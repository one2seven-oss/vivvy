use crate::error::{MemoryError, Result};
use crate::namespace::MemoryScope;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Identifies the semantic nature of a stored memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Preference,
    Fact,
    Instruction,
    Context,
    Episodic,
}

/// Lifecycle status of a memory in the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    Pending,
    Active,
    Deleted,
    Expired,
}

impl MemoryStatus {
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Active => "active",
            Self::Deleted => "deleted",
            Self::Expired => "expired",
        }
    }

    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(Self::Pending),
            "active" => Some(Self::Active),
            "deleted" => Some(Self::Deleted),
            "expired" => Some(Self::Expired),
            _ => None,
        }
    }
}

/// A stored memory record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub scope: MemoryScope,
    pub kind: MemoryKind,
    pub content: String,
    pub content_hash: Vec<u8>,
    pub embedding: Vec<f32>,
    pub embedding_model: String,
    pub embedding_dims: usize,
    pub importance: f32,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub last_accessed_at_ms: Option<i64>,
    pub access_count: u64,
    pub expires_at_ms: Option<i64>,
    pub status: MemoryStatus,
    pub revision: u64,
    pub metadata: HashMap<String, serde_json::Value>,
    pub source: HashMap<String, serde_json::Value>,
}

/// Request to store an observation or fact into memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RememberRequest {
    pub operation_id: Option<String>,
    pub scope: MemoryScope,
    pub content: String,
    pub embedding: Vec<f32>,
    pub kind: MemoryKind,
    pub importance: f32,
    pub expires_at_ms: Option<i64>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub source: HashMap<String, serde_json::Value>,
}

impl RememberRequest {
    pub fn validate(&self, expected_dims: usize) -> Result<()> {
        if self.content.trim().is_empty() {
            return Err(MemoryError::invalid_input("content cannot be empty"));
        }
        if self.embedding.len() != expected_dims {
            return Err(MemoryError::DimensionMismatch {
                code: crate::error::ErrorCode::DimensionMismatch,
                expected: expected_dims,
                actual: self.embedding.len(),
            });
        }
        if !(0.0..=1.0).contains(&self.importance) {
            return Err(MemoryError::invalid_input(
                "importance must be between 0.0 and 1.0",
            ));
        }
        Ok(())
    }
}

/// Filtering criteria for recall operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryFilter {
    pub kinds: Option<Vec<MemoryKind>>,
    pub min_importance: Option<f32>,
    pub metadata_eq: Option<HashMap<String, serde_json::Value>>,
    pub created_after_ms: Option<i64>,
    pub created_before_ms: Option<i64>,
}

impl MemoryFilter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn kind_in(kinds: impl IntoIterator<Item = MemoryKind>) -> Self {
        Self {
            kinds: Some(kinds.into_iter().collect()),
            min_importance: None,
            metadata_eq: None,
            created_after_ms: None,
            created_before_ms: None,
        }
    }

    #[must_use]
    pub fn with_min_importance(mut self, min_importance: f32) -> Self {
        self.min_importance = Some(min_importance);
        self
    }

    #[must_use]
    pub fn with_metadata_eq(
        mut self,
        key: impl Into<String>,
        val: impl Into<serde_json::Value>,
    ) -> Self {
        let mut map = self.metadata_eq.unwrap_or_default();
        map.insert(key.into(), val.into());
        self.metadata_eq = Some(map);
        self
    }

    #[must_use]
    pub fn with_time_range(mut self, after_ms: Option<i64>, before_ms: Option<i64>) -> Self {
        self.created_after_ms = after_ms;
        self.created_before_ms = before_ms;
        self
    }
}

/// Request to update an existing memory record with optimistic revision concurrency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRequest {
    pub operation_id: Option<String>,
    pub scope: MemoryScope,
    pub id: String,
    pub expected_revision: u64,
    pub content: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub kind: Option<MemoryKind>,
    pub importance: Option<f32>,
    pub expires_at_ms: Option<Option<i64>>,
    pub metadata_patch: Option<HashMap<String, serde_json::Value>>,
}

impl UpdateRequest {
    pub fn validate(&self, expected_dims: usize) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(MemoryError::invalid_input("id cannot be empty"));
        }
        if let Some(ref c) = self.content {
            if c.trim().is_empty() {
                return Err(MemoryError::invalid_input("content cannot be empty"));
            }
        }
        if let Some(ref emb) = self.embedding {
            if emb.len() != expected_dims {
                return Err(MemoryError::DimensionMismatch {
                    code: crate::error::ErrorCode::DimensionMismatch,
                    expected: expected_dims,
                    actual: emb.len(),
                });
            }
        }
        if let Some(imp) = self.importance {
            if !(0.0..=1.0).contains(&imp) {
                return Err(MemoryError::invalid_input(
                    "importance must be between 0.0 and 1.0",
                ));
            }
        }
        Ok(())
    }
}

/// Request to delete/forget a memory record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetRequest {
    pub operation_id: Option<String>,
    pub scope: MemoryScope,
    pub id: String,
}

impl ForgetRequest {
    pub fn new(scope: MemoryScope, id: impl Into<String>) -> Result<Self> {
        let id = id.into().trim().to_string();
        if id.is_empty() {
            return Err(MemoryError::invalid_input("id cannot be empty"));
        }
        Ok(Self {
            operation_id: None,
            scope,
            id,
        })
    }
}

fn default_weight_similarity() -> f32 {
    0.70
}

fn default_weight_importance() -> f32 {
    0.15
}

fn default_weight_recency() -> f32 {
    0.10
}

fn default_weight_reinforcement() -> f32 {
    0.05
}

fn default_recency_half_life_days() -> f32 {
    7.0
}

/// Configuration for memory reranking weights and temporal decay dynamics.
///
/// During candidate recall, Vivvy computes four independent factor scores for
/// each candidate memory:
/// - **Similarity** (`[0.0, 1.0]`): Dense cosine similarity to the query embedding.
/// - **Importance** (`[0.0, 1.0]`): User- or agent-declared importance.
/// - **Recency decay** (`[0.0, 1.0]`): Exponential decay based on memory age:
///   `exp(-age_ms / (recency_half_life_days * 86_400_000))`
/// - **Reinforcement** (`[0.0, 1.0]`): Access frequency saturation:
///   `min(access_count / 10.0, 1.0)`
///
/// The combined score is calculated as a weighted linear combination:
/// `total_score = w_sim * similarity + w_imp * importance + w_rec * recency + w_reinf * reinforcement`
///
/// # Default Values
/// - `weight_similarity`: `0.70`
/// - `weight_importance`: `0.15`
/// - `weight_recency`: `0.10`
/// - `weight_reinforcement`: `0.05`
/// - `recency_half_life_days`: `7.0` (days)
///
/// # Examples
/// ```rust
/// use vivvy_memory::ScoringConfig;
///
/// // Create default scoring configuration
/// let default_scoring = ScoringConfig::default();
/// assert_eq!(default_scoring.weight_similarity, 0.70);
///
/// // Use builder for custom workload tuning (e.g. documentation search)
/// let doc_scoring = ScoringConfig::builder()
///     .similarity(0.90)
///     .importance(0.10)
///     .recency(0.0)
///     .build()
///     .unwrap();
/// assert_eq!(doc_scoring.weight_recency, 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScoringConfig {
    /// Relative weight assigned to cosine vector similarity. Defaults to 0.70.
    #[serde(default = "default_weight_similarity")]
    pub weight_similarity: f32,

    /// Relative weight assigned to user/agent importance [0.0, 1.0]. Defaults to 0.15.
    #[serde(default = "default_weight_importance")]
    pub weight_importance: f32,

    /// Relative weight assigned to temporal recency decay. Defaults to 0.10.
    #[serde(default = "default_weight_recency")]
    pub weight_recency: f32,

    /// Relative weight assigned to memory access reinforcement. Defaults to 0.05.
    #[serde(default = "default_weight_reinforcement")]
    pub weight_reinforcement: f32,

    /// Half-life in days for exponential recency decay. Defaults to 7.0 days.
    #[serde(default = "default_recency_half_life_days")]
    pub recency_half_life_days: f32,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            weight_similarity: default_weight_similarity(),
            weight_importance: default_weight_importance(),
            weight_recency: default_weight_recency(),
            weight_reinforcement: default_weight_reinforcement(),
            recency_half_life_days: default_recency_half_life_days(),
        }
    }
}

impl ScoringConfig {
    /// Creates a new [`ScoringConfigBuilder`] initialized with default weights.
    #[must_use]
    pub fn builder() -> ScoringConfigBuilder {
        ScoringConfigBuilder::default()
    }

    /// Validates that all weights are non-negative, finite, and non-zero in aggregate,
    /// and that `recency_half_life_days` is strictly positive and finite.
    ///
    /// # Errors
    /// Returns [`MemoryError::InvalidInput`] if:
    /// - Any weight is negative, NaN, or infinite.
    /// - The sum of all weights is zero or negative.
    /// - `recency_half_life_days` is non-positive (`<= 0.0`), NaN, or infinite.
    pub fn validate(&self) -> Result<()> {
        let weights = [
            ("weight_similarity", self.weight_similarity),
            ("weight_importance", self.weight_importance),
            ("weight_recency", self.weight_recency),
            ("weight_reinforcement", self.weight_reinforcement),
        ];

        let mut sum = 0.0f32;
        for (name, val) in weights {
            if val.is_nan() || val.is_infinite() {
                return Err(MemoryError::invalid_input(format!(
                    "{name} must be a finite number, got {val}"
                )));
            }
            if val < 0.0 {
                return Err(MemoryError::invalid_input(format!(
                    "{name} must be non-negative, got {val}"
                )));
            }
            sum += val;
        }

        if sum <= 0.0 {
            return Err(MemoryError::invalid_input(
                "at least one scoring weight must be greater than zero",
            ));
        }

        if self.recency_half_life_days.is_nan() || self.recency_half_life_days.is_infinite() {
            return Err(MemoryError::invalid_input(
                "recency_half_life_days must be a finite number",
            ));
        }

        if self.recency_half_life_days <= 0.0 {
            return Err(MemoryError::invalid_input(format!(
                "recency_half_life_days must be strictly positive, got {}",
                self.recency_half_life_days
            )));
        }

        Ok(())
    }
}

/// Builder for constructing and validating a [`ScoringConfig`].
#[derive(Debug, Clone, Default)]
pub struct ScoringConfigBuilder {
    config: ScoringConfig,
}

impl ScoringConfigBuilder {
    /// Sets the relative weight assigned to cosine vector similarity.
    #[must_use]
    pub fn similarity(mut self, weight: f32) -> Self {
        self.config.weight_similarity = weight;
        self
    }

    /// Sets the relative weight assigned to user/agent importance.
    #[must_use]
    pub fn importance(mut self, weight: f32) -> Self {
        self.config.weight_importance = weight;
        self
    }

    /// Sets the relative weight assigned to temporal recency decay.
    #[must_use]
    pub fn recency(mut self, weight: f32) -> Self {
        self.config.weight_recency = weight;
        self
    }

    /// Sets the relative weight assigned to access reinforcement.
    #[must_use]
    pub fn reinforcement(mut self, weight: f32) -> Self {
        self.config.weight_reinforcement = weight;
        self
    }

    /// Sets the half-life in days for exponential recency decay.
    #[must_use]
    pub fn half_life_days(mut self, days: f32) -> Self {
        self.config.recency_half_life_days = days;
        self
    }

    /// Validates and returns the constructed [`ScoringConfig`].
    ///
    /// # Errors
    /// Returns [`MemoryError::InvalidInput`] if any weight or half-life invariant fails.
    pub fn build(self) -> Result<ScoringConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Request to recall memories for an agent context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallRequest {
    pub scope: MemoryScope,
    pub query_embedding: Vec<f32>,
    #[serde(default)]
    pub query_text: Option<String>,
    pub limit: usize,
    pub filters: MemoryFilter,
    pub include_explanations: bool,
    pub mmr_lambda: Option<f32>,
    /// Scoring configuration for ranking candidate memories. Defaults to [`ScoringConfig::default()`].
    #[serde(default)]
    pub scoring: ScoringConfig,
}

impl RecallRequest {
    /// Creates a new [`RecallRequest`] with default scoring and filters.
    #[must_use]
    pub fn new(scope: MemoryScope, query_embedding: Vec<f32>, limit: usize) -> Self {
        Self {
            scope,
            query_embedding,
            query_text: None,
            limit,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
            scoring: ScoringConfig::default(),
        }
    }

    /// Sets the scoring configuration.
    #[must_use]
    pub fn with_scoring(mut self, scoring: ScoringConfig) -> Self {
        self.scoring = scoring;
        self
    }

    pub fn validate(&self, expected_dims: usize) -> Result<()> {
        if self.limit == 0 {
            return Err(MemoryError::invalid_input("limit must be greater than 0"));
        }
        if self.query_embedding.len() != expected_dims {
            return Err(MemoryError::DimensionMismatch {
                code: crate::error::ErrorCode::DimensionMismatch,
                expected: expected_dims,
                actual: self.query_embedding.len(),
            });
        }
        if let Some(lambda) = self.mmr_lambda {
            if !(0.0..=1.0).contains(&lambda) {
                return Err(MemoryError::invalid_input(
                    "mmr_lambda must be between 0.0 and 1.0",
                ));
            }
        }
        self.scoring.validate()?;
        Ok(())
    }
}

/// Explanation detailing why a memory was selected during recall.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallExplanation {
    pub total_score: f32,
    pub similarity_score: f32,
    pub importance_score: f32,
    pub recency_score: f32,
    pub reinforcement_score: f32,
    pub policy_notes: Vec<String>,
}

/// Individual recalled item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallItem {
    pub memory: MemoryRecord,
    pub score: f32,
    pub explanation: Option<RecallExplanation>,
}

/// Complete response from a recall query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallResponse {
    pub items: Vec<RecallItem>,
    pub total_candidates: usize,
}

/// Configuration options for LLM context formatting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextFormatOptions {
    pub max_tokens: usize,
    pub template: String,
    pub header: Option<String>,
    pub footer: Option<String>,
}

impl Default for ContextFormatOptions {
    fn default() -> Self {
        Self {
            max_tokens: 1500,
            template: "- [{kind}] {content} (score: {score:.2})".to_string(),
            header: Some("### Relevant Agent Memories:".to_string()),
            footer: None,
        }
    }
}

impl ContextFormatOptions {
    pub fn validate(&self) -> Result<()> {
        if self.max_tokens == 0 {
            return Err(MemoryError::invalid_input(
                "max_tokens must be greater than 0",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoring_config_defaults() {
        let config = ScoringConfig::default();
        assert_eq!(config.weight_similarity, 0.70);
        assert_eq!(config.weight_importance, 0.15);
        assert_eq!(config.weight_recency, 0.10);
        assert_eq!(config.weight_reinforcement, 0.05);
        assert_eq!(config.recency_half_life_days, 7.0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_scoring_config_builder() {
        let config = ScoringConfig::builder()
            .similarity(0.85)
            .importance(0.10)
            .recency(0.05)
            .reinforcement(0.0)
            .half_life_days(3.5)
            .build()
            .unwrap();

        assert_eq!(config.weight_similarity, 0.85);
        assert_eq!(config.weight_importance, 0.10);
        assert_eq!(config.weight_recency, 0.05);
        assert_eq!(config.weight_reinforcement, 0.0);
        assert_eq!(config.recency_half_life_days, 3.5);
    }

    #[test]
    fn test_scoring_config_rejects_negative_weights() {
        let cases = [
            (
                "similarity",
                ScoringConfig {
                    weight_similarity: -0.1,
                    ..Default::default()
                },
            ),
            (
                "importance",
                ScoringConfig {
                    weight_importance: -0.1,
                    ..Default::default()
                },
            ),
            (
                "recency",
                ScoringConfig {
                    weight_recency: -0.1,
                    ..Default::default()
                },
            ),
            (
                "reinforcement",
                ScoringConfig {
                    weight_reinforcement: -0.1,
                    ..Default::default()
                },
            ),
        ];

        for (name, cfg) in cases {
            let err = cfg.validate().unwrap_err();
            assert!(
                err.to_string().contains("must be non-negative"),
                "Expected non-negative error for {name}, got: {err}"
            );
        }
    }

    #[test]
    fn test_scoring_config_rejects_nan_and_inf() {
        let nan_cfg = ScoringConfig {
            weight_similarity: f32::NAN,
            ..Default::default()
        };
        assert!(nan_cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("finite"));

        let inf_cfg = ScoringConfig {
            weight_importance: f32::INFINITY,
            ..Default::default()
        };
        assert!(inf_cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("finite"));

        let nan_decay = ScoringConfig {
            recency_half_life_days: f32::NAN,
            ..Default::default()
        };
        assert!(nan_decay
            .validate()
            .unwrap_err()
            .to_string()
            .contains("finite"));
    }

    #[test]
    fn test_scoring_config_rejects_zero_or_negative_half_life() {
        let zero_decay = ScoringConfig {
            recency_half_life_days: 0.0,
            ..Default::default()
        };
        assert!(zero_decay
            .validate()
            .unwrap_err()
            .to_string()
            .contains("strictly positive"));

        let neg_decay = ScoringConfig {
            recency_half_life_days: -5.0,
            ..Default::default()
        };
        assert!(neg_decay
            .validate()
            .unwrap_err()
            .to_string()
            .contains("strictly positive"));
    }

    #[test]
    fn test_scoring_config_rejects_all_zero_weights() {
        let all_zero = ScoringConfig {
            weight_similarity: 0.0,
            weight_importance: 0.0,
            weight_recency: 0.0,
            weight_reinforcement: 0.0,
            recency_half_life_days: 7.0,
        };
        assert!(all_zero
            .validate()
            .unwrap_err()
            .to_string()
            .contains("at least one scoring weight"));
    }

    #[test]
    fn test_scoring_config_serde_empty_json_uses_defaults() {
        let config: ScoringConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config, ScoringConfig::default());
    }

    #[test]
    fn test_scoring_config_serde_roundtrip() {
        let custom = ScoringConfig {
            weight_similarity: 0.60,
            weight_importance: 0.20,
            weight_recency: 0.15,
            weight_reinforcement: 0.05,
            recency_half_life_days: 14.0,
        };
        let json = serde_json::to_string(&custom).unwrap();
        let parsed: ScoringConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(custom, parsed);
    }

    #[test]
    fn test_scoring_config_validation() {
        // 1. Negative weights
        let neg_cfg = ScoringConfig {
            weight_similarity: -0.1,
            ..Default::default()
        };
        assert!(neg_cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must be non-negative"));

        // 2. NaNs and non-finite values
        let nan_cfg = ScoringConfig {
            weight_similarity: f32::NAN,
            ..Default::default()
        };
        assert!(nan_cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("finite"));

        // 3. Zero or negative half-life
        let zero_half_life = ScoringConfig {
            recency_half_life_days: 0.0,
            ..Default::default()
        };
        assert!(zero_half_life
            .validate()
            .unwrap_err()
            .to_string()
            .contains("strictly positive"));

        let neg_half_life = ScoringConfig {
            recency_half_life_days: -2.0,
            ..Default::default()
        };
        assert!(neg_half_life
            .validate()
            .unwrap_err()
            .to_string()
            .contains("strictly positive"));

        // 4. All-zero weights
        let all_zero = ScoringConfig {
            weight_similarity: 0.0,
            weight_importance: 0.0,
            weight_recency: 0.0,
            weight_reinforcement: 0.0,
            recency_half_life_days: 7.0,
        };
        assert!(all_zero
            .validate()
            .unwrap_err()
            .to_string()
            .contains("at least one scoring weight"));

        // Valid configuration
        let valid_cfg = ScoringConfig::builder()
            .similarity(0.8)
            .importance(0.2)
            .recency(0.0)
            .reinforcement(0.0)
            .half_life_days(5.0)
            .build();
        assert!(valid_cfg.is_ok());
    }

    #[test]
    fn test_recall_request_validate_checks_scoring() {
        let scope = MemoryScope::new("t", "ns").unwrap();
        let mut req = RecallRequest::new(scope, vec![0.1, 0.2, 0.3], 5);
        assert!(req.validate(3).is_ok());

        req.scoring.weight_similarity = -1.0;
        assert!(req.validate(3).is_err());
    }
}
