//! JSON envelopes exchanged across the FFI boundary.
//!
//! Embedding vectors never travel inside JSON — they cross as raw `float*` + `size_t`
//! pairs, since encoding thousands of floats as JSON numbers is wasteful on both sides.
//! Everything else (scope, metadata, filters, scoring) travels as JSON because the
//! underlying `vivvy-memory` types already derive `Serialize`/`Deserialize`, so JSON is
//! the natural, low-maintenance wire format for the structured, evolving parts of the API.

use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize};
use vivvy_memory::{MemoryKind, MemoryRecord};

fn default_max_recall_limit() -> usize {
    100
}

/// Body of `config_json` for `vivvy_store_open`.
#[derive(Debug, Deserialize)]
pub struct OpenConfigJson {
    pub dimensions: usize,
    pub embedding_model: String,
    #[serde(default = "default_max_recall_limit")]
    pub max_recall_limit: usize,
}

fn default_kind() -> String {
    "fact".to_string()
}

fn default_importance() -> f32 {
    0.5
}

/// Body of `record_json` for `vivvy_store_insert`. The embedding itself is
/// supplied separately via the `vector`/`dim` parameters.
#[derive(Debug, Deserialize)]
pub struct InsertRecordJson {
    #[serde(default)]
    pub operation_id: Option<String>,
    pub tenant_id: String,
    pub namespace: String,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
    pub content: String,
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default = "default_importance")]
    pub importance: f32,
    #[serde(default)]
    pub expires_at_ms: Option<i64>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub source: HashMap<String, serde_json::Value>,
}

/// Scope shared by get, forget, and batch-forget requests.
#[derive(Debug, Deserialize)]
pub struct ScopeJson {
    pub tenant_id: String,
    pub namespace: String,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
}

/// JSON portion of an update request. The optional replacement embedding is
/// supplied separately as a raw float buffer, just like inserts and recall.
#[derive(Debug, Deserialize)]
pub struct UpdateJson {
    #[serde(default)]
    pub operation_id: Option<String>,
    pub tenant_id: String,
    pub namespace: String,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
    pub id: String,
    pub expected_revision: u64,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub importance: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_expiration_update")]
    pub expires_at_ms: Option<ExpirationUpdateJson>,
    #[serde(default)]
    pub metadata_patch: Option<HashMap<String, serde_json::Value>>,
}

/// The update wire format distinguishes an omitted expiry (leave it as-is),
/// an integer expiry (set it), and JSON null (clear it).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ExpirationUpdateJson {
    Value(i64),
    Clear(()),
}

fn deserialize_expiration_update<'de, D>(
    deserializer: D,
) -> Result<Option<ExpirationUpdateJson>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Some(match Option::<i64>::deserialize(deserializer)? {
        Some(value) => ExpirationUpdateJson::Value(value),
        None => ExpirationUpdateJson::Clear(()),
    }))
}

fn default_true() -> bool {
    true
}

fn default_similarity() -> f32 {
    0.70
}

fn default_importance_weight() -> f32 {
    0.15
}

fn default_recency() -> f32 {
    0.10
}

fn default_reinforcement() -> f32 {
    0.05
}

fn default_half_life() -> f32 {
    7.0
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct ScoringConfigJson {
    #[serde(default = "default_similarity")]
    pub weight_similarity: f32,
    #[serde(default = "default_importance_weight")]
    pub weight_importance: f32,
    #[serde(default = "default_recency")]
    pub weight_recency: f32,
    #[serde(default = "default_reinforcement")]
    pub weight_reinforcement: f32,
    #[serde(default = "default_half_life")]
    pub recency_half_life_days: f32,
}

impl From<ScoringConfigJson> for vivvy_memory::ScoringConfig {
    fn from(json: ScoringConfigJson) -> Self {
        Self {
            weight_similarity: json.weight_similarity,
            weight_importance: json.weight_importance,
            weight_recency: json.weight_recency,
            weight_reinforcement: json.weight_reinforcement,
            recency_half_life_days: json.recency_half_life_days,
        }
    }
}

/// Body of `options_json` for both `vivvy_store_recall` and
/// `vivvy_store_format_context`: scope, filters, and scoring knobs. The query
/// vector/text and top_k stay as dedicated C parameters since every recall
/// call needs them.
#[derive(Debug, Deserialize)]
pub struct RecallOptionsJson {
    pub tenant_id: String,
    pub namespace: String,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default)]
    pub kinds: Option<Vec<String>>,
    #[serde(default)]
    pub min_importance: Option<f32>,
    #[serde(default)]
    pub metadata_eq: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub created_after_ms: Option<i64>,
    #[serde(default)]
    pub created_before_ms: Option<i64>,
    #[serde(default = "default_true")]
    pub include_explanations: bool,
    #[serde(default)]
    pub mmr_lambda: Option<f32>,
    #[serde(default)]
    pub scoring: Option<ScoringConfigJson>,
}

/// Body of the optional `format_json` parameter for `vivvy_store_format_context`.
#[derive(Debug, Default, Deserialize)]
pub struct FormatOptionsJson {
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub header: Option<String>,
    #[serde(default)]
    pub footer: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RecallResultsJson {
    pub total_candidates: usize,
    pub items: Vec<ScoredMemoryJson>,
}

#[derive(Debug, Serialize)]
pub struct ScoredMemoryJson {
    pub id: String,
    pub tenant_id: String,
    pub namespace: String,
    pub agent_id: Option<String>,
    pub user_id: Option<String>,
    pub kind: String,
    pub content: String,
    pub importance: f32,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub expires_at_ms: Option<i64>,
    pub revision: u64,
    pub metadata: HashMap<String, serde_json::Value>,
    pub source: HashMap<String, serde_json::Value>,
    pub score: f32,
    pub explanation: Option<ExplanationJson>,
}

/// A stored memory returned by `vivvy_store_get`.
#[derive(Debug, Serialize)]
pub struct MemoryRecordJson {
    pub id: String,
    pub tenant_id: String,
    pub namespace: String,
    pub agent_id: Option<String>,
    pub user_id: Option<String>,
    pub kind: String,
    pub content: String,
    pub importance: f32,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub expires_at_ms: Option<i64>,
    pub revision: u64,
    pub metadata: HashMap<String, serde_json::Value>,
    pub source: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct ExplanationJson {
    pub total_score: f32,
    pub similarity_score: f32,
    pub importance_score: f32,
    pub recency_score: f32,
    pub reinforcement_score: f32,
    pub policy_notes: Vec<String>,
}

/// Maps a free-form kind string onto `MemoryKind`. Unrecognized or absent
/// values default to `Fact`, matching `InsertRecordJson`'s default.
#[must_use]
pub fn parse_kind(s: &str) -> MemoryKind {
    match s.to_lowercase().as_str() {
        "preference" => MemoryKind::Preference,
        "instruction" => MemoryKind::Instruction,
        "context" => MemoryKind::Context,
        "episodic" => MemoryKind::Episodic,
        _ => MemoryKind::Fact,
    }
}

#[must_use]
pub fn kind_to_str(k: MemoryKind) -> &'static str {
    match k {
        MemoryKind::Fact => "fact",
        MemoryKind::Preference => "preference",
        MemoryKind::Instruction => "instruction",
        MemoryKind::Context => "context",
        MemoryKind::Episodic => "episodic",
    }
}

impl From<&MemoryRecord> for ScoredMemoryJson {
    /// Converts a recalled record into its wire form with `score`/`explanation`
    /// left at their zero values; callers fill those in from the `RecallItem`.
    fn from(r: &MemoryRecord) -> Self {
        ScoredMemoryJson {
            id: r.id.clone(),
            tenant_id: r.scope.tenant_id().to_string(),
            namespace: r.scope.namespace().to_string(),
            agent_id: r.scope.agent_id().map(|s| s.to_string()),
            user_id: r.scope.user_id().map(|s| s.to_string()),
            kind: kind_to_str(r.kind).to_string(),
            content: r.content.clone(),
            importance: r.importance,
            created_at_ms: r.created_at_ms,
            updated_at_ms: r.updated_at_ms,
            expires_at_ms: r.expires_at_ms,
            revision: r.revision,
            metadata: r.metadata.clone(),
            source: r.source.clone(),
            score: 0.0,
            explanation: None,
        }
    }
}

impl From<&MemoryRecord> for MemoryRecordJson {
    fn from(r: &MemoryRecord) -> Self {
        Self {
            id: r.id.clone(),
            tenant_id: r.scope.tenant_id().to_string(),
            namespace: r.scope.namespace().to_string(),
            agent_id: r.scope.agent_id().map(str::to_string),
            user_id: r.scope.user_id().map(str::to_string),
            kind: kind_to_str(r.kind).to_string(),
            content: r.content.clone(),
            importance: r.importance,
            created_at_ms: r.created_at_ms,
            updated_at_ms: r.updated_at_ms,
            expires_at_ms: r.expires_at_ms,
            revision: r.revision,
            metadata: r.metadata.clone(),
            source: r.source.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ExpirationUpdateJson, UpdateJson};

    #[test]
    fn update_expiry_preserves_omitted_set_and_clear_states() {
        let omitted: UpdateJson = serde_json::from_str(
            r#"{"tenant_id":"t","namespace":"n","id":"id","expected_revision":1}"#,
        )
        .unwrap();
        assert!(omitted.expires_at_ms.is_none());

        let clear: UpdateJson = serde_json::from_str(
            r#"{"tenant_id":"t","namespace":"n","id":"id","expected_revision":1,"expires_at_ms":null}"#,
        )
        .unwrap();
        assert!(matches!(
            clear.expires_at_ms,
            Some(ExpirationUpdateJson::Clear(()))
        ));

        let set: UpdateJson = serde_json::from_str(
            r#"{"tenant_id":"t","namespace":"n","id":"id","expected_revision":1,"expires_at_ms":123}"#,
        )
        .unwrap();
        assert!(matches!(
            set.expires_at_ms,
            Some(ExpirationUpdateJson::Value(123))
        ));
    }
}
