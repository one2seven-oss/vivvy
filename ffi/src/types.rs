//! JSON envelopes exchanged across the FFI boundary.
//!
//! Embedding vectors never travel inside JSON — they cross as raw `float*` + `size_t`
//! pairs, since encoding thousands of floats as JSON numbers is wasteful on both sides.
//! Everything else (scope, metadata, filters, scoring) travels as JSON because the
//! underlying `vivvy-memory` types already derive `Serialize`/`Deserialize`, so JSON is
//! the natural, low-maintenance wire format for the structured, evolving parts of the API.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
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

fn default_true() -> bool {
    true
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
