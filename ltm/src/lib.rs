pub mod cipher;
pub mod config;
pub mod crypto;
pub mod error;
pub mod health;
pub mod index_adapter;
pub mod journal;
pub mod model;
pub mod namespace;
pub mod repository;
pub mod telemetry;

pub use cipher::{FieldCipher, LexicalMode};
pub use config::MemoryConfig;
pub use crypto::{KeyProvider, MissingKeyProvider, NoOpDevKeyProvider};
pub use error::{ErrorCode, MemoryError, Result};
pub use health::StoreHealth;
pub use index_adapter::{InMemoryTestIndex, VectorIndex, VivvyVectorIndex};
pub use model::{
    ContextFormatOptions, ForgetRequest, MemoryFilter, MemoryKind, MemoryRecord, MemoryStatus,
    RecallExplanation, RecallItem, RecallRequest, RecallResponse, RememberRequest, ScoringConfig,
    ScoringConfigBuilder, UpdateRequest,
};
pub use namespace::MemoryScope;
pub use repository::Repository;
pub use telemetry::TelemetryRecord;

use crate::journal::JournalCoordinator;
use std::sync::Arc;
use uuid::Uuid;
use vivvy_core::distance::Metric;

/// A local, durable, namespace-isolated memory store for AI agents.
pub struct MemoryStore {
    config: Arc<MemoryConfig>,
    repo: Arc<Repository>,
    index: Arc<dyn VectorIndex>,
    coordinator: JournalCoordinator,
}

fn record_matches_recall(req: &RecallRequest, record: &MemoryRecord, now_ms: i64) -> bool {
    if record.status != MemoryStatus::Active {
        return false;
    }

    if record
        .expires_at_ms
        .is_some_and(|expires_at| now_ms >= expires_at)
    {
        return false;
    }

    if req
        .filters
        .kinds
        .as_ref()
        .is_some_and(|kinds| !kinds.contains(&record.kind))
    {
        return false;
    }

    if req
        .filters
        .min_importance
        .is_some_and(|minimum| record.importance < minimum)
    {
        return false;
    }

    if let Some(metadata_filter) = &req.filters.metadata_eq {
        if metadata_filter
            .iter()
            .any(|(key, expected)| record.metadata.get(key) != Some(expected))
        {
            return false;
        }
    }

    if req
        .filters
        .created_after_ms
        .is_some_and(|after| record.created_at_ms < after)
    {
        return false;
    }

    if req
        .filters
        .created_before_ms
        .is_some_and(|before| record.created_at_ms > before)
    {
        return false;
    }

    true
}

impl MemoryStore {
    /// Open or initialize a memory store at the configured directory.
    pub fn open(config: MemoryConfig) -> Result<Self> {
        let path = config.path();
        if !path.exists() {
            std::fs::create_dir_all(path).map_err(|e| MemoryError::IoError {
                code: ErrorCode::IoError,
                message: format!("Failed to create directory {:?}: {}", path, e),
            })?;
        }

        // Clean up stale index files from previous ephemeral index versions
        // (best-effort; ignore if files don't exist)
        let _ = std::fs::remove_file(path.join("index.wal"));
        let _ = std::fs::remove_file(path.join("index.manifest"));
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with(".vivvy") {
                        let _ = std::fs::remove_file(entry.path());
                    }
                }
            }
        }

        let db_path = path.join("memory.db");
        let cipher = config
            .key_provider()
            .map(|p| FieldCipher::new(p.clone(), config.lexical_mode()));
        let repo = Arc::new(Repository::open_with_encryption(
            db_path,
            cipher,
            config.lexical_mode(),
        )?);
        repo.enforce_store_metadata(config.embedding_model(), config.dimensions())?;

        let index = Arc::new(VivvyVectorIndex::new(config.dimensions(), Metric::Cosine)?);
        let coordinator = JournalCoordinator::new(repo.clone(), index.clone());

        // Run crash recovery / journal replay
        coordinator.recover_startup()?;

        Ok(Self {
            config: Arc::new(config),
            repo,
            index,
            coordinator,
        })
    }

    /// Access the store's immutable configuration.
    #[must_use]
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Store an observation/fact into durable memory.
    pub fn remember(&self, req: RememberRequest) -> Result<String> {
        req.validate(self.config.dimensions())?;

        let id = Uuid::new_v4().to_string();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&req.content, &mut hasher);
        let content_hash = std::hash::Hasher::finish(&hasher).to_le_bytes().to_vec();

        let record = MemoryRecord {
            id,
            scope: req.scope,
            kind: req.kind,
            content: req.content,
            content_hash,
            embedding: req.embedding,
            embedding_model: self.config.embedding_model().to_string(),
            embedding_dims: self.config.dimensions(),
            importance: req.importance,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            last_accessed_at_ms: None,
            access_count: 0,
            expires_at_ms: req.expires_at_ms,
            status: MemoryStatus::Pending,
            revision: 1,
            metadata: req.metadata,
            source: req.source,
        };

        self.coordinator
            .coordinate_remember(record, req.operation_id)
    }

    /// Store multiple observations/facts into durable memory in a single atomic transaction.
    pub fn remember_batch(&self, reqs: Vec<RememberRequest>) -> Result<Vec<String>> {
        if reqs.is_empty() {
            return Ok(Vec::new());
        }
        for req in &reqs {
            req.validate(self.config.dimensions())?;
        }

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let mut operation_ids = Vec::with_capacity(reqs.len());
        let records: Vec<MemoryRecord> = reqs
            .into_iter()
            .map(|req| {
                operation_ids.push(req.operation_id);
                let id = Uuid::new_v4().to_string();
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                std::hash::Hash::hash(&req.content, &mut hasher);
                let content_hash = std::hash::Hasher::finish(&hasher).to_le_bytes().to_vec();

                MemoryRecord {
                    id,
                    scope: req.scope,
                    kind: req.kind,
                    content: req.content,
                    content_hash,
                    embedding: req.embedding,
                    embedding_model: self.config.embedding_model().to_string(),
                    embedding_dims: self.config.dimensions(),
                    importance: req.importance,
                    created_at_ms: now_ms,
                    updated_at_ms: now_ms,
                    last_accessed_at_ms: None,
                    access_count: 0,
                    expires_at_ms: req.expires_at_ms,
                    status: MemoryStatus::Pending,
                    revision: 1,
                    metadata: req.metadata,
                    source: req.source,
                }
            })
            .collect();

        self.coordinator.coordinate_remember_batch(records, operation_ids)
    }

    /// Fetch a memory by ID within scope.
    pub fn get(&self, scope: &MemoryScope, id: &str) -> Result<Option<MemoryRecord>> {
        self.repo.get_by_scope_and_id(scope, id)
    }

    /// Retrieve matching memories for an agent context with explainability.
    pub fn recall(&self, req: RecallRequest) -> Result<RecallResponse> {
        req.validate(self.config.dimensions())?;

        let recall_limit = req.limit.min(self.config.max_recall_limit());
        if recall_limit == 0 {
            return Ok(RecallResponse {
                items: Vec::new(),
                total_candidates: 0,
            });
        }

        let mut candidate_limit = recall_limit.saturating_mul(3).max(1);

        let fts_candidate_ids = if let Some(ref text) = req.query_text {
            self.repo.search_fts(&req.scope, text, &req.filters, candidate_limit)?
        } else {
            Vec::new()
        };

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        let mut record_cache = std::collections::HashMap::new();
        let vector_candidates = loop {
            let vector_candidates = self.index.search(&req.query_embedding, candidate_limit)?;
            let mut eligible_count = 0;
            let mut seen_ids = std::collections::HashSet::new();

            for mem_id in fts_candidate_ids
                .iter()
                .chain(vector_candidates.iter().map(|(mem_id, _)| mem_id))
            {
                if !seen_ids.insert(mem_id.clone()) {
                    continue;
                }

                if !record_cache.contains_key(mem_id) {
                    let record = self.repo.get_by_scope_and_id(&req.scope, mem_id)?;
                    record_cache.insert(mem_id.clone(), record);
                }

                if record_cache
                    .get(mem_id)
                    .and_then(Option::as_ref)
                    .is_some_and(|record| record_matches_recall(&req, record, now_ms))
                {
                    eligible_count += 1;
                }
            }

            if eligible_count >= recall_limit || vector_candidates.len() < candidate_limit {
                break vector_candidates;
            }

            let next_limit = candidate_limit.saturating_mul(2);
            if next_limit <= candidate_limit {
                break vector_candidates;
            }
            candidate_limit = next_limit;
        };

        // Combine candidate IDs with RRF scoring & track sources
        let mut rrf_scores: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        let mut from_fts: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut from_vec: std::collections::HashSet<String> = std::collections::HashSet::new();

        for (rank_idx, (mem_id, _)) in vector_candidates.iter().enumerate() {
            let rrf = 1.0 / (60.0 + (rank_idx + 1) as f32);
            *rrf_scores.entry(mem_id.clone()).or_insert(0.0) += rrf;
            from_vec.insert(mem_id.clone());
        }

        for (rank_idx, mem_id) in fts_candidate_ids.iter().enumerate() {
            let rrf = 1.0 / (60.0 + (rank_idx + 1) as f32);
            *rrf_scores.entry(mem_id.clone()).or_insert(0.0) += rrf;
            from_fts.insert(mem_id.clone());
        }

        let mut unique_candidate_ids: Vec<String> = rrf_scores.keys().cloned().collect();
        unique_candidate_ids.sort_unstable_by(|a, b| {
            let score_a = rrf_scores.get(a).copied().unwrap_or(0.0);
            let score_b = rrf_scores.get(b).copied().unwrap_or(0.0);
            score_b.total_cmp(&score_a)
        });

        let mut items = Vec::new();
        let half_life_ms = req.scoring.recency_half_life_days * 86_400_000.0;
        let is_default_scoring = req.scoring == ScoringConfig::default();

        for mem_id in unique_candidate_ids {
            if let Some(record) = record_cache.get(&mem_id).and_then(Option::as_ref).cloned() {
                if !record_matches_recall(&req, &record, now_ms) {
                    continue;
                }

                let distance = vivvy_core::distance::cosine(&req.query_embedding, &record.embedding);
                let similarity = (1.0 - (distance / 2.0)).clamp(0.0, 1.0);
                let importance = record.importance.clamp(0.0, 1.0);

                let age_ms = (now_ms - record.created_at_ms).max(0) as f32;
                let recency_decay = (-age_ms / half_life_ms).exp();

                let reinforcement = (record.access_count as f32 / 10.0).clamp(0.0, 1.0);

                let total_score = req.scoring.weight_similarity * similarity
                    + req.scoring.weight_importance * importance
                    + req.scoring.weight_recency * recency_decay
                    + req.scoring.weight_reinforcement * reinforcement;

                let explanation = if req.include_explanations {
                    let mut notes = Vec::new();
                    if is_default_scoring {
                        notes.push("default_v1_scoring".to_string());
                    } else {
                        notes.push(format!(
                            "custom_scoring(sim={:.2},imp={:.2},rec={:.2},reinf={:.2},half_life={:.1}d)",
                            req.scoring.weight_similarity,
                            req.scoring.weight_importance,
                            req.scoring.weight_recency,
                            req.scoring.weight_reinforcement,
                            req.scoring.recency_half_life_days,
                        ));
                    }
                    if from_fts.contains(&mem_id) {
                        notes.push("fts5_lexical_candidate".to_string());
                    }
                    if from_vec.contains(&mem_id) && from_fts.contains(&mem_id) {
                        notes.push("reciprocal_rank_fusion".to_string());
                    }
                    Some(RecallExplanation {
                        total_score,
                        similarity_score: similarity,
                        importance_score: importance,
                        recency_score: recency_decay,
                        reinforcement_score: reinforcement,
                        policy_notes: notes,
                    })
                } else {
                    None
                };

                items.push(RecallItem {
                    memory: record,
                    score: total_score,
                    explanation,
                });
            }
        }

        // Apply MMR diversity if mmr_lambda is specified
        if let Some(lambda) = req.mmr_lambda {
            let mut selected: Vec<RecallItem> = Vec::new();
            let mut remaining = items;

            while !remaining.is_empty() && selected.len() < recall_limit {
                let mut best_idx = 0;
                let mut best_mmr_score = f32::NEG_INFINITY;

                for (idx, candidate) in remaining.iter().enumerate() {
                    // Relevance term
                    let rel = candidate.score;

                    // Redundancy term: max similarity to already selected items
                    let mut max_sim = 0.0f32;
                    for sel in &selected {
                        let dist = vivvy_core::distance::cosine(
                            &candidate.memory.embedding,
                            &sel.memory.embedding,
                        );
                        let sim = (1.0 - dist).clamp(0.0, 1.0);
                        if sim > max_sim {
                            max_sim = sim;
                        }
                    }

                    // MMR formula: lambda * rel - (1 - lambda) * max_sim
                    let mmr_score = lambda * rel - (1.0 - lambda) * max_sim;
                    if mmr_score > best_mmr_score {
                        best_mmr_score = mmr_score;
                        best_idx = idx;
                    }
                }

                let chosen = remaining.remove(best_idx);
                selected.push(chosen);
            }

            Ok(RecallResponse {
                total_candidates: selected.len(),
                items: selected,
            })
        } else {
            // Sort items by total score descending
            items.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
            items.truncate(recall_limit);

            Ok(RecallResponse {
                total_candidates: items.len(),
                items,
            })
        }
    }

    /// Update an existing memory record with optimistic concurrency.
    pub fn update(&self, req: UpdateRequest) -> Result<()> {
        req.validate(self.config.dimensions())?;

        let existing = self
            .repo
            .get_by_scope_and_id(&req.scope, &req.id)?
            .ok_or_else(|| MemoryError::not_found(&req.id))?;

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let new_content = req.content.unwrap_or(existing.content);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&new_content, &mut hasher);
        let new_content_hash = std::hash::Hasher::finish(&hasher).to_le_bytes().to_vec();

        let new_embedding = req.embedding.unwrap_or(existing.embedding);
        let new_kind = req.kind.unwrap_or(existing.kind);
        let new_importance = req.importance.unwrap_or(existing.importance);
        let new_expires_at = req.expires_at_ms.unwrap_or(existing.expires_at_ms);

        let mut new_metadata = existing.metadata;
        if let Some(patch) = req.metadata_patch {
            for (k, v) in patch {
                new_metadata.insert(k, v);
            }
        }

        let updated_record = MemoryRecord {
            id: req.id,
            scope: req.scope,
            kind: new_kind,
            content: new_content,
            content_hash: new_content_hash,
            embedding: new_embedding,
            embedding_model: self.config.embedding_model().to_string(),
            embedding_dims: self.config.dimensions(),
            importance: new_importance,
            created_at_ms: existing.created_at_ms,
            updated_at_ms: now_ms,
            last_accessed_at_ms: existing.last_accessed_at_ms,
            access_count: existing.access_count,
            expires_at_ms: new_expires_at,
            status: MemoryStatus::Active,
            revision: existing.revision + 1,
            metadata: new_metadata,
            source: existing.source,
        };

        self.coordinator
            .coordinate_update(updated_record, req.expected_revision, req.operation_id)
    }

    /// Forget/delete a memory record immediately hiding it from reads.
    pub fn forget(&self, req: ForgetRequest) -> Result<()> {
        self.coordinator
            .coordinate_forget(&req.scope, &req.id, req.operation_id)
    }

    /// Forget/delete multiple memory records in a single atomic transaction.
    pub fn forget_batch(&self, scope: &MemoryScope, ids: &[&str]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let operation_ids = vec![None; ids.len()];
        self.coordinator
            .coordinate_forget_batch(scope, ids, operation_ids)
    }

    /// Return a non-blocking operational health snapshot of the memory store.
    pub fn health(&self) -> Result<StoreHealth> {
        let (active_count, tombstoned_count, pending_ops_count) =
            self.repo.get_health_counts()?;

        let db_path = self.config.path().join("memory.db");
        let wal_path = self.config.path().join("memory.db-wal");

        let db_size_bytes = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
        let wal_size_bytes = std::fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);

        let is_healthy = self.index.is_healthy();

        Ok(StoreHealth {
            is_healthy,
            total_active_records: active_count,
            total_tombstoned_records: tombstoned_count,
            pending_operations_count: pending_ops_count,
            index_rebuild_required: !is_healthy,
            db_size_bytes,
            wal_size_bytes,
        })
    }

    /// Perform a resumable physical cleanup of tombstoned and expired memory records.
    pub fn vacuum_tombstones(&self, batch_size: usize) -> Result<usize> {
        if batch_size == 0 {
            return Ok(0);
        }

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let purged = self.repo.vacuum_tombstoned_records(batch_size, now_ms)?;
        if purged > 0 {
            self.rebuild_index()?;
        }
        Ok(purged)
    }

    /// Force a rebuild of the in-memory vector accelerator index from canonical active records.
    pub fn rebuild_index(&self) -> Result<()> {
        let active_records = self.repo.get_all_active_records()?;
        self.index.rebuild(Box::new(active_records.iter()))
    }

    /// Retrieve and format memories into a token-budgeted string for LLM context injection.
    pub fn format_context(
        &self,
        req: RecallRequest,
        options: ContextFormatOptions,
    ) -> Result<String> {
        options.validate()?;
        let response = self.recall(req)?;
        let mut output = String::new();

        if let Some(ref header) = options.header {
            if !header.is_empty() {
                output.push_str(header);
                output.push('\n');
            }
        }

        let mut current_token_estimate = estimate_tokens(&output);

        for item in response.items {
            let kind_str = format!("{:?}", item.memory.kind).to_lowercase();
            let line = options
                .template
                .replace("{kind}", &kind_str)
                .replace("{content}", &item.memory.content)
                .replace("{score:.2}", &format!("{:.2}", item.score))
                .replace("{score}", &format!("{:.4}", item.score));

            let line_tokens = estimate_tokens(&line);
            if current_token_estimate + line_tokens > options.max_tokens {
                break;
            }

            output.push_str(&line);
            output.push('\n');
            current_token_estimate += line_tokens;
        }

        if let Some(ref footer) = options.footer {
            if !footer.is_empty() {
                let footer_tokens = estimate_tokens(footer);
                if current_token_estimate + footer_tokens <= options.max_tokens {
                    output.push_str(footer);
                }
            }
        }

        Ok(output)
    }

    /// Create a zero-downtime, fully consistent backup of the memory store at `target_dir`.
    /// Backs up the SQLite database; the vector index is ephemeral and will be rebuilt on restore.
    pub fn backup(&self, target_dir: impl AsRef<std::path::Path>) -> Result<()> {
        let target = target_dir.as_ref();
        if !target.exists() {
            std::fs::create_dir_all(target).map_err(|e| MemoryError::IoError {
                code: ErrorCode::IoError,
                message: format!("Failed to create backup target directory: {}", e),
            })?;
        }

        let target_db = target.join("memory.db");
        if target_db.exists() {
            std::fs::remove_file(&target_db).map_err(|e| MemoryError::IoError {
                code: ErrorCode::IoError,
                message: format!("Failed to clear target db file: {}", e),
            })?;
        }

        // Perform SQLite online backup via VACUUM INTO (the canonical store)
        self.repo.backup_sqlite(&target_db)?;

        Ok(())
    }
}

fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    (text.len() as f32 / 4.0).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn recall_reuses_scope_lookup_results_for_scoring() {
        let dir = tempfile::tempdir().unwrap();
        let config = MemoryConfig::builder(dir.path())
            .dimensions(3)
            .embedding_model("test-model")
            .build()
            .unwrap();
        let store = MemoryStore::open(config).unwrap();
        let requested_scope = MemoryScope::new("tenant-a", "support").unwrap();
        let other_scope = MemoryScope::new("tenant-b", "support").unwrap();

        for index in 0..20 {
            store
                .remember(RememberRequest {
                    operation_id: None,
                    scope: other_scope.clone(),
                    content: format!("Other tenant memory {index}"),
                    embedding: vec![1.0, 0.0, 0.0],
                    kind: MemoryKind::Fact,
                    importance: 1.0,
                    expires_at_ms: None,
                    metadata: HashMap::new(),
                    source: HashMap::new(),
                })
                .unwrap();
        }

        let target_id = store
            .remember(RememberRequest {
                operation_id: None,
                scope: requested_scope.clone(),
                content: "Requested tenant memory".into(),
                embedding: vec![0.8, 0.6, 0.0],
                kind: MemoryKind::Preference,
                importance: 1.0,
                expires_at_ms: None,
                metadata: HashMap::new(),
                source: HashMap::new(),
            })
            .unwrap();

        let before_recall = store.repo.scope_lookup_count();
        let recall = store
            .recall(RecallRequest::new(
                requested_scope,
                vec![1.0, 0.0, 0.0],
                1,
            ))
            .unwrap();

        assert_eq!(recall.items.len(), 1);
        assert_eq!(recall.items[0].memory.id, target_id);
        assert_eq!(
            store.repo.scope_lookup_count() - before_recall,
            21,
            "each unique candidate, including cross-scope candidates cached as None, should be queried once"
        );
    }
}
