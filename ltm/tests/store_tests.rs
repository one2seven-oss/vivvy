use std::collections::HashMap;
use tempfile::tempdir;
use vivvy_memory::*;

fn remember_test_record(
    store: &MemoryStore,
    scope: &MemoryScope,
    content: impl Into<String>,
    embedding: Vec<f32>,
    kind: MemoryKind,
) -> String {
    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: content.into(),
            embedding,
            kind,
            importance: 1.0,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap()
}

#[test]
fn test_idempotent_remember() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "chat-support").unwrap();

    let req = RememberRequest {
        operation_id: Some("op-fixed-123".into()),
        scope: scope.clone(),
        content: "User loves dark mode.".into(),
        embedding: vec![1.0, 0.0, 0.0],
        kind: MemoryKind::Preference,
        importance: 0.9,
        expires_at_ms: None,
        metadata: HashMap::new(),
        source: HashMap::new(),
    };

    // First remember call
    let id1 = store.remember(req.clone()).unwrap();

    // Repeated remember call with same operation_id returns exact same ID
    let id2 = store.remember(req.clone()).unwrap();
    assert_eq!(id1, id2);

    // Verify record exists and is active
    let record = store.get(&scope, &id1).unwrap().expect("record found");
    assert_eq!(record.content, "User loves dark mode.");
    assert_eq!(record.status, MemoryStatus::Active);
}

#[test]
fn test_recall_and_explainability() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config.clone()).unwrap();
    let scope_a = MemoryScope::new("acme", "support").unwrap();
    let scope_b = MemoryScope::new("globex", "support").unwrap();

    // Insert record for scope_a
    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope_a.clone(),
            content: "Acme record".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Preference,
            importance: 0.8,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // Recall from scope_a finds the record
    let recall_a = store
        .recall(RecallRequest {
            scope: scope_a.clone(),
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: true,
            mmr_lambda: None,
            scoring: Default::default(),
        })
        .unwrap();

    assert_eq!(recall_a.items.len(), 1);
    assert_eq!(recall_a.items[0].memory.content, "Acme record");
    assert!(recall_a.items[0].explanation.is_some());
    let exp = recall_a.items[0].explanation.as_ref().unwrap();
    assert!(exp.total_score > 0.8);
    assert_eq!(exp.similarity_score, 1.0);

    // Recall from scope_b returns ZERO records (strict multi-tenant isolation)
    let recall_b = store
        .recall(RecallRequest {
            scope: scope_b,
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
            scoring: Default::default(),
        })
        .unwrap();

    assert_eq!(recall_b.items.len(), 0);
}

#[test]
fn test_recall_widens_candidates_past_other_scopes() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();
    let store = MemoryStore::open(config).unwrap();
    let requested_scope = MemoryScope::new("tenant-a", "support").unwrap();
    let other_scope = MemoryScope::new("tenant-b", "support").unwrap();

    for index in 0..20 {
        remember_test_record(
            &store,
            &other_scope,
            format!("Other tenant memory {index}"),
            vec![1.0, 0.0, 0.0],
            MemoryKind::Fact,
        );
    }
    let target_id = remember_test_record(
        &store,
        &requested_scope,
        "Requested tenant memory",
        vec![0.8, 0.6, 0.0],
        MemoryKind::Preference,
    );

    let recall = store
        .recall(RecallRequest::new(requested_scope, vec![1.0, 0.0, 0.0], 1))
        .unwrap();

    assert_eq!(recall.items.len(), 1);
    assert_eq!(recall.items[0].memory.id, target_id);
}

#[test]
fn test_recall_widens_candidates_past_filter_misses() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();
    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("tenant-a", "support").unwrap();

    for index in 0..20 {
        remember_test_record(
            &store,
            &scope,
            format!("Unfiltered memory {index}"),
            vec![1.0, 0.0, 0.0],
            MemoryKind::Fact,
        );
    }
    let target_id = remember_test_record(
        &store,
        &scope,
        "Filtered memory",
        vec![0.8, 0.6, 0.0],
        MemoryKind::Preference,
    );

    let mut request = RecallRequest::new(scope, vec![1.0, 0.0, 0.0], 1);
    request.filters = MemoryFilter::kind_in([MemoryKind::Preference]);
    let recall = store.recall(request).unwrap();

    assert_eq!(recall.items.len(), 1);
    assert_eq!(recall.items[0].memory.id, target_id);
}

#[test]
fn test_recall_widens_vector_search_past_tombstones() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();
    let store = MemoryStore::open(config).unwrap();
    let requested_scope = MemoryScope::new("tenant-a", "support").unwrap();
    let other_scope = MemoryScope::new("tenant-b", "support").unwrap();

    for index in 0..20 {
        let id = remember_test_record(
            &store,
            &other_scope,
            format!("Removed tenant memory {index}"),
            vec![1.0, 0.0, 0.0],
            MemoryKind::Fact,
        );
        store
            .forget(ForgetRequest {
                operation_id: None,
                scope: other_scope.clone(),
                id,
            })
            .unwrap();
    }
    let target_id = remember_test_record(
        &store,
        &requested_scope,
        "Requested tenant memory",
        vec![0.8, 0.6, 0.0],
        MemoryKind::Preference,
    );

    let recall = store
        .recall(RecallRequest::new(requested_scope, vec![1.0, 0.0, 0.0], 1))
        .unwrap();

    assert_eq!(recall.items.len(), 1);
    assert_eq!(recall.items[0].memory.id, target_id);
}

#[test]
fn test_crash_recovery_rebuild() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let scope = MemoryScope::new("tenant-x", "ns-1").unwrap();

    {
        let store = MemoryStore::open(config.clone()).unwrap();
        store
            .remember(RememberRequest {
                operation_id: Some("op-rec-1".into()),
                scope: scope.clone(),
                content: "Persistent fact".into(),
                embedding: vec![0.0, 1.0, 0.0],
                kind: MemoryKind::Fact,
                importance: 0.7,
                expires_at_ms: None,
                metadata: HashMap::new(),
                source: HashMap::new(),
            })
            .unwrap();
    }

    // Reopen store from disk
    {
        let store = MemoryStore::open(config).unwrap();
        let recall = store
            .recall(RecallRequest {
                scope: scope.clone(),
                query_embedding: vec![0.0, 1.0, 0.0],
                query_text: None,
                limit: 5,
                filters: MemoryFilter::default(),
                include_explanations: true,
                mmr_lambda: None,
                scoring: Default::default(),
            })
            .unwrap();

        assert_eq!(recall.items.len(), 1);
        assert_eq!(recall.items[0].memory.content, "Persistent fact");
    }
}

#[test]
fn test_update_revision_concurrency_and_forget() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config.clone()).unwrap();
    let scope = MemoryScope::new("acme", "chat").unwrap();

    let id = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Initial observation".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.5,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // Stale update with wrong revision fails with RevisionConflict
    let stale_res = store.update(UpdateRequest {
        operation_id: None,
        scope: scope.clone(),
        id: id.clone(),
        expected_revision: 99,
        content: Some("Wrong revision update".into()),
        embedding: None,
        kind: None,
        importance: None,
        expires_at_ms: None,
        metadata_patch: None,
    });
    assert!(matches!(
        stale_res.unwrap_err(),
        MemoryError::RevisionConflict { .. }
    ));

    // Valid update with expected_revision 1 succeeds and bumps to revision 2
    store
        .update(UpdateRequest {
            operation_id: None,
            scope: scope.clone(),
            id: id.clone(),
            expected_revision: 1,
            content: Some("Updated observation".into()),
            embedding: Some(vec![0.0, 1.0, 0.0]),
            kind: None,
            importance: Some(0.9),
            expires_at_ms: None,
            metadata_patch: None,
        })
        .unwrap();

    let updated = store.get(&scope, &id).unwrap().unwrap();
    assert_eq!(updated.content, "Updated observation");
    assert_eq!(updated.revision, 2);
    assert_eq!(updated.importance, 0.9);

    // Forget memory immediately hides it
    store
        .forget(ForgetRequest {
            operation_id: None,
            scope: scope.clone(),
            id: id.clone(),
        })
        .unwrap();

    let after_forget = store.get(&scope, &id).unwrap();
    assert_eq!(after_forget.unwrap().status, MemoryStatus::Deleted);

    // Recall returns 0 items
    let recall = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![0.0, 1.0, 0.0],
            query_text: None,
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
            scoring: Default::default(),
        })
        .unwrap();
    assert_eq!(recall.items.len(), 0);
}

#[test]
fn test_mmr_diversity_ranking() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "docs").unwrap();

    // Add 2 very similar records and 1 distinct record
    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Rust concurrency 1".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.9,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Rust concurrency 2 (near duplicate)".into(),
            embedding: vec![0.99, 0.01, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.89,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Python async IO".into(),
            embedding: vec![0.7, 0.7, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.85,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // Recall with MMR diversity (lambda = 0.5)
    let recall = store
        .recall(RecallRequest {
            scope,
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 2,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: Some(0.5),
            scoring: Default::default(),
        })
        .unwrap();

    assert_eq!(recall.items.len(), 2);
    assert_eq!(recall.items[0].memory.content, "Rust concurrency 1");
    // With MMR, the second selected item should be Python async IO rather than the duplicate
    assert_eq!(recall.items[1].memory.content, "Python async IO");
}

#[test]
fn test_hybrid_recall_fts_and_rrf() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "knowledge").unwrap();

    // Memory A: High semantic similarity to query embedding [1.0, 0.0, 0.0], generic content
    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "General information about system architecture".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.5,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // Memory B: Low semantic similarity [0.0, 0.0, 1.0], but contains exact keyword "Priya"
    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "User Priya prefers concise code reviews with performance metrics.".into(),
            embedding: vec![0.0, 0.0, 1.0],
            kind: MemoryKind::Preference,
            importance: 0.9,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // Recall with query_text = Some("Priya") and query_embedding = [1.0, 0.0, 0.0]
    let recall = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: Some("Priya".into()),
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: true,
            mmr_lambda: None,
            scoring: Default::default(),
        })
        .unwrap();

    // Both items must be present
    assert_eq!(recall.items.len(), 2);
    let fts_item = recall
        .items
        .iter()
        .find(|item| item.memory.content.contains("Priya"))
        .expect("Priya record found via FTS fusion");

    let notes = fts_item.explanation.as_ref().unwrap().policy_notes.clone();
    assert!(notes.contains(&"fts5_lexical_candidate".to_string()));
}

#[test]
fn test_remember_batch_and_forget_batch() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "batch-test").unwrap();

    // 1. Remember batch of 5 items
    let mut reqs = Vec::new();
    for i in 0..5 {
        reqs.push(RememberRequest {
            operation_id: Some(format!("op-batch-{}", i)),
            scope: scope.clone(),
            content: format!("Batch memory record {}", i),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.8,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        });
    }

    let ids = store.remember_batch(reqs).unwrap();
    assert_eq!(ids.len(), 5);

    // 2. Recall should find all 5 records
    let recall = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 10,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
            scoring: Default::default(),
        })
        .unwrap();
    assert_eq!(recall.items.len(), 5);

    // 3. Forget batch of first 3 items
    let to_delete: Vec<&str> = ids[0..3].iter().map(|s| s.as_str()).collect();
    store.forget_batch(&scope, &to_delete).unwrap();

    // 4. Recall should now only find remaining 2 records
    let recall_after = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 10,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
            scoring: Default::default(),
        })
        .unwrap();
    assert_eq!(recall_after.items.len(), 2);
}

#[test]
fn test_kv_metadata_filtering_in_recall() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "projects").unwrap();

    // Insert 5 records with metadata: {"project": "alpha"}
    for i in 0..5 {
        let mut meta = HashMap::new();
        meta.insert("project".to_string(), serde_json::Value::String("alpha".to_string()));
        meta.insert("env".to_string(), serde_json::Value::String("prod".to_string()));

        store
            .remember(RememberRequest {
                operation_id: None,
                scope: scope.clone(),
                content: format!("Alpha record {}", i),
                embedding: vec![1.0, 0.0, 0.0],
                kind: MemoryKind::Fact,
                importance: 0.8,
                expires_at_ms: None,
                metadata: meta,
                source: HashMap::new(),
            })
            .unwrap();
    }

    // Insert 5 records with metadata: {"project": "beta"}
    for i in 0..5 {
        let mut meta = HashMap::new();
        meta.insert("project".to_string(), serde_json::Value::String("beta".to_string()));
        meta.insert("env".to_string(), serde_json::Value::String("staging".to_string()));

        store
            .remember(RememberRequest {
                operation_id: None,
                scope: scope.clone(),
                content: format!("Beta record {}", i),
                embedding: vec![1.0, 0.0, 0.0],
                kind: MemoryKind::Fact,
                importance: 0.8,
                expires_at_ms: None,
                metadata: meta,
                source: HashMap::new(),
            })
            .unwrap();
    }

    // Recall with filter: project == "alpha"
    let filter_alpha = MemoryFilter::default().with_metadata_eq("project", "alpha");
    let recall_alpha = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 10,
            filters: filter_alpha,
            include_explanations: false,
            mmr_lambda: None,
            scoring: Default::default(),
        })
        .unwrap();

    assert_eq!(recall_alpha.items.len(), 5);
    for item in recall_alpha.items {
        assert!(item.memory.content.contains("Alpha"));
        assert_eq!(
            item.memory.metadata.get("project"),
            Some(&serde_json::Value::String("alpha".to_string()))
        );
    }

    // Recall with hybrid FTS text search + filter: project == "beta"
    let filter_beta = MemoryFilter::default().with_metadata_eq("project", "beta");
    let recall_beta = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: Some("Beta".to_string()),
            limit: 10,
            filters: filter_beta,
            include_explanations: false,
            mmr_lambda: None,
            scoring: Default::default(),
        })
        .unwrap();

    assert_eq!(recall_beta.items.len(), 5);
    for item in recall_beta.items {
        assert!(item.memory.content.contains("Beta"));
        assert_eq!(
            item.memory.metadata.get("project"),
            Some(&serde_json::Value::String("beta".to_string()))
        );
    }

    // Recall with non-matching metadata filter (project == "gamma") returns 0 items
    let filter_gamma = MemoryFilter::default().with_metadata_eq("project", "gamma");
    let recall_gamma = store
        .recall(RecallRequest {
            scope,
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 10,
            filters: filter_gamma,
            include_explanations: false,
            mmr_lambda: None,
            scoring: Default::default(),
        })
        .unwrap();

    assert_eq!(recall_gamma.items.len(), 0);
}

#[test]
fn test_format_context_token_budget() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "context-test").unwrap();

    // Insert 20 memories with long text
    for i in 0..20 {
        store
            .remember(RememberRequest {
                operation_id: None,
                scope: scope.clone(),
                content: format!("Long memory content record item number {} with detailed context information.", i),
                embedding: vec![1.0, 0.0, 0.0],
                kind: MemoryKind::Fact,
                importance: 0.8,
                expires_at_ms: None,
                metadata: HashMap::new(),
                source: HashMap::new(),
            })
            .unwrap();
    }

    let req = RecallRequest {
        scope,
        query_embedding: vec![1.0, 0.0, 0.0],
        query_text: None,
        limit: 20,
        filters: MemoryFilter::default(),
        include_explanations: false,
        mmr_lambda: None,
        scoring: Default::default(),
    };

    let options = ContextFormatOptions {
        max_tokens: 100,
        template: "- [{kind}] {content}".to_string(),
        header: Some("### System Context:".to_string()),
        footer: None,
    };

    let formatted = store.format_context(req, options).unwrap();

    assert!(formatted.starts_with("### System Context:"));
    // Heuristic: ~4 chars per token, so 100 tokens max ~400 chars
    let token_estimate = (formatted.len() as f32 / 4.0).ceil() as usize;
    assert!(token_estimate <= 100);
    assert!(formatted.ends_with('\n'));
}

#[test]
fn test_format_context_template_placeholders() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "template-test").unwrap();

    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "User prefers dark mode UI theme.".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Preference,
            importance: 0.9,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    let req = RecallRequest {
        scope,
        query_embedding: vec![1.0, 0.0, 0.0],
        query_text: None,
        limit: 5,
        filters: MemoryFilter::default(),
        include_explanations: false,
        mmr_lambda: None,
        scoring: Default::default(),
    };

    let options = ContextFormatOptions {
        max_tokens: 500,
        template: "* [{kind}] {content} (relevance: {score:.2})".to_string(),
        header: Some("=== MEMORY CONTEXT ===".to_string()),
        footer: Some("=== END CONTEXT ===".to_string()),
    };

    let formatted = store.format_context(req, options).unwrap();

    assert!(formatted.contains("=== MEMORY CONTEXT ==="));
    assert!(formatted.contains("* [preference] User prefers dark mode UI theme. (relevance:"));
    assert!(formatted.contains("=== END CONTEXT ==="));
}

#[test]
fn test_time_window_filtering_in_recall() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "temporal-test").unwrap();

    // Insert 3 records at T1 (1000ms), 3 at T2 (2000ms), 3 at T3 (3000ms)
    let mut ids_t1 = Vec::new();
    let mut ids_t2 = Vec::new();
    let mut ids_t3 = Vec::new();

    for i in 0..3 {
        let id = store
            .remember(RememberRequest {
                operation_id: None,
                scope: scope.clone(),
                content: format!("T1 record {}", i),
                embedding: vec![1.0, 0.0, 0.0],
                kind: MemoryKind::Fact,
                importance: 0.8,
                expires_at_ms: None,
                metadata: HashMap::new(),
                source: HashMap::new(),
            })
            .unwrap();
        ids_t1.push(id);
    }

    for i in 0..3 {
        let id = store
            .remember(RememberRequest {
                operation_id: None,
                scope: scope.clone(),
                content: format!("T2 record {}", i),
                embedding: vec![1.0, 0.0, 0.0],
                kind: MemoryKind::Fact,
                importance: 0.8,
                expires_at_ms: None,
                metadata: HashMap::new(),
                source: HashMap::new(),
            })
            .unwrap();
        ids_t2.push(id);
    }

    for i in 0..3 {
        let id = store
            .remember(RememberRequest {
                operation_id: None,
                scope: scope.clone(),
                content: format!("T3 record {}", i),
                embedding: vec![1.0, 0.0, 0.0],
                kind: MemoryKind::Fact,
                importance: 0.8,
                expires_at_ms: None,
                metadata: HashMap::new(),
                source: HashMap::new(),
            })
            .unwrap();
        ids_t3.push(id);
    }

    // Set timestamps in SQLite database
    let db_conn = rusqlite::Connection::open(dir.path().join("memory.db")).unwrap();
    for id in &ids_t1 {
        db_conn
            .execute("UPDATE memories SET created_at_ms = 1000 WHERE id = ?", rusqlite::params![id])
            .unwrap();
    }
    for id in &ids_t2 {
        db_conn
            .execute("UPDATE memories SET created_at_ms = 2000 WHERE id = ?", rusqlite::params![id])
            .unwrap();
    }
    for id in &ids_t3 {
        db_conn
            .execute("UPDATE memories SET created_at_ms = 3000 WHERE id = ?", rusqlite::params![id])
            .unwrap();
    }

    // 1. Filter exact window: created_after_ms = 2000, created_before_ms = 2000
    let filter_t2 = MemoryFilter::default().with_time_range(Some(2000), Some(2000));
    let recall_t2 = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 10,
            filters: filter_t2,
            include_explanations: false,
            mmr_lambda: None,
            scoring: Default::default(),
        })
        .unwrap();

    assert_eq!(recall_t2.items.len(), 3);
    for item in &recall_t2.items {
        assert_eq!(item.memory.created_at_ms, 2000);
        assert!(item.memory.content.contains("T2"));
    }

    // 2. Filter lower bound only: created_after_ms = 2000
    let filter_after_t2 = MemoryFilter::default().with_time_range(Some(2000), None);
    let recall_after_t2 = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 10,
            filters: filter_after_t2,
            include_explanations: false,
            mmr_lambda: None,
            scoring: Default::default(),
        })
        .unwrap();

    assert_eq!(recall_after_t2.items.len(), 6);
    for item in &recall_after_t2.items {
        assert!(item.memory.created_at_ms >= 2000);
    }

    // 3. Filter upper bound only: created_before_ms = 2000
    let filter_before_t2 = MemoryFilter::default().with_time_range(None, Some(2000));
    let recall_before_t2 = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 10,
            filters: filter_before_t2,
            include_explanations: false,
            mmr_lambda: None,
            scoring: Default::default(),
        })
        .unwrap();

    assert_eq!(recall_before_t2.items.len(), 6);
    for item in &recall_before_t2.items {
        assert!(item.memory.created_at_ms <= 2000);
    }
}

#[test]
fn test_time_window_empty_range() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "empty-range-test").unwrap();

    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Sample record".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.8,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // Range with created_after_ms (3000) > created_before_ms (1000)
    let filter_invalid = MemoryFilter::default().with_time_range(Some(3000), Some(1000));
    let recall = store
        .recall(RecallRequest {
            scope,
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 10,
            filters: filter_invalid,
            include_explanations: false,
            mmr_lambda: None,
            scoring: Default::default(),
        })
        .unwrap();

    assert_eq!(recall.items.len(), 0);
}

#[test]
fn test_fts_time_window_filtering() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "fts-temporal").unwrap();

    let id1 = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Quantum computing algorithms research paper".into(),
            embedding: vec![0.0, 0.0, 1.0],
            kind: MemoryKind::Fact,
            importance: 0.8,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    let id2 = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Quantum computing hardware breakthrough".into(),
            embedding: vec![0.0, 0.0, 1.0],
            kind: MemoryKind::Fact,
            importance: 0.8,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    let db_conn = rusqlite::Connection::open(dir.path().join("memory.db")).unwrap();
    db_conn
        .execute("UPDATE memories SET created_at_ms = 1000 WHERE id = ?", rusqlite::params![id1])
        .unwrap();
    db_conn
        .execute("UPDATE memories SET created_at_ms = 5000 WHERE id = ?", rusqlite::params![id2])
        .unwrap();

    // Query FTS for "Quantum" with time window created_after_ms = 4000
    let filter = MemoryFilter::default().with_time_range(Some(4000), None);
    let recall = store
        .recall(RecallRequest {
            scope,
            query_embedding: vec![0.0, 0.0, 1.0],
            query_text: Some("Quantum".into()),
            limit: 10,
            filters: filter,
            include_explanations: true,
            mmr_lambda: None,
            scoring: Default::default(),
        })
        .unwrap();

    assert_eq!(recall.items.len(), 1);
    assert_eq!(recall.items[0].memory.id, id2);
}

#[test]
fn test_stale_index_files_cleaned_up_on_open() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    // Create store and add some data
    let store = MemoryStore::open(config.clone()).unwrap();
    let scope = MemoryScope::new("acme", "test").unwrap();
    store
        .remember(RememberRequest {
            operation_id: None,
            scope,
            content: "Test record".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.8,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();
    drop(store);

    // Create stale index files that might be left from older versions
    let index_wal = dir.path().join("index.wal");
    let index_manifest = dir.path().join("index.manifest");
    let segment_file = dir.path().join("seg-001.vivvy");

    std::fs::write(&index_wal, "stale").unwrap();
    std::fs::write(&index_manifest, "stale").unwrap();
    std::fs::write(&segment_file, "stale").unwrap();

    // Verify stale files exist before opening
    assert!(index_wal.exists());
    assert!(index_manifest.exists());
    assert!(segment_file.exists());

    // Open store again — should clean up stale files
    let store = MemoryStore::open(config).unwrap();

    // Verify stale files are removed
    assert!(!index_wal.exists(), "index.wal should be cleaned up");
    assert!(!index_manifest.exists(), "index.manifest should be cleaned up");
    assert!(!segment_file.exists(), "segment files should be cleaned up");

    // Verify store still works correctly
    let scope = MemoryScope::new("acme", "test").unwrap();
    let recall = store
        .recall(RecallRequest {
            scope,
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 10,
            filters: Default::default(),
            include_explanations: false,
            mmr_lambda: None,
            scoring: Default::default(),
        })
        .unwrap();
    assert_eq!(recall.items.len(), 1);
}

#[test]
fn test_custom_scoring_weights_and_explanation_diagnostics() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "test").unwrap();

    let mem_id = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Domain fact with importance".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.8,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // 1. Recall with default scoring -> verify explanation contains default_v1_scoring
    let default_recall = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 5,
            filters: Default::default(),
            include_explanations: true,
            mmr_lambda: None,
            scoring: ScoringConfig::default(),
        })
        .unwrap();

    assert_eq!(default_recall.items.len(), 1);
    let default_expl = default_recall.items[0].explanation.as_ref().unwrap();
    assert!(
        default_expl.policy_notes.contains(&"default_v1_scoring".to_string()),
        "Expected default_v1_scoring in policy notes, got {:?}",
        default_expl.policy_notes
    );
    assert_eq!(default_recall.items[0].memory.id, mem_id);

    // 2. Recall with custom scoring (similarity 0.90, importance 0.10, recency 0.0, half_life 14.0d)
    let custom_scoring = ScoringConfig::builder()
        .similarity(0.90)
        .importance(0.10)
        .recency(0.0)
        .reinforcement(0.0)
        .half_life_days(14.0)
        .build()
        .unwrap();

    let custom_recall = store
        .recall(RecallRequest {
            scope,
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 5,
            filters: Default::default(),
            include_explanations: true,
            mmr_lambda: None,
            scoring: custom_scoring,
        })
        .unwrap();

    assert_eq!(custom_recall.items.len(), 1);
    let custom_expl = custom_recall.items[0].explanation.as_ref().unwrap();
    assert!(
        custom_expl
            .policy_notes
            .iter()
            .any(|note| note.starts_with("custom_scoring(")),
        "Expected custom_scoring note, got {:?}",
        custom_expl.policy_notes
    );

    // Similarity is 1.0 (cosine orthogonal 0 distance -> sim 1.0), importance is 0.8
    // Total score = 0.90 * 1.0 + 0.10 * 0.8 = 0.90 + 0.08 = 0.98
    let expected_score = 0.90 * custom_expl.similarity_score + 0.10 * custom_expl.importance_score;
    assert!((custom_expl.total_score - expected_score).abs() < 1e-4);
}

#[test]
fn test_default_scoring_bit_identical_to_legacy_calculation() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "bit-exact-test").unwrap();

    let mem_id = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Memory for bit-for-bit equivalence".into(),
            embedding: vec![0.8, 0.6, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.65,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    let recall = store
        .recall(RecallRequest {
            scope,
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 5,
            filters: Default::default(),
            include_explanations: true,
            mmr_lambda: None,
            scoring: ScoringConfig::default(),
        })
        .unwrap();

    assert_eq!(recall.items.len(), 1);
    let item = &recall.items[0];
    assert_eq!(item.memory.id, mem_id);
    let expl = item.explanation.as_ref().unwrap();

    // Recompute legacy formula using explanation factors:
    let legacy_score = 0.70 * expl.similarity_score
        + 0.15 * expl.importance_score
        + 0.10 * expl.recency_score
        + 0.05 * expl.reinforcement_score;

    assert_eq!(
        item.score.to_bits(),
        legacy_score.to_bits(),
        "Default scoring must be bit-for-bit identical to legacy constant calculation"
    );
}

#[test]
fn test_zero_recency_weight_ensures_memory_age_has_zero_impact() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "zero-recency-test").unwrap();

    // Insert two memories with identical embeddings and importance
    let fresh_id = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Fresh memory".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.8,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    let old_id = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Old memory".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.8,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // Artificially age the old memory by 90 days in SQLite
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    let db_conn = rusqlite::Connection::open(dir.path().join("memory.db")).unwrap();
    db_conn
        .execute(
            "UPDATE memories SET created_at_ms = ? WHERE id = ?",
            rusqlite::params![now_ms - 90 * 86_400_000, old_id],
        )
        .unwrap();

    // 1. With default scoring (recency weight = 0.10), fresh memory scores strictly higher than old memory
    let default_recall = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 5,
            filters: Default::default(),
            include_explanations: true,
            mmr_lambda: None,
            scoring: ScoringConfig::default(),
        })
        .unwrap();

    assert_eq!(default_recall.items.len(), 2);
    let fresh_item = default_recall.items.iter().find(|i| i.memory.id == fresh_id).unwrap();
    let old_item = default_recall.items.iter().find(|i| i.memory.id == old_id).unwrap();
    assert!(
        fresh_item.score > old_item.score,
        "Fresh memory should rank higher under default recency decay"
    );

    // 2. With weight_recency = 0.0, memory age has zero impact on total_score
    let no_recency_scoring = ScoringConfig::builder()
        .similarity(0.80)
        .importance(0.20)
        .recency(0.0)
        .reinforcement(0.0)
        .build()
        .unwrap();

    let no_recency_recall = store
        .recall(RecallRequest {
            scope,
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 5,
            filters: Default::default(),
            include_explanations: true,
            mmr_lambda: None,
            scoring: no_recency_scoring,
        })
        .unwrap();

    assert_eq!(no_recency_recall.items.len(), 2);
    let fresh_score = no_recency_recall.items.iter().find(|i| i.memory.id == fresh_id).unwrap().score;
    let old_score = no_recency_recall.items.iter().find(|i| i.memory.id == old_id).unwrap().score;
    assert_eq!(
        fresh_score.to_bits(),
        old_score.to_bits(),
        "When weight_recency is 0.0, total_score must be identical regardless of age"
    );
}

#[test]
fn test_recency_half_life_days_decay_rate() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "decay-rate-test").unwrap();

    let id = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Two days old memory".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.5,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // Set memory age to exactly 2 days ago
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    let db_conn = rusqlite::Connection::open(dir.path().join("memory.db")).unwrap();
    db_conn
        .execute(
            "UPDATE memories SET created_at_ms = ? WHERE id = ?",
            rusqlite::params![now_ms - 2 * 86_400_000, id],
        )
        .unwrap();

    // Recall with 7-day half-life (default)
    let recall_7d = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 5,
            filters: Default::default(),
            include_explanations: true,
            mmr_lambda: None,
            scoring: ScoringConfig::builder()
                .similarity(0.5)
                .importance(0.0)
                .recency(0.5)
                .reinforcement(0.0)
                .half_life_days(7.0)
                .build()
                .unwrap(),
        })
        .unwrap();

    // Recall with 1-day half-life
    let recall_1d = store
        .recall(RecallRequest {
            scope,
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 5,
            filters: Default::default(),
            include_explanations: true,
            mmr_lambda: None,
            scoring: ScoringConfig::builder()
                .similarity(0.5)
                .importance(0.0)
                .recency(0.5)
                .reinforcement(0.0)
                .half_life_days(1.0)
                .build()
                .unwrap(),
        })
        .unwrap();

    let recency_score_7d = recall_7d.items[0].explanation.as_ref().unwrap().recency_score;
    let recency_score_1d = recall_1d.items[0].explanation.as_ref().unwrap().recency_score;

    assert!(
        recency_score_1d < recency_score_7d,
        "Memory must decay faster with 1.0 day half-life ({recency_score_1d}) than 7.0 days ({recency_score_7d})"
    );
    // 2 days with 1-day half life is 2 half-lives: exp(-2) ≈ 0.1353
    // 2 days with 7-day half life is 2/7 half-life: exp(-2/7) ≈ 0.7515
    assert!((recency_score_1d - (-2.0f32 / 1.0).exp()).abs() < 0.05);
    assert!((recency_score_7d - (-2.0f32 / 7.0).exp()).abs() < 0.05);
}

#[test]
fn test_format_context_and_mmr_with_custom_scoring() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "pipeline-test").unwrap();

    // Memory A: high importance (1.0), lower similarity (orthogonal query)
    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "High importance rule".into(),
            embedding: vec![0.0, 1.0, 0.0],
            kind: MemoryKind::Instruction,
            importance: 1.0,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // Memory B: lower importance (0.1), high similarity (parallel to query)
    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "High similarity fact".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.1,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // Scoring where importance dominates (weight_importance: 0.95, weight_similarity: 0.05)
    let importance_heavy_scoring = ScoringConfig::builder()
        .similarity(0.05)
        .importance(0.95)
        .recency(0.0)
        .reinforcement(0.0)
        .build()
        .unwrap();

    let req = RecallRequest {
        scope,
        query_embedding: vec![1.0, 0.0, 0.0],
        query_text: None,
        limit: 2,
        filters: Default::default(),
        include_explanations: true,
        mmr_lambda: None,
        scoring: importance_heavy_scoring,
    };

    let formatted = store
        .format_context(
            req,
            ContextFormatOptions {
                max_tokens: 500,
                template: "- [{kind}] {content} (score: {score:.2})".into(),
                header: None,
                footer: None,
            },
        )
        .unwrap();

    // First formatted line should be the High importance rule because its total score is higher
    let first_line = formatted.lines().next().unwrap();
    assert!(
        first_line.contains("High importance rule"),
        "Context format should reflect custom scoring ranking, got: {first_line}"
    );
}

#[test]
fn test_refuse_to_mix_models_and_dimensions_on_reopen() {
    let dir = tempfile::tempdir().unwrap();

    let cfg1 = MemoryConfig::builder(dir.path())
        .dimensions(4)
        .embedding_model("model-v1")
        .build()
        .unwrap();

    // 1. Open and write a memory
    {
        let store = MemoryStore::open(cfg1).unwrap();
        let scope = MemoryScope::new("t1", "ns1").unwrap();
        store
            .remember(RememberRequest {
                operation_id: None,
                scope,
                content: "Fact 1".into(),
                embedding: vec![0.1, 0.2, 0.3, 0.4],
                kind: MemoryKind::Fact,
                importance: 0.8,
                expires_at_ms: None,
                metadata: HashMap::new(),
                source: HashMap::new(),
            })
            .unwrap();
    }

    // 2. Reopen with different embedding_model -> must fail
    let cfg_diff_model = MemoryConfig::builder(dir.path())
        .dimensions(4)
        .embedding_model("model-v2")
        .build()
        .unwrap();
    let err_model = MemoryStore::open(cfg_diff_model).err().unwrap();
    assert!(
        err_model.to_string().contains("refuse to mix incompatible embedding models"),
        "Unexpected error: {err_model}"
    );

    // 3. Reopen with different dimensions -> must fail
    let cfg_diff_dims = MemoryConfig::builder(dir.path())
        .dimensions(8)
        .embedding_model("model-v1")
        .build()
        .unwrap();
    let err_dims = MemoryStore::open(cfg_diff_dims).err().unwrap();
    assert!(
        err_dims.to_string().contains("refuse to mix incompatible dimensions"),
        "Unexpected error: {err_dims}"
    );

    // 4. Reopen with matching config -> must succeed
    let cfg_matching = MemoryConfig::builder(dir.path())
        .dimensions(4)
        .embedding_model("model-v1")
        .build()
        .unwrap();
    assert!(MemoryStore::open(cfg_matching).is_ok());
}

#[test]
fn test_weight_change_flips_ranking() {
    let dir = tempfile::tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(2)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "test").unwrap();

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    // Memory 1: High similarity (1.0), but old (14 days ago) and low importance (0.1)
    let fourteen_days_ms = 14 * 86_400_000;
    let old_high_sim_id = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Old high-similarity document".into(),
            embedding: vec![1.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.1,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // Manually age Memory 1 back by 14 days in the database
    {
        let db_path = dir.path().join("memory.db");
        let conn = rusqlite::Connection::open(db_path).unwrap();
        conn.execute(
            "UPDATE memories SET created_at_ms = ? WHERE id = ?",
            rusqlite::params![now_ms - fourteen_days_ms, old_high_sim_id],
        )
        .unwrap();
    }

    // Memory 2: Moderate similarity, but fresh (created right now) and high importance (1.0)
    let fresh_high_imp_id = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Fresh high-importance document".into(),
            embedding: vec![0.0, 1.0], // orthogonal direction -> lower cosine similarity
            kind: MemoryKind::Fact,
            importance: 1.0,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    let query_embedding = vec![1.0, 0.0];

    // Case A: Query with DEFAULT scoring weights (similarity 0.70 dominates)
    // Memory 1 should win because of high vector match despite being old.
    let default_recall = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: query_embedding.clone(),
            query_text: None,
            limit: 2,
            filters: Default::default(),
            include_explanations: true,
            mmr_lambda: None,
            scoring: ScoringConfig::default(),
        })
        .unwrap();

    assert_eq!(default_recall.items.len(), 2);
    assert_eq!(
        default_recall.items[0].memory.id, old_high_sim_id,
        "Under default weights, high similarity memory should rank #1"
    );

    // Case B: Query with RECENCY & IMPORTANCE tuned scoring
    // (similarity: 0.10, importance: 0.45, recency: 0.45, half_life: 3.0 days)
    let tuned_scoring = ScoringConfig::builder()
        .similarity(0.10)
        .importance(0.45)
        .recency(0.45)
        .reinforcement(0.0)
        .half_life_days(3.0)
        .build()
        .unwrap();

    let tuned_recall = store
        .recall(RecallRequest {
            scope,
            query_embedding,
            query_text: None,
            limit: 2,
            filters: Default::default(),
            include_explanations: true,
            mmr_lambda: None,
            scoring: tuned_scoring,
        })
        .unwrap();

    assert_eq!(tuned_recall.items.len(), 2);
    assert_eq!(
        tuned_recall.items[0].memory.id, fresh_high_imp_id,
        "Under recency/importance-heavy weights, ranking should invert and fresh memory should rank #1"
    );
}

#[test]
fn test_negative_scoring_weights_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(2)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "test").unwrap();

    let invalid_config = ScoringConfig {
        weight_similarity: -0.5,
        ..Default::default()
    };

    let err = store
        .recall(RecallRequest {
            scope,
            query_embedding: vec![1.0, 0.0],
            query_text: None,
            limit: 5,
            filters: Default::default(),
            include_explanations: false,
            mmr_lambda: None,
            scoring: invalid_config,
        })
        .unwrap_err();

    assert!(err.to_string().contains("must be non-negative"));
}

#[test]
fn test_default_scoring_preserves_ranking() {
    let dir = tempfile::tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(2)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "test").unwrap();

    let id = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Parity verification doc".into(),
            embedding: vec![1.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.8,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    let recall = store
        .recall(RecallRequest {
            scope,
            query_embedding: vec![1.0, 0.0],
            query_text: None,
            limit: 1,
            filters: Default::default(),
            include_explanations: true,
            mmr_lambda: None,
            scoring: ScoringConfig::default(),
        })
        .unwrap();

    assert_eq!(recall.items.len(), 1);
    let item = &recall.items[0];
    assert_eq!(item.memory.id, id);

    let explanation = item.explanation.as_ref().expect("explanation included");
    assert!(
        explanation.policy_notes.contains(&"default_v1_scoring".to_string()),
        "Expected default_v1_scoring policy note"
    );

    // Exact parity formula check: 0.70 * sim + 0.15 * imp + 0.10 * rec + 0.05 * reinf
    let expected_score = 0.70 * explanation.similarity_score
        + 0.15 * explanation.importance_score
        + 0.10 * explanation.recency_score
        + 0.05 * explanation.reinforcement_score;
    assert!(
        (item.score - expected_score).abs() < 1e-6,
        "Score {} does not match expected {}",
        item.score,
        expected_score
    );
}

#[test]
fn test_zero_similarity_weight_ablation() {
    let dir = tempfile::tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(2)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "test").unwrap();

    // Two documents with completely different vector directions (sim 1.0 vs sim 0.5)
    // but identical importance and created at virtually the same instant
    let _id_aligned = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Aligned vector".into(),
            embedding: vec![1.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.6,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    let _id_orthogonal = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Orthogonal vector".into(),
            embedding: vec![0.0, 1.0],
            kind: MemoryKind::Fact,
            importance: 0.6,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // With zero similarity weight, vector distance is ignored; importance dominates
    let ablation_scoring = ScoringConfig::builder()
        .similarity(0.0)
        .importance(1.0)
        .recency(0.0)
        .reinforcement(0.0)
        .build()
        .unwrap();

    let recall = store
        .recall(RecallRequest {
            scope,
            query_embedding: vec![1.0, 0.0],
            query_text: None,
            limit: 2,
            filters: Default::default(),
            include_explanations: true,
            mmr_lambda: None,
            scoring: ablation_scoring,
        })
        .unwrap();

    assert_eq!(recall.items.len(), 2);
    // Both items should have identical total score (1.0 * 0.6 = 0.6)
    assert!(
        (recall.items[0].score - 0.6).abs() < 1e-5,
        "Expected score 0.6, got {}",
        recall.items[0].score
    );
    assert!(
        (recall.items[1].score - 0.6).abs() < 1e-5,
        "Expected score 0.6, got {}",
        recall.items[1].score
    );
    assert!(
        (recall.items[0].score - recall.items[1].score).abs() < 1e-6,
        "Scores should be identical regardless of vector alignment"
    );
}

#[test]
fn test_half_life_decay_sensitivity() {
    let dir = tempfile::tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(2)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "test").unwrap();

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    // Memory aged 5 days ago
    let five_days_ms = 5 * 86_400_000;
    let mem_id = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Five days old memory".into(),
            embedding: vec![1.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.5,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    {
        let db_path = dir.path().join("memory.db");
        let conn = rusqlite::Connection::open(db_path).unwrap();
        conn.execute(
            "UPDATE memories SET created_at_ms = ? WHERE id = ?",
            rusqlite::params![now_ms - five_days_ms, mem_id],
        )
        .unwrap();
    }

    // Recall with long half-life (14 days)
    let long_halflife_scoring = ScoringConfig::builder()
        .similarity(0.0)
        .importance(0.0)
        .recency(1.0)
        .reinforcement(0.0)
        .half_life_days(14.0)
        .build()
        .unwrap();

    let recall_long = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![1.0, 0.0],
            query_text: None,
            limit: 1,
            filters: Default::default(),
            include_explanations: true,
            mmr_lambda: None,
            scoring: long_halflife_scoring,
        })
        .unwrap();

    // Recall with short half-life (2 days)
    let short_halflife_scoring = ScoringConfig::builder()
        .similarity(0.0)
        .importance(0.0)
        .recency(1.0)
        .reinforcement(0.0)
        .half_life_days(2.0)
        .build()
        .unwrap();

    let recall_short = store
        .recall(RecallRequest {
            scope,
            query_embedding: vec![1.0, 0.0],
            query_text: None,
            limit: 1,
            filters: Default::default(),
            include_explanations: true,
            mmr_lambda: None,
            scoring: short_halflife_scoring,
        })
        .unwrap();

    let score_long = recall_long.items[0].score;
    let score_short = recall_short.items[0].score;

    assert!(
        score_long > score_short,
        "Longer half-life ({score_long}) should retain higher recency than shorter half-life ({score_short})"
    );
    assert!(
        (score_long - (-5.0f32 / 14.0f32).exp()).abs() < 0.05,
        "Score should match exp(-5/14)"
    );
    assert!(
        (score_short - (-5.0f32 / 2.0f32).exp()).abs() < 0.05,
        "Score should match exp(-5/2)"
    );
}

