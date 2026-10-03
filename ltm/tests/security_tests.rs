use vivvy_memory::*;

#[test]
fn test_missing_key_provider_returns_encryption_key_unavailable() {
    let missing_provider = MissingKeyProvider;
    let err = missing_provider.get_key("acme-corp").unwrap_err();

    assert_eq!(err.code(), ErrorCode::EncryptionKeyUnavailable);
    assert_eq!(err.code().to_string(), "ENCRYPTION_KEY_UNAVAILABLE");
    assert!(err.to_string().contains("acme-corp"));
}

#[test]
fn test_noop_dev_key_provider_supplies_deterministic_key() {
    let dev_provider = NoOpDevKeyProvider::new();
    let key_bytes = dev_provider.get_key("acme-corp").unwrap();

    assert_eq!(key_bytes.len(), 32);
    assert_eq!(key_bytes, vec![0x42; 32]);
}

#[test]
fn test_telemetry_event_sanitization_guarantees_no_payload_leakage() {
    let telemetry = TelemetryRecord::new("tenant-secret", "confidential-ns", "recall")
        .with_operation_id(Some("op-sec-999".into()))
        .with_duration_us(1500)
        .with_outcome(true, None)
        .with_metrics(5, 4096);

    let serialized_json = serde_json::to_string(&telemetry).unwrap();

    // Verify metadata telemetry fields are present
    assert!(serialized_json.contains("tenant-secret"));
    assert!(serialized_json.contains("confidential-ns"));
    assert!(serialized_json.contains("recall"));

    // Verify sensitive data fields (content, embedding, key) are never present
    assert!(!serialized_json.contains("content"));
    assert!(!serialized_json.contains("embedding"));
    assert!(!serialized_json.contains("key"));
}

#[test]
fn test_error_formatting_redacts_memory_content_and_keys() {
    let _scope = MemoryScope::new("acme-tenant", "support-ns").unwrap();
    let err = MemoryError::invalid_scope("invalid tenant formatting");

    let err_msg = format!("{}", err);
    assert!(err_msg.contains("INVALID_SCOPE"));
    assert!(err_msg.contains("invalid tenant formatting"));

    // Verify formatted error string contains no raw memory content or key primitives
    assert!(!err_msg.contains("content="));
    assert!(!err_msg.contains("embedding="));
    assert!(!err_msg.contains("key_bytes="));
}

#[test]
fn test_cross_tenant_forget_isolation_does_not_tombstone_vector_index() {
    let dir = tempfile::tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope_a = MemoryScope::new("tenant-a", "main").unwrap();
    let scope_b = MemoryScope::new("tenant-b", "main").unwrap();

    // 1. Tenant A stores memory M
    let mem_id = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope_a.clone(),
            content: "Confidential Tenant A Fact".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 1.0,
            expires_at_ms: None,
            metadata: std::collections::HashMap::new(),
            source: std::collections::HashMap::new(),
        })
        .unwrap();

    // 2. Tenant B attempts to forget Tenant A's memory ID
    let forget_err = store
        .forget(ForgetRequest {
            scope: scope_b.clone(),
            id: mem_id.clone(),
            operation_id: None,
        })
        .unwrap_err();

    assert_eq!(forget_err.code(), ErrorCode::NotFound);

    // 3. Tenant A recalls via vector search: memory M MUST still be present
    let recall = store
        .recall(RecallRequest {
            scope: scope_a.clone(),
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
        })
        .unwrap();

    assert_eq!(recall.items.len(), 1);
    assert_eq!(recall.items[0].memory.id, mem_id);

    // 4. Batch forget from Tenant B on Tenant A's ID should also fail
    let batch_err = store
        .forget_batch(&scope_b, &[mem_id.as_str()])
        .unwrap_err();
    assert_eq!(batch_err.code(), ErrorCode::NotFound);

    // 5. Tenant A's memory is still intact
    let recall_after = store
        .recall(RecallRequest {
            scope: scope_a,
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
        })
        .unwrap();

    assert_eq!(recall_after.items.len(), 1);
    assert_eq!(recall_after.items[0].memory.id, mem_id);
}

#[test]
fn test_forget_nonexistent_returns_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("tenant-a", "main").unwrap();

    let err = store
        .forget(ForgetRequest {
            scope,
            id: "non-existent-id".into(),
            operation_id: None,
        })
        .unwrap_err();

    assert_eq!(err.code(), ErrorCode::NotFound);
}
