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

// ---------------------------------------------------------------------------
// Encryption-at-rest (field-level, per-tenant) integration tests
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::Arc;

const SECRET: &str = "the user's social security number is 078-05-1120";

fn encrypted_config(dir: &std::path::Path) -> MemoryConfig {
    MemoryConfig::builder(dir)
        .dimensions(3)
        .embedding_model("test-model")
        .key_provider(Arc::new(NoOpDevKeyProvider::new()))
        .build()
        .unwrap()
}

fn remember(store: &MemoryStore, scope: &MemoryScope, content: &str) -> String {
    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: content.into(),
            embedding: vec![0.1, 0.2, 0.3],
            kind: MemoryKind::Fact,
            importance: 0.7,
            expires_at_ms: None,
            metadata: HashMap::from([(
                "pii".to_string(),
                serde_json::Value::String(content.into()),
            )]),
            source: HashMap::new(),
        })
        .unwrap()
}

#[test]
fn test_encrypted_round_trip_returns_plaintext() {
    let dir = tempfile::tempdir().unwrap();
    let store = MemoryStore::open(encrypted_config(dir.path())).unwrap();
    let scope = MemoryScope::new("acme", "support").unwrap();

    let id = remember(&store, &scope, SECRET);

    // get() returns decrypted content and metadata
    let got = store.get(&scope, &id).unwrap().unwrap();
    assert_eq!(got.content, SECRET);
    assert_eq!(
        got.metadata.get("pii").unwrap(),
        &serde_json::Value::String(SECRET.into())
    );

    // recall() (vector path) also returns decrypted content
    let recall = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![0.1, 0.2, 0.3],
            query_text: None,
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
        })
        .unwrap();
    assert_eq!(recall.items.len(), 1);
    assert_eq!(recall.items[0].memory.content, SECRET);
}

#[test]
fn test_plaintext_never_hits_disk_when_encrypted() {
    let dir = tempfile::tempdir().unwrap();
    let store = MemoryStore::open(encrypted_config(dir.path())).unwrap();
    let scope = MemoryScope::new("acme", "support").unwrap();
    remember(&store, &scope, SECRET);

    // Flush a fully-consistent copy of the canonical store via VACUUM INTO.
    let backup_dir = tempfile::tempdir().unwrap();
    store.backup(backup_dir.path()).unwrap();

    let bytes = std::fs::read(backup_dir.path().join("memory.db")).unwrap();
    assert!(
        !contains(&bytes, SECRET.as_bytes()),
        "plaintext content leaked into the on-disk database"
    );
}

#[test]
fn test_opening_encrypted_store_without_key_fails() {
    let dir = tempfile::tempdir().unwrap();
    {
        let store = MemoryStore::open(encrypted_config(dir.path())).unwrap();
        let scope = MemoryScope::new("acme", "support").unwrap();
        remember(&store, &scope, SECRET);
    }

    // Reopen the same directory with no key provider: must be refused.
    let plain = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();
    match MemoryStore::open(plain) {
        Err(e) => assert_eq!(e.code(), ErrorCode::EncryptionKeyUnavailable),
        Ok(_) => panic!("expected opening an encrypted store without a key to fail"),
    }
}

#[test]
fn test_wrong_key_cannot_decrypt() {
    let dir = tempfile::tempdir().unwrap();
    let id;
    {
        let cfg = MemoryConfig::builder(dir.path())
            .dimensions(3)
            .embedding_model("test-model")
            .key_provider(Arc::new(NoOpDevKeyProvider::with_custom_key(vec![0xAA; 32])))
            .build()
            .unwrap();
        let store = MemoryStore::open(cfg).unwrap();
        let scope = MemoryScope::new("acme", "support").unwrap();
        id = remember(&store, &scope, SECRET);
    }

    // Reopen with a different key: AEAD must reject the ciphertext.
    let cfg = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .key_provider(Arc::new(NoOpDevKeyProvider::with_custom_key(vec![0xBB; 32])))
        .build()
        .unwrap();
    let store = MemoryStore::open(cfg);
    // Startup replay rebuilds the index from all active rows, which must
    // decrypt embeddings; a wrong key makes open() itself fail.
    match store {
        Err(e) => assert_eq!(e.code(), ErrorCode::DecryptionFailed),
        Ok(store) => {
            let scope = MemoryScope::new("acme", "support").unwrap();
            let err = store.get(&scope, &id).unwrap_err();
            assert_eq!(err.code(), ErrorCode::DecryptionFailed);
        }
    }
}

#[test]
fn test_blind_index_lexical_recall_under_encryption() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .key_provider(Arc::new(NoOpDevKeyProvider::new()))
        .lexical_mode(LexicalMode::BlindIndex)
        .build()
        .unwrap();
    let store = MemoryStore::open(cfg).unwrap();
    let scope = MemoryScope::new("acme", "support").unwrap();

    remember(&store, &scope, "user prefers concise technical answers");
    remember(&store, &scope, "completely unrelated episodic note");

    let recall = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![0.9, 0.9, 0.9], // far from stored vectors
            query_text: Some("technical".into()),
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
        })
        .unwrap();

    assert!(
        recall.items.iter().any(|i| i.memory.content.contains("technical")),
        "blind-index keyword recall should surface the matching memory"
    );

    // And the token index must not store the plaintext keyword.
    let backup_dir = tempfile::tempdir().unwrap();
    store.backup(backup_dir.path()).unwrap();
    let bytes = std::fs::read(backup_dir.path().join("memory.db")).unwrap();
    assert!(!contains(&bytes, b"technical"));
}

#[test]
fn test_legacy_plaintext_rows_readable_after_enabling_encryption() {
    let dir = tempfile::tempdir().unwrap();
    let legacy_id;
    {
        // Write a row with encryption OFF (enc_version = 0).
        let cfg = MemoryConfig::builder(dir.path())
            .dimensions(3)
            .embedding_model("test-model")
            .build()
            .unwrap();
        let store = MemoryStore::open(cfg).unwrap();
        let scope = MemoryScope::new("acme", "support").unwrap();
        legacy_id = remember(&store, &scope, "legacy plaintext memory");
    }

    // Reopen the same store WITH encryption enabled.
    let store = MemoryStore::open(encrypted_config(dir.path())).unwrap();
    let scope = MemoryScope::new("acme", "support").unwrap();

    // The legacy row still reads correctly...
    let got = store.get(&scope, &legacy_id).unwrap().unwrap();
    assert_eq!(got.content, "legacy plaintext memory");

    // ...and new rows are written sealed.
    let new_id = remember(&store, &scope, SECRET);
    assert_eq!(store.get(&scope, &new_id).unwrap().unwrap().content, SECRET);

    let backup_dir = tempfile::tempdir().unwrap();
    store.backup(backup_dir.path()).unwrap();
    let bytes = std::fs::read(backup_dir.path().join("memory.db")).unwrap();
    assert!(!contains(&bytes, SECRET.as_bytes()));
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}
