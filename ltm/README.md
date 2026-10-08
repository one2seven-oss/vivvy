# vivvy-memory

**Local, durable, namespace-isolated long-term memory for AI agents.** SQLite WAL is the canonical store of record; a [`vivvy-core`](https://crates.io/crates/vivvy-core) HNSW index rides alongside as a rebuildable accelerator — if it's ever lost or corrupted, it's regenerated from SQLite on boot, never the other way around.

`vivvy-memory` is the durable runtime layer of [Vivvy](https://github.com/one2seven-oss/vivvy). If you just need raw vector search with your own storage, [`vivvy-core`](https://crates.io/crates/vivvy-core) alone may be enough; reach for `vivvy-memory` when you want multi-tenant isolation, crash recovery, hybrid recall, and LLM-context formatting out of the box.

## Features

- **Multi-Tenant Isolation** — every record is scoped by tenant, namespace, and optionally agent/user; scope boundaries are enforced at the SQL layer, not just in application code.
- **Canonical SQLite WAL** — crash-safe durable storage with a 2-phase operation journal, so a crash mid-write never leaves the vector index and the record store disagreeing.
- **Hybrid Recall** — dense HNSW vector search fused with SQLite FTS5 lexical search via Reciprocal Rank Fusion (RRF).
- **Explainable Scoring & MMR** — a 4-factor weighted score (similarity, importance, recency, reinforcement) with an optional Maximal Marginal Relevance pass for result diversity.
- **Token-Budgeted Context Formatting** — render recall results straight into an LLM prompt with a custom template and a hard token budget.
- **Live Online Backups** — crash-consistent SQLite snapshots without stopping operations; vector index is ephemeral and rebuilt on restore.
- **Encryption at Rest** — optional per-tenant, field-level encryption (XChaCha20-Poly1305) of content, embeddings, metadata, and the operation journal, keyed via a pluggable `KeyProvider`. Scope, timestamps, and other query keys stay in plaintext so filtering and indexing keep working.

## Encryption at rest

Encryption is opt-in. Supply a [`KeyProvider`] and sensitive fields are sealed
per-tenant before they touch disk; non-sensitive columns used for querying
(ids, scope, timestamps, kind, status, importance) stay plaintext. Each sealed
field carries its own random nonce and is bound to its `(tenant, record, field)`
via AEAD additional data, so values cannot be swapped between rows or tenants.

```rust
use std::sync::Arc;
use vivvy_memory::{MemoryConfig, MemoryStore, NoOpDevKeyProvider, LexicalMode};

let config = MemoryConfig::builder("./agent_data")
    .dimensions(1536)
    .embedding_model("text-embedding-3-small")
    // Bring your own KeyProvider in production; NoOpDevKeyProvider is for tests.
    .key_provider(Arc::new(NoOpDevKeyProvider::new()))
    // Optional: enable keyed blind-index keyword recall. Defaults to Disabled.
    .lexical_mode(LexicalMode::BlindIndex)
    .build()?;
let store = MemoryStore::open(config)?;
# Ok::<(), vivvy_memory::MemoryError>(())
```

**Lexical recall trade-off.** Because `content` is encrypted, the plaintext FTS5
index would otherwise leak it. The `LexicalMode` controls this:

| Mode | Meaning |
| :--- | :--- |
| `Disabled` *(default when encrypted)* | No keyword index; recall is pure-vector. Leak-free. |
| `BlindIndex` | Keyword tokens are stored as keyed HMACs — exact-match keyword recall with no plaintext on disk. No BM25 ranking/phrase/prefix, and token frequency is observable to anyone with the database file. |
| `Plaintext` *(default when unencrypted)* | Normal FTS5. Rejected for an encrypted store. |

Once a store has been opened with encryption, it is recorded in the store
metadata and can no longer be opened without a `KeyProvider`. Existing
unencrypted stores can be upgraded in place: legacy rows stay readable and new
writes are sealed.

> **Not yet covered:** key rotation / re-encryption of existing rows, encryption
> of the SQLite WAL and temp files (a whole-database layer such as SQLCipher
> would be needed for those), and `content_hash`, which remains a
> non-cryptographic dedup digest.

## Install

```toml
[dependencies]
vivvy-memory = "0.1.0-alpha"
```

## Quick start

```rust
use vivvy_memory::{MemoryConfig, MemoryStore, MemoryScope, RememberRequest, RecallRequest, MemoryKind};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = MemoryConfig::builder("./agent_data")
        .dimensions(1536)
        .embedding_model("text-embedding-3-small")
        .build()?;
    let store = MemoryStore::open(config)?;

    let scope = MemoryScope::new("acme", "support")?;
    let id = store.remember(RememberRequest {
        operation_id: Some("op-01".into()),
        scope: scope.clone(),
        content: "User prefers concise answers.".into(),
        embedding: vec![0.01; 1536],
        kind: MemoryKind::Preference,
        importance: 0.9,
        expires_at_ms: None,
        metadata: HashMap::new(),
        source: HashMap::new(),
    })?;
    println!("stored {id}");

    let response = store.recall(RecallRequest {
        scope,
        query_embedding: vec![0.01; 1536],
        query_text: Some("concise answers".into()),
        limit: 5,
        filters: Default::default(),
        include_explanations: true,
        mmr_lambda: Some(0.5),
    })?;
    for item in response.items {
        println!("id={} score={:.4}", item.memory.id, item.score);
    }

    Ok(())
}
```

## Documentation

- [Rust API guide](https://docs.rs/vivvy-memory) (docs.rs, generated from source)
- [Full integration guide](https://github.com/one2seven-oss/vivvy/blob/main/Doc.md) — SQLite schema, crash-recovery protocol, telemetry/security invariants, and the contributor engineering guide
- [Workspace README](https://github.com/one2seven-oss/vivvy) — the four language bindings (Rust, Python, Go, and the C FFI layer) and benchmark numbers

## License

Licensed under the [Business Source License 1.1](https://github.com/one2seven-oss/vivvy/blob/main/LICENSE) (BUSL-1.1) — free for development, evaluation, and non-commercial or single-node production use. See the [licensing section](https://github.com/one2seven-oss/vivvy#licensing--commercial-terms) of the main repository for the full Additional Use Grant.
