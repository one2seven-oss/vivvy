# vivvy-memory

**Local, durable, namespace-isolated long-term memory for AI agents.** SQLite WAL is the canonical store of record; a [`vivvy-core`](https://crates.io/crates/vivvy-core) HNSW index rides alongside as a rebuildable accelerator — if it's ever lost or corrupted, it's regenerated from SQLite on boot, never the other way around.

`vivvy-memory` is the durable runtime layer of [Vivvy](https://github.com/one2seven-oss/vivvy). If you just need raw vector search with your own storage, [`vivvy-core`](https://crates.io/crates/vivvy-core) alone may be enough; reach for `vivvy-memory` when you want multi-tenant isolation, crash recovery, hybrid recall, and LLM-context formatting out of the box.

## Features

- **Multi-Tenant Isolation** — every record is scoped by tenant, namespace, and optionally agent/user; scope boundaries are enforced at the SQL layer, not just in application code.
- **Canonical SQLite WAL** — crash-safe durable storage with a 2-phase operation journal, so a crash mid-write never leaves the vector index and the record store disagreeing.
- **Hybrid Recall** — dense HNSW vector search fused with SQLite FTS5 lexical search via Reciprocal Rank Fusion (RRF).
- **Explainable Scoring & MMR** — a 4-factor weighted score (similarity, importance, recency, reinforcement) with an optional Maximal Marginal Relevance pass for result diversity.
- **Token-Budgeted Context Formatting** — render recall results straight into an LLM prompt with a custom template and a hard token budget.
- **Live Online Backups** — crash-consistent snapshots (SQLite + vector segments) without stopping operations.

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
