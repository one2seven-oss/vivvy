# vivvy-core

**Single-machine, memory-honest vector search.** A sharded HNSW index with Roaring-bitmap metadata filters, write-ahead logging, and crash-safe sealed segments — no server, no network hop, no external service.

`vivvy-core` is the low-level vector engine underneath [Vivvy](https://github.com/one2seven-oss/vivvy), a durable long-term memory runtime for AI agents. Reach for `vivvy-core` directly when you want fast in-process ANN search with your own persistence and multi-tenancy story; reach for [`vivvy-memory`](https://crates.io/crates/vivvy-memory) instead if you want the full durable, namespace-isolated memory store built on top of it.

## Features

- **HNSW Vector Index** — sharded graphs for concurrent, non-blocking approximate nearest neighbor search across L2, Cosine, and Dot metrics.
- **Roaring Bitmap Metadata Filters** — `AND`/`IN`/equality filter expressions evaluated alongside vector search, not as a slow post-filter pass.
- **Durable Storage** — a write-ahead log for uncommitted inserts plus atomic, memory-mapped sealed segments (`.vivvy` files) and a crash-safe manifest, so an in-flight write never corrupts the index.
- **64-bit ID Preservation** — safe explicit ID mapping; you choose the ID space, the index doesn't remap it.
- **Concurrency** — lock-free reads against sealed segments, fine-grained locking on the mutable delta shard for concurrent inserts.

## Install

```toml
[dependencies]
vivvy-core = "0.1.0-alpha"
```

## Quick start

```rust
use vivvy_core::concurrent::VivvyIndex;
use vivvy_core::distance::Metric;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // WAL + sealed-segment directories are optional — omit both for a pure in-memory index.
    let index = VivvyIndex::new(768, Metric::Cosine, Some("./wal.log"), Some("./segments"))?;

    index.insert_with_id(1001, vec![0.1; 768])?;

    let hits = index.search(&vec![0.1; 768], 5)?;
    for (id, distance) in hits {
        println!("id={id} distance={distance:.4}");
    }

    Ok(())
}
```

## Documentation

- [Rust API guide](https://docs.rs/vivvy-core) (docs.rs, generated from source)
- [Full integration guide](https://github.com/one2seven-oss/vivvy/blob/main/Doc.md) — architecture, on-disk formats (`.vivvy` segments, `.wal`, `manifest.idx`), and the contributor engineering guide
- [Workspace README](https://github.com/one2seven-oss/vivvy) — the four language bindings (Rust, Python, Go, and the C FFI layer) and benchmark numbers

## License

Licensed under the [Business Source License 1.1](https://github.com/one2seven-oss/vivvy/blob/main/LICENSE) (BUSL-1.1) — free for development, evaluation, and non-commercial or single-node production use. See the [licensing section](https://github.com/one2seven-oss/vivvy#licensing--commercial-terms) of the main repository for the full Additional Use Grant.
