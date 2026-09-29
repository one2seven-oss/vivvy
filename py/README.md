# vivvy-vdb

**Vector search that lives on one machine.** [Vivvy](https://github.com/one2seven-oss/vivvy)'s Python bindings: a local, durable long-term memory (LTM) runtime and single-machine vector engine for AI agents, built on a Rust core via [PyO3](https://pyo3.rs). No cloud database, no network hop, no daemon process to run alongside your app.

```bash
pip install vivvy-vdb
```

Two entry points, both in the `vivvy` module:

- **`vivvy.MemoryStore`** — the high-level agent memory API: durable, namespace-isolated storage with hybrid (vector + full-text) recall, explainable scoring, MMR diversity, and LLM-context formatting.
- **`vivvy.Index`** — a low-level vector index, for when you just need fast ANN search without memory-store semantics.

## Quick start: agent memory

```python
import vivvy

store = vivvy.MemoryStore.open(
    path="./agent_memory_data",
    dimensions=1536,
    embedding_model="text-embedding-3-small",
)

memory_id = store.remember(
    tenant_id="acme_corp",
    namespace="support_chat",
    content="User prefers Python over JavaScript for backend examples.",
    embedding=[0.012, -0.045, 0.089] + [0.0] * 1533,
    kind="preference",   # preference | fact | instruction | context | episodic
    importance=0.9,
)

results = store.recall(
    tenant_id="acme_corp",
    namespace="support_chat",
    query_embedding=[0.012, -0.045, 0.089] + [0.0] * 1533,
    query_text="Python backend examples",   # fused with vector search via RRF
    limit=5,
)
for mem_id, content, score in results:
    print(f"[{score:.4f}] {content}")

# Or render straight into an LLM prompt, inside a fixed token budget:
context = store.format_context(
    tenant_id="acme_corp",
    namespace="support_chat",
    query_embedding=[0.012, -0.045, 0.089] + [0.0] * 1533,
    max_tokens=1500,
)
```

## Quick start: low-level vector index

```python
import vivvy

index = vivvy.Index(dims=768, metric="cosine")  # cosine | l2 | dot
vector_id = index.insert(vector=[0.1] * 768, metadata={"category": "electronics"})
hits = index.search(query=[0.1] * 768, k=5, filter={"category": "electronics"})
```

## Why

- **Durable by construction** — SQLite WAL is the canonical store; the vector index is a rebuildable accelerator that's reconstructed from SQLite if it's ever lost, never the source of truth.
- **Hybrid recall** — dense HNSW vector search fused with SQLite FTS5 lexical search (Reciprocal Rank Fusion), explainable 4-factor scoring, and optional MMR diversity reranking.
- **Multi-tenant by default** — every call is scoped by `tenant_id`/`namespace` (and optionally `agent_id`/`user_id`), enforced at the SQL layer.
- **GIL-released compute** — every insert/search/recall call runs the Rust work with the GIL released, so it doesn't block other Python threads.
- **Zero-copy NumPy** — vector arguments accept a plain `list[float]` or a contiguous 1D `float32` NumPy array.

## Documentation

The full guide — every `MemoryStore`/`Index` method, memory kinds, production agent patterns (RAG retrieval, multi-tenant chat, episodic buffers, batch ingestion, background maintenance), on-disk formats, and the Rust/Go bindings this same engine also powers — lives in [`Doc.md`](https://github.com/one2seven-oss/vivvy/blob/main/Doc.md) in the main repository.

## License

Licensed under the [Business Source License 1.1](https://github.com/one2seven-oss/vivvy/blob/main/LICENSE) (BUSL-1.1) — free for development, evaluation, and non-commercial or single-node production use. See the [licensing section](https://github.com/one2seven-oss/vivvy#licensing--commercial-terms) of the main repository for the full Additional Use Grant.
