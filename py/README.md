# vivvy

**Vector search that lives on one machine.** [Vivvy](https://github.com/one2seven-oss/vivvy)'s Python bindings: a local, durable long-term memory (LTM) runtime and single-machine vector engine for AI agents, built on a Rust core via [PyO3](https://pyo3.rs). No cloud database, no network hop, no daemon process to run alongside your app.

```bash
pip install vivvy
```

Two entry points, both in the `vivvy` module:

- **`vivvy.MemoryStore`** — the high-level agent memory API: durable, namespace-isolated storage with hybrid (vector + full-text) recall, explainable scoring, MMR diversity, and LLM-context formatting.
- **`vivvy.Index`** — a low-level vector index, for when you just need fast ANN search without memory-store semantics.

## Quick start: agent memory

```python
import vivvy
from vivvy.embeddings import OpenAIEmbedder  # or SentenceTransformersEmbedder, OllamaEmbedder

# 1. Connect embedding adapter (or use local sentence-transformers / ollama)
embedder = OpenAIEmbedder(model="text-embedding-3-small")

# Dimensions and model name are automatically inferred from embedder:
store = vivvy.MemoryStore.open(
    path="./agent_memory_data",
    embedder=embedder,
)

# 2. Store text directly — embeddings generated and validated automatically
memory_id = store.remember_text(
    tenant_id="acme_corp",
    namespace="support_chat",
    content="User prefers Python over JavaScript for backend examples.",
    kind="preference",   # preference | fact | instruction | context | episodic
    importance=0.9,
)

# 3. Hybrid recall directly from query text (vector ANN fused with SQLite FTS5)
results = store.recall_text(
    tenant_id="acme_corp",
    namespace="support_chat",
    query_text="Python backend examples",
    limit=5,
)
for mem_id, content, score in results:
    print(f"[{score:.4f}] {content}")

# 3b. Tuning recall scoring weights for specialized agent personas
results = store.recall_text(
    tenant_id="acme_corp",
    namespace="agent_chat",
    query_text="Deployment architecture",
    scoring={
        "weight_similarity": 0.85,
        "weight_importance": 0.15,
        "weight_recency": 0.0,
    }
)

# 4. Or render straight into an LLM prompt inside a fixed token budget:
context = store.format_context_text(
    tenant_id="acme_corp",
    namespace="support_chat",
    query_text="Python backend examples",
    max_tokens=1500,
)
```

You can also pass pre-computed float vectors via `store.remember(...)` and `store.recall(...)` if managing embeddings manually.

### Workload Tuning Recipes

Pass `scoring={...}` or `vivvy.ScoringConfig(...)` to tailor recall reranking to your agent's workload:

| Workload | Recommended Weights | Focus |
| :--- | :--- | :--- |
| **Technical Docs / Code** | `{"weight_similarity": 0.90, "weight_importance": 0.10, "weight_recency": 0.0}` | High semantic precision; avoids penalizing older docs. |
| **Conversational Chat** | `{"weight_similarity": 0.35, "weight_importance": 0.15, "weight_recency": 0.50, "recency_half_life_days": 1.0}` | Recent dialogue turns dominate; rapid 24h decay. |
| **Core Persona / Rules** | `{"weight_similarity": 0.40, "weight_importance": 0.60, "weight_recency": 0.0}` | Critical rules and high-importance memories outrank chatter. |
| **Skills / Workflows** | `{"weight_similarity": 0.50, "weight_importance": 0.15, "weight_recency": 0.05, "weight_reinforcement": 0.30}` | Frequently executed routines and reinforced memories win. |

## Quick start: low-level vector index

```python
import vivvy

index = vivvy.Index(dims=768, metric="cosine")  # cosine | l2 | dot
vector_id = index.insert(vector=[0.1] * 768, metadata={"category": "electronics"})
hits = index.search(query=[0.1] * 768, k=5, filter={"category": "electronics"})
```

## Why

- **Durable by construction** — SQLite WAL is the canonical store; the vector index is a rebuildable accelerator that's reconstructed from SQLite if it's ever lost, never the source of truth.
- **Hybrid recall & configurable scoring** — dense HNSW vector search fused with SQLite FTS5 lexical search (Reciprocal Rank Fusion), configurable 4-factor scoring weights & half-life decay with zero heap allocation ($< 15\,\mu\text{s}$ per 1,000 candidates), and optional MMR diversity reranking.
- **Multi-tenant by default** — every call is scoped by `tenant_id`/`namespace` (and optionally `agent_id`/`user_id`), enforced at the SQL layer.
- **GIL-released compute** — every insert/search/recall call runs the Rust work with the GIL released, so it doesn't block other Python threads.
- **Zero-copy NumPy** — vector arguments accept a plain `list[float]` or a contiguous 1D `float32` NumPy array.

## Documentation

The full guide — every `MemoryStore`/`Index` method, memory kinds, production agent patterns (RAG retrieval, multi-tenant chat, episodic buffers, batch ingestion, background maintenance), on-disk formats, and the Rust/Go/TypeScript SDKs this same engine also powers — lives in [`Doc.md`](https://github.com/one2seven-oss/vivvy/blob/main/Doc.md) in the main repository.

## License

Licensed under the [Business Source License 1.1](https://github.com/one2seven-oss/vivvy/blob/main/LICENSE) (BUSL-1.1) — free for development, evaluation, and non-commercial or single-node production use. See the [licensing section](https://github.com/one2seven-oss/vivvy#licensing--commercial-terms) of the main repository for the full Additional Use Grant.
