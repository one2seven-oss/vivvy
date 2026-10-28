<div align="center">
  <img src="assets/vivvy.svg" alt="Vivvy" width="200" />
  <br><br>

  [![Crates.io](https://img.shields.io/crates/v/vivvy-core?label=vivvy-core)](https://crates.io/crates/vivvy-core)
  [![PyPI](https://img.shields.io/pypi/v/vivvy)](https://pypi.org/project/vivvy/)
  [![License](https://img.shields.io/badge/license-BSL--1.1-green)](LICENSE)
  [![Rust](https://img.shields.io/badge/rust-1.81%2B-orange)](https://www.rust-lang.org)
  [![HNSW](https://img.shields.io/badge/index-HNSW-8A2BE2)](#)
  [![WAL](https://img.shields.io/badge/crash--safe-WAL-blue)](#)
  [![FTS5](https://img.shields.io/badge/hybrid-FTS5%20%2B%20Vector%20RRF-blue)](#)
  [![PyO3](https://img.shields.io/badge/bindings-PyO3-yellow)](#)
  [![Go](https://img.shields.io/badge/bindings-cgo-00ADD8)](go)
  [![Release](https://img.shields.io/badge/status-v0.1.0--beta%20Ready-brightgreen)](#)
</div>

A local, durable long-term memory (LTM) runtime and single-machine vector engine for AI agents. No cloud databases, no network latency, no external daemons. Rust core with PyO3 Python bindings and Go bindings via a pre-compiled static C-FFI library.

---

## Workspace Packages

| Crate / Package | Path | Responsibility |
| :--- | :--- | :--- |
| **`vivvy`** | [`unified/`](unified) | Batteries-included Rust facade: re-exports vector search, durable memory, and embedder interfaces behind feature flags. |
| **`vivvy-memory`** | [`ltm/`](ltm) | Durable LTM runtime: multi-tenant isolation, SQLite WAL canonical store, operation journal, hybrid recall (FTS5 + HNSW vector RRF), configurable 4-factor scoring & MMR diversity. |
| **`vivvy-core`** | [`vec/`](vec) | In-process vector engine: HNSW graph search, Roaring bitmap filters, 64-bit ID safety, atomic segment manifests & WAL. |
| **`vivvy-embedder`** | [`embedders/`](embedders) | Provider-neutral Rust embedding interface; language SDKs supply native provider adapters. |
| **`vivvy-py`** | [`py/`](py) | PyO3 Python bindings exposing `vivvy.MemoryStore` and low-level `vivvy.Index` with GIL release. |
| **`vivvy-ffi`** | [`ffi/`](ffi) | Internal `extern "C"` layer (staticlib + cdylib) exposing memory and low-level vector APIs to language bindings. |
| **`vivvy-go`** | [`go/`](go) | Go bindings over `vivvy-ffi` via cgo and a pre-compiled static library — `go get`, no Rust toolchain or daemon required. |
| **`vivvy-ts`** | [`ts/`](ts) | TypeScript & Node.js bindings via Node-API — typed `Float32Array` ingestion, memory/vector APIs, and typed embedder adapters. |
| **`bench`** | [`sim/`](sim) | Synthetic data benchmark suite for QPS, latency, and ground-truth recall verification. |

---

## Benchmark Performance Matrix

Measured on single-machine benchmark suite (`cargo run --release --package bench`):

| Metric / Scenario | Measured Value | Description |
| :--- | :--- | :--- |
| **Cold Start Latency** | `~51 ms` | Time to initialize fresh database, SQLite WAL, & vector engine |
| **Warm Start Latency** | `~51 ms` | Time to recover startup & replay operation journal (500 records) |
| **Write Throughput** | `5,216 writes/sec` | Dual-write durability (SQLite WAL commit + HNSW graph update) |
| **Hybrid Recall Throughput** | `305.1 QPS` | Dense Vector KNN + SQLite FTS5 + RRF Fusion + MMR Rerank |
| **Hybrid Recall Latency** | `3.27 ms` | End-to-end mean search & reranking latency |
| **Reranking Overhead** | `< 15 µs` | 4-factor scoring & decay for 1,000 candidates (zero heap allocations) |
| **Vector Engine (64-dim)** | `6,012 QPS` | Pure HNSW vector search throughput (`0.16 ms` mean latency) |
| **Vector Engine (512-dim)** | `1,433 QPS` | Pure HNSW vector search throughput (`0.69 ms` mean latency) |
| **Tombstone Vacuum Speed** | `74 ms` | Resumable physical tombstone scrubbing (100 rows batch) |

---

## Key Capabilities

* **Durable Agent Memory (`vivvy-memory`)**: SQLite WAL serves as canonical truth for memory records, revisions, and operation journal. Vector index acts as a derived, auto-rebuildable accelerator.
* **Time-Window Temporal Filtering**: Filter candidate memories by temporal lower and upper bounds (`created_after_ms` & `created_before_ms`) at SQLite query entry and candidate evaluation.
* **Strict Multi-Tenant Isolation**: Enforces tenant, namespace, agent, and user boundaries at API entry and SQL query level. Zero cross-tenant data leakage.
* **Hybrid Candidate Recall**: Combines SQLite FTS5 lexical keyword matching with dense HNSW vector search using Reciprocal Rank Fusion (RRF).
* **Token-Budgeted Context Formatter (`format_context`)**: Formats recalled agent memories into custom templated prompt blocks bounded by exact LLM token budgets.
* **Live Zero-Downtime Store Backups (`backup`)**: Creates crash-consistent online backup snapshots of SQLite (via `VACUUM INTO`) without interrupting store operation; vector index is ephemeral and rebuilt on restore.
* **Zero-Copy NumPy & PyTorch Ingestion**: Direct PyO3 C-contiguous buffer protocol ingestion for `np.ndarray` float32 arrays with GIL release during search & inserts.
* **Configurable 4-Factor Reranking & MMR**: Dynamic scoring weights (Similarity, Importance, Recency, Reinforcement) and tunable exponential half-life decay with zero heap allocation and $O(1)$ register arithmetic, plus optional Maximal Marginal Relevance (MMR) deduplication.
* **Security & Operations Primitives**: Encrypted storage interfaces (`KeyProvider`), telemetry redaction (`TelemetryRecord`), non-blocking health checks (`StoreHealth`), and resumable tombstone vacuuming (`vacuum_tombstones`).
* **High Performance Vector Search (`vivvy-core`)**: HNSW vector graph with non-blocking inserts and Roaring bitmap metadata filtering.

---

## Architecture

```text
+-----------------------------------------------------------------+
| Caller / Python Agent / LLM Application                         |
+-----------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
| vivvy-memory (LTM Runtime Layer)                                |
| - MemoryScope (tenant_id, namespace, agent_id, user_id)         |
| - Operation Journal (idempotency, 2-phase state machine)        |
| - Temporal Filtering (created_after_ms, created_before_ms)      |
| - Hybrid Reciprocal Rank Fusion (SQLite FTS5 + HNSW Vector RRF) |
| - Explainable Reranker, Context Formatter & MMR Diversity       |
+-----------------------------------------------------------------+
           |                                           |
           v                                           v
+-----------------------------------+   +-------------------------+
| SQLite (Canonical Store / WAL)    |   | vivvy-core (Vector ANN) |
| - memories (Content, Provenance)  |   | - Rebuildable HNSW      |
| - memories_fts (FTS5 Lexical)     |   | - Fast Cosine Retrieval |
| - operations (Journal state)      |   | - Derived Accelerators  |
+-----------------------------------+   +-------------------------+
```

---

## Getting Started

Refer to [**`Doc.md`**](Doc.md) for the complete, production-grade integration guide, full API reference, and runnable agent workflow patterns.

### Quick Example

```python
import vivvy

# Open durable long-term memory store
store = vivvy.MemoryStore.open(path="./agent_data", dimensions=3, embedding_model="test-model")

# Store observation into durable memory
mem_id = store.remember(
    tenant_id="acme",
    namespace="support",
    content="User prefers concise technical responses.",
    embedding=[0.1, 0.2, 0.3],
    kind="preference"
)

# Recall relevant memories with hybrid text + vector search
results = store.recall(
    tenant_id="acme",
    namespace="support",
    query_embedding=[0.1, 0.2, 0.3],
    query_text="concise technical",
    limit=5
)

for memory_id, content, score in results:
    print(f"[{score:.2f}] {content}")
```

#### TypeScript & Node.js Example

```typescript
import { MemoryStore, OpenAIEmbedder } from 'vivvy';

const embedder = new OpenAIEmbedder({ model: 'text-embedding-3-small' });
const store = await MemoryStore.open({ path: './agent_data', embedder });

// Store observation into durable agent memory
const memoryId = await store.rememberText({
  tenantId: 'acme',
  namespace: 'support',
  content: 'User prefers concise technical responses.',
  kind: 'preference',
  importance: 0.9,
});

// Recall relevant memories with hybrid text + vector search
const recall = await store.recallText({
  tenantId: 'acme',
  namespace: 'support',
  queryText: 'concise technical',
  topK: 5,
});

for (const item of recall.items) {
  console.log(`[${item.score.toFixed(2)}] ${item.content}`);
}
```

---

## Build & Verification

```sh
# Run Rust workspace test suite (60 unit, integration & security tests)
cargo test --workspace

# Run the full Rust + Go + TypeScript contract matrix on a supported Unix host
make verify

# Run zero-warning Clippy check
cargo clippy --workspace --all-targets -- -D warnings

# Build & install Python extension module
maturin develop --manifest-path py/Cargo.toml

# Run Python PyO3 integration test suite (44 tests)
pytest py/

# Build and run TypeScript test suite (Node.js & Bun)
make ts-test

# Run performance benchmark suite
cargo run --release --package bench
```

## `0.1.0-beta.1` Release Surface

The beta release publishes four Rust libraries with one aligned version:

- `vivvy` — unified, batteries-included SDK.
- `vivvy-core` — vector search only.
- `vivvy-memory` — durable agent memory, built on `vivvy-core`.
- `vivvy-embedder` — provider-neutral embedding interface.

Python and TypeScript expose the unified experience as `vivvy`; Go releases
through the repository tag and its prebuilt native libraries. See
[RELEASING.md](RELEASING.md) for the ordered release procedure.

---

## Licensing & Commercial Terms

**Vivvy** components (`vivvy`, `vivvy-core`, `vivvy-memory`, `vivvy-embedder`, language SDKs, and internal build artifacts) are published under **The Business Source License 1.1 (BSL-1.1)** (see [`LICENSE`](LICENSE)).

### Permitted Uses (BSL 1.1 Additional Use Grant)
- **Non-Production & Evaluation**: Free use for development, testing, research, and evaluation.
- **Production Workloads**: Free use in production for non-commercial applications, single-node deployments, and internal AI agent workloads, provided Vivvy is not offered as a managed SaaS or cloud API vector service to third parties.

### Commercial Pro / Enterprise Licensing
For managed cloud service providers or enterprise deployments requiring custom SLAs:
- Cloud cluster synchronization & distributed multi-region replication.
- Hardware Security Module (HSM) & AWS KMS / GCP KMS `KeyProvider` integration.
- Role-Based Access Control (RBAC) policy enforcement engine.

---
