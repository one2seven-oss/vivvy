<div align="center">
  <img src="assets/vivy.png" alt="Vivy" width="200" />
  <br><br>

  [![Crates.io](https://img.shields.io/crates/v/vivy-core?label=vivy-core)](https://crates.io/crates/vivy-core)
  [![PyPI](https://img.shields.io/badge/pypi-vivy--vdb-blue)](https://pypi.org/project/vivy-vdb/)
  [![License](https://img.shields.io/badge/license-BSL--1.1-green)](LICENSE)
  [![Rust](https://img.shields.io/badge/rust-1.81%2B-orange)](https://www.rust-lang.org)
  [![HNSW](https://img.shields.io/badge/index-HNSW-8A2BE2)](#)
  [![WAL](https://img.shields.io/badge/crash--safe-WAL-blue)](#)
  [![FTS5](https://img.shields.io/badge/hybrid-FTS5%20%2B%20Vector%20RRF-blue)](#)
  [![PyO3](https://img.shields.io/badge/bindings-PyO3-yellow)](#)
  [![Release Candidate](https://img.shields.io/badge/status-v0.1.0--RC1%20Ready-brightgreen)](#)
</div>

A local, durable long-term memory (LTM) runtime and single-machine vector engine for AI agents. No cloud databases, no network latency, no external daemons. Rust core with PyO3 Python bindings.

---

## Workspace Packages

| Crate / Package | Path | Responsibility |
| :--- | :--- | :--- |
| **`vivy-memory`** | [`ltm/`](ltm) | Durable LTM runtime: multi-tenant isolation, SQLite WAL canonical store, operation journal, hybrid recall (FTS5 + HNSW vector RRF), explainable scoring & MMR diversity. |
| **`vivy-core`** | [`vec/`](vec) | In-process vector engine: HNSW graph search, Roaring bitmap filters, 64-bit ID safety, atomic segment manifests & WAL. |
| **`vivy-py`** | [`py/`](py) | PyO3 Python bindings exposing `vivy.MemoryStore` and low-level `vivy.Index` with GIL release. |
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
| **Vector Engine (64-dim)** | `6,012 QPS` | Pure HNSW vector search throughput (`0.16 ms` mean latency) |
| **Vector Engine (512-dim)** | `1,433 QPS` | Pure HNSW vector search throughput (`0.69 ms` mean latency) |
| **Tombstone Vacuum Speed** | `74 ms` | Resumable physical tombstone scrubbing (100 rows batch) |

---

## Key Capabilities

* **Durable Agent Memory (`vivy-memory`)**: SQLite WAL serves as canonical truth for memory records, revisions, and operation journal. Vector index acts as a derived, auto-rebuildable accelerator.
* **Time-Window Temporal Filtering**: Filter candidate memories by temporal lower and upper bounds (`created_after_ms` & `created_before_ms`) at SQLite query entry and candidate evaluation.
* **Strict Multi-Tenant Isolation**: Enforces tenant, namespace, agent, and user boundaries at API entry and SQL query level. Zero cross-tenant data leakage.
* **Hybrid Candidate Recall**: Combines SQLite FTS5 lexical keyword matching with dense HNSW vector search using Reciprocal Rank Fusion (RRF).
* **Token-Budgeted Context Formatter (`format_context`)**: Formats recalled agent memories into custom templated prompt blocks bounded by exact LLM token budgets.
* **Live Zero-Downtime Store Backups (`backup`)**: Creates crash-consistent online backup snapshots of SQLite (via `VACUUM INTO`) and vector engine segments without interrupting store operation.
* **Zero-Copy NumPy & PyTorch Ingestion**: Direct PyO3 C-contiguous buffer protocol ingestion for `np.ndarray` float32 arrays with GIL release during search & inserts.
* **Transparent Reranking & MMR**: 4-component weighted scoring (Similarity, Importance, Recency, Reinforcement) plus optional Maximal Marginal Relevance (MMR) deduplication.
* **Security & Operations Primitives**: Encrypted storage interfaces (`KeyProvider`), telemetry redaction (`TelemetryRecord`), non-blocking health checks (`StoreHealth`), and resumable tombstone vacuuming (`vacuum_tombstones`).
* **High Performance Vector Search (`vivy-core`)**: HNSW vector graph with non-blocking inserts and Roaring bitmap metadata filtering.

---

## Architecture

```text
+-----------------------------------------------------------------+
| Caller / Python Agent / LLM Application                         |
+-----------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
| vivy-memory (LTM Runtime Layer)                                 |
| - MemoryScope (tenant_id, namespace, agent_id, user_id)         |
| - Operation Journal (idempotency, 2-phase state machine)        |
| - Temporal Filtering (created_after_ms, created_before_ms)      |
| - Hybrid Reciprocal Rank Fusion (SQLite FTS5 + HNSW Vector RRF) |
| - Explainable Reranker, Context Formatter & MMR Diversity       |
+-----------------------------------------------------------------+
           |                                           |
           v                                           v
+-----------------------------------+   +-------------------------+
| SQLite (Canonical Store / WAL)    |   | vivy-core (Vector ANN)  |
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
import vivy

# Open durable long-term memory store
store = vivy.MemoryStore.open(path="./agent_data", dimensions=3, embedding_model="test-model")

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

---

## Build & Verification

```sh
# Run Rust workspace test suite (60 unit, integration & security tests)
cargo test --workspace

# Run zero-warning Clippy check
cargo clippy --workspace --all-targets -- -D warnings

# Build & install Python extension module
maturin develop --manifest-path py/Cargo.toml

# Run Python PyO3 integration test suite (44 tests)
pytest py/

# Run performance benchmark suite
cargo run --release --package bench
```

---

## Licensing & Commercial Terms

**Vivy** components (`vivy-core`, `vivy-memory`, `vivy-py`, `bench`) are published under **The Business Source License 1.1 (BSL-1.1)** (see [`LICENSE`](LICENSE)).

### Permitted Uses (BSL 1.1 Additional Use Grant)
- **Non-Production & Evaluation**: Free use for development, testing, research, and evaluation.
- **Production Workloads**: Free use in production for non-commercial applications, single-node deployments, and internal AI agent workloads, provided Vivy is not offered as a managed SaaS or cloud API vector service to third parties.

### Commercial Pro / Enterprise Licensing
For managed cloud service providers or enterprise deployments requiring custom SLAs:
- Cloud cluster synchronization & distributed multi-region replication.
- Hardware Security Module (HSM) & AWS KMS / GCP KMS `KeyProvider` integration.
- Role-Based Access Control (RBAC) policy enforcement engine.

---
