# Vivvy Post-v0.1.0 Updates & Changelog

This document provides a comprehensive, point-by-point record of all updates, bug fixes, security enhancements, and new features implemented in **Vivvy** following the last stable release (`v0.1.0` / `go/v0.1.0`).

---

## 1. Bug Fixes & Reliability Improvements

### 1.1. Cross-Tenant Forget Vulnerability Fix
* **Reference:** PR #21 (Commits `2681eba`, `dc5f9a0`), Fixes Issue #1
* **Components Affected:** `ltm/src/repository.rs`, `ltm/tests/security_tests.rs`
* **Issue:**
  * In `Repository::delete_memory` and batch delete operations, Vivvy issued SQL updates scoped by tenant and namespace (`UPDATE memories SET status = 'deleted' WHERE id = ? AND tenant_id = ? AND namespace = ?`).
  * However, the query execution did not inspect the count of affected rows. If a tenant attempted to delete an ID owned by another tenant, SQLite updated zero rows, but the function returned successfully. Execution then proceeded to tombstone the vector index entry, causing an unauthorized vector eviction across tenant boundaries.
* **Resolution & Details:**
  * **Affected Rows Verification:** Explicitly asserted `rows_affected > 0`. If 0 rows are updated, the repository now returns `MemoryError::NotFound`, immediately halting before any vector index modification occurs.
  * **Tombstone Status Guard:** Added `AND status != 'deleted'` to prevent repeated deletion invocations on already-tombstoned memories.
  * **Security Test Suite:** Added `test_cross_tenant_forget_isolation_does_not_tombstone_vector_index` and `test_forget_nonexistent_returns_not_found` to prevent regression.

---

### 1.2. Tenant-Scoped Idempotent Operation Journal
* **Reference:** PR #22 (Commits `6ec53ed`, `633101f`, `77bc759`), Fixes Issue #2
* **Components Affected:** `ltm/src/journal.rs`, `ltm/src/repository.rs`, `ltm/tests/operations_tests.rs`
* **Issue:**
  * The `operations` idempotency table keyed records solely by `operation_id` globally. This allowed operation IDs from different tenants to collide or leak metadata.
  * Furthermore, submitting the same `operation_id` with conflicting or mutated payloads was not verified, returning stale memory IDs silently.
* **Resolution & Details:**
  * **Composite Primary Key:** Migrated the SQLite schema to `PRIMARY KEY (tenant_id, namespace, operation_id)`.
  * **Scoped Queries:** Scoped all repository lookups and status transitions (`get_operation`, `mark_operation_applied`, `mark_operation_applied_batch`) strictly by `MemoryScope`.
  * **Payload Validation on Retry:** When an existing `operation_id` is replayed, Vivvy now deserializes the stored payload and validates that both `content` and `embedding` match. If any field differs, it returns `MemoryError::InvalidInput("Operation ID already used with a different payload")`.
  * **Delete Idempotency:** Added checks so re-executing a delete operation with the same `operation_id` cleanly returns `Ok(())`.

---

### 1.3. Ephemeral Vector Index & Persistence Reconciliation
* **Reference:** PR #23 (Commits `b4ede93`, `5681535`, `32044f7`, `8ffb6f1`), Fixes Issue #3
* **Components Affected:** `ltm/src/index_adapter.rs`, `ltm/src/lib.rs`, `ltm/tests/store_tests.rs`, documentation
* **Issue:**
  * The LTM engine previously attempted dual persistence: SQLite recorded memories, while the vector index independently flushed `.vivvy` segment files and WAL logs (`index.wal`, `index.manifest`).
  * Crashes or rebuilds led to out-of-sync states between the database and the vector segment files.
* **Resolution & Details:**
  * **SQLite as Canonical Ground Truth:** Configured the LTM vector index as strictly ephemeral in memory. On `MemoryStore::open()`, the vector index is reconstructed directly from the SQLite store.
  * **Stale Ephemeral File Cleanup:** Added automatic detection and cleanup of obsolete index files (`index.wal`, `index.manifest`, and any lingering `*.vivvy` files) during store initialization.
  * **Consistent Live Backups:** Simplified zero-downtime backups (`MemoryStore::backup`) to perform online SQLite snapshots via `VACUUM INTO`. This eliminated backup race conditions where vector segments and SQLite could diverge.
  * **Verification:** Added `test_stale_index_files_cleaned_up_on_open`.

---

### 1.4. Scope Starvation Resolution in Vector Recall
* **Reference:** PR #27 (Commits `49dedc2`, `fa6ed23`, `460f7ca`), Fixes Issue #5
* **Components Affected:** `ltm/src/index_adapter.rs`, `ltm/src/lib.rs`, `ltm/src/repository.rs`, `ltm/tests/store_tests.rs`
* **Issue:**
  * When multiple tenants or namespaces populated the shared vector space, a fixed top-$k$ nearest-neighbor query would often return vectors exclusively belonging to other tenants or tombstoned records.
  * Post-filtering in the memory layer then dropped all foreign results, leaving the caller with 0 items even when matching records existed further down the similarity ranking (Scope Starvation).
* **Resolution & Details:**
  * **Adaptive Candidate Expansion Loop:** Replaced static top-$k$ search with a progressive fetch loop in `VivvyVectorIndex::search`. If returned candidates hit other scopes or tombstones, search depth $k_{\text{fetch}}$ dynamically doubles until sufficient matching candidates are gathered or the index is exhausted.
  * **Candidate Filter Matching:** Implemented `record_matches_recall` to evaluate candidate match against scope, active status, expiration (`expires_at_ms`), metadata criteria, and kinds.
  * **Record Cache Reuse:** Introduced an in-memory `record_cache` during fusion and reranking passes in `ltm/src/lib.rs` to avoid redundant database reads for candidates evaluated multiple times.
  * **Verification:** Added `test_recall_widens_candidates_past_other_scopes`, `test_recall_widens_candidates_past_filter_misses`, and `test_recall_widens_vector_search_past_tombstones`.

---

### 1.5. HNSW Upsert Semantics, Deletions & Node Staling Fix
* **Reference:** Branch `fix/hnsw-upsert-stale-nodes` (Commit `d4f204a`)
* **Components Affected:** `vec/src/hnsw.rs`, `vec/src/flat.rs`, `vec/src/concurrent.rs`, `vec/src/storage/wal.rs`, `ltm/src/index_adapter.rs`
* **Issue:**
  * Inserting a vector with an existing ID previously appended a new entry without evicting the previous vector representation in the HNSW graph. Queries near the old position still matched the stale node, corrupting distance metrics.
  * Vector index had no tombstoning or deletion mechanism in WAL or segment compaction.
* **Resolution & Details:**
  * **True HNSW Upsert Semantics:** In `HnswIndex::insert`, if an ID already exists, the old node is tombstoned and unlinked from external mappings before the new vector is indexed.
  * **Tombstone Set & Search Filtering:** Added active `HashSet<u64>` tracking deleted nodes in `HnswIndex`. Graph traversal ignores tombstoned nodes during search.
  * **Segment Compaction & Drain:** Updated `drain()` and `snapshot()` to exclude tombstoned entries so flushes write only live vectors.
  * **Roaring Bitmap Tracking in Concurrent Index:** Added `RoaringTreemap` in `VivvyIndex` for memory-efficient deletion tracking across concurrent shards.
  * **WAL Deletion Opcode:** Extended the Write-Ahead Log format with opcode `0x02` (`WalEntry::Delete { id: u64 }`) ensuring deletes survive crash recovery.
  * **Adapter Propagation:** Updated `VivvyVectorIndex::remove` and `remove_batch` to invoke `VivvyIndex::delete`.
  * **Verification:** Added `test_hnsw_upsert`, `test_hnsw_delete`, `test_vivvy_vector_index_upsert_moves_candidate`, and `test_wal_delete_roundtrip`.

---

### 1.6. Numeric Safety, Boundary Checks & Trait Hardening
* **Reference:** Commit `b567930`
* **Components Affected:** `vec/src/storage/segments.rs`, `vec/src/distance.rs`, `vec/src/hnsw.rs`, `vec/src/flat.rs`, `ltm/src/config.rs`, `ffi/src/error.rs`, `ffi/src/types.rs`
* **Issue:**
  * Unchecked memory slice conversions (`try_into().unwrap()`) in memory-mapped segment reading could panic on corrupted files.
  * Floating-point sorting using `partial_cmp().unwrap()` could trigger runtime panics on `NaN` or unnormalized vectors.
* **Resolution & Details:**
  * **Safe Segment Decoding:** Wrapped slice parsing in explicit buffer length validation, returning `Result<_, SegmentError::Truncated>` on out-of-bounds reads.
  * **Safe Float Ordering:** Replaced all instances of `partial_cmp().unwrap()` with `total_cmp()`.
  * **API Annotations:** Decorated builder methods and getters with `#[must_use]` attributes across FFI, config, and repository layers.

---

## 2. New Features & Architectural Enhancements

### 2.1. Per-Tenant Field-Level Encryption at Rest
* **Reference:** PR #24 (Commits `268878d`, `de7cc79`)
* **Components Affected:** `ltm/src/cipher.rs`, `ltm/src/config.rs`, `ltm/src/repository.rs`, `ltm/src/error.rs`, `ffi/`, `go/`
* **Capabilities:**
  * **AEAD Cryptographic Engine:** Implemented `FieldCipher` using **XChaCha20-Poly1305** with 24-byte random nonces per field.
  * **Tamper-Proof AAD Binding:** Ciphertext is bound via Additional Authenticated Data:
    $$\text{AAD} = \text{tenant\_id} \parallel \text{record\_id} \parallel \text{field\_name}$$
    This prevents cross-tenant, cross-row, or cross-column ciphertext swapping.
  * **Encrypted Sensitive Fields:** Encrypts `content`, `embedding`, `metadata_json`, `source_json`, and operation journal payloads. Query fields (`id`, `tenant_id`, `namespace`, `created_at_ms`, `kind`, `status`, `importance`) remain unencrypted to support fast SQL indexing.
  * **Pluggable `KeyProvider`:** Added `KeyProvider` trait for external KMS/vault integration, alongside `NoOpDevKeyProvider` for development.
  * **Lexical Privacy Modes (`LexicalMode`):**
    * `Disabled`: Full-text search disabled on encrypted data; recall runs via pure vector search.
    * `BlindIndex`: Indexes tokens using keyed HMAC-SHA256 digests, enabling exact keyword matching without persisting plaintext words on disk.
    * `Plaintext`: Standard FTS5 indexing (disallowed when encryption is active).
  * **In-Place Upgrades:** Unencrypted legacy databases can be opened with encryption enabled; historical rows remain readable while new writes are sealed.

---

### 2.2. Python Embedding Adapters & Direct Text Ingestion
* **Reference:** PR #25 (Commits `8f9500b`, `16f51de`, `2795b3d`, `7bbb320`)
* **Components Affected:** `py/vivvy/embeddings.py`, `py/vivvy/__init__.py`, `py/src/lib.rs`, `ltm/src/repository.rs`
* **Capabilities:**
  * **Model & Dimension Persistence:** Stored `dimensions` and `embedding_model` in the `store_meta` SQLite table. Opening a store with conflicting vector dimensions or model identifiers fails with an explicit configuration error.
  * **Standard Embedding Adapters (`py/vivvy/embeddings.py`):**
    * `OpenAIEmbeddingAdapter` (e.g. `text-embedding-3-small`, `text-embedding-3-large`)
    * `CohereEmbeddingAdapter` (e.g. `embed-english-v3.0`)
    * `FastEmbedEmbeddingAdapter` (local ONNX runtime)
    * `HuggingFaceEmbeddingAdapter` / `SentenceTransformersEmbeddingAdapter` (local PyTorch models)
  * **Direct Text APIs:**
    * `store.remember_text(content, ...)`: Automatically computes text embeddings via the configured adapter and persists memory records.
    * `store.recall_text(query_text, ...)`: Automatically embeds search queries and executes hybrid vector/lexical retrieval.
  * **PyO3 Bindings & Typing:** Exposed `.dimensions` and `.embedding_model` attributes on `MemoryStore` and packaged inline type definitions (`py.typed`).

---

### 2.3. Dynamic Multi-Factor Scoring Engine & SDK Bindings
* **Reference:** PR #26 (Commits `8f8d171`, `d5b5344`, `dff13b5`, `bdcf26d`, `fde51c1`, `6ba2c93`, `6102a43`, `8556c3b`)
* **Components Affected:** `ltm/src/model.rs`, `ltm/src/lib.rs`, `ffi/src/types.rs`, `go/vivvy.go`, `py/vivvy/scoring.py`
* **Capabilities:**
  * **Dynamic Scoring Formula:** Replaced fixed weights with a fully configurable `ScoringConfig`:
    $$\text{FinalScore} = w_{\text{sim}} \cdot S_{\text{similarity}} + w_{\text{rec}} \cdot S_{\text{recency}} + w_{\text{imp}} \cdot S_{\text{importance}} + w_{\text{reinf}} \cdot S_{\text{reinforcement}}$$
  * **Customizable Decay & MMR:** Configurable exponential decay half-life (`decay_half_life_days`), logarithmic reinforcement scaling, and Maximal Marginal Relevance (`mmr_lambda`) diversity control.
  * **Cross-Language SDK Parity:**
    * **Rust:** Added `ScoringConfig::builder()` attached to `RecallRequest`.
    * **Python:** Added `ScoringConfig` dataclass and integrated it into `recall()` and `recall_text()`.
    * **C-FFI:** Serialized `ScoringConfig` into the JSON FFI boundary (`RecallRequestJson`).
    * **Go SDK:** Added native Go `ScoringConfig` struct in `go/vivvy.go` with FFI translation.
  * **Recipes & Documentation:** Published parameter tuning recipes, zero-overhead execution guarantees, and rank-inversion test suites.

---

### 2.4. Vector WAL Deletion Durability & Segment Garbage Collection
* **Reference:** Branch `fix/hnsw-upsert-stale-nodes` (Commit `d4f204a`)
* **Components Affected:** `vec/src/storage/wal.rs`, `vec/src/concurrent.rs`, `vec/src/hnsw.rs`
* **Capabilities:**
  * **Persistent WAL Deletions:** Added WAL opcode `0x02` for write-ahead durability of vector deletions.
  * **Roaring Bitmap Tombstone Set:** High-speed, bit-compressed membership testing using `RoaringTreemap` across multi-sharded vector indexes.
  * **Garbage-Collected Segment Flushes:** Segment snapshots automatically prune tombstoned nodes during compaction passes, reclaiming memory and disk storage without downtime.

---

## 3. Summary of Commits & Pull Requests

| Commit / PR | Type | Title / Focus |
| :--- | :--- | :--- |
| `b567930` | **Fix** | Numeric safety, slice bounds checks, and API ergonomics |
| **PR #21** (`2681eba`, `dc5f9a0`) | **Fix** | Fix cross-tenant forget vulnerability (Issue #1) |
| **PR #22** (`6ec53ed`, `633101f`, `77bc759`) | **Fix** | Scope operation lookups and validate idempotent payloads (Issue #2) |
| **PR #23** (`b4ede93`, `5681535`, `32044f7`, `8ffb6f1`) | **Fix** | Make vector index ephemeral to resolve rebuild persistence mismatch (Issue #3) |
| **PR #24** (`268878d`, `de7cc79`) | **Feature** | Per-tenant encryption at rest with key management and migration support |
| **PR #25** (`8f9500b`, `16f51de`, `2795b3d`, `7bbb320`) | **Feature** | Built-in embedding adapters and `remember_text`/`recall_text` methods |
| **PR #26** (`8f8d171`..`8556c3b`) | **Feature** | Configurable scoring weights and reranking engine across Python, C-FFI, and Go SDKs |
| **PR #27** (`49dedc2`, `fa6ed23`, `460f7ca`) | **Fix** | Scope starvation resolution and candidate widening in vector recall (Issue #5) |
| `d4f204a` (`fix/hnsw-upsert-stale-nodes`) | **Fix & Feature** | HNSW upsert semantics, tombstoning, and WAL deletion support |
