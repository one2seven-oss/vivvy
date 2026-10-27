# Vivvy benchmark results

This document consolidates every benchmark run currently recorded in this
workspace. Detailed per-run artifacts remain in
[`benchmarks/reports/`](benchmarks/reports/).

## Important interpretation limits

- These are **single-threaded, single-machine** measurements. They do not
  represent multi-client throughput or distributed-database capacity.
- The corpus uses deterministic random embeddings and unique text tokens.
  Synthetic Recall@k verifies HNSW mechanics, hybrid-path correctness, and
  tenant isolation; it does **not** measure semantic retrieval quality.
- The runner measures Vivvy after an embedding is available. Remote/local
  embedding latency is not included.
- Each profile was run once. Repeat a profile at least five times on a quiet
  benchmark host and report median plus spread before making public claims.

## Test environment

| Field | Value |
| --- | --- |
| Git revision | `419490a-dirty` |
| OS / kernel | Linux / `7.3.0-4-cachyos` |
| CPU | 11th Gen Intel Core i5-11400H @ 2.70 GHz |
| Logical CPUs | 12 |
| RAM | 16,082,584 KiB (~15.3 GiB) |
| Rust | `rustc 1.97.1 (8bab26f4f 2026-07-14)` |
| Benchmark model label | `synthetic-deterministic-v1` |
| RNG seed | `0x5649565659` |

## Workload profiles

| Profile | Run timestamp (Unix) | Records | Dimensions | Tenants | Queries / warm-up | top-k | Tombstones |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Smoke | `1792998021` | 1,000 | 384 | 10 | 200 / 20 | 10 | 10% (100) |
| Standard | `1792997667` | 10,000 | 768 | 50 | 1,000 / 100 | 10 | 10% (1,000) |

## Headline comparison

| Scenario | Smoke: p95 / throughput | Standard: p95 / throughput |
| --- | --- | --- |
| Core ANN recall | 2.899 ms / 380.8 QPS | 26.983 ms / 39.6 QPS |
| Durable single-record write | 2.021 ms / 761.0 writes/s | 11.009 ms / 144.0 writes/s |
| Warm start + index rebuild | 965.531 ms | 45,369.420 ms |
| Scoped vector recall | 39.075 ms / 36.7 QPS | 274.927 ms / 4.6 QPS |
| Hybrid recall + MMR | 12.576 ms / 84.7 QPS | 62.220 ms / 17.5 QPS |
| Hybrid filtered recall | 12.432 ms / 85.0 QPS | 63.502 ms / 17.5 QPS |
| Context formatting | 12.299 ms / 85.0 QPS | 89.498 ms / 15.6 QPS |
| Backup + restore open | 1,011.460 ms | 43,114.649 ms |
| Tombstone write | 0.633 ms / 2,216.8 writes/s | 3.662 ms / 312.5 writes/s |
| Vacuum + index rebuild | 894.503 ms | 41,992.959 ms |

The 10K profile shows the present scaling constraint: reconstruction of the
ephemeral HNSW accelerator dominates warm start, restored-store readiness, and
post-vacuum work.

## Smoke profile — 1,000 records, 384 dimensions

Command: `cargo run --release --package bench -- --profile smoke`

| Scenario | Throughput | Mean ms | p50 ms | p95 ms | p99 ms | Correctness |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Synthetic core ANN recall | 380.8 QPS | 2.626 | 2.615 | 2.899 | 2.983 | Exact-oracle Recall@10 = 100.00% |
| Cold start (empty store) | 18.5 opens/s | 54.098 | 54.098 | 54.098 | 54.098 | Store opened |
| Durable single-record writes | 761.0 writes/s | 1.314 | 1.572 | 2.021 | 2.187 | 1,000 IDs persisted |
| Warm start + index rebuild | 1.0 open/s | 965.531 | 965.531 | 965.531 | 965.531 | 1,000 active records rebuilt |
| Vector recall (scoped) | 36.7 QPS | 27.280 | 27.007 | 39.075 | 40.625 | Target Recall@10 = 100.00%; cross-scope results = 0 |
| Hybrid recall + MMR (scoped) | 84.7 QPS | 11.797 | 11.760 | 12.576 | 12.669 | Target Recall@10 = 100.00%; cross-scope results = 0 |
| Hybrid filtered recall (scope + kind) | 85.0 QPS | 11.762 | 11.765 | 12.432 | 12.582 | Target Recall@10 = 100.00%; cross-scope results = 0 |
| Token-budgeted context formatting | 85.0 QPS | 11.763 | 11.721 | 12.299 | 12.460 | 100 outputs respected the 512-token estimator budget |
| Online backup + restore open | 982.9 records/s | 1,011.460 | 1,011.460 | 1,011.460 | 1,011.460 | Restored 1,000 active records |
| Tombstone writes | 2,216.8 writes/s | 0.451 | 0.425 | 0.633 | 0.851 | 100 records hidden from reads |
| Tombstone vacuum + index rebuild | 111.8 records/s | 894.503 | 894.503 | 894.503 | 894.503 | 100 tombstones physically purged |

| Storage metric | Before vacuum | After vacuum |
| --- | ---: | ---: |
| SQLite database | 2,437,120 bytes | 2,453,504 bytes |
| SQLite WAL | 0 bytes | 4,173,592 bytes |

Additional notes:

- Core ANN used 1,000 random L2 vectors; HNSW build took 0.892 s (1,121.4 inserts/s).
- Sample ranges are recorded in [the smoke report](benchmarks/reports/smoke.md).

## Standard profile — 10,000 records, 768 dimensions

Command: `cargo run --release --package bench -- --profile standard`

| Scenario | Throughput | Mean ms | p50 ms | p95 ms | p99 ms | Correctness |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Synthetic core ANN recall | 39.6 QPS | 25.244 | 25.300 | 26.983 | 27.432 | Exact-oracle Recall@10 = 98.55% |
| Cold start (empty store) | 18.2 opens/s | 54.896 | 54.896 | 54.896 | 54.896 | Store opened |
| Durable single-record writes | 144.0 writes/s | 6.945 | 6.867 | 11.009 | 12.857 | 10,000 IDs persisted |
| Warm start + index rebuild | 0.022 opens/s | 45,369.420 | 45,369.420 | 45,369.420 | 45,369.420 | 10,000 active records rebuilt |
| Vector recall (scoped) | 4.6 QPS | 216.963 | 195.129 | 274.927 | 294.402 | Target Recall@10 = 98.30%; cross-scope results = 0 |
| Hybrid recall + MMR (scoped) | 17.5 QPS | 57.120 | 56.738 | 62.220 | 67.029 | Target Recall@10 = 100.00%; cross-scope results = 0 |
| Hybrid filtered recall (scope + kind) | 17.5 QPS | 57.185 | 56.639 | 63.502 | 70.051 | Target Recall@10 = 100.00%; cross-scope results = 0 |
| Token-budgeted context formatting | 15.6 QPS | 64.187 | 61.325 | 89.498 | 96.051 | 100 outputs respected the 512-token estimator budget |
| Online backup + restore open | 231.4 records/s | 43,114.649 | 43,114.649 | 43,114.649 | 43,114.649 | Restored 10,000 active records |
| Tombstone writes | 312.5 writes/s | 3.199 | 3.144 | 3.662 | 4.035 | 1,000 records hidden from reads |
| Tombstone vacuum + index rebuild | 23.8 records/s | 41,992.959 | 41,992.959 | 41,992.959 | 41,992.959 | 1,000 tombstones physically purged |

| Storage metric | Before vacuum | After vacuum |
| --- | ---: | ---: |
| SQLite database | 44,306,432 bytes | 44,310,528 bytes |
| SQLite WAL | 0 bytes | 4,148,872 bytes |

Additional notes:

- Core ANN used 10,000 random L2 vectors; HNSW build took 44.045 s (227.0 inserts/s).
- Sample ranges are recorded in [the standard report](benchmarks/reports/standard.md).

## What each scenario includes

| Scenario | Included Vivvy work |
| --- | --- |
| Durable writes | SQLite persistence and WAL, operation journal, FTS maintenance, HNSW insert |
| Warm start | Startup recovery plus rebuild of the derived in-memory vector index from SQLite |
| Vector recall | Dense HNSW candidate retrieval with tenant/namespace scope validation |
| Hybrid recall | FTS5, dense HNSW, reciprocal-rank fusion, default four-factor scoring, MMR (`lambda=0.70`) |
| Filtered hybrid recall | Hybrid recall plus `MemoryKind` filter and scope isolation validation |
| Context formatting | Hybrid recall plus `format_context` using a 512-token char/4 estimate |
| Backup + restore | SQLite `VACUUM INTO` backup and restored-store index rebuild |
| Vacuum | Physical tombstone purge followed by derived-index rebuild |

## Publication checklist

Do not quote the synthetic Recall@10 values as relevance quality. Before a
public benchmark, add a real embedding model and labelled agent-memory dataset,
then record dataset/evaluator versions, model and normalisation, HNSW settings,
filters, scoring/MMR configuration, concurrency, and run-to-run variability.
