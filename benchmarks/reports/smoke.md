# Vivvy benchmark report

- Run timestamp (Unix seconds): `1792998021`
- Profile: `smoke`
- Git revision: `419490a-dirty`
- Command: `cargo run --release --package bench -- --profile smoke`

## Workload declaration

| Field | Value |
| --- | ---: |
| Records | 1000 |
| Embedding dimensions | 384 |
| Embedding model label | `synthetic-deterministic-v1` |
| Tenants / namespace | 10 / `agent-memory` |
| Measured queries / warm-up queries | 200 / 20 |
| Recall top-k | 10 |
| Delete rate before vacuum | 10% |
| RNG seed | `0x5649565659` |

## Environment

| Field | Value |
| --- | --- |
| OS / kernel | linux / 7.3.0-4-cachyos |
| CPU | 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz |
| Logical CPUs | 12 |
| Memory | 16082584 KiB |
| Rust | rustc 1.97.1 (8bab26f4f 2026-07-14) |

## Results

| Scenario | Ops/s | Mean ms | p50 ms | p95 ms | p99 ms | Correctness |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| synthetic core ANN recall | 380.8 | 2.626 | 2.615 | 2.899 | 2.983 | exact-oracle Recall@10=100.00% |
| cold start (empty store) | 18.5 | 54.098 | 54.098 | 54.098 | 54.098 | store opened |
| durable single-record writes | 761.0 | 1.314 | 1.572 | 2.021 | 2.187 | 1000 IDs persisted |
| warm start + index rebuild | 1.0 | 965.531 | 965.531 | 965.531 | 965.531 | 1000 active records rebuilt |
| vector recall (scoped) | 36.7 | 27.280 | 27.007 | 39.075 | 40.625 | synthetic target Recall@10=100.00%; cross-scope results=0 |
| hybrid recall + MMR (scoped) | 84.7 | 11.797 | 11.760 | 12.576 | 12.669 | synthetic target Recall@10=100.00%; cross-scope results=0 |
| hybrid filtered recall (scope + kind) | 85.0 | 11.762 | 11.765 | 12.432 | 12.582 | synthetic target Recall@10=100.00%; cross-scope results=0 |
| token-budgeted context formatting | 85.0 | 11.763 | 11.721 | 12.299 | 12.460 | 100 outputs respected 512-token estimator budget |
| online backup + restore open | 982.9 | 1011.460 | 1011.460 | 1011.460 | 1011.460 | restored 1000 active records |
| tombstone writes | 2216.8 | 0.451 | 0.425 | 0.633 | 0.851 | 100 records hidden from reads |
| tombstone vacuum + index rebuild | 111.8 | 894.503 | 894.503 | 894.503 | 894.503 | 100 tombstones physically purged |

## Storage lifecycle

| Measure | Before vacuum | After vacuum |
| --- | ---: | ---: |
| SQLite database bytes | 2437120 | 2453504 |
| SQLite WAL bytes | 0 | 4173592 |

## Interpretation and limits

- End-to-end scenarios exercise SQLite WAL, the operation journal, FTS5, HNSW, scope isolation, scoring, MMR, backup/restore, and tombstone vacuuming.
- Latency uses nearest-rank percentiles. Throughput is measured wall-clock operations divided by the measured interval. This runner is **single-threaded**; it does not claim multi-client capacity.
- ANN and retrieval targets use deterministic random vectors and unique text tokens. Their Recall@k validates mechanics and isolation only; it is **not** semantic memory quality.
- Before public claims, repeat each profile at least five times on a quiet pinned host, report median and spread, and run labelled LoCoMo/LongMemEval (or equivalent) with the exact embedding model and evaluator version.
- Keep embedding-service latency separate from MemoryStore latency.

## Scenario notes

- **synthetic core ANN recall** — Random L2 vectors (N=1000); build 0.892s (1121.4 inserts/s). Validates ANN mechanics only, not semantic quality.
  Samples: 200; min/max: 2.285/3.375 ms.
- **cold start (empty store)** — Includes SQLite schema/WAL initialization and empty index creation.
  Samples: 1; min/max: 54.098/54.098 ms.
- **durable single-record writes** — Each operation includes SQLite persistence, FTS maintenance, operation journal, and HNSW insertion.
  Samples: 1000; min/max: 0.131/3.036 ms.
- **warm start + index rebuild** — SQLite is canonical; opening rebuilds the ephemeral vector accelerator.
  Samples: 1; min/max: 965.531/965.531 ms.
- **vector recall (scoped)** — Dense HNSW candidate recall scoped by MemoryScope.
  Samples: 200; min/max: 4.536/41.752 ms.
- **hybrid recall + MMR (scoped)** — FTS5 + dense HNSW + RRF + default scoring + MMR(lambda=0.70).
  Samples: 200; min/max: 10.790/12.809 ms.
- **hybrid filtered recall (scope + kind)** — Hybrid recall with tenant scope and MemoryKind predicate.
  Samples: 200; min/max: 10.491/12.834 ms.
- **token-budgeted context formatting** — Includes hybrid recall and formatting; token counting is the implementation's char/4 approximation.
  Samples: 100; min/max: 10.346/12.543 ms.
- **online backup + restore open** — Elapsed includes SQLite VACUUM INTO backup and restore index rebuild; latency columns represent restore-open.
  Samples: 1; min/max: 1011.460/1011.460 ms.
- **tombstone writes** — Logical delete; physical reclamation is measured by vacuum.
  Samples: 100; min/max: 0.369/0.898 ms.
- **tombstone vacuum + index rebuild** — Physical deletion triggers a derived-vector-index rebuild in this implementation.
  Samples: 1; min/max: 894.503/894.503 ms.
