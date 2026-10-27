# Vivvy benchmark report

- Run timestamp (Unix seconds): `1792997667`
- Profile: `standard`
- Git revision: `419490a-dirty`
- Command: `cargo run --release --package bench -- --profile standard`

## Workload declaration

| Field | Value |
| --- | ---: |
| Records | 10000 |
| Embedding dimensions | 768 |
| Embedding model label | `synthetic-deterministic-v1` |
| Tenants / namespace | 50 / `agent-memory` |
| Measured queries / warm-up queries | 1000 / 100 |
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
| synthetic core ANN recall | 39.6 | 25.244 | 25.300 | 26.983 | 27.432 | exact-oracle Recall@10=98.55% |
| cold start (empty store) | 18.2 | 54.896 | 54.896 | 54.896 | 54.896 | store opened |
| durable single-record writes | 144.0 | 6.945 | 6.867 | 11.009 | 12.857 | 10000 IDs persisted |
| warm start + index rebuild | 0.0 | 45369.420 | 45369.420 | 45369.420 | 45369.420 | 10000 active records rebuilt |
| vector recall (scoped) | 4.6 | 216.963 | 195.129 | 274.927 | 294.402 | synthetic target Recall@10=98.30%; cross-scope results=0 |
| hybrid recall + MMR (scoped) | 17.5 | 57.120 | 56.738 | 62.220 | 67.029 | synthetic target Recall@10=100.00%; cross-scope results=0 |
| hybrid filtered recall (scope + kind) | 17.5 | 57.185 | 56.639 | 63.502 | 70.051 | synthetic target Recall@10=100.00%; cross-scope results=0 |
| token-budgeted context formatting | 15.6 | 64.187 | 61.325 | 89.498 | 96.051 | 100 outputs respected 512-token estimator budget |
| online backup + restore open | 231.4 | 43114.649 | 43114.649 | 43114.649 | 43114.649 | restored 10000 active records |
| tombstone writes | 312.5 | 3.199 | 3.144 | 3.662 | 4.035 | 1000 records hidden from reads |
| tombstone vacuum + index rebuild | 23.8 | 41992.959 | 41992.959 | 41992.959 | 41992.959 | 1000 tombstones physically purged |

## Storage lifecycle

| Measure | Before vacuum | After vacuum |
| --- | ---: | ---: |
| SQLite database bytes | 44306432 | 44310528 |
| SQLite WAL bytes | 0 | 4148872 |

## Interpretation and limits

- End-to-end scenarios exercise SQLite WAL, the operation journal, FTS5, HNSW, scope isolation, scoring, MMR, backup/restore, and tombstone vacuuming.
- Latency uses nearest-rank percentiles. Throughput is measured wall-clock operations divided by the measured interval. This runner is **single-threaded**; it does not claim multi-client capacity.
- ANN and retrieval targets use deterministic random vectors and unique text tokens. Their Recall@k validates mechanics and isolation only; it is **not** semantic memory quality.
- Before public claims, repeat each profile at least five times on a quiet pinned host, report median and spread, and run labelled LoCoMo/LongMemEval (or equivalent) with the exact embedding model and evaluator version.
- Keep embedding-service latency separate from MemoryStore latency.

## Scenario notes

- **synthetic core ANN recall** — Random L2 vectors (N=10000); build 44.045s (227.0 inserts/s). Validates ANN mechanics only, not semantic quality.
  Samples: 200; min/max: 22.199/27.672 ms.
- **cold start (empty store)** — Includes SQLite schema/WAL initialization and empty index creation.
  Samples: 1; min/max: 54.896/54.896 ms.
- **durable single-record writes** — Each operation includes SQLite persistence, FTS maintenance, operation journal, and HNSW insertion.
  Samples: 10000; min/max: 0.164/16.635 ms.
- **warm start + index rebuild** — SQLite is canonical; opening rebuilds the ephemeral vector accelerator.
  Samples: 1; min/max: 45369.420/45369.420 ms.
- **vector recall (scoped)** — Dense HNSW candidate recall scoped by MemoryScope.
  Samples: 1000; min/max: 84.892/360.937 ms.
- **hybrid recall + MMR (scoped)** — FTS5 + dense HNSW + RRF + default scoring + MMR(lambda=0.70).
  Samples: 1000; min/max: 49.782/96.048 ms.
- **hybrid filtered recall (scope + kind)** — Hybrid recall with tenant scope and MemoryKind predicate.
  Samples: 1000; min/max: 48.195/112.971 ms.
- **token-budgeted context formatting** — Includes hybrid recall and formatting; token counting is the implementation's char/4 approximation.
  Samples: 100; min/max: 53.574/101.580 ms.
- **online backup + restore open** — Elapsed includes SQLite VACUUM INTO backup and restore index rebuild; latency columns represent restore-open.
  Samples: 1; min/max: 43114.649/43114.649 ms.
- **tombstone writes** — Logical delete; physical reclamation is measured by vacuum.
  Samples: 1000; min/max: 2.603/6.381 ms.
- **tombstone vacuum + index rebuild** — Physical deletion triggers a derived-vector-index rebuild in this implementation.
  Samples: 1; min/max: 41992.959/41992.959 ms.
