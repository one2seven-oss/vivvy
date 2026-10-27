# Vivvy benchmark runner

`bench` measures Vivvy's production-facing, single-machine `MemoryStore` path
rather than only a pure-vector happy path. It writes a self-contained Markdown
report with tail latency and correctness checks.

```sh
# Fast local validation
cargo run --release --package bench -- --profile smoke

# Representative 10K-record, 768-dimension local-agent workload
cargo run --release --package bench -- --profile standard
```

Reports are written to `benchmarks/reports/<profile>.md`; override that with
`--report path/to/report.md`. `large` is a 100K-record profile for a quiet,
provisioned benchmark host.

## Coverage

- pure HNSW throughput and Recall@k against an exact flat-index oracle;
- cold open, durable per-record writes, warm reopen/index rebuild;
- scoped vector recall, hybrid FTS5/vector RRF + MMR, and filtered recall;
- token-budgeted context formatting;
- online backup plus restored-store open; and
- tombstone writes, vacuum/index rebuild, and SQLite/WAL footprint.

The runner fails if persistence, restore, vacuum, or tenant-scope invariants
fail. It intentionally uses deterministic synthetic records for repeatable
operational measurement. Synthetic Recall@k must never be quoted as semantic
agent-memory quality.

For public relevance claims, use labelled data and report the dataset/version,
embedding model and normalisation, qrels/evaluator version, scoring/filter/MMR
configuration, and at least five independent runs (median plus spread). Keep
remote or local embedding latency separate from MemoryStore latency.
