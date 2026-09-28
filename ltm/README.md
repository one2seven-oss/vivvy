# vivy-memory

Local, durable, namespace-isolated long-term memory runtime for AI agents.

## Features

- **Multi-Tenant Isolation**: Scoped by tenant, namespace, agent, and user.
- **Canonical SQLite WAL**: Crash-safe durable storage with operation journaling.
- **Hybrid Recall**: Dense HNSW vector search + SQLite FTS5 lexical search with Reciprocal Rank Fusion (RRF).
- **Explainable Scoring & MMR**: Weighted scoring (Similarity, Importance, Recency, Reinforcement) and diversity reranking.
- **Token-Budgeted Context Formatting**: Custom prompt templates with exact token bounds.
- **Live Online Backups**: Crash-consistent snapshots without stopping operations.

## License

BSL-1.1
