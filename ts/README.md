# vivvy (TypeScript & JavaScript)

**Single-machine vector engine and durable long-term memory for AI agents.**

High-performance, namespace-isolated long-term memory (LTM) runtime for Node.js and Bun, powered by Vivvy's native Rust core via Node-API (N-API).

- **Durable by construction**: SQLite WAL is the canonical store; HNSW vector index is a fast, auto-rebuildable accelerator.
- **Worker-backed memory operations**: `remember`, `recall`, and context formatting run off the JavaScript event loop; explicit lifecycle/maintenance APIs are synchronous.
- **Typed-vector ingestion**: `Float32Array` avoids JSON encoding; the native
  bridge copies values into Rust-owned memory before durable/index insertion.
- **Multi-tenant isolation**: Strict tenant and namespace boundaries enforced at the query layer.
- **Hybrid recall**: SQLite FTS5 lexical keyword matching fused with dense vector search (Reciprocal Rank Fusion) and explainable 4-factor scoring.
- **Token-budgeted context formatting**: Render recalled memories directly into prompt blocks within exact LLM token budgets.

---

## Quick Start

```typescript
import { MemoryStore, DeterministicEmbedder, OpenAIEmbedder } from 'vivvy';

// 1. Initialize with an embedder adapter (or local Ollama / OpenAI)
const embedder = new OpenAIEmbedder({ model: 'text-embedding-3-small' });

const store = await MemoryStore.open({
  path: './agent_memory_data',
  embedder,
});

// 2. Remember facts, instructions, or preferences
const id = await store.rememberText({
  tenantId: 'acme_corp',
  namespace: 'support_chat',
  content: 'User prefers concise TypeScript examples without semicolons.',
  kind: 'preference', // 'fact' | 'preference' | 'instruction' | 'context' | 'episodic'
  importance: 0.9,
});

// 3. Hybrid Recall (Vector + SQLite FTS5 Lexical Search)
const recall = await store.recallText({
  tenantId: 'acme_corp',
  namespace: 'support_chat',
  queryText: 'coding preferences',
  topK: 5,
});

for (const item of recall.items) {
  console.log(`[Score: ${item.score.toFixed(3)}] ${item.content}`);
}

// 4. Render directly into an LLM prompt block bounded by token budget
const promptContext = await store.formatContextText({
  tenantId: 'acme_corp',
  namespace: 'support_chat',
  queryText: 'coding preferences',
  maxTokens: 1000,
});
console.log(promptContext);

// 5. Close store when done
store.close();
```

---

## Custom Scoring Weights

Tailor reranking to specialized agent workloads:

```typescript
const results = await store.recallText({
  tenantId: 'acme_corp',
  namespace: 'support_chat',
  queryText: 'TypeScript patterns',
  scoring: {
    weightSimilarity: 0.85,
    weightImportance: 0.15,
    weightRecency: 0.0,
  },
});
```

---

## Direct Vector Ingestion

You can also bypass embedders and supply pre-computed `Float32Array` vectors directly:

```typescript
const vector = new Float32Array([0.1, 0.4, -0.2, /* ... */]);
await store.remember({
  tenantId: 'team_a',
  namespace: 'docs',
  content: 'Raw vector ingested document',
}, vector);
```

## Memory lifecycle and raw vector search

The unified package also exposes CRUD/maintenance methods and a standalone
`VectorIndex` for applications that do not need durable agent-memory records.

```typescript
import { MemoryStore, VectorIndex } from 'vivvy';

const ids = await store.rememberBatch(records, vectors);
const scope = { tenantId: 'team_a', namespace: 'docs' };
const record = store.get(scope, ids[0]);
store.update({ ...scope, id: ids[0], expectedRevision: record!.revision, content: 'revised' });
store.forget(scope, ids[0]);
console.log(store.health());

const index = VectorIndex.open({ dimensions: 768, metric: 'cosine' });
const id = index.insert(new Float32Array(768), { category: 'docs' });
const hits = index.search(new Float32Array(768), 10, {
  eq: { field: 'category', value: 'docs' },
});
index.delete(id);
index.close();
```

`remember`, `recall`, and `formatContext` run through the Node worker pool.
The lifecycle/maintenance methods above are synchronous native calls; run
large rebuild or compaction work outside latency-sensitive request handlers.
