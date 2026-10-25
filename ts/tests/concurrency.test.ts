import test from 'node:test';
import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

import { MemoryStore, DeterministicEmbedder } from '../src/index.ts';

function createTempDir(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), `vivvy-test-${prefix}-`));
}

test('Concurrency: Parallel asynchronous inserts and recalls', async () => {
  const dir = createTempDir('concurrency-async');
  try {
    const embedder = new DeterministicEmbedder('model-concurrent', 16);
    const store = await MemoryStore.open({ path: dir, embedder });

    // 1. Parallel batch insert across 20 asynchronous tasks
    const count = 20;
    const insertPromises = Array.from({ length: count }, (_, i) => {
      return store.rememberText({
        tenantId: 'concurrent_tenant',
        namespace: 'batch_ns',
        content: `Concurrent record payload index ${i} with unique marker ${i * 7}`,
        kind: i % 2 === 0 ? 'fact' : 'preference',
        importance: 0.5 + (i / 100),
      });
    });

    const ids = await Promise.all(insertPromises);
    assert.equal(ids.length, count);
    const uniqueIds = new Set(ids);
    assert.equal(uniqueIds.size, count);

    // 2. Parallel concurrent recalls
    const recallPromises = Array.from({ length: 5 }, (_, i) => {
      return store.recallText({
        tenantId: 'concurrent_tenant',
        namespace: 'batch_ns',
        queryText: `payload index ${i}`,
        topK: 3,
      });
    });

    const results = await Promise.all(recallPromises);
    assert.equal(results.length, 5);
    for (const res of results) {
      assert.ok(res.items.length >= 1);
    }

    store.close();
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
