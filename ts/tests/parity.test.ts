import test from 'node:test';
import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

import { MemoryStore, VectorIndex } from '../src/index.ts';

function tempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'vivvy-parity-'));
}

test('MemoryStore CRUD, batch, health, and rebuild', async () => {
  const dir = tempDir();
  try {
    const store = MemoryStore.openSync({ path: dir, dimensions: 2, embeddingModel: 'test' });
    const ids = await store.rememberBatch([
      { tenantId: 'acme', namespace: 'default', content: 'first' },
      { tenantId: 'acme', namespace: 'default', content: 'second' },
    ], [new Float32Array([1, 0]), new Float32Array([0, 1])]);
    assert.equal(ids.length, 2);

    const scope = { tenantId: 'acme', namespace: 'default' };
    const first = store.get(scope, ids[0]);
    assert.equal(first?.content, 'first');
    store.update({ ...scope, id: ids[0], expectedRevision: first!.revision, content: 'updated' });
    assert.equal(store.get(scope, ids[0])?.content, 'updated');
    assert.equal(store.health().totalActiveRecords, 2);
    store.rebuildIndex();
    store.forget(scope, ids[0]);
    assert.equal(store.get(scope, ids[0]), null);
    store.forgetBatch(scope, [ids[1]]);
    store.close();
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test('VectorIndex inserts, filters, searches, and deletes', () => {
  const index = VectorIndex.open({ dimensions: 2, metric: 'l2' });
  try {
    const red = index.insert(new Float32Array([1, 0]), { color: 'red' });
    index.insert(new Float32Array([0, 1]), { color: 'blue' });
    assert.equal(index.insertBatch([new Float32Array([0.5, 0.5])]).length, 1);
    index.insertWithId(99n, new Float32Array([0.25, 0.75]));
    const results = index.search(new Float32Array([1, 0]), 5, { eq: { field: 'color', value: 'red' } });
    assert.equal(results.length, 1);
    assert.equal(BigInt(results[0].id), red);
    assert.equal(index.delete(red), true);
  } finally { index.close(); }
});
