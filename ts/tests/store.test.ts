import test from 'node:test';
import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

import {
  MemoryStore,
  DeterministicEmbedder,
  VivvyError,
  VivvyErrorCode,
} from '../src/index.ts';

function createTempDir(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), `vivvy-test-${prefix}-`));
}

test('MemoryStore: open and basic remember with vector', async () => {
  const dir = createTempDir('store-basic');
  try {
    const store = await MemoryStore.open({
      path: dir,
      dimensions: 4,
      embeddingModel: 'test-dim-4',
    });

    assert.equal(store.dimensions, 4);
    assert.equal(store.embeddingModel, 'test-dim-4');

    const vec = new Float32Array([1, 0, 0, 0]);
    const id = await store.remember(
      {
        tenantId: 'tenant_1',
        namespace: 'default',
        content: 'Fact number one',
        kind: 'fact',
        importance: 0.8,
      },
      vec
    );

    assert.ok(typeof id === 'string' && id.length > 0);

    const queryVec = new Float32Array([0.9, 0.1, 0, 0]);
    const res = await store.recall(
      {
        tenantId: 'tenant_1',
        namespace: 'default',
      },
      queryVec,
      null,
      5
    );

    assert.equal(res.items.length, 1);
    assert.equal(res.items[0].id, id);
    assert.equal(res.items[0].content, 'Fact number one');
    assert.equal(res.items[0].kind, 'fact');
    assert.ok(res.items[0].score > 0);

    store.close();
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('MemoryStore: dimension mismatch validation', async () => {
  const dir = createTempDir('store-dim-mismatch');
  try {
    const store = await MemoryStore.open({
      path: dir,
      dimensions: 4,
      embeddingModel: 'test-dim-4',
    });

    const wrongVec = new Float32Array([1, 0, 0]); // 3 dims instead of 4

    await assert.rejects(
      async () => {
        await store.remember(
          {
            tenantId: 'tenant_1',
            namespace: 'default',
            content: 'Invalid dimension vector',
          },
          wrongVec
        );
      },
      (err: any) => {
        assert.ok(err instanceof VivvyError);
        assert.equal(err.code, VivvyErrorCode.DimensionMismatch);
        return true;
      }
    );

    store.close();
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('MemoryStore: embedder mismatch validation on open', async () => {
  const dir = createTempDir('store-embedder-mismatch');
  try {
    const embedder = new DeterministicEmbedder('model-a', 8);

    assert.throws(
      () => {
        MemoryStore.openSync({
          path: dir,
          dimensions: 16, // Declares 16 but embedder is 8
          embedder,
        });
      },
      (err: any) => {
        assert.ok(err instanceof VivvyError);
        assert.equal(err.code, VivvyErrorCode.DimensionMismatch);
        return true;
      }
    );
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
