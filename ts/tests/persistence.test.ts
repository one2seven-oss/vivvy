import test from 'node:test';
import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

import { MemoryStore, DeterministicEmbedder } from '../src/index.ts';

function createTempDir(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), `vivvy-test-${prefix}-`));
}

test('Persistence: Reopening existing database recovers memories', async () => {
  const dir = createTempDir('persistence-reopen');
  try {
    const embedder = new DeterministicEmbedder('model-persist', 16);

    // 1. Open and insert records
    let store = await MemoryStore.open({ path: dir, embedder });
    const id1 = await store.rememberText({
      tenantId: 'persist_t1',
      namespace: 'persist_ns',
      content: 'Persistent memory across process restart',
      kind: 'fact',
      importance: 0.85,
    });
    assert.ok(id1);

    // 2. Close store
    store.close();

    // 3. Re-open store at the exact same directory
    store = await MemoryStore.open({ path: dir, embedder });

    // 4. Recall record to prove SQLite WAL persistence
    const recall = await store.recallText({
      tenantId: 'persist_t1',
      namespace: 'persist_ns',
      queryText: 'Persistent memory restart',
      topK: 5,
    });

    assert.equal(recall.items.length, 1);
    assert.equal(recall.items[0].id, id1);
    assert.equal(recall.items[0].content, 'Persistent memory across process restart');

    store.close();
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('Persistence: Live online backup and restore', async () => {
  const mainDir = createTempDir('backup-source');
  const backupDir = createTempDir('backup-dest');
  try {
    const embedder = new DeterministicEmbedder('model-backup', 16);

    const store = await MemoryStore.open({ path: mainDir, embedder });
    await store.rememberText({
      tenantId: 't_backup',
      namespace: 'ns_backup',
      content: 'Critical memory backed up without downtime',
      kind: 'instruction',
      importance: 0.99,
    });

    // Create online backup
    await store.backup(backupDir);
    store.close();

    // Open the backup store directory
    const backupStore = await MemoryStore.open({ path: backupDir, embedder });
    const recall = await backupStore.recallText({
      tenantId: 't_backup',
      namespace: 'ns_backup',
      queryText: 'Critical memory backed up',
      topK: 5,
    });

    assert.equal(recall.items.length, 1);
    assert.equal(recall.items[0].content, 'Critical memory backed up without downtime');

    backupStore.close();
  } finally {
    fs.rmSync(mainDir, { recursive: true, force: true });
    fs.rmSync(backupDir, { recursive: true, force: true });
  }
});
