import test from 'node:test';
import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

import { MemoryStore, DeterministicEmbedder } from '../src/index.ts';

function createTempDir(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), `vivvy-test-${prefix}-`));
}

test('Multi-Tenant Isolation: Tenant boundary enforcement', async () => {
  const dir = createTempDir('multitenant-isolation');
  try {
    const embedder = new DeterministicEmbedder('model-mt', 16);
    const store = await MemoryStore.open({ path: dir, embedder });

    // Store record under Tenant Alpha
    await store.rememberText({
      tenantId: 'tenant_alpha',
      namespace: 'shared_topic',
      content: 'Alpha secret corporate strategy',
      kind: 'fact',
      importance: 1.0,
    });

    // Store record under Tenant Beta
    await store.rememberText({
      tenantId: 'tenant_beta',
      namespace: 'shared_topic',
      content: 'Beta secret financial plan',
      kind: 'fact',
      importance: 1.0,
    });

    // Recall from Tenant Alpha — MUST NEVER return Beta records!
    const alphaRecall = await store.recallText({
      tenantId: 'tenant_alpha',
      namespace: 'shared_topic',
      queryText: 'secret plan strategy',
      topK: 10,
    });

    assert.equal(alphaRecall.items.length, 1);
    assert.equal(alphaRecall.items[0].tenantId, 'tenant_alpha');
    assert.equal(alphaRecall.items[0].content, 'Alpha secret corporate strategy');

    // Recall from Tenant Beta — MUST NEVER return Alpha records!
    const betaRecall = await store.recallText({
      tenantId: 'tenant_beta',
      namespace: 'shared_topic',
      queryText: 'secret plan strategy',
      topK: 10,
    });

    assert.equal(betaRecall.items.length, 1);
    assert.equal(betaRecall.items[0].tenantId, 'tenant_beta');
    assert.equal(betaRecall.items[0].content, 'Beta secret financial plan');

    // Recall from non-existent Tenant Gamma — MUST return 0 records
    const gammaRecall = await store.recallText({
      tenantId: 'tenant_gamma',
      namespace: 'shared_topic',
      queryText: 'secret plan strategy',
      topK: 10,
    });

    assert.equal(gammaRecall.items.length, 0);

    store.close();
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
