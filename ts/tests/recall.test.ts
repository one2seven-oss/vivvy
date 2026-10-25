import test from 'node:test';
import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

import { MemoryStore, DeterministicEmbedder } from '../src/index.ts';

function createTempDir(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), `vivvy-test-${prefix}-`));
}

test('Recall: hybrid search and text convenience methods', async () => {
  const dir = createTempDir('recall-hybrid');
  try {
    const embedder = new DeterministicEmbedder('model-hybrid', 32);
    const store = await MemoryStore.open({ path: dir, embedder });

    await store.rememberText({
      tenantId: 't1',
      namespace: 'ns_chat',
      content: 'The user prefers dark mode theme across all UI settings.',
      kind: 'preference',
      importance: 0.9,
    });

    await store.rememberText({
      tenantId: 't1',
      namespace: 'ns_chat',
      content: 'Antigravity AI is an agentic pair programmer by DeepMind.',
      kind: 'fact',
      importance: 0.8,
    });

    await store.rememberText({
      tenantId: 't1',
      namespace: 'ns_chat',
      content: 'Always format code blocks with triple backticks and language name.',
      kind: 'instruction',
      importance: 1.0,
    });

    // 1. Recall by text
    const recallRes = await store.recallText({
      tenantId: 't1',
      namespace: 'ns_chat',
      queryText: 'dark mode UI preferences',
      topK: 3,
    });

    assert.ok(recallRes.items.length >= 1);
    const prefItem = recallRes.items.find((i) => i.kind === 'preference');
    assert.ok(prefItem, 'Expected preference item to be recalled');
    assert.match(prefItem.content, /dark mode/);

    // 2. Filter by kind
    const instructionRecall = await store.recallText({
      tenantId: 't1',
      namespace: 'ns_chat',
      queryText: 'formatting settings',
      kinds: ['instruction'],
      topK: 5,
    });

    assert.equal(instructionRecall.items.length, 1);
    assert.equal(instructionRecall.items[0].kind, 'instruction');
    assert.match(instructionRecall.items[0].content, /triple backticks/);

    // 3. Format Context
    const contextPrompt = await store.formatContextText({
      tenantId: 't1',
      namespace: 'ns_chat',
      queryText: 'code block instructions',
      maxTokens: 500,
    });

    assert.match(contextPrompt, /Relevant Agent Memories/);
    assert.match(contextPrompt, /instruction/);

    store.close();
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('Recall: custom scoring weights (recency and similarity)', async () => {
  const dir = createTempDir('recall-scoring');
  try {
    const embedder = new DeterministicEmbedder('model-scoring', 16);
    const store = await MemoryStore.open({ path: dir, embedder });

    await store.rememberText({
      tenantId: 't1',
      namespace: 'ns_scoring',
      content: 'Core architecture documentation',
      kind: 'fact',
      importance: 0.5,
    });

    const res = await store.recallText({
      tenantId: 't1',
      namespace: 'ns_scoring',
      queryText: 'architecture',
      scoring: {
        weightSimilarity: 0.9,
        weightImportance: 0.1,
        weightRecency: 0.0,
      },
    });

    assert.equal(res.items.length, 1);
    assert.ok(res.items[0].explanation);
    assert.ok(res.items[0].explanation.similarityScore >= 0);

    store.close();
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
