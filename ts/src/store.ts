import { createRequire } from 'node:module';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { VivvyError, VivvyErrorCode } from './errors.ts';
import type { Embedder } from './embeddings.ts';
import type {
  FormatContextOptions,
  FormatContextTextOptions,
  MemoryKind,
  RecallItem,
  RecallOptions,
  RecallResponse,
  RecallTextOptions,
  RememberOptions,
  StoreConfig,
} from './types.ts';

// Resolve native addon path reliably across ESM and CJS
function loadNativeAddon() {
  const currentDir = typeof __dirname !== 'undefined'
    ? __dirname
    : (import.meta.url ? path.dirname(fileURLToPath(import.meta.url)) : process.cwd());

  const req = typeof require !== 'undefined'
    ? require
    : createRequire(import.meta.url || `file://${process.cwd()}/`);

  const possiblePaths = [
    path.join(currentDir, '../vivvy.node'),
    path.join(currentDir, 'vivvy.node'),
    path.join(process.cwd(), 'ts/vivvy.node'),
    path.join(process.cwd(), 'vivvy.node'),
  ];

  for (const p of possiblePaths) {
    try {
      return req(p);
    } catch {
      // try next
    }
  }

  // Fallback to direct require
  return req('./vivvy.node');
}

let nativeBinding: any = null;
function getNative() {
  if (!nativeBinding) {
    nativeBinding = loadNativeAddon();
  }
  return nativeBinding;
}

export class MemoryStore {
  private _nativeHandle: any;
  public readonly path: string;
  public readonly dimensions: number;
  public readonly embeddingModel: string;
  public embedder?: Embedder;
  private _closed = false;

  private constructor(
    nativeHandle: any,
    storePath: string,
    dimensions: number,
    embeddingModel: string,
    embedder?: Embedder
  ) {
    this._nativeHandle = nativeHandle;
    this.path = storePath;
    this.dimensions = dimensions;
    this.embeddingModel = embeddingModel;
    this.embedder = embedder;
  }

  /**
   * Open or create a durable MemoryStore.
   */
  static async open(config: StoreConfig): Promise<MemoryStore> {
    return MemoryStore.openSync(config);
  }

  /**
   * Synchronously open or create a durable MemoryStore.
   */
  static openSync(config: StoreConfig): MemoryStore {
    let dims = config.dimensions;
    let model = config.embeddingModel;

    if (config.embedder) {
      if (dims === undefined) {
        dims = config.embedder.dimensions;
      } else if (dims !== config.embedder.dimensions) {
        throw new VivvyError(
          VivvyErrorCode.DimensionMismatch,
          `Embedding dimension mismatch: embedder declared ${config.embedder.dimensions} dimensions, but store was configured with ${dims}`
        );
      }

      if (model === undefined) {
        model = config.embedder.name;
      } else if (model !== config.embedder.name) {
        throw new VivvyError(
          VivvyErrorCode.EmbeddingModelMismatch,
          `Embedding model mismatch: embedder model '${config.embedder.name}' does not match store model '${model}'`
        );
      }
    }

    if (dims === undefined || dims <= 0) {
      throw new VivvyError(
        VivvyErrorCode.InvalidInput,
        'dimensions must be specified or an embedder must be provided'
      );
    }

    if (!model) {
      throw new VivvyError(
        VivvyErrorCode.InvalidInput,
        'embeddingModel must be specified or an embedder must be provided'
      );
    }

    const native = getNative();
    const configPayload = {
      dimensions: dims,
      embedding_model: model,
      max_recall_limit: config.maxRecallLimit ?? 100,
    };

    try {
      const handle = native.storeOpen(config.path, JSON.stringify(configPayload));
      return new MemoryStore(handle, config.path, dims, model, config.embedder);
    } catch (err) {
      throw VivvyError.fromNativeError(err);
    }
  }

  private ensureOpen(): void {
    if (this._closed || !this._nativeHandle) {
      throw new VivvyError(VivvyErrorCode.StoreIOError, 'Store is closed');
    }
  }

  /**
   * Close the store, releasing SQLite handles and vector index memory.
   */
  close(): void {
    if (this._closed) return;
    this._closed = true;
    if (this._nativeHandle) {
      try {
        const native = getNative();
        native.storeClose(this._nativeHandle);
      } finally {
        this._nativeHandle = null;
      }
    }
  }

  /**
   * Store a memory record with a pre-computed vector embedding.
   */
  async remember(options: RememberOptions, vector: Float32Array): Promise<string> {
    this.ensureOpen();
    if (vector.length !== this.dimensions) {
      throw new VivvyError(
        VivvyErrorCode.DimensionMismatch,
        `Vector dimension mismatch: expected ${this.dimensions}, got ${vector.length}`
      );
    }

    const payload = this.marshalRemember(options);
    const native = getNative();
    try {
      return await native.storeInsert(this._nativeHandle, JSON.stringify(payload), vector);
    } catch (err) {
      throw VivvyError.fromNativeError(err);
    }
  }

  /**
   * Synchronously store a memory record with a pre-computed vector embedding.
   */
  rememberSync(options: RememberOptions, vector: Float32Array): string {
    this.ensureOpen();
    if (vector.length !== this.dimensions) {
      throw new VivvyError(
        VivvyErrorCode.DimensionMismatch,
        `Vector dimension mismatch: expected ${this.dimensions}, got ${vector.length}`
      );
    }

    const payload = this.marshalRemember(options);
    const native = getNative();
    try {
      return native.storeInsertSync(this._nativeHandle, JSON.stringify(payload), vector);
    } catch (err) {
      throw VivvyError.fromNativeError(err);
    }
  }

  /**
   * Automatically generate embeddings via the attached embedder and store text.
   */
  async rememberText(options: RememberOptions): Promise<string> {
    if (!this.embedder) {
      throw new VivvyError(
        VivvyErrorCode.InvalidInput,
        'Cannot call rememberText without an attached embedder. Pass embedder to MemoryStore.open or use remember() with a vector.'
      );
    }
    const vector = await this.embedder.embed(options.content);
    return this.remember(options, vector);
  }

  /**
   * Hybrid vector and full-text recall.
   */
  async recall(
    options: RecallOptions,
    queryVector: Float32Array,
    queryText?: string | null,
    topK = 10
  ): Promise<RecallResponse> {
    this.ensureOpen();
    if (queryVector.length !== this.dimensions) {
      throw new VivvyError(
        VivvyErrorCode.DimensionMismatch,
        `Query vector dimension mismatch: expected ${this.dimensions}, got ${queryVector.length}`
      );
    }

    const payload = this.marshalRecall(options);
    const native = getNative();
    try {
      const jsonStr = await native.storeRecall(
        this._nativeHandle,
        JSON.stringify(payload),
        queryVector,
        queryText ?? null,
        topK
      );
      return this.parseRecallResponse(jsonStr);
    } catch (err) {
      throw VivvyError.fromNativeError(err);
    }
  }

  /**
   * Synchronously perform hybrid recall.
   */
  recallSync(
    options: RecallOptions,
    queryVector: Float32Array,
    queryText?: string | null,
    topK = 10
  ): RecallResponse {
    this.ensureOpen();
    if (queryVector.length !== this.dimensions) {
      throw new VivvyError(
        VivvyErrorCode.DimensionMismatch,
        `Query vector dimension mismatch: expected ${this.dimensions}, got ${queryVector.length}`
      );
    }

    const payload = this.marshalRecall(options);
    const native = getNative();
    try {
      const jsonStr = native.storeRecallSync(
        this._nativeHandle,
        JSON.stringify(payload),
        queryVector,
        queryText ?? null,
        topK
      );
      return this.parseRecallResponse(jsonStr);
    } catch (err) {
      throw VivvyError.fromNativeError(err);
    }
  }

  /**
   * Automatically embed query text and execute hybrid recall.
   */
  async recallText(options: RecallTextOptions): Promise<RecallResponse> {
    if (!this.embedder) {
      throw new VivvyError(
        VivvyErrorCode.InvalidInput,
        'Cannot call recallText without an attached embedder.'
      );
    }
    const queryVector = await this.embedder.embed(options.queryText);
    return this.recall(options, queryVector, options.queryText, options.topK ?? 10);
  }

  /**
   * Format recall results directly into an LLM prompt block within a token budget.
   */
  async formatContext(
    options: RecallOptions,
    queryVector: Float32Array,
    queryText?: string | null,
    topK = 10,
    format?: FormatContextOptions
  ): Promise<string> {
    this.ensureOpen();
    const payload = this.marshalRecall(options);
    const formatPayload = format
      ? {
          max_tokens: format.maxTokens,
          template: format.template,
          header: format.header,
          footer: format.footer,
        }
      : null;

    const native = getNative();
    try {
      return await native.storeFormatContext(
        this._nativeHandle,
        JSON.stringify(payload),
        queryVector,
        queryText ?? null,
        topK,
        formatPayload ? JSON.stringify(formatPayload) : null
      );
    } catch (err) {
      throw VivvyError.fromNativeError(err);
    }
  }

  /**
   * Automatically embed text and format prompt context block within a token budget.
   */
  async formatContextText(options: FormatContextTextOptions): Promise<string> {
    if (!this.embedder) {
      throw new VivvyError(
        VivvyErrorCode.InvalidInput,
        'Cannot call formatContextText without an attached embedder.'
      );
    }
    const queryVector = await this.embedder.embed(options.queryText);
    return this.formatContext(
      options,
      queryVector,
      options.queryText,
      options.topK ?? 10,
      options.format
    );
  }

  /**
   * Create an online, crash-consistent backup of the store database.
   */
  async backup(targetPath: string): Promise<void> {
    this.ensureOpen();
    const native = getNative();
    try {
      native.storeBackupSync(this._nativeHandle, targetPath);
    } catch (err) {
      throw VivvyError.fromNativeError(err);
    }
  }

  /**
   * Purge expired/deleted tombstones and rebuild accelerator index.
   */
  async vacuum(batchSize = 100): Promise<number> {
    this.ensureOpen();
    const native = getNative();
    try {
      return native.storeVacuumSync(this._nativeHandle, batchSize);
    } catch (err) {
      throw VivvyError.fromNativeError(err);
    }
  }

  // Helper marshaling routines
  private marshalRemember(opt: RememberOptions): Record<string, unknown> {
    const payload: Record<string, unknown> = {
      tenant_id: opt.tenantId,
      namespace: opt.namespace,
      content: opt.content,
      kind: opt.kind ?? 'fact',
      importance: opt.importance ?? 0.5,
      metadata: opt.metadata ?? {},
      source: opt.source ?? {},
    };
    if (opt.agentId) payload.agent_id = opt.agentId;
    if (opt.userId) payload.user_id = opt.userId;
    if (opt.expiresAtMs != null) payload.expires_at_ms = opt.expiresAtMs;
    if (opt.operationId) payload.operation_id = opt.operationId;
    return payload;
  }

  private marshalRecall(opt: RecallOptions): Record<string, unknown> {
    const scoring = opt.scoring
      ? {
          weight_similarity: opt.scoring.weightSimilarity,
          weight_importance: opt.scoring.weightImportance,
          weight_recency: opt.scoring.weightRecency,
          weight_reinforcement: opt.scoring.weightReinforcement,
          recency_half_life_days: opt.scoring.recencyHalfLifeDays,
        }
      : undefined;

    return {
      tenant_id: opt.tenantId,
      namespace: opt.namespace,
      agent_id: opt.agentId ?? null,
      user_id: opt.userId ?? null,
      kinds: opt.kinds,
      min_importance: opt.minImportance,
      metadata_eq: opt.metadataEq,
      created_after_ms: opt.createdAfterMs,
      created_before_ms: opt.createdBeforeMs,
      include_explanations: opt.includeExplanations ?? true,
      mmr_lambda: opt.mmrLambda,
      scoring,
    };
  }

  private parseRecallResponse(jsonStr: string): RecallResponse {
    const raw = JSON.parse(jsonStr);
    const items: RecallItem[] = (raw.items || []).map((item: any) => ({
      id: item.id,
      tenantId: item.tenant_id,
      namespace: item.namespace,
      agentId: item.agent_id,
      userId: item.user_id,
      kind: item.kind as MemoryKind,
      content: item.content,
      importance: item.importance,
      createdAtMs: item.created_at_ms,
      updatedAtMs: item.updated_at_ms,
      expiresAtMs: item.expires_at_ms,
      revision: item.revision,
      metadata: item.metadata || {},
      source: item.source || null,
      score: item.score,
      explanation: item.explanation
        ? {
            totalScore: item.explanation.total_score,
            similarityScore: item.explanation.similarity_score,
            importanceScore: item.explanation.importance_score,
            recencyScore: item.explanation.recency_score,
            reinforcementScore: item.explanation.reinforcement_score,
            policyNotes: item.explanation.policy_notes,
          }
        : null,
    }));

    return {
      totalCandidates: raw.total_candidates ?? items.length,
      items,
    };
  }
}
