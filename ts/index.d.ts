/**
 * TypeScript definitions for vivvy.
 */

export type MemoryKind = 'fact' | 'preference' | 'instruction' | 'context' | 'episodic';

export const VivvyErrorCode: {
  readonly NullPointer: -1;
  readonly InvalidUtf8: -2;
  readonly InvalidJson: -3;
  readonly Panic: -4;
  readonly IO: -5;
  readonly InvalidScope: 1;
  readonly DimensionMismatch: 2;
  readonly EmbeddingModelMismatch: 3;
  readonly NotFound: 4;
  readonly RevisionConflict: 5;
  readonly StoreBusy: 6;
  readonly RecoveryRequired: 7;
  readonly CorruptStore: 8;
  readonly EncryptionKeyUnavailable: 9;
  readonly PolicyDenied: 10;
  readonly InvalidFilter: 11;
  readonly InvalidInput: 12;
  readonly StoreIOError: 13;
  readonly DatabaseError: 14;
  readonly EncryptionFailed: 15;
  readonly DecryptionFailed: 16;
};

export type VivvyErrorCode = (typeof VivvyErrorCode)[keyof typeof VivvyErrorCode];

export class VivvyError extends Error {
  public readonly code: VivvyErrorCode;
  constructor(code: number | VivvyErrorCode, message: string);
  static fromNativeError(err: unknown): VivvyError;
}

export interface Embedder {
  readonly name: string;
  readonly dimensions: number;
  embed(text: string): Promise<Float32Array>;
  embedBatch(texts: string[]): Promise<Float32Array[]>;
}

export class DeterministicEmbedder implements Embedder {
  readonly name: string;
  readonly dimensions: number;
  constructor(name?: string, dimensions?: number);
  embed(text: string): Promise<Float32Array>;
  embedBatch(texts: string[]): Promise<Float32Array[]>;
}

export class OpenAIEmbedder implements Embedder {
  readonly name: string;
  readonly dimensions: number;
  constructor(options?: {
    model?: string;
    dimensions?: number;
    apiKey?: string;
    baseUrl?: string;
  });
  embed(text: string): Promise<Float32Array>;
  embedBatch(texts: string[]): Promise<Float32Array[]>;
}

export class OllamaEmbedder implements Embedder {
  readonly name: string;
  readonly dimensions: number;
  constructor(options?: {
    model?: string;
    dimensions?: number;
    baseUrl?: string;
  });
  embed(text: string): Promise<Float32Array>;
  embedBatch(texts: string[]): Promise<Float32Array[]>;
}

export interface StoreConfig {
  path: string;
  dimensions?: number;
  embeddingModel?: string;
  maxRecallLimit?: number;
  embedder?: Embedder;
}

export interface MemoryRecord {
  id: string;
  tenantId: string;
  namespace: string;
  agentId?: string | null;
  userId?: string | null;
  kind: MemoryKind;
  content: string;
  importance: number;
  createdAtMs: number;
  updatedAtMs: number;
  expiresAtMs?: number | null;
  revision: number;
  metadata: Record<string, unknown>;
  source?: Record<string, unknown> | null;
}

export interface MemoryScope {
  tenantId: string;
  namespace: string;
  agentId?: string | null;
  userId?: string | null;
}

export interface UpdateOptions extends MemoryScope {
  id: string;
  expectedRevision: number;
  operationId?: string | null;
  content?: string;
  kind?: MemoryKind;
  importance?: number;
  expiresAtMs?: number | null;
  metadataPatch?: Record<string, unknown>;
}

export interface StoreHealth {
  isHealthy: boolean;
  totalActiveRecords: number;
  totalTombstonedRecords: number;
  pendingOperationsCount: number;
  indexRebuildRequired: boolean;
  dbSizeBytes: number;
  walSizeBytes: number;
}

export interface IndexConfig {
  dimensions: number;
  metric?: 'l2' | 'cosine' | 'dot';
  walPath?: string;
  dataDir?: string;
}

export interface IndexFilter {
  eq?: { field: string; value: string };
  in?: { field: string; values: string[] };
  and?: IndexFilter[];
  or?: IndexFilter[];
  not?: IndexFilter;
}

export interface VectorSearchResult {
  id: number;
  distance: number;
}

export interface RecallExplanation {
  totalScore: number;
  similarityScore: number;
  importanceScore: number;
  recencyScore: number;
  reinforcementScore: number;
  policyNotes?: string[];
}

export interface RecallItem extends MemoryRecord {
  score: number;
  explanation?: RecallExplanation | null;
}

export interface RecallResponse {
  totalCandidates: number;
  items: RecallItem[];
}

export interface ScoringWeights {
  weightSimilarity?: number;
  weightImportance?: number;
  weightRecency?: number;
  weightReinforcement?: number;
  recencyHalfLifeDays?: number;
}

export interface RecallOptions {
  tenantId: string;
  namespace: string;
  agentId?: string | null;
  userId?: string | null;
  kinds?: MemoryKind[];
  minImportance?: number;
  metadataEq?: Record<string, unknown>;
  createdAfterMs?: number;
  createdBeforeMs?: number;
  includeExplanations?: boolean;
  mmrLambda?: number;
  scoring?: ScoringWeights;
}

export interface FormatContextOptions {
  maxTokens?: number;
  template?: string;
  header?: string;
  footer?: string;
}

export interface RememberOptions {
  tenantId: string;
  namespace: string;
  agentId?: string | null;
  userId?: string | null;
  content: string;
  kind?: MemoryKind;
  importance?: number;
  expiresAtMs?: number | null;
  operationId?: string | null;
  metadata?: Record<string, unknown>;
  source?: Record<string, unknown>;
}

export interface RecallTextOptions extends RecallOptions {
  queryText: string;
  topK?: number;
}

export interface FormatContextTextOptions extends RecallOptions {
  queryText: string;
  topK?: number;
  format?: FormatContextOptions;
}

export class MemoryStore {
  readonly path: string;
  readonly dimensions: number;
  readonly embeddingModel: string;
  embedder?: Embedder;

  static open(config: StoreConfig): Promise<MemoryStore>;
  static openSync(config: StoreConfig): MemoryStore;

  close(): void;

  remember(options: RememberOptions, vector: Float32Array): Promise<string>;
  rememberSync(options: RememberOptions, vector: Float32Array): string;
  rememberBatch(options: RememberOptions[], vectors: Float32Array[]): Promise<string[]>;
  rememberText(options: RememberOptions): Promise<string>;

  get(scope: MemoryScope, id: string): MemoryRecord | null;
  update(options: UpdateOptions, vector?: Float32Array): void;
  forget(scope: MemoryScope, id: string): void;
  forgetBatch(scope: MemoryScope, ids: string[]): void;
  health(): StoreHealth;
  rebuildIndex(): void;

  recall(
    options: RecallOptions,
    queryVector: Float32Array,
    queryText?: string | null,
    topK?: number
  ): Promise<RecallResponse>;
  recallSync(
    options: RecallOptions,
    queryVector: Float32Array,
    queryText?: string | null,
    topK?: number
  ): RecallResponse;
  recallText(options: RecallTextOptions): Promise<RecallResponse>;

  formatContext(
    options: RecallOptions,
    queryVector: Float32Array,
    queryText?: string | null,
    topK?: number,
    format?: FormatContextOptions
  ): Promise<string>;
  formatContextSync(
    options: RecallOptions,
    queryVector: Float32Array,
    queryText?: string | null,
    topK?: number,
    format?: FormatContextOptions
  ): string;
  formatContextText(options: FormatContextTextOptions): Promise<string>;

  backup(targetPath: string): Promise<void>;
  vacuum(batchSize?: number): Promise<number>;
}

export class VectorIndex {
  static open(config: IndexConfig): VectorIndex;
  close(): void;
  insert(vector: Float32Array, metadata?: Record<string, string>): bigint;
  insertWithId(id: bigint, vector: Float32Array): void;
  insertBatch(vectors: Float32Array[]): bigint[];
  search(query: Float32Array, k: number, filter?: IndexFilter): VectorSearchResult[];
  delete(id: bigint): boolean;
  compact(): void;
}
