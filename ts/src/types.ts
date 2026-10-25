/**
 * Types and interfaces for the Vivvy long-term agent memory runtime.
 */

export type MemoryKind = 'fact' | 'preference' | 'instruction' | 'context' | 'episodic';

export interface StoreConfig {
  path: string;
  dimensions?: number;
  embeddingModel?: string;
  maxRecallLimit?: number;
  embedder?: import('./embeddings').Embedder;
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
