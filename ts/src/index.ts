/**
 * Vivvy: Single-machine vector search and durable long-term memory for AI agents.
 */

export { MemoryStore } from './store.ts';
export { VectorIndex } from './vector-index.ts';
export { VivvyError, VivvyErrorCode } from './errors.ts';
export {
  DeterministicEmbedder,
  OpenAIEmbedder,
  OllamaEmbedder,
} from './embeddings.ts';
export type {
  Embedder,
  MemoryKind,
  MemoryScope,
  StoreConfig,
  StoreHealth,
  UpdateOptions,
  MemoryRecord,
  RecallExplanation,
  RecallItem,
  RecallResponse,
  ScoringWeights,
  RecallOptions,
  FormatContextOptions,
  RememberOptions,
  RecallTextOptions,
  FormatContextTextOptions,
  IndexConfig,
  IndexFilter,
  VectorSearchResult,
} from './types.ts';
