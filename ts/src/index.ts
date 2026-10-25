/**
 * Vivvy: Single-machine vector search and durable long-term memory for AI agents.
 */

export { MemoryStore } from './store.ts';
export { VivvyError, VivvyErrorCode } from './errors.ts';
export {
  DeterministicEmbedder,
  OpenAIEmbedder,
  OllamaEmbedder,
} from './embeddings.ts';
export type {
  Embedder,
  MemoryKind,
  StoreConfig,
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
} from './types.ts';
