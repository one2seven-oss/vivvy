import { VivvyError, VivvyErrorCode } from './errors.ts';
import { getNative } from './store.ts';
import type { IndexConfig, IndexFilter, VectorSearchResult } from './types.ts';

/** Low-level HNSW vector index, independent from MemoryStore. */
export class VectorIndex {
  private handle: any;
  private closed = false;

  private constructor(handle: any) { this.handle = handle; }

  static open(config: IndexConfig): VectorIndex {
    if (!Number.isInteger(config.dimensions) || config.dimensions <= 0) {
      throw new VivvyError(VivvyErrorCode.InvalidInput, 'dimensions must be a positive integer');
    }
    try {
      const handle = getNative().indexOpen(JSON.stringify({
        dimensions: config.dimensions,
        metric: config.metric ?? 'l2',
        wal_path: config.walPath,
        data_dir: config.dataDir,
      }));
      return new VectorIndex(handle);
    } catch (err) { throw VivvyError.fromNativeError(err); }
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    getNative().indexClose(this.handle);
    this.handle = null;
  }

  insert(vector: Float32Array, metadata?: Record<string, string>): bigint {
    this.ensureOpen();
    try { return getNative().indexInsert(this.handle, vector, metadata ? JSON.stringify(metadata) : null); }
    catch (err) { throw VivvyError.fromNativeError(err); }
  }

  insertWithId(id: bigint, vector: Float32Array): void {
    this.ensureOpen();
    try { getNative().indexInsertWithId(this.handle, id, vector); }
    catch (err) { throw VivvyError.fromNativeError(err); }
  }

  insertBatch(vectors: Float32Array[]): bigint[] {
    this.ensureOpen();
    if (vectors.length === 0) return [];
    const dim = vectors[0].length;
    if (dim === 0 || vectors.some((vector) => vector.length !== dim)) {
      throw new VivvyError(VivvyErrorCode.InvalidInput, 'all vectors must have the same non-zero dimension');
    }
    const flat = new Float32Array(vectors.length * dim);
    vectors.forEach((vector, i) => flat.set(vector, i * dim));
    try { return JSON.parse(getNative().indexInsertBatch(this.handle, flat, dim, vectors.length)).map(BigInt); }
    catch (err) { throw VivvyError.fromNativeError(err); }
  }

  search(query: Float32Array, k: number, filter?: IndexFilter): VectorSearchResult[] {
    this.ensureOpen();
    if (!Number.isInteger(k) || k <= 0) throw new VivvyError(VivvyErrorCode.InvalidInput, 'k must be positive');
    try {
      return JSON.parse(getNative().indexSearch(this.handle, query, k, filter ? JSON.stringify(filter) : null));
    } catch (err) { throw VivvyError.fromNativeError(err); }
  }

  delete(id: bigint): boolean {
    this.ensureOpen();
    try { return getNative().indexDelete(this.handle, id); }
    catch (err) { throw VivvyError.fromNativeError(err); }
  }

  compact(): void {
    this.ensureOpen();
    try { getNative().indexCompact(this.handle); }
    catch (err) { throw VivvyError.fromNativeError(err); }
  }

  private ensureOpen(): void {
    if (this.closed || !this.handle) throw new VivvyError(VivvyErrorCode.StoreIOError, 'Index is closed');
  }
}
