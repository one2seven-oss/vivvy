/**
 * Stable error codes mirrored 1:1 from vivvy_memory::ErrorCode and vivvy.h.
 */
export const VivvyErrorCode = {
  // FFI-boundary failure codes
  NullPointer: -1,
  InvalidUtf8: -2,
  InvalidJson: -3,
  Panic: -4,
  IO: -5,

  // Domain failure codes
  InvalidScope: 1,
  DimensionMismatch: 2,
  EmbeddingModelMismatch: 3,
  NotFound: 4,
  RevisionConflict: 5,
  StoreBusy: 6,
  RecoveryRequired: 7,
  CorruptStore: 8,
  EncryptionKeyUnavailable: 9,
  PolicyDenied: 10,
  InvalidFilter: 11,
  InvalidInput: 12,
  StoreIOError: 13,
  DatabaseError: 14,
  EncryptionFailed: 15,
  DecryptionFailed: 16,
} as const;

export type VivvyErrorCode = (typeof VivvyErrorCode)[keyof typeof VivvyErrorCode];

/**
 * Standard typed error thrown by Vivvy operations.
 */
export class VivvyError extends Error {
  public readonly code: VivvyErrorCode;

  constructor(code: number | VivvyErrorCode, message: string) {
    super(`vivvy: [${code}] ${message}`);
    this.name = 'VivvyError';
    this.code = code as VivvyErrorCode;
    // Maintain proper prototype chain
    Object.setPrototypeOf(this, VivvyError.prototype);
  }

  static fromNativeError(err: unknown): VivvyError {
    if (err instanceof VivvyError) {
      return err;
    }
    if (err && typeof err === 'object') {
      const code = 'code' in err && typeof (err as { code: unknown }).code === 'number'
        ? (err as { code: number }).code
        : VivvyErrorCode.DatabaseError;
      const message = 'message' in err && typeof (err as { message: unknown }).message === 'string'
        ? (err as { message: string }).message
        : String(err);
      return new VivvyError(code, message.replace(/^vivvy:\s*\[\d+\]\s*/, ''));
    }
    return new VivvyError(VivvyErrorCode.DatabaseError, String(err));
  }
}
