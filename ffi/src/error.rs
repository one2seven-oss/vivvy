//! Error codes and the thread-local last-error message returned across the FFI boundary.
//!
//! Convention: every `extern "C"` function returns a `c_int`. `0` is success. Negative
//! codes (`ERR_*`) are FFI-boundary failures (bad pointers, invalid UTF-8/JSON, a caught
//! panic). Positive codes mirror `vivvy_memory::ErrorCode` one-to-one so Go callers can
//! branch on specific store failures (e.g. a revision conflict) without string matching.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;

use vivvy_memory::ErrorCode;

pub const ERR_NULL_POINTER: i32 = -1;
pub const ERR_INVALID_UTF8: i32 = -2;
pub const ERR_INVALID_JSON: i32 = -3;
pub const ERR_PANIC: i32 = -4;
pub const ERR_IO: i32 = -5;

pub const ERR_INVALID_SCOPE: i32 = 1;
pub const ERR_DIMENSION_MISMATCH: i32 = 2;
pub const ERR_EMBEDDING_MODEL_MISMATCH: i32 = 3;
pub const ERR_NOT_FOUND: i32 = 4;
pub const ERR_REVISION_CONFLICT: i32 = 5;
pub const ERR_STORE_BUSY: i32 = 6;
pub const ERR_RECOVERY_REQUIRED: i32 = 7;
pub const ERR_CORRUPT_STORE: i32 = 8;
pub const ERR_ENCRYPTION_KEY_UNAVAILABLE: i32 = 9;
pub const ERR_POLICY_DENIED: i32 = 10;
pub const ERR_INVALID_FILTER: i32 = 11;
pub const ERR_INVALID_INPUT: i32 = 12;
pub const ERR_STORE_IO_ERROR: i32 = 13;
pub const ERR_DATABASE_ERROR: i32 = 14;

/// Every failure mode an FFI function body can produce, from bad C arguments
/// through to a domain error surfaced by `vivvy-memory`.
#[derive(Debug)]
pub enum FfiError {
    NullPointer(&'static str),
    InvalidUtf8(&'static str),
    InvalidJson {
        field: &'static str,
        source: serde_json::Error,
    },
    Memory(vivvy_memory::MemoryError),
    Io(std::io::Error),
}

impl FfiError {
    pub fn code(&self) -> i32 {
        match self {
            FfiError::NullPointer(_) => ERR_NULL_POINTER,
            FfiError::InvalidUtf8(_) => ERR_INVALID_UTF8,
            FfiError::InvalidJson { .. } => ERR_INVALID_JSON,
            FfiError::Io(_) => ERR_IO,
            FfiError::Memory(e) => match e.code() {
                ErrorCode::InvalidScope => ERR_INVALID_SCOPE,
                ErrorCode::DimensionMismatch => ERR_DIMENSION_MISMATCH,
                ErrorCode::EmbeddingModelMismatch => ERR_EMBEDDING_MODEL_MISMATCH,
                ErrorCode::NotFound => ERR_NOT_FOUND,
                ErrorCode::RevisionConflict => ERR_REVISION_CONFLICT,
                ErrorCode::StoreBusy => ERR_STORE_BUSY,
                ErrorCode::RecoveryRequired => ERR_RECOVERY_REQUIRED,
                ErrorCode::CorruptStore => ERR_CORRUPT_STORE,
                ErrorCode::EncryptionKeyUnavailable => ERR_ENCRYPTION_KEY_UNAVAILABLE,
                ErrorCode::PolicyDenied => ERR_POLICY_DENIED,
                ErrorCode::InvalidFilter => ERR_INVALID_FILTER,
                ErrorCode::InvalidInput => ERR_INVALID_INPUT,
                ErrorCode::IoError => ERR_STORE_IO_ERROR,
                ErrorCode::DatabaseError => ERR_DATABASE_ERROR,
            },
        }
    }

    pub fn message(&self) -> String {
        match self {
            FfiError::NullPointer(field) => {
                format!("null pointer passed for required argument '{field}'")
            }
            FfiError::InvalidUtf8(field) => format!("argument '{field}' is not valid UTF-8"),
            FfiError::InvalidJson { field, source } => {
                format!("invalid JSON in '{field}': {source}")
            }
            FfiError::Io(e) => format!("I/O error: {e}"),
            FfiError::Memory(e) => e.to_string(),
        }
    }
}

impl From<vivvy_memory::MemoryError> for FfiError {
    fn from(e: vivvy_memory::MemoryError) -> Self {
        FfiError::Memory(e)
    }
}

impl From<std::io::Error> for FfiError {
    fn from(e: std::io::Error) -> Self {
        FfiError::Io(e)
    }
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Records `msg` as the calling thread's last error. Overwritten by every
/// subsequent FFI call on the same thread, mirroring `errno` semantics.
pub fn set_last_error(msg: impl Into<String>) {
    let sanitized = msg.into().replace('\0', "");
    let c_string = CString::new(sanitized)
        .unwrap_or_else(|_| CString::new("<error message unavailable>").unwrap());
    LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(c_string));
}

/// Returns the calling thread's last recorded error message, or an empty
/// string if none has been set. The returned pointer is borrowed: it is
/// valid until the next `vivvy_*` call made on this thread and must never
/// be freed by the caller.
pub fn last_error_ptr() -> *const c_char {
    thread_local! {
        static EMPTY: CString = CString::new("").unwrap();
    }
    LAST_ERROR.with(|cell| match &*cell.borrow() {
        Some(c) => c.as_ptr(),
        None => EMPTY.with(|e| e.as_ptr()),
    })
}

/// Extracts a human-readable message from a caught panic payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// Runs `f` behind `catch_unwind`, translating any error or panic into the
/// last-error thread-local and returning the stable numeric code. Every
/// `extern "C"` function in this crate funnels its fallible body through here
/// so a Rust panic can never unwind across the FFI boundary (which is
/// undefined behavior).
pub fn run_guarded<T>(
    f: impl FnOnce() -> Result<T, FfiError> + std::panic::UnwindSafe,
) -> Result<T, i32> {
    match std::panic::catch_unwind(f) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => {
            let code = e.code();
            set_last_error(e.message());
            Err(code)
        }
        Err(payload) => {
            set_last_error(format!("internal panic: {}", panic_message(&payload)));
            Err(ERR_PANIC)
        }
    }
}
