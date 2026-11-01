//! C-compatible FFI layer for Vivvy.
//!
//! Built as both a `staticlib` and a `cdylib` so downstream hosts (the Go bindings in `go/`,
//! or any other C-ABI consumer) can link against Vivvy without a Rust toolchain. The public
//! surface is entirely `extern "C"` functions in [`store`] and [`index`]; see
//! `go/include/vivvy.h` for the canonical C99 declarations and `error` for the
//! status-code/last-error-message convention every function follows.
//!
//! The memory API is backed by `vivvy-memory::MemoryStore`; the low-level
//! vector API is backed by `vivvy-core::concurrent::VivvyIndex`.

pub mod error;
pub mod index;
pub mod store;
mod types;

pub use index::CVivvyIndex;
pub use store::CVivvyStore;
