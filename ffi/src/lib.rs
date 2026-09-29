//! C-compatible FFI layer for Vivvy.
//!
//! Built as both a `staticlib` and a `cdylib` so downstream hosts (the Go bindings in `go/`,
//! or any other C-ABI consumer) can link against Vivvy without a Rust toolchain. The public
//! surface is entirely `extern "C"` functions in [`store`]; see `go/include/vivvy.h` for the
//! canonical C99 declarations and `error` for the status-code/last-error-message convention
//! every function follows.
//!
//! `vivvy-core` is a declared workspace dependency (per the FFI crate's design brief) but is
//! not referenced directly yet: every operation exposed today goes through `vivvy-memory`'s
//! `MemoryStore`. It stays a direct dependency so a future low-level `Index` FFI — mirroring
//! the Python bindings' `vivvy.Index` — can be added without a `Cargo.toml` change.

pub mod error;
pub mod store;
mod types;

pub use store::CVivvyStore;
