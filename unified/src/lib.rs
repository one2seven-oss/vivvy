//! Unified Vivvy Rust SDK.
//!
//! Use this crate for the batteries-included experience. Advanced consumers
//! can instead depend directly on `vivvy-core`, `vivvy-memory`, or
//! `vivvy-embedder` to keep their dependency surface minimal.

#[cfg(feature = "core")]
pub use vivvy_core as core;
#[cfg(feature = "embedder")]
pub use vivvy_embedder as embedder;
#[cfg(feature = "memory")]
pub use vivvy_memory as memory;

/// Convenient imports for applications using the unified SDK.
pub mod prelude {
    #[cfg(feature = "core")]
    pub use vivvy_core::{concurrent::VivvyIndex, distance::Metric};
    #[cfg(feature = "embedder")]
    pub use vivvy_embedder::{Embedder, EmbedderError};
    #[cfg(feature = "memory")]
    pub use vivvy_memory::{MemoryConfig, MemoryStore};
}
