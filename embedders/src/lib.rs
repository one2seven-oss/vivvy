//! Provider-neutral embedding interfaces for Vivvy.
//!
//! This crate deliberately contains no network client or API key handling.
//! Language SDKs can implement [`Embedder`] using their native HTTP/client
//! ecosystem while applications share a small, stable contract.

use thiserror::Error;

/// Errors returned by an embedding provider implementation.
#[derive(Debug, Error)]
pub enum EmbedderError {
    #[error("embedding provider error: {0}")]
    Provider(String),
    #[error("embedding provider returned {actual} vectors for {expected} inputs")]
    BatchSizeMismatch { expected: usize, actual: usize },
    #[error("embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// A synchronous, provider-neutral text embedding interface.
///
/// Implementations should return vectors whose width always equals
/// [`Embedder::dimensions`]. Network clients are intentionally left to
/// language-specific packages rather than bundled into this core contract.
pub trait Embedder: Send + Sync {
    /// Stable provider/model identifier, such as `text-embedding-3-small`.
    fn name(&self) -> &str;

    /// Width of every vector returned by this embedder.
    fn dimensions(&self) -> usize;

    /// Embed one input string.
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError>;

    /// Embed multiple inputs. The default implementation preserves order and
    /// delegates to [`Embedder::embed`].
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        let vectors = texts
            .iter()
            .map(|text| self.embed(text))
            .collect::<Result<Vec<_>, _>>()?;
        if vectors.len() != texts.len() {
            return Err(EmbedderError::BatchSizeMismatch {
                expected: texts.len(),
                actual: vectors.len(),
            });
        }
        for vector in &vectors {
            if vector.len() != self.dimensions() {
                return Err(EmbedderError::DimensionMismatch {
                    expected: self.dimensions(),
                    actual: vector.len(),
                });
            }
        }
        Ok(vectors)
    }
}
