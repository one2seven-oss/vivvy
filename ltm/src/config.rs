use crate::cipher::LexicalMode;
use crate::crypto::KeyProvider;
use crate::error::{MemoryError, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Configuration for opening or creating a `MemoryStore`.
#[derive(Clone)]
pub struct MemoryConfig {
    pub(crate) path: PathBuf,
    pub(crate) dimensions: usize,
    pub(crate) embedding_model: String,
    pub(crate) max_recall_limit: usize,
    pub(crate) key_provider: Option<Arc<dyn KeyProvider>>,
    pub(crate) lexical_mode: LexicalMode,
}

impl std::fmt::Debug for MemoryConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryConfig")
            .field("path", &self.path)
            .field("dimensions", &self.dimensions)
            .field("embedding_model", &self.embedding_model)
            .field("max_recall_limit", &self.max_recall_limit)
            .field("encrypted", &self.key_provider.is_some())
            .field("lexical_mode", &self.lexical_mode)
            .finish()
    }
}

impl MemoryConfig {
    #[must_use]
    pub fn builder(path: impl AsRef<Path>) -> MemoryConfigBuilder {
        MemoryConfigBuilder::new(path)
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    #[must_use]
    pub fn embedding_model(&self) -> &str {
        &self.embedding_model
    }

    #[must_use]
    pub fn max_recall_limit(&self) -> usize {
        self.max_recall_limit
    }

    /// The configured per-tenant key provider, if encryption-at-rest is enabled.
    #[must_use]
    pub fn key_provider(&self) -> Option<&Arc<dyn KeyProvider>> {
        self.key_provider.as_ref()
    }

    /// Whether this store encrypts sensitive fields at rest.
    #[must_use]
    pub fn is_encrypted(&self) -> bool {
        self.key_provider.is_some()
    }

    /// The lexical (keyword) recall strategy for this store.
    #[must_use]
    pub fn lexical_mode(&self) -> LexicalMode {
        self.lexical_mode
    }
}

pub struct MemoryConfigBuilder {
    path: PathBuf,
    dimensions: Option<usize>,
    embedding_model: Option<String>,
    max_recall_limit: usize,
    key_provider: Option<Arc<dyn KeyProvider>>,
    lexical_mode: Option<LexicalMode>,
}

impl MemoryConfigBuilder {
    #[must_use]
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            dimensions: None,
            embedding_model: None,
            max_recall_limit: 100,
            key_provider: None,
            lexical_mode: None,
        }
    }

    /// Enable encryption-at-rest for sensitive fields, sealing them per-tenant
    /// with keys from `provider`. When set, lexical recall defaults to
    /// [`LexicalMode::Disabled`] unless overridden via [`lexical_mode`].
    #[must_use]
    pub fn key_provider(mut self, provider: Arc<dyn KeyProvider>) -> Self {
        self.key_provider = Some(provider);
        self
    }

    /// Override the lexical (keyword) recall strategy. Validated in `build`
    /// against whether encryption is enabled.
    #[must_use]
    pub fn lexical_mode(mut self, mode: LexicalMode) -> Self {
        self.lexical_mode = Some(mode);
        self
    }

    #[must_use]
    pub fn dimensions(mut self, dims: usize) -> Self {
        self.dimensions = Some(dims);
        self
    }

    #[must_use]
    pub fn embedding_model(mut self, model: impl Into<String>) -> Self {
        self.embedding_model = Some(model.into());
        self
    }

    #[must_use]
    pub fn max_recall_limit(mut self, limit: usize) -> Self {
        self.max_recall_limit = limit;
        self
    }

    pub fn build(self) -> Result<MemoryConfig> {
        let dimensions = self.dimensions.ok_or_else(|| {
            MemoryError::invalid_input("MemoryConfig: embedding dimensions must be specified")
        })?;

        if dimensions == 0 {
            return Err(MemoryError::invalid_input(
                "MemoryConfig: dimensions must be greater than 0",
            ));
        }

        let embedding_model = self.embedding_model.ok_or_else(|| {
            MemoryError::invalid_input("MemoryConfig: embedding_model must be specified")
        })?;

        if embedding_model.trim().is_empty() {
            return Err(MemoryError::invalid_input(
                "MemoryConfig: embedding_model cannot be empty",
            ));
        }

        // Resolve lexical mode and validate it against the encryption setting.
        let encrypted = self.key_provider.is_some();
        let lexical_mode = match self.lexical_mode {
            Some(mode) => mode,
            None if encrypted => LexicalMode::Disabled,
            None => LexicalMode::Plaintext,
        };

        match (encrypted, lexical_mode) {
            (true, LexicalMode::Plaintext) => {
                return Err(MemoryError::invalid_input(
                    "MemoryConfig: LexicalMode::Plaintext would store plaintext in the FTS index \
                     and is not allowed for an encrypted store; use Disabled or BlindIndex",
                ));
            }
            (false, LexicalMode::BlindIndex) => {
                return Err(MemoryError::invalid_input(
                    "MemoryConfig: LexicalMode::BlindIndex requires a key_provider",
                ));
            }
            _ => {}
        }

        Ok(MemoryConfig {
            path: self.path,
            dimensions,
            embedding_model,
            max_recall_limit: self.max_recall_limit,
            key_provider: self.key_provider,
            lexical_mode,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_config() {
        let config = MemoryConfig::builder("./test_mem")
            .dimensions(1536)
            .embedding_model("text-embedding-3-small")
            .build()
            .unwrap();

        assert_eq!(config.dimensions(), 1536);
        assert_eq!(config.embedding_model(), "text-embedding-3-small");
        assert_eq!(config.max_recall_limit(), 100);
    }

    #[test]
    fn test_missing_fields_fail() {
        assert!(MemoryConfig::builder("./test_mem").build().is_err());
        assert!(MemoryConfig::builder("./test_mem")
            .dimensions(1536)
            .build()
            .is_err());
        assert!(MemoryConfig::builder("./test_mem")
            .embedding_model("test")
            .build()
            .is_err());
        assert!(MemoryConfig::builder("./test_mem")
            .dimensions(0)
            .embedding_model("test")
            .build()
            .is_err());
    }
}
