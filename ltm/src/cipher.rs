//! Per-tenant, field-level encryption-at-rest for the memory store.
//!
//! Sensitive columns (`content`, `embedding`, `metadata_json`, `source_json`,
//! and the operation-journal `payload_json`) are sealed with an AEAD cipher
//! keyed by the tenant's key, as supplied by a [`KeyProvider`]. Non-sensitive
//! columns used as query keys or indexes (ids, scope, timestamps, kind,
//! status, importance, revision, embedding model/dims) are left in plaintext
//! so SQL filtering and indexing keep working.
//!
//! Scheme: XChaCha20-Poly1305 with a random 24-byte nonce per field. Each
//! ciphertext is bound to its row and column via Additional Authenticated Data
//! (AAD = `tenant_id || record_id || field_name`), so a sealed value cannot be
//! moved to a different row, column, or tenant without detection.

use crate::crypto::KeyProvider;
use crate::error::{MemoryError, Result};
use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    Key, XChaCha20Poly1305, XNonce,
};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::sync::Arc;
use zeroize::Zeroizing;

type HmacSha256 = Hmac<Sha256>;

/// Canonical name of the on-disk encryption scheme, recorded in `store_meta`.
pub const SCHEME_XCHACHA20POLY1305: &str = "xchacha20poly1305";

const ENVELOPE_VERSION: u8 = 1;
const KEY_VERSION: u8 = 1;
const NONCE_LEN: usize = 24;
const HEADER_LEN: usize = 2 + NONCE_LEN; // version + key_version + nonce
const AAD_SEP: u8 = 0x1f;

/// How lexical (keyword) recall behaves when the store is encrypted.
///
/// Encrypting `content` means the plaintext FTS5 index would otherwise leak
/// the very data being protected, so the caller must choose a strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexicalMode {
    /// Plaintext FTS5 index. Only valid when the store is NOT encrypted; this
    /// is the historical behavior for unencrypted stores.
    Plaintext,
    /// No lexical index is maintained; recall falls back to pure vector search.
    /// Default for encrypted stores — leak-free and simple.
    Disabled,
    /// Keyed blind index: each content token is stored as `HMAC(tenant_key,
    /// token)`, enabling exact keyword match with no plaintext on disk. No BM25
    /// ranking, phrase, or prefix matching, and token-frequency is observable
    /// to anyone with the database file. Opt-in.
    BlindIndex,
}

/// Field names used as part of the AEAD AAD. Keep these stable — changing a
/// name makes previously sealed values undecryptable.
pub mod field {
    pub const CONTENT: &str = "content";
    pub const EMBEDDING: &str = "embedding";
    pub const METADATA: &str = "metadata_json";
    pub const SOURCE: &str = "source_json";
    pub const OP_PAYLOAD: &str = "op_payload";
}

/// Field-level cipher bound to a [`KeyProvider`] and a [`LexicalMode`].
#[derive(Clone)]
pub struct FieldCipher {
    provider: Arc<dyn KeyProvider>,
    lexical_mode: LexicalMode,
}

impl FieldCipher {
    pub fn new(provider: Arc<dyn KeyProvider>, lexical_mode: LexicalMode) -> Self {
        Self {
            provider,
            lexical_mode,
        }
    }

    pub fn lexical_mode(&self) -> LexicalMode {
        self.lexical_mode
    }

    fn cipher_for(&self, tenant_id: &str) -> Result<(XChaCha20Poly1305, Zeroizing<Vec<u8>>)> {
        let key_bytes = Zeroizing::new(self.provider.get_key(tenant_id)?);
        if key_bytes.len() != 32 {
            return Err(MemoryError::encryption_key_unavailable(format!(
                "tenant '{tenant_id}' key must be 32 bytes, got {}",
                key_bytes.len()
            )));
        }
        let cipher = XChaCha20Poly1305::new(Key::from_slice(key_bytes.as_slice()));
        Ok((cipher, key_bytes))
    }

    /// Whether `bytes` are a sealed envelope produced by [`seal`](Self::seal)
    /// rather than legacy plaintext. JSON payloads never start with the
    /// envelope version byte, so this is unambiguous for journal payloads.
    pub fn is_sealed(bytes: &[u8]) -> bool {
        bytes.len() > HEADER_LEN && bytes[0] == ENVELOPE_VERSION
    }

    fn aad(tenant_id: &str, record_id: &str, field_name: &str) -> Vec<u8> {
        let mut aad = Vec::with_capacity(tenant_id.len() + record_id.len() + field_name.len() + 2);
        aad.extend_from_slice(tenant_id.as_bytes());
        aad.push(AAD_SEP);
        aad.extend_from_slice(record_id.as_bytes());
        aad.push(AAD_SEP);
        aad.extend_from_slice(field_name.as_bytes());
        aad
    }

    /// Seal `plaintext` for `(tenant_id, record_id, field_name)`, returning the
    /// self-describing envelope bytes.
    pub fn seal(
        &self,
        tenant_id: &str,
        record_id: &str,
        field_name: &str,
        plaintext: &[u8],
    ) -> Result<Vec<u8>> {
        let (cipher, _key) = self.cipher_for(tenant_id)?;

        let mut nonce_bytes = [0u8; NONCE_LEN];
        getrandom::getrandom(&mut nonce_bytes).map_err(|e| {
            MemoryError::encryption_failed(format!("failed to generate nonce: {e}"))
        })?;
        let nonce = XNonce::from_slice(&nonce_bytes);

        let aad = Self::aad(tenant_id, record_id, field_name);
        let ciphertext = cipher
            .encrypt(
                nonce,
                Payload {
                    msg: plaintext,
                    aad: &aad,
                },
            )
            .map_err(|_| MemoryError::encryption_failed("AEAD seal failed"))?;

        let mut out = Vec::with_capacity(HEADER_LEN + ciphertext.len());
        out.push(ENVELOPE_VERSION);
        out.push(KEY_VERSION);
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Open an envelope produced by [`seal`](Self::seal) for the same
    /// `(tenant_id, record_id, field_name)`. Fails if the key is wrong, the
    /// bytes are tampered, or the AAD context does not match.
    pub fn open(
        &self,
        tenant_id: &str,
        record_id: &str,
        field_name: &str,
        envelope: &[u8],
    ) -> Result<Vec<u8>> {
        if envelope.len() < HEADER_LEN {
            return Err(MemoryError::decryption_failed("ciphertext envelope too short"));
        }
        let version = envelope[0];
        if version != ENVELOPE_VERSION {
            return Err(MemoryError::decryption_failed(format!(
                "unsupported envelope version {version}"
            )));
        }
        // envelope[1] is the key_version, reserved for key rotation.
        let nonce = XNonce::from_slice(&envelope[2..HEADER_LEN]);
        let ciphertext = &envelope[HEADER_LEN..];

        let (cipher, _key) = self.cipher_for(tenant_id)?;
        let aad = Self::aad(tenant_id, record_id, field_name);
        cipher
            .decrypt(
                nonce,
                Payload {
                    msg: ciphertext,
                    aad: &aad,
                },
            )
            .map_err(|_| {
                MemoryError::decryption_failed(
                    "AEAD open failed (wrong key, tampered data, or context mismatch)",
                )
            })
    }

    /// Build the blind-index token string for `content`: a space-separated
    /// list of `HMAC(tenant_key, token)` hex digests, suitable for storage in
    /// the FTS5 `content` column under [`LexicalMode::BlindIndex`].
    pub fn blind_index_tokens(&self, tenant_id: &str, content: &str) -> Result<String> {
        let key = Zeroizing::new(self.provider.get_key(tenant_id)?);
        let tokens = tokenize(content);
        let mut out = String::new();
        for (i, tok) in tokens.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(&hmac_token(key.as_slice(), tok)?);
        }
        Ok(out)
    }

    /// Build an FTS5 MATCH expression for a blind-index query: the query text's
    /// tokens are hashed with the same keyed HMAC and OR-combined.
    pub fn blind_index_query(&self, tenant_id: &str, query_text: &str) -> Result<String> {
        let key = Zeroizing::new(self.provider.get_key(tenant_id)?);
        let tokens = tokenize(query_text);
        if tokens.is_empty() {
            return Ok(String::new());
        }
        let mut terms = Vec::with_capacity(tokens.len());
        for tok in &tokens {
            terms.push(format!("\"{}\"", hmac_token(key.as_slice(), tok)?));
        }
        Ok(terms.join(" OR "))
    }
}

/// Lowercase, alphanumeric/`_`/`-` word tokenizer, matching the sanitizer used
/// for plaintext FTS so blind-index and plaintext behave consistently.
fn tokenize(input: &str) -> Vec<String> {
    input
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

fn hmac_token(key: &[u8], token: &str) -> Result<String> {
    let mut mac = <HmacSha256 as Mac>::new_from_slice(key)
        .map_err(|e| MemoryError::encryption_failed(format!("invalid HMAC key: {e}")))?;
    mac.update(token.as_bytes());
    let digest = mac.finalize().into_bytes();
    // 16 bytes / 32 hex chars is ample to avoid collisions for a token index.
    let mut hex = String::with_capacity(32);
    for b in &digest[..16] {
        hex.push_str(&format!("{b:02x}"));
    }
    Ok(hex)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::NoOpDevKeyProvider;

    fn cipher() -> FieldCipher {
        FieldCipher::new(Arc::new(NoOpDevKeyProvider::new()), LexicalMode::Disabled)
    }

    #[test]
    fn round_trip_restores_plaintext() {
        let c = cipher();
        let pt = b"user prefers concise technical answers";
        let env = c.seal("acme", "mem-1", field::CONTENT, pt).unwrap();
        assert_ne!(&env[HEADER_LEN..], pt, "ciphertext must differ from plaintext");
        let out = c.open("acme", "mem-1", field::CONTENT, &env).unwrap();
        assert_eq!(out, pt);
    }

    #[test]
    fn wrong_aad_context_fails() {
        let c = cipher();
        let env = c.seal("acme", "mem-1", field::CONTENT, b"secret").unwrap();
        assert!(c.open("acme", "mem-2", field::CONTENT, &env).is_err());
        assert!(c.open("other", "mem-1", field::CONTENT, &env).is_err());
        assert!(c.open("acme", "mem-1", field::SOURCE, &env).is_err());
    }

    #[test]
    fn tampered_ciphertext_fails() {
        let c = cipher();
        let mut env = c.seal("acme", "mem-1", field::CONTENT, b"secret").unwrap();
        let last = env.len() - 1;
        env[last] ^= 0x01;
        assert!(c.open("acme", "mem-1", field::CONTENT, &env).is_err());
    }

    #[test]
    fn wrong_key_fails() {
        let a = FieldCipher::new(
            Arc::new(NoOpDevKeyProvider::with_custom_key(vec![0x01; 32])),
            LexicalMode::Disabled,
        );
        let b = FieldCipher::new(
            Arc::new(NoOpDevKeyProvider::with_custom_key(vec![0x02; 32])),
            LexicalMode::Disabled,
        );
        let env = a.seal("acme", "mem-1", field::CONTENT, b"secret").unwrap();
        assert!(b.open("acme", "mem-1", field::CONTENT, &env).is_err());
    }

    #[test]
    fn blind_index_matches_shared_tokens() {
        let c = cipher();
        let idx = c.blind_index_tokens("acme", "Concise Technical Responses").unwrap();
        let q = c.blind_index_query("acme", "technical").unwrap();
        let term = q.trim_matches('"');
        assert!(idx.split(' ').any(|t| t == term));
    }
}
