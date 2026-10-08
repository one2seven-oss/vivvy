use crate::cipher::{field, FieldCipher, LexicalMode, SCHEME_XCHACHA20POLY1305};
use crate::error::{ErrorCode, MemoryError, Result};
use crate::model::{MemoryFilter, MemoryKind, MemoryRecord, MemoryStatus};
use crate::namespace::MemoryScope;
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashMap;
use std::path::Path;

fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks.iter().map(|&chunk| f32::from_le_bytes(chunk)).collect()
}

fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Raw, still-sealed column values for one `memories` row, read inside a
/// rusqlite closure. Decryption (which needs the key provider and may fail) is
/// performed afterwards by [`Repository::materialize`], outside the closure.
struct RawMemoryRow {
    id: String,
    tenant_id: String,
    namespace: String,
    agent_id: Option<String>,
    user_id: Option<String>,
    kind_str: String,
    content: Vec<u8>,
    content_hash: Vec<u8>,
    embedding: Vec<u8>,
    embedding_model: String,
    embedding_dims: i64,
    importance: f64,
    created_at_ms: i64,
    updated_at_ms: i64,
    last_accessed_at_ms: Option<i64>,
    access_count: i64,
    expires_at_ms: Option<i64>,
    status_str: String,
    revision: i64,
    metadata: Vec<u8>,
    source: Vec<u8>,
    enc_version: i64,
}

/// Column list shared by every full-record SELECT, with `enc_version` last.
const RECORD_COLUMNS: &str = "id, tenant_id, namespace, agent_id, user_id, kind, \
     content, content_hash, embedding, embedding_model, embedding_dims, \
     importance, created_at_ms, updated_at_ms, last_accessed_at_ms, \
     access_count, expires_at_ms, status, revision, metadata_json, source_json, enc_version";

fn read_raw_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RawMemoryRow> {
    Ok(RawMemoryRow {
        id: row.get(0)?,
        tenant_id: row.get(1)?,
        namespace: row.get(2)?,
        agent_id: row.get(3)?,
        user_id: row.get(4)?,
        kind_str: row.get(5)?,
        content: row.get(6)?,
        content_hash: row.get(7)?,
        embedding: row.get(8)?,
        embedding_model: row.get(9)?,
        embedding_dims: row.get(10)?,
        importance: row.get(11)?,
        created_at_ms: row.get(12)?,
        updated_at_ms: row.get(13)?,
        last_accessed_at_ms: row.get(14)?,
        access_count: row.get(15)?,
        expires_at_ms: row.get(16)?,
        status_str: row.get(17)?,
        revision: row.get(18)?,
        metadata: row.get(19)?,
        source: row.get(20)?,
        enc_version: row.get(21)?,
    })
}

const SCHEMA_V1: &str = r#"
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    agent_id TEXT,
    user_id TEXT,
    kind TEXT NOT NULL,
    content BLOB NOT NULL,
    content_hash BLOB NOT NULL,
    embedding BLOB NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dims INTEGER NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    last_accessed_at_ms INTEGER,
    access_count INTEGER NOT NULL DEFAULT 0,
    expires_at_ms INTEGER,
    status TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    source_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS memories_scope_active
    ON memories(tenant_id, namespace, status, expires_at_ms);

CREATE TABLE IF NOT EXISTS operations (
    operation_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    applied_at_ms INTEGER
);
"#;

const SCHEMA_V2: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id UNINDEXED,
    tenant_id UNINDEXED,
    namespace UNINDEXED,
    content,
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(id, tenant_id, namespace, content)
    VALUES (new.id, new.tenant_id, new.namespace, CAST(new.content AS TEXT));
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    DELETE FROM memories_fts WHERE id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    DELETE FROM memories_fts WHERE id = old.id;
    INSERT INTO memories_fts(id, tenant_id, namespace, content)
    VALUES (new.id, new.tenant_id, new.namespace, CAST(new.content AS TEXT));
END;
"#;

const SCHEMA_V3: &str = r#"
CREATE TABLE IF NOT EXISTS operations_v3 (
    tenant_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    applied_at_ms INTEGER,
    PRIMARY KEY (tenant_id, namespace, operation_id)
);

INSERT OR IGNORE INTO operations_v3
SELECT m.tenant_id, m.namespace, o.operation_id, o.memory_id, o.kind, o.payload_json, o.state, o.created_at_ms, o.applied_at_ms
FROM operations o
JOIN memories m ON o.memory_id = m.id;

DROP TABLE operations;
ALTER TABLE operations_v3 RENAME TO operations;
"#;

// v4 adds encryption-at-rest support:
//   * `enc_version` marks whether a row's sensitive columns are sealed (1) or
//     legacy plaintext (0), so old stores keep reading after an upgrade.
//   * The FTS5 triggers are dropped: content may now be ciphertext, so the
//     lexical index is maintained from Rust (plaintext, blind-index, or not at
//     all) depending on the store's LexicalMode.
//   * `store_meta` records the encryption scheme so an encrypted store cannot
//     be silently opened without a key, and vice versa.
const SCHEMA_V4: &str = r#"
ALTER TABLE memories ADD COLUMN enc_version INTEGER NOT NULL DEFAULT 0;

DROP TRIGGER IF EXISTS memories_ai;
DROP TRIGGER IF EXISTS memories_ad;
DROP TRIGGER IF EXISTS memories_au;

CREATE TABLE IF NOT EXISTS store_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"#;

fn sanitize_fts_query(input: &str) -> String {
    let clean: String = input
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { ' ' })
        .collect();
    let tokens: Vec<&str> = clean.split_whitespace().collect();
    if tokens.is_empty() {
        return String::new();
    }
    let phrase = format!("\"{}\"", tokens.join(" "));
    let term_ors = tokens
        .iter()
        .map(|t| format!("\"{}\"", t))
        .collect::<Vec<_>>()
        .join(" OR ");
    format!("{} OR {}", phrase, term_ors)
}

pub struct Repository {
    conn: Mutex<Connection>,
    cipher: Option<FieldCipher>,
    lexical_mode: LexicalMode,
}

impl Repository {
    /// Open an unencrypted store with plaintext lexical recall (historical
    /// behavior). Equivalent to [`open_with_encryption`](Self::open_with_encryption)
    /// with no cipher.
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_encryption(db_path, None, LexicalMode::Plaintext)
    }

    /// Open a store, optionally sealing sensitive fields per-tenant with
    /// `cipher`. `lexical_mode` selects how the keyword index is maintained and
    /// must be consistent with whether `cipher` is present (validated by the
    /// config builder).
    pub fn open_with_encryption(
        db_path: impl AsRef<Path>,
        cipher: Option<FieldCipher>,
        lexical_mode: LexicalMode,
    ) -> Result<Self> {
        let conn = Connection::open(db_path.as_ref()).map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to open SQLite database: {}", e),
        })?;

        // Configure WAL mode, synchronous=NORMAL, and foreign keys
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to set WAL mode: {}", e),
            })?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to set synchronous mode: {}", e),
            })?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to enable foreign keys: {}", e),
            })?;

        let repo = Self {
            conn: Mutex::new(conn),
            cipher,
            lexical_mode,
        };
        repo.migrate()?;
        repo.enforce_encryption_contract()?;
        repo.integrity_check()?;
        Ok(repo)
    }

    /// Guard against opening an encrypted store without a key (which would make
    /// every record unreadable) or changing the on-disk scheme out from under
    /// existing sealed data. Records this store's encryption state on first use.
    fn enforce_encryption_contract(&self) -> Result<()> {
        let conn = self.conn.lock();
        let existing: Option<String> = conn
            .query_row(
                "SELECT value FROM store_meta WHERE key = 'encryption_scheme'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to read store encryption metadata: {}", e),
            })?;

        let current = if self.cipher.is_some() {
            SCHEME_XCHACHA20POLY1305
        } else {
            "none"
        };

        match existing.as_deref() {
            // Fresh store, or a plaintext store being opened (optionally being
            // upgraded to encryption). Legacy rows stay readable via enc_version.
            None | Some("none") => {}
            // Previously encrypted with the scheme we support: a key is required.
            Some(scheme) if scheme == SCHEME_XCHACHA20POLY1305 => {
                if self.cipher.is_none() {
                    return Err(MemoryError::encryption_key_unavailable(
                        "store contains encrypted records but was opened without a KeyProvider",
                    ));
                }
            }
            // Anything else is an unknown/incompatible on-disk scheme.
            Some(other) => {
                return Err(MemoryError::CorruptStore {
                    code: ErrorCode::CorruptStore,
                    message: format!(
                        "store encryption scheme mismatch: on-disk '{other}', configured '{current}'"
                    ),
                });
            }
        }

        // Record/refresh the marker. Once encryption is adopted the marker stays
        // encrypted, so the store can never later be reopened without a key.
        conn.execute(
            "INSERT INTO store_meta (key, value) VALUES ('encryption_scheme', ?)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![current],
        )
        .map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to record store encryption metadata: {}", e),
        })?;

        Ok(())
    }

    #[inline]
    fn enc_version(&self) -> i64 {
        if self.cipher.is_some() {
            1
        } else {
            0
        }
    }

    /// Seal a field for storage, or pass it through unchanged when the store is
    /// not encrypted.
    fn seal_field(
        &self,
        tenant_id: &str,
        record_id: &str,
        field_name: &str,
        plaintext: &[u8],
    ) -> Result<Vec<u8>> {
        match &self.cipher {
            Some(c) => c.seal(tenant_id, record_id, field_name, plaintext),
            None => Ok(plaintext.to_vec()),
        }
    }

    /// Open a stored field, honoring the row's `enc_version` so that legacy
    /// plaintext rows (0) and sealed rows (1) are both handled.
    fn open_field(
        &self,
        enc_version: i64,
        tenant_id: &str,
        record_id: &str,
        field_name: &str,
        stored: &[u8],
    ) -> Result<Vec<u8>> {
        if enc_version == 0 {
            return Ok(stored.to_vec());
        }
        match &self.cipher {
            Some(c) => c.open(tenant_id, record_id, field_name, stored),
            None => Err(MemoryError::encryption_key_unavailable(
                "record is sealed but store was opened without a KeyProvider",
            )),
        }
    }

    /// The value to store in the FTS `content` column for a record, or `None`
    /// when no lexical index is maintained.
    fn fts_content_for(&self, record: &MemoryRecord) -> Result<Option<String>> {
        match self.lexical_mode {
            LexicalMode::Plaintext => Ok(Some(record.content.clone())),
            LexicalMode::Disabled => Ok(None),
            LexicalMode::BlindIndex => {
                let c = self.cipher.as_ref().ok_or_else(|| {
                    MemoryError::encryption_key_unavailable(
                        "BlindIndex lexical mode requires a KeyProvider",
                    )
                })?;
                Ok(Some(
                    c.blind_index_tokens(record.scope.tenant_id(), &record.content)?,
                ))
            }
        }
    }

    /// Reconstruct a decrypted [`MemoryRecord`] from a raw row.
    fn materialize(&self, raw: RawMemoryRow) -> Result<MemoryRecord> {
        let content_bytes = self.open_field(
            raw.enc_version,
            &raw.tenant_id,
            &raw.id,
            field::CONTENT,
            &raw.content,
        )?;
        let embedding_bytes = self.open_field(
            raw.enc_version,
            &raw.tenant_id,
            &raw.id,
            field::EMBEDDING,
            &raw.embedding,
        )?;
        let metadata_bytes = self.open_field(
            raw.enc_version,
            &raw.tenant_id,
            &raw.id,
            field::METADATA,
            &raw.metadata,
        )?;
        let source_bytes = self.open_field(
            raw.enc_version,
            &raw.tenant_id,
            &raw.id,
            field::SOURCE,
            &raw.source,
        )?;

        let mut scope = MemoryScope::new(raw.tenant_id, raw.namespace)?;
        if let Some(agent) = raw.agent_id {
            scope = scope.with_agent(agent)?;
        }
        if let Some(user) = raw.user_id {
            scope = scope.with_user(user)?;
        }

        let kind = match raw.kind_str.as_str() {
            "preference" => MemoryKind::Preference,
            "fact" => MemoryKind::Fact,
            "instruction" => MemoryKind::Instruction,
            "context" => MemoryKind::Context,
            _ => MemoryKind::Episodic,
        };

        let metadata: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&metadata_bytes).unwrap_or_default();
        let source: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&source_bytes).unwrap_or_default();

        Ok(MemoryRecord {
            id: raw.id,
            scope,
            kind,
            content: String::from_utf8_lossy(&content_bytes).to_string(),
            content_hash: raw.content_hash,
            embedding: bytes_to_embedding(&embedding_bytes),
            embedding_model: raw.embedding_model,
            embedding_dims: raw.embedding_dims as usize,
            importance: raw.importance as f32,
            created_at_ms: raw.created_at_ms,
            updated_at_ms: raw.updated_at_ms,
            last_accessed_at_ms: raw.last_accessed_at_ms,
            access_count: raw.access_count as u64,
            expires_at_ms: raw.expires_at_ms,
            status: MemoryStatus::parse(&raw.status_str).unwrap_or(MemoryStatus::Pending),
            revision: raw.revision as u64,
            metadata,
            source,
        })
    }

    /// Perform a crash-consistent online backup of the SQLite database using VACUUM INTO.
    pub fn backup_sqlite(&self, target_db_path: &Path) -> Result<()> {
        let conn = self.conn.lock();
        let path_str = target_db_path.to_str().ok_or_else(|| MemoryError::InvalidInput {
            code: ErrorCode::InvalidInput,
            message: "invalid target backup path".to_string(),
        })?;

        conn.execute("VACUUM INTO ?", params![path_str])
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("SQLite VACUUM INTO failed: {}", e),
            })?;
        Ok(())
    }

    pub fn migrate(&self) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to start migration transaction: {}", e),
        })?;

        let table_exists: bool = tx
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'",
                [],
                |_| Ok(true),
            )
            .optional()
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to check table existence: {}", e),
            })?
            .unwrap_or(false);

        let current_ver: i64 = if table_exists {
            tx.query_row(
                "SELECT MAX(version) FROM schema_migrations",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to check migration version: {}", e),
            })?
            .flatten()
            .unwrap_or(0)
        } else {
            0
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        if current_ver < 1 {
            tx.execute_batch(SCHEMA_V1)
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to apply schema v1: {}", e),
                })?;
            tx.execute(
                "INSERT INTO schema_migrations (version, applied_at_ms) VALUES (1, ?)",
                params![now],
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to record schema v1 migration: {}", e),
            })?;
        }

        if current_ver < 2 {
            tx.execute_batch(SCHEMA_V2)
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to apply schema v2: {}", e),
                })?;
            tx.execute(
                r#"
                INSERT OR IGNORE INTO memories_fts(id, tenant_id, namespace, content)
                SELECT id, tenant_id, namespace, CAST(content AS TEXT)
                FROM memories
                WHERE status != 'deleted'
                "#,
                [],
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to backfill memories_fts: {}", e),
            })?;
            tx.execute(
                "INSERT INTO schema_migrations (version, applied_at_ms) VALUES (2, ?)",
                params![now],
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to record schema v2 migration: {}", e),
            })?;
        }

        if current_ver < 3 {
            tx.execute_batch(SCHEMA_V3)
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to apply schema v3: {}", e),
                })?;
            tx.execute(
                "INSERT INTO schema_migrations (version, applied_at_ms) VALUES (3, ?)",
                params![now],
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to record schema v3 migration: {}", e),
            })?;
        }

        if current_ver < 4 {
            tx.execute_batch(SCHEMA_V4)
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to apply schema v4: {}", e),
                })?;
            tx.execute(
                "INSERT INTO schema_migrations (version, applied_at_ms) VALUES (4, ?)",
                params![now],
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to record schema v4 migration: {}", e),
            })?;
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit migration transaction: {}", e),
        })?;

        Ok(())
    }

    pub fn integrity_check(&self) -> Result<()> {
        let conn = self.conn.lock();
        let check_status: String = conn
            .query_row("PRAGMA integrity_check;", [], |row| row.get(0))
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to run PRAGMA integrity_check: {}", e),
            })?;

        if check_status.to_lowercase() != "ok" {
            return Err(MemoryError::CorruptStore {
                code: ErrorCode::CorruptStore,
                message: format!("SQLite integrity check failed: {}", check_status),
            });
        }
        Ok(())
    }

    pub fn insert_pending_memory(
        &self,
        record: &MemoryRecord,
        operation_id: Option<&str>,
        now_ms: i64,
    ) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to begin transaction: {}", e),
        })?;

        let tenant = record.scope.tenant_id();
        let content_blob =
            self.seal_field(tenant, &record.id, field::CONTENT, record.content.as_bytes())?;
        let embedding_blob = self.seal_field(
            tenant,
            &record.id,
            field::EMBEDDING,
            &embedding_to_bytes(&record.embedding),
        )?;
        let metadata_json = serde_json::to_string(&record.metadata).unwrap_or_else(|_| "{}".into());
        let source_json = serde_json::to_string(&record.source).unwrap_or_else(|_| "{}".into());
        let metadata_blob =
            self.seal_field(tenant, &record.id, field::METADATA, metadata_json.as_bytes())?;
        let source_blob =
            self.seal_field(tenant, &record.id, field::SOURCE, source_json.as_bytes())?;

        tx.execute(
            r#"
            INSERT INTO memories (
                id, tenant_id, namespace, agent_id, user_id, kind,
                content, content_hash, embedding, embedding_model, embedding_dims,
                importance, created_at_ms, updated_at_ms, last_accessed_at_ms,
                access_count, expires_at_ms, status, revision, metadata_json, source_json,
                enc_version
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?
            )
            "#,
            params![
                record.id,
                record.scope.tenant_id(),
                record.scope.namespace(),
                record.scope.agent_id(),
                record.scope.user_id(),
                format!("{:?}", record.kind).to_lowercase(),
                content_blob,
                record.content_hash,
                embedding_blob,
                record.embedding_model,
                record.embedding_dims as i64,
                record.importance,
                record.created_at_ms,
                record.updated_at_ms,
                record.last_accessed_at_ms,
                record.access_count as i64,
                record.expires_at_ms,
                record.status.as_str(),
                record.revision as i64,
                metadata_blob,
                source_blob,
                self.enc_version(),
            ],
        )
        .map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to insert memory: {}", e),
        })?;

        self.fts_upsert(&tx, record)?;

        if let Some(op_id) = operation_id {
            let payload = serde_json::to_string(record).unwrap_or_else(|_| "{}".into());
            let payload_blob =
                self.seal_field(tenant, op_id, field::OP_PAYLOAD, payload.as_bytes())?;
            tx.execute(
                r#"
                INSERT INTO operations (
                    tenant_id, namespace, operation_id, memory_id, kind, payload_json, state, created_at_ms, applied_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                "#,
                params![
                    record.scope.tenant_id(),
                    record.scope.namespace(),
                    op_id,
                    record.id,
                    "remember",
                    payload_blob,
                    "pending",
                    now_ms,
                ],
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to insert operation journal: {}", e),
            })?;
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit memory transaction: {}", e),
        })?;

        Ok(())
    }

    /// Insert (or replace) the FTS row for a record according to the store's
    /// lexical mode. No-op under `Disabled`. Runs inside an existing
    /// transaction so FTS stays consistent with the `memories` write.
    fn fts_upsert(&self, tx: &rusqlite::Transaction<'_>, record: &MemoryRecord) -> Result<()> {
        tx.execute("DELETE FROM memories_fts WHERE id = ?", params![record.id])
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to clear FTS row: {}", e),
            })?;
        if let Some(fts_content) = self.fts_content_for(record)? {
            tx.execute(
                "INSERT INTO memories_fts(id, tenant_id, namespace, content) VALUES (?, ?, ?, ?)",
                params![
                    record.id,
                    record.scope.tenant_id(),
                    record.scope.namespace(),
                    fts_content,
                ],
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to insert FTS row: {}", e),
            })?;
        }
        Ok(())
    }

    fn fts_delete(&self, tx: &rusqlite::Transaction<'_>, id: &str) -> Result<()> {
        tx.execute("DELETE FROM memories_fts WHERE id = ?", params![id])
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to delete FTS row: {}", e),
            })?;
        Ok(())
    }

    pub fn insert_pending_memory_batch(
        &self,
        records: &[MemoryRecord],
        operation_ids: &[Option<String>],
        now_ms: i64,
    ) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to begin transaction: {}", e),
        })?;

        {
            let mut mem_stmt = tx.prepare(
                r#"
                INSERT INTO memories (
                    id, tenant_id, namespace, agent_id, user_id, kind,
                    content, content_hash, embedding, embedding_model, embedding_dims,
                    importance, created_at_ms, updated_at_ms, last_accessed_at_ms,
                    access_count, expires_at_ms, status, revision, metadata_json, source_json,
                    enc_version
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?
                )
                "#,
            ).map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to prepare batch memory insert statement: {}", e),
            })?;

            let mut op_stmt = tx.prepare(
                r#"
                INSERT INTO operations (
                    tenant_id, namespace, operation_id, memory_id, kind, payload_json, state, created_at_ms, applied_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                "#,
            ).map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to prepare batch operation insert statement: {}", e),
            })?;

            let mut fts_del_stmt = tx.prepare("DELETE FROM memories_fts WHERE id = ?")
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to prepare batch FTS delete statement: {}", e),
                })?;
            let mut fts_ins_stmt = tx.prepare(
                "INSERT INTO memories_fts(id, tenant_id, namespace, content) VALUES (?, ?, ?, ?)",
            ).map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to prepare batch FTS insert statement: {}", e),
            })?;

            for (i, record) in records.iter().enumerate() {
                let tenant = record.scope.tenant_id();
                let content_blob =
                    self.seal_field(tenant, &record.id, field::CONTENT, record.content.as_bytes())?;
                let embedding_blob = self.seal_field(
                    tenant,
                    &record.id,
                    field::EMBEDDING,
                    &embedding_to_bytes(&record.embedding),
                )?;
                let metadata_json = serde_json::to_string(&record.metadata).unwrap_or_else(|_| "{}".into());
                let source_json = serde_json::to_string(&record.source).unwrap_or_else(|_| "{}".into());
                let metadata_blob =
                    self.seal_field(tenant, &record.id, field::METADATA, metadata_json.as_bytes())?;
                let source_blob =
                    self.seal_field(tenant, &record.id, field::SOURCE, source_json.as_bytes())?;

                mem_stmt.execute(params![
                    record.id,
                    record.scope.tenant_id(),
                    record.scope.namespace(),
                    record.scope.agent_id(),
                    record.scope.user_id(),
                    format!("{:?}", record.kind).to_lowercase(),
                    content_blob,
                    record.content_hash,
                    embedding_blob,
                    record.embedding_model,
                    record.embedding_dims as i64,
                    record.importance,
                    record.created_at_ms,
                    record.updated_at_ms,
                    record.last_accessed_at_ms,
                    record.access_count as i64,
                    record.expires_at_ms,
                    record.status.as_str(),
                    record.revision as i64,
                    metadata_blob,
                    source_blob,
                    self.enc_version(),
                ]).map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to insert memory in batch: {}", e),
                })?;

                fts_del_stmt.execute(params![record.id]).map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to clear FTS row in batch: {}", e),
                })?;
                if let Some(fts_content) = self.fts_content_for(record)? {
                    fts_ins_stmt.execute(params![
                        record.id,
                        record.scope.tenant_id(),
                        record.scope.namespace(),
                        fts_content,
                    ]).map_err(|e| MemoryError::DatabaseError {
                        code: ErrorCode::DatabaseError,
                        message: format!("Failed to insert FTS row in batch: {}", e),
                    })?;
                }

                if let Some(op_id) = operation_ids.get(i).and_then(|opt| opt.as_deref()) {
                    let payload = serde_json::to_string(record).unwrap_or_else(|_| "{}".into());
                    let payload_blob =
                        self.seal_field(tenant, op_id, field::OP_PAYLOAD, payload.as_bytes())?;
                    op_stmt.execute(params![
                        record.scope.tenant_id(),
                        record.scope.namespace(),
                        op_id,
                        record.id,
                        "remember",
                        payload_blob,
                        "pending",
                        now_ms,
                    ]).map_err(|e| MemoryError::DatabaseError {
                        code: ErrorCode::DatabaseError,
                        message: format!("Failed to insert operation journal in batch: {}", e),
                    })?;
                }
            }
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit batch memory transaction: {}", e),
        })?;

        Ok(())
    }

    pub fn get_by_scope_and_id(
        &self,
        scope: &MemoryScope,
        id: &str,
    ) -> Result<Option<MemoryRecord>> {
        let raw = {
            let conn = self.conn.lock();
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT {RECORD_COLUMNS} FROM memories WHERE id = ? AND tenant_id = ? AND namespace = ?"
                ))
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to prepare query: {}", e),
                })?;

            stmt.query_row(
                params![id, scope.tenant_id(), scope.namespace()],
                read_raw_row,
            )
            .optional()
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Query failed: {}", e),
            })?
        };

        match raw {
            Some(raw) => Ok(Some(self.materialize(raw)?)),
            None => Ok(None),
        }
    }

    pub fn set_status(&self, id: &str, status: MemoryStatus) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE memories SET status = ? WHERE id = ?",
            params![status.as_str(), id],
        )
        .map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to update status: {}", e),
        })?;
        Ok(())
    }

    pub fn set_status_batch(&self, ids: &[&str], status: MemoryStatus) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to begin transaction for set_status_batch: {}", e),
        })?;

        {
            let mut stmt = tx.prepare("UPDATE memories SET status = ? WHERE id = ?")
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to prepare set_status_batch statement: {}", e),
                })?;

            for id in ids {
                stmt.execute(params![status.as_str(), id])
                    .map_err(|e| MemoryError::DatabaseError {
                        code: ErrorCode::DatabaseError,
                        message: format!("Failed to update status in batch: {}", e),
                    })?;
            }
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit set_status_batch transaction: {}", e),
        })?;
        Ok(())
    }

    pub fn get_operation(&self, scope: &MemoryScope, op_id: &str) -> Result<Option<(String, String)>> {
        let raw: Option<(String, Vec<u8>)> = {
            let conn = self.conn.lock();
            conn.query_row(
                "SELECT memory_id, payload_json FROM operations WHERE tenant_id = ? AND namespace = ? AND operation_id = ?",
                params![scope.tenant_id(), scope.namespace(), op_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to query operation: {}", e),
            })?
        };

        match raw {
            Some((memory_id, payload_bytes)) => {
                let payload_json = self.open_operation_payload(
                    scope.tenant_id(),
                    op_id,
                    &payload_bytes,
                )?;
                Ok(Some((memory_id, payload_json)))
            }
            None => Ok(None),
        }
    }

    /// Decrypt a journal payload if it is a sealed envelope, otherwise return it
    /// as plaintext. Journal payloads may be legacy plaintext (pre-encryption),
    /// the literal `{}` of a delete op, or sealed JSON on an encrypted store.
    fn open_operation_payload(&self, tenant_id: &str, op_id: &str, bytes: &[u8]) -> Result<String> {
        if let Some(c) = &self.cipher {
            if FieldCipher::is_sealed(bytes) {
                let pt = c.open(tenant_id, op_id, field::OP_PAYLOAD, bytes)?;
                return Ok(String::from_utf8_lossy(&pt).to_string());
            }
        }
        Ok(String::from_utf8_lossy(bytes).to_string())
    }

    pub fn mark_operation_applied(&self, scope: &MemoryScope, op_id: &str, applied_at_ms: i64) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE operations SET state = 'applied', applied_at_ms = ? WHERE tenant_id = ? AND namespace = ? AND operation_id = ?",
            params![applied_at_ms, scope.tenant_id(), scope.namespace(), op_id],
        )
        .map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to mark operation applied: {}", e),
        })?;
        Ok(())
    }

    pub fn mark_operation_applied_batch(&self, ops: &[(&MemoryScope, &str)], applied_at_ms: i64) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to begin transaction for mark_operation_applied_batch: {}", e),
        })?;

        {
            let mut stmt = tx.prepare("UPDATE operations SET state = 'applied', applied_at_ms = ? WHERE tenant_id = ? AND namespace = ? AND operation_id = ?")
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to prepare mark_operation_applied_batch statement: {}", e),
                })?;

            for (scope, op_id) in ops {
                stmt.execute(params![applied_at_ms, scope.tenant_id(), scope.namespace(), op_id])
                    .map_err(|e| MemoryError::DatabaseError {
                        code: ErrorCode::DatabaseError,
                        message: format!("Failed to mark operation applied in batch: {}", e),
                    })?;
            }
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit mark_operation_applied_batch transaction: {}", e),
        })?;
        Ok(())
    }

    pub fn get_pending_operations(&self) -> Result<Vec<(String, String, String)>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare("SELECT operation_id, memory_id, kind FROM operations WHERE state = 'pending' ORDER BY created_at_ms ASC")
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to prepare pending operations query: {}", e),
            })?;

        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Query pending operations failed: {}", e),
            })?;

        let mut ops = Vec::new();
        for r in rows {
            ops.push(r.map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Row error: {}", e),
            })?);
        }
        Ok(ops)
    }

    pub fn get_record_by_id_internal(&self, id: &str) -> Result<Option<MemoryRecord>> {
        let raw = {
            let conn = self.conn.lock();
            let mut stmt = conn
                .prepare(&format!("SELECT {RECORD_COLUMNS} FROM memories WHERE id = ?"))
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to prepare internal record query: {}", e),
                })?;

            stmt.query_row(params![id], read_raw_row)
                .optional()
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Internal query failed: {}", e),
                })?
        };

        match raw {
            Some(raw) => Ok(Some(self.materialize(raw)?)),
            None => Ok(None),
        }
    }

    pub fn get_all_active_records(&self) -> Result<Vec<MemoryRecord>> {
        let raws = {
            let conn = self.conn.lock();
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT {RECORD_COLUMNS} FROM memories WHERE status = 'active'"
                ))
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to prepare active records query: {}", e),
                })?;

            let rows = stmt
                .query_map([], read_raw_row)
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Query active records failed: {}", e),
                })?;

            let mut raws = Vec::new();
            for r in rows {
                raws.push(r.map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Row error: {}", e),
                })?);
            }
            raws
        };

        let mut records = Vec::with_capacity(raws.len());
        for raw in raws {
            records.push(self.materialize(raw)?);
        }
        Ok(records)
    }

    pub fn update_memory(
        &self,
        record: &MemoryRecord,
        expected_revision: u64,
        operation_id: Option<&str>,
        now_ms: i64,
    ) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to begin update transaction: {}", e),
        })?;

        // Verify current revision and scope
        let (current_revision, current_status): (i64, String) = tx
            .query_row(
                "SELECT revision, status FROM memories WHERE id = ? AND tenant_id = ? AND namespace = ?",
                params![record.id, record.scope.tenant_id(), record.scope.namespace()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to fetch current revision: {}", e),
            })?
            .ok_or_else(|| MemoryError::not_found(&record.id))?;

        if current_status == "deleted" {
            return Err(MemoryError::not_found(&record.id));
        }

        if current_revision as u64 != expected_revision {
            return Err(MemoryError::RevisionConflict {
                code: ErrorCode::RevisionConflict,
                id: record.id.clone(),
                expected: expected_revision,
                current: current_revision as u64,
            });
        }

        let tenant = record.scope.tenant_id();
        let content_blob =
            self.seal_field(tenant, &record.id, field::CONTENT, record.content.as_bytes())?;
        let embedding_blob = self.seal_field(
            tenant,
            &record.id,
            field::EMBEDDING,
            &embedding_to_bytes(&record.embedding),
        )?;
        let metadata_json = serde_json::to_string(&record.metadata).unwrap_or_else(|_| "{}".into());
        let metadata_blob =
            self.seal_field(tenant, &record.id, field::METADATA, metadata_json.as_bytes())?;

        tx.execute(
            r#"
            UPDATE memories SET
                content = ?,
                content_hash = ?,
                embedding = ?,
                importance = ?,
                updated_at_ms = ?,
                expires_at_ms = ?,
                status = ?,
                revision = ?,
                metadata_json = ?,
                enc_version = ?
            WHERE id = ? AND revision = ?
            "#,
            params![
                content_blob,
                record.content_hash,
                embedding_blob,
                record.importance,
                record.updated_at_ms,
                record.expires_at_ms,
                record.status.as_str(),
                (expected_revision + 1) as i64,
                metadata_blob,
                self.enc_version(),
                record.id,
                expected_revision as i64,
            ],
        )
        .map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to update memory record: {}", e),
        })?;

        self.fts_upsert(&tx, record)?;

        if let Some(op_id) = operation_id {
            let payload = serde_json::to_string(record).unwrap_or_else(|_| "{}".into());
            let payload_blob =
                self.seal_field(tenant, op_id, field::OP_PAYLOAD, payload.as_bytes())?;
            tx.execute(
                r#"
                INSERT INTO operations (
                    tenant_id, namespace, operation_id, memory_id, kind, payload_json, state, created_at_ms, applied_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                "#,
                params![
                    record.scope.tenant_id(),
                    record.scope.namespace(),
                    op_id,
                    record.id,
                    "update",
                    payload_blob,
                    "pending",
                    now_ms,
                ],
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to record update operation: {}", e),
            })?;
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit update transaction: {}", e),
        })?;

        Ok(())
    }

    pub fn delete_memory(
        &self,
        scope: &MemoryScope,
        id: &str,
        operation_id: Option<&str>,
        now_ms: i64,
    ) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to begin delete transaction: {}", e),
        })?;

        let rows_affected = tx.execute(
            r#"
            UPDATE memories SET
                status = 'deleted',
                updated_at_ms = ?
            WHERE id = ? AND tenant_id = ? AND namespace = ? AND status != 'deleted'
            "#,
            params![now_ms, id, scope.tenant_id(), scope.namespace()],
        )
        .map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to tombstone memory: {}", e),
        })?;

        if rows_affected == 0 {
            return Err(MemoryError::not_found(id));
        }

        self.fts_delete(&tx, id)?;

        if let Some(op_id) = operation_id {
            tx.execute(
                r#"
                INSERT INTO operations (
                    tenant_id, namespace, operation_id, memory_id, kind, payload_json, state, created_at_ms, applied_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                "#,
                params![scope.tenant_id(), scope.namespace(), op_id, id, "delete", "{}", "applied", now_ms, now_ms],
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to insert delete operation: {}", e),
            })?;
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit delete transaction: {}", e),
        })?;

        Ok(())
    }

    pub fn delete_memory_batch(
        &self,
        scope: &MemoryScope,
        ids: &[&str],
        operation_ids: &[Option<String>],
        now_ms: i64,
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }

        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to begin delete batch transaction: {}", e),
        })?;

        {
            let mut del_stmt = tx.prepare(
                r#"
                UPDATE memories SET
                    status = 'deleted',
                    updated_at_ms = ?
                WHERE id = ? AND tenant_id = ? AND namespace = ? AND status != 'deleted'
                "#,
            ).map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to prepare delete batch statement: {}", e),
            })?;

            let mut op_stmt = tx.prepare(
                r#"
                INSERT INTO operations (
                    tenant_id, namespace, operation_id, memory_id, kind, payload_json, state, created_at_ms, applied_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                "#,
            ).map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to prepare delete operation batch statement: {}", e),
            })?;

            let mut fts_del_stmt = tx.prepare("DELETE FROM memories_fts WHERE id = ?")
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to prepare batch FTS delete statement: {}", e),
                })?;

            for (i, id) in ids.iter().enumerate() {
                let rows = del_stmt.execute(params![now_ms, id, scope.tenant_id(), scope.namespace()])
                    .map_err(|e| MemoryError::DatabaseError {
                        code: ErrorCode::DatabaseError,
                        message: format!("Failed to tombstone memory in batch: {}", e),
                    })?;

                if rows == 0 {
                    return Err(MemoryError::not_found(*id));
                }

                fts_del_stmt.execute(params![id]).map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to delete FTS row in batch: {}", e),
                })?;

                if let Some(op_id) = operation_ids.get(i).and_then(|opt| opt.as_deref()) {
                    op_stmt.execute(params![scope.tenant_id(), scope.namespace(), op_id, id, "delete", "{}", "applied", now_ms, now_ms])
                        .map_err(|e| MemoryError::DatabaseError {
                            code: ErrorCode::DatabaseError,
                            message: format!("Failed to insert delete operation in batch: {}", e),
                        })?;
                }
            }
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit delete batch transaction: {}", e),
        })?;

        Ok(())
    }

    pub fn search_fts(
        &self,
        scope: &MemoryScope,
        query_text: &str,
        filter: &MemoryFilter,
        limit: usize,
    ) -> Result<Vec<String>> {
        let match_expr = match self.lexical_mode {
            // No lexical index is maintained; recall is pure-vector only.
            LexicalMode::Disabled => return Ok(Vec::new()),
            LexicalMode::Plaintext => sanitize_fts_query(query_text),
            LexicalMode::BlindIndex => {
                let c = self.cipher.as_ref().ok_or_else(|| {
                    MemoryError::encryption_key_unavailable(
                        "BlindIndex lexical mode requires a KeyProvider",
                    )
                })?;
                c.blind_index_query(scope.tenant_id(), query_text)?
            }
        };
        if match_expr.is_empty() {
            return Ok(Vec::new());
        }

        let mut query_sql = String::from(
            r#"
            SELECT f.id
            FROM memories_fts f
            JOIN memories m ON f.id = m.id
            WHERE f.tenant_id = ?
              AND f.namespace = ?
              AND f.memories_fts MATCH ?
              AND m.status = 'active'
            "#,
        );

        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = vec![
            Box::new(scope.tenant_id().to_string()),
            Box::new(scope.namespace().to_string()),
            Box::new(match_expr),
        ];

        if let Some(ref meta) = filter.metadata_eq {
            for (key, val) in meta {
                query_sql.push_str(" AND json_extract(m.metadata_json, ?) = ?");
                params_vec.push(Box::new(format!("$.{}", key)));
                match val {
                    serde_json::Value::String(s) => params_vec.push(Box::new(s.clone())),
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            params_vec.push(Box::new(i));
                        } else if let Some(f) = n.as_f64() {
                            params_vec.push(Box::new(f));
                        } else {
                            params_vec.push(Box::new(val.to_string()));
                        }
                    }
                    serde_json::Value::Bool(b) => params_vec.push(Box::new(if *b { 1i64 } else { 0i64 })),
                    _ => params_vec.push(Box::new(val.to_string())),
                }
            }
        }

        if let Some(after) = filter.created_after_ms {
            query_sql.push_str(" AND m.created_at_ms >= ?");
            params_vec.push(Box::new(after));
        }

        if let Some(before) = filter.created_before_ms {
            query_sql.push_str(" AND m.created_at_ms <= ?");
            params_vec.push(Box::new(before));
        }

        query_sql.push_str(" ORDER BY rank LIMIT ?");
        params_vec.push(Box::new(limit as i64));

        let conn = self.conn.lock();
        let mut stmt = conn.prepare(&query_sql).map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to prepare FTS query: {}", e),
        })?;

        let param_refs: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();
        let rows = stmt
            .query_map(&param_refs[..], |row| row.get(0))
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("FTS query failed: {}", e),
            })?;

        let ids = rows.flatten().collect();
        Ok(ids)
    }

    pub fn get_health_counts(&self) -> Result<(usize, usize, usize)> {
        let conn = self.conn.lock();
        let active_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE status = 'active'",
                [],
                |row| row.get(0),
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to count active records: {}", e),
            })?;

        let tombstoned_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE status = 'deleted'",
                [],
                |row| row.get(0),
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to count tombstoned records: {}", e),
            })?;

        let pending_ops_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM operations WHERE state = 'pending'",
                [],
                |row| row.get(0),
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to count pending operations: {}", e),
            })?;

        Ok((
            active_count as usize,
            tombstoned_count as usize,
            pending_ops_count as usize,
        ))
    }

    pub fn vacuum_tombstoned_records(&self, batch_size: usize, now_ms: i64) -> Result<usize> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to start vacuum transaction: {}", e),
        })?;

        // Collect the batch of ids to purge first so the FTS index can be kept
        // consistent (the AFTER DELETE trigger was removed in schema v4).
        let purge_ids: Vec<String> = {
            let mut stmt = tx
                .prepare(
                    r#"
                SELECT id FROM memories
                WHERE status = 'deleted'
                   OR (expires_at_ms IS NOT NULL AND expires_at_ms <= ?)
                LIMIT ?
                "#,
                )
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to prepare vacuum scan: {}", e),
                })?;
            let rows = stmt
                .query_map(params![now_ms, batch_size as i64], |row| row.get::<_, String>(0))
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to scan vacuum candidates: {}", e),
                })?;
            let mut ids = Vec::new();
            for r in rows {
                ids.push(r.map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Vacuum scan row error: {}", e),
                })?);
            }
            ids
        };

        let mut purged_count = 0usize;
        {
            let mut del_mem = tx
                .prepare("DELETE FROM memories WHERE id = ?")
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to prepare vacuum delete: {}", e),
                })?;
            let mut del_fts = tx
                .prepare("DELETE FROM memories_fts WHERE id = ?")
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to prepare vacuum FTS delete: {}", e),
                })?;
            for id in &purge_ids {
                purged_count += del_mem.execute(params![id]).map_err(|e| {
                    MemoryError::DatabaseError {
                        code: ErrorCode::DatabaseError,
                        message: format!("Failed to vacuum tombstoned records: {}", e),
                    }
                })?;
                del_fts.execute(params![id]).map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to vacuum FTS rows: {}", e),
                })?;
            }
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit vacuum transaction: {}", e),
        })?;

        Ok(purged_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_repository_lifecycle_and_reopen() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memory.db");

        let record = MemoryRecord {
            id: "mem-123".into(),
            scope: MemoryScope::new("tenant-a", "ns-1").unwrap(),
            kind: MemoryKind::Fact,
            content: "The sky is blue".into(),
            content_hash: vec![1, 2, 3, 4],
            embedding: vec![0.1, 0.2, 0.3],
            embedding_model: "test-model".into(),
            embedding_dims: 3,
            importance: 0.8,
            created_at_ms: 1000,
            updated_at_ms: 1000,
            last_accessed_at_ms: None,
            access_count: 0,
            expires_at_ms: None,
            status: MemoryStatus::Pending,
            revision: 1,
            metadata: HashMap::new(),
            source: HashMap::new(),
        };

        {
            let repo = Repository::open(&db_path).unwrap();
            repo.insert_pending_memory(&record, Some("op-1"), 1000)
                .unwrap();
            let fetched = repo
                .get_by_scope_and_id(&record.scope, &record.id)
                .unwrap()
                .expect("record should exist");
            assert_eq!(fetched.id, "mem-123");
            assert_eq!(fetched.content, "The sky is blue");
            assert_eq!(fetched.embedding, vec![0.1, 0.2, 0.3]);
            assert_eq!(fetched.status, MemoryStatus::Pending);
        }

        // Reopen database
        {
            let repo = Repository::open(&db_path).unwrap();
            let fetched = repo
                .get_by_scope_and_id(&record.scope, &record.id)
                .unwrap()
                .expect("record should persist across reopen");
            assert_eq!(fetched.id, "mem-123");
            assert_eq!(fetched.content, "The sky is blue");
            assert_eq!(fetched.embedding, vec![0.1, 0.2, 0.3]);
        }
    }

    #[test]
    fn test_cross_tenant_isolation_in_sql() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memory.db");
        let repo = Repository::open(&db_path).unwrap();

        let record = MemoryRecord {
            id: "mem-secret".into(),
            scope: MemoryScope::new("tenant-a", "ns-1").unwrap(),
            kind: MemoryKind::Fact,
            content: "Secret data".into(),
            content_hash: vec![0],
            embedding: vec![1.0, 0.0],
            embedding_model: "test".into(),
            embedding_dims: 2,
            importance: 0.5,
            created_at_ms: 100,
            updated_at_ms: 100,
            last_accessed_at_ms: None,
            access_count: 0,
            expires_at_ms: None,
            status: MemoryStatus::Pending,
            revision: 1,
            metadata: HashMap::new(),
            source: HashMap::new(),
        };

        repo.insert_pending_memory(&record, None, 100).unwrap();

        // Querying from tenant-b returns None
        let other_scope = MemoryScope::new("tenant-b", "ns-1").unwrap();
        assert!(repo
            .get_by_scope_and_id(&other_scope, "mem-secret")
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_fts_retrieval_and_scope_isolation() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memory.db");
        let repo = Repository::open(&db_path).unwrap();

        let scope_a = MemoryScope::new("tenant-a", "ns-1").unwrap();
        let scope_b = MemoryScope::new("tenant-b", "ns-1").unwrap();

        let record_a = MemoryRecord {
            id: "mem-fts-1".into(),
            scope: scope_a.clone(),
            kind: MemoryKind::Fact,
            content: "Quantum computing breakthrough in silicon quantum dots".into(),
            content_hash: vec![1],
            embedding: vec![0.1, 0.9],
            embedding_model: "test".into(),
            embedding_dims: 2,
            importance: 0.9,
            created_at_ms: 100,
            updated_at_ms: 100,
            last_accessed_at_ms: None,
            access_count: 0,
            expires_at_ms: None,
            status: MemoryStatus::Active,
            revision: 1,
            metadata: HashMap::new(),
            source: HashMap::new(),
        };

        repo.insert_pending_memory(&record_a, None, 100).unwrap();
        repo.set_status("mem-fts-1", MemoryStatus::Active).unwrap();

        let record_b = MemoryRecord {
            id: "mem-fts-2".into(),
            scope: scope_b.clone(),
            kind: MemoryKind::Fact,
            content: "Quantum computing in silicon".into(),
            content_hash: vec![2],
            embedding: vec![0.1, 0.9],
            embedding_model: "test".into(),
            embedding_dims: 2,
            importance: 0.9,
            created_at_ms: 100,
            updated_at_ms: 100,
            last_accessed_at_ms: None,
            access_count: 0,
            expires_at_ms: None,
            status: MemoryStatus::Active,
            revision: 1,
            metadata: HashMap::new(),
            source: HashMap::new(),
        };

        repo.insert_pending_memory(&record_b, None, 100).unwrap();
        repo.set_status("mem-fts-2", MemoryStatus::Active).unwrap();

        // Search scope_a for "Quantum silicon"
        let matches_a = repo.search_fts(&scope_a, "Quantum silicon", &MemoryFilter::default(), 10).unwrap();
        assert_eq!(matches_a, vec!["mem-fts-1"]);

        // Search scope_b for "Quantum silicon"
        let matches_b = repo.search_fts(&scope_b, "Quantum silicon", &MemoryFilter::default(), 10).unwrap();
        assert_eq!(matches_b, vec!["mem-fts-2"]);

        // Tombstone mem-fts-1 and verify FTS search omits deleted
        repo.delete_memory(&scope_a, "mem-fts-1", None, 200).unwrap();
        let matches_after_del = repo.search_fts(&scope_a, "Quantum silicon", &MemoryFilter::default(), 10).unwrap();
        assert!(matches_after_del.is_empty());
    }
}
