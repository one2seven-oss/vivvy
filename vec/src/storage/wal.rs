use log::warn;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use thiserror::Error;

const WAL_MAGIC: [u8; 8] = *b"VIVWAL\0\0";
const WAL_VERSION: u32 = 2;
const WAL_HEADER_LEN: u64 = (WAL_MAGIC.len() + std::mem::size_of::<u32>()) as u64;
const RECORD_LEN_LEN: u64 = std::mem::size_of::<u32>() as u64;
const RECORD_CRC_LEN: u64 = std::mem::size_of::<u32>() as u64;
const INSERT_FIXED_LEN: usize = 1 + 8 + 4;
const DELETE_BODY_LEN: usize = 1 + 8;

/// A deliberately finite upper bound before a WAL record is allocated during
/// recovery. Index construction additionally requires every insert to match
/// the configured index dimension.
pub const MAX_WAL_DIM: usize = 1_048_576;
const MAX_RECORD_BODY_LEN: usize = INSERT_FIXED_LEN + MAX_WAL_DIM * std::mem::size_of::<f32>();

/// Controls whether a completed WAL append is forced to stable storage.
/// `Sync` is the conservative default. `Flush` makes entries visible to the
/// operating system but permits their loss in a power failure; callers may use
/// [`WalWriter::sync`] as an explicit durability barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WalDurability {
    #[default]
    Sync,
    Flush,
}

#[derive(Debug, Error)]
pub enum WalError {
    #[error("I/O: {0}")]
    Io(#[from] io::Error),
    #[error("corrupt entry at offset {0}")]
    Corrupt(u64),
    #[error("unsupported WAL version {0}")]
    UnsupportedVersion(u32),
    #[error("WAL vector dimension {dim} exceeds the maximum {max}")]
    DimensionTooLarge { dim: usize, max: usize },
    #[error("WAL vector dimension {actual} does not match index dimension {expected}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Single recorded operation in the WAL.
#[derive(Debug, Clone, PartialEq)]
pub enum WalEntry {
    Insert { id: u64, vector: Vec<f32> },
    Delete { id: u64 },
}

/// append-only, crash-safe.
pub struct WalWriter {
    file: BufWriter<File>,
    path: Box<Path>,
    committed: u64,
    needs_dir_sync: bool,
    durability: WalDurability,
}

impl WalWriter {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WalError> {
        Self::open_with_durability(path, WalDurability::Sync)
    }

    pub fn open_with_durability(
        path: impl AsRef<Path>,
        durability: WalDurability,
    ) -> Result<Self, WalError> {
        let path = path.as_ref();
        let existed = path.exists();

        // Never append behind an invalid tail. This also converts a v1 WAL
        // before a v2 record can be added to it.
        if existed && path.metadata()?.len() > 0 {
            Self::replay(path, |_| {})?;
        }

        Self::open_existing_recovered(path, existed, durability)
    }

    /// Open a WAL whose existing contents were just replayed by the caller.
    /// This prevents index startup from parsing a healthy WAL twice.
    pub(crate) fn open_after_recovery(
        path: impl AsRef<Path>,
        durability: WalDurability,
    ) -> Result<Self, WalError> {
        let path = path.as_ref();
        Self::open_existing_recovered(path, path.exists(), durability)
    }

    fn open_existing_recovered(
        path: &Path,
        existed: bool,
        durability: WalDurability,
    ) -> Result<Self, WalError> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(path)?;
        let mut file = BufWriter::new(file);
        let mut committed = file.get_ref().metadata()?.len();
        if committed == 0 {
            write_header(&mut file)?;
            // The header becomes durable with the next commit. A process that
            // crashes before then merely leaves an empty WAL, which is valid.
            committed = 0;
        }
        Ok(Self {
            file,
            path: path.into(),
            committed,
            needs_dir_sync: !existed,
            durability,
        })
    }

    pub fn append(&mut self, entry: &WalEntry) -> Result<(), WalError> {
        write_record(&mut self.file, entry)
    }

    pub fn commit(&mut self) -> Result<(), WalError> {
        self.file.flush()?;
        if self.durability == WalDurability::Sync {
            self.sync()?;
        }
        self.committed = self.file.get_ref().metadata()?.len();
        Ok(())
    }

    /// Force all pending records to stable storage regardless of the
    /// configured durability mode.
    pub fn sync(&mut self) -> Result<(), WalError> {
        self.file.flush()?;
        self.file.get_ref().sync_all()?;
        if self.needs_dir_sync {
            sync_parent_dir(&self.path)?;
            self.needs_dir_sync = false;
        }
        self.committed = self.file.get_ref().metadata()?.len();
        Ok(())
    }

    /// Remove all records after delta segments are compacted, retaining the
    /// durable v2 file header.
    pub fn reset(&mut self) -> Result<(), WalError> {
        // Flush first so replacing this BufWriter cannot later drop buffered
        // bytes into the file after it was truncated.
        self.file.flush()?;
        let file = self.file.get_mut();
        file.set_len(WAL_HEADER_LEN)?;
        file.seek(SeekFrom::Start(WAL_HEADER_LEN))?;
        file.sync_all()?;
        sync_parent_dir(&self.path)?;
        self.committed = WAL_HEADER_LEN;
        self.needs_dir_sync = false;
        Ok(())
    }

    /// Replay all entries from the WAL, calling `f` for each.
    pub fn replay(path: impl AsRef<Path>, f: impl FnMut(WalEntry)) -> Result<(), WalError> {
        Self::replay_with_expected_dim(path, None, f)
    }

    /// Replay and recover a WAL, rejecting insert records that do not match
    /// `expected_dim` when it is supplied by an owning index.
    pub fn replay_with_expected_dim(
        path: impl AsRef<Path>,
        expected_dim: Option<usize>,
        mut f: impl FnMut(WalEntry),
    ) -> Result<(), WalError> {
        if let Some(dim) = expected_dim {
            validate_dim(dim)?;
        }

        let path = path.as_ref();
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        drop(file);

        if has_v2_prefix(path)? {
            replay_v2(path, expected_dim, &mut f)
        } else {
            replay_v1_and_migrate(path, expected_dim, &mut f)
        }
    }
}

fn write_header(mut writer: impl Write) -> io::Result<()> {
    writer.write_all(&WAL_MAGIC)?;
    writer.write_all(&WAL_VERSION.to_le_bytes())
}

fn validate_dim(dim: usize) -> Result<(), WalError> {
    if dim > MAX_WAL_DIM {
        return Err(WalError::DimensionTooLarge {
            dim,
            max: MAX_WAL_DIM,
        });
    }
    Ok(())
}

fn body_for_entry(entry: &WalEntry) -> Result<Vec<u8>, WalError> {
    let mut body = match entry {
        WalEntry::Insert { vector, .. } => {
            validate_dim(vector.len())?;
            Vec::with_capacity(INSERT_FIXED_LEN + vector.len() * std::mem::size_of::<f32>())
        }
        WalEntry::Delete { .. } => Vec::with_capacity(DELETE_BODY_LEN),
    };

    match entry {
        WalEntry::Insert { id, vector } => {
            let dim = u32::try_from(vector.len()).map_err(|_| WalError::DimensionTooLarge {
                dim: vector.len(),
                max: MAX_WAL_DIM,
            })?;
            body.push(0x01);
            body.extend_from_slice(&id.to_le_bytes());
            body.extend_from_slice(&dim.to_le_bytes());
            for value in vector {
                body.extend_from_slice(&value.to_le_bytes());
            }
        }
        WalEntry::Delete { id } => {
            body.push(0x02);
            body.extend_from_slice(&id.to_le_bytes());
        }
    }
    Ok(body)
}

fn write_record(mut writer: impl Write, entry: &WalEntry) -> Result<(), WalError> {
    let body = body_for_entry(entry)?;
    let len = u32::try_from(body.len()).map_err(|_| WalError::Corrupt(0))?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&body)?;
    writer.write_all(&crc32fast::hash(&body).to_le_bytes())?;
    Ok(())
}

fn has_v2_prefix(path: &Path) -> Result<bool, WalError> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len() as usize;
    if len == 0 {
        return Ok(false);
    }
    let mut prefix = vec![0u8; len.min(WAL_MAGIC.len())];
    file.read_exact(&mut prefix)?;
    Ok(prefix == WAL_MAGIC[..prefix.len()])
}

fn replay_v2(
    path: &Path,
    expected_dim: Option<usize>,
    f: &mut impl FnMut(WalEntry),
) -> Result<(), WalError> {
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    let original_len = file.metadata()?.len();
    if original_len < WAL_HEADER_LEN {
        return truncate_tail(&mut file, path, 0, original_len);
    }

    let mut magic = [0u8; WAL_MAGIC.len()];
    file.read_exact(&mut magic)?;
    let mut version = [0u8; 4];
    file.read_exact(&mut version)?;
    let version = u32::from_le_bytes(version);
    if magic != WAL_MAGIC {
        return Err(WalError::Corrupt(0));
    }
    if version != WAL_VERSION {
        return Err(WalError::UnsupportedVersion(version));
    }

    let mut last_good = WAL_HEADER_LEN;
    while last_good < original_len {
        let record_start = last_good;
        let mut len = [0u8; 4];
        if let Err(e) = file.read_exact(&mut len) {
            return recover_or_io(e, &mut file, path, record_start, original_len);
        }
        let body_len = u32::from_le_bytes(len) as usize;
        if !(DELETE_BODY_LEN..=MAX_RECORD_BODY_LEN).contains(&body_len) {
            return truncate_tail(&mut file, path, record_start, original_len);
        }
        let record_end = record_start
            .checked_add(RECORD_LEN_LEN)
            .and_then(|n| n.checked_add(body_len as u64))
            .and_then(|n| n.checked_add(RECORD_CRC_LEN))
            .ok_or(WalError::Corrupt(record_start))?;
        if record_end > original_len {
            return truncate_tail(&mut file, path, record_start, original_len);
        }

        let mut body = vec![0u8; body_len];
        if let Err(e) = file.read_exact(&mut body) {
            return recover_or_io(e, &mut file, path, record_start, original_len);
        }
        let mut checksum = [0u8; 4];
        if let Err(e) = file.read_exact(&mut checksum) {
            return recover_or_io(e, &mut file, path, record_start, original_len);
        }
        if crc32fast::hash(&body) != u32::from_le_bytes(checksum) {
            return truncate_tail(&mut file, path, record_start, original_len);
        }

        let entry = match parse_v2_body(&body, expected_dim, record_start) {
            Ok(entry) => entry,
            // A syntactically malformed record, including an out-of-range
            // dimension, is recoverable WAL tail damage. A dimension mismatch
            // is different: it indicates the caller opened this WAL using the
            // wrong index configuration and must not destroy data.
            Err(WalError::Corrupt(_)) | Err(WalError::DimensionTooLarge { .. }) => {
                return truncate_tail(&mut file, path, record_start, original_len);
            }
            Err(error) => return Err(error),
        };
        f(entry);
        last_good = record_end;
    }
    Ok(())
}

fn parse_v2_body(
    body: &[u8],
    expected_dim: Option<usize>,
    offset: u64,
) -> Result<WalEntry, WalError> {
    if body.len() < DELETE_BODY_LEN {
        return Err(WalError::Corrupt(offset));
    }
    let id = u64::from_le_bytes(body[1..9].try_into().expect("fixed-width ID"));
    match body[0] {
        0x01 => {
            if body.len() < INSERT_FIXED_LEN {
                return Err(WalError::Corrupt(offset));
            }
            let dim =
                u32::from_le_bytes(body[9..13].try_into().expect("fixed-width dimension")) as usize;
            validate_dim(dim)?;
            if expected_dim.is_some_and(|expected| expected != dim) {
                return Err(WalError::DimensionMismatch {
                    expected: expected_dim.expect("checked above"),
                    actual: dim,
                });
            }
            let payload_len = dim
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(WalError::Corrupt(offset))?;
            if body.len() != INSERT_FIXED_LEN + payload_len {
                return Err(WalError::Corrupt(offset));
            }
            let vector = body[INSERT_FIXED_LEN..]
                .as_chunks::<{ std::mem::size_of::<f32>() }>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect();
            Ok(WalEntry::Insert { id, vector })
        }
        0x02 if body.len() == DELETE_BODY_LEN => Ok(WalEntry::Delete { id }),
        _ => Err(WalError::Corrupt(offset)),
    }
}

fn recover_or_io(
    error: io::Error,
    file: &mut File,
    path: &Path,
    offset: u64,
    original_len: u64,
) -> Result<(), WalError> {
    if error.kind() == io::ErrorKind::UnexpectedEof {
        truncate_tail(file, path, offset, original_len)
    } else {
        Err(error.into())
    }
}

fn truncate_tail(
    file: &mut File,
    path: &Path,
    offset: u64,
    original_len: u64,
) -> Result<(), WalError> {
    warn!(
        "WAL replay: invalid tail at offset {offset}; truncating WAL from {original_len} to {offset} bytes"
    );
    file.set_len(offset)?;
    file.sync_all()?;
    sync_parent_dir(path)?;
    Ok(())
}

fn replay_v1_and_migrate(
    path: &Path,
    expected_dim: Option<usize>,
    f: &mut impl FnMut(WalEntry),
) -> Result<(), WalError> {
    let input = File::open(path)?;
    let original_len = input.metadata()?.len();
    let mut reader = BufReader::new(input);
    let temp_path = path.with_file_name(format!(
        ".{}.wal-v2-{}-{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("wal"),
        std::process::id(),
        fastrand::u64(..)
    ));
    let temp = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)?;
    let mut output = BufWriter::new(temp);
    write_header(&mut output)?;

    let mut offset = 0u64;
    let mut recovered_tail = false;
    while offset < original_len {
        let record_start = offset;
        let mut tag = [0u8; 1];
        if let Err(e) = reader.read_exact(&mut tag) {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                recovered_tail = true;
                break;
            }
            return Err(e.into());
        }
        offset += 1;
        let entry = match tag[0] {
            0x01 => {
                let mut id = [0u8; 8];
                let mut dim = [0u8; 4];
                if read_legacy_exact(&mut reader, &mut id)?
                    || read_legacy_exact(&mut reader, &mut dim)?
                {
                    recovered_tail = true;
                    break;
                }
                offset += 12;
                let dim = u32::from_le_bytes(dim) as usize;
                if validate_dim(dim).is_err() {
                    recovered_tail = true;
                    break;
                }
                if expected_dim.is_some_and(|expected| expected != dim) {
                    return Err(WalError::DimensionMismatch {
                        expected: expected_dim.expect("checked above"),
                        actual: dim,
                    });
                }
                let payload_len = dim
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or(WalError::Corrupt(record_start))?;
                let mut payload = vec![0u8; payload_len];
                if read_legacy_exact(&mut reader, &mut payload)? {
                    recovered_tail = true;
                    break;
                }
                offset += payload_len as u64;
                let vector = payload
                    .as_chunks::<{ std::mem::size_of::<f32>() }>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_ne_bytes(*bytes))
                    .collect();
                WalEntry::Insert {
                    id: u64::from_le_bytes(id),
                    vector,
                }
            }
            0x02 => {
                let mut id = [0u8; 8];
                if read_legacy_exact(&mut reader, &mut id)? {
                    recovered_tail = true;
                    break;
                }
                offset += 8;
                WalEntry::Delete {
                    id: u64::from_le_bytes(id),
                }
            }
            _ => {
                recovered_tail = true;
                break;
            }
        };
        write_record(&mut output, &entry)?;
        f(entry);
    }

    output.flush()?;
    output.get_ref().sync_all()?;
    drop(output);
    std::fs::rename(&temp_path, path)?;
    sync_parent_dir(path)?;
    if recovered_tail {
        warn!(
            "WAL replay: invalid legacy tail at offset {offset}; migrated valid prefix to checksummed WAL"
        );
    }
    Ok(())
}

/// Returns true when an EOF interrupted this legacy record. Other I/O errors
/// remain fatal rather than being mistaken for crash recovery.
fn read_legacy_exact(reader: &mut impl Read, buffer: &mut [u8]) -> Result<bool, WalError> {
    match reader.read_exact(buffer) {
        Ok(()) => Ok(false),
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(true),
        Err(error) => Err(error.into()),
    }
}

fn sync_parent_dir(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        File::open(parent)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn record_end(bytes: &[u8], start: usize) -> usize {
        let body_len = u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap()) as usize;
        start + 4 + body_len + 4
    }

    fn v1_insert(id: u64, vector: &[f32]) -> Vec<u8> {
        let mut bytes = vec![0x01];
        bytes.extend_from_slice(&id.to_le_bytes());
        bytes.extend_from_slice(&(vector.len() as u32).to_le_bytes());
        for value in vector {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    #[test]
    fn test_wal_roundtrip() {
        let dir = std::env::temp_dir().join("vivvy-wal-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.log");

        {
            let mut wal = WalWriter::open(&path).unwrap();
            wal.append(&WalEntry::Insert {
                id: 1,
                vector: vec![1.0, 2.0, 3.0],
            })
            .unwrap();
            wal.commit().unwrap();
        }

        let mut entries = Vec::new();
        WalWriter::replay(&path, |e| entries.push(e)).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0],
            WalEntry::Insert {
                id: 1,
                vector: vec![1.0, 2.0, 3.0],
            }
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_wal_delete_roundtrip() {
        let dir = std::env::temp_dir().join("vivvy-wal-delete-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.log");

        {
            let mut wal = WalWriter::open(&path).unwrap();
            wal.append(&WalEntry::Insert {
                id: 10,
                vector: vec![4.0, 5.0, 6.0],
            })
            .unwrap();
            wal.append(&WalEntry::Delete { id: 10 }).unwrap();
            wal.commit().unwrap();
        }

        let mut entries = Vec::new();
        WalWriter::replay(&path, |e| entries.push(e)).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0],
            WalEntry::Insert {
                id: 10,
                vector: vec![4.0, 5.0, 6.0],
            }
        );
        assert_eq!(entries[1], WalEntry::Delete { id: 10 });

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn replay_truncates_a_torn_final_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");
        {
            let mut wal = WalWriter::open(&path).unwrap();
            wal.append(&WalEntry::Insert {
                id: 1,
                vector: vec![1.0, 2.0, 3.0],
            })
            .unwrap();
            wal.append(&WalEntry::Delete { id: 1 }).unwrap();
            wal.commit().unwrap();
        }

        let bytes = std::fs::read(&path).unwrap();
        let first_end = record_end(&bytes, WAL_HEADER_LEN as usize);
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len((first_end + 3) as u64)
            .unwrap();

        let mut entries = Vec::new();
        WalWriter::replay(&path, |entry| entries.push(entry)).unwrap();
        assert_eq!(
            entries,
            vec![WalEntry::Insert {
                id: 1,
                vector: vec![1.0, 2.0, 3.0],
            }]
        );
        assert_eq!(std::fs::metadata(&path).unwrap().len(), first_end as u64);
    }

    #[test]
    fn opening_an_existing_wal_recovers_before_appending() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");
        {
            let mut wal = WalWriter::open(&path).unwrap();
            wal.append(&WalEntry::Delete { id: 1 }).unwrap();
            wal.append(&WalEntry::Delete { id: 2 }).unwrap();
            wal.commit().unwrap();
        }
        let bytes = std::fs::read(&path).unwrap();
        let first_end = record_end(&bytes, WAL_HEADER_LEN as usize);
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len((bytes.len() - 1) as u64)
            .unwrap();

        let mut wal = WalWriter::open(&path).unwrap();
        wal.append(&WalEntry::Delete { id: 3 }).unwrap();
        wal.commit().unwrap();
        drop(wal);

        let mut entries = Vec::new();
        WalWriter::replay(&path, |entry| entries.push(entry)).unwrap();
        assert_eq!(
            entries,
            vec![WalEntry::Delete { id: 1 }, WalEntry::Delete { id: 3 }]
        );
        assert!(std::fs::metadata(&path).unwrap().len() > first_end as u64);
    }

    #[test]
    fn replay_truncates_checksum_failure_at_the_prior_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");
        {
            let mut wal = WalWriter::open(&path).unwrap();
            wal.append(&WalEntry::Delete { id: 10 }).unwrap();
            wal.append(&WalEntry::Insert {
                id: 20,
                vector: vec![4.0, 5.0],
            })
            .unwrap();
            wal.commit().unwrap();
        }

        let mut bytes = std::fs::read(&path).unwrap();
        let first_end = record_end(&bytes, WAL_HEADER_LEN as usize);
        bytes[first_end + 4 + INSERT_FIXED_LEN] ^= 0x80;
        std::fs::write(&path, bytes).unwrap();

        let mut entries = Vec::new();
        WalWriter::replay(&path, |entry| entries.push(entry)).unwrap();
        assert_eq!(entries, vec![WalEntry::Delete { id: 10 }]);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), first_end as u64);
    }

    #[test]
    fn replay_rejects_oversized_dimension_without_allocating() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");
        let mut file = File::create(&path).unwrap();
        write_header(&mut file).unwrap();
        let mut body = Vec::new();
        body.push(0x01);
        body.extend_from_slice(&7u64.to_le_bytes());
        body.extend_from_slice(&((MAX_WAL_DIM as u32) + 1).to_le_bytes());
        file.write_all(&(body.len() as u32).to_le_bytes()).unwrap();
        file.write_all(&body).unwrap();
        file.write_all(&crc32fast::hash(&body).to_le_bytes())
            .unwrap();
        file.sync_all().unwrap();

        WalWriter::replay(&path, |_| {}).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), WAL_HEADER_LEN);
    }

    #[test]
    fn legacy_wal_is_migrated_to_checksums() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");
        let mut legacy = v1_insert(42, &[1.5, -2.0]);
        legacy.extend_from_slice(&[0x02]);
        legacy.extend_from_slice(&42u64.to_le_bytes());
        std::fs::write(&path, legacy).unwrap();

        let mut entries = Vec::new();
        WalWriter::replay_with_expected_dim(&path, Some(2), |entry| entries.push(entry)).unwrap();
        assert_eq!(
            entries,
            vec![
                WalEntry::Insert {
                    id: 42,
                    vector: vec![1.5, -2.0],
                },
                WalEntry::Delete { id: 42 },
            ]
        );
        assert!(has_v2_prefix(&path).unwrap());

        let mut reread = Vec::new();
        WalWriter::replay_with_expected_dim(&path, Some(2), |entry| reread.push(entry)).unwrap();
        assert_eq!(reread, entries);
    }

    #[test]
    fn configured_dimension_mismatch_does_not_truncate_wal() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");
        {
            let mut wal = WalWriter::open(&path).unwrap();
            wal.append(&WalEntry::Insert {
                id: 1,
                vector: vec![1.0, 2.0],
            })
            .unwrap();
            wal.commit().unwrap();
        }
        let before = std::fs::metadata(&path).unwrap().len();
        assert!(matches!(
            WalWriter::replay_with_expected_dim(&path, Some(3), |_| {}),
            Err(WalError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        ));
        assert_eq!(std::fs::metadata(&path).unwrap().len(), before);
    }

    #[test]
    fn flush_durability_accepts_an_explicit_sync_barrier() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");
        let mut wal = WalWriter::open_with_durability(&path, WalDurability::Flush).unwrap();
        wal.append(&WalEntry::Delete { id: 9 }).unwrap();
        wal.commit().unwrap();
        wal.sync().unwrap();
        drop(wal);

        let mut entries = Vec::new();
        WalWriter::replay(&path, |entry| entries.push(entry)).unwrap();
        assert_eq!(entries, vec![WalEntry::Delete { id: 9 }]);
    }
}
