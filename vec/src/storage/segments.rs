use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use std::io::Read;
use std::ops::Range;
use std::path::Path;
use thiserror::Error;

const MAGIC: [u8; 8] = *b"VIVVYSEG";
const CURRENT_VERSION: u32 = 1;
const MAX_SEGMENT_BYTES: usize = 1 << 30;
const MAX_NODES_PER_SEGMENT: usize = 16_000_000;
const MAX_DIMS: usize = 1 << 20;
const MAX_LEVEL: usize = 64;
const MAX_NEIGHBORS_PER_LAYER: usize = 65_536;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Header {
    magic: [u8; 8],
    version: u32,
    num_nodes: u32,
    dims: u32,
    m: u32,
    m_max: u32,
    pq_subvectors: u32,
    pq_enabled: u8,
    _reserved: [u8; 31],
}

#[derive(Debug, Error)]
pub enum SegmentError {
    #[error("invalid magic bytes")]
    BadMagic,
    #[error("unsupported version {0}")]
    UnsupportedVersion(u32),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("file too small")]
    Truncated,
    #[error("dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch { expected: usize, found: usize },
    #[error("segment layout arithmetic overflow")]
    ArithmeticOverflow,
    #[error("segment exceeds the maximum supported size")]
    TooLarge,
    #[error("unable to allocate memory for segment")]
    AllocationFailed,
    #[error("invalid segment layout: {0}")]
    InvalidLayout(&'static str),
    #[error("segment limit exceeded: {0}")]
    LimitExceeded(&'static str),
    #[error("invalid byte alignment in segment payload")]
    InvalidAlignment,
}

#[derive(Debug, Clone)]
pub struct NodeRecord {
    pub id: u64,
    pub level: u32,
    pub neighbors: Vec<Vec<u32>>,
    pub pq_code: Option<Vec<u8>>,
    pub vector: Option<Vec<f32>>,
}

pub struct SealedSegment {
    // Segments are read into owned immutable memory before validation. A memory map can
    // SIGBUS if another process truncates its backing file after it has been validated.
    bytes: Box<[u8]>,
    offset_table_off: usize,
    data_off: usize,
    num_nodes: usize,
    dims: usize,
    #[allow(dead_code)]
    m: usize,
    #[allow(dead_code)]
    m_max: usize,
    pq_enabled: bool,
    pq_subvectors: usize,
    has_codebook: bool,
    codebook_off: usize,
}

impl SealedSegment {
    /// Open a sealed segment from a file path.
    ///
    /// # Errors
    /// Returns `SegmentError` if the file cannot be opened or if the format is invalid.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SegmentError> {
        let file = std::fs::File::open(path.as_ref())?;
        Self::from_file(file)
    }

    /// Open a sealed segment and require its stored dimensions to match the index.
    pub fn open_with_dims(
        path: impl AsRef<Path>,
        expected_dims: usize,
    ) -> Result<Self, SegmentError> {
        let segment = Self::open(path)?;
        segment.validate_dims(expected_dims)?;
        Ok(segment)
    }

    /// Open a sealed segment from an open file handle.
    ///
    /// # Errors
    /// Returns `SegmentError` if mapping or validation fails.
    pub fn from_file(mut file: std::fs::File) -> Result<Self, SegmentError> {
        let expected_len = usize::try_from(file.metadata()?.len())
            .map_err(|_| SegmentError::ArithmeticOverflow)?;
        if expected_len > MAX_SEGMENT_BYTES {
            return Err(SegmentError::TooLarge);
        }
        let read_limit = u64::try_from(checked_add(expected_len, 1)?)
            .map_err(|_| SegmentError::ArithmeticOverflow)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(expected_len)
            .map_err(|_| SegmentError::AllocationFailed)?;
        file.by_ref().take(read_limit).read_to_end(&mut bytes)?;
        if bytes.len() != expected_len {
            return Err(SegmentError::InvalidLayout("file changed while being read"));
        }
        Self::from_bytes(bytes)
    }

    /// Build a validated segment from an immutable byte image.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, SegmentError> {
        if bytes.len() > MAX_SEGMENT_BYTES {
            return Err(SegmentError::TooLarge);
        }
        if bytes.len() < size_of::<Header>() {
            return Err(SegmentError::Truncated);
        }

        let h: Header = bytemuck::pod_read_unaligned(&bytes[..size_of::<Header>()]);
        if h.magic != MAGIC {
            return Err(SegmentError::BadMagic);
        }
        if h.version != CURRENT_VERSION {
            return Err(SegmentError::UnsupportedVersion(h.version));
        }
        if h.pq_enabled > 1 {
            return Err(SegmentError::InvalidLayout("invalid PQ flag"));
        }

        let num_nodes =
            usize::try_from(h.num_nodes).map_err(|_| SegmentError::ArithmeticOverflow)?;
        let dims = usize::try_from(h.dims).map_err(|_| SegmentError::ArithmeticOverflow)?;
        let m = usize::try_from(h.m).map_err(|_| SegmentError::ArithmeticOverflow)?;
        let m_max = usize::try_from(h.m_max).map_err(|_| SegmentError::ArithmeticOverflow)?;
        let pq_subvectors =
            usize::try_from(h.pq_subvectors).map_err(|_| SegmentError::ArithmeticOverflow)?;
        let pq_enabled = h.pq_enabled == 1;

        if num_nodes > MAX_NODES_PER_SEGMENT {
            return Err(SegmentError::LimitExceeded("nodes"));
        }
        if dims == 0 || dims > MAX_DIMS {
            return Err(SegmentError::LimitExceeded("dimensions"));
        }
        if m == 0 || m_max < m || m_max > MAX_NEIGHBORS_PER_LAYER {
            return Err(SegmentError::InvalidLayout("invalid HNSW degree"));
        }
        if pq_enabled && pq_subvectors == 0 {
            return Err(SegmentError::InvalidLayout("PQ enabled without subvectors"));
        }

        let offset_table_off = size_of::<Header>();
        let offset_table_len = checked_mul(num_nodes, size_of::<u64>())?;
        let codebook_off = checked_add(offset_table_off, offset_table_len)?;
        ensure_range(codebook_off, 0, bytes.len())?;

        let codebook_len = if pq_enabled {
            checked_mul(checked_mul(pq_subvectors, 256)?, size_of::<f32>())?
        } else {
            0
        };
        let data_off = checked_add(codebook_off, codebook_len)?;
        ensure_range(data_off, 0, bytes.len())?;

        let segment = Self {
            bytes: bytes.into_boxed_slice(),
            offset_table_off,
            data_off,
            num_nodes,
            dims,
            m,
            m_max,
            pq_enabled,
            pq_subvectors,
            has_codebook: pq_enabled,
            codebook_off,
        };
        segment.validate_layout()?;
        Ok(segment)
    }

    /// Construct a segment by copying a stable memory map into owned memory.
    ///
    /// # Safety
    /// The mapping must remain backed by a stable file for the duration of this call.
    /// Concurrent truncation can SIGBUS while copying any `Mmap`; callers that do not
    /// control the backing file must use [`Self::open`] or [`Self::from_file`] instead.
    pub unsafe fn from_mmap(mmap: Mmap) -> Result<Self, SegmentError> {
        Self::from_bytes(mmap[..].to_vec())
    }

    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    #[must_use]
    pub fn dims(&self) -> usize {
        self.dims
    }

    #[must_use]
    pub fn pq_enabled(&self) -> bool {
        self.pq_enabled
    }

    pub fn validate_dims(&self, expected_dims: usize) -> Result<(), SegmentError> {
        if self.dims != expected_dims {
            return Err(SegmentError::DimensionMismatch {
                expected: expected_dims,
                found: self.dims,
            });
        }
        Ok(())
    }

    fn offset_of(&self, idx: usize) -> Result<Range<usize>, SegmentError> {
        if idx >= self.num_nodes {
            return Err(SegmentError::Truncated);
        }
        let start = self.node_start(idx)?;
        let end = if idx + 1 < self.num_nodes {
            self.node_start(idx + 1)?
        } else {
            self.bytes.len()
        };
        if start >= end || end > self.bytes.len() {
            return Err(SegmentError::InvalidLayout("invalid node offset range"));
        }
        Ok(start..end)
    }

    #[must_use]
    pub fn codebook(&self) -> Option<&[f32]> {
        if !self.has_codebook {
            return None;
        }
        let len = self
            .pq_subvectors
            .checked_mul(256)?
            .checked_mul(size_of::<f32>())?;
        let end = self.codebook_off.checked_add(len)?;
        bytemuck::try_cast_slice(&self.bytes[self.codebook_off..end]).ok()
    }

    pub fn read_node(&self, idx: usize) -> Result<NodeRecord, SegmentError> {
        let range = self.offset_of(idx)?;
        let buf = &self.bytes[range];
        let mut cursor = Cursor::new(buf);
        let id = cursor.read_u64()?;
        let level =
            usize::try_from(cursor.read_u32()?).map_err(|_| SegmentError::ArithmeticOverflow)?;
        let layer_count = checked_add(level, 1)?;
        if level > MAX_LEVEL {
            return Err(SegmentError::LimitExceeded("node level"));
        }

        let mut neighbors = Vec::with_capacity(layer_count);
        for layer_idx in 0..layer_count {
            let n = usize::try_from(cursor.read_u32()?)
                .map_err(|_| SegmentError::ArithmeticOverflow)?;
            self.validate_neighbor_count(layer_idx, n)?;
            let mut layer = Vec::with_capacity(n);
            for _ in 0..n {
                layer.push(cursor.read_u32()?);
            }
            neighbors.push(layer);
        }

        let (pq_code, vector) = if self.pq_enabled {
            (Some(cursor.take(self.pq_subvectors)?.to_vec()), None)
        } else {
            let vbytes = cursor.take(self.vector_byte_len()?)?;
            let v = bytemuck::try_cast_slice(vbytes)
                .map_err(|_| SegmentError::InvalidAlignment)?
                .to_vec();
            (None, Some(v))
        };
        if !cursor.is_empty() {
            return Err(SegmentError::InvalidLayout("trailing node bytes"));
        }

        Ok(NodeRecord {
            id,
            level: level as u32,
            neighbors,
            pq_code,
            vector,
        })
    }

    pub fn read_pq_code(&self, idx: usize) -> Result<&[u8], SegmentError> {
        if !self.pq_enabled {
            return Err(SegmentError::InvalidLayout(
                "PQ code requested from raw-vector segment",
            ));
        }
        let (buf, payload_range) = self.node_payload(idx)?;
        Ok(&buf[payload_range])
    }

    pub fn id_at(&self, idx: usize) -> Result<u64, SegmentError> {
        let range = self.offset_of(idx)?;
        let mut cursor = Cursor::new(&self.bytes[range]);
        cursor.read_u64()
    }

    pub fn vector_at(&self, idx: usize) -> Result<&[f32], SegmentError> {
        if self.pq_enabled {
            return Err(SegmentError::InvalidLayout(
                "vector requested from PQ segment",
            ));
        }
        let (buf, payload_range) = self.node_payload(idx)?;
        bytemuck::try_cast_slice(&buf[payload_range]).map_err(|_| SegmentError::InvalidAlignment)
    }

    fn validate_layout(&self) -> Result<(), SegmentError> {
        if self.pq_enabled {
            let codebook_len =
                checked_mul(checked_mul(self.pq_subvectors, 256)?, size_of::<f32>())?;
            ensure_range(self.codebook_off, codebook_len, self.bytes.len())?;
            bytemuck::try_cast_slice::<u8, f32>(&self.bytes[self.codebook_off..self.data_off])
                .map_err(|_| SegmentError::InvalidAlignment)?;
        }

        let mut previous = None;
        for idx in 0..self.num_nodes {
            let start = self.node_start(idx)?;
            if idx == 0 && start != self.data_off {
                return Err(SegmentError::InvalidLayout(
                    "first node does not start at data offset",
                ));
            }
            if let Some(prev) = previous {
                if start <= prev {
                    return Err(SegmentError::InvalidLayout(
                        "node offsets are not strictly monotonic",
                    ));
                }
            }
            previous = Some(start);
        }

        for idx in 0..self.num_nodes {
            let range = self.offset_of(idx)?;
            self.validate_node_record(&self.bytes[range])?;
        }
        Ok(())
    }

    fn node_start(&self, idx: usize) -> Result<usize, SegmentError> {
        let table_entry = checked_add(self.offset_table_off, checked_mul(idx, size_of::<u64>())?)?;
        ensure_range(table_entry, size_of::<u64>(), self.bytes.len())?;
        let raw: [u8; 8] = self.bytes[table_entry..table_entry + size_of::<u64>()]
            .try_into()
            .map_err(|_| SegmentError::Truncated)?;
        let relative = usize::try_from(u64::from_le_bytes(raw))
            .map_err(|_| SegmentError::ArithmeticOverflow)?;
        let start = checked_add(self.data_off, relative)?;
        if start < self.data_off || start >= self.bytes.len() {
            return Err(SegmentError::Truncated);
        }
        Ok(start)
    }

    fn validate_node_record(&self, buf: &[u8]) -> Result<(), SegmentError> {
        let mut cursor = Cursor::new(buf);
        cursor.read_u64()?;
        let level =
            usize::try_from(cursor.read_u32()?).map_err(|_| SegmentError::ArithmeticOverflow)?;
        if level > MAX_LEVEL {
            return Err(SegmentError::LimitExceeded("node level"));
        }
        let layer_count = checked_add(level, 1)?;
        for layer_idx in 0..layer_count {
            let count = usize::try_from(cursor.read_u32()?)
                .map_err(|_| SegmentError::ArithmeticOverflow)?;
            self.validate_neighbor_count(layer_idx, count)?;
            cursor.skip(checked_mul(count, size_of::<u32>())?)?;
        }
        let payload_len = if self.pq_enabled {
            self.pq_subvectors
        } else {
            self.vector_byte_len()?
        };
        cursor.take(payload_len)?;
        if !cursor.is_empty() {
            return Err(SegmentError::InvalidLayout("trailing node bytes"));
        }
        Ok(())
    }

    fn node_payload(&self, idx: usize) -> Result<(&[u8], Range<usize>), SegmentError> {
        let range = self.offset_of(idx)?;
        let buf = &self.bytes[range];
        let mut cursor = Cursor::new(buf);
        cursor.read_u64()?;
        let level =
            usize::try_from(cursor.read_u32()?).map_err(|_| SegmentError::ArithmeticOverflow)?;
        if level > MAX_LEVEL {
            return Err(SegmentError::LimitExceeded("node level"));
        }
        for layer_idx in 0..checked_add(level, 1)? {
            let count = usize::try_from(cursor.read_u32()?)
                .map_err(|_| SegmentError::ArithmeticOverflow)?;
            self.validate_neighbor_count(layer_idx, count)?;
            cursor.skip(checked_mul(count, size_of::<u32>())?)?;
        }
        let payload_len = if self.pq_enabled {
            self.pq_subvectors
        } else {
            self.vector_byte_len()?
        };
        let start = cursor.position();
        cursor.take(payload_len)?;
        if !cursor.is_empty() {
            return Err(SegmentError::InvalidLayout("trailing node bytes"));
        }
        Ok((buf, start..checked_add(start, payload_len)?))
    }

    fn validate_neighbor_count(&self, layer_idx: usize, count: usize) -> Result<(), SegmentError> {
        let max_degree = if layer_idx == 0 { self.m_max } else { self.m };
        if count > max_degree || count > MAX_NEIGHBORS_PER_LAYER {
            return Err(SegmentError::LimitExceeded("neighbors per layer"));
        }
        Ok(())
    }

    fn vector_byte_len(&self) -> Result<usize, SegmentError> {
        checked_mul(self.dims, size_of::<f32>())
    }
}

fn checked_add(lhs: usize, rhs: usize) -> Result<usize, SegmentError> {
    lhs.checked_add(rhs).ok_or(SegmentError::ArithmeticOverflow)
}

fn checked_mul(lhs: usize, rhs: usize) -> Result<usize, SegmentError> {
    lhs.checked_mul(rhs).ok_or(SegmentError::ArithmeticOverflow)
}

fn ensure_range(start: usize, len: usize, total: usize) -> Result<(), SegmentError> {
    let end = checked_add(start, len)?;
    if end > total {
        return Err(SegmentError::Truncated);
    }
    Ok(())
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn is_empty(&self) -> bool {
        self.pos == self.bytes.len()
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], SegmentError> {
        let end = checked_add(self.pos, len)?;
        if end > self.bytes.len() {
            return Err(SegmentError::Truncated);
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn skip(&mut self, len: usize) -> Result<(), SegmentError> {
        self.take(len).map(|_| ())
    }

    fn read_u32(&mut self) -> Result<u32, SegmentError> {
        let bytes: [u8; 4] = self
            .take(size_of::<u32>())?
            .try_into()
            .map_err(|_| SegmentError::Truncated)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, SegmentError> {
        let bytes: [u8; 8] = self
            .take(size_of::<u64>())?
            .try_into()
            .map_err(|_| SegmentError::Truncated)?;
        Ok(u64::from_le_bytes(bytes))
    }
}

#[allow(clippy::type_complexity)]
pub struct SegmentWriter<W: std::io::Write + std::io::Seek> {
    inner: W,
    header: Header,
    nodes: Vec<(u64, u32, Vec<Vec<u32>>, Vec<f32>)>,
}

impl<W: std::io::Write + std::io::Seek> SegmentWriter<W> {
    pub fn new(writer: W, dims: u32, m: u32, m_max: u32) -> Self {
        let header = Header {
            magic: MAGIC,
            version: CURRENT_VERSION,
            num_nodes: 0,
            dims,
            m,
            m_max,
            pq_subvectors: 0,
            pq_enabled: 0,
            _reserved: [0u8; 31],
        };
        Self {
            inner: writer,
            header,
            nodes: Vec::new(),
        }
    }

    pub fn push(
        &mut self,
        id: u64,
        level: u32,
        neighbors: Vec<Vec<u32>>,
        vector: Vec<f32>,
    ) -> Result<(), SegmentError> {
        if vector.len() != self.header.dims as usize {
            return Err(SegmentError::DimensionMismatch {
                expected: self.header.dims as usize,
                found: vector.len(),
            });
        }
        let layer_count = usize::try_from(level)
            .map_err(|_| SegmentError::ArithmeticOverflow)?
            .checked_add(1)
            .ok_or(SegmentError::ArithmeticOverflow)?;
        if level as usize > MAX_LEVEL {
            return Err(SegmentError::LimitExceeded("node level"));
        }
        if neighbors.len() != layer_count {
            return Err(SegmentError::InvalidLayout(
                "neighbor layer count does not match level",
            ));
        }
        for (layer_idx, layer) in neighbors.iter().enumerate() {
            let max_degree = if layer_idx == 0 {
                self.header.m_max as usize
            } else {
                self.header.m as usize
            };
            if layer.len() > max_degree || layer.len() > MAX_NEIGHBORS_PER_LAYER {
                return Err(SegmentError::LimitExceeded("neighbors per layer"));
            }
            u32::try_from(layer.len())
                .map_err(|_| SegmentError::LimitExceeded("neighbors per layer"))?;
        }
        self.nodes.push((id, level, neighbors, vector));
        Ok(())
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn write(mut self) -> Result<(), SegmentError> {
        let dims = self.header.dims as usize;
        let m = self.header.m as usize;
        let m_max = self.header.m_max as usize;
        if dims == 0 || dims > MAX_DIMS {
            return Err(SegmentError::LimitExceeded("dimensions"));
        }
        if m == 0 || m_max < m || m_max > MAX_NEIGHBORS_PER_LAYER {
            return Err(SegmentError::InvalidLayout("invalid HNSW degree"));
        }
        self.header.num_nodes =
            u32::try_from(self.nodes.len()).map_err(|_| SegmentError::LimitExceeded("nodes"))?;
        if self.nodes.len() > MAX_NODES_PER_SEGMENT {
            return Err(SegmentError::LimitExceeded("nodes"));
        }
        let hdr_bytes: &[u8] = bytemuck::bytes_of(&self.header);
        self.inner.write_all(hdr_bytes)?;

        let offset_pos = self.inner.stream_position()?;
        let num_nodes = self.nodes.len();
        for _ in 0..num_nodes {
            self.inner.write_all(&[0u8; 8])?;
        }

        let node_data_start = self.inner.stream_position()?;
        let mut offsets = Vec::with_capacity(num_nodes);
        for (id, level, neighbors, vector) in &self.nodes {
            let off = self.inner.stream_position()? - node_data_start;
            offsets.push(off);
            self.inner.write_all(&id.to_le_bytes())?;
            self.inner.write_all(&level.to_le_bytes())?;
            for layer in neighbors {
                let layer_len = u32::try_from(layer.len())
                    .map_err(|_| SegmentError::LimitExceeded("neighbors per layer"))?;
                self.inner.write_all(&layer_len.to_le_bytes())?;
                for &n in layer {
                    self.inner.write_all(&n.to_le_bytes())?;
                }
            }
            let vbytes: &[u8] = bytemuck::cast_slice(vector.as_slice());
            let vector_bytes = checked_mul(self.header.dims as usize, size_of::<f32>())?;
            self.inner.write_all(&vbytes[..vector_bytes])?;
        }

        let end_pos = self.inner.stream_position()?;
        self.inner.seek(std::io::SeekFrom::Start(offset_pos))?;
        for off in offsets {
            self.inner.write_all(&off.to_le_bytes())?;
        }
        self.inner.seek(std::io::SeekFrom::Start(end_pos))?;
        self.inner.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufWriter, Read, Seek, SeekFrom};
    use tempfile::tempfile;

    fn segment_bytes(dims: u32, nodes: usize) -> Vec<u8> {
        let mut file = tempfile().unwrap();
        {
            let writer = BufWriter::new(file.try_clone().unwrap());
            let mut seg_w = SegmentWriter::new(writer, dims, 16, 32);
            for idx in 0..nodes {
                seg_w
                    .push(idx as u64, 0, vec![Vec::new()], vec![1.0; dims as usize])
                    .unwrap();
            }
            seg_w.write().unwrap();
        }
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        bytes
    }

    fn write_offset(bytes: &mut [u8], idx: usize, value: u64) {
        let start = size_of::<Header>() + idx * size_of::<u64>();
        bytes[start..start + size_of::<u64>()].copy_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn test_roundtrip() {
        let mut file = tempfile().unwrap();
        {
            let writer = BufWriter::new(file.try_clone().unwrap());
            let mut seg_w = SegmentWriter::new(writer, 4, 16, 32);
            seg_w
                .push(42, 1, vec![vec![1, 2], vec![3]], vec![1.0, 2.0, 3.0, 4.0])
                .unwrap();
            seg_w
                .push(7, 0, vec![vec![0]], vec![5.0, 6.0, 7.0, 8.0])
                .unwrap();
            seg_w.write().unwrap();
        }
        file.seek(SeekFrom::Start(0)).unwrap();

        let seg = SealedSegment::from_file(file).unwrap();
        assert_eq!(seg.num_nodes(), 2);
        assert_eq!(seg.dims(), 4);

        let n0 = seg.read_node(0).unwrap();
        assert_eq!(n0.id, 42);
        assert_eq!(n0.level, 1);
        assert_eq!(n0.neighbors.len(), 2);
        assert_eq!(n0.neighbors[0], vec![1, 2]);
        assert_eq!(n0.vector.as_deref(), Some(&[1.0, 2.0, 3.0, 4.0][..]));

        let n1 = seg.read_node(1).unwrap();
        assert_eq!(n1.id, 7);
        assert_eq!(n1.level, 0);
    }

    #[test]
    fn rejects_out_of_bounds_or_non_monotonic_offsets() {
        let mut out_of_bounds = segment_bytes(4, 1);
        write_offset(&mut out_of_bounds, 0, u64::MAX);
        assert!(matches!(
            SealedSegment::from_bytes(out_of_bounds),
            Err(SegmentError::ArithmeticOverflow | SegmentError::Truncated)
        ));

        let mut non_monotonic = segment_bytes(4, 2);
        write_offset(&mut non_monotonic, 1, 0);
        assert!(matches!(
            SealedSegment::from_bytes(non_monotonic),
            Err(SegmentError::InvalidLayout(_))
        ));
    }

    #[test]
    fn rejects_untrusted_level_and_neighbor_counts_before_allocation() {
        let data_offset = size_of::<Header>() + size_of::<u64>();

        let mut huge_level = segment_bytes(4, 1);
        huge_level[data_offset + size_of::<u64>()..data_offset + 12]
            .copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            SealedSegment::from_bytes(huge_level),
            Err(SegmentError::LimitExceeded("node level"))
        ));

        let mut huge_neighbor_count = segment_bytes(4, 1);
        huge_neighbor_count[data_offset + 12..data_offset + 16]
            .copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            SealedSegment::from_bytes(huge_neighbor_count),
            Err(SegmentError::LimitExceeded("neighbors per layer"))
        ));
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let segment = SealedSegment::from_bytes(segment_bytes(4, 1)).unwrap();
        assert!(matches!(
            segment.validate_dims(3),
            Err(SegmentError::DimensionMismatch {
                expected: 3,
                found: 4
            })
        ));
    }

    #[test]
    fn owned_segment_remains_safe_after_backing_file_is_truncated() {
        let mut file = tempfile().unwrap();
        {
            let writer = BufWriter::new(file.try_clone().unwrap());
            let mut seg_w = SegmentWriter::new(writer, 3, 16, 32);
            seg_w
                .push(7, 0, vec![Vec::new()], vec![1.0, 2.0, 3.0])
                .unwrap();
            seg_w.write().unwrap();
        }
        file.seek(SeekFrom::Start(0)).unwrap();
        let segment = SealedSegment::from_file(file.try_clone().unwrap()).unwrap();
        file.set_len(0).unwrap();

        assert_eq!(segment.id_at(0).unwrap(), 7);
        assert_eq!(segment.vector_at(0).unwrap(), &[1.0, 2.0, 3.0]);
    }
}
