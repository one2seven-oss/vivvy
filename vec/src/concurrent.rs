use crate::distance::{self, Metric};
use crate::filter::{FilterExpr, FilterIndex};
use crate::hnsw::HnswIndex;
use crate::storage::manifest::Manifest;
use crate::storage::segments::{SealedSegment, SegmentWriter};
use crate::storage::wal::{WalEntry, WalError, WalWriter};
use arc_swap::ArcSwap;
use log::{info, warn};
use parking_lot::{Mutex, RwLock};
use roaring::RoaringTreemap;
use std::collections::HashSet;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::error::VivvyError;

pub const NUM_SHARDS: usize = 8;

#[inline]
fn shard_idx(id: u64, num_shards: usize) -> usize {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    id.hash(&mut hasher);
    (hasher.finish() as usize) % num_shards
}

/// Thread-safe vector index (mutable delta segment + atomically-swappable sealed list)
pub struct VivvyIndex {
    dims: usize,
    pub(crate) shards: Arc<Vec<RwLock<HnswIndex>>>,
    sealed: Arc<ArcSwap<Vec<Arc<SealedSegment>>>>,
    pub(crate) filter_index: RwLock<FilterIndex>,
    wal: Option<Arc<Mutex<WalWriter>>>,
    running: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
    pub(crate) next_id: AtomicU64,
    data_dir: Option<PathBuf>,
    tombstoned_ids: Arc<RwLock<RoaringTreemap>>,
    // Serializes mutations with compaction so a WAL entry cannot be lost while
    // the compactor snapshots the delta and resets the WAL.
    mutation_lock: Arc<Mutex<()>>,
}

impl VivvyIndex {
    pub fn new(
        dims: usize,
        metric: Metric,
        wal_path: Option<impl AsRef<Path>>,
        data_dir: Option<impl AsRef<Path>>,
    ) -> Result<Self, WalError> {
        let shards = (0..NUM_SHARDS)
            .map(|_| RwLock::new(HnswIndex::new(dims, metric)))
            .collect::<Vec<_>>();
        let shards = Arc::new(shards);

        let mut next_id = 1u64;
        let tombstoned_ids = Arc::new(RwLock::new(RoaringTreemap::new()));

        //  Replay uncompacted WAL entries if existing WAL is present
        if let Some(ref path) = wal_path {
            WalWriter::replay(path.as_ref(), |entry| match entry {
                WalEntry::Insert { id, vector } => {
                    tombstoned_ids.write().remove(id);
                    let shard = shard_idx(id, NUM_SHARDS);
                    shards[shard].write().insert(id, vector);
                    if id >= next_id {
                        next_id = id.saturating_add(1);
                    }
                }
                WalEntry::Delete { id } => {
                    tombstoned_ids.write().insert(id);
                    let shard = shard_idx(id, NUM_SHARDS);
                    shards[shard].write().delete(id);
                }
            })?;
        }

        let wal = wal_path
            .as_ref()
            .map(WalWriter::open)
            .transpose()?
            .map(|w| Arc::new(Mutex::new(w)));
        let data_dir = data_dir.map(|p| p.as_ref().to_path_buf());

        //  Discover existing sealed segments in data directory using manifest
        let mut initial_sealed = Vec::new();
        if let Some(ref dir) = data_dir {
            if dir.exists() {
                let manifest = Manifest::load(dir).unwrap_or(None);
                let seg_files: Vec<String> = match manifest {
                    Some(m) => m.segments,
                    None => {
                        let mut found = Vec::new();
                        if let Ok(entries) = std::fs::read_dir(dir) {
                            for entry in entries.flatten() {
                                let path = entry.path();
                                if path.extension().and_then(|s| s.to_str()) == Some("vivvy") {
                                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                                        found.push(name.to_string());
                                    }
                                }
                            }
                        }
                        found.sort();
                        if !found.is_empty() {
                            let m = Manifest::new(found.clone());
                            let _ = m.save(dir);
                        }
                        found
                    }
                };

                for fname in seg_files {
                    let path = dir.join(&fname);
                    match SealedSegment::open(&path) {
                        Ok(seg) => {
                            for idx in 0..seg.num_nodes() {
                                if let Ok(id) = seg.id_at(idx) {
                                    if id >= next_id {
                                        next_id = id.saturating_add(1);
                                    }
                                }
                            }
                            initial_sealed.push(Arc::new(seg));
                        }
                        Err(e) => {
                            warn!("Failed to open existing sealed segment {:?}: {:?}", path, e);
                        }
                    }
                }
            }
        }

        let sealed: Arc<ArcSwap<Vec<Arc<SealedSegment>>>> =
            Arc::new(ArcSwap::new(Arc::new(initial_sealed)));

        let mut idx = Self {
            dims,
            shards: shards.clone(),
            sealed: sealed.clone(),
            filter_index: RwLock::new(FilterIndex::new()),
            wal: wal.clone(),
            running: Arc::new(AtomicBool::new(true)),
            handle: None,
            next_id: AtomicU64::new(next_id),
            data_dir,
            tombstoned_ids: tombstoned_ids.clone(),
            mutation_lock: Arc::new(Mutex::new(())),
        };

        let params = CompactorParams {
            running: idx.running.clone(),
            dims,
            shards,
            sealed,
            data_dir: idx.data_dir.clone(),
            wal,
            tombstoned_ids,
            mutation_lock: idx.mutation_lock.clone(),
        };

        let h = thread::Builder::new()
            .name("vivvy-compactor".into())
            .spawn(move || compactor_loop(params))
            .expect("compactor thread");
        idx.handle = Some(h);

        Ok(idx)
    }

    /// Insert with auto-generated ID
    pub fn insert(&self, vector: Vec<f32>) -> Result<u64, VivvyError> {
        if vector.len() != self.dims {
            return Err(VivvyError::DimensionMismatch);
        }
        let _mutation_guard = self.mutation_lock.lock();
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        self.tombstoned_ids.write().remove(id);

        if let Some(ref wal_mutex) = self.wal {
            let mut wal = wal_mutex.lock();
            wal.append(&WalEntry::Insert {
                id,
                vector: vector.clone(),
            })?;
            wal.commit()?;
        }

        let shard = shard_idx(id, NUM_SHARDS);
        self.shards[shard].write().insert(id, vector);
        Ok(id)
    }

    /// Insert with explicit ID, replacing any previous node with this ID (upsert semantics).
    pub fn insert_with_id(&self, id: u64, vector: Vec<f32>) -> Result<(), VivvyError> {
        if vector.len() != self.dims {
            return Err(VivvyError::DimensionMismatch);
        }
        let _mutation_guard = self.mutation_lock.lock();
        self.next_id
            .fetch_max(id.saturating_add(1), Ordering::SeqCst);

        // Clear prior tombstone so the newly inserted vector is active
        self.tombstoned_ids.write().remove(id);

        if let Some(ref wal_mutex) = self.wal {
            let mut wal = wal_mutex.lock();
            wal.append(&WalEntry::Insert {
                id,
                vector: vector.clone(),
            })?;
            wal.commit()?;
        }
        let shard = shard_idx(id, NUM_SHARDS);
        self.shards[shard].write().insert(id, vector);
        Ok(())
    }

    /// Soft-delete an ID from the vector index.
    ///
    /// Records a delete entry in the WAL (if enabled), records the ID in `tombstoned_ids`
    /// (to mask older instances in sealed segments), and marks the node deleted in its
    /// corresponding shard's HNSW graph.
    pub fn delete(&self, id: u64) -> Result<bool, VivvyError> {
        let _mutation_guard = self.mutation_lock.lock();
        if let Some(ref wal_mutex) = self.wal {
            let mut wal = wal_mutex.lock();
            wal.append(&WalEntry::Delete { id })?;
            wal.commit()?;
        }
        self.tombstoned_ids.write().insert(id);

        let shard = shard_idx(id, NUM_SHARDS);
        let deleted = self.shards[shard].write().delete(id);
        Ok(deleted)
    }

    /// Insert with metadata fields for filtering
    pub fn insert_with_metadata(
        &self,
        vector: Vec<f32>,
        metadata: Vec<(String, String)>,
    ) -> Result<u64, VivvyError> {
        if vector.len() != self.dims {
            return Err(VivvyError::DimensionMismatch);
        }
        let _mutation_guard = self.mutation_lock.lock();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.tombstoned_ids.write().remove(id);

        if let Some(ref wal_mutex) = self.wal {
            let mut wal = wal_mutex.lock();
            wal.append(&WalEntry::Insert {
                id,
                vector: vector.clone(),
            })?;
            wal.commit()?;
        }

        {
            let mut fi = self.filter_index.write();
            for (field, value) in &metadata {
                fi.insert(id, field, value);
            }
        }

        let shard = shard_idx(id, NUM_SHARDS);
        self.shards[shard].write().insert(id, vector);
        Ok(id)
    }

    /// Batch insert with auto-generated IDs
    pub fn insert_batch(&self, vectors: Vec<Vec<f32>>) -> Result<Vec<u64>, VivvyError> {
        self.insert_batch_with_metadata(vectors, None)
    }

    /// Batch insert with metadata fields
    pub fn insert_batch_with_metadata(
        &self,
        vectors: Vec<Vec<f32>>,
        metadata: Option<Vec<Vec<(String, String)>>>,
    ) -> Result<Vec<u64>, VivvyError> {
        let _mutation_guard = self.mutation_lock.lock();
        let mut ids = Vec::with_capacity(vectors.len());
        for _ in 0..vectors.len() {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            ids.push(id);
        }

        for v in &vectors {
            if v.len() != self.dims {
                return Err(VivvyError::DimensionMismatch);
            }
        }

        {
            let mut tombstones = self.tombstoned_ids.write();
            for id in &ids {
                tombstones.remove(*id);
            }
        }

        if let Some(ref wal_mutex) = self.wal {
            let mut wal = wal_mutex.lock();
            for (id, vector) in ids.iter().zip(vectors.iter()) {
                wal.append(&WalEntry::Insert {
                    id: *id,
                    vector: vector.clone(),
                })?;
            }
            wal.commit()?;
        }

        if let Some(ref metas) = metadata {
            let mut fi = self.filter_index.write();
            for (id, meta) in ids.iter().zip(metas.iter()) {
                for (field, value) in meta {
                    fi.insert(*id, field, value);
                }
            }
        }

        for (id, vector) in ids.iter().zip(vectors) {
            let shard = shard_idx(*id, NUM_SHARDS);
            self.shards[shard].write().insert(*id, vector);
        }

        Ok(ids)
    }

    /// Search with an optional filter expression
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&FilterExpr>,
    ) -> Result<Vec<(u64, f32)>, VivvyError> {
        if query.len() != self.dims {
            return Err(VivvyError::DimensionMismatch);
        }
        let (filter_bitmap, has_filter) = match filter {
            Some(expr) => {
                let fi = self.filter_index.read();
                let bitmap = fi.evaluate(expr);
                let empty = bitmap.is_empty();
                (if empty { None } else { Some(bitmap) }, true)
            }
            None => (None, false),
        };

        // If a filter was provided but no items match, return empty
        if has_filter && filter_bitmap.is_none() {
            return Ok(Vec::new());
        }

        // An ID in the mutable delta is newer than every sealed instance of that
        // ID. Populate this from *all* live delta nodes, not just the delta
        // search hits: a moved vector can be outside the delta top-k for a query
        // near its old position, and must still mask that stale sealed vector.
        let shard_guards: Vec<_> = self.shards.iter().map(|shard| shard.read()).collect();
        let mut seen_ids: HashSet<u64> = shard_guards
            .iter()
            .flat_map(|shard| shard.active_ids())
            .collect();
        let (mut results, metric) = {
            let mut results = Vec::new();
            let metric = shard_guards[0].metric();
            for shard in &shard_guards {
                results.extend(shard.search_filtered(query, k, filter_bitmap.as_ref()));
            }
            (results, metric)
        };

        // Search sealed segments newest to oldest. Record an ID as soon as its
        // newest sealed copy is encountered, even when that vector is not a
        // candidate. This prevents an older copy from becoming searchable simply
        // because the newer copy was outside the requested top-k.
        let tombstones = self.tombstoned_ids.read();
        let sealed_list = self.sealed.load();
        for seg in sealed_list.iter().rev() {
            for idx in 0..seg.num_nodes() {
                if let Ok(id) = seg.id_at(idx) {
                    if tombstones.contains(id) || !seen_ids.insert(id) {
                        continue;
                    }
                    let passes_filter = filter_bitmap.as_ref().is_none_or(|bm| bm.contains(id));
                    if !passes_filter {
                        continue;
                    }
                    if let Ok(v) = seg.vector_at(idx) {
                        let d = distance::compute(metric, query, v);
                        seen_ids.insert(id);
                        results.push((id, d));
                    }
                }
            }
        }

        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);
        Ok(results)
    }

    /// Search across delta + all sealed segments.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, VivvyError> {
        self.search_filtered(query, k, None)
    }

    #[must_use]
    pub fn delta_len(&self) -> usize {
        self.shards.iter().map(|s| s.read().len()).sum()
    }

    #[must_use]
    pub fn num_sealed(&self) -> usize {
        self.sealed.load().len()
    }

    /// Force immediate compaction.
    pub fn compact_now(&self) {
        if let Some(ref dir) = self.data_dir {
            let wal_ref = self.wal.as_deref();
            if let Err(e) = run_compaction(
                self.dims,
                &self.shards,
                &self.sealed,
                dir,
                wal_ref,
                &self.tombstoned_ids,
                &self.mutation_lock,
            ) {
                warn!("compaction failed: {e}");
            }
        } else if let Err(e) = run_in_memory_compaction(
            self.dims,
            &self.shards,
            &self.tombstoned_ids,
            &self.mutation_lock,
        ) {
            warn!("in-memory compaction failed: {e}");
        }
    }

    /// Flush any pending delta vectors to disk and copy all sealed vector segments (.vivvy) and manifest (.idx) to target_dir.
    pub fn flush_and_copy_segments(&self, target_dir: &Path) -> Result<(), VivvyError> {
        if !target_dir.exists() {
            std::fs::create_dir_all(target_dir).map_err(|e| VivvyError::Wal(WalError::Io(e)))?;
        }
        if let Some(ref dir) = self.data_dir {
            self.compact_now();
            if dir.exists() {
                let manifest_src = Manifest::manifest_path(dir);
                if manifest_src.exists() {
                    let manifest_dst = Manifest::manifest_path(target_dir);
                    std::fs::copy(&manifest_src, &manifest_dst)
                        .map_err(|e| VivvyError::Wal(WalError::Io(e)))?;
                }

                if let Ok(entries) = std::fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().and_then(|s| s.to_str()) == Some("vivvy") {
                            if let Some(fname) = path.file_name() {
                                let dst = target_dir.join(fname);
                                std::fs::copy(&path, &dst)
                                    .map_err(|e| VivvyError::Wal(WalError::Io(e)))?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl Drop for VivvyIndex {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

struct CompactorParams {
    running: Arc<AtomicBool>,
    dims: usize,
    shards: Arc<Vec<RwLock<HnswIndex>>>,
    sealed: Arc<ArcSwap<Vec<Arc<SealedSegment>>>>,
    data_dir: Option<PathBuf>,
    wal: Option<Arc<Mutex<WalWriter>>>,
    tombstoned_ids: Arc<RwLock<RoaringTreemap>>,
    mutation_lock: Arc<Mutex<()>>,
}

fn compactor_loop(params: CompactorParams) {
    let threshold = 10_000usize;
    let mut tick_counter = 0u32;
    while params.running.load(Ordering::Acquire) {
        thread::sleep(Duration::from_millis(50));
        tick_counter += 1;
        if tick_counter >= 100 {
            tick_counter = 0;
            // Use physical node count, not live-ID count: repeated upserts of a
            // single ID create tombstoned graph nodes that must still trigger
            // compaction and reclamation.
            let physical_len: usize = params
                .shards
                .iter()
                .map(|shard| shard.read().physical_len())
                .sum();
            if physical_len >= threshold {
                if let Some(ref dir) = params.data_dir {
                    let wal_ref = params.wal.as_deref();
                    if let Err(e) = run_compaction(
                        params.dims,
                        &params.shards,
                        &params.sealed,
                        dir,
                        wal_ref,
                        &params.tombstoned_ids,
                        &params.mutation_lock,
                    ) {
                        warn!("compaction failed: {e}");
                    }
                } else if let Err(e) = run_in_memory_compaction(
                    params.dims,
                    &params.shards,
                    &params.tombstoned_ids,
                    &params.mutation_lock,
                ) {
                    warn!("in-memory compaction failed: {e}");
                }
            }
        }
    }
}

fn run_compaction(
    dims: usize,
    shards: &[RwLock<HnswIndex>],
    sealed: &ArcSwap<Vec<Arc<SealedSegment>>>,
    data_dir: &Path,
    wal: Option<&Mutex<WalWriter>>,
    tombstoned_ids: &RwLock<RoaringTreemap>,
    mutation_lock: &Mutex<()>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Keep writers out until the replacement segment and WAL reset form one
    // durable state. The old delta remains searchable until that point, so an
    // I/O failure cannot discard live vectors.
    let _mutation_guard = mutation_lock.lock();
    let mut entries = Vec::new();
    for shard in shards {
        entries.extend(shard.read().snapshot());
    }

    let previous_segments = sealed.load_full();
    let previous_manifest = Manifest::load(data_dir)?.unwrap_or_else(|| Manifest::new(Vec::new()));
    let tombstones = tombstoned_ids.read();
    let mut seen_ids: HashSet<u64> = entries.iter().map(|(id, _)| *id).collect();

    // Preserve only the newest live sealed version of each ID. Delta vectors
    // take precedence over every sealed copy, which physically reclaims vectors
    // superseded by insert_with_id.
    for seg in previous_segments.iter().rev() {
        for idx in 0..seg.num_nodes() {
            let id = seg.id_at(idx)?;
            if tombstones.contains(id) || !seen_ids.insert(id) {
                continue;
            }
            entries.push((id, seg.vector_at(idx)?.to_vec()));
        }
    }
    drop(tombstones);

    if entries.is_empty() && previous_manifest.segments.is_empty() {
        return Ok(());
    }
    info!(
        "compacting {} live vectors into a replacement sealed segment",
        entries.len()
    );

    let (new_manifest, new_sealed) = if entries.is_empty() {
        (Manifest::new(Vec::new()), Vec::new())
    } else {
        let seg_filename = format!("seg-{}.vivvy", timestamp_ns());
        let seg_path = data_dir.join(&seg_filename);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&seg_path)?;
        let writer = BufWriter::new(file.try_clone()?);

        let mut seg_writer = SegmentWriter::new(writer, dims as u32, 16, 32);
        for (id, vector) in &entries {
            seg_writer.push(*id, 0, vec![Vec::new()], vector.clone())?;
        }
        seg_writer.write()?;
        file.sync_all()?;

        (
            Manifest::new(vec![seg_filename]),
            vec![Arc::new(SealedSegment::open(&seg_path)?)],
        )
    };

    // The manifest is written atomically before old files are retired. A crash
    // before this step leaves only an orphan; a crash after it has a complete,
    // self-contained replacement segment to recover from.
    new_manifest.save(data_dir)?;
    sealed.store(Arc::new(new_sealed));

    for shard in shards {
        shard.write().drain();
    }

    if let Some(wal_mutex) = wal {
        let mut wal_guard = wal_mutex.lock();
        wal_guard.reset()?;
        wal_guard.commit()?;
    }
    tombstoned_ids.write().clear();

    // Unlink only files that are no longer reachable from the new manifest.
    // An unlink failure leaves a harmless orphan and must not undo compaction.
    for filename in previous_manifest.segments {
        let path = data_dir.join(filename);
        if let Err(e) = std::fs::remove_file(&path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                warn!("failed to remove obsolete segment {:?}: {}", path, e);
            }
        }
    }

    info!(
        "compaction complete: {} live vectors, {} obsolete segments reclaimed",
        entries.len(),
        previous_segments.len()
    );
    Ok(())
}

fn run_in_memory_compaction(
    dims: usize,
    shards: &[RwLock<HnswIndex>],
    tombstoned_ids: &RwLock<RoaringTreemap>,
    mutation_lock: &Mutex<()>,
) -> Result<(), Box<dyn std::error::Error>> {
    let _mutation_guard = mutation_lock.lock();
    for shard in shards {
        let mut current = shard.write();
        let metric = current.metric();
        let live_entries = current.snapshot();
        let mut rebuilt = HnswIndex::new(dims, metric);
        for (id, vector) in live_entries {
            rebuilt.insert(id, vector);
        }
        *current = rebuilt;
    }
    // Without sealed segments there is nothing left for these tombstones to
    // mask; the rebuilt delta already excludes their deleted nodes.
    tombstoned_ids.write().clear();
    Ok(())
}

fn timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_basic_insert_search() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let segments_dir = dir.path().join("segments");
        std::fs::create_dir_all(&segments_dir).unwrap();

        let idx = VivvyIndex::new(3, Metric::L2, Some(&wal), Some(&segments_dir)).unwrap();
        idx.insert(vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(vec![0.0, 1.0, 0.0]).unwrap();

        let hits = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert!(!hits.is_empty());
        drop(idx);
    }

    #[test]
    fn test_compaction() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let segments_dir = dir.path().join("segments");
        std::fs::create_dir_all(&segments_dir).unwrap();

        let idx = VivvyIndex::new(3, Metric::L2, Some(&wal), Some(&segments_dir)).unwrap();
        for i in 0..5 {
            idx.insert(vec![i as f32, 0.0, 0.0]).unwrap();
        }
        assert_eq!(idx.delta_len(), 5);
        idx.compact_now();
        assert_eq!(idx.delta_len(), 0);
        assert_eq!(idx.num_sealed(), 1);

        let hits = idx.search(&[3.0, 0.0, 0.0], 1).unwrap();
        assert!(!hits.is_empty());
        drop(idx);
    }

    #[test]
    fn test_compaction_reclaims_superseded_sealed_segments() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let segments_dir = dir.path().join("segments");
        std::fs::create_dir_all(&segments_dir).unwrap();

        let idx = VivvyIndex::new(3, Metric::L2, Some(&wal), Some(&segments_dir)).unwrap();
        idx.insert_with_id(10, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert_with_id(20, vec![0.0, 1.0, 0.0]).unwrap();
        idx.compact_now();

        let segment_count = || {
            std::fs::read_dir(&segments_dir)
                .unwrap()
                .flatten()
                .filter(|entry| {
                    entry.path().extension().and_then(|ext| ext.to_str()) == Some("vivvy")
                })
                .count()
        };
        assert_eq!(segment_count(), 1);

        idx.insert_with_id(10, vec![0.0, 0.0, 100.0]).unwrap();
        idx.compact_now();

        assert_eq!(idx.num_sealed(), 1);
        assert_eq!(
            segment_count(),
            1,
            "obsolete segment file must be reclaimed"
        );
        assert_eq!(std::fs::metadata(&wal).unwrap().len(), 0);
        assert_eq!(idx.search(&[1.0, 0.0, 0.0], 1).unwrap()[0].0, 20);

        drop(idx);
        let reopened = VivvyIndex::new(3, Metric::L2, Some(&wal), Some(&segments_dir)).unwrap();
        assert_eq!(reopened.num_sealed(), 1);
        assert_eq!(reopened.search(&[0.0, 0.0, 100.0], 1).unwrap()[0].0, 10);
    }

    #[test]
    fn test_in_memory_compaction_reclaims_upsert_tombstones() {
        let idx = VivvyIndex::new(3, Metric::L2, None::<&Path>, None::<&Path>).unwrap();
        for value in 0..10 {
            idx.insert_with_id(10, vec![value as f32, 0.0, 0.0])
                .unwrap();
        }

        let physical_before: usize = idx
            .shards
            .iter()
            .map(|shard| shard.read().physical_len())
            .sum();
        assert_eq!(physical_before, 10);

        idx.compact_now();

        let physical_after: usize = idx
            .shards
            .iter()
            .map(|shard| shard.read().physical_len())
            .sum();
        assert_eq!(physical_after, 1);
        assert_eq!(idx.search(&[9.0, 0.0, 0.0], 1).unwrap()[0].0, 10);
    }

    #[test]
    fn test_512_dim() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let segments_dir = dir.path().join("segments");
        std::fs::create_dir_all(&segments_dir).unwrap();

        let idx = VivvyIndex::new(512, Metric::L2, Some(&wal), Some(&segments_dir)).unwrap();
        let v1 = vec![1.0; 512];
        let v2 = vec![0.0; 512];

        idx.insert(v1.clone()).unwrap();
        idx.insert(v2.clone()).unwrap();

        let hits = idx.search(&v1, 1).unwrap();
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn test_64bit_id_preservation_and_filtering() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let segments_dir = dir.path().join("segments");
        std::fs::create_dir_all(&segments_dir).unwrap();

        let idx = VivvyIndex::new(3, Metric::L2, Some(&wal), Some(&segments_dir)).unwrap();
        let large_id_1 = (1u64 << 40) + 123;
        let large_id_2 = u64::MAX - 99;

        idx.insert_with_id(large_id_1, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert_with_id(large_id_2, vec![0.0, 1.0, 0.0]).unwrap();

        let matches1 = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(matches1[0].0, large_id_1);

        let matches2 = idx.search(&[0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(matches2[0].0, large_id_2);

        idx.compact_now();
        assert_eq!(idx.num_sealed(), 1);

        let sealed_matches = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(sealed_matches[0].0, large_id_1);
    }

    #[test]
    fn test_explicit_id_high_water_mark() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let idx = VivvyIndex::new(3, Metric::L2, Some(&wal), Option::<&Path>::None).unwrap();

        idx.insert_with_id(500, vec![1.0, 0.0, 0.0]).unwrap();
        let auto_id = idx.insert(vec![0.0, 1.0, 0.0]).unwrap();
        assert!(
            auto_id > 500,
            "auto-generated ID must be strictly greater than explicit high-water mark, got {}",
            auto_id
        );
    }

    #[test]
    fn test_reopen_wal_and_sealed_recovery() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let segments_dir = dir.path().join("segments");
        std::fs::create_dir_all(&segments_dir).unwrap();

        {
            let idx = VivvyIndex::new(3, Metric::L2, Some(&wal), Some(&segments_dir)).unwrap();
            idx.insert_with_id(10, vec![1.0, 0.0, 0.0]).unwrap();
            idx.insert_with_id(20, vec![0.0, 1.0, 0.0]).unwrap();
            // Compact 10 & 20 into a sealed segment
            idx.compact_now();
            assert_eq!(idx.num_sealed(), 1);

            // Insert 30 into WAL (uncompacted delta)
            idx.insert_with_id(30, vec![0.0, 0.0, 1.0]).unwrap();
            assert_eq!(idx.delta_len(), 1);
        }

        // Reopen index from the same directory & wal
        let reopened = VivvyIndex::new(3, Metric::L2, Some(&wal), Some(&segments_dir)).unwrap();
        assert_eq!(reopened.num_sealed(), 1);
        assert_eq!(reopened.delta_len(), 1);

        let match_10 = reopened.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(match_10[0].0, 10);

        let match_20 = reopened.search(&[0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(match_20[0].0, 20);

        let match_30 = reopened.search(&[0.0, 0.0, 1.0], 1).unwrap();
        assert_eq!(match_30[0].0, 30);

        // Verify high-water mark after reopen
        let next_auto = reopened.insert(vec![0.5, 0.5, 0.0]).unwrap();
        assert!(
            next_auto > 30,
            "next auto ID must be > 30, got {}",
            next_auto
        );
    }

    #[test]
    fn test_upsert_replaces_old_vector() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let idx = VivvyIndex::new(3, Metric::L2, Some(&wal), Option::<&Path>::None).unwrap();

        idx.insert_with_id(10, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert_with_id(20, vec![0.0, 1.0, 0.0]).unwrap();

        let hits = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].0, 10);

        // Update ID 10 to a new position (0, 0, 10)
        idx.insert_with_id(10, vec![0.0, 0.0, 10.0]).unwrap();

        // Querying at (1, 0, 0) should now find 20 first, NOT stale 10
        let hits_after = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits_after[0].0, 20);

        // Querying at (0, 0, 10) should find 10
        let hits_new = idx.search(&[0.0, 0.0, 10.0], 1).unwrap();
        assert_eq!(hits_new[0].0, 10);
    }

    #[test]
    fn test_upsert_masks_old_sealed_vector_when_new_vector_is_not_a_delta_hit() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let segments_dir = dir.path().join("segments");
        std::fs::create_dir_all(&segments_dir).unwrap();

        let idx = VivvyIndex::new(3, Metric::L2, Some(&wal), Some(&segments_dir)).unwrap();
        idx.insert_with_id(10, vec![1.0, 0.0, 0.0]).unwrap();
        idx.compact_now();

        // Move 10 far away, then add enough nearby delta vectors that 10 is not
        // returned by the delta graph for this query.
        idx.insert_with_id(10, vec![0.0, 0.0, 100.0]).unwrap();
        for id in 20..140 {
            idx.insert_with_id(id, vec![1.0, 0.0, 0.0]).unwrap();
        }

        let hits = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_ne!(hits[0].0, 10, "old sealed vector must not be returned");

        // The same invariant must hold after the updated vector is sealed too.
        idx.compact_now();
        let hits_after_compaction = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_ne!(hits_after_compaction[0].0, 10);
    }

    #[test]
    fn test_delete_and_sealed_tombstone() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let segments_dir = dir.path().join("segments");
        std::fs::create_dir_all(&segments_dir).unwrap();

        let idx = VivvyIndex::new(3, Metric::L2, Some(&wal), Some(&segments_dir)).unwrap();
        idx.insert_with_id(10, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert_with_id(20, vec![0.0, 1.0, 0.0]).unwrap();

        // Compact into sealed segment
        idx.compact_now();
        assert_eq!(idx.num_sealed(), 1);

        // Delete sealed ID 10
        assert!(idx.delete(10).is_ok());

        let hits = idx.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 20);
    }
}
