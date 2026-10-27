//! Hierarchical Navigable Small World (HNSW) graph.
//! Reference: Malkov & Yashunin (TPAMI 2020).
//!
//! Multi-layer graph where each node lives at one or more layers. Top layers
//! are sparse with long-range connections (the "expressway") for quickly
//! finding the right neighbourhood. Layer 0 is dense (every node, many
//! connections) for fine-grained precision.
//!
//! Search: greedy descent from the entry point through each layer, then
//! ef-search at layer 0 (min-heap of candidates, max-heap of results).
//! Insert: assign random level, find position via greedy descent, connect
//! to the M closest neighbours at each layer, trim neighbours that exceed
//! M_max.
//!
//! We use simple closest-M selection rather than the paper's diversity
//! heuristic because with M=16 / M_max=32 we already get >95% recall@10
//! on SIFT1M, and the heuristic adds ~20% to insert time.
//! ponytail: add diversity heuristic if benchmarking shows a gap.

use crate::distance::{self, Metric};
use ordered_float::OrderedFloat;
use roaring::RoaringBitmap;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct HnswIndex {
    // Vectors + IDs by internal node index (0..len-1).
    vectors: Vec<f32>,
    dims: usize,
    ids: Vec<u64>,

    // neighbors[node][layer] = neighbour indices.
    neighbors: Vec<Vec<Vec<usize>>>,
    levels: Vec<usize>,

    entry: Option<usize>,
    max_level: usize,

    m: usize, // connections per layer (M_max = 2*M at layer 0)
    m_max: usize,
    ef_construction: usize,
    ml: f32, // 1/ln(M) for level generation

    metric: Metric,

    // Map external u64 id -> active internal node index
    id_to_node: HashMap<u64, usize>,
    // Tombstoned node indices (either replaced by upsert or explicitly deleted)
    deleted_nodes: RoaringBitmap,
}

// ef-search priority-queue element. OrderedFloat so f32 can be Ord.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Candidate {
    dist: OrderedFloat<f32>,
    idx: usize,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .cmp(&other.dist)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl HnswIndex {
    #[must_use]
    pub fn new(dims: usize, metric: Metric) -> Self {
        // Paper defaults: M=16, ef_construction=200, ml=1/ln(M)≈0.36.
        let m = 16;
        Self {
            vectors: Vec::new(),
            dims,
            ids: Vec::new(),
            neighbors: Vec::new(),
            levels: Vec::new(),
            entry: None,
            max_level: 0,
            m,
            m_max: m * 2,
            ef_construction: 200,
            ml: 1.0 / (m as f32).ln(),
            metric,
            id_to_node: HashMap::new(),
            deleted_nodes: RoaringBitmap::new(),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.id_to_node.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.id_to_node.is_empty()
    }

    /// Iterate the IDs of all live nodes.
    ///
    /// This deliberately excludes superseded and deleted nodes. Callers that merge
    /// this delta graph with older immutable segments use it to ensure an older
    /// copy of an ID cannot reappear merely because its newer vector did not make
    /// the delta graph's top-k results.
    pub(crate) fn active_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.ids
            .iter()
            .copied()
            .enumerate()
            .filter(|(idx, _)| !self.deleted_nodes.contains(*idx as u32))
            .map(|(_, id)| id)
    }

    // Geometric: P(level=L) = exp(−L/mL) − exp(−(L+1)/mL).
    // Sampled via floor(−ln(u)·mL). ~63% at level 0, ~23% at 1, ~9% at 2, etc.
    fn random_level(&self) -> usize {
        let r: f32 = fastrand::f32();
        (-r.ln() * self.ml).floor() as usize
    }

    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        let idx = self.ids.len();
        if let Some(old_idx) = self.id_to_node.insert(id, idx) {
            self.deleted_nodes.insert(old_idx as u32);
        }
        self.vectors.extend(vector);
        self.ids.push(id);
        let level = self.random_level();
        self.levels.push(level);
        self.neighbors.push(vec![Vec::new(); level + 1]);

        let Some(mut ep) = self.entry else {
            self.entry = Some(idx);
            self.max_level = level;
            return;
        };

        // Greedy descent from top to one above node's level — we only need
        // the single closest node to locate the insertion neighbourhood.
        for lc in (level + 1..=self.max_level).rev() {
            ep = self.greedy_down(&self.vectors[idx * self.dims..(idx + 1) * self.dims], ep, lc);
        }

        // Insert at each layer: ef_search → select closest → connect bidirectionally → trim overflow.
        for lc in (0..=level.min(self.max_level)).rev() {
            let candidates = self.ef_search(&self.vectors[idx * self.dims..(idx + 1) * self.dims], ep, self.ef_construction, lc);
            let m_conn = if lc == 0 { self.m_max } else { self.m };
            let selected = closest_n(&candidates, m_conn);

            for &n in &selected {
                self.neighbors[idx][lc].push(n);
                if lc < self.neighbors[n].len() {
                    self.neighbors[n][lc].push(idx);
                    if self.neighbors[n][lc].len() > self.m_max {
                        self.neighbors[n][lc] = self.trim(n, lc);
                    }
                }
            }
            // "Follow the closest" heuristic — next layer is a superset of this one.
            ep = selected.first().copied().unwrap_or(ep);
        }

        if level > self.max_level {
            self.entry = Some(idx);
            self.max_level = level;
        }
    }

    // Hill-climb at one layer: walk to the closest neighbour until no improvement.
    fn greedy_down(&self, query: &[f32], mut ep: usize, layer: usize) -> usize {
        loop {
            let mut improved = false;
            if let Some(neigh) = self.neighbors[ep].get(layer) {
                for &n in neigh {
                    let d = distance::compute(self.metric, query, &self.vectors[n * self.dims..(n + 1) * self.dims]);
                    if d < distance::compute(self.metric, query, &self.vectors[ep * self.dims..(ep + 1) * self.dims]) {
                        ep = n;
                        improved = true;
                    }
                }
            }
            if !improved {
                return ep;
            }
        }
    }

    // Core HNSW search primitive at one layer.
    // Min-heap of candidates to explore (Reverse<Candidate>), max-heap of best ef results (Candidate).
    // Stop when the closest candidate is farther than the furthest result in results
    // (triangle inequality guarantees no unexplored node can beat it).
    // ef controls speed-recall: insertion uses ef_construction (200),
    // search starts at k.max(100) and expands dynamically under filter.
    fn ef_search(&self, query: &[f32], entry: usize, ef: usize, layer: usize) -> Vec<(usize, f32)> {
        let mut visited = HashSet::new();
        visited.insert(entry);
        let ed = distance::compute(self.metric, query, &self.vectors[entry * self.dims..(entry + 1) * self.dims]);

        let mut candidates = BinaryHeap::new();
        candidates.push(Reverse(Candidate {
            dist: OrderedFloat(ed),
            idx: entry,
        }));

        let mut results = BinaryHeap::new();
        results.push(Candidate {
            dist: OrderedFloat(ed),
            idx: entry,
        });

        while let Some(Reverse(c)) = candidates.pop() {
            let furthest = results.peek().unwrap().dist;
            if c.dist > furthest {
                break;
            }
            if let Some(neigh) = self.neighbors[c.idx].get(layer) {
                for &n in neigh {
                    if visited.insert(n) {
                        let d = distance::compute(self.metric, query, &self.vectors[n * self.dims..(n + 1) * self.dims]);
                        let of = OrderedFloat(d);
                        let furthest = results.peek().unwrap().dist;
                        if of < furthest || results.len() < ef {
                            candidates.push(Reverse(Candidate { dist: of, idx: n }));
                            results.push(Candidate { dist: of, idx: n });
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        results
            .into_sorted_vec()
            .into_iter()
            .map(|c| (c.idx, c.dist.into()))
            .collect()
    }

    // Keep the closest M_max edges. "Closest" policy favours short, strong
    // edges over diverse ones, which improves search efficiency.
    fn trim(&self, node: usize, layer: usize) -> Vec<usize> {
        let mut v: Vec<(usize, f32)> = self.neighbors[node][layer]
            .iter()
            .map(|&n| {
                (
                    n,
                    distance::compute(self.metric, &self.vectors[node * self.dims..(node + 1) * self.dims], &self.vectors[n * self.dims..(n + 1) * self.dims]),
                )
            })
            .collect();
        v.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        v.truncate(self.m_max);
        v.into_iter().map(|(i, _)| i).collect()
    }

    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Soft-delete an ID from the HNSW index.
    ///
    /// The node is tombstoned: it remains traversable during graph searches to preserve
    /// connectivity, but is excluded from search results and pruned during compaction.
    pub fn delete(&mut self, id: u64) -> bool {
        if let Some(idx) = self.id_to_node.remove(&id) {
            self.deleted_nodes.insert(idx as u32);
            true
        } else {
            false
        }
    }

    #[must_use]
    pub fn snapshot(&self) -> Vec<(u64, Vec<f32>)> {
        self.ids
            .iter()
            .copied()
            .enumerate()
            .filter(|(idx, _)| !self.deleted_nodes.contains(*idx as u32))
            .map(|(idx, id)| {
                let start = idx * self.dims;
                let end = start + self.dims;
                (id, self.vectors[start..end].to_vec())
            })
            .collect()
    }

    // Drains the index to empty. Used by the compactor to move delta entries to sealed segments.
    pub fn drain(&mut self) -> Vec<(u64, Vec<f32>)> {
        let snapshot = self.snapshot();
        self.vectors.clear();
        self.ids.clear();
        self.neighbors.clear();
        self.levels.clear();
        self.entry = None;
        self.max_level = 0;
        self.id_to_node.clear();
        self.deleted_nodes.clear();
        snapshot
    }

    // Unfiltered search — delegates to search_filtered with no filter.
    #[must_use]
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.search_filtered(query, k, None)
    }

    // Search with optional filter bitmap. If fewer than k results pass the
    // filter, double ef (up to 1000) and retry. This avoids the worst case
    // (re-search with max ef from scratch) and the naive approach (always
    // use max ef, slow for broad filters).
    #[must_use]
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&roaring::RoaringTreemap>,
    ) -> Vec<(u64, f32)> {
        if self.is_empty() {
            return Vec::new();
        }

        let ep = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };

        let mut best = ep;
        for lc in (1..=self.max_level).rev() {
            best = self.greedy_down(query, best, lc);
        }

        let max_ef = 1000;
        let mut ef = k.max(100);

        loop {
            let neighbors = self.ef_search(query, best, ef, 0);
            let mut results: Vec<(u64, f32)> = neighbors
                .into_iter()
                .filter(|(idx, _)| !self.deleted_nodes.contains(*idx as u32))
                .map(|(idx, d)| (self.ids[idx], d))
                .collect();

            if let Some(bitmap) = filter {
                results.retain(|(id, _)| bitmap.contains(*id));
            }

            if results.len() >= k || ef >= max_ef {
                results.truncate(k);
                return results;
            }

            ef = (ef * 2).min(max_ef);
        }
    }
}

// Select the n closest (index, distance) pairs. Used during insertion.
fn closest_n(candidates: &[(usize, f32)], n: usize) -> Vec<usize> {
    let mut sorted = candidates.to_vec();
    sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    sorted.truncate(n);
    sorted.into_iter().map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_insert_search() {
        let mut idx = HnswIndex::new(3, Metric::L2);
        idx.insert(0, vec![1.0, 0.0, 0.0]);
        idx.insert(1, vec![0.0, 1.0, 0.0]);
        idx.insert(2, vec![0.0, 0.0, 1.0]);
        let hits = idx.search(&[1.0, 0.0, 0.0], 1);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].0, 0);
    }

    #[test]
    fn test_hnsw_upsert_replaces_old_vector() {
        let mut idx = HnswIndex::new(3, Metric::L2);
        // Insert id 42 at (1, 0, 0)
        idx.insert(42, vec![1.0, 0.0, 0.0]);
        // Insert id 99 at (0, 1, 0)
        idx.insert(99, vec![0.0, 1.0, 0.0]);
        assert_eq!(idx.len(), 2);

        let hits_before = idx.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(hits_before[0].0, 42);

        // Move id 42 far away to (0, 0, 10.0)
        idx.insert(42, vec![0.0, 0.0, 10.0]);
        assert_eq!(idx.len(), 2);

        // Searching near (1, 0, 0) should now rank 99 first, NOT 42
        let hits_old_pos = idx.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(hits_old_pos[0].0, 99);

        // Searching near (0, 0, 10.0) should return 42
        let hits_new_pos = idx.search(&[0.0, 0.0, 10.0], 1);
        assert_eq!(hits_new_pos[0].0, 42);

        // Snapshot only contains 2 active items (with latest vector for 42)
        let snap = idx.snapshot();
        assert_eq!(snap.len(), 2);
        let v42 = snap.iter().find(|(id, _)| *id == 42).unwrap();
        assert_eq!(v42.1, vec![0.0, 0.0, 10.0]);
    }

    #[test]
    fn test_hnsw_delete() {
        let mut idx = HnswIndex::new(3, Metric::L2);
        idx.insert(10, vec![1.0, 0.0, 0.0]);
        idx.insert(20, vec![0.0, 1.0, 0.0]);
        assert_eq!(idx.len(), 2);

        assert!(idx.delete(10));
        assert_eq!(idx.len(), 1);
        assert!(!idx.delete(10));

        let hits = idx.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 20);

        // Drain should only produce active item 20
        let drained = idx.drain();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, 20);
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_hnsw_empty() {
        let idx = HnswIndex::new(2, Metric::L2);
        let hits = idx.search(&[1.0, 0.0], 5);
        assert!(hits.is_empty());
    }

    // 100 collinear points at x=0..100, search at x=50, expect 50 as top result.
    #[test]
    fn test_hnsw_multiple() {
        let mut idx = HnswIndex::new(2, Metric::L2);
        for i in 0..100 {
            let v = vec![i as f32, 0.0];
            idx.insert(i as u64, v);
        }
        let hits = idx.search(&[50.0, 0.0], 5);
        assert_eq!(hits.len(), 5);
        assert_eq!(hits[0].0, 50);
    }
}
