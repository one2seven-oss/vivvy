// Package vivvy provides idiomatic Go bindings to Vivvy — a local, durable,
// namespace-isolated long-term memory (LTM) runtime and single-machine
// vector engine for AI agents — via a pre-compiled static C-FFI library
// (cgo + staticlib). Vivvy runs in-process: no Rust toolchain or external
// daemon is required at build or run time. See the module's README
// (https://github.com/one2seven-oss/vivvy/blob/main/go/README.md) for how
// to populate lib/ with the platform-matching static library, and the
// repository's Doc.md
// (https://github.com/one2seven-oss/vivvy/blob/main/Doc.md) for the full
// integration guide covering all four language bindings.
//
// # Overview
//
// Open a [Store], insert scored, namespace-isolated memory records with
// [Store.Insert], and retrieve them with [Store.Recall] — hybrid vector
// search fused with optional full-text search via Reciprocal Rank Fusion,
// explainable scoring, and optional Maximal Marginal Relevance diversity
// reranking. [Store.FormatContext] renders recall results straight into a
// token-budgeted prompt string for LLM context injection. See the Example
// functions below for runnable end-to-end usage.
//
// A [*Store] is safe for concurrent use by multiple goroutines: Insert,
// Recall, FormatContext, Backup, and Vacuum may all run concurrently
// against the same Store. [Store.Close] must be called exactly once.
//
// Errors returned by store operations are [*Error], carrying a stable
// numeric Code alongside a human-readable Message; use [errors.As] to
// recover one and branch on Code rather than parsing Message strings — see
// the ErrCode* constants for the full list, mirrored exactly from
// include/vivvy.h and vivvy_memory::ErrorCode on the Rust side.
package vivvy

/*
#include "vivvy.h"
#include <stdlib.h>
*/
import "C"

import (
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// MemoryKind identifies the semantic nature of a stored memory.
type MemoryKind string

const (
	KindFact        MemoryKind = "fact"
	KindPreference  MemoryKind = "preference"
	KindInstruction MemoryKind = "instruction"
	KindContext     MemoryKind = "context"
	KindEpisodic    MemoryKind = "episodic"
)

// Stable FFI-boundary and domain error codes, mirroring vivvy.h /
// vivvy_memory::ErrorCode. Compare against (*Error).Code, or use
// errors.As to recover an *Error from a wrapped error.
const (
	ErrCodeNullPointer = -1
	ErrCodeInvalidUTF8 = -2
	ErrCodeInvalidJSON = -3
	ErrCodePanic       = -4
	ErrCodeIO          = -5

	ErrCodeInvalidScope             = 1
	ErrCodeDimensionMismatch        = 2
	ErrCodeEmbeddingModelMismatch   = 3
	ErrCodeNotFound                 = 4
	ErrCodeRevisionConflict         = 5
	ErrCodeStoreBusy                = 6
	ErrCodeRecoveryRequired         = 7
	ErrCodeCorruptStore             = 8
	ErrCodeEncryptionKeyUnavailable = 9
	ErrCodePolicyDenied             = 10
	ErrCodeInvalidFilter            = 11
	ErrCodeInvalidInput             = 12
	ErrCodeStoreIOError             = 13
	ErrCodeDatabaseError            = 14
	ErrCodeEncryptionFailed         = 15
	ErrCodeDecryptionFailed         = 16
)

// ErrClosed is returned by any Store method called after Close.
var ErrClosed = errors.New("vivvy: store is closed")

// Error wraps a failure reported by the underlying Rust store, carrying the
// stable numeric code from vivvy.h alongside the human-readable message.
type Error struct {
	Code    int
	Message string
}

func (e *Error) Error() string {
	return fmt.Sprintf("vivvy: [%d] %s", e.Code, e.Message)
}

// lastError must be called immediately after a failing cgo call, before any
// other vivvy_* call on this goroutine's underlying OS thread, since the
// message lives in Rust thread-local storage.
func lastError(code C.int) error {
	msg := C.GoString(C.vivvy_last_error_message())
	return &Error{Code: int(code), Message: msg}
}

// OpenConfig configures a new or existing store.
type OpenConfig struct {
	// Dimensions is the fixed embedding width every vector in this store must
	// match. Required.
	Dimensions int
	// EmbeddingModel identifies the embedding model whose vectors this store
	// holds; purely descriptive, not enforced against caller-supplied
	// vectors. Required.
	EmbeddingModel string
	// MaxRecallLimit caps Recall/FormatContext result counts. Zero uses the
	// library default (100).
	MaxRecallLimit int
}

// MemoryRecord describes an observation to store via Store.Insert. The
// embedding itself is passed as a separate argument, not a field here.
type MemoryRecord struct {
	// ID, timestamps, and Revision are populated when a record is returned by
	// Get. They are ignored by Insert.
	ID string
	// OperationID, if set, makes this insert idempotent: retrying with the
	// same OperationID returns the same result rather than duplicating data.
	OperationID string
	TenantID    string
	Namespace   string
	AgentID     string
	UserID      string
	Content     string
	// Kind defaults to KindFact if empty.
	Kind MemoryKind
	// Importance must be in [0, 1]. Nil uses the library default (0.5).
	Importance  *float32
	ExpiresAtMs *int64
	Metadata    map[string]any
	Source      map[string]any
	CreatedAtMs int64
	UpdatedAtMs int64
	Revision    uint64
}

// Scope identifies the tenant and namespace containing a memory. AgentID and
// UserID further narrow the scope when they were used at insert time.
type Scope struct {
	TenantID  string
	Namespace string
	AgentID   string
	UserID    string
}

// UpdateRequest atomically changes a record when ExpectedRevision matches its
// current revision. Leave optional fields nil to preserve their existing
// values. Set ClearExpiresAt to remove an existing expiration.
type UpdateRequest struct {
	OperationID      string
	Scope            Scope
	ID               string
	ExpectedRevision uint64
	Content          *string
	Kind             *MemoryKind
	Importance       *float32
	ExpiresAtMs      *int64
	ClearExpiresAt   bool
	MetadataPatch    map[string]any
}

// StoreHealth is a privacy-safe operational snapshot; it never includes
// memory content or embeddings.
type StoreHealth struct {
	IsHealthy              bool
	TotalActiveRecords     int
	TotalTombstonedRecords int
	PendingOperationsCount int
	IndexRebuildRequired   bool
	DBSizeBytes            uint64
	WALSizeBytes           uint64
}

// IndexConfig configures a low-level vector index. Omit WALPath and DataDir
// for an in-memory index; set both for durable WAL replay and compaction.
type IndexConfig struct {
	Dimensions int
	Metric     string
	WALPath    string
	DataDir    string
}

// Filter is the recursive metadata-filter grammar accepted by Index.Search.
// Exactly one operator should be set on each Filter value.
type Filter struct {
	Eq  *FilterEquals `json:"eq,omitempty"`
	In  *FilterIn     `json:"in,omitempty"`
	And []Filter      `json:"and,omitempty"`
	Or  []Filter      `json:"or,omitempty"`
	Not *Filter       `json:"not,omitempty"`
}

type FilterEquals struct {
	Field string `json:"field"`
	Value string `json:"value"`
}

type FilterIn struct {
	Field  string   `json:"field"`
	Values []string `json:"values"`
}

// SearchResult is one nearest-neighbour result from a low-level Index.
type SearchResult struct {
	ID       uint64  `json:"id"`
	Distance float32 `json:"distance"`
}

// ScoringConfig customizes candidate reranking weights and decay.
type ScoringConfig struct {
	WeightSimilarity    float32 `json:"weight_similarity,omitempty"`
	WeightImportance    float32 `json:"weight_importance,omitempty"`
	WeightRecency       float32 `json:"weight_recency,omitempty"`
	WeightReinforcement float32 `json:"weight_reinforcement,omitempty"`
	RecencyHalfLifeDays float32 `json:"recency_half_life_days,omitempty"`
}

// RecallOptions selects and filters candidates for Store.Recall and
// Store.FormatContext.
type RecallOptions struct {
	TenantID  string
	Namespace string
	AgentID   string
	UserID    string

	// QueryText, if non-empty, enables hybrid recall: lexical FTS5 matches
	// are fused with vector search results via Reciprocal Rank Fusion.
	QueryText string
	// TopK is the maximum number of results to return. Zero defaults to 10.
	TopK int

	Kinds           []MemoryKind
	MinImportance   *float32
	MetadataEq      map[string]any
	CreatedAfterMs  *int64
	CreatedBeforeMs *int64

	// IncludeExplanations attaches a score breakdown to each result.
	IncludeExplanations bool
	// MMRLambda, if set (in [0, 1]), enables Maximal Marginal Relevance
	// diversity reranking: 1.0 is pure relevance, 0.0 is pure diversity.
	MMRLambda *float32

	// Scoring customizes candidate reranking weights and decay.
	Scoring *ScoringConfig
}

// Explanation breaks down how a ScoredMemory's score was computed.
type Explanation struct {
	TotalScore         float32
	SimilarityScore    float32
	ImportanceScore    float32
	RecencyScore       float32
	ReinforcementScore float32
	PolicyNotes        []string
}

// ScoredMemory is a single recalled record together with its relevance score.
type ScoredMemory struct {
	ID          string
	TenantID    string
	Namespace   string
	AgentID     string
	UserID      string
	Kind        MemoryKind
	Content     string
	Importance  float32
	CreatedAtMs int64
	UpdatedAtMs int64
	ExpiresAtMs *int64
	Revision    uint64
	Metadata    map[string]any
	Source      map[string]any
	Score       float32
	// Explanation is nil unless RecallOptions.IncludeExplanations was true.
	Explanation *Explanation
}

// RecallResult is the outcome of Store.Recall.
type RecallResult struct {
	TotalCandidates int
	Items           []ScoredMemory
}

// FormatOptions controls Store.FormatContext's token-budgeted prompt
// rendering. All fields are optional; zero values fall back to library
// defaults (max 1500 tokens, a default line template and header, no footer).
type FormatOptions struct {
	MaxTokens *int
	// Template supports the placeholders {kind}, {content}, {score} and
	// {score:.2}.
	Template string
	Header   *string
	Footer   *string
}

// -- wire-format DTOs (unexported; mirror ffi/src/types.rs) --------------------------------

type openConfigJSON struct {
	Dimensions     int    `json:"dimensions"`
	EmbeddingModel string `json:"embedding_model"`
	MaxRecallLimit int    `json:"max_recall_limit,omitempty"`
}

type insertRecordJSON struct {
	OperationID *string        `json:"operation_id,omitempty"`
	TenantID    string         `json:"tenant_id"`
	Namespace   string         `json:"namespace"`
	AgentID     *string        `json:"agent_id,omitempty"`
	UserID      *string        `json:"user_id,omitempty"`
	Content     string         `json:"content"`
	Kind        string         `json:"kind,omitempty"`
	Importance  *float32       `json:"importance,omitempty"`
	ExpiresAtMs *int64         `json:"expires_at_ms,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
	Source      map[string]any `json:"source,omitempty"`
}

type scopeJSON struct {
	TenantID  string  `json:"tenant_id"`
	Namespace string  `json:"namespace"`
	AgentID   *string `json:"agent_id,omitempty"`
	UserID    *string `json:"user_id,omitempty"`
}

type updateRequestJSON struct {
	OperationID      *string        `json:"operation_id,omitempty"`
	TenantID         string         `json:"tenant_id"`
	Namespace        string         `json:"namespace"`
	AgentID          *string        `json:"agent_id,omitempty"`
	UserID           *string        `json:"user_id,omitempty"`
	ID               string         `json:"id"`
	ExpectedRevision uint64         `json:"expected_revision"`
	Content          *string        `json:"content,omitempty"`
	Kind             *string        `json:"kind,omitempty"`
	Importance       *float32       `json:"importance,omitempty"`
	MetadataPatch    map[string]any `json:"metadata_patch,omitempty"`
}

type memoryRecordJSON struct {
	ID          string         `json:"id"`
	TenantID    string         `json:"tenant_id"`
	Namespace   string         `json:"namespace"`
	AgentID     *string        `json:"agent_id"`
	UserID      *string        `json:"user_id"`
	Kind        string         `json:"kind"`
	Content     string         `json:"content"`
	Importance  float32        `json:"importance"`
	CreatedAtMs int64          `json:"created_at_ms"`
	UpdatedAtMs int64          `json:"updated_at_ms"`
	ExpiresAtMs *int64         `json:"expires_at_ms"`
	Revision    uint64         `json:"revision"`
	Metadata    map[string]any `json:"metadata"`
	Source      map[string]any `json:"source"`
}

type storeHealthJSON struct {
	IsHealthy              bool   `json:"is_healthy"`
	TotalActiveRecords     int    `json:"total_active_records"`
	TotalTombstonedRecords int    `json:"total_tombstoned_records"`
	PendingOperationsCount int    `json:"pending_operations_count"`
	IndexRebuildRequired   bool   `json:"index_rebuild_required"`
	DBSizeBytes            uint64 `json:"db_size_bytes"`
	WALSizeBytes           uint64 `json:"wal_size_bytes"`
}

type indexConfigJSON struct {
	Dimensions int     `json:"dimensions"`
	Metric     string  `json:"metric"`
	WALPath    *string `json:"wal_path,omitempty"`
	DataDir    *string `json:"data_dir,omitempty"`
}

type recallOptionsJSON struct {
	TenantID            string         `json:"tenant_id"`
	Namespace           string         `json:"namespace"`
	AgentID             *string        `json:"agent_id,omitempty"`
	UserID              *string        `json:"user_id,omitempty"`
	Kinds               []string       `json:"kinds,omitempty"`
	MinImportance       *float32       `json:"min_importance,omitempty"`
	MetadataEq          map[string]any `json:"metadata_eq,omitempty"`
	CreatedAfterMs      *int64         `json:"created_after_ms,omitempty"`
	CreatedBeforeMs     *int64         `json:"created_before_ms,omitempty"`
	IncludeExplanations bool           `json:"include_explanations"`
	MMRLambda           *float32       `json:"mmr_lambda,omitempty"`
	Scoring             *ScoringConfig `json:"scoring,omitempty"`
}

type formatOptionsJSON struct {
	MaxTokens *int    `json:"max_tokens,omitempty"`
	Template  *string `json:"template,omitempty"`
	Header    *string `json:"header,omitempty"`
	Footer    *string `json:"footer,omitempty"`
}

type explanationJSON struct {
	TotalScore         float32  `json:"total_score"`
	SimilarityScore    float32  `json:"similarity_score"`
	ImportanceScore    float32  `json:"importance_score"`
	RecencyScore       float32  `json:"recency_score"`
	ReinforcementScore float32  `json:"reinforcement_score"`
	PolicyNotes        []string `json:"policy_notes"`
}

type scoredMemoryJSON struct {
	ID          string           `json:"id"`
	TenantID    string           `json:"tenant_id"`
	Namespace   string           `json:"namespace"`
	AgentID     *string          `json:"agent_id"`
	UserID      *string          `json:"user_id"`
	Kind        string           `json:"kind"`
	Content     string           `json:"content"`
	Importance  float32          `json:"importance"`
	CreatedAtMs int64            `json:"created_at_ms"`
	UpdatedAtMs int64            `json:"updated_at_ms"`
	ExpiresAtMs *int64           `json:"expires_at_ms"`
	Revision    uint64           `json:"revision"`
	Metadata    map[string]any   `json:"metadata"`
	Source      map[string]any   `json:"source"`
	Score       float32          `json:"score"`
	Explanation *explanationJSON `json:"explanation"`
}

type recallResultsJSON struct {
	TotalCandidates int                `json:"total_candidates"`
	Items           []scoredMemoryJSON `json:"items"`
}

// -- marshaling helpers ---------------------------------------------------------------------

func nonEmpty(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

func marshalScope(scope Scope) ([]byte, error) {
	return json.Marshal(scopeJSON{
		TenantID:  scope.TenantID,
		Namespace: scope.Namespace,
		AgentID:   nonEmpty(scope.AgentID),
		UserID:    nonEmpty(scope.UserID),
	})
}

func marshalUpdate(req UpdateRequest) ([]byte, error) {
	var kind *string
	if req.Kind != nil {
		value := string(*req.Kind)
		kind = &value
	}
	payload, err := json.Marshal(updateRequestJSON{
		OperationID:      nonEmpty(req.OperationID),
		TenantID:         req.Scope.TenantID,
		Namespace:        req.Scope.Namespace,
		AgentID:          nonEmpty(req.Scope.AgentID),
		UserID:           nonEmpty(req.Scope.UserID),
		ID:               req.ID,
		ExpectedRevision: req.ExpectedRevision,
		Content:          req.Content,
		Kind:             kind,
		Importance:       req.Importance,
		MetadataPatch:    req.MetadataPatch,
	})
	if err != nil || (!req.ClearExpiresAt && req.ExpiresAtMs == nil) {
		return payload, err
	}

	// A Go struct cannot represent the three wire states required here:
	// omitted = preserve, integer = set, JSON null = clear. Add the final
	// field after normal marshaling to preserve that distinction.
	var fields map[string]any
	if err := json.Unmarshal(payload, &fields); err != nil {
		return nil, fmt.Errorf("vivvy: unmarshal update payload: %w", err)
	}
	if req.ClearExpiresAt {
		fields["expires_at_ms"] = nil
	} else {
		fields["expires_at_ms"] = *req.ExpiresAtMs
	}
	return json.Marshal(fields)
}

func convertMemoryRecord(raw memoryRecordJSON) *MemoryRecord {
	record := &MemoryRecord{
		ID:          raw.ID,
		TenantID:    raw.TenantID,
		Namespace:   raw.Namespace,
		Kind:        MemoryKind(raw.Kind),
		Content:     raw.Content,
		Importance:  &raw.Importance,
		ExpiresAtMs: raw.ExpiresAtMs,
		Metadata:    raw.Metadata,
		Source:      raw.Source,
		CreatedAtMs: raw.CreatedAtMs,
		UpdatedAtMs: raw.UpdatedAtMs,
		Revision:    raw.Revision,
	}
	if raw.AgentID != nil {
		record.AgentID = *raw.AgentID
	}
	if raw.UserID != nil {
		record.UserID = *raw.UserID
	}
	return record
}

func kindStrings(kinds []MemoryKind) []string {
	if len(kinds) == 0 {
		return nil
	}
	out := make([]string, len(kinds))
	for i, k := range kinds {
		out[i] = string(k)
	}
	return out
}

func marshalRecallOptions(opts RecallOptions) ([]byte, error) {
	payload, err := json.Marshal(recallOptionsJSON{
		TenantID:            opts.TenantID,
		Namespace:           opts.Namespace,
		AgentID:             nonEmpty(opts.AgentID),
		UserID:              nonEmpty(opts.UserID),
		Kinds:               kindStrings(opts.Kinds),
		MinImportance:       opts.MinImportance,
		MetadataEq:          opts.MetadataEq,
		CreatedAfterMs:      opts.CreatedAfterMs,
		CreatedBeforeMs:     opts.CreatedBeforeMs,
		IncludeExplanations: opts.IncludeExplanations,
		MMRLambda:           opts.MMRLambda,
		Scoring:             opts.Scoring,
	})
	if err != nil {
		return nil, fmt.Errorf("vivvy: marshal recall options: %w", err)
	}
	return payload, nil
}

func convertRecallResults(raw recallResultsJSON) *RecallResult {
	items := make([]ScoredMemory, 0, len(raw.Items))
	for _, it := range raw.Items {
		sm := ScoredMemory{
			ID:          it.ID,
			TenantID:    it.TenantID,
			Namespace:   it.Namespace,
			Kind:        MemoryKind(it.Kind),
			Content:     it.Content,
			Importance:  it.Importance,
			CreatedAtMs: it.CreatedAtMs,
			UpdatedAtMs: it.UpdatedAtMs,
			ExpiresAtMs: it.ExpiresAtMs,
			Revision:    it.Revision,
			Metadata:    it.Metadata,
			Source:      it.Source,
			Score:       it.Score,
		}
		if it.AgentID != nil {
			sm.AgentID = *it.AgentID
		}
		if it.UserID != nil {
			sm.UserID = *it.UserID
		}
		if it.Explanation != nil {
			sm.Explanation = &Explanation{
				TotalScore:         it.Explanation.TotalScore,
				SimilarityScore:    it.Explanation.SimilarityScore,
				ImportanceScore:    it.Explanation.ImportanceScore,
				RecencyScore:       it.Explanation.RecencyScore,
				ReinforcementScore: it.Explanation.ReinforcementScore,
				PolicyNotes:        it.Explanation.PolicyNotes,
			}
		}
		items = append(items, sm)
	}
	return &RecallResult{TotalCandidates: raw.TotalCandidates, Items: items}
}

// floatPtr returns a pointer to vector's backing array (or nil for an empty
// slice) suitable for passing to a `const float*` C parameter, plus its
// length. C.float and Go's float32 share an identical IEEE-754 layout on
// every platform this module targets, so no copy is needed.
func floatPtr(vector []float32) (*C.float, C.size_t) {
	if len(vector) == 0 {
		return nil, 0
	}
	return (*C.float)(unsafe.Pointer(&vector[0])), C.size_t(len(vector))
}

// -- Store ----------------------------------------------------------------------------------

// Store is a durable, namespace-isolated long-term memory store backed by
// Vivvy's Rust engine. A *Store is safe for concurrent use by multiple
// goroutines: Insert, Recall, FormatContext, Backup and Vacuum may all be
// called concurrently on the same Store, guarded on the Go side by an
// RWMutex and on the Rust side by the store's own internal locking. Close
// must be called exactly once, after which every other method returns
// ErrClosed.
type Store struct {
	mu             sync.RWMutex
	ptr            *C.VivvyStore
	closed         bool
	dimensions     int
	embeddingModel string
}

// Index is the low-level vector-search API. It is independent from Store:
// it has no memory scopes, full-text search, or revision management.
type Index struct {
	mu     sync.RWMutex
	ptr    *C.VivvyIndex
	closed bool
}

// OpenIndex opens an in-memory or durable low-level vector index.
func OpenIndex(cfg IndexConfig) (*Index, error) {
	if cfg.Dimensions <= 0 {
		return nil, fmt.Errorf("vivvy: IndexConfig.Dimensions must be > 0")
	}
	if cfg.Metric == "" {
		cfg.Metric = "l2"
	}
	payload, err := json.Marshal(indexConfigJSON{
		Dimensions: cfg.Dimensions,
		Metric:     cfg.Metric,
		WALPath:    nonEmpty(cfg.WALPath),
		DataDir:    nonEmpty(cfg.DataDir),
	})
	if err != nil {
		return nil, fmt.Errorf("vivvy: marshal index config: %w", err)
	}
	cConfig := C.CString(string(payload))
	defer C.free(unsafe.Pointer(cConfig))
	var handle *C.VivvyIndex
	if code := C.vivvy_index_open(cConfig, &handle); code != 0 {
		return nil, lastError(code)
	}
	index := &Index{ptr: handle}
	runtime.SetFinalizer(index, (*Index).finalize)
	return index, nil
}

func (i *Index) withHandle(fn func(*C.VivvyIndex) error) error {
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.closed {
		return ErrClosed
	}
	return fn(i.ptr)
}

// Close stops index background work and releases its native resources.
func (i *Index) Close() error {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.closed {
		return ErrClosed
	}
	C.vivvy_index_close(i.ptr)
	i.ptr = nil
	i.closed = true
	runtime.SetFinalizer(i, nil)
	return nil
}

func (i *Index) finalize() {
	i.mu.Lock()
	defer i.mu.Unlock()
	if !i.closed {
		C.vivvy_index_close(i.ptr)
		i.ptr = nil
		i.closed = true
	}
}

// Insert adds one vector, optionally with string metadata used by Filtered
// Search, and returns its auto-generated ID.
func (i *Index) Insert(vector []float32, metadata map[string]string) (uint64, error) {
	var id uint64
	err := i.withHandle(func(h *C.VivvyIndex) error {
		var cMetadata *C.char
		if len(metadata) > 0 {
			payload, mErr := json.Marshal(metadata)
			if mErr != nil {
				return fmt.Errorf("vivvy: marshal metadata: %w", mErr)
			}
			cMetadata = C.CString(string(payload))
			defer C.free(unsafe.Pointer(cMetadata))
		}
		vecPtr, vecLen := floatPtr(vector)
		var outID C.uint64_t
		if code := C.vivvy_index_insert(h, vecPtr, vecLen, cMetadata, &outID); code != 0 {
			return lastError(code)
		}
		runtime.KeepAlive(vector)
		id = uint64(outID)
		return nil
	})
	return id, err
}

// InsertBatch atomically inserts vectors with auto-generated IDs. Metadata is
// intentionally unavailable on this fast path; use Insert when filtering
// fields are needed.
func (i *Index) InsertBatch(vectors [][]float32) ([]uint64, error) {
	if len(vectors) == 0 {
		return []uint64{}, nil
	}
	var ids []uint64
	err := i.withHandle(func(h *C.VivvyIndex) error {
		dim := len(vectors[0])
		flat := make([]float32, 0, dim*len(vectors))
		for _, vector := range vectors {
			if len(vector) != dim {
				return fmt.Errorf("vivvy: all batch vectors must have the same dimension")
			}
			flat = append(flat, vector...)
		}
		vectorPtr, _ := floatPtr(flat)
		var outJSON *C.char
		if code := C.vivvy_index_insert_batch(h, vectorPtr, C.size_t(dim), C.size_t(len(vectors)), &outJSON); code != 0 {
			return lastError(code)
		}
		runtime.KeepAlive(flat)
		defer C.vivvy_free_string(outJSON)
		if uErr := json.Unmarshal([]byte(C.GoString(outJSON)), &ids); uErr != nil {
			return fmt.Errorf("vivvy: unmarshal index batch IDs: %w", uErr)
		}
		return nil
	})
	return ids, err
}

// InsertWithID inserts or replaces a vector with an application-supplied ID.
func (i *Index) InsertWithID(id uint64, vector []float32) error {
	return i.withHandle(func(h *C.VivvyIndex) error {
		vecPtr, vecLen := floatPtr(vector)
		if code := C.vivvy_index_insert_with_id(h, C.uint64_t(id), vecPtr, vecLen); code != 0 {
			return lastError(code)
		}
		runtime.KeepAlive(vector)
		return nil
	})
}

// Delete soft-deletes an ID and reports whether it existed in the mutable
// portion of the index.
func (i *Index) Delete(id uint64) (bool, error) {
	var deleted bool
	err := i.withHandle(func(h *C.VivvyIndex) error {
		var outDeleted C.int
		if code := C.vivvy_index_delete(h, C.uint64_t(id), &outDeleted); code != 0 {
			return lastError(code)
		}
		deleted = outDeleted != 0
		return nil
	})
	return deleted, err
}

// Search returns up to k nearest vectors. Pass nil for an unfiltered search.
func (i *Index) Search(query []float32, k int, filter *Filter) ([]SearchResult, error) {
	if k <= 0 {
		return []SearchResult{}, nil
	}
	var results []SearchResult
	err := i.withHandle(func(h *C.VivvyIndex) error {
		var cFilter *C.char
		if filter != nil {
			payload, mErr := json.Marshal(filter)
			if mErr != nil {
				return fmt.Errorf("vivvy: marshal filter: %w", mErr)
			}
			cFilter = C.CString(string(payload))
			defer C.free(unsafe.Pointer(cFilter))
		}
		queryPtr, queryLen := floatPtr(query)
		var outJSON *C.char
		if code := C.vivvy_index_search(h, queryPtr, queryLen, C.size_t(k), cFilter, &outJSON); code != 0 {
			return lastError(code)
		}
		runtime.KeepAlive(query)
		defer C.vivvy_free_string(outJSON)
		if uErr := json.Unmarshal([]byte(C.GoString(outJSON)), &results); uErr != nil {
			return fmt.Errorf("vivvy: unmarshal index search results: %w", uErr)
		}
		return nil
	})
	return results, err
}

// Compact immediately compacts mutable vectors into the current index form.
func (i *Index) Compact() error {
	return i.withHandle(func(h *C.VivvyIndex) error {
		if code := C.vivvy_index_compact(h); code != 0 {
			return lastError(code)
		}
		return nil
	})
}

// Open opens (or creates) a durable memory store at path.
func Open(path string, cfg OpenConfig) (*Store, error) {
	if cfg.Dimensions <= 0 {
		return nil, fmt.Errorf("vivvy: OpenConfig.Dimensions must be > 0")
	}
	if cfg.EmbeddingModel == "" {
		return nil, fmt.Errorf("vivvy: OpenConfig.EmbeddingModel must not be empty")
	}

	cfgJSON, err := json.Marshal(openConfigJSON{
		Dimensions:     cfg.Dimensions,
		EmbeddingModel: cfg.EmbeddingModel,
		MaxRecallLimit: cfg.MaxRecallLimit,
	})
	if err != nil {
		return nil, fmt.Errorf("vivvy: marshal config: %w", err)
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	cCfg := C.CString(string(cfgJSON))
	defer C.free(unsafe.Pointer(cCfg))

	var handle *C.VivvyStore
	code := C.vivvy_store_open(cPath, cCfg, &handle)
	if code != 0 {
		return nil, lastError(code)
	}

	s := &Store{ptr: handle, dimensions: cfg.Dimensions, embeddingModel: cfg.EmbeddingModel}
	// Backstop for callers that forget to Close explicitly; Close cancels
	// this finalizer so the common, correct path never touches the GC.
	runtime.SetFinalizer(s, (*Store).finalize)
	return s, nil
}

// withHandle serializes against Close (via RLock, so concurrent operations
// still run in parallel with each other) and rejects calls made after Close.
func (s *Store) withHandle(fn func(*C.VivvyStore) error) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return ErrClosed
	}
	return fn(s.ptr)
}

// Close releases all resources held by the store. Safe to call at most
// once; subsequent calls return ErrClosed.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrClosed
	}
	C.vivvy_store_close(s.ptr)
	s.ptr = nil
	s.closed = true
	runtime.SetFinalizer(s, nil)
	return nil
}

func (s *Store) finalize() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.closed {
		C.vivvy_store_close(s.ptr)
		s.ptr = nil
		s.closed = true
	}
}

// Insert stores a new memory record with its embedding and returns the
// newly assigned record ID.
func (s *Store) Insert(rec MemoryRecord, vector []float32) (string, error) {
	var id string
	err := s.withHandle(func(h *C.VivvyStore) error {
		kind := string(rec.Kind)
		if kind == "" {
			kind = string(KindFact)
		}

		payload, mErr := json.Marshal(insertRecordJSON{
			OperationID: nonEmpty(rec.OperationID),
			TenantID:    rec.TenantID,
			Namespace:   rec.Namespace,
			AgentID:     nonEmpty(rec.AgentID),
			UserID:      nonEmpty(rec.UserID),
			Content:     rec.Content,
			Kind:        kind,
			Importance:  rec.Importance,
			ExpiresAtMs: rec.ExpiresAtMs,
			Metadata:    rec.Metadata,
			Source:      rec.Source,
		})
		if mErr != nil {
			return fmt.Errorf("vivvy: marshal record: %w", mErr)
		}

		cRecord := C.CString(string(payload))
		defer C.free(unsafe.Pointer(cRecord))
		vecPtr, vecLen := floatPtr(vector)

		var outID *C.char
		code := C.vivvy_store_insert(h, cRecord, vecPtr, vecLen, &outID)
		runtime.KeepAlive(vector)
		if code != 0 {
			return lastError(code)
		}
		defer C.vivvy_free_string(outID)
		id = C.GoString(outID)
		return nil
	})
	return id, err
}

// InsertBatch stores records in one atomic transaction. Each vector must
// correspond to the record at the same index.
func (s *Store) InsertBatch(records []MemoryRecord, vectors [][]float32) ([]string, error) {
	if len(records) != len(vectors) {
		return nil, fmt.Errorf("vivvy: records and vectors must have the same length")
	}
	if len(records) == 0 {
		return []string{}, nil
	}

	var ids []string
	err := s.withHandle(func(h *C.VivvyStore) error {
		dim := len(vectors[0])
		flat := make([]float32, 0, len(records)*dim)
		payloads := make([]insertRecordJSON, 0, len(records))
		for i, rec := range records {
			if len(vectors[i]) != dim {
				return fmt.Errorf("vivvy: all batch vectors must have the same dimension")
			}
			kind := string(rec.Kind)
			if kind == "" {
				kind = string(KindFact)
			}
			payloads = append(payloads, insertRecordJSON{
				OperationID: nonEmpty(rec.OperationID),
				TenantID:    rec.TenantID,
				Namespace:   rec.Namespace,
				AgentID:     nonEmpty(rec.AgentID),
				UserID:      nonEmpty(rec.UserID),
				Content:     rec.Content,
				Kind:        kind,
				Importance:  rec.Importance,
				ExpiresAtMs: rec.ExpiresAtMs,
				Metadata:    rec.Metadata,
				Source:      rec.Source,
			})
			flat = append(flat, vectors[i]...)
		}
		payload, mErr := json.Marshal(payloads)
		if mErr != nil {
			return fmt.Errorf("vivvy: marshal batch records: %w", mErr)
		}
		cRecords := C.CString(string(payload))
		defer C.free(unsafe.Pointer(cRecords))
		vecPtr, _ := floatPtr(flat)

		var outIDs *C.char
		code := C.vivvy_store_insert_batch(h, cRecords, vecPtr, C.size_t(dim), C.size_t(len(records)), &outIDs)
		runtime.KeepAlive(flat)
		if code != 0 {
			return lastError(code)
		}
		defer C.vivvy_free_string(outIDs)
		if uErr := json.Unmarshal([]byte(C.GoString(outIDs)), &ids); uErr != nil {
			return fmt.Errorf("vivvy: unmarshal batch IDs: %w", uErr)
		}
		return nil
	})
	return ids, err
}

// Get fetches an active record by ID within a scope. It returns nil, nil when
// the record does not exist or has been forgotten.
func (s *Store) Get(scope Scope, id string) (*MemoryRecord, error) {
	var record *MemoryRecord
	err := s.withHandle(func(h *C.VivvyStore) error {
		payload, mErr := marshalScope(scope)
		if mErr != nil {
			return fmt.Errorf("vivvy: marshal scope: %w", mErr)
		}
		cScope := C.CString(string(payload))
		defer C.free(unsafe.Pointer(cScope))
		cID := C.CString(id)
		defer C.free(unsafe.Pointer(cID))

		var outJSON *C.char
		code := C.vivvy_store_get(h, cScope, cID, &outJSON)
		if code != 0 {
			return lastError(code)
		}
		defer C.vivvy_free_string(outJSON)
		if C.GoString(outJSON) == "null" {
			return nil
		}
		var raw memoryRecordJSON
		if uErr := json.Unmarshal([]byte(C.GoString(outJSON)), &raw); uErr != nil {
			return fmt.Errorf("vivvy: unmarshal record: %w", uErr)
		}
		record = convertMemoryRecord(raw)
		return nil
	})
	return record, err
}

// Update atomically updates a record when ExpectedRevision matches. An empty
// vector preserves the current embedding; a non-empty vector replaces it.
func (s *Store) Update(req UpdateRequest, vector []float32) error {
	if req.ClearExpiresAt && req.ExpiresAtMs != nil {
		return fmt.Errorf("vivvy: ClearExpiresAt and ExpiresAtMs cannot both be set")
	}
	return s.withHandle(func(h *C.VivvyStore) error {
		payload, mErr := marshalUpdate(req)
		if mErr != nil {
			return fmt.Errorf("vivvy: marshal update: %w", mErr)
		}
		cUpdate := C.CString(string(payload))
		defer C.free(unsafe.Pointer(cUpdate))
		vecPtr, vecLen := floatPtr(vector)
		code := C.vivvy_store_update(h, cUpdate, vecPtr, vecLen)
		runtime.KeepAlive(vector)
		if code != 0 {
			return lastError(code)
		}
		return nil
	})
}

// Forget soft-deletes one record, immediately hiding it from Get and Recall.
func (s *Store) Forget(scope Scope, id string) error {
	return s.withHandle(func(h *C.VivvyStore) error {
		payload, mErr := marshalScope(scope)
		if mErr != nil {
			return fmt.Errorf("vivvy: marshal scope: %w", mErr)
		}
		cScope := C.CString(string(payload))
		defer C.free(unsafe.Pointer(cScope))
		cID := C.CString(id)
		defer C.free(unsafe.Pointer(cID))
		if code := C.vivvy_store_forget(h, cScope, cID); code != 0 {
			return lastError(code)
		}
		return nil
	})
}

// ForgetBatch soft-deletes IDs in a scope as one transaction.
func (s *Store) ForgetBatch(scope Scope, ids []string) error {
	return s.withHandle(func(h *C.VivvyStore) error {
		payload, mErr := marshalScope(scope)
		if mErr != nil {
			return fmt.Errorf("vivvy: marshal scope: %w", mErr)
		}
		idsJSON, mErr := json.Marshal(ids)
		if mErr != nil {
			return fmt.Errorf("vivvy: marshal IDs: %w", mErr)
		}
		cScope := C.CString(string(payload))
		defer C.free(unsafe.Pointer(cScope))
		cIDs := C.CString(string(idsJSON))
		defer C.free(unsafe.Pointer(cIDs))
		if code := C.vivvy_store_forget_batch(h, cScope, cIDs); code != 0 {
			return lastError(code)
		}
		return nil
	})
}

// Health returns a privacy-safe operational health snapshot.
func (s *Store) Health() (*StoreHealth, error) {
	var health *StoreHealth
	err := s.withHandle(func(h *C.VivvyStore) error {
		var outJSON *C.char
		if code := C.vivvy_store_health(h, &outJSON); code != 0 {
			return lastError(code)
		}
		defer C.vivvy_free_string(outJSON)
		var raw storeHealthJSON
		if uErr := json.Unmarshal([]byte(C.GoString(outJSON)), &raw); uErr != nil {
			return fmt.Errorf("vivvy: unmarshal health: %w", uErr)
		}
		health = &StoreHealth{
			IsHealthy: raw.IsHealthy, TotalActiveRecords: raw.TotalActiveRecords,
			TotalTombstonedRecords: raw.TotalTombstonedRecords,
			PendingOperationsCount: raw.PendingOperationsCount,
			IndexRebuildRequired:   raw.IndexRebuildRequired,
			DBSizeBytes:            raw.DBSizeBytes, WALSizeBytes: raw.WALSizeBytes,
		}
		return nil
	})
	return health, err
}

// RebuildIndex rebuilds the in-memory vector accelerator from durable active
// records.
func (s *Store) RebuildIndex() error {
	return s.withHandle(func(h *C.VivvyStore) error {
		if code := C.vivvy_store_rebuild_index(h); code != 0 {
			return lastError(code)
		}
		return nil
	})
}

// Recall runs hybrid (vector + optional full-text) search and returns
// scored, filtered matches.
func (s *Store) Recall(opts RecallOptions, vector []float32) (*RecallResult, error) {
	var recalled *RecallResult
	err := s.withHandle(func(h *C.VivvyStore) error {
		optsJSON, mErr := marshalRecallOptions(opts)
		if mErr != nil {
			return mErr
		}
		cOpts := C.CString(string(optsJSON))
		defer C.free(unsafe.Pointer(cOpts))

		var cQueryText *C.char
		if opts.QueryText != "" {
			cQueryText = C.CString(opts.QueryText)
			defer C.free(unsafe.Pointer(cQueryText))
		}

		vecPtr, vecLen := floatPtr(vector)
		topK := opts.TopK
		if topK <= 0 {
			topK = 10
		}

		var outJSON *C.char
		code := C.vivvy_store_recall(h, cOpts, vecPtr, vecLen, cQueryText, C.size_t(topK), &outJSON)
		runtime.KeepAlive(vector)
		if code != 0 {
			return lastError(code)
		}
		defer C.vivvy_free_string(outJSON)

		var raw recallResultsJSON
		if uErr := json.Unmarshal([]byte(C.GoString(outJSON)), &raw); uErr != nil {
			return fmt.Errorf("vivvy: unmarshal recall results: %w", uErr)
		}
		recalled = convertRecallResults(raw)
		return nil
	})
	return recalled, err
}

// FormatContext runs the same recall as Recall and renders the results into
// a token-budgeted string ready for direct LLM prompt injection.
func (s *Store) FormatContext(opts RecallOptions, vector []float32, fmtOpts FormatOptions) (string, error) {
	var out string
	err := s.withHandle(func(h *C.VivvyStore) error {
		optsJSON, mErr := marshalRecallOptions(opts)
		if mErr != nil {
			return mErr
		}
		cOpts := C.CString(string(optsJSON))
		defer C.free(unsafe.Pointer(cOpts))

		var cQueryText *C.char
		if opts.QueryText != "" {
			cQueryText = C.CString(opts.QueryText)
			defer C.free(unsafe.Pointer(cQueryText))
		}

		vecPtr, vecLen := floatPtr(vector)
		topK := opts.TopK
		if topK <= 0 {
			topK = 10
		}

		fmtPayload, fErr := json.Marshal(formatOptionsJSON{
			MaxTokens: fmtOpts.MaxTokens,
			Template:  nonEmpty(fmtOpts.Template),
			Header:    fmtOpts.Header,
			Footer:    fmtOpts.Footer,
		})
		if fErr != nil {
			return fmt.Errorf("vivvy: marshal format options: %w", fErr)
		}
		cFmt := C.CString(string(fmtPayload))
		defer C.free(unsafe.Pointer(cFmt))

		var outContext *C.char
		code := C.vivvy_store_format_context(h, cOpts, vecPtr, vecLen, cQueryText, C.size_t(topK), cFmt, &outContext)
		runtime.KeepAlive(vector)
		if code != 0 {
			return lastError(code)
		}
		defer C.vivvy_free_string(outContext)
		out = C.GoString(outContext)
		return nil
	})
	return out, err
}

// Backup creates a zero-downtime, crash-consistent copy of the store at
// targetPath (created if it does not exist).
func (s *Store) Backup(targetPath string) error {
	return s.withHandle(func(h *C.VivvyStore) error {
		cPath := C.CString(targetPath)
		defer C.free(unsafe.Pointer(cPath))
		code := C.vivvy_store_backup(h, cPath)
		if code != 0 {
			return lastError(code)
		}
		return nil
	})
}

// Vacuum physically purges up to batchSize tombstoned/expired records and
// rebuilds the vector accelerator if anything was purged. It returns the
// number of records removed.
func (s *Store) Vacuum(batchSize int) (int, error) {
	var purged int
	err := s.withHandle(func(h *C.VivvyStore) error {
		var outPurged C.size_t
		code := C.vivvy_store_vacuum(h, C.size_t(batchSize), &outPurged)
		if code != 0 {
			return lastError(code)
		}
		purged = int(outPurged)
		return nil
	})
	return purged, err
}
