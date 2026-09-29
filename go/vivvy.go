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
	mu     sync.RWMutex
	ptr    *C.VivvyStore
	closed bool
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

	s := &Store{ptr: handle}
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
