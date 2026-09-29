package vivvy

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

func f32ptr(v float32) *float32 { return &v }

func openTestStore(t *testing.T, dims int) *Store {
	t.Helper()
	dir := t.TempDir()
	s, err := Open(dir, OpenConfig{Dimensions: dims, EmbeddingModel: "test-model"})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil && !errors.Is(err, ErrClosed) {
			t.Errorf("Close: %v", err)
		}
	})
	return s
}

// -- store creation -------------------------------------------------------------------------

func TestOpenCreatesStoreDirectory(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "nested", "store")
	s, err := Open(dir, OpenConfig{Dimensions: 3, EmbeddingModel: "test-model"})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer s.Close()

	if _, err := os.Stat(filepath.Join(dir, "memory.db")); err != nil {
		t.Fatalf("expected memory.db to exist: %v", err)
	}
}

func TestOpenRejectsInvalidConfig(t *testing.T) {
	if _, err := Open(t.TempDir(), OpenConfig{Dimensions: 0, EmbeddingModel: "m"}); err == nil {
		t.Fatal("expected error for zero dimensions")
	}
	if _, err := Open(t.TempDir(), OpenConfig{Dimensions: 3, EmbeddingModel: ""}); err == nil {
		t.Fatal("expected error for empty embedding model")
	}
}

func TestCloseIsIdempotentAndRejectsFurtherUse(t *testing.T) {
	s, err := Open(t.TempDir(), OpenConfig{Dimensions: 3, EmbeddingModel: "test-model"})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := s.Close(); !errors.Is(err, ErrClosed) {
		t.Fatalf("second Close: got %v, want ErrClosed", err)
	}
	if _, err := s.Insert(MemoryRecord{TenantID: "t", Namespace: "n", Content: "x"}, []float32{1, 2, 3}); !errors.Is(err, ErrClosed) {
		t.Fatalf("Insert after close: got %v, want ErrClosed", err)
	}
	if _, err := s.Recall(RecallOptions{TenantID: "t", Namespace: "n", TopK: 1}, []float32{1, 2, 3}); !errors.Is(err, ErrClosed) {
		t.Fatalf("Recall after close: got %v, want ErrClosed", err)
	}
}

// -- insertion & vector search ---------------------------------------------------------------

func TestInsertAndVectorSearch(t *testing.T) {
	s := openTestStore(t, 3)

	seeds := []struct {
		content string
		vec     []float32
	}{
		{"apple", []float32{1, 0, 0}},
		{"banana", []float32{0, 1, 0}},
		{"cherry", []float32{0, 0, 1}},
	}
	for _, sd := range seeds {
		id, err := s.Insert(MemoryRecord{
			TenantID: "acme", Namespace: "default",
			Content: sd.content, Kind: KindFact, Importance: f32ptr(0.5),
		}, sd.vec)
		if err != nil {
			t.Fatalf("Insert(%q): %v", sd.content, err)
		}
		if id == "" {
			t.Fatalf("Insert(%q): expected non-empty id", sd.content)
		}
	}

	res, err := s.Recall(RecallOptions{
		TenantID: "acme", Namespace: "default", TopK: 1,
	}, []float32{0.9, 0.1, 0})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(res.Items) != 1 {
		t.Fatalf("expected 1 result, got %d", len(res.Items))
	}
	if res.Items[0].Content != "apple" {
		t.Fatalf("expected nearest neighbor 'apple', got %q", res.Items[0].Content)
	}
}

func TestDimensionMismatchReturnsTypedError(t *testing.T) {
	s := openTestStore(t, 3)
	_, err := s.Insert(MemoryRecord{TenantID: "acme", Namespace: "default", Content: "x"}, []float32{1, 2})
	var ferr *Error
	if !errors.As(err, &ferr) {
		t.Fatalf("expected *vivvy.Error, got %T: %v", err, err)
	}
	if ferr.Code != ErrCodeDimensionMismatch {
		t.Fatalf("expected ErrCodeDimensionMismatch, got %d (%s)", ferr.Code, ferr.Message)
	}
}

func TestInvalidScopeReturnsTypedError(t *testing.T) {
	s := openTestStore(t, 2)
	_, err := s.Insert(MemoryRecord{TenantID: "", Namespace: "default", Content: "x"}, []float32{1, 1})
	var ferr *Error
	if !errors.As(err, &ferr) || ferr.Code != ErrCodeInvalidScope {
		t.Fatalf("expected ErrCodeInvalidScope, got %v", err)
	}
}

// -- hybrid recall & filtering ----------------------------------------------------------------

func TestHybridRecallWithQueryText(t *testing.T) {
	s := openTestStore(t, 3)

	insert := func(content string, vec []float32) {
		t.Helper()
		if _, err := s.Insert(MemoryRecord{
			TenantID: "acme", Namespace: "default", Content: content,
		}, vec); err != nil {
			t.Fatalf("Insert(%q): %v", content, err)
		}
	}
	insert("the quick brown fox", []float32{1, 0, 0})
	insert("a slow green turtle", []float32{0, 1, 0})

	// Query vector is deliberately far from both seeds; only the FTS side of
	// the hybrid RRF fusion should be able to surface "fox".
	res, err := s.Recall(RecallOptions{
		TenantID: "acme", Namespace: "default",
		QueryText: "fox", TopK: 5, IncludeExplanations: true,
	}, []float32{0, 0, 1})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	var found bool
	for _, it := range res.Items {
		if it.Content == "the quick brown fox" {
			found = true
			if it.Explanation == nil {
				t.Error("expected explanation to be populated when IncludeExplanations is true")
			}
		}
	}
	if !found {
		t.Fatalf("expected FTS match for 'fox', got %+v", res.Items)
	}
}

func TestRecallFilters(t *testing.T) {
	s := openTestStore(t, 2)

	mustInsert := func(kind MemoryKind, importance float32, meta map[string]any) {
		t.Helper()
		if _, err := s.Insert(MemoryRecord{
			TenantID: "acme", Namespace: "default", Content: "content",
			Kind: kind, Importance: f32ptr(importance), Metadata: meta,
		}, []float32{1, 1}); err != nil {
			t.Fatalf("Insert: %v", err)
		}
	}
	mustInsert(KindFact, 0.9, map[string]any{"team": "infra"})
	mustInsert(KindPreference, 0.2, map[string]any{"team": "growth"})

	res, err := s.Recall(RecallOptions{
		TenantID: "acme", Namespace: "default", TopK: 10,
		Kinds: []MemoryKind{KindFact}, MinImportance: f32ptr(0.5),
	}, []float32{1, 1})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(res.Items) != 1 || res.Items[0].Kind != KindFact {
		t.Fatalf("expected exactly 1 Fact result, got %+v", res.Items)
	}
}

func TestFormatContext(t *testing.T) {
	s := openTestStore(t, 2)
	if _, err := s.Insert(MemoryRecord{
		TenantID: "acme", Namespace: "default", Content: "remember the milk",
		Kind: KindInstruction, Importance: f32ptr(0.8),
	}, []float32{1, 0}); err != nil {
		t.Fatalf("Insert: %v", err)
	}

	maxTokens := 500
	text, err := s.FormatContext(RecallOptions{
		TenantID: "acme", Namespace: "default", TopK: 5,
	}, []float32{1, 0}, FormatOptions{MaxTokens: &maxTokens})
	if err != nil {
		t.Fatalf("FormatContext: %v", err)
	}
	if !strings.Contains(text, "remember the milk") {
		t.Fatalf("expected formatted context to contain the memory, got %q", text)
	}
}

// -- backup & vacuum --------------------------------------------------------------------------

func TestBackupAndReopen(t *testing.T) {
	s, err := Open(t.TempDir(), OpenConfig{Dimensions: 2, EmbeddingModel: "test-model"})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer s.Close()

	if _, err := s.Insert(MemoryRecord{
		TenantID: "acme", Namespace: "default", Content: "backed up memory",
	}, []float32{1, 1}); err != nil {
		t.Fatalf("Insert: %v", err)
	}

	backupDir := filepath.Join(t.TempDir(), "backup")
	if err := s.Backup(backupDir); err != nil {
		t.Fatalf("Backup: %v", err)
	}

	restored, err := Open(backupDir, OpenConfig{Dimensions: 2, EmbeddingModel: "test-model"})
	if err != nil {
		t.Fatalf("Open(backup): %v", err)
	}
	defer restored.Close()

	res, err := restored.Recall(RecallOptions{TenantID: "acme", Namespace: "default", TopK: 5}, []float32{1, 1})
	if err != nil {
		t.Fatalf("Recall(backup): %v", err)
	}
	if len(res.Items) != 1 || res.Items[0].Content != "backed up memory" {
		t.Fatalf("expected restored memory, got %+v", res.Items)
	}
}

func TestVacuumOnCleanStoreIsNoop(t *testing.T) {
	s := openTestStore(t, 2)
	if _, err := s.Insert(MemoryRecord{TenantID: "acme", Namespace: "default", Content: "x"}, []float32{1, 1}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	purged, err := s.Vacuum(100)
	if err != nil {
		t.Fatalf("Vacuum: %v", err)
	}
	if purged != 0 {
		t.Fatalf("expected 0 purged on a store with no tombstones, got %d", purged)
	}
}

func TestVacuumPurgesExpiredRecords(t *testing.T) {
	s := openTestStore(t, 2)
	past := time.Now().Add(-time.Hour).UnixMilli()
	if _, err := s.Insert(MemoryRecord{
		TenantID: "acme", Namespace: "default", Content: "expired",
		ExpiresAtMs: &past,
	}, []float32{1, 1}); err != nil {
		t.Fatalf("Insert: %v", err)
	}

	purged, err := s.Vacuum(100)
	if err != nil {
		t.Fatalf("Vacuum: %v", err)
	}
	if purged != 1 {
		t.Fatalf("expected 1 purged expired record, got %d", purged)
	}
}

// -- concurrency -------------------------------------------------------------------------------

func TestConcurrentInsertAndRecall(t *testing.T) {
	s := openTestStore(t, 4)

	const workers = 16
	const perWorker = 25

	var wg sync.WaitGroup
	errCh := make(chan error, workers*perWorker*2)

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < perWorker; i++ {
				vec := []float32{float32(w), float32(i), 0, 0}
				content := fmt.Sprintf("worker-%d-item-%d", w, i)

				id, err := s.Insert(MemoryRecord{
					TenantID: "acme", Namespace: "default", Content: content,
				}, vec)
				if err != nil {
					errCh <- fmt.Errorf("insert w=%d i=%d: %w", w, i, err)
					continue
				}
				if id == "" {
					errCh <- fmt.Errorf("insert w=%d i=%d: empty id", w, i)
				}

				if _, err := s.Recall(RecallOptions{
					TenantID: "acme", Namespace: "default", TopK: 3,
				}, vec); err != nil {
					errCh <- fmt.Errorf("recall w=%d i=%d: %w", w, i, err)
				}
			}
		}(w)
	}

	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}
}

// TestConcurrentCloseIsSafe races Close against in-flight operations to
// exercise Store's RWMutex: every Insert must either complete against a
// still-valid handle or observe ErrClosed — never touch a freed pointer.
// Run with -race to get real assurance from this test.
func TestConcurrentCloseIsSafe(t *testing.T) {
	s, err := Open(t.TempDir(), OpenConfig{Dimensions: 2, EmbeddingModel: "test-model"})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := s.Insert(MemoryRecord{TenantID: "acme", Namespace: "default", Content: "x"}, []float32{1, 1})
			if err != nil && !errors.Is(err, ErrClosed) {
				var ferr *Error
				if !errors.As(err, &ferr) {
					t.Errorf("unexpected error type from racing Insert: %v", err)
				}
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = s.Close()
	}()
	wg.Wait()
}

// -- resource teardown / leak smoke test --------------------------------------------------------

// TestMemoryStressNoUnboundedGrowth performs many insert/recall cycles and
// checks process RSS does not grow without bound. This is a coarse smoke
// test, not a substitute for running the Go test binary under valgrind or
// ASan in CI (see the repository Makefile's `leak-check` target) — but a
// per-call leak of even a few hundred bytes (e.g. a forgotten
// vivvy_free_string) would dwarf the threshold below over this many
// iterations.
func TestMemoryStressNoUnboundedGrowth(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in -short mode")
	}
	s := openTestStore(t, 8)

	vec := make([]float32, 8)
	for i := range vec {
		vec[i] = float32(i) / 8
	}

	readRSSKB := func() (uint64, bool) {
		data, err := os.ReadFile("/proc/self/status")
		if err != nil {
			return 0, false
		}
		for _, line := range strings.Split(string(data), "\n") {
			if !strings.HasPrefix(line, "VmRSS:") {
				continue
			}
			fields := strings.Fields(line)
			if len(fields) < 2 {
				return 0, false
			}
			var kb uint64
			if _, err := fmt.Sscanf(fields[1], "%d", &kb); err != nil {
				return 0, false
			}
			return kb, true
		}
		return 0, false
	}

	// Warm up allocators/caches before measuring the baseline.
	for i := 0; i < 200; i++ {
		if _, err := s.Insert(MemoryRecord{TenantID: "acme", Namespace: "default", Content: "warmup"}, vec); err != nil {
			t.Fatalf("Insert (warmup): %v", err)
		}
	}
	runtime.GC()
	before, ok := readRSSKB()
	if !ok {
		t.Skip("VmRSS not available on this platform")
	}

	const iterations = 2000
	for i := 0; i < iterations; i++ {
		if _, err := s.Insert(MemoryRecord{
			TenantID: "acme", Namespace: "default",
			Content: fmt.Sprintf("stress item %d", i),
		}, vec); err != nil {
			t.Fatalf("Insert: %v", err)
		}
		if _, err := s.Recall(RecallOptions{
			TenantID: "acme", Namespace: "default", TopK: 5,
		}, vec); err != nil {
			t.Fatalf("Recall: %v", err)
		}
	}

	runtime.GC()
	after, ok := readRSSKB()
	if !ok {
		t.Skip("VmRSS not available on this platform")
	}

	const maxGrowthKB = 300_000 // 300 MB — a coarse net, not a tight bound; see doc comment.
	if after > before && after-before > maxGrowthKB {
		t.Fatalf("RSS grew by %d KB over %d iterations (before=%d after=%d); possible leak",
			after-before, iterations, before, after)
	}
}
