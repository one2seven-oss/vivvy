package vivvy_test

// Runnable examples for pkg.go.dev: `go test` executes each Example
// function and checks its stdout against the trailing `// Output:` comment,
// so these are verified documentation, not just illustrative snippets.

import (
	"fmt"
	"os"
	"strings"

	"github.com/one2seven-oss/vivvy/go"
)

func openExampleStore(dims int) (*vivvy.Store, func()) {
	dir, err := os.MkdirTemp("", "vivvy-example-*")
	if err != nil {
		panic(err)
	}
	store, err := vivvy.Open(dir, vivvy.OpenConfig{
		Dimensions:     dims,
		EmbeddingModel: "test-model",
	})
	if err != nil {
		panic(err)
	}
	return store, func() {
		store.Close()
		os.RemoveAll(dir)
	}
}

// Example demonstrates the golden path: open a store, store an observation
// with its embedding, and recall it back by vector similarity.
func Example() {
	store, cleanup := openExampleStore(3)
	defer cleanup()

	_, err := store.Insert(vivvy.MemoryRecord{
		TenantID:  "acme",
		Namespace: "support",
		Content:   "the customer prefers email over phone",
		Kind:      vivvy.KindPreference,
	}, []float32{0.1, 0.2, 0.3})
	if err != nil {
		panic(err)
	}

	res, err := store.Recall(vivvy.RecallOptions{
		TenantID:  "acme",
		Namespace: "support",
		TopK:      5,
	}, []float32{0.1, 0.2, 0.3})
	if err != nil {
		panic(err)
	}

	fmt.Println(res.Items[0].Content)
	// Output: the customer prefers email over phone
}

// ExampleStore_Recall demonstrates hybrid recall: fusing full-text matching
// (QueryText) with vector similarity, plus filtering by Kind and a minimum
// Importance threshold.
func ExampleStore_Recall() {
	store, cleanup := openExampleStore(3)
	defer cleanup()

	insert := func(content string, kind vivvy.MemoryKind, importance float32, vec []float32) {
		if _, err := store.Insert(vivvy.MemoryRecord{
			TenantID: "acme", Namespace: "support",
			Content: content, Kind: kind, Importance: &importance,
		}, vec); err != nil {
			panic(err)
		}
	}
	insert("the customer prefers email over phone", vivvy.KindPreference, 0.9, []float32{1, 0, 0})
	insert("outage postmortem notes from last week", vivvy.KindEpisodic, 0.2, []float32{0, 1, 0})

	minImportance := float32(0.5)
	res, err := store.Recall(vivvy.RecallOptions{
		TenantID:      "acme",
		Namespace:     "support",
		QueryText:     "email",
		TopK:          5,
		Kinds:         []vivvy.MemoryKind{vivvy.KindPreference},
		MinImportance: &minImportance,
	}, []float32{1, 0, 0})
	if err != nil {
		panic(err)
	}

	fmt.Println(len(res.Items))
	// Output: 1
}

// ExampleStore_FormatContext demonstrates rendering recall results directly
// into a token-budgeted prompt string for LLM context injection.
func ExampleStore_FormatContext() {
	store, cleanup := openExampleStore(2)
	defer cleanup()

	if _, err := store.Insert(vivvy.MemoryRecord{
		TenantID: "acme", Namespace: "support",
		Content: "remember the milk", Kind: vivvy.KindInstruction,
	}, []float32{1, 0}); err != nil {
		panic(err)
	}

	maxTokens := 500
	text, err := store.FormatContext(vivvy.RecallOptions{
		TenantID: "acme", Namespace: "support", TopK: 5,
	}, []float32{1, 0}, vivvy.FormatOptions{MaxTokens: &maxTokens})
	if err != nil {
		panic(err)
	}

	fmt.Println(strings.Contains(text, "remember the milk"))
	// Output: true
}
