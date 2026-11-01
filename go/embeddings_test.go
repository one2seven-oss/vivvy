package vivvy

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

type testEmbedder struct{}

func (testEmbedder) Name() string    { return "test-model" }
func (testEmbedder) Dimensions() int { return 2 }
func (testEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	if text == "second" {
		return []float32{0, 1}, nil
	}
	return []float32{1, 0}, nil
}
func (e testEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for i, text := range texts {
		vector, err := e.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		vectors[i] = vector
	}
	return vectors, nil
}

func TestTextHelpers(t *testing.T) {
	s := openTestStore(t, 2)
	embedder := testEmbedder{}
	if _, err := s.RememberText(context.Background(), embedder, MemoryRecord{
		TenantID: "acme", Namespace: "default", Content: "first",
	}); err != nil {
		t.Fatalf("RememberText: %v", err)
	}
	result, err := s.RecallText(context.Background(), embedder, RecallOptions{
		TenantID: "acme", Namespace: "default", QueryText: "first", TopK: 1,
	})
	if err != nil {
		t.Fatalf("RecallText: %v", err)
	}
	if len(result.Items) != 1 || result.Items[0].Content != "first" {
		t.Fatalf("unexpected text recall result: %+v", result)
	}
}

func TestOpenAIAndOllamaAdapters(t *testing.T) {
	openAIServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("unexpected authorization header")
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{
			map[string]any{"index": 1, "embedding": []float32{0, 1}},
			map[string]any{"index": 0, "embedding": []float32{1, 0}},
		}})
	}))
	defer openAIServer.Close()
	openAI, err := NewOpenAIEmbedder(OpenAIEmbedderConfig{
		APIKey: "test-key", Model: "test-model", Dimensions: 2, Endpoint: openAIServer.URL,
	})
	if err != nil {
		t.Fatalf("NewOpenAIEmbedder: %v", err)
	}
	vectors, err := openAI.EmbedBatch(context.Background(), []string{"one", "two"})
	if err != nil || vectors[0][0] != 1 || vectors[1][1] != 1 {
		t.Fatalf("OpenAI vectors=%v err=%v", vectors, err)
	}

	ollamaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"embeddings": [][]float32{{1, 0}}})
	}))
	defer ollamaServer.Close()
	ollama, err := NewOllamaEmbedder(OllamaEmbedderConfig{
		Model: "test-model", Dimensions: 2, Endpoint: ollamaServer.URL,
	})
	if err != nil {
		t.Fatalf("NewOllamaEmbedder: %v", err)
	}
	vector, err := ollama.Embed(context.Background(), "one")
	if err != nil || vector[0] != 1 {
		t.Fatalf("Ollama vector=%v err=%v", vector, err)
	}
}
