package vivvy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Embedder supplies vector embeddings for text-first Store helpers. Provider
// implementations live in Go so API keys and HTTP clients never cross the C
// ABI boundary.
type Embedder interface {
	Name() string
	Dimensions() int
	Embed(ctx context.Context, text string) ([]float32, error)
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}

func (s *Store) validateEmbedder(embedder Embedder) error {
	if embedder == nil {
		return fmt.Errorf("vivvy: embedder is required")
	}
	if embedder.Dimensions() != s.dimensions {
		return fmt.Errorf("vivvy: embedder dimensions %d do not match store dimensions %d", embedder.Dimensions(), s.dimensions)
	}
	if embedder.Name() != s.embeddingModel {
		return fmt.Errorf("vivvy: embedder model %q does not match store model %q", embedder.Name(), s.embeddingModel)
	}
	return nil
}

// RememberText embeds record.Content and stores the resulting vector.
func (s *Store) RememberText(ctx context.Context, embedder Embedder, record MemoryRecord) (string, error) {
	if err := s.validateEmbedder(embedder); err != nil {
		return "", err
	}
	vector, err := embedder.Embed(ctx, record.Content)
	if err != nil {
		return "", fmt.Errorf("vivvy: embed record: %w", err)
	}
	return s.Insert(record, vector)
}

// RecallText embeds opts.QueryText and performs hybrid recall, retaining the
// original text for the store's full-text/RRF path.
func (s *Store) RecallText(ctx context.Context, embedder Embedder, opts RecallOptions) (*RecallResult, error) {
	if err := s.validateEmbedder(embedder); err != nil {
		return nil, err
	}
	if strings.TrimSpace(opts.QueryText) == "" {
		return nil, fmt.Errorf("vivvy: RecallText requires RecallOptions.QueryText")
	}
	vector, err := embedder.Embed(ctx, opts.QueryText)
	if err != nil {
		return nil, fmt.Errorf("vivvy: embed query: %w", err)
	}
	return s.Recall(opts, vector)
}

// RememberTextBatch embeds and stores all records atomically.
func (s *Store) RememberTextBatch(ctx context.Context, embedder Embedder, records []MemoryRecord) ([]string, error) {
	if err := s.validateEmbedder(embedder); err != nil {
		return nil, err
	}
	texts := make([]string, len(records))
	for i, record := range records {
		texts[i] = record.Content
	}
	vectors, err := embedder.EmbedBatch(ctx, texts)
	if err != nil {
		return nil, fmt.Errorf("vivvy: embed batch: %w", err)
	}
	if len(vectors) != len(records) {
		return nil, fmt.Errorf("vivvy: embedder returned %d vectors for %d records", len(vectors), len(records))
	}
	return s.InsertBatch(records, vectors)
}

// OpenAIEmbedderConfig configures the OpenAI-compatible /v1/embeddings API.
type OpenAIEmbedderConfig struct {
	APIKey     string
	Model      string
	Dimensions int
	Endpoint   string
	HTTPClient *http.Client
}

// OpenAIEmbedder uses the standard OpenAI embeddings HTTP API.
type OpenAIEmbedder struct{ config OpenAIEmbedderConfig }

func NewOpenAIEmbedder(config OpenAIEmbedderConfig) (*OpenAIEmbedder, error) {
	if config.APIKey == "" || config.Model == "" || config.Dimensions <= 0 {
		return nil, fmt.Errorf("vivvy: OpenAI API key, model, and dimensions are required")
	}
	if config.Endpoint == "" {
		config.Endpoint = "https://api.openai.com/v1/embeddings"
	}
	if config.HTTPClient == nil {
		config.HTTPClient = http.DefaultClient
	}
	return &OpenAIEmbedder{config: config}, nil
}

func (e *OpenAIEmbedder) Name() string    { return e.config.Model }
func (e *OpenAIEmbedder) Dimensions() int { return e.config.Dimensions }
func (e *OpenAIEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	vectors, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return vectors[0], nil
}
func (e *OpenAIEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	return embedOpenAICompatible(ctx, e.config.HTTPClient, e.config.Endpoint, e.config.APIKey, e.config.Model, texts, e.config.Dimensions)
}

// OllamaEmbedderConfig configures Ollama's /api/embed endpoint.
type OllamaEmbedderConfig struct {
	Model      string
	Dimensions int
	Endpoint   string
	HTTPClient *http.Client
}

// OllamaEmbedder uses Ollama's local embedding endpoint.
type OllamaEmbedder struct{ config OllamaEmbedderConfig }

func NewOllamaEmbedder(config OllamaEmbedderConfig) (*OllamaEmbedder, error) {
	if config.Model == "" || config.Dimensions <= 0 {
		return nil, fmt.Errorf("vivvy: Ollama model and dimensions are required")
	}
	if config.Endpoint == "" {
		config.Endpoint = "http://localhost:11434/api/embed"
	}
	if config.HTTPClient == nil {
		config.HTTPClient = http.DefaultClient
	}
	return &OllamaEmbedder{config: config}, nil
}

func (e *OllamaEmbedder) Name() string    { return e.config.Model }
func (e *OllamaEmbedder) Dimensions() int { return e.config.Dimensions }
func (e *OllamaEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	vectors, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return vectors[0], nil
}
func (e *OllamaEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	payload, err := json.Marshal(map[string]any{"model": e.config.Model, "input": texts})
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, e.config.Endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := e.config.HTTPClient.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 8<<20))
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("ollama embeddings: HTTP %d", response.StatusCode)
	}
	var parsed struct {
		Embeddings [][]float32 `json:"embeddings"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, err
	}
	return validateVectors(parsed.Embeddings, len(texts), e.config.Dimensions)
}

func embedOpenAICompatible(ctx context.Context, client *http.Client, endpoint, key, model string, texts []string, dimensions int) ([][]float32, error) {
	payload, err := json.Marshal(map[string]any{"model": model, "input": texts})
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+key)
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 8<<20))
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("openai embeddings: HTTP %d", response.StatusCode)
	}
	var parsed struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, err
	}
	vectors := make([][]float32, len(texts))
	for _, item := range parsed.Data {
		if item.Index < 0 || item.Index >= len(vectors) {
			return nil, fmt.Errorf("openai embeddings: invalid result index")
		}
		vectors[item.Index] = item.Embedding
	}
	return validateVectors(vectors, len(texts), dimensions)
}

func validateVectors(vectors [][]float32, expected, dimensions int) ([][]float32, error) {
	if len(vectors) != expected {
		return nil, fmt.Errorf("embedding provider returned %d vectors for %d inputs", len(vectors), expected)
	}
	for _, vector := range vectors {
		if len(vector) != dimensions {
			return nil, fmt.Errorf("embedding provider returned dimension %d; expected %d", len(vector), dimensions)
		}
	}
	return vectors, nil
}
