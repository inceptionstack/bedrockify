package bedrockify

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// --- Mock Converser ---

type mockConverser struct {
	converseResp *ChatResponse
	converseErr  error
	streamEvents []StreamEvent
	streamErr    error
	models       []ModelInfo
	modelsErr    error
}

func (m *mockConverser) Converse(_ context.Context, _ *ChatRequest) (*ChatResponse, error) {
	return m.converseResp, m.converseErr
}

func (m *mockConverser) ConverseStream(_ context.Context, _ *ChatRequest) (<-chan StreamEvent, error) {
	if m.streamErr != nil {
		return nil, m.streamErr
	}
	ch := make(chan StreamEvent, len(m.streamEvents))
	for _, e := range m.streamEvents {
		ch <- e
	}
	close(ch)
	return ch, nil
}

func (m *mockConverser) ListModels(_ context.Context) ([]ModelInfo, error) {
	return m.models, m.modelsErr
}

// --- Mock Embedder ---

type mockEmbedder struct {
	EmbedFunc func(ctx context.Context, text string) ([]float64, error)
}

func (m *mockEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	if m.EmbedFunc != nil {
		return m.EmbedFunc(ctx, text)
	}
	return make([]float64, 1024), nil
}

// --- Chat Handler Tests ---

func TestHandlerHealth(t *testing.T) {
	h := NewHandlerWithModel(&mockConverser{}, "test-model", "dev", "us-east-1")
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp HealthResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Status != "ok" {
		t.Errorf("expected status=ok, got %q", resp.Status)
	}
	if resp.Model != "test-model" {
		t.Errorf("expected model=test-model, got %q", resp.Model)
	}
}

func TestHandlerHealthWithEmbedder(t *testing.T) {
	h := NewHandlerFull(&mockConverser{}, &mockEmbedder{}, "chat-model", "embed-model", "dev", "us-east-1")
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp HealthResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Status != "ok" {
		t.Errorf("expected status=ok, got %q", resp.Status)
	}
	if resp.Model != "chat-model" {
		t.Errorf("expected model=chat-model, got %q", resp.Model)
	}
	if resp.EmbedModel != "embed-model" {
		t.Errorf("expected embed_model=embed-model, got %q", resp.EmbedModel)
	}
}

func TestHandlerChatCompletions_NonStreaming(t *testing.T) {
	mock := &mockConverser{
		converseResp: &ChatResponse{
			ID:     "chatcmpl-test",
			Object: "chat.completion",
			Model:  "anthropic.claude-3",
			Choices: []Choice{
				{
					Index:        0,
					Message:      Message{Role: "assistant", Content: "Hello!"},
					FinishReason: "stop",
				},
			},
			Usage: Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		},
	}

	h := NewHandlerWithModel(mock, "anthropic.claude-3", "dev", "us-east-1")
	body := `{"model":"anthropic.claude-3","messages":[{"role":"user","content":"Hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp ChatResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.ID != "chatcmpl-test" {
		t.Errorf("unexpected ID: %s", resp.ID)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(resp.Choices))
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("unexpected finish_reason: %s", resp.Choices[0].FinishReason)
	}
}

func TestHandlerChatCompletions_EmptyMessages(t *testing.T) {
	h := NewHandlerWithModel(&mockConverser{}, "test-model", "dev", "us-east-1")
	body := `{"model":"test-model","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}
}

func TestHandlerChatCompletions_DefaultModel(t *testing.T) {
	mock := &mockConverser{
		converseResp: &ChatResponse{
			ID:      "x",
			Object:  "chat.completion",
			Model:   "default-model",
			Choices: []Choice{{Message: Message{Role: "assistant", Content: "hi"}, FinishReason: "stop"}},
		},
	}

	h := NewHandlerWithModel(mock, "default-model", "dev", "us-east-1")
	body := `{"messages":[{"role":"user","content":"Hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp ChatResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Model != "default-model" {
		t.Errorf("expected model 'default-model', got '%s'", resp.Model)
	}
}

func TestHandlerModels(t *testing.T) {
	mock := &mockConverser{
		models: []ModelInfo{
			{ID: "anthropic.claude-3", Object: "model", OwnedBy: "anthropic"},
			{ID: "amazon.titan-text", Object: "model", OwnedBy: "amazon"},
		},
	}
	h := NewHandlerWithModel(mock, "test-model", "dev", "us-east-1")
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp ModelsResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Object != "list" {
		t.Errorf("expected object=list, got %q", resp.Object)
	}
	if len(resp.Data) != 2 {
		t.Errorf("expected 2 models, got %d", len(resp.Data))
	}
}

func TestHandlerNotFound(t *testing.T) {
	h := NewHandlerWithModel(&mockConverser{}, "test-model", "dev", "us-east-1")
	req := httptest.NewRequest(http.MethodGet, "/nonexistent", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", w.Code)
	}
}

func TestHandlerMethodNotAllowed(t *testing.T) {
	h := NewHandlerWithModel(&mockConverser{}, "test-model", "dev", "us-east-1")
	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", w.Code)
	}
}

func TestHandlerCORSPreflight(t *testing.T) {
	h := NewHandlerWithModel(&mockConverser{}, "test-model", "dev", "us-east-1")
	req := httptest.NewRequest(http.MethodOptions, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	if w.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Error("expected CORS header")
	}
}

func TestHandlerStreaming(t *testing.T) {
	mock := &mockConverser{
		streamEvents: []StreamEvent{
			{Text: "Hello"},
			{Text: " world"},
			{FinishReason: "stop"},
			{Usage: &Usage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7}},
		},
	}

	h := NewHandlerWithModel(mock, "test-model", "dev", "us-east-1")
	body := `{"model":"test-model","messages":[{"role":"user","content":"Hi"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	respBody := w.Body.String()
	if !strings.Contains(respBody, "data: ") {
		t.Error("expected SSE data lines")
	}
	if !strings.Contains(respBody, "[DONE]") {
		t.Error("expected [DONE] terminator")
	}
	if !strings.Contains(respBody, "Hello") {
		t.Error("expected 'Hello' in stream")
	}

	// Verify SSE lines are valid JSON
	lines := strings.Split(respBody, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || line == "data: [DONE]" {
			continue
		}
		if strings.HasPrefix(line, "data: ") {
			jsonStr := strings.TrimPrefix(line, "data: ")
			var chunk StreamChunk
			if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
				t.Errorf("invalid JSON in SSE line: %s — error: %v", line, err)
			}
			if chunk.Object != "chat.completion.chunk" {
				t.Errorf("expected object=chat.completion.chunk, got %q", chunk.Object)
			}
		}
	}
}

func TestHandlerStreamingToolCalls(t *testing.T) {
	mock := &mockConverser{
		streamEvents: []StreamEvent{
			{ToolCallID: "call_1", ToolName: "get_weather", ToolArgs: `{"city":"NYC"}`},
			{FinishReason: "tool_calls"},
			{Usage: &Usage{PromptTokens: 10, CompletionTokens: 15, TotalTokens: 25}},
		},
	}

	h := NewHandlerWithModel(mock, "test-model", "dev", "us-east-1")
	body := `{"model":"test-model","messages":[{"role":"user","content":"Weather?"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	respBody := w.Body.String()
	if !strings.Contains(respBody, "get_weather") {
		t.Error("expected tool name in stream")
	}
	if !strings.Contains(respBody, "tool_calls") {
		t.Error("expected tool_calls finish_reason in stream")
	}
	if !strings.Contains(respBody, "[DONE]") {
		t.Error("expected [DONE]")
	}
}

func TestHandlerStreamingError(t *testing.T) {
	mock := &mockConverser{
		streamErr: fmt.Errorf("bedrock converse stream: throttling"),
	}

	h := NewHandlerWithModel(mock, "test-model", "dev", "us-east-1")
	body := `{"model":"test-model","messages":[{"role":"user","content":"Hi"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code == http.StatusOK {
		respBody := w.Body.String()
		if !strings.Contains(respBody, "throttling") {
			t.Error("expected throttling error in response")
		}
	}
}

func TestHandlerNonStreamingError(t *testing.T) {
	mock := &mockConverser{
		converseErr: fmt.Errorf("bedrock converse: ValidationException: invalid model"),
	}

	h := NewHandlerWithModel(mock, "test-model", "dev", "us-east-1")
	body := `{"model":"bad-model","messages":[{"role":"user","content":"Hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code == http.StatusOK {
		t.Fatal("expected error status, got 200")
	}

	var errResp ErrorResponse
	if err := json.NewDecoder(w.Body).Decode(&errResp); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if errResp.Error.Type != "invalid_request_error" {
		t.Errorf("expected type=invalid_request_error, got %q", errResp.Error.Type)
	}
}

func TestHandlerInvalidJSON(t *testing.T) {
	h := NewHandlerWithModel(&mockConverser{}, "test-model", "dev", "us-east-1")
	body := `{invalid json`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}
}

func TestHandlerEmptyBody(t *testing.T) {
	h := NewHandlerWithModel(&mockConverser{}, "test-model", "dev", "us-east-1")
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(""))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}
}

// --- Embedding Handler Tests ---

func TestEmbeddingHealthEndpoint(t *testing.T) {
	h := NewHandlerFull(&mockConverser{}, &mockEmbedder{}, "chat-model", defaultEmbedModel, "dev", "us-east-1")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var resp HealthResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Status != "ok" {
		t.Errorf("expected status 'ok', got '%s'", resp.Status)
	}
	if resp.EmbedModel != defaultEmbedModel {
		t.Errorf("expected default embed model, got '%s'", resp.EmbedModel)
	}
}

func TestSingleEmbedding(t *testing.T) {
	mock := &mockEmbedder{
		EmbedFunc: func(ctx context.Context, text string) ([]float64, error) {
			return make([]float64, 1024), nil
		},
	}
	h := NewHandlerFull(&mockConverser{}, mock, "chat-model", defaultEmbedModel, "dev", "us-east-1")

	body, _ := json.Marshal(EmbeddingRequest{Input: "test", Model: "amazon.titan-embed-text-v2:0"})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp EmbeddingResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Object != "list" {
		t.Errorf("expected object 'list', got '%s'", resp.Object)
	}
	if len(resp.Data) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(resp.Data))
	}
	if len(resp.Data[0].Embedding) != 1024 {
		t.Errorf("expected 1024 dims, got %d", len(resp.Data[0].Embedding))
	}
	if resp.Data[0].Index != 0 {
		t.Errorf("expected index 0, got %d", resp.Data[0].Index)
	}
}

func TestBatchEmbeddings(t *testing.T) {
	callCount := 0
	mock := &mockEmbedder{
		EmbedFunc: func(ctx context.Context, text string) ([]float64, error) {
			callCount++
			return make([]float64, 1024), nil
		},
	}
	h := NewHandlerFull(&mockConverser{}, mock, "chat-model", defaultEmbedModel, "dev", "us-east-1")

	body, _ := json.Marshal(EmbeddingRequestBatch{
		Input: []string{"first", "second", "third"},
		Model: "amazon.titan-embed-text-v2:0",
	})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var resp EmbeddingResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if len(resp.Data) != 3 {
		t.Fatalf("expected 3 embeddings, got %d", len(resp.Data))
	}
	if callCount != 3 {
		t.Errorf("expected 3 embed calls, got %d", callCount)
	}
	for i, d := range resp.Data {
		if d.Index != i {
			t.Errorf("expected index %d, got %d", i, d.Index)
		}
	}
}

func TestCohereV4SingleEmbedding(t *testing.T) {
	mock := &mockEmbedder{
		EmbedFunc: func(ctx context.Context, text string) ([]float64, error) {
			return make([]float64, 1536), nil
		},
	}
	h := NewHandlerFull(&mockConverser{}, mock, "chat-model", "cohere.embed-v4:0", "dev", "us-east-1")

	body, _ := json.Marshal(EmbeddingRequest{Input: "test cohere v4", Model: "cohere.embed-v4:0"})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var resp EmbeddingResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Model != "cohere.embed-v4:0" {
		t.Errorf("expected model 'cohere.embed-v4:0', got '%s'", resp.Model)
	}
	if len(resp.Data) != 1 || len(resp.Data[0].Embedding) != 1536 {
		t.Errorf("expected 1 embedding with 1536 dims")
	}
}

func TestEmbeddingDefaultModelFallback(t *testing.T) {
	mock := &mockEmbedder{
		EmbedFunc: func(ctx context.Context, text string) ([]float64, error) {
			return make([]float64, 1536), nil
		},
	}
	h := NewHandlerFull(&mockConverser{}, mock, "chat-model", "cohere.embed-v4:0", "dev", "us-east-1")

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader([]byte(`{"input":"test"}`)))
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(w, req)

	var resp EmbeddingResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Model != "cohere.embed-v4:0" {
		t.Errorf("expected default model 'cohere.embed-v4:0', got '%s'", resp.Model)
	}
}

func TestEmbeddingModelMismatchReturns400(t *testing.T) {
	mock := &mockEmbedder{
		EmbedFunc: func(ctx context.Context, text string) ([]float64, error) {
			return make([]float64, 256), nil
		},
	}
	h := NewHandlerFull(&mockConverser{}, mock, "chat-model", defaultEmbedModel, "dev", "us-east-1")

	body, _ := json.Marshal(EmbeddingRequest{Input: "test", Model: "cohere.embed-english-v3"})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for model mismatch, got %d", w.Code)
	}
}

func TestEmbeddingMethodNotAllowed(t *testing.T) {
	h := NewHandlerFull(&mockConverser{}, &mockEmbedder{}, "chat-model", defaultEmbedModel, "dev", "us-east-1")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodPut, "/v1/embeddings", nil))

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", w.Code)
	}
}

func TestEmbeddingEmptyBody(t *testing.T) {
	h := NewHandlerFull(&mockConverser{}, &mockEmbedder{}, "chat-model", defaultEmbedModel, "dev", "us-east-1")
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader([]byte{}))
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestEmbedderError(t *testing.T) {
	mock := &mockEmbedder{
		EmbedFunc: func(ctx context.Context, text string) ([]float64, error) {
			return nil, &EmbedError{Message: "model not available"}
		},
	}
	h := NewHandlerFull(&mockConverser{}, mock, "chat-model", defaultEmbedModel, "dev", "us-east-1")

	body, _ := json.Marshal(EmbeddingRequest{Input: "test", Model: "amazon.titan-embed-text-v2:0"})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("expected 500, got %d", w.Code)
	}
}

func TestEmbeddingsCORSHeaders(t *testing.T) {
	h := NewHandlerFull(&mockConverser{}, &mockEmbedder{}, "chat-model", defaultEmbedModel, "dev", "us-east-1")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodOptions, "/v1/embeddings", nil))

	if w.Code != http.StatusOK {
		t.Errorf("expected 200 for OPTIONS, got %d", w.Code)
	}
	if w.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Error("missing CORS header")
	}
}

func TestEmbeddingCancelledContextReturnsError(t *testing.T) {
	mock := &mockEmbedder{
		EmbedFunc: func(ctx context.Context, text string) ([]float64, error) {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			return make([]float64, 256), nil
		},
	}
	h := NewHandlerFull(&mockConverser{}, mock, "chat-model", defaultEmbedModel, "dev", "us-east-1")

	body, _ := json.Marshal(EmbeddingRequest{Input: "test", Model: "amazon.titan-embed-text-v2:0"})
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	ctx, cancel := context.WithCancel(req.Context())
	cancel()
	req = req.WithContext(ctx)

	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500 for cancelled context, got %d", w.Code)
	}
}

func TestSensitiveErrorsNotLeaked(t *testing.T) {
	sensitiveMsg := "AccessDeniedException: User arn:aws:iam::123456789:role/Foo is not authorized"
	mock := &mockEmbedder{
		EmbedFunc: func(ctx context.Context, text string) ([]float64, error) {
			return nil, fmt.Errorf("%s", sensitiveMsg)
		},
	}
	h := NewHandlerFull(&mockConverser{}, mock, "chat-model", defaultEmbedModel, "dev", "us-east-1")

	body, _ := json.Marshal(EmbeddingRequest{Input: "test", Model: "amazon.titan-embed-text-v2:0"})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d", w.Code)
	}

	respBody := w.Body.String()
	if strings.Contains(respBody, "arn:aws:iam") {
		t.Errorf("response body leaked sensitive error: %s", respBody)
	}
	if strings.Contains(respBody, "AccessDeniedException") {
		t.Errorf("response body leaked AWS error type: %s", respBody)
	}

	var errResp ErrorResponse
	json.NewDecoder(strings.NewReader(respBody)).Decode(&errResp)
	if errResp.Error.Message != "embedding failed" {
		t.Errorf("expected generic 'embedding failed', got '%s'", errResp.Error.Message)
	}
}

func TestTokenCountApproximation(t *testing.T) {
	mock := &mockEmbedder{
		EmbedFunc: func(ctx context.Context, text string) ([]float64, error) {
			return make([]float64, 256), nil
		},
	}
	h := NewHandlerFull(&mockConverser{}, mock, "chat-model", defaultEmbedModel, "dev", "us-east-1")

	input := strings.Repeat("a", 100)
	body, _ := json.Marshal(EmbeddingRequest{Input: input, Model: "amazon.titan-embed-text-v2:0"})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp EmbeddingResponse
	json.NewDecoder(w.Body).Decode(&resp)

	expectedApprox := len(input) / 4 // 25
	if resp.Usage.PromptTokens == len(input) {
		t.Errorf("prompt_tokens should not equal raw byte count (%d)", len(input))
	}
	if resp.Usage.PromptTokens != expectedApprox {
		t.Errorf("expected ~%d prompt_tokens, got %d", expectedApprox, resp.Usage.PromptTokens)
	}
	if resp.Usage.TotalTokens != resp.Usage.PromptTokens {
		t.Errorf("total_tokens (%d) should equal prompt_tokens (%d)", resp.Usage.TotalTokens, resp.Usage.PromptTokens)
	}
}

// --- Nil embedder tests ---

func TestEmbeddingsWithNilEmbedderReturns404(t *testing.T) {
	h := NewHandlerWithModel(&mockConverser{}, "chat-model", "dev", "us-east-1")

	body, _ := json.Marshal(EmbeddingRequest{Input: "test"})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Fatalf("expected 404 when embedder is nil, got %d", w.Code)
	}
}

// --- parseEmbedInput tests ---

func TestParseEmbedInputSingle(t *testing.T) {
	body := []byte(`{"input":"hello world","model":"test-model"}`)
	inputs, model, err := parseEmbedInput(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(inputs) != 1 || inputs[0] != "hello world" {
		t.Errorf("expected ['hello world'], got %v", inputs)
	}
	if model != "test-model" {
		t.Errorf("expected model 'test-model', got '%s'", model)
	}
}

func TestParseEmbedInputBatch(t *testing.T) {
	body := []byte(`{"input":["a","b","c"],"model":"test-model"}`)
	inputs, model, err := parseEmbedInput(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(inputs) != 3 {
		t.Errorf("expected 3 inputs, got %d", len(inputs))
	}
	if model != "test-model" {
		t.Errorf("expected model 'test-model', got '%s'", model)
	}
}

func TestParseEmbedInputNoModel(t *testing.T) {
	body := []byte(`{"input":"hello"}`)
	inputs, model, err := parseEmbedInput(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(inputs) != 1 {
		t.Errorf("expected 1 input, got %d", len(inputs))
	}
	if model != "" {
		t.Errorf("expected empty model, got '%s'", model)
	}
}

func TestParseEmbedInputInvalid(t *testing.T) {
	body := []byte(`{"input":123}`)
	_, _, err := parseEmbedInput(body)
	if err == nil {
		t.Error("expected error for numeric input")
	}
}

// --- Chat error sanitization tests ---

func TestChatBedrockErrorSanitized(t *testing.T) {
	tests := []struct {
		name       string
		err        string
		wantStatus int
		wantType   string
		wantNoLeak string
	}{
		{
			name:       "validation error",
			err:        "bedrock converse: operation error Bedrock Runtime: Converse, ValidationException: The provided model identifier is invalid.",
			wantStatus: http.StatusBadRequest,
			wantType:   "invalid_request_error",
			wantNoLeak: "ValidationException",
		},
		{
			name:       "access denied",
			err:        "AccessDeniedException: User arn:aws:iam::123456789:role/Foo is not authorized to perform bedrock:InvokeModel",
			wantStatus: http.StatusUnauthorized,
			wantType:   "authentication_error",
			wantNoLeak: "arn:aws:iam",
		},
		{
			name:       "throttling",
			err:        "ThrottlingException: Too many requests, please slow down",
			wantStatus: http.StatusTooManyRequests,
			wantType:   "rate_limit_error",
			wantNoLeak: "ThrottlingException",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mock := &mockConverser{
				converseErr: fmt.Errorf("%s", tt.err),
			}
			h := NewHandlerWithModel(mock, "test-model", "dev", "us-east-1")
			body := `{"model":"test-model","messages":[{"role":"user","content":"Hi"}]}`
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
			w := httptest.NewRecorder()
			h.ServeHTTP(w, req)

			if w.Code != tt.wantStatus {
				t.Errorf("expected %d, got %d", tt.wantStatus, w.Code)
			}

			respBody := w.Body.String()
			if strings.Contains(respBody, tt.wantNoLeak) {
				t.Errorf("response leaked %q: %s", tt.wantNoLeak, respBody)
			}

			var errResp ErrorResponse
			json.NewDecoder(strings.NewReader(respBody)).Decode(&errResp)
			if errResp.Error.Type != tt.wantType {
				t.Errorf("expected type=%q, got %q", tt.wantType, errResp.Error.Type)
			}
		})
	}
}

// --- Embedding alias resolution via handler ---

func TestEmbeddingAliasResolution(t *testing.T) {
	mock := &mockEmbedder{
		EmbedFunc: func(ctx context.Context, text string) ([]float64, error) {
			return make([]float64, 256), nil
		},
	}
	h := NewHandlerFull(&mockConverser{}, mock, "chat-model", "amazon.titan-embed-text-v2:0", "dev", "us-east-1")

	// Send OpenAI-compatible model name — should resolve via alias
	body, _ := json.Marshal(EmbeddingRequest{Input: "test", Model: "titan-embed-v2"})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp EmbeddingResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Model != "amazon.titan-embed-text-v2:0" {
		t.Errorf("expected resolved model, got '%s'", resp.Model)
	}
}

// --- Both endpoints on same handler ---

func TestUnifiedHandlerBothEndpoints(t *testing.T) {
	chatMock := &mockConverser{
		converseResp: &ChatResponse{
			ID: "chat-1", Object: "chat.completion", Model: "test-chat",
			Choices: []Choice{{Message: Message{Role: "assistant", Content: "hi"}, FinishReason: "stop"}},
		},
	}
	embedMock := &mockEmbedder{
		EmbedFunc: func(ctx context.Context, text string) ([]float64, error) {
			return make([]float64, 512), nil
		},
	}
	h := NewHandlerFull(chatMock, embedMock, "test-chat", "test-embed", "dev", "us-east-1")

	// Chat
	chatReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"messages":[{"role":"user","content":"Hi"}]}`))
	chatW := httptest.NewRecorder()
	h.ServeHTTP(chatW, chatReq)
	if chatW.Code != http.StatusOK {
		t.Fatalf("chat: expected 200, got %d", chatW.Code)
	}

	// Embed
	embedBody, _ := json.Marshal(EmbeddingRequest{Input: "test", Model: "test-embed"})
	embedReq := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(embedBody))
	embedW := httptest.NewRecorder()
	h.ServeHTTP(embedW, embedReq)
	if embedW.Code != http.StatusOK {
		t.Fatalf("embed: expected 200, got %d: %s", embedW.Code, embedW.Body.String())
	}

	// Health shows both
	healthReq := httptest.NewRequest(http.MethodGet, "/", nil)
	healthW := httptest.NewRecorder()
	h.ServeHTTP(healthW, healthReq)
	var health HealthResponse
	json.NewDecoder(healthW.Body).Decode(&health)
	if health.Model != "test-chat" || health.EmbedModel != "test-embed" {
		t.Errorf("health: expected both models, got chat=%q embed=%q", health.Model, health.EmbedModel)
	}
}

// --- Model alias via chat handler ---

func TestChatModelAliasResolution(t *testing.T) {
	var capturedModel string
	mock := &mockConverser{
		converseResp: &ChatResponse{
			ID: "x", Object: "chat.completion",
			Choices: []Choice{{Message: Message{Role: "assistant", Content: "hi"}, FinishReason: "stop"}},
		},
	}
	// Wrap to capture the model after alias resolution
	h := NewHandlerWithModel(mock, "us.anthropic.claude-opus-4-6-v1", "dev", "us-east-1")

	// Send OpenRouter-style alias
	body := `{"model":"anthropic/claude-opus-4.6","messages":[{"role":"user","content":"Hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	_ = capturedModel
}
