package bedrockify

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// maxRequestBodySize limits request body reads to 10MB to prevent OOM.
const maxRequestBodySize = 10 * 1024 * 1024

const defaultChatModel = "us.anthropic.claude-sonnet-4-6"
const defaultEmbedModel = "amazon.titan-embed-text-v2:0"

// Handler serves the OpenAI-compatible chat completions and embeddings API.
type Handler struct {
	converser      Converser
	embedder       Embedder
	defaultModel   string
	embeddingModel string
	region         string
	version        string
}

// NewHandler creates a handler with the default model name.
func NewHandler(converser Converser) *Handler {
	return &Handler{
		converser:    converser,
		defaultModel: defaultChatModel,
		region:       "us-east-1",
	}
}

// NewHandlerWithModel creates a handler with specific default model names and optional embedder.
func NewHandlerWithModel(converser Converser, model, version, region string) *Handler {
	if region == "" {
		region = "us-east-1"
	}
	return &Handler{
		converser:    converser,
		defaultModel: model,
		version:      version,
		region:       region,
	}
}

// NewHandlerFull creates a unified handler with both converser and embedder.
func NewHandlerFull(converser Converser, embedder Embedder, chatModel, embedModel, version, region string) *Handler {
	if region == "" {
		region = "us-east-1"
	}
	if embedModel == "" {
		embedModel = defaultEmbedModel
	}
	return &Handler{
		converser:      converser,
		embedder:       embedder,
		defaultModel:   chatModel,
		embeddingModel: embedModel,
		version:        version,
		region:         region,
	}
}

// ServeHTTP dispatches requests to the appropriate handler.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	path := strings.TrimSuffix(r.URL.Path, "/")

	switch {
	case r.Method == http.MethodGet && (path == "" || path == "/"):
		h.handleHealth(w)
	case path == "/v1/chat/completions" && r.Method == http.MethodPost:
		h.handleChatCompletions(w, r)
	case path == "/v1/chat/completions":
		h.writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
	case path == "/v1/embeddings" && r.Method == http.MethodPost:
		h.handleEmbeddings(w, r)
	case path == "/v1/embeddings" && r.Method == http.MethodOptions:
		w.WriteHeader(http.StatusOK)
	case path == "/v1/embeddings":
		h.writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
	case path == "/v1/models" && r.Method == http.MethodGet:
		h.handleModels(w, r)
	case path == "/v1/models":
		h.writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
	default:
		h.writeError(w, http.StatusNotFound, "not found", "invalid_request_error")
	}
}

// handleHealth returns a simple health check JSON.
func (h *Handler) handleHealth(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	resp := HealthResponse{
		Status:  "ok",
		Model:   h.defaultModel,
		Version: h.version,
	}
	if h.embedder != nil {
		resp.EmbedModel = h.embeddingModel
	}
	json.NewEncoder(w).Encode(resp)
}

// readBody reads and returns the request body with a size limit.
func readBody(r *http.Request) ([]byte, error) {
	r.Body = http.MaxBytesReader(nil, r.Body, maxRequestBodySize)
	return io.ReadAll(r.Body)
}

// handleChatCompletions handles POST /v1/chat/completions.
func (h *Handler) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	body, err := readBody(r)
	if err != nil || len(body) == 0 {
		h.writeError(w, http.StatusBadRequest, "empty or invalid request body", "invalid_request_error")
		return
	}

	var req ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		h.writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(), "invalid_request_error")
		return
	}

	if len(req.Messages) == 0 {
		h.writeError(w, http.StatusBadRequest, "messages must not be empty", "invalid_request_error")
		return
	}

	// Apply default model if not specified
	if req.Model == "" {
		req.Model = h.defaultModel
	}

	// Resolve model aliases (OpenRouter IDs, short names, etc.) to Bedrock IDs
	req.Model, _ = ResolveModelAlias(req.Model, h.region)

	start := time.Now()

	if req.Stream {
		h.handleStreaming(w, r, &req, start)
	} else {
		h.handleNonStreaming(w, r, &req, start)
	}
}

// handleNonStreaming handles a regular (non-SSE) chat completion.
func (h *Handler) handleNonStreaming(w http.ResponseWriter, r *http.Request, req *ChatRequest, start time.Time) {
	resp, err := h.converser.Converse(r.Context(), req)
	if err != nil {
		log.Printf("converse error model=%s latency=%s: %v", req.Model, time.Since(start).Round(time.Millisecond), err)
		h.writeBedrockError(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleStreaming handles a streaming SSE chat completion.
func (h *Handler) handleStreaming(w http.ResponseWriter, r *http.Request, req *ChatRequest, start time.Time) {
	ch, err := h.converser.ConverseStream(r.Context(), req)
	if err != nil {
		log.Printf("converse-stream error model=%s latency=%s: %v", req.Model, time.Since(start).Round(time.Millisecond), err)
		h.writeBedrockError(w, err)
		return
	}

	reqID := newRequestID()
	includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage
	StreamWithOptions(w, ch, req.Model, reqID, includeUsage)
}

// handleEmbeddings handles POST /v1/embeddings.
func (h *Handler) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if h.embedder == nil {
		h.writeError(w, http.StatusNotFound, "embeddings endpoint not configured; start with --embed-model to enable", "not_found_error")
		return
	}

	body, err := readBody(r)
	if err != nil || len(body) == 0 {
		h.writeError(w, http.StatusBadRequest, "empty or invalid request body", "invalid_request_error")
		return
	}

	inputs, model, err := parseEmbedInput(body)
	if err != nil {
		h.writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}

	// Resolve embedding model alias
	if model != "" {
		model, _ = ResolveEmbeddingAlias(model)
	}

	if model == "" {
		model = h.embeddingModel
	} else if model != h.embeddingModel {
		h.writeError(w, http.StatusBadRequest,
			"model '"+model+"' is not available; this server is configured with '"+h.embeddingModel+"'",
			"invalid_request_error")
		return
	}

	data := make([]EmbeddingData, 0, len(inputs))
	promptTokens := 0
	for i, text := range inputs {
		embedding, err := h.embedder.Embed(r.Context(), text)
		if err != nil {
			log.Printf("embedding error: %v", err)
			h.writeError(w, http.StatusInternalServerError, "embedding failed", "server_error")
			return
		}
		data = append(data, EmbeddingData{
			Object:    "embedding",
			Index:     i,
			Embedding: embedding,
		})
		promptTokens += len(text) / 4 // ~4 chars per BPE token
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(EmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  model,
		Usage:  Usage{PromptTokens: promptTokens, TotalTokens: promptTokens},
	})
}

// handleModels handles GET /v1/models.
func (h *Handler) handleModels(w http.ResponseWriter, r *http.Request) {
	models, err := h.converser.ListModels(r.Context())
	if err != nil {
		log.Printf("list models error: %v", err)
		h.writeError(w, http.StatusInternalServerError, "failed to list models: "+err.Error(), "server_error")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ModelsResponse{
		Object: "list",
		Data:   models,
	})
}

// writeError writes a standard OpenAI error response.
func (h *Handler) writeError(w http.ResponseWriter, status int, message, errType string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(ErrorResponse{
		Error: ErrorDetail{Message: message, Type: errType},
	})
}

// writeBedrockError translates a Bedrock error into an OpenAI error response.
// Logs the full error server-side but returns a sanitized message to the client.
func (h *Handler) writeBedrockError(w http.ResponseWriter, err error) {
	msg := err.Error()
	status := http.StatusInternalServerError
	errType := "server_error"
	clientMsg := "internal server error"

	lower := strings.ToLower(msg)
	switch {
	case strings.Contains(lower, "throttlingexception") || strings.Contains(lower, "too many requests"):
		status = http.StatusTooManyRequests
		errType = "rate_limit_error"
		clientMsg = "rate limit exceeded — try again later"
	case strings.Contains(lower, "validationexception") || strings.Contains(lower, "invalid"):
		status = http.StatusBadRequest
		errType = "invalid_request_error"
		clientMsg = "invalid request — check model ID and parameters"
	case strings.Contains(lower, "accessdenied") || strings.Contains(lower, "unauthorized"):
		status = http.StatusUnauthorized
		errType = "authentication_error"
		clientMsg = "authentication failed"
	case strings.Contains(lower, "resourcenotfound") || strings.Contains(lower, "not found"):
		status = http.StatusNotFound
		errType = "not_found_error"
		clientMsg = "model not found"
	}

	h.writeError(w, status, clientMsg, errType)
}

// parseEmbedInput extracts text inputs and model from an OpenAI-format request body.
func parseEmbedInput(body []byte) ([]string, string, error) {
	// Try batch (array input)
	var batch EmbeddingRequestBatch
	if err := json.Unmarshal(body, &batch); err == nil && len(batch.Input) > 0 {
		return batch.Input, batch.Model, nil
	}

	// Try single string input
	var single EmbeddingRequest
	if err := json.Unmarshal(body, &single); err == nil && single.Input != "" {
		return []string{single.Input}, single.Model, nil
	}

	// Fallback: raw JSON with generic input field
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, "", err
	}

	model, _ := raw["model"].(string)

	switch v := raw["input"].(type) {
	case string:
		if v == "" {
			return nil, "", &EmbedError{Message: "input is empty"}
		}
		return []string{v}, model, nil
	case []interface{}:
		inputs := make([]string, 0, len(v))
		for _, item := range v {
			s, ok := item.(string)
			if !ok {
				return nil, "", &EmbedError{Message: "input array must contain strings"}
			}
			inputs = append(inputs, s)
		}
		if len(inputs) == 0 {
			return nil, "", &EmbedError{Message: "input array is empty"}
		}
		return inputs, model, nil
	default:
		return nil, "", &EmbedError{Message: "input must be a string or array of strings"}
	}
}
