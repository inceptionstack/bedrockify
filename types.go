package bedrockify

import (
	"context"
	"fmt"
)

// --- Core interfaces ---

// Converser handles chat completion requests against a backend model API.
type Converser interface {
	Converse(ctx context.Context, req *ChatRequest) (*ChatResponse, error)
	ConverseStream(ctx context.Context, req *ChatRequest) (<-chan StreamEvent, error)
	ListModels(ctx context.Context) ([]ModelInfo, error)
}

// Embedder generates embedding vectors from text.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float64, error)
}

// --- OpenAI-compatible chat request types ---

// ChatRequest is the OpenAI /v1/chat/completions request body.
type ChatRequest struct {
	Model               string                 `json:"model"`
	Messages            []Message              `json:"messages"`
	MaxTokens           int                    `json:"max_tokens,omitempty"`
	MaxCompletionTokens int                    `json:"max_completion_tokens,omitempty"`
	Temperature         *float64               `json:"temperature,omitempty"`
	TopP                *float64               `json:"top_p,omitempty"`
	Stream              bool                   `json:"stream,omitempty"`
	StreamOptions       *StreamOptions         `json:"stream_options,omitempty"`
	Stop                []string               `json:"stop,omitempty"`
	Tools               []Tool                 `json:"tools,omitempty"`
	ToolChoice          interface{}            `json:"tool_choice,omitempty"`
	N                   int                    `json:"n,omitempty"`
	ReasoningEffort     string                 `json:"reasoning_effort,omitempty"`
	ExtraBody           map[string]interface{} `json:"extra_body,omitempty"`
}

// StreamOptions controls streaming behavior.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// Message is a single chat message.
type Message struct {
	Role             string      `json:"role"`
	Content          interface{} `json:"content"` // string or []ContentPart
	ToolCalls        []ToolCall  `json:"tool_calls,omitempty"`
	ToolCallID       string      `json:"tool_call_id,omitempty"`
	Name             string      `json:"name,omitempty"`
	ReasoningContent string      `json:"reasoning_content,omitempty"`
}

// ContentPart is a typed content element (text, image_url, tool_result, etc.)
type ContentPart struct {
	Type       string      `json:"type"`
	Text       string      `json:"text,omitempty"`
	ImageURL   *ImageURL   `json:"image_url,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
	Content    interface{} `json:"content,omitempty"`
}

// ImageURL holds image data for vision messages.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// Tool defines a function the model may call.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction is the function definition within a tool.
type ToolFunction struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	Parameters  interface{} `json:"parameters,omitempty"`
}

// ToolCall is a function invocation requested by the model.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction holds the function name and stringified arguments.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// --- OpenAI-compatible chat response types ---

// ChatResponse is the non-streaming /v1/chat/completions response.
type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Choice is a single completion option.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage reports token consumption (shared by chat and embeddings).
type Usage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens,omitempty"`
	TotalTokens             int                      `json:"total_tokens"`
	PromptTokensDetails     *PromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// PromptTokensDetails holds granular prompt token breakdown.
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// CompletionTokensDetails holds granular completion token breakdown.
type CompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// --- Streaming types ---

// StreamChunk is the SSE delta payload for streaming responses.
type StreamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
	Usage   *Usage         `json:"usage,omitempty"`
}

// StreamChoice is a single delta choice in a streaming chunk.
type StreamChoice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

// Delta carries the incremental content for a streaming chunk.
type Delta struct {
	Role             string     `json:"role,omitempty"`
	Content          string     `json:"content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
}

// StreamEvent is an internal event emitted during streaming.
type StreamEvent struct {
	// Text delta content
	Text string
	// Reasoning/thinking content delta
	ReasoningContent string
	// Reasoning signature (feature 5.3)
	ReasoningSignature string
	// Tool call being built up
	ToolCallID string
	ToolName   string
	ToolArgs   string
	// Finish signal
	FinishReason string
	// Usage stats (sent at end)
	Usage *Usage
	// Error
	Err error
}

// --- Models list ---

// ModelInfo is a single model in the /v1/models list.
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created,omitempty"`
	OwnedBy string `json:"owned_by"`
}

// ModelsResponse is the /v1/models response.
type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

// --- Embedding types ---

// EmbeddingRequest is a single-string input request.
type EmbeddingRequest struct {
	Input          string `json:"input"`
	Model          string `json:"model"`
	EncodingFormat string `json:"encoding_format,omitempty"`
	Dimensions     int    `json:"dimensions,omitempty"`
}

// EmbeddingRequestBatch is an array input request.
type EmbeddingRequestBatch struct {
	Input          []string `json:"input"`
	Model          string   `json:"model"`
	EncodingFormat string   `json:"encoding_format,omitempty"`
	Dimensions     int      `json:"dimensions,omitempty"`
}

// EmbeddingResponse is the top-level response envelope.
type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  Usage           `json:"usage"`
}

// EmbeddingData is a single embedding result.
// Embedding field is interface{} to support both []float64 and base64 string.
type EmbeddingData struct {
	Object    string      `json:"object"`
	Index     int         `json:"index"`
	Embedding interface{} `json:"embedding"`
}

// --- API response types ---

// HealthResponse is returned by GET /.
type HealthResponse struct {
	Status     string `json:"status"`
	Model      string `json:"model,omitempty"`
	EmbedModel string `json:"embed_model,omitempty"`
	Version    string `json:"version,omitempty"`
}

// ErrorResponse wraps errors in OpenAI-compatible format.
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail is the error payload.
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

// --- Internal helpers ---

// ProxyError represents a proxy-level error.
type ProxyError struct {
	Message string
	Code    int
}

func (e *ProxyError) Error() string {
	return fmt.Sprintf("proxy error %d: %s", e.Code, e.Message)
}

// EmbedError represents an embedding failure.
type EmbedError struct {
	Message string
}

func (e *EmbedError) Error() string {
	return fmt.Sprintf("embed error: %s", e.Message)
}

// MessageContent extracts the string content from a Message.Content (which may be
// a string or []ContentPart). Returns the concatenated text.
func MessageContent(m Message) string {
	switch v := m.Content.(type) {
	case string:
		return v
	case []interface{}:
		var sb string
		for _, part := range v {
			if p, ok := part.(map[string]interface{}); ok {
				if t, ok := p["text"].(string); ok {
					sb += t
				}
			}
		}
		return sb
	}
	return ""
}
