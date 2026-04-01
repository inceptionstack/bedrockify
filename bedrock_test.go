package bedrockify

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

func TestMapStopReason(t *testing.T) {
	cases := []struct {
		input    string
		expected string
	}{
		{"end_turn", "stop"},
		{"tool_use", "tool_calls"},
		{"max_tokens", "length"},
		{"stop_sequence", "stop"},
		{"guardrail_intervened", "content_filter"},
		{"", ""},
		{"unknown_reason", "stop"},
	}
	for _, c := range cases {
		got := mapStopReason(c.input)
		if got != c.expected {
			t.Errorf("mapStopReason(%q) = %q, want %q", c.input, got, c.expected)
		}
	}
}

func TestMessageContent(t *testing.T) {
	cases := []struct {
		name     string
		msg      Message
		expected string
	}{
		{
			name:     "string content",
			msg:      Message{Role: "user", Content: "hello"},
			expected: "hello",
		},
		{
			name: "parts content",
			msg: Message{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{"type": "text", "text": "hello "},
					map[string]interface{}{"type": "text", "text": "world"},
				},
			},
			expected: "hello world",
		},
		{
			name:     "nil content",
			msg:      Message{Role: "user", Content: nil},
			expected: "",
		},
		{
			name:     "empty string",
			msg:      Message{Role: "user", Content: ""},
			expected: "",
		},
		{
			name: "mixed parts with non-text",
			msg: Message{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{"type": "text", "text": "hello"},
					map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": "data:image/png;base64,abc"}},
				},
			},
			expected: "hello",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := MessageContent(c.msg)
			if got != c.expected {
				t.Errorf("got %q, want %q", got, c.expected)
			}
		})
	}
}

func TestConvertMessagesSystemRole(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
	}
	bedrockMsgs, system, err := convertMessages(msgs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(system) != 1 {
		t.Fatalf("expected 1 system block, got %d", len(system))
	}
	if len(bedrockMsgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(bedrockMsgs))
	}
}

func TestConvertMessagesMultipleSystemBlocks(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "Rule 1"},
		{Role: "system", Content: "Rule 2"},
		{Role: "user", Content: "Hello"},
	}
	bedrockMsgs, system, err := convertMessages(msgs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(system) != 2 {
		t.Fatalf("expected 2 system blocks, got %d", len(system))
	}
	if len(bedrockMsgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(bedrockMsgs))
	}
}

func TestConvertMessagesToolResult(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: "Call a tool"},
		{Role: "assistant", Content: nil, ToolCalls: []ToolCall{
			{
				ID:   "call_123",
				Type: "function",
				Function: ToolCallFunction{Name: "my_func", Arguments: `{"arg":"val"}`},
			},
		}},
		{Role: "tool", Content: "tool result", ToolCallID: "call_123"},
	}

	bedrockMsgs, _, err := convertMessages(msgs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(bedrockMsgs) != 3 {
		t.Fatalf("expected 3 bedrock messages, got %d", len(bedrockMsgs))
	}
}

func TestConvertMessagesMultipleToolResults(t *testing.T) {
	// When assistant calls 2 tools, both tool results should merge into one user message
	msgs := []Message{
		{Role: "user", Content: "Do two things"},
		{Role: "assistant", Content: nil, ToolCalls: []ToolCall{
			{ID: "call_1", Type: "function", Function: ToolCallFunction{Name: "func_a", Arguments: "{}"}},
			{ID: "call_2", Type: "function", Function: ToolCallFunction{Name: "func_b", Arguments: "{}"}},
		}},
		{Role: "tool", Content: "result a", ToolCallID: "call_1"},
		{Role: "tool", Content: "result b", ToolCallID: "call_2"},
	}

	bedrockMsgs, _, err := convertMessages(msgs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// user, assistant, user (merged tool results)
	if len(bedrockMsgs) != 3 {
		t.Fatalf("expected 3 bedrock messages (tool results merged), got %d", len(bedrockMsgs))
	}
	// The third message should have 2 content blocks (both tool results)
	if len(bedrockMsgs[2].Content) != 2 {
		t.Errorf("expected 2 content blocks in merged tool result, got %d", len(bedrockMsgs[2].Content))
	}
}

func TestConvertMessagesUserOnly(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: "Just a question"},
	}
	bedrockMsgs, system, err := convertMessages(msgs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(system) != 0 {
		t.Errorf("expected no system blocks, got %d", len(system))
	}
	if len(bedrockMsgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(bedrockMsgs))
	}
}

func TestConvertMessagesMultiTurn(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "Be helpful"},
		{Role: "user", Content: "Hi"},
		{Role: "assistant", Content: "Hello!"},
		{Role: "user", Content: "How are you?"},
		{Role: "assistant", Content: "I'm good!"},
		{Role: "user", Content: "Great"},
	}
	bedrockMsgs, system, err := convertMessages(msgs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(system) != 1 {
		t.Errorf("expected 1 system block, got %d", len(system))
	}
	if len(bedrockMsgs) != 5 {
		t.Errorf("expected 5 messages (3 user + 2 assistant), got %d", len(bedrockMsgs))
	}
}

func TestConvertToolsEmpty(t *testing.T) {
	toolConfig, err := convertTools(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if toolConfig == nil {
		t.Fatal("expected non-nil tool config")
	}
	if len(toolConfig.Tools) != 0 {
		t.Errorf("expected 0 tools, got %d", len(toolConfig.Tools))
	}
}

func TestConvertTools(t *testing.T) {
	tools := []Tool{
		{
			Type: "function",
			Function: ToolFunction{
				Name:        "get_weather",
				Description: "Get weather",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "City name",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	toolConfig, err := convertTools(tools)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(toolConfig.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(toolConfig.Tools))
	}
}

func TestConvertToolsMultiple(t *testing.T) {
	tools := []Tool{
		{Type: "function", Function: ToolFunction{Name: "func_a", Description: "A"}},
		{Type: "function", Function: ToolFunction{Name: "func_b", Description: "B"}},
		{Type: "not_function", Function: ToolFunction{Name: "skipped"}}, // should be skipped
	}
	toolConfig, err := convertTools(tools)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(toolConfig.Tools) != 2 {
		t.Errorf("expected 2 tools (non-function skipped), got %d", len(toolConfig.Tools))
	}
}

func TestNewRequestID(t *testing.T) {
	id1 := newRequestID()
	id2 := newRequestID()
	if id1 == "" || id2 == "" {
		t.Error("expected non-empty IDs")
	}
	if !strings.HasPrefix(id1, "chatcmpl-") {
		t.Errorf("expected chatcmpl- prefix, got %s", id1)
	}
}

func TestChatRequestJSON(t *testing.T) {
	raw := `{
		"model": "anthropic.claude-3",
		"messages": [
			{"role": "system", "content": "You are helpful."},
			{"role": "user", "content": "Hello"}
		],
		"max_tokens": 1024,
		"stream": false,
		"temperature": 0.7
	}`

	var req ChatRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if req.Model != "anthropic.claude-3" {
		t.Errorf("unexpected model: %s", req.Model)
	}
	if len(req.Messages) != 2 {
		t.Errorf("expected 2 messages, got %d", len(req.Messages))
	}
	if req.MaxTokens != 1024 {
		t.Errorf("expected max_tokens=1024, got %d", req.MaxTokens)
	}
	if req.Temperature == nil || *req.Temperature != 0.7 {
		t.Errorf("unexpected temperature: %v", req.Temperature)
	}
}

func TestChatRequestJSONWithTools(t *testing.T) {
	raw := `{
		"model": "anthropic.claude-3",
		"messages": [{"role": "user", "content": "Hello"}],
		"tools": [{
			"type": "function",
			"function": {
				"name": "search",
				"description": "Search the web",
				"parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
			}
		}],
		"tool_choice": "auto"
	}`
	var req ChatRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(req.Tools) != 1 {
		t.Errorf("expected 1 tool, got %d", len(req.Tools))
	}
	if req.Tools[0].Function.Name != "search" {
		t.Errorf("unexpected tool name: %s", req.Tools[0].Function.Name)
	}
}

func TestChatRequestJSONWithStreamAndStop(t *testing.T) {
	raw := `{
		"model": "test",
		"messages": [{"role": "user", "content": "Hi"}],
		"stream": true,
		"stop": ["END", "STOP"],
		"top_p": 0.9
	}`
	var req ChatRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !req.Stream {
		t.Error("expected stream=true")
	}
	if len(req.Stop) != 2 {
		t.Errorf("expected 2 stop sequences, got %d", len(req.Stop))
	}
	if req.TopP == nil || *req.TopP != 0.9 {
		t.Errorf("unexpected top_p: %v", req.TopP)
	}
}

func TestChatResponseJSON(t *testing.T) {
	resp := ChatResponse{
		ID:      "chatcmpl-test",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "anthropic.claude-3",
		Choices: []Choice{
			{Index: 0, Message: Message{Role: "assistant", Content: "Hi"}, FinishReason: "stop"},
		},
		Usage: Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}

	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var decoded ChatResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if decoded.ID != "chatcmpl-test" {
		t.Errorf("unexpected ID: %s", decoded.ID)
	}
	if decoded.Usage.TotalTokens != 15 {
		t.Errorf("unexpected total_tokens: %d", decoded.Usage.TotalTokens)
	}
	if len(decoded.Choices) != 1 || decoded.Choices[0].FinishReason != "stop" {
		t.Errorf("unexpected choices: %+v", decoded.Choices)
	}
}

func TestStreamChunkJSON(t *testing.T) {
	reason := "stop"
	chunk := StreamChunk{
		ID:      "chatcmpl-stream",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "test-model",
		Choices: []StreamChoice{
			{Index: 0, Delta: Delta{Content: "hello"}, FinishReason: nil},
		},
	}

	data, err := json.Marshal(chunk)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(data), `"content":"hello"`) {
		t.Error("expected content in JSON")
	}
	if strings.Contains(string(data), `"finish_reason":"stop"`) {
		t.Error("finish_reason should be null, not stop")
	}

	// With finish reason
	chunk.Choices[0].FinishReason = &reason
	data, err = json.Marshal(chunk)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(data), `"finish_reason":"stop"`) {
		t.Error("expected finish_reason=stop in JSON")
	}
}

func TestErrorResponseJSON(t *testing.T) {
	resp := ErrorResponse{
		Error: ErrorDetail{
			Message: "model not found",
			Type:    "not_found_error",
			Code:    "model_not_found",
		},
	}
	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(data), "model not found") {
		t.Error("expected error message in JSON")
	}
	if !strings.Contains(string(data), "not_found_error") {
		t.Error("expected error type in JSON")
	}
}

func TestProxyError(t *testing.T) {
	err := &ProxyError{Message: "bad request", Code: 400}
	if err.Error() != "proxy error 400: bad request" {
		t.Errorf("unexpected error string: %s", err.Error())
	}
}

func TestBuildConverseInputDefaults(t *testing.T) {
	req := &ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
	}
	input, err := buildConverseInput("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if *input.ModelId != "test-model" {
		t.Errorf("expected model=test-model, got %s", *input.ModelId)
	}
	if input.InferenceConfig != nil {
		t.Error("expected nil InferenceConfig when no params set")
	}
	if input.ToolConfig != nil {
		t.Error("expected nil ToolConfig when no tools")
	}
}

func TestBuildConverseInputWithConfig(t *testing.T) {
	temp := 0.5
	topP := 0.9
	req := &ChatRequest{
		Messages:    []Message{{Role: "user", Content: "Hi"}},
		MaxTokens:   100,
		Temperature: &temp,
		TopP:        &topP,
		Stop:        []string{"END"},
	}
	input, err := buildConverseInput("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if input.InferenceConfig == nil {
		t.Fatal("expected InferenceConfig to be set")
	}
	if *input.InferenceConfig.MaxTokens != 100 {
		t.Errorf("expected max_tokens=100, got %d", *input.InferenceConfig.MaxTokens)
	}
	if *input.InferenceConfig.Temperature != 0.5 {
		t.Errorf("expected temperature=0.5, got %f", *input.InferenceConfig.Temperature)
	}
	if *input.InferenceConfig.TopP != 0.9 {
		t.Errorf("expected top_p=0.9, got %f", *input.InferenceConfig.TopP)
	}
	if len(input.InferenceConfig.StopSequences) != 1 {
		t.Errorf("expected 1 stop sequence, got %d", len(input.InferenceConfig.StopSequences))
	}
}

func TestBuildConverseInputWithTools(t *testing.T) {
	req := &ChatRequest{
		Messages: []Message{{Role: "user", Content: "Hi"}},
		Tools: []Tool{
			{Type: "function", Function: ToolFunction{Name: "test", Description: "A test tool"}},
		},
	}
	input, err := buildConverseInput("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if input.ToolConfig == nil {
		t.Fatal("expected ToolConfig to be set")
	}
	if len(input.ToolConfig.Tools) != 1 {
		t.Errorf("expected 1 tool, got %d", len(input.ToolConfig.Tools))
	}
}

func TestBuildConverseInputSystemPrompt(t *testing.T) {
	req := &ChatRequest{
		Messages: []Message{
			{Role: "system", Content: "Be concise"},
			{Role: "user", Content: "Hi"},
		},
	}
	input, err := buildConverseInput("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(input.System) != 1 {
		t.Errorf("expected 1 system block, got %d", len(input.System))
	}
	if len(input.Messages) != 1 {
		t.Errorf("expected 1 message (system extracted), got %d", len(input.Messages))
	}
}

func TestParseDataURLImageJPEG(t *testing.T) {
	// Valid base64 for a tiny 1-byte "image"
	block, err := parseDataURLImage("data:image/jpeg;base64,/9j/4A==")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if block == nil {
		t.Fatal("expected non-nil block")
	}
}

func TestParseDataURLImagePNG(t *testing.T) {
	block, err := parseDataURLImage("data:image/png;base64,iVBORw0K")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if block == nil {
		t.Fatal("expected non-nil block")
	}
}

func TestParseDataURLImageInvalid(t *testing.T) {
	_, err := parseDataURLImage("not-a-data-url")
	if err == nil {
		t.Error("expected error for invalid data URL")
	}
}

func TestModelsResponseJSON(t *testing.T) {
	resp := ModelsResponse{
		Object: "list",
		Data: []ModelInfo{
			{ID: "anthropic.claude-3", Object: "model", OwnedBy: "Anthropic"},
			{ID: "amazon.titan-text", Object: "model", OwnedBy: "Amazon"},
		},
	}
	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(data), "anthropic.claude-3") {
		t.Error("expected model ID in JSON")
	}
}

func TestHealthResponseJSON(t *testing.T) {
	resp := HealthResponse{Status: "ok", Model: "test", Version: "1.0"}
	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(data), `"status":"ok"`) {
		t.Error("expected status in JSON")
	}
	if !strings.Contains(string(data), `"version":"1.0"`) {
		t.Error("expected version in JSON")
	}
}

// ============================================================
// Feature 1: Reasoning / Extended Thinking
// ============================================================

func TestReasoningEffortMapping(t *testing.T) {
	tests := []struct {
		effort    string
		maxTokens int
		wantMin   int
		wantMax   int
	}{
		// low = 30% of max_tokens, min 1024
		{"low", 4096, 1024, 1228},
		// medium = 60% of max_tokens
		{"medium", 4096, 2000, 2500},
		// high = 100% of max_tokens
		{"high", 4096, 4000, 4096},
		// min budget floor = 1024
		{"low", 100, 1024, 1024},
		{"medium", 100, 1024, 1024},
	}
	for _, tt := range tests {
		t.Run(tt.effort+"_"+strings.ReplaceAll(fmt.Sprintf("%d", tt.maxTokens), "0", "0"), func(t *testing.T) {
			got := computeReasoningBudget(tt.effort, tt.maxTokens)
			if got < tt.wantMin || got > tt.wantMax {
				t.Errorf("computeReasoningBudget(%q, %d) = %d, want [%d, %d]",
					tt.effort, tt.maxTokens, got, tt.wantMin, tt.wantMax)
			}
		})
	}
}

func TestBuildConverseInputWithReasoning(t *testing.T) {
	req := &ChatRequest{
		Messages:        []Message{{Role: "user", Content: "Think hard"}},
		MaxTokens:       4096,
		ReasoningEffort: "high",
	}
	input, err := buildConverseInput("anthropic.claude-sonnet-4", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Verify AdditionalModelRequestFields is set (ThinkingConfig)
	if input.AdditionalModelRequestFields == nil {
		t.Fatal("expected AdditionalModelRequestFields to be set for reasoning")
	}
	var fields map[string]interface{}
	// Note: UnmarshalSmithyDocument may return a secondary validation error but still
	// populates the map correctly — we check the data regardless.
	_ = input.AdditionalModelRequestFields.UnmarshalSmithyDocument(&fields)
	thinking, ok := fields["thinking"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected 'thinking' key in additional fields, got %+v", fields)
	}
	if thinking["type"] != "enabled" {
		t.Errorf("expected thinking.type=enabled, got %v", thinking["type"])
	}
	budget, ok := thinking["budget_tokens"]
	if !ok {
		t.Error("expected budget_tokens in thinking config")
	}
	_ = budget
}

func TestReasoningNotSetWhenAbsent(t *testing.T) {
	req := &ChatRequest{
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		MaxTokens: 1024,
	}
	input, err := buildConverseInput("anthropic.claude-sonnet-4", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if input.AdditionalModelRequestFields != nil {
		var fields map[string]interface{}
		_ = input.AdditionalModelRequestFields.UnmarshalSmithyDocument(&fields)
		if _, hasThinking := fields["thinking"]; hasThinking {
			t.Error("expected no ThinkingConfig when ReasoningEffort is not set")
		}
	}
}

func TestParseConverseOutputWithReasoning(t *testing.T) {
	// Build a mock ConverseOutput with a reasoning content block
	resp := buildMockConverseOutputWithReasoning("I reasoned this out.", "The answer is 42.")
	result, err := parseConverseOutput(resp, "test-model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Choices[0].Message.ReasoningContent != "I reasoned this out." {
		t.Errorf("expected reasoning_content=%q, got %q",
			"I reasoned this out.", result.Choices[0].Message.ReasoningContent)
	}
	content, _ := result.Choices[0].Message.Content.(string)
	if content != "The answer is 42." {
		t.Errorf("expected content=%q, got %q", "The answer is 42.", content)
	}
}

func TestStreamingWithReasoningContent(t *testing.T) {
	mock := &mockConverser{
		streamEvents: []StreamEvent{
			{ReasoningContent: "thinking..."},
			{Text: "The answer."},
			{FinishReason: "stop"},
		},
	}
	h := NewHandlerWithModel(mock, "test-model", "dev", "us-east-1")
	body := `{"model":"test-model","messages":[{"role":"user","content":"Think"}],"stream":true,"reasoning_effort":"high","max_tokens":4096}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	respBody := w.Body.String()
	if !strings.Contains(respBody, "thinking...") {
		t.Error("expected reasoning content in stream output")
	}
}

func TestReasoningEffortChatRequestJSON(t *testing.T) {
	raw := `{"model":"claude-sonnet","messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"medium","max_completion_tokens":8192}`
	var req ChatRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if req.ReasoningEffort != "medium" {
		t.Errorf("expected reasoning_effort=medium, got %q", req.ReasoningEffort)
	}
	if req.MaxCompletionTokens != 8192 {
		t.Errorf("expected max_completion_tokens=8192, got %d", req.MaxCompletionTokens)
	}
}

// ============================================================
// Feature 2: Interleaved Thinking
// ============================================================

func TestExtraBodyThinkingEnabled(t *testing.T) {
	req := &ChatRequest{
		Messages:  []Message{{Role: "user", Content: "Think step by step"}},
		MaxTokens: 8192,
		ExtraBody: map[string]interface{}{
			"thinking": map[string]interface{}{
				"type":          "enabled",
				"budget_tokens": 5000,
			},
		},
	}
	input, err := buildConverseInput("anthropic.claude-sonnet-4", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if input.AdditionalModelRequestFields == nil {
		t.Fatal("expected AdditionalModelRequestFields to be set for interleaved thinking")
	}
	var fields map[string]interface{}
	// UnmarshalSmithyDocument fills the map even when returning secondary error
	_ = input.AdditionalModelRequestFields.UnmarshalSmithyDocument(&fields)
	thinking, ok := fields["thinking"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected thinking key, got %+v", fields)
	}
	if thinking["type"] != "enabled" {
		t.Errorf("expected type=enabled, got %v", thinking["type"])
	}
}

func TestExtraBodyIgnoredWhenEmpty(t *testing.T) {
	req := &ChatRequest{
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		MaxTokens: 1024,
		ExtraBody: map[string]interface{}{},
	}
	input, err := buildConverseInput("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if input.AdditionalModelRequestFields != nil {
		var fields map[string]interface{}
		_ = input.AdditionalModelRequestFields.UnmarshalSmithyDocument(&fields)
		if _, hasThinking := fields["thinking"]; hasThinking {
			t.Error("expected no thinking config for empty extra_body")
		}
	}
}

func TestExtraBodyChatRequestJSON(t *testing.T) {
	raw := `{"model":"claude-sonnet","messages":[{"role":"user","content":"Hi"}],"extra_body":{"thinking":{"type":"enabled","budget_tokens":5000},"anthropic_beta":["interleaved-thinking-2025-05-14"]}}`
	var req ChatRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if req.ExtraBody == nil {
		t.Fatal("expected extra_body to be set")
	}
	thinking, ok := req.ExtraBody["thinking"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected thinking in extra_body, got %+v", req.ExtraBody)
	}
	if thinking["type"] != "enabled" {
		t.Errorf("expected thinking.type=enabled, got %v", thinking["type"])
	}
}

func TestInterleavedThinkingResponse(t *testing.T) {
	// Multi-block response: thinking + tool_use + thinking + text
	resp := buildMockConverseOutputWithReasoning("reasoning about tool", "final answer after tool")
	result, err := parseConverseOutput(resp, "test-model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Choices[0].Message.ReasoningContent == "" {
		t.Error("expected reasoning_content in response with thinking")
	}
}

// ============================================================
// Feature 3: Prompt Caching
// ============================================================

func TestPromptCachingSystemEnabled(t *testing.T) {
	req := &ChatRequest{
		Messages: []Message{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "Hello"},
		},
		ExtraBody: map[string]interface{}{
			"prompt_caching": map[string]interface{}{
				"system": true,
			},
		},
	}
	input, err := buildConverseInput("anthropic.claude-sonnet-4", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// System should have a CachePoint at end
	hasCachePoint := false
	for _, block := range input.System {
		if isCachePointSystemBlock(block) {
			hasCachePoint = true
		}
	}
	if !hasCachePoint {
		t.Error("expected cache point in system blocks when prompt_caching.system=true")
	}
}

func TestPromptCachingDisabledByDefault(t *testing.T) {
	req := &ChatRequest{
		Messages: []Message{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "Hello"},
		},
	}
	input, err := buildConverseInput("anthropic.claude-sonnet-4", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, block := range input.System {
		if isCachePointSystemBlock(block) {
			t.Error("expected no cache point when prompt_caching not requested")
		}
	}
}

func TestPromptCachingMessagesEnabled(t *testing.T) {
	req := &ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there"},
			{Role: "user", Content: "How are you?"},
		},
		ExtraBody: map[string]interface{}{
			"prompt_caching": map[string]interface{}{
				"messages": true,
			},
		},
	}
	input, err := buildConverseInput("anthropic.claude-sonnet-4", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// At least the penultimate message should have a cache point
	hasCachePoint := false
	for _, msg := range input.Messages {
		for _, block := range msg.Content {
			if isCachePointContentBlock(block) {
				hasCachePoint = true
			}
		}
	}
	if !hasCachePoint {
		t.Error("expected cache point in messages when prompt_caching.messages=true")
	}
}

func TestPromptCachingBothEnabled(t *testing.T) {
	req := &ChatRequest{
		Messages: []Message{
			{Role: "system", Content: "Be helpful"},
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi"},
			{Role: "user", Content: "Final"},
		},
		ExtraBody: map[string]interface{}{
			"prompt_caching": map[string]interface{}{
				"system":   true,
				"messages": true,
			},
		},
	}
	input, err := buildConverseInput("anthropic.claude-sonnet-4", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	hasSysCache := false
	for _, block := range input.System {
		if isCachePointSystemBlock(block) {
			hasSysCache = true
		}
	}
	if !hasSysCache {
		t.Error("expected cache point in system blocks")
	}
	hasMsgCache := false
	for _, msg := range input.Messages {
		for _, block := range msg.Content {
			if isCachePointContentBlock(block) {
				hasMsgCache = true
			}
		}
	}
	if !hasMsgCache {
		t.Error("expected cache point in messages")
	}
}

func TestPromptCachingJSONParsing(t *testing.T) {
	raw := `{"model":"claude-sonnet","messages":[{"role":"user","content":"Hi"}],"extra_body":{"prompt_caching":{"system":true,"messages":false}}}`
	var req ChatRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if req.ExtraBody == nil {
		t.Fatal("expected extra_body")
	}
	pc, ok := req.ExtraBody["prompt_caching"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected prompt_caching map, got %T", req.ExtraBody["prompt_caching"])
	}
	if pc["system"] != true {
		t.Errorf("expected system=true, got %v", pc["system"])
	}
}

// ============================================================
// Test helpers
// ============================================================

// buildMockConverseOutputWithReasoning builds a ConverseOutput with a reasoning
// block followed by a text block.
func buildMockConverseOutputWithReasoning(reasoning, text string) *bedrockruntime.ConverseOutput {
	content := []brtypes.ContentBlock{
		&brtypes.ContentBlockMemberReasoningContent{
			Value: &brtypes.ReasoningContentBlockMemberReasoningText{
				Value: brtypes.ReasoningTextBlock{
					Text: aws.String(reasoning),
				},
			},
		},
		&brtypes.ContentBlockMemberText{Value: text},
	}
	return &bedrockruntime.ConverseOutput{
		Output: &brtypes.ConverseOutputMemberMessage{
			Value: brtypes.Message{
				Role:    brtypes.ConversationRoleAssistant,
				Content: content,
			},
		},
		StopReason: brtypes.StopReasonEndTurn,
		Usage: &brtypes.TokenUsage{
			InputTokens:  aws.Int32(10),
			OutputTokens: aws.Int32(20),
		},
	}
}

// isCachePointSystemBlock checks if a SystemContentBlock is a CachePoint.
func isCachePointSystemBlock(block brtypes.SystemContentBlock) bool {
	_, ok := block.(*brtypes.SystemContentBlockMemberCachePoint)
	return ok
}

// isCachePointContentBlock checks if a ContentBlock is a CachePoint.
func isCachePointContentBlock(block brtypes.ContentBlock) bool {
	_, ok := block.(*brtypes.ContentBlockMemberCachePoint)
	return ok
}
