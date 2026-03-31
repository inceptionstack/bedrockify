package bedrockify

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// SSEWriter writes Server-Sent Events to an http.ResponseWriter.
type SSEWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
}

// NewSSEWriter prepares a ResponseWriter for SSE streaming.
// Returns an error if the ResponseWriter doesn't support flushing.
func NewSSEWriter(w http.ResponseWriter) (*SSEWriter, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, fmt.Errorf("streaming not supported by ResponseWriter")
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	return &SSEWriter{w: w, flusher: flusher}, nil
}

// SendChunk writes a single SSE data line and flushes.
func (s *SSEWriter) SendChunk(chunk StreamChunk) error {
	data, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(s.w, "data: %s\n\n", data)
	if err != nil {
		return err
	}
	s.flusher.Flush()
	return nil
}

// SendDone writes the terminal SSE line.
func (s *SSEWriter) SendDone() {
	fmt.Fprintf(s.w, "data: [DONE]\n\n")
	s.flusher.Flush()
}

// Stream reads events from ch and writes SSE chunks to the client.
// model and reqID are used to populate the chunk metadata.
func Stream(w http.ResponseWriter, ch <-chan StreamEvent, model, reqID string) {
	sse, err := NewSSEWriter(w)
	if err != nil {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	created := time.Now().Unix()

	// Role delta first
	roleChunk := StreamChunk{
		ID:      reqID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []StreamChoice{
			{
				Index: 0,
				Delta: Delta{Role: "assistant"},
			},
		},
	}
	_ = sse.SendChunk(roleChunk)

	for event := range ch {
		if event.Err != nil {
			// Send an error in SSE stream format
			errChunk := map[string]interface{}{
				"error": map[string]string{
					"message": event.Err.Error(),
					"type":    "server_error",
				},
			}
			data, _ := json.Marshal(errChunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			return
		}

		if event.Text != "" {
			chunk := StreamChunk{
				ID:      reqID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []StreamChoice{
					{
						Index: 0,
						Delta: Delta{Content: event.Text},
					},
				},
			}
			_ = sse.SendChunk(chunk)
		}

		if event.ToolCallID != "" {
			tc := ToolCall{
				ID:   event.ToolCallID,
				Type: "function",
				Function: ToolCallFunction{
					Name:      event.ToolName,
					Arguments: event.ToolArgs,
				},
			}

			chunk := StreamChunk{
				ID:      reqID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []StreamChoice{
					{
						Index: 0,
						Delta: Delta{ToolCalls: []ToolCall{tc}},
					},
				},
			}
			_ = sse.SendChunk(chunk)
		}

		if event.FinishReason != "" {
			reason := event.FinishReason
			var finishUsage *Usage
			// Collect usage from next event if available
			if event.Usage != nil {
				finishUsage = event.Usage
			}
			chunk := StreamChunk{
				ID:      reqID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []StreamChoice{
					{
						Index:        0,
						Delta:        Delta{},
						FinishReason: &reason,
					},
				},
				Usage: finishUsage,
			}
			_ = sse.SendChunk(chunk)
		}

		if event.FinishReason == "" && event.Usage != nil {
			// Usage-only event (after finish reason was already sent)
			chunk := StreamChunk{
				ID:      reqID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []StreamChoice{},
				Usage:   event.Usage,
			}
			_ = sse.SendChunk(chunk)
		}
	}

	sse.SendDone()
}
