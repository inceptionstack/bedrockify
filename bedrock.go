package bedrockify

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrock"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brdoc "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	smithybearer "github.com/aws/smithy-go/auth/bearer"
)

// BedrockConverser implements Converser against Amazon Bedrock Converse APIs.
type BedrockConverser struct {
	rtClient *bedrockruntime.Client
	brClient *bedrock.Client
	modelID  string
	region   string
}

// NewBedrockConverser creates a Bedrock-backed converser using the default AWS credential chain.
func NewBedrockConverser(region, modelID, baseURL string) (*BedrockConverser, error) {
	opts := []func(*config.LoadOptions) error{
		config.WithRegion(region),
	}
	cfg, err := config.LoadDefaultConfig(context.Background(), opts...)
	if err != nil {
		return nil, err
	}

	var rtOpts []func(*bedrockruntime.Options)
	var brOpts []func(*bedrock.Options)
	if baseURL != "" {
		rtOpts = append(rtOpts, func(o *bedrockruntime.Options) { o.BaseEndpoint = aws.String(baseURL) })
		brOpts = append(brOpts, func(o *bedrock.Options) { o.BaseEndpoint = aws.String(baseURL) })
	}

	return &BedrockConverser{
		rtClient: bedrockruntime.NewFromConfig(cfg, rtOpts...),
		brClient: bedrock.NewFromConfig(cfg, brOpts...),
		modelID:  modelID,
		region:   region,
	}, nil
}

// NewBedrockConverserWithBearerToken creates a Bedrock-backed converser using a
// Bedrock API key (bearer token) for authentication instead of SigV4.
func NewBedrockConverserWithBearerToken(region, modelID, token, baseURL string) (*BedrockConverser, error) {
	opts := []func(*config.LoadOptions) error{
		config.WithRegion(region),
		config.WithBearerAuthTokenProvider(smithybearer.StaticTokenProvider{
			Token: smithybearer.Token{Value: token},
		}),
	}
	cfg, err := config.LoadDefaultConfig(context.Background(), opts...)
	if err != nil {
		return nil, err
	}

	var rtOpts []func(*bedrockruntime.Options)
	var brOpts []func(*bedrock.Options)
	if baseURL != "" {
		rtOpts = append(rtOpts, func(o *bedrockruntime.Options) { o.BaseEndpoint = aws.String(baseURL) })
		brOpts = append(brOpts, func(o *bedrock.Options) { o.BaseEndpoint = aws.String(baseURL) })
	}

	return &BedrockConverser{
		rtClient: bedrockruntime.NewFromConfig(cfg, rtOpts...),
		brClient: bedrock.NewFromConfig(cfg, brOpts...),
		modelID:  modelID,
		region:   region,
	}, nil
}

// --- Converse (non-streaming) ---

// Converse sends a chat request to Bedrock and returns the full response.
func (b *BedrockConverser) Converse(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	modelID := req.Model
	if modelID == "" {
		modelID = b.modelID
	}

	input, err := buildConverseInput(modelID, req)
	if err != nil {
		return nil, err
	}

	start := time.Now()
	resp, err := b.rtClient.Converse(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("bedrock converse: %w", err)
	}
	latency := time.Since(start)

	result, err := parseConverseOutput(resp, modelID)
	if err != nil {
		return nil, err
	}

	log.Printf("converse model=%s prompt_tokens=%d completion_tokens=%d latency=%s",
		modelID, result.Usage.PromptTokens, result.Usage.CompletionTokens, latency.Round(time.Millisecond))

	return result, nil
}

// --- ConverseStream (streaming) ---

// ConverseStream sends a chat request and streams delta events on the returned channel.
func (b *BedrockConverser) ConverseStream(ctx context.Context, req *ChatRequest) (<-chan StreamEvent, error) {
	modelID := req.Model
	if modelID == "" {
		modelID = b.modelID
	}

	input, err := buildConverseInput(modelID, req)
	if err != nil {
		return nil, err
	}

	streamInput := &bedrockruntime.ConverseStreamInput{
		ModelId:                      input.ModelId,
		Messages:                     input.Messages,
		System:                       input.System,
		InferenceConfig:              input.InferenceConfig,
		ToolConfig:                   input.ToolConfig,
		AdditionalModelRequestFields: input.AdditionalModelRequestFields,
	}

	start := time.Now()
	resp, err := b.rtClient.ConverseStream(ctx, streamInput)
	if err != nil {
		return nil, fmt.Errorf("bedrock converse stream: %w", err)
	}

	ch := make(chan StreamEvent, 64)
	go func() {
		defer close(ch)
		stream := resp.GetStream()
		defer stream.Close()

		var (
			currentToolCallID    string
			currentToolName      string
			currentToolArgsAccum strings.Builder
		)

		for event := range stream.Events() {
			switch e := event.(type) {

			case *brtypes.ConverseStreamOutputMemberContentBlockStart:
				switch s := e.Value.Start.(type) {
				case *brtypes.ContentBlockStartMemberToolUse:
					currentToolCallID = aws.ToString(s.Value.ToolUseId)
					currentToolName = aws.ToString(s.Value.Name)
					currentToolArgsAccum.Reset()
				}

			case *brtypes.ConverseStreamOutputMemberContentBlockDelta:
				switch d := e.Value.Delta.(type) {
				case *brtypes.ContentBlockDeltaMemberText:
					ch <- StreamEvent{Text: d.Value}

				case *brtypes.ContentBlockDeltaMemberToolUse:
					if d.Value.Input != nil {
						currentToolArgsAccum.WriteString(aws.ToString(d.Value.Input))
					}

				case *brtypes.ContentBlockDeltaMemberReasoningContent:
					// Emit reasoning/thinking delta
					if rt, ok := d.Value.(*brtypes.ReasoningContentBlockDeltaMemberText); ok {
						ch <- StreamEvent{ReasoningContent: rt.Value}
					}
				}

			case *brtypes.ConverseStreamOutputMemberContentBlockStop:
				// If we were accumulating tool args, emit them now
				if currentToolCallID != "" {
					ch <- StreamEvent{
						ToolCallID: currentToolCallID,
						ToolName:   currentToolName,
						ToolArgs:   currentToolArgsAccum.String(),
					}
					currentToolCallID = ""
					currentToolName = ""
					currentToolArgsAccum.Reset()
				}

			case *brtypes.ConverseStreamOutputMemberMessageStop:
				reason := mapStopReason(string(e.Value.StopReason))
				ch <- StreamEvent{FinishReason: reason}

			case *brtypes.ConverseStreamOutputMemberMetadata:
				if e.Value.Usage != nil {
					inputTokens := int32(0)
					outputTokens := int32(0)
					if e.Value.Usage.InputTokens != nil {
						inputTokens = *e.Value.Usage.InputTokens
					}
					if e.Value.Usage.OutputTokens != nil {
						outputTokens = *e.Value.Usage.OutputTokens
					}
					u := &Usage{
						PromptTokens:     int(inputTokens),
						CompletionTokens: int(outputTokens),
						TotalTokens:      int(inputTokens + outputTokens),
					}
					ch <- StreamEvent{Usage: u}
					log.Printf("converse-stream model=%s prompt_tokens=%d completion_tokens=%d latency=%s",
						modelID, u.PromptTokens, u.CompletionTokens, time.Since(start).Round(time.Millisecond))
				}
			}
		}

		if err := stream.Err(); err != nil {
			ch <- StreamEvent{Err: err}
		}
	}()

	return ch, nil
}

// --- ListModels ---

// ListModels returns available Bedrock foundation models.
func (b *BedrockConverser) ListModels(ctx context.Context) ([]ModelInfo, error) {
	resp, err := b.brClient.ListFoundationModels(ctx, &bedrock.ListFoundationModelsInput{})
	if err != nil {
		return nil, fmt.Errorf("bedrock list models: %w", err)
	}

	models := make([]ModelInfo, 0, len(resp.ModelSummaries))
	for _, m := range resp.ModelSummaries {
		models = append(models, ModelInfo{
			ID:      aws.ToString(m.ModelId),
			Object:  "model",
			OwnedBy: aws.ToString(m.ProviderName),
		})
	}
	return models, nil
}

// --- Input builder ---

func buildConverseInput(modelID string, req *ChatRequest) (*bedrockruntime.ConverseInput, error) {
	// Determine effective max tokens (max_completion_tokens takes precedence)
	maxTokens := req.MaxTokens
	if req.MaxCompletionTokens > 0 {
		maxTokens = req.MaxCompletionTokens
	}

	// Determine if prompt caching is requested via extra_body
	cachingSystem, cachingMessages := extractPromptCachingConfig(req.ExtraBody)

	messages, system, err := convertMessages(req.Messages)
	if err != nil {
		return nil, err
	}

	// Apply system prompt caching if requested
	if cachingSystem && len(system) > 0 {
		system = append(system, &brtypes.SystemContentBlockMemberCachePoint{
			Value: brtypes.CachePointBlock{Type: brtypes.CachePointTypeDefault},
		})
	}

	// Apply message prompt caching if requested (add cache point to penultimate user message)
	if cachingMessages && len(messages) > 1 {
		// Add cache point after the last-but-one turn (before the final user message)
		idx := len(messages) - 2
		if idx >= 0 {
			messages[idx].Content = append(messages[idx].Content,
				&brtypes.ContentBlockMemberCachePoint{
					Value: brtypes.CachePointBlock{Type: brtypes.CachePointTypeDefault},
				})
		}
	}

	input := &bedrockruntime.ConverseInput{
		ModelId:  aws.String(modelID),
		Messages: messages,
		System:   system,
	}

	// Inference config
	ic := &brtypes.InferenceConfiguration{}
	hasIC := false
	if maxTokens > 0 {
		ic.MaxTokens = aws.Int32(int32(maxTokens))
		hasIC = true
	}
	if req.Temperature != nil {
		ic.Temperature = aws.Float32(float32(*req.Temperature))
		hasIC = true
	}
	if req.TopP != nil {
		ic.TopP = aws.Float32(float32(*req.TopP))
		hasIC = true
	}
	if len(req.Stop) > 0 {
		ic.StopSequences = req.Stop
		hasIC = true
	}
	if hasIC {
		input.InferenceConfig = ic
	}

	// Tool config
	if len(req.Tools) > 0 {
		toolConfig, err := convertTools(req.Tools)
		if err != nil {
			return nil, err
		}
		input.ToolConfig = toolConfig
	}

	// Thinking / reasoning config via AdditionalModelRequestFields
	thinkingConfig := buildThinkingConfig(req, maxTokens)
	if thinkingConfig != nil {
		input.AdditionalModelRequestFields = brdoc.NewLazyDocument(thinkingConfig)
	}

	return input, nil
}

// buildThinkingConfig returns the AdditionalModelRequestFields map for thinking/reasoning,
// or nil if no thinking config should be set.
func buildThinkingConfig(req *ChatRequest, maxTokens int) map[string]interface{} {
	// Priority 1: explicit extra_body.thinking config (Feature 2: Interleaved Thinking)
	if req.ExtraBody != nil {
		if thinking, ok := req.ExtraBody["thinking"].(map[string]interface{}); ok {
			if ttype, ok := thinking["type"].(string); ok && ttype == "enabled" {
				return map[string]interface{}{
					"thinking": thinking,
				}
			}
		}
	}

	// Priority 2: reasoning_effort field (Feature 1: Reasoning)
	if req.ReasoningEffort != "" {
		budget := computeReasoningBudget(req.ReasoningEffort, maxTokens)
		return map[string]interface{}{
			"thinking": map[string]interface{}{
				"type":          "enabled",
				"budget_tokens": budget,
			},
		}
	}

	return nil
}

// computeReasoningBudget maps a reasoning effort level to a token budget.
// low=30%, medium=60%, high=100% of maxTokens, with a minimum of 1024.
func computeReasoningBudget(effort string, maxTokens int) int {
	if maxTokens <= 0 {
		maxTokens = 8192 // sensible default
	}
	var fraction float64
	switch effort {
	case "low":
		fraction = 0.30
	case "medium":
		fraction = 0.60
	case "high":
		fraction = 1.0
	default:
		fraction = 0.60
	}
	budget := int(float64(maxTokens) * fraction)
	if budget < 1024 {
		budget = 1024
	}
	return budget
}

// extractPromptCachingConfig reads prompt_caching config from extra_body.
// Returns (cachingSystem, cachingMessages).
func extractPromptCachingConfig(extraBody map[string]interface{}) (bool, bool) {
	if extraBody == nil {
		return false, false
	}
	pc, ok := extraBody["prompt_caching"].(map[string]interface{})
	if !ok {
		return false, false
	}
	cachingSystem, _ := pc["system"].(bool)
	cachingMessages, _ := pc["messages"].(bool)
	return cachingSystem, cachingMessages
}

// --- Message conversion ---

// convertMessages transforms OpenAI messages into Bedrock Converse messages + system blocks.
func convertMessages(msgs []Message) ([]brtypes.Message, []brtypes.SystemContentBlock, error) {
	var bedrockMsgs []brtypes.Message
	var system []brtypes.SystemContentBlock

	for _, m := range msgs {
		switch m.Role {
		case "system":
			text := MessageContent(m)
			system = append(system, &brtypes.SystemContentBlockMemberText{Value: text})

		case "user":
			blocks, err := contentToBlocks(m)
			if err != nil {
				return nil, nil, err
			}
			bedrockMsgs = append(bedrockMsgs, brtypes.Message{
				Role:    brtypes.ConversationRoleUser,
				Content: blocks,
			})

		case "assistant":
			blocks, err := assistantContentToBlocks(m)
			if err != nil {
				return nil, nil, err
			}
			bedrockMsgs = append(bedrockMsgs, brtypes.Message{
				Role:    brtypes.ConversationRoleAssistant,
				Content: blocks,
			})

		case "tool":
			// Tool result — must be packaged as a user message with toolResult block
			block, err := toolResultToBlock(m)
			if err != nil {
				return nil, nil, err
			}
			// If the previous message was also a user role (from a prior tool result), merge.
			if len(bedrockMsgs) > 0 && bedrockMsgs[len(bedrockMsgs)-1].Role == brtypes.ConversationRoleUser {
				bedrockMsgs[len(bedrockMsgs)-1].Content = append(
					bedrockMsgs[len(bedrockMsgs)-1].Content, block)
			} else {
				bedrockMsgs = append(bedrockMsgs, brtypes.Message{
					Role:    brtypes.ConversationRoleUser,
					Content: []brtypes.ContentBlock{block},
				})
			}
		}
	}

	return bedrockMsgs, system, nil
}

// contentToBlocks converts a user Message to Bedrock content blocks.
func contentToBlocks(m Message) ([]brtypes.ContentBlock, error) {
	switch v := m.Content.(type) {
	case string:
		return []brtypes.ContentBlock{
			&brtypes.ContentBlockMemberText{Value: v},
		}, nil
	case []interface{}:
		var blocks []brtypes.ContentBlock
		for _, part := range v {
			p, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			switch p["type"] {
			case "text":
				if t, ok := p["text"].(string); ok {
					blocks = append(blocks, &brtypes.ContentBlockMemberText{Value: t})
				}
			case "image_url":
				if iu, ok := p["image_url"].(map[string]interface{}); ok {
					if url, ok := iu["url"].(string); ok && strings.HasPrefix(url, "data:") {
						imgBlock, err := parseDataURLImage(url)
						if err == nil {
							blocks = append(blocks, imgBlock)
						}
					}
				}
			}
		}
		if len(blocks) == 0 {
			blocks = append(blocks, &brtypes.ContentBlockMemberText{Value: ""})
		}
		return blocks, nil
	case nil:
		return []brtypes.ContentBlock{
			&brtypes.ContentBlockMemberText{Value: ""},
		}, nil
	}
	return nil, fmt.Errorf("unsupported content type: %T", m.Content)
}

// assistantContentToBlocks converts an assistant Message to Bedrock content blocks.
func assistantContentToBlocks(m Message) ([]brtypes.ContentBlock, error) {
	var blocks []brtypes.ContentBlock

	text := MessageContent(m)
	if text != "" {
		blocks = append(blocks, &brtypes.ContentBlockMemberText{Value: text})
	}

	// Tool calls → toolUse blocks
	for _, tc := range m.ToolCalls {
		var inputMap map[string]interface{}
		if tc.Function.Arguments != "" {
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &inputMap); err != nil {
				inputMap = map[string]interface{}{"_raw": tc.Function.Arguments}
			}
		} else {
			inputMap = map[string]interface{}{}
		}
		blocks = append(blocks, &brtypes.ContentBlockMemberToolUse{
			Value: brtypes.ToolUseBlock{
				ToolUseId: aws.String(tc.ID),
				Name:      aws.String(tc.Function.Name),
				Input:     brdoc.NewLazyDocument(inputMap),
			},
		})
	}

	if len(blocks) == 0 {
		blocks = append(blocks, &brtypes.ContentBlockMemberText{Value: ""})
	}

	return blocks, nil
}

// toolResultToBlock converts a tool-role message to a Bedrock toolResult block.
func toolResultToBlock(m Message) (brtypes.ContentBlock, error) {
	text := MessageContent(m)
	return &brtypes.ContentBlockMemberToolResult{
		Value: brtypes.ToolResultBlock{
			ToolUseId: aws.String(m.ToolCallID),
			Content: []brtypes.ToolResultContentBlock{
				&brtypes.ToolResultContentBlockMemberText{Value: text},
			},
		},
	}, nil
}

// parseDataURLImage parses a data: URL into a Bedrock image block.
func parseDataURLImage(dataURL string) (brtypes.ContentBlock, error) {
	parts := strings.SplitN(dataURL, ",", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid data URL")
	}
	header := parts[0]

	mediaType := "image/jpeg"
	if strings.Contains(header, ":") {
		mt := strings.TrimPrefix(header, "data:")
		mt = strings.TrimSuffix(mt, ";base64")
		if mt != "" {
			mediaType = mt
		}
	}

	var format brtypes.ImageFormat
	switch mediaType {
	case "image/jpeg":
		format = brtypes.ImageFormatJpeg
	case "image/png":
		format = brtypes.ImageFormatPng
	case "image/gif":
		format = brtypes.ImageFormatGif
	case "image/webp":
		format = brtypes.ImageFormatWebp
	default:
		format = brtypes.ImageFormatJpeg
	}

	imgBytes, err := base64.StdEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, fmt.Errorf("invalid base64 image: %w", err)
	}

	return &brtypes.ContentBlockMemberImage{
		Value: brtypes.ImageBlock{
			Format: format,
			Source: &brtypes.ImageSourceMemberBytes{
				Value: imgBytes,
			},
		},
	}, nil
}

// --- Tool conversion ---

func convertTools(tools []Tool) (*brtypes.ToolConfiguration, error) {
	var bedrockTools []brtypes.Tool
	for _, t := range tools {
		if t.Type != "function" {
			continue
		}
		bedrockTools = append(bedrockTools, &brtypes.ToolMemberToolSpec{
			Value: brtypes.ToolSpecification{
				Name:        aws.String(t.Function.Name),
				Description: aws.String(t.Function.Description),
				InputSchema: &brtypes.ToolInputSchemaMemberJson{
					Value: brdoc.NewLazyDocument(t.Function.Parameters),
				},
			},
		})
	}
	return &brtypes.ToolConfiguration{Tools: bedrockTools}, nil
}

// --- Output parsing ---

func parseConverseOutput(resp *bedrockruntime.ConverseOutput, modelID string) (*ChatResponse, error) {
	msg, ok := resp.Output.(*brtypes.ConverseOutputMemberMessage)
	if !ok {
		return nil, fmt.Errorf("unexpected output type: %T", resp.Output)
	}

	var textParts []string
	var toolCalls []ToolCall
	var reasoningParts []string

	for _, block := range msg.Value.Content {
		switch b := block.(type) {
		case *brtypes.ContentBlockMemberText:
			textParts = append(textParts, b.Value)

		case *brtypes.ContentBlockMemberToolUse:
			// Unmarshal the document.Interface back to a map, then marshal to JSON string
			var inputMap map[string]interface{}
			if b.Value.Input != nil {
				if err := b.Value.Input.UnmarshalSmithyDocument(&inputMap); err != nil {
					inputMap = map[string]interface{}{}
				}
			}
			argsBytes, _ := json.Marshal(inputMap)
			toolCalls = append(toolCalls, ToolCall{
				ID:   aws.ToString(b.Value.ToolUseId),
				Type: "function",
				Function: ToolCallFunction{
					Name:      aws.ToString(b.Value.Name),
					Arguments: string(argsBytes),
				},
			})

		case *brtypes.ContentBlockMemberReasoningContent:
			// Extract reasoning/thinking text
			if rt, ok := b.Value.(*brtypes.ReasoningContentBlockMemberReasoningText); ok {
				reasoningParts = append(reasoningParts, aws.ToString(rt.Value.Text))
			}
		}
	}

	content := strings.Join(textParts, "")
	reasoningContent := strings.Join(reasoningParts, "")
	finishReason := mapStopReason(string(resp.StopReason))

	respMsg := Message{
		Role:    "assistant",
		Content: content,
	}
	if reasoningContent != "" {
		respMsg.ReasoningContent = reasoningContent
	}
	if len(toolCalls) > 0 {
		respMsg.ToolCalls = toolCalls
		if content == "" {
			respMsg.Content = nil
		}
	}

	var usage Usage
	if resp.Usage != nil {
		inputTokens := int32(0)
		outputTokens := int32(0)
		if resp.Usage.InputTokens != nil {
			inputTokens = *resp.Usage.InputTokens
		}
		if resp.Usage.OutputTokens != nil {
			outputTokens = *resp.Usage.OutputTokens
		}
		usage = Usage{
			PromptTokens:     int(inputTokens),
			CompletionTokens: int(outputTokens),
			TotalTokens:      int(inputTokens + outputTokens),
		}
	}

	return &ChatResponse{
		ID:      newRequestID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   modelID,
		Choices: []Choice{
			{
				Index:        0,
				Message:      respMsg,
				FinishReason: finishReason,
			},
		},
		Usage: usage,
	}, nil
}

// --- Helpers ---

// mapStopReason maps Bedrock stop reasons to OpenAI finish_reason values.
func mapStopReason(reason string) string {
	switch reason {
	case "end_turn":
		return "stop"
	case "tool_use":
		return "tool_calls"
	case "max_tokens":
		return "length"
	case "stop_sequence":
		return "stop"
	case "guardrail_intervened":
		return "content_filter"
	default:
		if reason == "" {
			return ""
		}
		return "stop"
	}
}

// newRequestID generates a simple request ID.
func newRequestID() string {
	return fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
}
