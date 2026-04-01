package bedrockify

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
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

// awsRetryConfig holds testable retry configuration.
type awsRetryConfig struct {
	maxAttempts int
	mode        string
}

// buildAWSRetryConfig returns the standard retry configuration used by Bedrock clients.
func buildAWSRetryConfig() awsRetryConfig {
	return awsRetryConfig{
		maxAttempts: 8,
		mode:        "adaptive",
	}
}

// NewBedrockConverser creates a Bedrock-backed converser using the default AWS credential chain.
func NewBedrockConverser(region, modelID, baseURL string) (*BedrockConverser, error) {
	retryCfg := buildAWSRetryConfig()
	opts := []func(*config.LoadOptions) error{
		config.WithRegion(region),
		config.WithRetryMaxAttempts(retryCfg.maxAttempts),
		config.WithRetryMode("adaptive"),
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
					switch rt := d.Value.(type) {
					case *brtypes.ReasoningContentBlockDeltaMemberText:
						ch <- StreamEvent{ReasoningContent: rt.Value}
					case *brtypes.ReasoningContentBlockDeltaMemberSignature:
						// Feature 5.3: signature delta — emit as SignatureBlock event
						ch <- StreamEvent{ReasoningSignature: rt.Value}
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

// noPrefillModels is the set of model ID substrings that reject conversations
// ending with an assistant message (no-prefill constraint).
var noPrefillModels = []string{
	"claude-opus-4-6",
}

// isNoPrefillModel returns true if the model rejects assistant-ending conversations.
func isNoPrefillModel(modelID string) bool {
	lower := strings.ToLower(modelID)
	for _, s := range noPrefillModels {
		if strings.Contains(lower, s) {
			return true
		}
	}
	return false
}

// conflictModels is the set of model ID substrings where Bedrock rejects
// requests that include both temperature AND topP simultaneously.
// For these models, we drop topP when both are specified.
var conflictModels = []string{
	"claude-sonnet-4-5",
	"claude-haiku-4-5",
	"claude-opus-4-5",
}

// isConflictModel returns true if the model rejects temp+topP simultaneously.
func isConflictModel(modelID string) bool {
	lower := strings.ToLower(modelID)
	for _, s := range conflictModels {
		if strings.Contains(lower, s) {
			return true
		}
	}
	return false
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

	// Feature 1.6: no-prefill — append user continuation when last message is assistant
	if len(messages) > 0 &&
		messages[len(messages)-1].Role == brtypes.ConversationRoleAssistant &&
		isNoPrefillModel(modelID) {
		messages = append(messages, brtypes.Message{
			Role: brtypes.ConversationRoleUser,
			Content: []brtypes.ContentBlock{
				&brtypes.ContentBlockMemberText{Value: "Please continue."},
			},
		})
		input.Messages = messages
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

	// For conflict models, drop topP when both temperature and topP are set
	if ic.Temperature != nil && ic.TopP != nil && isConflictModel(modelID) {
		ic.TopP = nil
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
	thinkingConfig, budgetTokens := buildThinkingConfig(req, maxTokens)
	if thinkingConfig != nil {
		// Merge extra_body passthrough keys into thinkingConfig
		merged := mergeExtraBodyPassthrough(thinkingConfig, req.ExtraBody)
		input.AdditionalModelRequestFields = brdoc.NewLazyDocument(merged)
		// Bedrock rejects requests with topP when thinking/reasoning is active.
		if ic.TopP != nil {
			ic.TopP = nil
		}
		// Bedrock requires max_tokens > budget_tokens for thinking.
		// Auto-bump max_tokens if needed so the request doesn't get rejected.
		if budgetTokens > 0 && maxTokens <= budgetTokens {
			adjusted := budgetTokens + 1024
			ic.MaxTokens = aws.Int32(int32(adjusted))
			if !hasIC {
				hasIC = true
			}
			input.InferenceConfig = ic
		}
	} else {
		// No thinking config — still pass through any non-caching extra_body keys
		passthrough := buildExtraBodyPassthrough(req.ExtraBody)
		if len(passthrough) > 0 {
			input.AdditionalModelRequestFields = brdoc.NewLazyDocument(passthrough)
		}
	}

	return input, nil
}

// buildThinkingConfig returns the AdditionalModelRequestFields map for thinking/reasoning
// and the budget_tokens value (0 if no thinking config). Returns (nil, 0) if no thinking
// config should be set.
func buildThinkingConfig(req *ChatRequest, maxTokens int) (map[string]interface{}, int) {
	modelID := req.Model

	// Priority 1: explicit extra_body.thinking config (Feature 2: Interleaved Thinking)
	if req.ExtraBody != nil {
		if thinking, ok := req.ExtraBody["thinking"].(map[string]interface{}); ok {
			if ttype, ok := thinking["type"].(string); ok && ttype == "enabled" {
				budget := 0
				if bt, ok := thinking["budget_tokens"].(float64); ok {
					budget = int(bt)
				}
				return map[string]interface{}{
					"thinking": thinking,
				}, budget
			}
		}
	}

	// Priority 2: reasoning_effort field (Feature 1: Reasoning)
	if req.ReasoningEffort != "" {
		budget := computeReasoningBudget(req.ReasoningEffort, maxTokens)

		// DeepSeek v3 uses string format: reasoning_config: "high"
		if isDeepSeekModel(modelID) {
			return map[string]interface{}{
				"reasoning_config": req.ReasoningEffort,
			}, budget
		}

		return map[string]interface{}{
			"thinking": map[string]interface{}{
				"type":          "enabled",
				"budget_tokens": budget,
			},
		}, budget
	}

	return nil, 0
}

// isDeepSeekModel returns true if the model is a DeepSeek variant.
func isDeepSeekModel(modelID string) bool {
	lower := strings.ToLower(modelID)
	return strings.Contains(lower, "deepseek")
}

// computeReasoningBudget maps a reasoning effort level to a token budget.
// low=30%, medium=60%, high=max_tokens-1 (BAG spec), with a minimum of 1024.
func computeReasoningBudget(effort string, maxTokens int) int {
	if maxTokens <= 0 {
		maxTokens = 8192 // sensible default
	}
	var budget int
	switch effort {
	case "low":
		budget = int(float64(maxTokens) * 0.30)
	case "medium":
		budget = int(float64(maxTokens) * 0.60)
	case "high":
		// BAG spec: high = max_tokens - 1
		budget = maxTokens - 1
	default:
		budget = int(float64(maxTokens) * 0.60)
	}
	if budget < 1024 {
		budget = 1024
	}
	return budget
}

// extractPromptCachingConfig reads prompt_caching config from extra_body,
// falling back to the ENABLE_PROMPT_CACHING environment variable as global default.
// Returns (cachingSystem, cachingMessages).
func extractPromptCachingConfig(extraBody map[string]interface{}) (bool, bool) {
	globalDefault := os.Getenv("ENABLE_PROMPT_CACHING") == "1" || os.Getenv("ENABLE_PROMPT_CACHING") == "true"

	if extraBody == nil {
		return globalDefault, globalDefault
	}
	pc, ok := extraBody["prompt_caching"].(map[string]interface{})
	if !ok {
		return globalDefault, globalDefault
	}
	// explicit false overrides global default
	cachingSystem := globalDefault
	cachingMessages := globalDefault
	if v, ok := pc["system"].(bool); ok {
		cachingSystem = v
	}
	if v, ok := pc["messages"].(bool); ok {
		cachingMessages = v
	}
	return cachingSystem, cachingMessages
}

// filteredExtraBodyKeys are keys in extra_body that are handled separately
// and should NOT be passed through as-is to AdditionalModelRequestFields.
var filteredExtraBodyKeys = map[string]bool{
	"prompt_caching": true,
	"thinking":       true, // handled by buildThinkingConfig
}

// buildExtraBodyPassthrough returns a map of extra_body keys that should be
// passed through to AdditionalModelRequestFields (excluding known handled keys).
func buildExtraBodyPassthrough(extraBody map[string]interface{}) map[string]interface{} {
	if extraBody == nil {
		return nil
	}
	result := make(map[string]interface{})
	for k, v := range extraBody {
		if !filteredExtraBodyKeys[k] {
			result[k] = v
		}
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

// mergeExtraBodyPassthrough merges extra_body passthrough keys into an existing
// AdditionalModelRequestFields map without overwriting existing keys.
func mergeExtraBodyPassthrough(base map[string]interface{}, extraBody map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{}, len(base))
	for k, v := range base {
		result[k] = v
	}
	for k, v := range extraBody {
		if !filteredExtraBodyKeys[k] {
			if _, exists := result[k]; !exists {
				result[k] = v
			}
		}
	}
	return result
}

// --- Message conversion ---

// convertMessages transforms OpenAI messages into Bedrock Converse messages + system blocks.
func convertMessages(msgs []Message) ([]brtypes.Message, []brtypes.SystemContentBlock, error) {
	var bedrockMsgs []brtypes.Message
	var system []brtypes.SystemContentBlock

	for _, m := range msgs {
		switch m.Role {
		case "system", "developer":
			text := MessageContent(m)
			system = append(system, &brtypes.SystemContentBlockMemberText{Value: text})

		case "user":
			blocks, err := contentToBlocks(m)
			if err != nil {
				return nil, nil, err
			}
			// Coalesce consecutive same-role messages
			if len(bedrockMsgs) > 0 && bedrockMsgs[len(bedrockMsgs)-1].Role == brtypes.ConversationRoleUser {
				bedrockMsgs[len(bedrockMsgs)-1].Content = append(bedrockMsgs[len(bedrockMsgs)-1].Content, blocks...)
			} else {
				bedrockMsgs = append(bedrockMsgs, brtypes.Message{
					Role:    brtypes.ConversationRoleUser,
					Content: blocks,
				})
			}

		case "assistant":
			blocks, err := assistantContentToBlocks(m)
			if err != nil {
				return nil, nil, err
			}
			// Coalesce consecutive same-role messages
			if len(bedrockMsgs) > 0 && bedrockMsgs[len(bedrockMsgs)-1].Role == brtypes.ConversationRoleAssistant {
				bedrockMsgs[len(bedrockMsgs)-1].Content = append(bedrockMsgs[len(bedrockMsgs)-1].Content, blocks...)
			} else {
				bedrockMsgs = append(bedrockMsgs, brtypes.Message{
					Role:    brtypes.ConversationRoleAssistant,
					Content: blocks,
				})
			}

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
					if url, ok := iu["url"].(string); ok {
						var imgBlock brtypes.ContentBlock
						var err error
						if strings.HasPrefix(url, "data:") {
							imgBlock, err = parseDataURLImage(url)
						} else if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
							imgBlock, err = fetchRemoteImage(url)
						}
						if err == nil && imgBlock != nil {
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

// fetchRemoteImage fetches an image from a URL and returns a Bedrock image block.
func fetchRemoteImage(url string) (brtypes.ContentBlock, error) {
	resp, err := http.Get(url) //nolint:gosec
	if err != nil {
		return nil, fmt.Errorf("fetch image: %w", err)
	}
	defer resp.Body.Close()

	imgBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read image: %w", err)
	}

	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "image/jpeg"
	}
	// Strip parameters (e.g. "image/png; charset=...")
	if idx := strings.Index(contentType, ";"); idx >= 0 {
		contentType = strings.TrimSpace(contentType[:idx])
	}

	var format brtypes.ImageFormat
	switch contentType {
	case "image/jpeg", "image/jpg":
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

	return &brtypes.ContentBlockMemberImage{
		Value: brtypes.ImageBlock{
			Format: format,
			Source: &brtypes.ImageSourceMemberBytes{
				Value: imgBytes,
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
		// Feature 3.1: prompt_tokens_details.cached_tokens
		cachedRead := int32(0)
		if resp.Usage.CacheReadInputTokens != nil {
			cachedRead = *resp.Usage.CacheReadInputTokens
		}
		if resp.Usage.CacheWriteInputTokens != nil {
			// both read and write count as cached
			cachedRead += *resp.Usage.CacheWriteInputTokens
		}
		if cachedRead > 0 {
			usage.PromptTokensDetails = &PromptTokensDetails{CachedTokens: int(cachedRead)}
		}
		// Feature 3.2: completion_tokens_details.reasoning_tokens
		if reasoningContent != "" {
			// Estimate: ~1 token per 4 chars
			estimated := len(reasoningContent) / 4
			if estimated < 1 {
				estimated = 1
			}
			usage.CompletionTokensDetails = &CompletionTokensDetails{ReasoningTokens: estimated}
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
