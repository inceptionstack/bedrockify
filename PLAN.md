# Bedrockify Feature Plan — 4 Features, TDD

## Context
Bedrockify is an OpenAI-compatible proxy for Amazon Bedrock (Go, ~4K LOC).
Repo: /tmp/bedrockify-src
Tests: `go test ./... -v`

All features should be implemented using Red-Green-Refactor TDD:
1. Write failing test first
2. Write minimum code to pass
3. Refactor
4. Repeat

## Existing Architecture
- `types.go` — Request/response types (ChatRequest, ChatResponse, StreamEvent, etc.)
- `bedrock.go` — BedrockConverser implementing Converser interface (Converse, ConverseStream, ListModels)
- `handler.go` — HTTP handlers (handleChatCompletions, handleEmbeddings, handleModels)
- `stream.go` — SSE streaming helpers
- `embed_bedrock.go` — BedrockEmbedder
- `aliases.go` — Model alias resolution
- `config.go` — TOML config + env + CLI flag loading

## Feature 1: Reasoning / Extended Thinking
### What
Support `reasoning_effort` parameter (low/medium/high) in ChatRequest.
Return `reasoning_content` field in response messages (CoT/thoughts).
Works for Claude 3.7/4/4.5 and DeepSeek R1.

### Implementation
1. Add `ReasoningEffort` and `MaxCompletionTokens` fields to `ChatRequest` in types.go
2. Add `ReasoningContent` field to `Message` and `Delta` in types.go
3. In `bedrock.go` `buildConverseInput()`: when `ReasoningEffort` is set, add `PerformanceConfig` with `ThinkingConfig` to the Converse input (budget_tokens computed from reasoning_effort level + max_tokens)
4. In `parseConverseOutput()`: extract thinking blocks from response and put in `ReasoningContent`
5. In streaming (`ConverseStream`): emit reasoning deltas as `StreamEvent` with a new `ReasoningContent` field
6. In `stream.go`: include `reasoning_content` in Delta when present
7. In handler: pass through the new fields

### Tests (write FIRST)
- `TestReasoningEffortMapping` — low=30%, medium=60%, high=100% of max_tokens, min 1024
- `TestBuildConverseInputWithReasoning` — verify ThinkingConfig is set in Converse input
- `TestParseConverseOutputWithReasoning` — response with thinking blocks → reasoning_content field
- `TestStreamingWithReasoningContent` — streaming chunks include reasoning_content delta
- `TestReasoningNotSetWhenAbsent` — no reasoning_effort → no ThinkingConfig
- `TestDeepSeekR1Reasoning` — DeepSeek model reasoning passthrough (no explicit reasoning_effort needed)

## Feature 2: Interleaved Thinking
### What
Support `extra_body.thinking` and `extra_body.anthropic_beta` for Claude 4/4.5 interleaved thinking between tool calls.

### Implementation
1. Add `ExtraBody` field to `ChatRequest` (map[string]interface{})
2. In `buildConverseInput()`: when `extra_body.thinking.type == "enabled"`, set ThinkingConfig with the specified budget_tokens and enable interleaved thinking beta
3. In response parsing: handle multiple thinking blocks interleaved with tool use blocks
4. In streaming: emit thinking deltas between tool call events

### Tests (write FIRST)
- `TestExtraBodyThinkingEnabled` — extra_body with thinking config → correct Converse input
- `TestInterleavedThinkingResponse` — response with thinking + tool_use + thinking + text → correct ordering
- `TestInterleavedThinkingStreaming` — streaming with interleaved blocks
- `TestExtraBodyIgnoredWhenEmpty` — no extra_body → normal behavior
- `TestAnthropic BetaHeader` — anthropic_beta array passed through

## Feature 3: Prompt Caching
### What
Support `extra_body.prompt_caching` to enable Bedrock prompt caching for Claude and Nova models.

### Implementation
1. Parse `prompt_caching` from extra_body: `{system: bool, messages: bool}`
2. In `buildConverseInput()`: when prompt_caching.system is true, add cache point after system content blocks
3. When prompt_caching.messages is true, add cache points to appropriate message content
4. Also support global `ENABLE_PROMPT_CACHING=true` env var
5. In response: expose cache hit/miss stats in usage (prompt_tokens_details.cached_tokens)

### Tests (write FIRST)
- `TestPromptCachingSystemEnabled` — cache point added after system blocks
- `TestPromptCachingMessagesEnabled` — cache points added to message content
- `TestPromptCachingBothEnabled` — both system and messages cached
- `TestPromptCachingDisabledByDefault` — no cache points when not requested
- `TestPromptCachingGlobalEnvVar` — ENABLE_PROMPT_CACHING=true enables caching
- `TestPromptCachingUsageStats` — cached_tokens in response usage

## Feature 4: Application Inference Profiles
### What
Support passing ARN as model ID for cost tracking via Application Inference Profiles.

### Implementation
1. In `buildConverseInput()`: detect ARN format (`arn:aws:bedrock:...`) and pass through as model ID without alias resolution
2. In `resolveModel()` (aliases.go): skip alias lookup for ARN-format model IDs
3. In response: preserve the ARN as the model field

### Tests (write FIRST)
- `TestARNModelIDPassthrough` — ARN passed directly, no alias resolution
- `TestARNModelIDInResponse` — response.model contains the ARN
- `TestARNModelIDStreaming` — streaming with ARN model works
- `TestNonARNStillResolvesAliases` — regular model IDs still go through alias resolution

## Implementation Order
1. Feature 4 (Application Inference Profiles) — smallest, isolated change
2. Feature 1 (Reasoning) — core feature, needed for Feature 2
3. Feature 2 (Interleaved Thinking) — builds on reasoning
4. Feature 3 (Prompt Caching) — independent but needs Bedrock API understanding

## Rules
- TDD: write failing test → write code → pass → refactor
- Run `go test ./... -v` after each feature
- Keep existing tests passing
- Do not break backward compatibility
- Use the existing code style (check existing tests for patterns)
