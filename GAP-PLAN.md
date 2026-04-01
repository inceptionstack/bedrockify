# Bedrockify Gap Closure Plan — Sequential TDD

## Execution Model
- One feature per sub-agent spawn
- After each: review code quality, test coverage, DRY compliance
- Fix issues found in review before moving to next feature
- All work in /tmp/bedrockify-src

## Phase 1: Critical Request Handling (breaks real requests)

### 1.1 — topP removal when reasoning/thinking enabled
Bedrock rejects requests with topP when thinking is active.
- In `buildConverseInput()`: when thinkingConfig is non-nil, strip TopP from InferenceConfig
- Test: request with reasoning_effort + top_p → top_p stripped from Converse input
- Test: request without reasoning → top_p preserved

### 1.2 — temperature/topP conflict for specific models
Claude Sonnet 4.5, Haiku 4.5, Opus 4.5 reject both temp+topP simultaneously.
- Add conflictModels set in bedrock.go
- In `buildConverseInput()`: when model matches + both set, drop topP
- Test: conflict model with both → topP removed
- Test: non-conflict model with both → both kept

### 1.3 — reasoning_effort=high should use max_tokens-1 (not max_tokens)
BAG uses `max_tokens - 1` for high. Our auto-bump works but is wasteful.
- Fix `computeReasoningBudget()`: high = maxTokens - 1 (not maxTokens * 1.0)
- Remove or simplify the auto-bump logic (no longer needed for high)
- Keep auto-bump for extra_body.thinking where user sets explicit budget
- Test: high + max=4096 → budget=4095, no auto-bump triggered
- Test: extra_body.thinking with budget > max → still auto-bumps

### 1.4 — DeepSeek v3 reasoning format (string, not object)
DeepSeek v3 uses `reasoning_config: "high"` (plain string).
Claude uses `reasoning_config: {type: "enabled", budget_tokens: N}`.
- In `buildThinkingConfig()`: detect deepseek model → return string format
- Test: deepseek model + reasoning_effort=high → `{"reasoning_config": "high"}`
- Test: claude model + reasoning_effort=high → `{"thinking": {...}}`

### 1.5 — Message role coalescing
Bedrock Converse rejects consecutive same-role messages.
- In `convertMessages()`: merge consecutive same-role messages
- Test: [user, user, assistant] → [user(merged), assistant]
- Test: [user, assistant, user] → unchanged

### 1.6 — No-assistant-prefill for specific models
claude-opus-4-6 rejects conversations ending with assistant message.
- In `buildConverseInput()`: if last message is assistant + model matches, append "please continue" user message
- Test: opus-4-6 ending with assistant → user continuation appended
- Test: other model ending with assistant → unchanged

## Phase 2: Missing Request Features

### 2.1 — developer role support
OpenAI's `developer` role = system message.
- In `convertMessages()`: treat developer role same as system
- Test: developer message → system content block
- Test: mixed system + developer → both in system blocks

### 2.2 — extra_body generic passthrough
BAG passes ALL extra_body keys (except prompt_caching) to additionalModelRequestFields.
- In `buildConverseInput()`: merge remaining extra_body keys into AdditionalModelRequestFields
- Must not conflict with thinking config (merge, don't overwrite)
- Test: extra_body with custom key → passed through
- Test: extra_body with thinking + custom → both present
- Test: prompt_caching key → filtered out

### 2.3 — ENABLE_PROMPT_CACHING global env var
- Add config field + env var support
- When set, enable caching by default (extra_body can override)
- Test: env var set + no extra_body → caching enabled
- Test: env var set + extra_body.prompt_caching.system=false → caching disabled

### 2.4 — stream_options.include_usage
OpenAI param to control usage chunk behavior.
- Add StreamOptions to ChatRequest
- When include_usage=true, send usage as separate final chunk with empty choices
- When include_usage=false or absent, current behavior (usage in last content chunk)
- Test: include_usage=true → separate usage chunk
- Test: absent → current behavior

## Phase 3: Response Quality

### 3.1 — prompt_tokens_details.cached_tokens in response
- Parse cacheReadInputTokens / cacheWriteInputTokens from Bedrock response
- Add PromptTokensDetails to Usage type
- Test: response with cache metrics → cached_tokens populated
- Test: response without cache → nil details

### 3.2 — completion_tokens_details.reasoning_tokens
- Estimate reasoning tokens from reasoning_content length
- Add CompletionTokensDetails to Usage type
- Test: response with reasoning → reasoning_tokens estimated
- Test: response without reasoning → nil details

### 3.3 — Reasoning token estimation in streaming
- Accumulate reasoning tokens during streaming
- Patch into final usage chunk
- Test: streaming with reasoning → usage chunk has reasoning_tokens

## Phase 4: Embedding Enhancements

### 4.1 — Nova Multimodal Embeddings v2
- New embedder implementation for amazon.nova-2-multimodal-embeddings-v1:0
- Supports configurable dimensions (256, 384, 1024, 3072)
- Test: nova embed request → correct InvokeModel call
- Test: invalid dimensions → error

### 4.2 — base64 encoding_format for embeddings
- Add encoding_format to EmbeddingRequest
- When "base64", encode float array as base64 bytes
- Test: float format → float array (existing)
- Test: base64 format → base64 encoded string

### 4.3 — dimensions parameter passthrough
- Add dimensions to EmbeddingRequest
- Pass through to Nova embeddings
- Test: dimensions=384 → passed to model

## Phase 5: Edge Cases & Robustness

### 5.1 — Remote image URL fetching
- In contentToBlocks(): when image_url is http/https (not data:), fetch and convert to base64
- Test: http URL → fetched and converted
- Test: data: URL → existing behavior

### 5.2 — Adaptive retries + connection pooling
- Configure AWS SDK client with retries (adaptive, max 8) and connection pool (50)
- Config-level change, minimal test needed
- Test: verify client config has retry settings

### 5.3 — Reasoning signature handling in streaming
- Handle signature_delta in reasoning blocks
- Test: signature block in stream → handled without error

## Implementation Order (sequential)
1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 2.1 → 2.2 → 2.3 → 2.4 → 3.1 → 3.2 → 3.3 → 4.1 → 4.2 → 4.3 → 5.1 → 5.2 → 5.3
