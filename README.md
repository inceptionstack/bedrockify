# bedrockify ⚡

> One binary. One port. Full OpenAI-compatible API — chat completions, embeddings, reasoning, tool use, vision, streaming, prompt caching — all backed by Amazon Bedrock.

```
POST /v1/chat/completions  →  Bedrock Converse API   (Claude, Nova, Llama, Mistral, DeepSeek…)
POST /v1/embeddings        →  Bedrock InvokeModel    (Titan Embed, Cohere Embed v3/v4, Nova Embed v2)
GET  /v1/models            →  Lists foundation models
GET  /                     →  Health check
```

---

## Quick Start

```bash
# 1. Install
curl -fsSL https://github.com/inceptionstack/bedrockify/releases/latest/download/install.sh | bash

# 2. Run as daemon (chat + embeddings on port 8090)
sudo bedrockify install-daemon \
  --region us-east-1 \
  --model us.anthropic.claude-opus-4-6-v1 \
  --embed-model amazon.titan-embed-text-v2:0

sudo systemctl daemon-reload && sudo systemctl enable --now bedrockify

# 3. Chat
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-opus","messages":[{"role":"user","content":"Hello!"}]}'

# 4. Embed
curl http://127.0.0.1:8090/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"semantic search rocks","model":"amazon.titan-embed-text-v2:0"}'
```

---

## Supported APIs

| API | Endpoint | Description |
|-----|----------|-------------|
| **Chat Completions** | `POST /v1/chat/completions` | Non-streaming and SSE streaming |
| **Embeddings** | `POST /v1/embeddings` | Titan, Cohere, Nova embedding models |
| **Models** | `GET /v1/models` | List available Bedrock foundation models |
| **Health** | `GET /` | Health check with config info |

### Chat Completions — Supported Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model ID, alias, cross-region ID, or ARN |
| `messages` | array | Chat messages (system, developer, user, assistant, tool) |
| `max_tokens` | int | Maximum output tokens |
| `max_completion_tokens` | int | Max output tokens (takes precedence over max_tokens) |
| `temperature` | float | Sampling temperature (0.0–2.0) |
| `top_p` | float | Nucleus sampling (auto-stripped when reasoning is active) |
| `stream` | bool | Enable SSE streaming |
| `stream_options` | object | `{include_usage: true}` — separate usage chunk |
| `stop` | array | Stop sequences |
| `tools` | array | Function definitions for tool calling |
| `tool_choice` | string/object | `auto`, `required`, or specific tool |
| `reasoning_effort` | string | `low`/`medium`/`high` — enable extended thinking |
| `extra_body` | object | Passthrough to Bedrock `additionalModelRequestFields` |
| `extra_body.thinking` | object | Interleaved thinking config |
| `extra_body.prompt_caching` | object | Prompt caching control |
| `n` | int | Number of completions (only n=1 supported) |

### Chat Completions — Response Fields

| Field | Description |
|-------|-------------|
| `choices[].message.content` | Model response text |
| `choices[].message.reasoning_content` | Chain-of-thought reasoning (when reasoning enabled) |
| `choices[].message.tool_calls` | Function calls requested by the model |
| `choices[].finish_reason` | `stop`, `tool_calls`, `length`, `content_filter` |
| `usage.prompt_tokens` | Input token count |
| `usage.completion_tokens` | Output token count |
| `usage.prompt_tokens_details.cached_tokens` | Tokens served from cache (when caching active) |
| `usage.completion_tokens_details.reasoning_tokens` | Estimated reasoning tokens |

### Embeddings — Supported Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | string/array | Text(s) to embed |
| `model` | string | Embedding model ID |
| `encoding_format` | string | `float` (default) or `base64` |
| `dimensions` | int | Output dimensions (Nova models: 256/384/1024/3072) |

---

## Features

### Core
- **Unified proxy** — chat completions + embeddings on a single port
- **OpenAI-compatible API** — drop-in for any OpenAI SDK client
- **Streaming** — SSE streaming for chat completions with `stream_options.include_usage` support
- **Tool use** — function calling via Bedrock Converse (`auto`, `required`, specific tool)
- **Vision** — image inputs via base64 data URLs AND remote HTTP/HTTPS URLs
- **Model aliases** — short names, OpenRouter IDs, bare Bedrock IDs all work
- **Cross-region inference** — auto-prefixes (`us.`, `eu.`, `ap.`, `global.`) based on region

### Intelligence
- **Reasoning / Extended Thinking** — `reasoning_effort` (low/medium/high) for Claude 3.7/4/4.5 with `reasoning_content` in responses and streaming
- **DeepSeek R1 Reasoning** — automatic format detection (string format for DeepSeek, object format for Claude)
- **Interleaved Thinking** — `extra_body.thinking` for Claude 4/4.5 thinking between tool calls
- **Prompt Caching** — `extra_body.prompt_caching` for up to 90% cost reduction, with `ENABLE_PROMPT_CACHING` env var for global default

### Compatibility
- **Application Inference Profiles** — pass ARN as model ID for cost tracking (no alias mangling)
- **Developer role** — `developer` messages treated as system (OpenAI compatibility)
- **Message coalescing** — consecutive same-role messages automatically merged (Bedrock requirement)
- **No-prefill handling** — models that reject assistant-ending conversations (e.g. claude-opus-4-6) get automatic user continuation
- **temperature/topP conflict** — auto-stripped for models that reject both simultaneously (Claude 4.5, Haiku 4.5, Opus 4.5)
- **Extra body passthrough** — unknown `extra_body` keys forwarded to Bedrock `additionalModelRequestFields`

### Embedding Models
- **Amazon Titan Embed v2** — 1024 dimensions, English + multilingual
- **Cohere Embed v3/v4** — English, multilingual, latest v4 (1536 dims)
- **Amazon Nova Multimodal Embeddings v2** — configurable dimensions (256/384/1024/3072)
- **base64 encoding** — `encoding_format: "base64"` for compact responses

### Infrastructure
- **Auth** — IAM/SigV4 (default) or Bedrock API key (bearer token)
- **Adaptive retries** — max 8 attempts with adaptive backoff
- **Config** — TOML file, env vars, CLI flags (layered priority)
- **Systemd daemon** — `install-daemon` subcommand
- **Self-update** — `update` subcommand

---

## Chat Models

| Alias | Bedrock Model |
|-------|---------------|
| `claude-opus` | `anthropic.claude-opus-4-6-v1` |
| `claude-sonnet` | `anthropic.claude-sonnet-4-6-v1` |
| `claude-haiku` | `anthropic.claude-haiku-4-5-20251001-v1:0` |
| `claude-3.5-sonnet` | `anthropic.claude-3-5-sonnet-20241022-v2:0` |
| `nova-pro` | `amazon.nova-pro-v1:0` |
| `nova-lite` | `amazon.nova-lite-v1:0` |
| `nova-micro` | `amazon.nova-micro-v1:0` |
| `llama-4-maverick` | `meta.llama4-maverick-17b-instruct-v1:0` |
| `deepseek-r1` | `deepseek.deepseek-r1-v1:0` |
| `mistral-large` | `mistral.mistral-large-2407-v1:0` |

OpenRouter-style IDs (e.g. `anthropic/claude-opus-4.6`) also work. Cross-region inference prefixes (`us.`, `eu.`, `ap.`) are added automatically based on your `--region`.

---

## Embedding Models

| Model ID | Dims | Notes |
|----------|------|-------|
| `amazon.titan-embed-text-v2:0` | 1024 | Default, English + multilingual |
| `cohere.embed-english-v3` | 1024 | Cohere v3, English |
| `cohere.embed-multilingual-v3` | 1024 | Cohere v3, multilingual |
| `cohere.embed-v4:0` | 1536 | Cohere v4, latest |
| `amazon.nova-2-multimodal-embeddings-v1:0` | 256–3072 | Nova v2, configurable dimensions |

---

## Reasoning / Extended Thinking

Enable chain-of-thought reasoning for Claude 3.7/4/4.5 and DeepSeek R1. The model's thinking is returned in a separate `reasoning_content` field.

```bash
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet",
    "messages": [{"role": "user", "content": "Which is larger: 9.11 or 9.8?"}],
    "max_completion_tokens": 4096,
    "reasoning_effort": "low"
  }'
```

**Response includes `reasoning_content`:**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "9.8 is larger than 9.11.",
      "reasoning_content": "Comparing 9.11 and 9.8:\n9.8 = 9.80\n9.80 > 9.11"
    }
  }],
  "usage": {
    "completion_tokens_details": {"reasoning_tokens": 12}
  }
}
```

| Level | Budget | Use Case |
|-------|--------|----------|
| `low` | 30% of max_tokens (min 1024) | Quick comparisons, simple reasoning |
| `medium` | 60% of max_tokens (min 1024) | Standard analysis |
| `high` | max_tokens - 1 | Complex math, coding, deep analysis |

**DeepSeek R1** uses a different format automatically — no changes needed on the client side.

Streaming also works — reasoning deltas arrive as `reasoning_content` in delta objects before content deltas.

---

## Interleaved Thinking

For Claude 4/4.5, enable thinking between tool calls using `extra_body.thinking`:

```python
response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[{"role": "user", "content": "What is 15 * 17?"}],
    max_tokens=2048,
    extra_body={"thinking": {"type": "enabled", "budget_tokens": 4096}}
)

print(response.choices[0].message.reasoning_content)  # Chain of thought
print(response.choices[0].message.content)              # Final answer
```

> **Note:** `budget_tokens` can exceed `max_tokens` — bedrockify auto-adjusts `max_tokens` so the request is valid.

---

## Prompt Caching

Reduce costs by up to 90% and latency by up to 85% for repeated prompts. Works with Claude and Nova models.

```python
response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[
        {"role": "system", "content": "Long system prompt..."},
        {"role": "user", "content": "Question"}
    ],
    extra_body={"prompt_caching": {"system": True}}
)

# Check cache hit
if response.usage.prompt_tokens_details:
    print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens}")
```

| Option | Effect |
|--------|--------|
| `{"system": true}` | Cache system prompt |
| `{"messages": true}` | Cache conversation history |
| `{"system": true, "messages": true}` | Cache both |

**Global default:** Set `ENABLE_PROMPT_CACHING=true` env var to enable caching for all requests (per-request `extra_body` overrides).

Requirements: prompt ≥1,024 tokens. Cache TTL is 5 minutes (resets on each hit).

---

## Application Inference Profiles

Pass an Application Inference Profile ARN as the model ID for per-application cost tracking:

```bash
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-app",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

ARNs pass through directly — no alias resolution or cross-region prefix applied.

---

## Usage

### OpenClaw Configuration

```json
{
  "llm": {
    "baseUrl": "http://127.0.0.1:8090/v1",
    "model": "claude-opus",
    "embeddingBaseUrl": "http://127.0.0.1:8090/v1",
    "embeddingModel": "amazon.titan-embed-text-v2:0"
  }
}
```

### Any OpenAI SDK

```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:8090/v1",
    api_key="not-used",  # bedrockify uses AWS credentials
)

# Chat
response = client.chat.completions.create(
    model="claude-opus",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Chat with reasoning
response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[{"role": "user", "content": "Solve: 15 * 17"}],
    max_completion_tokens=4096,
    reasoning_effort="low",
)
print(response.choices[0].message.reasoning_content)

# Embeddings
embeddings = client.embeddings.create(
    model="amazon.titan-embed-text-v2:0",
    input="semantic search rocks",
)

# Embeddings (base64 format)
embeddings = client.embeddings.create(
    model="amazon.titan-embed-text-v2:0",
    input="semantic search rocks",
    encoding_format="base64",
)
```

---

## Configuration

### Config File

Copy `bedrockify.example.toml` to `bedrockify.toml` (next to binary, CWD, or `~/.config/bedrockify/`):

```toml
region      = "us-east-1"
model       = "us.anthropic.claude-opus-4-6-v1"
embed_model = "amazon.titan-embed-text-v2:0"
port        = 8090
host        = "127.0.0.1"

# Optional: Bedrock API key instead of IAM
# bearer_token = "ABSK..."
```

### CLI Flags

```
--region        AWS region (default: us-east-1)
--model         Default chat model (default: us.anthropic.claude-sonnet-4-6)
--embed-model   Default embedding model (default: amazon.titan-embed-text-v2:0)
--port          Port to listen on (default: 8090)
--host          Bind host (default: 127.0.0.1)
--bearer-token  Bedrock API key (alternative to IAM/SigV4)
--base-url      Custom Bedrock endpoint URL
--init          Write example bedrockify.toml and exit
--version       Show version
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `AWS_BEARER_TOKEN_BEDROCK` | Bedrock API key (overrides config) |
| `ENABLE_PROMPT_CACHING` | Set `true` to enable prompt caching globally |
| Standard AWS env vars | `AWS_ACCESS_KEY_ID`, `AWS_REGION`, etc. |

---

## Architecture

```
Client (OpenAI SDK / OpenClaw / Hermes / curl)
         │
         ▼
  bedrockify :8090
         │
    ┌────┴────────────────────────┐
    │                             │
    ▼                             ▼
POST /v1/chat/completions   POST /v1/embeddings
    │                             │
    ▼                             ▼
BedrockConverser            BedrockEmbedder
(Converse API)              (InvokeModel API)
    │                             │
    ├─ Reasoning config           ├─ Titan Embed v2
    ├─ Prompt caching             ├─ Cohere Embed v3/v4
    ├─ Tool config                ├─ Nova Embed v2
    ├─ Message coalescing         └─ base64 encoding
    ├─ topP conflict resolution
    ├─ No-prefill handling
    └─ Image fetching
    │                             │
    ▼                             ▼
Amazon Bedrock              Amazon Bedrock
(Claude/Nova/Llama/         (Titan/Cohere/
 Mistral/DeepSeek)           Nova)
```

---

## IAM Permissions

Minimum IAM policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:Converse",
        "bedrock:ConverseStream",
        "bedrock:ListFoundationModels"
      ],
      "Resource": "*"
    }
  ]
}
```

---

## Building from Source

```bash
git clone https://github.com/inceptionstack/bedrockify
cd bedrockify
go build -o bedrockify ./cmd/bedrockify/
go test ./... -v  # 202 tests
```

---

## License

MIT — see [LICENSE](LICENSE)
