# bedrockify ⚡

> You want access to bedrock native embedding models for your OpenClaw/Hermes/Other agent.
> You want access to Anthropic modles in bedrock under OpenAI compatible completions API for OpenClaw/Hermes/Other
> Now you can have both. One binary. One port. OpenAI-compatible **chat completions** AND **embeddings** — both backed by Amazon Bedrock.

```
POST /v1/chat/completions  →  Bedrock Converse API   (Claude, Nova, Llama, Mistral, DeepSeek…)
POST /v1/embeddings        →  Bedrock InvokeModel    (Titan Embed, Cohere Embed v3/v4)
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

## Features

- **Unified proxy** — chat completions + embeddings on a single port
- **OpenAI-compatible API** — drop-in for any OpenAI SDK client
- **Chat models**: Claude Opus/Sonnet/Haiku, Nova Pro/Lite/Micro, Llama 4, Mistral Large, DeepSeek R1
- **Embedding models**: Amazon Titan Embed v2, Cohere Embed v3/v4 (English + Multilingual)
- **Streaming** — SSE streaming for chat completions
- **Tool use** — function calling via Bedrock Converse
- **Vision** — image inputs (base64 data URLs)
- **Reasoning / Extended Thinking** — `reasoning_effort` (low/medium/high) for Claude 3.7/4/4.5 + DeepSeek R1 with `reasoning_content` in responses
- **Interleaved Thinking** — `extra_body.thinking` for Claude 4/4.5 thinking between tool calls
- **Prompt Caching** — `extra_body.prompt_caching` for up to 90% cost reduction on repeated prompts
- **Application Inference Profiles** — pass ARN as model ID for cost tracking
- **Model aliases** — short names, OpenRouter IDs, bare Bedrock IDs all work
- **Auth** — IAM/SigV4 (default) or Bedrock API key (bearer token)
- **Config** — TOML file, env vars, CLI flags (layered priority)
- **Systemd daemon** — `install-daemon` subcommand
- **Self-update** — `update` subcommand

---

## Usage

### OpenClaw Configuration

In your OpenClaw `config.json` or equivalent:

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

Both chat and embeddings route to the same bedrockify instance on the same port.

### Hermes / LiteLLM / Any OpenAI SDK

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

# Embeddings
embeddings = client.embeddings.create(
    model="amazon.titan-embed-text-v2:0",
    input="semantic search rocks",
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

| Variable                  | Description                                 |
|---------------------------|---------------------------------------------|
| `AWS_BEARER_TOKEN_BEDROCK` | Bedrock API key (overrides config if set)  |
| Standard AWS env vars      | `AWS_ACCESS_KEY_ID`, `AWS_REGION`, etc.    |

---

## Model Aliases (Chat)

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

OpenRouter-style IDs (e.g. `anthropic/claude-opus-4.6`) also work.

Cross-region inference prefixes (`us.`, `eu.`, `ap.`) are added automatically based on your `--region`.

---

## Embedding Models

| Model ID | Dims | Notes |
|----------|------|-------|
| `amazon.titan-embed-text-v2:0` | 1024 | Default, English + multilingual |
| `cohere.embed-english-v3` | 1024 | Cohere v3, English |
| `cohere.embed-multilingual-v3` | 1024 | Cohere v3, multilingual |
| `cohere.embed-v4:0` | 1536 | Cohere v4, latest |

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
      "reasoning_content": "Comparing 9.11 and 9.8:\n9.8 = 9.80\n9.80 > 9.11\nSo 9.8 is larger."
    }
  }]
}
```

| Level | Budget | Use Case |
|-------|--------|----------|
| `low` | 30% of max_tokens (min 1024) | Quick comparisons, simple reasoning |
| `medium` | 60% of max_tokens (min 1024) | Standard analysis |
| `high` | 100% of max_tokens (min 1024) | Complex math, coding, deep analysis |

Streaming also works — reasoning deltas arrive before content deltas:
```bash
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-sonnet", "messages": [{"role": "user", "content": "9.11 vs 9.8?"}], "max_completion_tokens": 4096, "reasoning_effort": "low", "stream": true}'
```

---

## Interleaved Thinking

For Claude 4/4.5, enable thinking between tool calls using `extra_body.thinking`. The model thinks before and after each tool result, producing better agentic decisions.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="not-used")

response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[{"role": "user", "content": "What is 15 * 17?"}],
    max_tokens=2048,
    extra_body={
        "thinking": {"type": "enabled", "budget_tokens": 4096}
    }
)

print(response.choices[0].message.reasoning_content)  # Chain of thought
print(response.choices[0].message.content)              # Final answer
```

> **Note:** `budget_tokens` can exceed `max_tokens` — bedrockify auto-adjusts `max_tokens` so the request is valid.

---

## Prompt Caching

Reduce costs by up to 90% and latency by up to 85% for workloads with repeated prompts (system prompts, few-shot examples). Works with Claude and Nova models.

```python
response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[
        {"role": "system", "content": "You are an expert assistant with..."},  # Long system prompt
        {"role": "user", "content": "Help me with this task"}
    ],
    extra_body={
        "prompt_caching": {"system": True}  # Cache the system prompt
    }
)
```

**Cache options:**

| Option | Effect |
|--------|--------|
| `{"system": true}` | Cache system prompt |
| `{"messages": true}` | Cache conversation history |
| `{"system": true, "messages": true}` | Cache both |

Requirements: prompt must be ≥1,024 tokens for caching to activate. Cache TTL is 5 minutes (resets on each cache hit).

---

## Application Inference Profiles

Pass an Application Inference Profile ARN as the model ID for per-application cost tracking:

```bash
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-app",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

ARNs pass through directly — no alias resolution or cross-region prefix is applied.

---

## Architecture

```
Client (OpenAI SDK / OpenClaw / curl)
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
    ▼                             ▼
Amazon Bedrock              Amazon Bedrock
(Claude/Nova/etc.)          (Titan/Cohere)
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
go test ./... -v
```

---

## License

MIT — see [LICENSE](LICENSE)
