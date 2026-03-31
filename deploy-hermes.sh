#!/usr/bin/env bash
# ============================================================================
# Bedrockify + Hermes Agent — Automated Deploy Script
# ============================================================================
# Deploys bedrockify (OpenAI-compatible Bedrock proxy) and Hermes Agent
# on a fresh EC2 instance with IAM instance profile for Bedrock access.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/inceptionstack/bedrockify/main/deploy-hermes.sh | bash
#
# Prerequisites:
#   - EC2 instance with IAM role that has bedrock:InvokeModel permissions
#   - Ubuntu 22.04+ or Amazon Linux 2023 (arm64 or amd64)
#
# What it does:
#   1. Installs bedrockify and starts it as a systemd daemon on port 8090
#   2. Installs Hermes Agent via official installer (--skip-setup)
#   3. Writes config pointing Hermes at bedrockify (localhost:8090)
#   4. Verifies both services work
# ============================================================================

set -euo pipefail

# Configuration — override with env vars before running
BEDROCK_REGION="${BEDROCK_REGION:-us-east-1}"
BEDROCK_CHAT_MODEL="${BEDROCK_CHAT_MODEL:-us.anthropic.claude-opus-4-6-v1}"
BEDROCK_EMBED_MODEL="${BEDROCK_EMBED_MODEL:-amazon.titan-embed-text-v2:0}"
BEDROCKIFY_PORT="${BEDROCKIFY_PORT:-8090}"
HERMES_MODEL="${HERMES_MODEL:-anthropic/claude-opus-4.6}"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

log()   { echo -e "${CYAN}→${NC} $1"; }
ok()    { echo -e "${GREEN}✓${NC} $1"; }
warn()  { echo -e "${YELLOW}⚠${NC} $1"; }
fail()  { echo -e "${RED}✗${NC} $1"; exit 1; }

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Bedrockify + Hermes Agent — Automated Deploy${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ============================================================================
# Step 1: Install bedrockify
# ============================================================================

log "Installing bedrockify..."
curl -fsSL https://raw.githubusercontent.com/inceptionstack/bedrockify/main/install.sh | bash

# Verify binary
if ! command -v bedrockify &>/dev/null; then
  if [ -x /usr/local/bin/bedrockify ]; then
    export PATH="/usr/local/bin:$PATH"
  else
    fail "bedrockify binary not found after install"
  fi
fi

bedrockify --version
ok "bedrockify installed"

# ============================================================================
# Step 2: Install bedrockify daemon
# ============================================================================

log "Setting up bedrockify daemon (port ${BEDROCKIFY_PORT})..."

# Stop existing if running
sudo systemctl stop bedrockify 2>/dev/null || true

sudo bedrockify install-daemon \
  --region "$BEDROCK_REGION" \
  --model "$BEDROCK_CHAT_MODEL" \
  --embed-model "$BEDROCK_EMBED_MODEL" \
  --port "$BEDROCKIFY_PORT"

sudo systemctl daemon-reload
sudo systemctl enable bedrockify
sudo systemctl start bedrockify

# Wait for it to be ready
sleep 2
for i in 1 2 3 4 5; do
  if curl -sf "http://127.0.0.1:${BEDROCKIFY_PORT}/" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# Verify
if curl -sf "http://127.0.0.1:${BEDROCKIFY_PORT}/" | grep -q '"status":"ok"'; then
  ok "bedrockify daemon running on port ${BEDROCKIFY_PORT}"
else
  fail "bedrockify daemon failed to start. Check: sudo journalctl -u bedrockify"
fi

# ============================================================================
# Step 3: Install Hermes Agent
# ============================================================================

log "Installing Hermes Agent (non-interactive)..."
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --skip-setup

# Ensure hermes is on PATH for current session
export PATH="$HOME/.local/bin:$PATH"

if ! command -v hermes &>/dev/null; then
  fail "hermes command not found after install"
fi

ok "Hermes Agent installed"

# ============================================================================
# Step 4: Configure Hermes to use bedrockify
# ============================================================================

log "Configuring Hermes to use bedrockify..."

mkdir -p ~/.hermes

cat > ~/.hermes/config.yaml << EOF
# Hermes Agent — configured to use bedrockify (Bedrock proxy)
model:
  default: "${HERMES_MODEL}"
  provider: "custom"
  base_url: "http://127.0.0.1:${BEDROCKIFY_PORT}/v1"

terminal:
  backend: "local"
  cwd: "."
  timeout: 180

agent:
  max_turns: 60
  reasoning_effort: "medium"

compression:
  enabled: true
  threshold: 0.50
  summary_model: "${HERMES_MODEL}"

display:
  streaming: true
  tool_progress: all
EOF

cat > ~/.hermes/.env << EOF
# bedrockify handles auth via IAM — no API key needed
OPENAI_API_KEY=not-needed
EOF

ok "Hermes configured → bedrockify at localhost:${BEDROCKIFY_PORT}"

# ============================================================================
# Step 5: Verify end-to-end
# ============================================================================

log "Testing bedrockify chat endpoint..."

CHAT_RESP=$(curl -sf "http://127.0.0.1:${BEDROCKIFY_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${HERMES_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in exactly 3 words.\"}]}" \
  --max-time 30 2>&1) || true

if echo "$CHAT_RESP" | grep -q '"choices"'; then
  ok "Chat completions working"
  REPLY=$(echo "$CHAT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "(parse failed)")
  echo "  → Model replied: $REPLY"
else
  warn "Chat test failed (IAM role may lack bedrock:InvokeModel permission)"
  echo "  Response: $CHAT_RESP"
fi

# ============================================================================
# Done
# ============================================================================

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  ✓ Deploy Complete${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  bedrockify:  http://127.0.0.1:${BEDROCKIFY_PORT} (systemd daemon)"
echo "  hermes:      hermes  (CLI chat agent)"
echo ""
echo "  Start chatting:  hermes"
echo "  Check proxy:     curl http://127.0.0.1:${BEDROCKIFY_PORT}/"
echo "  Proxy logs:      sudo journalctl -u bedrockify -f"
echo ""
