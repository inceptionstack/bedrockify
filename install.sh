#!/usr/bin/env bash
set -euo pipefail

REPO="inceptionstack/bedrockify"
INSTALL_DIR="/usr/local/bin"
BINARY="bedrockify"

# Detect OS
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$OS" in
  linux)  OS="linux" ;;
  darwin) OS="darwin" ;;
  *) echo "❌ Unsupported OS: $OS" >&2; exit 1 ;;
esac

# Detect arch
ARCH="$(uname -m)"
case "$ARCH" in
  aarch64|arm64) ARCH="arm64" ;;
  x86_64|amd64)  ARCH="amd64" ;;
  *) echo "❌ Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

URL="https://github.com/${REPO}/releases/latest/download/${BINARY}-${OS}-${ARCH}"

echo "⬇️  Downloading bedrockify (${OS}/${ARCH})..."

# Use sudo if not root and install dir isn't writable
SUDO=""
if [ ! -w "$INSTALL_DIR" ] 2>/dev/null; then
  if command -v sudo &>/dev/null; then
    SUDO="sudo"
  else
    echo "❌ ${INSTALL_DIR} is not writable and sudo is not available." >&2
    echo "   Run as root or install manually." >&2
    exit 1
  fi
fi

$SUDO curl -fsSL "$URL" -o "${INSTALL_DIR}/${BINARY}"
$SUDO chmod +x "${INSTALL_DIR}/${BINARY}"

echo "✅ bedrockify installed to ${INSTALL_DIR}/${BINARY}"
echo ""
bedrockify --version 2>/dev/null || true

# --- Optional daemon setup (Linux with systemd only) ---

# Skip daemon prompt if non-interactive (piped), in a container, or not Linux
if [ ! -t 0 ] || [ "$OS" != "linux" ] || [ ! -d /etc/systemd/system ]; then
  exit 0
fi

echo ""
read -rp "🔧 Install bedrockify as a systemd daemon? [y/N] " INSTALL_DAEMON
case "$INSTALL_DAEMON" in
  [yY]|[yY][eE][sS]) ;;
  *) echo "Skipping daemon setup. Run 'sudo bedrockify install-daemon' later if needed."; exit 0 ;;
esac

read -rp "  AWS region [us-east-1]: " BR_REGION
BR_REGION="${BR_REGION:-us-east-1}"

read -rp "  Default chat model [us.anthropic.claude-opus-4-6-v1]: " BR_MODEL
BR_MODEL="${BR_MODEL:-us.anthropic.claude-opus-4-6-v1}"

read -rp "  Default embed model [amazon.titan-embed-text-v2:0]: " BR_EMBED_MODEL
BR_EMBED_MODEL="${BR_EMBED_MODEL:-amazon.titan-embed-text-v2:0}"

read -rp "  Port [8090]: " BR_PORT
BR_PORT="${BR_PORT:-8090}"

read -rp "  Bedrock API key (bearer token, leave empty for IAM/SigV4): " BR_TOKEN

echo ""
echo "🔧 Installing daemon..."
if [ -n "$BR_TOKEN" ]; then
  $SUDO bedrockify install-daemon --region "$BR_REGION" --model "$BR_MODEL" --embed-model "$BR_EMBED_MODEL" --port "$BR_PORT" --bearer-token "$BR_TOKEN"
else
  $SUDO bedrockify install-daemon --region "$BR_REGION" --model "$BR_MODEL" --embed-model "$BR_EMBED_MODEL" --port "$BR_PORT"
fi

$SUDO systemctl daemon-reload
$SUDO systemctl enable bedrockify
$SUDO systemctl start bedrockify

echo ""
echo "✅ bedrockify daemon is running!"
echo "   Check status: sudo systemctl status bedrockify"
echo "   View logs:    sudo journalctl -u bedrockify -f"
echo ""
echo "Endpoints available on http://127.0.0.1:${BR_PORT}:"
echo "   POST /v1/chat/completions   (chat)"
echo "   POST /v1/embeddings         (embeddings)"
echo "   GET  /v1/models             (list models)"
echo "   GET  /                      (health)"
