package bedrockify

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

// Config holds all bedrockify settings.
type Config struct {
	Region      string `toml:"region"`
	Model       string `toml:"model"`
	EmbedModel  string `toml:"embed_model"`
	BaseURL     string `toml:"base_url"`
	BearerToken string `toml:"bearer_token"`
	Port        int    `toml:"port"`
	Host        string `toml:"host"`
}

// DefaultConfig returns config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		Region:     "us-east-1",
		Model:      "us.anthropic.claude-sonnet-4-6",
		EmbedModel: "amazon.titan-embed-text-v2:0",
		Port:       8090,
		Host:       "127.0.0.1",
	}
}

// LoadConfig loads configuration from the first config file found, then
// overlays environment variables. CLI flags should be applied by the caller
// after this returns.
//
// Search order: ./bedrockify.toml, ~/.config/bedrockify/bedrockify.toml, /etc/bedrockify/bedrockify.toml
func LoadConfig() (Config, string, error) {
	cfg := DefaultConfig()

	// Search for config file
	paths := configSearchPaths()
	var loadedFrom string
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			if _, err := toml.DecodeFile(p, &cfg); err != nil {
				return cfg, "", fmt.Errorf("error parsing %s: %w", p, err)
			}
			loadedFrom = p
			break
		}
	}

	// Environment variables override config file
	if v := os.Getenv("AWS_BEARER_TOKEN_BEDROCK"); v != "" && cfg.BearerToken == "" {
		cfg.BearerToken = v
	}

	return cfg, loadedFrom, nil
}

// configSearchPaths returns the ordered list of config file locations.
func configSearchPaths() []string {
	var paths []string

	// 1. Next to the binary
	if exe, err := os.Executable(); err == nil {
		paths = append(paths, filepath.Join(filepath.Dir(exe), "bedrockify.toml"))
	}
	// Also check CWD (useful during development)
	if cwd, err := os.Getwd(); err == nil {
		paths = append(paths, filepath.Join(cwd, "bedrockify.toml"))
	}

	// 2. User config dir
	if home, err := os.UserHomeDir(); err == nil {
		paths = append(paths, filepath.Join(home, ".config", "bedrockify", "bedrockify.toml"))
	}

	// 3. System config
	paths = append(paths, "/etc/bedrockify/bedrockify.toml")

	return paths
}

// WriteExampleConfig writes a commented example config to the given path.
func WriteExampleConfig(path string) error {
	content := `# bedrockify configuration
# Docs: https://github.com/inceptionstack/bedrockify

# AWS region for Bedrock
region = "us-east-1"

# Default chat model ID (client can override per-request)
model = "us.anthropic.claude-opus-4-6-v1"

# Default embedding model ID
embed_model = "amazon.titan-embed-text-v2:0"

# Custom Bedrock endpoint URL (optional — overrides region-based endpoint)
# Useful for VPC endpoints or non-standard deployments.
# base_url = "https://vpce-xxx.bedrock-runtime.us-east-1.vpce.amazonaws.com"

# Bedrock API key for bearer token auth (optional)
# Leave empty to use IAM/SigV4 from instance profile or env credentials.
# bearer_token = "ABSK..."

# Proxy server settings
port = 8090
host = "127.0.0.1"
`
	return os.WriteFile(path, []byte(content), 0644)
}
