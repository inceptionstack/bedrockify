package bedrockify

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.Region != "us-east-1" {
		t.Errorf("expected region=us-east-1, got %q", cfg.Region)
	}
	if cfg.Port != 8090 {
		t.Errorf("expected port=8090, got %d", cfg.Port)
	}
	if cfg.Host != "127.0.0.1" {
		t.Errorf("expected host=127.0.0.1, got %q", cfg.Host)
	}
	if cfg.EmbedModel != "amazon.titan-embed-text-v2:0" {
		t.Errorf("expected embed_model=amazon.titan-embed-text-v2:0, got %q", cfg.EmbedModel)
	}
}

func TestLoadConfigFromFile(t *testing.T) {
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "bedrockify.toml")
	content := `
region = "eu-west-1"
model = "anthropic.claude-3-haiku"
embed_model = "cohere.embed-english-v3"
port = 9999
host = "0.0.0.0"
bearer_token = "test-token"
base_url = "https://custom.endpoint.com"
`
	if err := os.WriteFile(cfgFile, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	// Change to temp dir so CWD search finds it
	orig, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(orig)

	cfg, loadedFrom, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if loadedFrom == "" {
		t.Fatal("expected config to be loaded from file")
	}
	if cfg.Region != "eu-west-1" {
		t.Errorf("expected region=eu-west-1, got %q", cfg.Region)
	}
	if cfg.Model != "anthropic.claude-3-haiku" {
		t.Errorf("expected model=anthropic.claude-3-haiku, got %q", cfg.Model)
	}
	if cfg.EmbedModel != "cohere.embed-english-v3" {
		t.Errorf("expected embed_model=cohere.embed-english-v3, got %q", cfg.EmbedModel)
	}
	if cfg.Port != 9999 {
		t.Errorf("expected port=9999, got %d", cfg.Port)
	}
	if cfg.Host != "0.0.0.0" {
		t.Errorf("expected host=0.0.0.0, got %q", cfg.Host)
	}
	if cfg.BearerToken != "test-token" {
		t.Errorf("expected bearer_token=test-token, got %q", cfg.BearerToken)
	}
	if cfg.BaseURL != "https://custom.endpoint.com" {
		t.Errorf("expected base_url=https://custom.endpoint.com, got %q", cfg.BaseURL)
	}
}

func TestLoadConfigEnvOverride(t *testing.T) {
	dir := t.TempDir()
	orig, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(orig)

	// No config file — defaults only
	os.Setenv("AWS_BEARER_TOKEN_BEDROCK", "env-token")
	defer os.Unsetenv("AWS_BEARER_TOKEN_BEDROCK")

	cfg, _, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.BearerToken != "env-token" {
		t.Errorf("expected bearer_token=env-token from env, got %q", cfg.BearerToken)
	}
}

func TestLoadConfigFileOverridesDefaults(t *testing.T) {
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "bedrockify.toml")
	// Only set port — everything else should stay default
	if err := os.WriteFile(cfgFile, []byte(`port = 7777`), 0644); err != nil {
		t.Fatal(err)
	}

	orig, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(orig)

	cfg, _, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.Port != 7777 {
		t.Errorf("expected port=7777, got %d", cfg.Port)
	}
	if cfg.Region != "us-east-1" {
		t.Errorf("expected default region=us-east-1, got %q", cfg.Region)
	}
}

func TestWriteExampleConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bedrockify.toml")
	if err := WriteExampleConfig(path); err != nil {
		t.Fatalf("WriteExampleConfig: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) == 0 {
		t.Error("expected non-empty config file")
	}
}
