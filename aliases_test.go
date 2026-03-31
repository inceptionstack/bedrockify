package bedrockify

import "testing"

func TestResolveModelAlias(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		region   string
		want     string
		wantHit  bool
	}{
		// Already valid cross-region IDs — pass through
		{"cross-region opus", "us.anthropic.claude-opus-4-6-v1", "us-east-1", "us.anthropic.claude-opus-4-6-v1", false},
		{"cross-region eu", "eu.anthropic.claude-sonnet-4-6-v1", "eu-west-1", "eu.anthropic.claude-sonnet-4-6-v1", false},

		// Raw Bedrock IDs — add region prefix
		{"bare bedrock id us", "anthropic.claude-opus-4-6-v1", "us-east-1", "us.anthropic.claude-opus-4-6-v1", true},
		{"bare bedrock id eu", "anthropic.claude-opus-4-6-v1", "eu-west-1", "eu.anthropic.claude-opus-4-6-v1", true},
		{"bare bedrock id ap", "anthropic.claude-opus-4-6-v1", "ap-northeast-1", "ap.anthropic.claude-opus-4-6-v1", true},

		// OpenRouter style aliases
		{"openrouter opus", "anthropic/claude-opus-4.6", "us-east-1", "us.anthropic.claude-opus-4-6-v1", true},
		{"openrouter sonnet", "anthropic/claude-sonnet-4.6", "us-east-1", "us.anthropic.claude-sonnet-4-6-v1", true},
		{"openrouter haiku", "anthropic/claude-haiku-4.5", "us-east-1", "us.anthropic.claude-haiku-4-5-20251001-v1:0", true},
		{"openrouter opus eu", "anthropic/claude-opus-4.6", "eu-west-1", "eu.anthropic.claude-opus-4-6-v1", true},

		// Short name aliases
		{"short opus", "claude-opus", "us-east-1", "us.anthropic.claude-opus-4-6-v1", true},
		{"short sonnet", "claude-sonnet", "us-east-1", "us.anthropic.claude-sonnet-4-6-v1", true},
		{"short haiku", "claude-haiku", "us-east-1", "us.anthropic.claude-haiku-4-5-20251001-v1:0", true},

		// Dot-version aliases
		{"dot version opus", "claude-opus-4.6", "us-east-1", "us.anthropic.claude-opus-4-6-v1", true},
		{"dot version sonnet", "claude-sonnet-4.6", "ap-southeast-1", "ap.anthropic.claude-sonnet-4-6-v1", true},

		// Non-Anthropic models
		{"nova pro", "nova-pro", "us-east-1", "us.amazon.nova-pro-v1:0", true},
		{"deepseek", "deepseek-r1", "us-east-1", "us.deepseek.deepseek-r1-v1:0", true},
		{"mistral", "mistral-large", "eu-west-1", "eu.mistral.mistral-large-2407-v1:0", true},

		// Case insensitivity
		{"case insensitive", "Claude-Opus-4.6", "us-east-1", "us.anthropic.claude-opus-4-6-v1", true},
		{"case insensitive openrouter", "Anthropic/Claude-Opus-4.6", "us-east-1", "us.anthropic.claude-opus-4-6-v1", true},

		// Unknown model — pass through
		{"unknown model", "some-random-model", "us-east-1", "some-random-model", false},
		{"empty model", "", "us-east-1", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, gotHit := ResolveModelAlias(tt.model, tt.region)
			if got != tt.want {
				t.Errorf("ResolveModelAlias(%q, %q) = %q, want %q", tt.model, tt.region, got, tt.want)
			}
			if gotHit != tt.wantHit {
				t.Errorf("ResolveModelAlias(%q, %q) hit = %v, want %v", tt.model, tt.region, gotHit, tt.wantHit)
			}
		})
	}
}

func TestRegionToCrossRegionPrefix(t *testing.T) {
	tests := []struct {
		region string
		want   string
	}{
		{"us-east-1", "us"},
		{"us-west-2", "us"},
		{"eu-west-1", "eu"},
		{"eu-central-1", "eu"},
		{"ap-northeast-1", "ap"},
		{"ap-southeast-1", "ap"},
		{"unknown-region-1", "us"}, // default fallback
	}

	for _, tt := range tests {
		t.Run(tt.region, func(t *testing.T) {
			got := regionToCrossRegionPrefix(tt.region)
			if got != tt.want {
				t.Errorf("regionToCrossRegionPrefix(%q) = %q, want %q", tt.region, got, tt.want)
			}
		})
	}
}
