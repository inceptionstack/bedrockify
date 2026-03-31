package bedrockify

import (
	"testing"
)

func TestIsCohere(t *testing.T) {
	tests := []struct {
		modelID string
		want    bool
	}{
		{"cohere.embed-english-v3", true},
		{"cohere.embed-multilingual-v3", true},
		{"cohere.embed-v4:0", true},
		{"amazon.titan-embed-text-v2:0", false},
		{"amazon.titan-embed-g1-text-02", false},
		{"", false},
	}
	for _, tt := range tests {
		if got := isCohere(tt.modelID); got != tt.want {
			t.Errorf("isCohere(%q) = %v, want %v", tt.modelID, got, tt.want)
		}
	}
}

func TestIsCohereV4(t *testing.T) {
	tests := []struct {
		modelID string
		want    bool
	}{
		{"cohere.embed-v4:0", true},
		{"cohere.embed-v4", true},
		{"cohere.embed-english-v3", false},
		{"cohere.embed-multilingual-v3", false},
		{"amazon.titan-embed-text-v2:0", false},
	}
	for _, tt := range tests {
		if got := isCohereV4(tt.modelID); got != tt.want {
			t.Errorf("isCohereV4(%q) = %v, want %v", tt.modelID, got, tt.want)
		}
	}
}
