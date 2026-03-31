package bedrockify

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// BedrockEmbedder calls Amazon Bedrock to generate embeddings.
// Supports Titan and Cohere model families, auto-detected by model ID prefix.
type BedrockEmbedder struct {
	client  *bedrockruntime.Client
	modelID string
}

// NewBedrockEmbedder creates an embedder backed by Bedrock.
func NewBedrockEmbedder(region, modelID string) (*BedrockEmbedder, error) {
	cfg, err := config.LoadDefaultConfig(context.Background(),
		config.WithRegion(region),
	)
	if err != nil {
		return nil, err
	}

	return &BedrockEmbedder{
		client:  bedrockruntime.NewFromConfig(cfg),
		modelID: modelID,
	}, nil
}

// Embed generates an embedding vector for the given text.
// Routes to the correct Bedrock model format based on model ID.
func (b *BedrockEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	if isCohere(b.modelID) {
		return b.embedCohere(ctx, text)
	}
	return b.embedTitan(ctx, text)
}

// --- Model detection ---

// isCohere returns true for any Cohere model ID.
func isCohere(modelID string) bool {
	return strings.HasPrefix(modelID, "cohere.")
}

// isCohereV4 returns true for Cohere Embed v4 (dict response format).
func isCohereV4(modelID string) bool {
	return strings.HasPrefix(modelID, "cohere.embed-v4")
}

// --- Titan format ---

type titanRequest struct {
	InputText string `json:"inputText"`
}

type titanResponse struct {
	Embedding []float64 `json:"embedding"`
}

func (b *BedrockEmbedder) embedTitan(ctx context.Context, text string) ([]float64, error) {
	body, err := json.Marshal(titanRequest{InputText: text})
	if err != nil {
		return nil, err
	}

	resp, err := b.invokeModel(ctx, body)
	if err != nil {
		return nil, err
	}

	var result titanResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, err
	}
	return result.Embedding, nil
}

// --- Cohere format ---

type cohereRequest struct {
	Texts     []string `json:"texts"`
	InputType string   `json:"input_type"`
}

// cohereV3Response: {"embeddings": [[float, ...]]}
type cohereV3Response struct {
	Embeddings [][]float64 `json:"embeddings"`
}

// cohereV4Response: {"embeddings": {"float": [[float, ...]]}}
type cohereV4Response struct {
	Embeddings struct {
		Float [][]float64 `json:"float"`
	} `json:"embeddings"`
}

func (b *BedrockEmbedder) embedCohere(ctx context.Context, text string) ([]float64, error) {
	body, err := json.Marshal(cohereRequest{
		Texts:     []string{text},
		InputType: "search_query",
	})
	if err != nil {
		return nil, err
	}

	resp, err := b.invokeModel(ctx, body)
	if err != nil {
		return nil, err
	}

	if isCohereV4(b.modelID) {
		var result cohereV4Response
		if err := json.Unmarshal(resp, &result); err != nil {
			return nil, fmt.Errorf("cohere v4 parse error: %w", err)
		}
		if len(result.Embeddings.Float) == 0 {
			return nil, &EmbedError{Message: "cohere v4 returned no embeddings"}
		}
		return result.Embeddings.Float[0], nil
	}

	var result cohereV3Response
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("cohere v3 parse error: %w", err)
	}
	if len(result.Embeddings) == 0 {
		return nil, &EmbedError{Message: "cohere v3 returned no embeddings"}
	}
	return result.Embeddings[0], nil
}

// --- Shared Bedrock call ---

func (b *BedrockEmbedder) invokeModel(ctx context.Context, body []byte) ([]byte, error) {
	contentType := "application/json"
	accept := "application/json"
	resp, err := b.client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     &b.modelID,
		ContentType: &contentType,
		Accept:      &accept,
		Body:        body,
	})
	if err != nil {
		return nil, err
	}
	return resp.Body, nil
}
