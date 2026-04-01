package bedrockify

import (
	"context"
	"encoding/binary"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// BedrockEmbedder calls Amazon Bedrock to generate embeddings.
// Supports Titan, Cohere, and Nova Multimodal model families,
// auto-detected by model ID prefix.
type BedrockEmbedder struct {
	client     *bedrockruntime.Client
	modelID    string
	dimensions int // 0 means use model default
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

// NewBedrockEmbedderWithDimensions creates an embedder with custom output dimensions.
func NewBedrockEmbedderWithDimensions(region, modelID string, dimensions int) (*BedrockEmbedder, error) {
	e, err := NewBedrockEmbedder(region, modelID)
	if err != nil {
		return nil, err
	}
	e.dimensions = dimensions
	return e, nil
}

// Embed generates an embedding vector for the given text.
// Routes to the correct Bedrock model format based on model ID.
func (b *BedrockEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	if isNovaMultimodal(b.modelID) {
		return b.embedNova(ctx, text)
	}
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

// isNovaMultimodal returns true for Nova Multimodal Embeddings v2.
func isNovaMultimodal(modelID string) bool {
	return strings.Contains(modelID, "nova-2-multimodal-embeddings")
}

// validNovaDimensions are the allowed dimension counts for Nova Multimodal.
var validNovaDimensions = map[int]bool{
	256:  true,
	384:  true,
	1024: true,
	3072: true,
}

// validateNovaDimensions checks that dimensions is valid for Nova Multimodal.
// dimensions=0 means use model default (1024), which is always valid.
func (b *BedrockEmbedder) validateNovaDimensions() error {
	if b.dimensions == 0 {
		return nil // default
	}
	if !validNovaDimensions[b.dimensions] {
		return fmt.Errorf("invalid dimensions %d for Nova Multimodal; valid: 256, 384, 1024, 3072", b.dimensions)
	}
	return nil
}

// --- Nova Multimodal format ---

type novaEmbedRequestFull struct {
	InputText       string                 `json:"inputText"`
	EmbeddingConfig map[string]interface{} `json:"embeddingConfig,omitempty"`
}

type novaEmbedResponse struct {
	Embedding []float64 `json:"embedding"`
}

func (b *BedrockEmbedder) embedNova(ctx context.Context, text string) ([]float64, error) {
	if err := b.validateNovaDimensions(); err != nil {
		return nil, err
	}

	req := novaEmbedRequestFull{
		InputText: text,
	}
	if b.dimensions > 0 {
		req.EmbeddingConfig = map[string]interface{}{
			"outputEmbeddingLength": b.dimensions,
		}
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	resp, err := b.invokeModel(ctx, body)
	if err != nil {
		return nil, err
	}

	var result novaEmbedResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("nova embed parse error: %w", err)
	}
	return result.Embedding, nil
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

// --- Encoding helpers ---

// encodeEmbeddingBase64 encodes a float64 slice as little-endian float32 bytes in base64.
// This matches the OpenAI base64 encoding format.
func encodeEmbeddingBase64(floats []float64) string {
	buf := make([]byte, len(floats)*4)
	for i, f := range floats {
		bits := math.Float32bits(float32(f))
		binary.LittleEndian.PutUint32(buf[i*4:], bits)
	}
	return base64.StdEncoding.EncodeToString(buf)
}

// formatEmbedding converts a float64 slice to the requested encoding format.
// Returns []float64 for default/"float" format, string for "base64" format.
func formatEmbedding(floats []float64, encodingFormat string) interface{} {
	if encodingFormat == "base64" {
		return encodeEmbeddingBase64(floats)
	}
	return floats
}
