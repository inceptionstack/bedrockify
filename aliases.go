package bedrockify

import (
	"log"
	"strings"
)

// bedrockModel holds the base model ID (without region prefix) and whether it
// supports cross-region inference prefixing.
type bedrockModel struct {
	baseID      string // e.g. "anthropic.claude-opus-4-6-v1"
	crossRegion bool   // true = can be prefixed with us./eu./ap.
}

// modelAliases maps common model names, OpenRouter IDs, short names, etc. to
// Bedrock model definitions. Keys are lowercase for case-insensitive lookup.
var modelAliases = map[string]bedrockModel{
	// --- Claude Opus 4.6 ---
	"claude-opus-4-6":                  {baseID: "anthropic.claude-opus-4-6-v1", crossRegion: true},
	"claude-opus-4.6":                  {baseID: "anthropic.claude-opus-4-6-v1", crossRegion: true},
	"claude-opus":                      {baseID: "anthropic.claude-opus-4-6-v1", crossRegion: true},
	"anthropic/claude-opus-4.6":        {baseID: "anthropic.claude-opus-4-6-v1", crossRegion: true},
	"anthropic/claude-opus-4-6":        {baseID: "anthropic.claude-opus-4-6-v1", crossRegion: true},
	"anthropic/claude-opus":            {baseID: "anthropic.claude-opus-4-6-v1", crossRegion: true},
	"anthropic.claude-opus-4-6-v1":     {baseID: "anthropic.claude-opus-4-6-v1", crossRegion: true},

	// --- Claude Sonnet 4.6 ---
	"claude-sonnet-4-6":                {baseID: "anthropic.claude-sonnet-4-6-v1", crossRegion: true},
	"claude-sonnet-4.6":                {baseID: "anthropic.claude-sonnet-4-6-v1", crossRegion: true},
	"claude-sonnet":                    {baseID: "anthropic.claude-sonnet-4-6-v1", crossRegion: true},
	"anthropic/claude-sonnet-4.6":      {baseID: "anthropic.claude-sonnet-4-6-v1", crossRegion: true},
	"anthropic/claude-sonnet-4-6":      {baseID: "anthropic.claude-sonnet-4-6-v1", crossRegion: true},
	"anthropic/claude-sonnet":          {baseID: "anthropic.claude-sonnet-4-6-v1", crossRegion: true},
	"anthropic.claude-sonnet-4-6-v1":   {baseID: "anthropic.claude-sonnet-4-6-v1", crossRegion: true},
	"anthropic.claude-sonnet-4-6":      {baseID: "anthropic.claude-sonnet-4-6-v1", crossRegion: true},

	// --- Claude Sonnet 4 ---
	"claude-sonnet-4":                  {baseID: "anthropic.claude-sonnet-4-20250514-v1:0", crossRegion: true},
	"claude-sonnet-4.0":                {baseID: "anthropic.claude-sonnet-4-20250514-v1:0", crossRegion: true},
	"anthropic/claude-sonnet-4":        {baseID: "anthropic.claude-sonnet-4-20250514-v1:0", crossRegion: true},
	"anthropic/claude-sonnet-4.0":      {baseID: "anthropic.claude-sonnet-4-20250514-v1:0", crossRegion: true},

	// --- Claude Haiku 4.5 ---
	"claude-haiku-4-5":                 {baseID: "anthropic.claude-haiku-4-5-20251001-v1:0", crossRegion: true},
	"claude-haiku-4.5":                 {baseID: "anthropic.claude-haiku-4-5-20251001-v1:0", crossRegion: true},
	"claude-haiku":                     {baseID: "anthropic.claude-haiku-4-5-20251001-v1:0", crossRegion: true},
	"anthropic/claude-haiku-4.5":       {baseID: "anthropic.claude-haiku-4-5-20251001-v1:0", crossRegion: true},
	"anthropic/claude-haiku-4-5":       {baseID: "anthropic.claude-haiku-4-5-20251001-v1:0", crossRegion: true},
	"anthropic/claude-haiku":           {baseID: "anthropic.claude-haiku-4-5-20251001-v1:0", crossRegion: true},

	// --- Claude 3.5 Sonnet (v2) ---
	"claude-3.5-sonnet":                {baseID: "anthropic.claude-3-5-sonnet-20241022-v2:0", crossRegion: true},
	"claude-3-5-sonnet":                {baseID: "anthropic.claude-3-5-sonnet-20241022-v2:0", crossRegion: true},
	"anthropic/claude-3.5-sonnet":      {baseID: "anthropic.claude-3-5-sonnet-20241022-v2:0", crossRegion: true},
	"anthropic/claude-3-5-sonnet":      {baseID: "anthropic.claude-3-5-sonnet-20241022-v2:0", crossRegion: true},

	// --- Claude 3.5 Haiku ---
	"claude-3.5-haiku":                 {baseID: "anthropic.claude-3-5-haiku-20241022-v1:0", crossRegion: true},
	"claude-3-5-haiku":                 {baseID: "anthropic.claude-3-5-haiku-20241022-v1:0", crossRegion: true},
	"anthropic/claude-3.5-haiku":       {baseID: "anthropic.claude-3-5-haiku-20241022-v1:0", crossRegion: true},
	"anthropic/claude-3-5-haiku":       {baseID: "anthropic.claude-3-5-haiku-20241022-v1:0", crossRegion: true},

	// --- Meta Llama 4 Maverick ---
	"llama-4-maverick":                 {baseID: "meta.llama4-maverick-17b-instruct-v1:0", crossRegion: true},
	"llama4-maverick":                  {baseID: "meta.llama4-maverick-17b-instruct-v1:0", crossRegion: true},
	"meta/llama-4-maverick":            {baseID: "meta.llama4-maverick-17b-instruct-v1:0", crossRegion: true},

	// --- Meta Llama 4 Scout ---
	"llama-4-scout":                    {baseID: "meta.llama4-scout-17b-instruct-v1:0", crossRegion: true},
	"llama4-scout":                     {baseID: "meta.llama4-scout-17b-instruct-v1:0", crossRegion: true},
	"meta/llama-4-scout":               {baseID: "meta.llama4-scout-17b-instruct-v1:0", crossRegion: true},

	// --- Mistral Large ---
	"mistral-large":                    {baseID: "mistral.mistral-large-2407-v1:0", crossRegion: true},
	"mistralai/mistral-large":          {baseID: "mistral.mistral-large-2407-v1:0", crossRegion: true},

	// --- Amazon Nova ---
	"nova-pro":                         {baseID: "amazon.nova-pro-v1:0", crossRegion: true},
	"nova-lite":                        {baseID: "amazon.nova-lite-v1:0", crossRegion: true},
	"nova-micro":                       {baseID: "amazon.nova-micro-v1:0", crossRegion: true},
	"amazon/nova-pro":                  {baseID: "amazon.nova-pro-v1:0", crossRegion: true},
	"amazon/nova-lite":                 {baseID: "amazon.nova-lite-v1:0", crossRegion: true},
	"amazon/nova-micro":                {baseID: "amazon.nova-micro-v1:0", crossRegion: true},

	// --- DeepSeek R1 ---
	"deepseek-r1":                      {baseID: "deepseek.deepseek-r1-v1:0", crossRegion: true},
	"deepseek/deepseek-r1":             {baseID: "deepseek.deepseek-r1-v1:0", crossRegion: true},
}

// embeddingAliases maps common embedding model names to Bedrock model IDs.
// Embedding models don't use cross-region inference prefixes.
var embeddingAliases = map[string]string{
	// Titan Embed v2
	"text-embedding-ada-002":              "amazon.titan-embed-text-v2:0",
	"titan-embed-v2":                      "amazon.titan-embed-text-v2:0",
	"titan-embed-text-v2":                 "amazon.titan-embed-text-v2:0",
	"amazon/titan-embed-text-v2":          "amazon.titan-embed-text-v2:0",

	// Titan Embed v1
	"titan-embed-v1":                      "amazon.titan-embed-g1-text-02",
	"titan-embed-text-v1":                 "amazon.titan-embed-g1-text-02",
	"amazon/titan-embed-g1":               "amazon.titan-embed-g1-text-02",

	// Cohere Embed v4
	"cohere-embed-v4":                     "cohere.embed-v4:0",
	"cohere/embed-v4":                     "cohere.embed-v4:0",
	"embed-v4":                            "cohere.embed-v4:0",

	// Cohere Embed English v3
	"cohere-embed-english-v3":             "cohere.embed-english-v3",
	"cohere/embed-english-v3":             "cohere.embed-english-v3",

	// Cohere Embed Multilingual v3
	"cohere-embed-multilingual-v3":        "cohere.embed-multilingual-v3",
	"cohere/embed-multilingual-v3":        "cohere.embed-multilingual-v3",

	// Titan Multimodal
	"titan-embed-image-v1":                "amazon.titan-embed-image-v1:0",
	"amazon/titan-embed-image-v1":         "amazon.titan-embed-image-v1:0",
}

// regionPrefix maps AWS region prefixes (first segment) to cross-region inference prefixes.
var regionPrefixMap = map[string]string{
	"us":  "us",
	"eu":  "eu",
	"ap":  "ap",
}

// regionToCrossRegionPrefix derives the cross-region prefix from a full region name.
// e.g. "us-east-1" → "us", "eu-west-1" → "eu", "ap-northeast-1" → "ap"
func regionToCrossRegionPrefix(region string) string {
	parts := strings.SplitN(region, "-", 2)
	if len(parts) >= 1 {
		if prefix, ok := regionPrefixMap[parts[0]]; ok {
			return prefix
		}
	}
	return "us" // default to us
}

// ResolveModelAlias resolves a model name to a Bedrock model ID with the
// appropriate cross-region inference prefix based on the configured region.
// If the model is already a valid Bedrock ID (contains a dot like "us.anthropic.xxx"
// or "anthropic.xxx"), it is returned as-is.
// Returns the resolved model ID and whether an alias was matched.
func ResolveModelAlias(model, region string) (string, bool) {
	if model == "" {
		return model, false
	}

	// If it already has a cross-region prefix (e.g. "us.anthropic.claude-..."), pass through.
	if hasCrossRegionPrefix(model) {
		return model, false
	}

	// If it's a raw Bedrock model ID without cross-region prefix (e.g. "anthropic.claude-opus-4-6-v1"),
	// add the region prefix.
	if looksLikeBedrockID(model) {
		prefix := regionToCrossRegionPrefix(region)
		resolved := prefix + "." + model
		log.Printf("model alias: %q → %q (added region prefix)", model, resolved)
		return resolved, true
	}

	// Look up in alias table (case-insensitive)
	key := strings.ToLower(model)
	if bm, ok := modelAliases[key]; ok {
		resolved := bm.baseID
		if bm.crossRegion {
			prefix := regionToCrossRegionPrefix(region)
			resolved = prefix + "." + resolved
		}
		log.Printf("model alias: %q → %q", model, resolved)
		return resolved, true
	}

	// No match — pass through as-is (let Bedrock reject if invalid)
	return model, false
}

// ResolveEmbeddingAlias resolves common embedding model names to Bedrock model IDs.
// Embedding models don't use cross-region inference prefixes.
// Returns the resolved model ID and whether an alias was matched.
func ResolveEmbeddingAlias(model string) (string, bool) {
	if model == "" {
		return model, false
	}

	// Already a Bedrock embedding ID (has a dot)
	if strings.Contains(model, ".") {
		return model, false
	}

	key := strings.ToLower(model)
	if resolved, ok := embeddingAliases[key]; ok {
		log.Printf("embedding alias: %q → %q", model, resolved)
		return resolved, true
	}

	return model, false
}

// hasCrossRegionPrefix checks if model starts with "us.", "eu.", "ap." etc.
func hasCrossRegionPrefix(model string) bool {
	prefixes := []string{"us.", "eu.", "ap."}
	lower := strings.ToLower(model)
	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			return true
		}
	}
	return false
}

// looksLikeBedrockID checks if model looks like a native Bedrock ID (provider.model-name).
// e.g. "anthropic.claude-opus-4-6-v1" or "meta.llama4-scout-17b-instruct-v1:0"
func looksLikeBedrockID(model string) bool {
	// Must contain a dot, first segment must be a known provider
	dotIdx := strings.Index(model, ".")
	if dotIdx <= 0 {
		return false
	}
	provider := strings.ToLower(model[:dotIdx])
	knownProviders := []string{"anthropic", "meta", "mistral", "amazon", "cohere", "ai21", "stability", "deepseek"}
	for _, p := range knownProviders {
		if provider == p {
			return true
		}
	}
	return false
}
