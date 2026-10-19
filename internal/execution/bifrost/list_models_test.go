package bifrost

import (
	"reflect"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

func TestSanitizeListModelsResponsePreservesOpenRouterEmbeddingModels(t *testing.T) {
	t.Parallel()

	response := &schemas.BifrostListModelsResponse{Data: []schemas.Model{
		{ID: "openrouter/text", Architecture: &schemas.Architecture{OutputModalities: []string{"text"}}},
		{ID: "openrouter/embedding-singular", Architecture: &schemas.Architecture{OutputModalities: []string{" Embedding "}}},
		{ID: "openrouter/embedding-plural", Architecture: &schemas.Architecture{OutputModalities: []string{"EMBEDDINGS"}}},
		{ID: "openrouter/mixed", Architecture: &schemas.Architecture{OutputModalities: []string{"text", "embeddings"}}},
	}}

	got := sanitizeListModelsResponse(schemas.OpenRouter, response)
	gotIDs := make([]string, 0, len(got.Data))
	for _, model := range got.Data {
		gotIDs = append(gotIDs, model.ID)
	}
	if want := []string{"text", "embedding-singular", "embedding-plural", "mixed"}; !reflect.DeepEqual(gotIDs, want) {
		t.Fatalf("sanitized OpenRouter models = %v, want %v", gotIDs, want)
	}
	originalIDs := make([]string, 0, len(response.Data))
	for _, model := range response.Data {
		originalIDs = append(originalIDs, model.ID)
	}
	if want := []string{
		"openrouter/text",
		"openrouter/embedding-singular",
		"openrouter/embedding-plural",
		"openrouter/mixed",
	}; !reflect.DeepEqual(originalIDs, want) {
		t.Fatalf("sanitize mutated source response: %#v", response.Data)
	}

	other := sanitizeListModelsResponse(schemas.OpenAI, &schemas.BifrostListModelsResponse{Data: []schemas.Model{{
		ID: "openai/embedding-only", Architecture: &schemas.Architecture{OutputModalities: []string{"embeddings"}},
	}}})
	if len(other.Data) != 1 || other.Data[0].ID != "embedding-only" {
		t.Fatalf("non-OpenRouter embedding model was filtered: %#v", other.Data)
	}
}

// TestSanitizeListModelsResponseDropsOpenRouterDecisionModels pins the GPT-Load
// contract for OpenRouter's model list: it publishes only models whose declared
// output modalities GPT-Load can actually serve (chat and embeddings), not the
// decision models Bifrost v1.13.0 started merging in from
// /v1/models?output_modalities=decisions, which have no GPT-Load route.
//
// The decision catalog is excluded whether or not its items carry an
// architecture: a model whose modalities are only "decisions" declares
// capabilities GPT-Load does not proxy, and a model that declares no modality at
// all cannot be shown to be chat or embed. A mixed entry such as
// ["text","decisions"] is kept, because its text output is supported.
func TestSanitizeListModelsResponseDropsOpenRouterDecisionModels(t *testing.T) {
	t.Parallel()

	response := &schemas.BifrostListModelsResponse{Data: []schemas.Model{
		{ID: "openrouter/openai/gpt-4o", Architecture: &schemas.Architecture{OutputModalities: []string{"text"}}},
		{ID: "openrouter/perplexity/pplx-decider", Architecture: &schemas.Architecture{OutputModalities: []string{"decisions"}}},
		{ID: "openrouter/microsoft/microsoft-decision-1"},
		{ID: "openrouter/acme/hybrid-decider", Architecture: &schemas.Architecture{OutputModalities: []string{"DECISIONS", "text"}}},
		{ID: "openrouter/acme/decision-embed", Architecture: &schemas.Architecture{OutputModalities: []string{"decisions", "embeddings"}}},
		{ID: "openrouter/acme/empty-modalities", Architecture: &schemas.Architecture{OutputModalities: []string{" "}}},
	}}

	got := sanitizeListModelsResponse(schemas.OpenRouter, response)
	gotIDs := make([]string, 0, len(got.Data))
	for _, model := range got.Data {
		gotIDs = append(gotIDs, model.ID)
	}
	if want := []string{"openai/gpt-4o", "acme/hybrid-decider", "acme/decision-embed"}; !reflect.DeepEqual(gotIDs, want) {
		t.Fatalf("sanitized OpenRouter models = %v, want %v", gotIDs, want)
	}
	if len(response.Data) != 6 || response.Data[0].ID != "openrouter/openai/gpt-4o" {
		t.Fatalf("sanitize mutated the provider response: %v", response.Data)
	}
}
