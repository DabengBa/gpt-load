package channel

import (
	"crypto/sha256"
	"fmt"
	"strings"
	"testing"

	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

// TestBuiltInRouteGolden freezes the explicit built-in route contract. It is
// based on the pre-module behavior with the intentionally unreachable
// OpenAI Responses model-list routes removed.
func TestBuiltInRouteGolden(t *testing.T) {
	t.Parallel()

	registry := NewRegistry()
	got := make([]string, 0)
	for _, descriptor := range registry.List() {
		// New presets have explicit capability assertions in TestU006PresetCapabilities;
		// retain this historical snapshot for the original channels.
		switch descriptor.ID {
		case Cerebras, Mistral, Nebius, Parasail, Wafer, HuggingFace, Cohere, OpenCodeGo, OpenCodeZen:
			continue
		}
		definition, ok := registry.lookup(descriptor.ID)
		if !ok {
			t.Fatalf("lookup(%q) missing", descriptor.ID)
		}
		for _, clientProtocol := range protocol.DataPlaneProtocols() {
			// New embedding routes have direct capability and HTTP behavior proofs.
			if clientProtocol == protocol.GeminiEmbeddings || descriptor.ID == Gemini && clientProtocol == protocol.OpenAIEmbeddings {
				continue
			}
			for _, operation := range allGoldenOperations() {
				mode, exists := definition.modes[clientProtocol][operation]
				if !exists {
					continue
				}
				got = append(got, fmt.Sprintf(
					"%s|%s|%s|%s",
					descriptor.ID,
					clientProtocol,
					operation,
					mode,
				))
			}
		}
	}

	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(strings.Join(got, "\n"))))
	const wantDigest = "9e3e6d2882f5d7902525b6ef1782364d3b1a2e79aef14c3711e53a171b80c5d2"
	if digest != wantDigest {
		t.Fatalf("built-in routes changed: digest = %s, want %s\n%s", digest, wantDigest, strings.Join(got, "\n"))
	}
}

func allGoldenOperations() []execution.Operation {
	return []execution.Operation{
		execution.OperationChatCompletion,
		execution.OperationResponsesCreate,
		execution.OperationResponsesRetrieve,
		execution.OperationResponsesDelete,
		execution.OperationResponsesCancel,
		execution.OperationResponsesInputItems,
		execution.OperationResponsesCompact,
		execution.OperationResponsesInputTokens,
		execution.OperationCountTokens,
		execution.OperationResponsesPassthrough,
		execution.OperationImagesGenerate,
		execution.OperationImagesEdit,
		execution.OperationEmbeddingsCreate,
		execution.OperationRerank,
		execution.OperationListModels,
		execution.OperationProbe,
	}
}
