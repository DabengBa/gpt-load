package parameteroverride

import (
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"testing"
)

func TestGeminiEmbeddingsSupportsConfiguredParameters(t *testing.T) {
	if !supports(protocol.GeminiEmbeddings, execution.OperationEmbeddingsCreate) {
		t.Fatal("Gemini embeddings overrides disabled")
	}
	if supports(protocol.GeminiEmbeddings, execution.OperationChatCompletion) {
		t.Fatal("generation must not be enabled")
	}
}
