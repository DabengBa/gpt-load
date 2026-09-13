package dialect

import (
	"encoding/json"
	"strings"
	"testing"

	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

func TestSupportsReasoningEffortOverride(t *testing.T) {
	tests := []struct {
		name      string
		client    protocol.Protocol
		operation execution.Operation
		want      bool
	}{
		{name: "openai chat", client: protocol.OpenAICompletions, operation: execution.OperationChatCompletion, want: true},
		{name: "responses create", client: protocol.OpenAIResponses, operation: execution.OperationResponsesCreate, want: true},
		{name: "anthropic chat", client: protocol.Anthropic, operation: execution.OperationChatCompletion, want: true},
		{name: "gemini chat", client: protocol.Gemini, operation: execution.OperationChatCompletion, want: true},
		{name: "responses input tokens", client: protocol.OpenAIResponses, operation: execution.OperationResponsesInputTokens},
		{name: "anthropic count tokens", client: protocol.Anthropic, operation: execution.OperationCountTokens},
		{name: "gemini count tokens", client: protocol.Gemini, operation: execution.OperationCountTokens},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := SupportsReasoningEffortOverride(test.client, test.operation); got != test.want {
				t.Fatalf("SupportsReasoningEffortOverride() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestOverrideReasoningEffort(t *testing.T) {
	tests := []struct {
		name     string
		client   protocol.Protocol
		body     string
		wantPath []string
		present  bool
		changed  bool
	}{
		{"chat", protocol.OpenAICompletions, `{"reasoning_effort":"max","keep":true}`, []string{"reasoning_effort"}, true, true},
		{"responses", protocol.OpenAIResponses, `{"reasoning":{"effort":"max","summary":"auto"}}`, []string{"reasoning", "effort"}, true, true},
		{"anthropic", protocol.Anthropic, `{"output_config":{"effort":"max"}}`, []string{"output_config", "effort"}, true, true},
		{"gemini", protocol.Gemini, `{"generationConfig":{"thinkingConfig":{"thinkingLevel":"HIGH"}}}`, []string{"generationConfig", "thinkingConfig", "thinkingLevel"}, true, true},
		{"missing effort is not injected", protocol.OpenAIResponses, `{"reasoning":{"summary":"auto"}}`, nil, false, false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := HasReasoningEffort([]byte(test.body), test.client); got != test.present {
				t.Fatalf("HasReasoningEffort() = %v, want %v", got, test.present)
			}
			body, changed, err := OverrideReasoningEffort([]byte(test.body), "high", test.client)
			if err != nil || changed != test.changed {
				t.Fatalf("OverrideReasoningEffort() = %s, %v, %v", body, changed, err)
			}
			if !changed {
				return
			}
			var value map[string]any
			if err := json.Unmarshal(body, &value); err != nil {
				t.Fatal(err)
			}
			var current any = value
			for _, key := range test.wantPath {
				current = current.(map[string]any)[key]
			}
			if current != "high" {
				t.Fatalf("effort = %#v", current)
			}
		})
	}
}

func TestOverrideReasoningEffortPreservesUnrelatedIntegerLiterals(t *testing.T) {
	body, changed, err := OverrideReasoningEffort(
		[]byte(`{"reasoning_effort":"low","seed":9007199254740993,"metadata":{"sequence":9007199254740993}}`),
		"high",
		protocol.OpenAICompletions,
	)
	if err != nil || !changed {
		t.Fatalf("OverrideReasoningEffort() = %s, %v, %v", body, changed, err)
	}
	if strings.Count(string(body), "9007199254740993") != 2 {
		t.Fatalf("unrelated integer literals changed: %s", body)
	}
}

func TestOverrideReasoningEffortPreservesUnrelatedDuplicateFields(t *testing.T) {
	input := []byte(`{"reasoning":{"effort":"low","metadata":{"x":1},"metadata":{"x":2}},"metadata":{"x":1},"metadata":{"x":2}}`)
	want := `{"reasoning":{"effort":"high","metadata":{"x":1},"metadata":{"x":2}},"metadata":{"x":1},"metadata":{"x":2}}`

	body, changed, err := OverrideReasoningEffort(input, "high", protocol.OpenAIResponses)
	if err != nil || !changed {
		t.Fatalf("OverrideReasoningEffort() = %s, %v, %v", body, changed, err)
	}
	if got := string(body); got != want {
		t.Fatalf("body = %s, want %s", got, want)
	}
}
