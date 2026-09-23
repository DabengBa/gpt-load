package dialect

import (
	"encoding/json"
	"testing"

	"gpt-load/internal/protocol"
)

func TestSetReasoningEffortReplacesOrInjectsProtocolField(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		client protocol.Protocol
		body   string
		path   []string
	}{
		{name: "OpenAI Chat replaces", client: protocol.OpenAICompletions, body: `{"reasoning_effort":"low","keep":9007199254740993}`, path: []string{"reasoning_effort"}},
		{name: "OpenAI Responses creates parent", client: protocol.OpenAIResponses, body: `{}`, path: []string{"reasoning", "effort"}},
		{name: "Anthropic creates parent", client: protocol.Anthropic, body: `{}`, path: []string{"output_config", "effort"}},
		{name: "Gemini creates parents", client: protocol.Gemini, body: `{}`, path: []string{"generationConfig", "thinkingConfig", "thinkingLevel"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, changed, err := SetReasoningEffort([]byte(test.body), "none", test.client)
			if err != nil || !changed {
				t.Fatalf("SetReasoningEffort() = %s, %v, %v", got, changed, err)
			}
			var current any
			if err := json.Unmarshal(got, &current); err != nil {
				t.Fatal(err)
			}
			for _, part := range test.path {
				current = current.(map[string]any)[part]
			}
			if current != "none" {
				t.Fatalf("reasoning effort = %#v, want none", current)
			}
			if test.client == protocol.OpenAICompletions && string(got) != `{"reasoning_effort":"none","keep":9007199254740993}` {
				t.Fatalf("unrelated JSON changed: %s", got)
			}
		})
	}
}

func TestSetReasoningEffortLeavesBodyUnchangedWhenUnconfigured(t *testing.T) {
	t.Parallel()
	input := []byte(`{"model":"m","reasoning_effort":"low"}`)
	got, changed, err := SetReasoningEffort(input, "", protocol.OpenAICompletions)
	if err != nil || changed || string(got) != string(input) {
		t.Fatalf("SetReasoningEffort() = %s, %v, %v; want unchanged", got, changed, err)
	}
}
