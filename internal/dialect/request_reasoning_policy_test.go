package dialect

import (
	"encoding/json"
	"errors"
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

func TestSetReasoningEffortRejectsNoneForGemini(t *testing.T) {
	_, _, err := SetReasoningEffort([]byte(`{"generationConfig":{"thinkingConfig":{"thinkingLevel":"high"}}}`), "none", protocol.Gemini)
	if !errors.Is(err, ErrReasoningPolicyConflict) {
		t.Fatalf("error = %v, want ErrReasoningPolicyConflict", err)
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

func TestSetReasoningEffortReconcilesExistingChatFields(t *testing.T) {
	for _, tc := range []struct{ effort, input, want string }{
		{"none", `{"enable_thinking":true,"chat_template_kwargs":{"enable_thinking":true,"keep":1},"reasoning":{"effort":"high","summary":"auto"}}`, `{"enable_thinking":false,"chat_template_kwargs":{"enable_thinking":false,"keep":1},"reasoning":{"effort":"none","summary":"auto"},"reasoning_effort":"none"}`},
		{"low", `{"enable_thinking":false,"chat_template_kwargs":{"enable_thinking":false},"reasoning":{"effort":"none"}}`, `{"enable_thinking":true,"chat_template_kwargs":{"enable_thinking":true},"reasoning":{"effort":"low"},"reasoning_effort":"low"}`},
		{"low", `{"enable_thinking":"custom","chat_template_kwargs":{"keep":1}}`, `{"enable_thinking":"custom","chat_template_kwargs":{"keep":1},"reasoning_effort":"low"}`},
	} {
		t.Run(tc.effort+tc.input, func(t *testing.T) {
			got, _, err := SetReasoningEffort([]byte(tc.input), tc.effort, protocol.OpenAICompletions)
			if err != nil || string(got) != tc.want {
				t.Fatalf("got %s, %v; want %s", got, err, tc.want)
			}
		})
	}
}

func TestSetReasoningEffortRejectsNoneWithAnthropicEnabledThinking(t *testing.T) {
	input := []byte(`{"thinking":{"type":"enabled","budget_tokens":2048},"max_tokens":4096}`)
	_, _, err := SetReasoningEffort(input, "none", protocol.Anthropic)
	if err == nil {
		t.Fatal("expected conflict instead of sending enabled thinking with effort none")
	}
}
