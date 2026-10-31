package bifrost

import (
	"encoding/json"
	"testing"
)

func TestNormalizeAnthropicNativeReasoningUpdatesEnabledBudget(t *testing.T) {
	body := []byte(`{"model":"claude-sonnet-4-6","max_tokens":4096,"thinking":{"type":"enabled","budget_tokens":4096,"custom":"keep"},"output_config":{"effort":"low"}}`)

	got, err := normalizeAnthropicNativeReasoning(body, "low")
	if err != nil {
		t.Fatalf("normalizeAnthropicNativeReasoning() error = %v", err)
	}
	want := `{"model":"claude-sonnet-4-6","max_tokens":4096,"thinking":{"type":"enabled","budget_tokens":1484,"custom":"keep"},"output_config":{"effort":"low"}}`
	var gotJSON, wantJSON any
	if err := json.Unmarshal(got, &gotJSON); err != nil {
		t.Fatalf("decode normalized body: %v", err)
	}
	if err := json.Unmarshal([]byte(want), &wantJSON); err != nil {
		t.Fatalf("decode expected body: %v", err)
	}
	if !anthropicJSONEqual(gotJSON, wantJSON) {
		t.Fatalf("normalizeAnthropicNativeReasoning() = %s, want %s", got, want)
	}
}

func anthropicJSONEqual(a, b any) bool {
	left, _ := json.Marshal(a)
	right, _ := json.Marshal(b)
	return string(left) == string(right)
}

func TestNormalizeAnthropicNativeReasoningLeavesUnfitRequestsForUpstream(t *testing.T) {
	tests := []string{
		`{"max_tokens":1024,"thinking":{"type":"enabled","budget_tokens":2048}}`,
		`{"thinking":{"type":"enabled"}}`,
		`{"max_tokens":"4096","thinking":{"type":"enabled"}}`,
		`{"max_tokens":4096,"thinking":{"type":"adaptive"}}`,
	}
	for _, body := range tests {
		got, err := normalizeAnthropicNativeReasoning([]byte(body), "low")
		if err != nil {
			t.Fatalf("normalizeAnthropicNativeReasoning(%s) error = %v", body, err)
		}
		if string(got) != body {
			t.Fatalf("normalizeAnthropicNativeReasoning(%s) = %s, want unchanged", body, got)
		}
	}
}
