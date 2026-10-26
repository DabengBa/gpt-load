package channel

import (
	"encoding/json"
	"testing"
)

func TestChannelDescriptorOmitsSubscriptionNotices(t *testing.T) {
	t.Parallel()
	descriptor, ok := NewRegistry().Get(OpenAI)
	if !ok {
		t.Fatal("OpenAI channel missing")
	}
	encoded, err := json.Marshal(descriptor)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &fields); err != nil {
		t.Fatal(err)
	}
	if _, exists := fields["notices"]; exists {
		t.Fatalf("subscription notice DTO remains: %s", encoded)
	}
}

func TestRemovedSubscriptionChannels(t *testing.T) {
	t.Parallel()
	r := NewRegistry()
	for _, id := range []ID{"codex", "claude", "antigravity", "grok"} {
		if _, ok := r.Get(id); ok {
			t.Errorf("removed subscription channel %q remains registered", id)
		}
	}
	for _, id := range []ID{"openai", "anthropic", "gemini", "xai", "azure_openai", "aws_bedrock", "google_vertex", "cliproxyapi"} {
		d, ok := r.Get(id)
		if !ok || d.Connection.CredentialInput != "batch_text" {
			t.Errorf("ordinary channel %q missing or without a batch credential input: %#v", id, d)
		}
	}
}
