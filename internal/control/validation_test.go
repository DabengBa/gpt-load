package control

import (
	"encoding/json"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/state"
)

func probeTestGroupView(t *testing.T, channelID channel.ID, connectionType string, models ...string) state.GroupView {
	t.Helper()
	registry := channel.NewRegistry()
	params := json.RawMessage(`{}`)
	if channelID == channel.OpenAICompatible {
		params = json.RawMessage(`{"base_url":"https://probe.example/v1"}`)
	}
	resolved, err := registry.Resolve(channelID, params)
	if err != nil {
		t.Fatalf("Resolve(%q) error = %v", channelID, err)
	}
	configured := make([]state.ModelConfig, 0, len(models))
	for _, model := range models {
		configured = append(configured, state.ModelConfig{ID: model})
	}
	return state.GroupView{
		ID:             1,
		ChannelID:      channelID,
		ConnectionType: connectionType,
		ResolvedTarget: resolved,
		Models:         configured,
		HeaderRules:    state.HeaderRules{Set: map[string]string{}},
	}
}

func TestBuildGroupProbeTargetUsesExplicitProviderContract(t *testing.T) {
	t.Parallel()
	group := probeTestGroupView(t, channel.OpenAI, "api_key", "gpt-4o")
	target, ok := buildGroupProbeTarget(group, "")
	if !ok {
		t.Fatal("buildGroupProbeTarget() = unsupported, want OpenAI probe contract")
	}
	if target.protocol != "openai-responses" {
		t.Fatalf("probe protocol = %q, want openai-responses", target.protocol)
	}
	if !target.routeMode.Valid() {
		t.Fatalf("probe route mode = %q, want valid route", target.routeMode)
	}
	if target.model != "gpt-4o" {
		t.Fatalf("probe model = %q, want first configured model", target.model)
	}
}

func TestBuildGroupProbeTargetRejectsUnsupportedSubscriptionAndMissingContract(t *testing.T) {
	t.Parallel()
	if _, ok := buildGroupProbeTarget(probeTestGroupView(t, channel.Claude, "subscription", "claude-3"), ""); ok {
		t.Fatal("subscription probe unexpectedly supported")
	}
	if _, ok := buildGroupProbeTarget(probeTestGroupView(t, channel.OpenAI, "api_key"), ""); ok {
		t.Fatal("probe without a configured model unexpectedly supported")
	}
}
