package state

import (
	"encoding/json"
	"reflect"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/platform/config"
)

func TestParseReasoningEffortOverridesCanonicalizesEntries(t *testing.T) {
	result, err := ParseReasoningEffortOverrides(map[string]any{
		" upstream-a ": " HIGH ",
	})
	if err != nil {
		t.Fatalf("ParseReasoningEffortOverrides() error = %v", err)
	}
	if want := map[string]string{"upstream-a": "high"}; !reflect.DeepEqual(result, want) {
		t.Fatalf("ParseReasoningEffortOverrides() = %#v, want %#v", result, want)
	}
}

func TestParseReasoningEffortOverridesRejectsAmbiguousEntries(t *testing.T) {
	for _, value := range []any{
		map[string]any{},
		map[string]any{"upstream-a": "high", " upstream-a ": "low"},
		map[string]any{"upstream-a": "unsupported"},
	} {
		if _, err := ParseReasoningEffortOverrides(value); err == nil {
			t.Fatalf("ParseReasoningEffortOverrides(%#v) error = nil", value)
		}
	}
}

func TestCompileRejectsReasoningEffortOverrideForUnknownGroupModel(t *testing.T) {
	_, err := Compile(CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []GroupConfig{{
			ID: 1, Name: "group", ChannelID: channel.OpenAI, ConnectionType: "api_key",
			Params: json.RawMessage(`{}`),
			Models: []ModelConfig{{ID: "upstream-a"}},
			Settings: config.Settings{
				SettingReasoningEffortOverrides: map[string]any{"upstream-b": "high"},
			},
			Enabled: true,
		}},
	})
	if err == nil {
		t.Fatal("Compile() accepted an override for an unknown group model")
	}
}

func TestCompileCopiesReasoningEffortOverrides(t *testing.T) {
	overrides := map[string]any{"upstream-a": "low"}
	snapshot, err := Compile(CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []GroupConfig{{
			ID: 1, Name: "group", ChannelID: channel.OpenAI, ConnectionType: "api_key",
			Params:   json.RawMessage(`{}`),
			Models:   []ModelConfig{{ID: "upstream-a"}},
			Settings: config.Settings{SettingReasoningEffortOverrides: overrides},
			Enabled:  true,
		}},
	})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	overrides["upstream-a"] = "high"
	if got := snapshot.Groups[1].ReasoningEffortOverrides["upstream-a"]; got != "low" {
		t.Fatalf("snapshot override = %q, want copied low value", got)
	}
}
