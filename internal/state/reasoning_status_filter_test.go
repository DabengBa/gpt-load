package state

import (
	"testing"

	"gpt-load/internal/platform/config"
)

func TestResponsesReasoningStatusFilterIsGroupOnlyAndDisabledByDefault(t *testing.T) {
	if IsRuntimeSettingKey(SettingResponsesReasoningStatusFilterEnabled) {
		t.Fatal("group-only setting is exposed as a system runtime setting")
	}

	base := DefaultRuntimeSettings()
	resolved, err := ResolveGroupRuntimeSettings(base, nil)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.ResponsesReasoningStatusFilterEnabled {
		t.Fatal("filter is enabled by default")
	}

	resolved, err = ResolveGroupRuntimeSettings(base, config.Settings{
		SettingResponsesReasoningStatusFilterEnabled: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !resolved.ResponsesReasoningStatusFilterEnabled {
		t.Fatal("filter override was not resolved")
	}

	if _, err := ResolveGroupRuntimeSettings(base, config.Settings{
		SettingResponsesReasoningStatusFilterEnabled: "true",
	}); err == nil {
		t.Fatal("non-boolean filter override was accepted")
	}
}
