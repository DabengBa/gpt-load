package control

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution/bifrost"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/state"
)

func TestModelProbeGeminiPreparedWire(t *testing.T) {
	bodies := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"4"}]},"finishReason":"STOP"}],"modelVersion":"gemini-test"}`))
	}))
	defer server.Close()
	fixture := newServiceFixture(t)
	group, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		Name: stringPointer("gemini-wire"), ChannelID: channel.Gemini,
		Params:      json.RawMessage(`{"base_url":"` + server.URL + `"}`),
		Models:      optionalGroupModels{Set: true, Values: []GroupModel{{ID: "gemini-test", Alias: "alias-a", AliasEnabled: true}}},
		Credentials: "provider-key", ConfirmSameTarget: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = fixture.service.UpdateGroupSettings(t.Context(), group.GroupID, GroupSettingsUpdateRequest{
		Overrides: optionalField[config.Settings]{Set: true, Value: config.Settings{
			state.SettingParameterOverrides: []any{map[string]any{"match": map[string]any{"model": "alias-a"}, "set": map[string]any{"generationConfig": map[string]any{"temperature": 0.3}}, "remove": []any{"/generationConfig/maxOutputTokens"}}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	runtime, err := bifrost.NewRuntime(t.Context(), fixture.channelRegistry)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Shutdown()
	fixture.service.executor = runtime
	result, err := fixture.service.ProbeGroupModels(t.Context(), ModelProbeRequest{Targets: []ModelProbeTargetRequest{{GroupID: group.GroupID, Model: "gemini-test"}}})
	if err != nil || len(result.Results) != 1 {
		t.Fatalf("probe = %#v, %v", result, err)
	}
	if result.Results[0].Outcome != ProbeOutcomePassed {
		t.Errorf("probe outcome = %s, reason = %v", result.Results[0].Outcome, result.Results[0].Reason)
	}
	select {
	case body := <-bodies:
		generation, _ := body["generationConfig"].(map[string]any)
		if generation["temperature"] != 0.3 {
			t.Errorf("wire missing alias override: %#v", body)
		}
		if _, ok := generation["maxOutputTokens"]; ok {
			t.Errorf("wire retained removed budget: %#v", body)
		}
	default:
		t.Fatalf("upstream not called: %#v", result.Results[0])
	}
}
