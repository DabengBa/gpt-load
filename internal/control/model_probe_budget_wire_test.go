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

func TestModelProbeResponsesRespectsRemovedBudgetOnWire(t *testing.T) {
	bodies := make(chan map[string]any, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","status":"completed","model":"gpt-5.4","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"4","annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()
	fixture := newServiceFixture(t)
	group, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		Name: stringPointer("removed-budget"), ChannelID: channel.OpenAI,
		Params:      json.RawMessage(`{"base_url":"` + upstream.URL + `"}`),
		Models:      optionalGroupModels{Set: true, Values: []GroupModel{{ID: "gpt-5.4"}}},
		Credentials: "provider-key", ConfirmSameTarget: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = fixture.service.UpdateGroupSettings(t.Context(), group.GroupID, GroupSettingsUpdateRequest{
		Overrides: optionalField[config.Settings]{Set: true, Value: config.Settings{
			state.SettingParameterOverrides: []any{map[string]any{"remove": []any{"/max_output_tokens"}}},
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
	result, err := fixture.service.ProbeGroupModels(t.Context(), ModelProbeRequest{
		Targets: []ModelProbeTargetRequest{{GroupID: group.GroupID, Model: "gpt-5.4"}},
	})
	if err != nil || len(result.Results) != 1 || result.Results[0].Outcome != ProbeOutcomePassed {
		t.Fatalf("probe = %#v, %v", result, err)
	}
	select {
	case body := <-bodies:
		if _, exists := body["max_output_tokens"]; exists {
			t.Fatalf("provider wire retained removed budget: %#v", body)
		}
	default:
		t.Fatal("upstream not called")
	}
}
