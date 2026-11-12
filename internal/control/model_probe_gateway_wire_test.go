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

func TestModelProbeMultiProtocolGatewayPreparedWire(t *testing.T) {
	bodies := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chat_1","object":"chat.completion","created":1,"model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":"4"},"finish_reason":"stop"}]}`))
	}))
	defer server.Close()
	fixture := newServiceFixture(t)
	group, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		Name: stringPointer("gateway-probe"), ChannelID: channel.GPTLoad,
		Params:      json.RawMessage(`{"base_url":"` + server.URL + `"}`),
		Models:      optionalGroupModels{Set: true, Values: []GroupModel{{ID: "test-model", Alias: "alias-a", AliasEnabled: true}}},
		Credentials: "provider-key", ConfirmSameTarget: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = fixture.service.UpdateGroupSettings(t.Context(), group.GroupID, GroupSettingsUpdateRequest{
		Overrides: optionalField[config.Settings]{Set: true, Value: config.Settings{
			state.SettingParameterOverrides: []any{map[string]any{"match": map[string]any{"model": "alias-a"}, "set": map[string]any{"temperature": 0.3}, "remove": []any{"/max_tokens"}}},
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
	result, err := fixture.service.ProbeGroupModels(t.Context(), ModelProbeRequest{Targets: []ModelProbeTargetRequest{{GroupID: group.GroupID, Model: "test-model"}}})
	if err != nil || len(result.Results) != 1 || result.Results[0].Outcome != ProbeOutcomePassed {
		t.Fatalf("probe = %#v, %v", result, err)
	}
	select {
	case body := <-bodies:
		if body["temperature"] != 0.3 || body["model"] != "test-model" {
			t.Errorf("gateway body missing prepared fields: %#v", body)
		}
		if _, ok := body["max_tokens"]; ok {
			t.Errorf("gateway body retained removed budget: %#v", body)
		}
		if body["stream"] == true {
			t.Errorf("probe unexpectedly requests streaming: %#v", body)
		}
	default:
		t.Fatal("gateway was not called")
	}
}
