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

func TestModelProbeRerankPreparedWire(t *testing.T) {
	bodies := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"r1","results":[{"index":0,"relevance_score":0.9}]}`))
	}))
	defer server.Close()
	fixture := newServiceFixture(t)
	group, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		Name: stringPointer("rerank-wire"), ChannelID: channel.Cohere,
		Params:      json.RawMessage(`{"base_url":"` + server.URL + `"}`),
		Models:      optionalGroupModels{Set: true, Values: []GroupModel{{ID: "rerank-model", Alias: "alias-a", AliasEnabled: true}}},
		Credentials: "provider-key", ConfirmSameTarget: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = fixture.service.UpdateGroupSettings(t.Context(), group.GroupID, GroupSettingsUpdateRequest{
		Overrides: optionalField[config.Settings]{Set: true, Value: config.Settings{
			state.SettingParameterOverrides: []any{map[string]any{"match": map[string]any{"model": "alias-a"}, "set": map[string]any{"query": "override query"}, "remove": []any{"/top_n"}}},
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
	result, err := fixture.service.ProbeGroupModels(t.Context(), ModelProbeRequest{Targets: []ModelProbeTargetRequest{{GroupID: group.GroupID, Model: "rerank-model"}}})
	if err != nil || len(result.Results) != 1 {
		t.Fatalf("probe = %#v, %v", result, err)
	}
	select {
	case body := <-bodies:
		if body["query"] != "override query" {
			t.Errorf("wire missing alias override: %#v", body)
		}
		if _, ok := body["top_n"]; ok {
			t.Errorf("wire retained removed top_n: %#v", body)
		}
	default:
		t.Fatalf("upstream not called: %#v", result.Results[0])
	}
}
