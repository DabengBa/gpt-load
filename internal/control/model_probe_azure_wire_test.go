package control

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution/bifrost"
	"gpt-load/internal/gateway"
	"gpt-load/internal/health"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/state"
)

func TestModelProbeAzureConvertedPreparedWire(t *testing.T) {
	bodies := make(chan map[string]any, 2)
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
		Name: stringPointer("azure-wire"), ChannelID: channel.AzureOpenAI,
		Params:      json.RawMessage(`{"endpoint":"` + server.URL + `"}`),
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
	if err != nil || len(result.Results) != 1 {
		t.Fatalf("probe = %#v, %v", result, err)
	}
	if result.Results[0].Outcome != ProbeOutcomePassed {
		t.Errorf("probe outcome = %s, reason = %v", result.Results[0].Outcome, result.Results[0].Reason)
	}
	if result.Results[0].RouteMode == nil || *result.Results[0].RouteMode != channel.RouteConverted {
		t.Errorf("probe route = %v, want converted", result.Results[0].RouteMode)
	}
	var probeBody map[string]any
	select {
	case body := <-bodies:
		probeBody = body
		if body["temperature"] != 0.3 {
			t.Errorf("wire missing alias override: %#v", body)
		}
		if _, ok := body["max_tokens"]; ok {
			t.Errorf("wire retained removed budget: %#v", body)
		}
	default:
		t.Fatalf("upstream not called: %#v", result.Results[0])
	}
	key, err := fixture.service.CreateAccessKey(t.Context(), AccessKeyCreateRequest{Name: "azure-comparison"})
	if err != nil {
		t.Fatal(err)
	}
	handler := gateway.NewHandler(fixture.manager, fixture.registry, fixture.encryption,
		gateway.NewExecutionForwarder(runtime), dialect.NewSet(dialect.NewOpenAI()),
		health.NewStatsStore(), health.NewMutationCoordinator(), nil, nil, nil)
	engine := gin.New()
	registerGatewayRoutes(t, engine, handler)
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"alias-a","messages":[{"role":"user","content":"What is 2 + 2? Please answer briefly."}],"max_tokens":128}`))
	request.Header.Set("Authorization", "Bearer "+key.Key)
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("ordinary gateway response = %d %s", response.Code, response.Body.String())
	}
	select {
	case ordinaryBody := <-bodies:
		if !reflect.DeepEqual(probeBody, ordinaryBody) {
			t.Errorf("converted HTTP wire differs: probe=%#v ordinary=%#v", probeBody, ordinaryBody)
		}
	default:
		t.Fatal("ordinary request did not reach upstream")
	}
}
