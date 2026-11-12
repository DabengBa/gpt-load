package control

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
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

func TestModelProbeUsesSelectedGroupsPreparedWire(t *testing.T) {
	bodies := make(chan []byte, 2)
	selected := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, _ := io.ReadAll(request.Body)
		bodies <- body
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"id":"resp_1","object":"response","status":"completed","model":"gpt-5.4","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"4","annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`))
	}))
	defer selected.Close()
	otherRequests := make(chan struct{}, 1)
	other := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		otherRequests <- struct{}{}
		writer.WriteHeader(http.StatusInternalServerError)
	}))
	defer other.Close()
	fixture := newServiceFixture(t)
	create := func(name, url string) uint {
		t.Helper()
		group, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
			Name: stringPointer(name), ChannelID: channel.OpenAI,
			Params:      json.RawMessage(`{"base_url":"` + url + `"}`),
			Models:      optionalGroupModels{Set: true, Values: []GroupModel{{ID: "gpt-5.4", Alias: "client-alias", AliasEnabled: true}}},
			Credentials: "provider-key", ConfirmSameTarget: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		return group.GroupID
	}
	groupID := create("selected-wire-group", selected.URL)
	create("other-wire-group", other.URL)
	if got := fixture.manager.Current().Groups[groupID].Models[0].Alias; got != "client-alias" {
		t.Fatalf("saved alias = %q, want client-alias", got)
	}
	_, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
		Overrides: optionalField[config.Settings]{Set: true, Value: config.Settings{
			state.SettingParameterOverrides: []any{map[string]any{
				"match": map[string]any{"model": "client-alias"}, "set": map[string]any{"temperature": 0.3, "fallbacks": []string{"other/model"}},
			}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	models, err := fixture.service.GetGroupModels(t.Context(), groupID)
	if err != nil || len(models.Items) != 1 {
		t.Fatalf("models = %#v, %v", models, err)
	}
	revision := fixture.manager.Current().Revision
	_, err = fixture.service.UpdateModelRouteSchedule(t.Context(), modelRouteSchedulePatchRequest{
		SnapshotRevision: &revision,
		Updates: []modelRouteSchedulePatchUpdate{{GroupID: groupID, EntryID: models.Items[0].EntryID,
			ReasoningEffort: optionalField[string]{Set: true, Value: "high"}}},
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
	response, err := fixture.service.ProbeGroupModels(t.Context(), ModelProbeRequest{
		Targets: []ModelProbeTargetRequest{{GroupID: groupID, Model: "gpt-5.4"}},
	})
	if err != nil || len(response.Results) != 1 {
		t.Fatalf("probe = %#v, %v", response, err)
	}
	select {
	case <-otherRequests:
		t.Fatal("probe reached the other group")
	default:
	}
	select {
	case body := <-bodies:
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatal(err)
		}
		reasoning, _ := payload["reasoning"].(map[string]any)
		if _, leaked := payload["fallbacks"]; leaked {
			t.Fatalf("control field leaked to provider wire: %s", body)
		}
		if payload["model"] != "gpt-5.4" || payload["temperature"] != 0.3 || reasoning["effort"] != "high" || payload["input"] == nil {
			t.Fatalf("provider wire = %s, want selected upstream model, preset, alias override and reasoning", body)
		}
	default:
		t.Fatal("selected upstream not called")
	}
	if response.Results[0].Outcome != ProbeOutcomePassed {
		t.Fatalf("probe result = %#v, want passed", response.Results[0])
	}
}

func TestReasoningPolicyPersistedScheduleReachesProviderWire(t *testing.T) {
	var upstreamBody []byte
	var upstreamPath string
	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		upstreamPath = request.URL.Path
		var err error
		upstreamBody, err = io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"id":"chatcmpl-policy","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	fixture := newServiceFixture(t)
	accessKey, err := fixture.service.CreateAccessKey(t.Context(), AccessKeyCreateRequest{Name: "reasoning-wire-client"})
	if err != nil {
		t.Fatal(err)
	}
	created, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		Name:              stringPointer("reasoning-wire-group"),
		ChannelID:         channel.OpenAI,
		Params:            json.RawMessage(`{"base_url":"` + upstream.URL + `"}`),
		Models:            optionalGroupModels{Set: true, Values: []GroupModel{{ID: "gpt-5.4"}}},
		Credentials:       "provider-wire-key",
		ConfirmSameTarget: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	models, err := fixture.service.GetGroupModels(t.Context(), created.GroupID)
	if err != nil || len(models.Items) != 1 {
		t.Fatalf("GetGroupModels() = %#v, %v", models, err)
	}
	revision := fixture.manager.Current().Revision
	if _, err := fixture.service.UpdateModelRouteSchedule(t.Context(), modelRouteSchedulePatchRequest{
		SnapshotRevision: &revision,
		Updates: []modelRouteSchedulePatchUpdate{{
			GroupID: created.GroupID, EntryID: models.Items[0].EntryID,
			ReasoningEffort: optionalField[string]{Set: true, Value: "high"},
		}},
	}); err != nil {
		t.Fatalf("UpdateModelRouteSchedule() error = %v", err)
	}
	row, err := fixture.service.GetGroupModels(t.Context(), created.GroupID)
	if err != nil || row.Items[0].EntryID == "" {
		t.Fatalf("persisted route entry = %#v, %v", row, err)
	}
	if got := fixture.manager.Current().Groups[created.GroupID].Models[0].ReasoningEffort; got != "high" {
		t.Fatalf("snapshot item reasoning override = %q, want high", got)
	}

	runtime, err := bifrost.NewRuntime(t.Context(), fixture.channelRegistry)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Shutdown()
	handler := gateway.NewHandler(
		fixture.manager,
		fixture.registry,
		fixture.encryption,
		gateway.NewExecutionForwarder(runtime),
		dialect.NewSet(dialect.NewOpenAI()),
		health.NewStatsStore(),
		health.NewMutationCoordinator(),
		nil,
		nil,
		nil,
	)
	engine := gin.New()
	registerGatewayRoutes(t, engine, handler)
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-5.4","messages":[{"role":"user","content":"hello"}]}`))
	request.Header.Set("Authorization", "Bearer "+accessKey.Key)
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("gateway response = %d %s", response.Code, response.Body.String())
	}
	if upstreamPath != "/v1/chat/completions" {
		t.Fatalf("upstream path = %q, want /v1/chat/completions", upstreamPath)
	}
	if got := string(upstreamBody); !bytes.Contains(upstreamBody, []byte(`"reasoning_effort":"high"`)) {
		t.Fatalf("provider wire body = %s, want reasoning_effort high", got)
	}
	if !bytes.Contains(upstreamBody, []byte(`"model":"gpt-5.4"`)) {
		t.Fatalf("provider wire body = %s, want upstream model", upstreamBody)
	}
}
