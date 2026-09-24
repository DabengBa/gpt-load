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
)

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
		ConnectionType:    "api_key",
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
