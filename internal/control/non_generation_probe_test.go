package control

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"

	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution/bifrost"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
)

func TestModelProbeTargetUsesExplicitNativeProtocol(t *testing.T) {
	group := probeTestGroup(channel.Gemini, json.RawMessage(`{}`), "embedding-model")
	group.ClientProtocols = []protocol.Protocol{protocol.Gemini, protocol.GeminiEmbeddings}
	target, ok := buildGroupProbeTarget(group, "embedding-model", protocol.GeminiEmbeddings)
	if !ok || target.protocol != protocol.GeminiEmbeddings || target.maxOutputTokens != 0 {
		t.Fatalf("explicit embedding target = %+v, ok=%t", target, ok)
	}
}

func TestModelProbeManagementCompileHTTPVectorAndProtocolRecovery(t *testing.T) {
	fixture := newServiceFixture(t)
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		requests++
		body, _ := io.ReadAll(request.Body)
		var payload struct {
			Content struct {
				Parts []map[string]string `json:"parts"`
			} `json:"content"`
		}
		if request.Method != http.MethodPost || request.URL.Path != "/v1beta/models/embedding-model:embedContent" {
			t.Errorf("native embedding request = %s %s", request.Method, request.URL.Path)
		}
		if err := json.Unmarshal(body, &payload); err != nil || payload.Content.Parts[0]["text"] != "ping" {
			t.Errorf("native embedding body = %s", body)
		}
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusOK)
		if requests > 1 {
			_, _ = writer.Write([]byte(`{"embedding":{"values":["invalid"]}}`))
			return
		}
		_, _ = writer.Write([]byte(`{"embedding":{"values":[0.25,0.5]}}`))
	}))
	defer server.Close()
	name := "compiled-gemini-embedding-probe"
	created, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		Name: &name, ChannelID: channel.Gemini,
		Params:      json.RawMessage(`{"base_url":"` + server.URL + `/v1beta"}`),
		Models:      optionalGroupModels{Set: true, Values: []GroupModel{{ID: "embedding-model"}}},
		Credentials: "mock-key", ConnectionType: "api_key", ConfirmSameTarget: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	group := fixture.manager.Current().Groups[created.GroupID]
	entryID := probeModelEntryID(group, "embedding-model")
	if entryID == "" {
		t.Fatal("compiled group has no model route entry")
	}
	runtime, err := bifrost.NewRuntime(t.Context(), fixture.channelRegistry)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Shutdown()
	fixture.service.executor = runtime
	entryKey := state.RouteEntryKey{GroupID: created.GroupID, EntryID: entryID}
	if _, ok := fixture.registry.SetEntryBlacklistedWithChange(entryKey); !ok {
		t.Fatal("failed to seed recovery route")
	}
	response, err := fixture.service.ProbeGroupModels(t.Context(), ModelProbeRequest{Targets: []ModelProbeTargetRequest{
		{GroupID: created.GroupID, Model: "embedding-model", Protocol: protocol.GeminiEmbeddings},
		{GroupID: created.GroupID, Model: "embedding-model", Protocol: protocol.GeminiEmbeddings},
	}})
	if err != nil || len(response.Results) != 1 {
		t.Fatalf("management probe response = %#v, err=%v", response, err)
	}
	if response.Results[0].Protocol == nil || *response.Results[0].Protocol != protocol.GeminiEmbeddings {
		t.Fatalf("result protocol = %#v", response.Results[0].Protocol)
	}
	if response.Results[0].Outcome != ProbeOutcomePassed || requests != 1 {
		t.Fatalf("mock HTTP requests=%d, want one vector probe", requests)
	}
	if !response.Results[0].Recovered {
		t.Fatal("matching selected protocol did not recover route")
	}

	before := requests
	invalid, err := fixture.service.ProbeGroupModels(t.Context(), ModelProbeRequest{Targets: []ModelProbeTargetRequest{
		{GroupID: created.GroupID, Model: "embedding-model", Protocol: protocol.OpenAICompletions},
	}})
	if err != nil || len(invalid.Results) != 1 || invalid.Results[0].LogID != nil || requests != before {
		t.Fatalf("invalid protocol result=%#v err=%v requests=%d before=%d", invalid, err, requests, before)
	}
	if _, ok := fixture.registry.SetEntryBlacklistedWithChange(entryKey); !ok {
		t.Fatal("failed to reseed recovery route")
	}
	malformed, err := fixture.service.ProbeGroupModels(t.Context(), ModelProbeRequest{Targets: []ModelProbeTargetRequest{
		{GroupID: created.GroupID, Model: "embedding-model", Protocol: protocol.GeminiEmbeddings},
	}})
	if err != nil || len(malformed.Results) != 1 || malformed.Results[0].Recovered || requests != before+1 {
		t.Fatalf("invalid vector result=%#v err=%v requests=%d", malformed, err, requests)
	}
}

func TestEmbeddingOnlyGroupSelectsProbeBeforeDispatch(t *testing.T) {
	group := probeTestGroup(channel.Gemini, json.RawMessage(`{}`), "arbitrary-model")
	group.ClientProtocols = []protocol.Protocol{protocol.GeminiEmbeddings}
	target, ok := buildGroupProbeTarget(group, "", "")
	if !ok || target.protocol != protocol.Gemini || target.maxOutputTokens == 0 {
		t.Fatalf("embedding target = %+v, ok=%t", target, ok)
	}
	group.ClientProtocols = []protocol.Protocol{protocol.GeminiEmbeddings, protocol.Gemini}
	target, ok = buildGroupProbeTarget(group, "", protocol.GeminiEmbeddings)
	if !ok || target.protocol != protocol.GeminiEmbeddings || target.maxOutputTokens != 0 {
		t.Fatalf("mixed explicit target = %+v, ok=%t", target, ok)
	}
}
