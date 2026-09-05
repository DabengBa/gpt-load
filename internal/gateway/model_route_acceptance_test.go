package gateway

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"gpt-load/internal/state"
)

func TestModelRouteEntriesAcceptanceRewritesAndLogsModels(t *testing.T) {
	forwarder := &scriptedForwarder{results: []UpstreamResult{modelUnavailableScriptedResult(), successScriptedResult()}}
	engine, handler, _ := newModelRewriteTestRuntime(t, forwarder, []state.ModelConfig{{ID: "up-a", Alias: "pub"}, {ID: "up-b", Alias: "pub"}}, "sk-one")
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"pub","messages":[{"role":"user","content":"acceptance"}]}`))
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)
	if response.Code != http.StatusOK || len(forwarder.inputs) != 2 {
		t.Fatalf("status/attempts=%d/%d body=%s", response.Code, len(forwarder.inputs), response.Body.String())
	}
	seen := map[string]bool{}
	for i, input := range forwarder.inputs {
		seen[input.UpstreamModelID] = true
		if input.ExternalModel != "pub" {
			t.Fatalf("attempt %d client model=%q", i, input.ExternalModel)
		}
		if input.UpstreamModelID == "" {
			t.Fatalf("attempt %d has no upstream model; body=%s", i, input.Request.Body)
		}
	}
	if !seen["up-a"] || !seen["up-b"] {
		t.Fatalf("attempt upstream models=%v, want both entries", seen)
	}
	events := sink.snapshot()
	if len(events) != 1 {
		t.Fatalf("request log events=%d, want 1", len(events))
	}
	event := events[0]
	if event.ClientModel != "pub" || event.UpstreamModel != forwarder.inputs[1].UpstreamModelID {
		t.Fatalf("request log client/upstream=%q/%q, want pub/%q", event.ClientModel, event.UpstreamModel, forwarder.inputs[1].UpstreamModelID)
	}
	if len(event.Attempts) != 2 || event.Attempts[0].UpstreamModel == "" || event.Attempts[1].UpstreamModel == "" {
		t.Fatalf("request log attempts=%#v", event.Attempts)
	}
}
