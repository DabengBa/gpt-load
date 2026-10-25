package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/state"
)

func TestResponsesRecoveryUnavailableAfterHeartbeatKeepsTerminalFraming(t *testing.T) {
	var handler *Handler
	calls := 0
	executor := fakeExecutionExecutor{stream: func(_ context.Context, _ execution.AttemptSpec, sink execution.StreamSink) execution.StreamResult {
		calls++
		if err := sink(execution.StreamEvent{Sequence: 1, Kind: execution.StreamEventReady, StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}}); err != nil {
			t.Error(err)
		}
		handler.registry.(*state.CredentialRegistry).SetCooldown(1, time.Now().Add(time.Hour))
		return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindProvider, OriginHint: execution.ErrorOriginUpstream, Code: "invalid_encrypted_content", Summary: "encrypted content could not be verified"}}
	}}
	engine, h := newBufferedResponsesFramingRuntime(t, NewExecutionForwarder(executor), 3, dialectGatewayGroup{id: 1, name: "only", upstreamURL: "https://unused.example", apiKeys: []string{"sk-first"}})
	handler = h
	body := append([]byte(`{"stream":true,"store":false,`), []byte(recoveryHistory)[1:]...)
	response := performBufferedResponsesFramingRequest(t, engine, string(body))
	if calls != 1 {
		t.Fatalf("attempts = %d, want 1", calls)
	}
	if got := response.Body.String(); got != bufferedStreamHeartbeat+bufferedResponsesFramingAmbiguousErrorEvent {
		t.Fatalf("unavailable replay body = %q, want heartbeat + terminal SSE error", got)
	}
}

func TestResponsesRecoveryPreservesCandidateOverrides(t *testing.T) {
	var bodies [][]byte
	var auth []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var b json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&b); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, b)
		auth = append(auth, r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		switch len(bodies) {
		case 1:
			w.WriteHeader(400)
			_, _ = w.Write([]byte(`{"error":{"code":"response_item_resource_mismatch","message":"rejected item"}}`))
		case 2:
			w.WriteHeader(429)
			_, _ = w.Write([]byte(`{"error":{"code":"rate_limit_exceeded","message":"rate limited"}}`))
		default:
			_, _ = w.Write(storedResponse("recovered").Body)
		}
	}))
	t.Cleanup(server.Close)
	settings := func(field string) config.Settings {
		return config.Settings{state.SettingParameterOverrides: []any{map[string]any{"set": map[string]any{field: true}}}}
	}
	engine, _ := newBufferedResponsesFramingRuntime(t, newConcreteTestExecutionForwarder(t), 3,
		dialectGatewayGroup{id: 1, name: "first", upstreamURL: server.URL, apiKeys: []string{"sk-first"}, settings: settings("first_only")},
		dialectGatewayGroup{id: 2, name: "second", upstreamURL: server.URL, apiKeys: []string{"sk-second"}, settings: settings("second_only")},
	)
	serveContinuation(t, engine, "gl-client", recoveryHistory, 200)
	if len(bodies) != 3 {
		t.Fatalf("attempts = %d, want 3", len(bodies))
	}
	if auth[0] != auth[1] || auth[1] == auth[2] {
		t.Fatalf("identity order = %v", auth)
	}
	for i, b := range bodies {
		field := []byte("first_only")
		other := []byte("second_only")
		if i == 2 {
			field, other = other, field
		}
		if !bytes.Contains(b, field) || bytes.Contains(b, other) {
			t.Errorf("candidate overrides at %d: %s", i, b)
		}
		if i > 0 && (bytes.Contains(b, []byte("fc_old")) || bytes.Contains(b, []byte("opaque"))) {
			t.Errorf("cleanup not retained at %d: %s", i, b)
		}
	}
}

func TestResponsesRecoveryContinuationKeepsOwner(t *testing.T) {
	var auth []string
	var bodies [][]byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var b json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&b); err != nil {
			t.Error(err)
		}
		auth = append(auth, r.Header.Get("Authorization"))
		bodies = append(bodies, b)
		w.Header().Set("Content-Type", "application/json")
		if len(bodies) == 2 {
			w.WriteHeader(400)
			_, _ = w.Write([]byte(`{"error":{"code":"invalid_encrypted_content"}}`))
			return
		}
		_, _ = w.Write(storedResponse("owner-response").Body)
	}))
	t.Cleanup(server.Close)
	h, engine, _ := newContinuationFixture(t, newConcreteTestExecutionForwarder(t))
	setContinuationChannel(t, h, channel.OpenAI, server.URL)
	h.manager.Current().Settings.RetryCount = 3
	useAffinityRandomValues(h, 0, 1, 1)
	serveContinuation(t, engine, "gl-client", `{"model":"gpt-4o","input":"initial","store":true}`, 200)
	body := append([]byte(`{"previous_response_id":"owner-response",`), []byte(recoveryHistory)[1:]...)
	serveContinuation(t, engine, "gl-client", string(body), 200)
	if len(bodies) != 3 {
		t.Fatalf("attempts = %d, want 3", len(bodies))
	}
	for i := 1; i < 3; i++ {
		if auth[i] != auth[0] || !bytes.Contains(bodies[i], []byte(`"previous_response_id":"owner-response"`)) {
			t.Errorf("owner binding bypassed: %v %s", auth, bodies[i])
		}
	}
}

func TestResponsesRecoveryNoReplayAfterVisiblePayload(t *testing.T) {
	forwarder := &scriptedForwarder{streamResults: []UpstreamResult{{StatusCode: 200, Committed: true, HTTPCommitted: true, BufferedStream: true, PayloadReleased: true, ClientVisibleBytes: 100,
		DispatchState: execution.DispatchMaybeSent, ExecutionError: &execution.ErrorEvidence{Kind: execution.ErrorKindProvider, Code: "invalid_encrypted_content", Summary: "encrypted content could not be verified"}, Stream: StreamObservation{EndReason: StreamEndSSEError}}, storedResponse("must-not-run")}}
	_, engine, _ := newContinuationFixture(t, forwarder)
	serveContinuation(t, engine, "gl-client", `{"model":"gpt-4o","stream":true,"input":[{"type":"reasoning","encrypted_content":"opaque"}]}`, 200)
	if len(forwarder.streamInputs) != 1 {
		t.Fatalf("attempts after visible payload = %d, want 1", len(forwarder.streamInputs))
	}
}
