package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/platform/config"
)

const recoveryHistory = `{"model":"gpt-4o","input":[{"type":"reasoning","id":"rs_old","encrypted_content":"opaque","summary":[]},{"type":"function_call","id":"fc_old","call_id":"call_keep","name":"lookup","arguments":"{}"},{"type":"function_call_output","id":"out_old","call_id":"call_keep","output":"result"}]}`

func TestResponsesRecoveryBufferedHeartbeat(t *testing.T) {
	var bodies [][]byte
	var auth []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		auth = append(auth, r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "text/event-stream")
		if len(bodies) == 1 {
			_, _ = w.Write([]byte("event: error\ndata: {\"type\":\"error\",\"error\":{\"code\":\"invalid_encrypted_content\",\"message\":\"encrypted content could not be verified\"}}\n\n"))
		} else {
			_, _ = w.Write([]byte(bufferedResponsesFramingSuccessFixture))
		}
		w.(http.Flusher).Flush()
	}))
	t.Cleanup(server.Close)
	capture := &capturingExecutionForwarder{delegate: newConcreteTestExecutionForwarder(t)}
	engine, _ := newBufferedResponsesFramingRuntime(t, capture, 3,
		dialectGatewayGroup{id: 1, name: "first", upstreamURL: server.URL, apiKeys: []string{"sk-first"}, settings: config.Settings{}},
		dialectGatewayGroup{id: 2, name: "second", upstreamURL: server.URL, apiKeys: []string{"sk-second"}, settings: config.Settings{}},
	)
	body := strings.TrimSuffix(recoveryHistory, "}") + `,"stream":true,"store":false}`
	response := performBufferedResponsesFramingRequest(t, engine, body)
	if len(capture.results) != 2 {
		t.Fatalf("attempts = %d, want 2; body=%s", len(capture.results), response.Body.String())
	}
	first := capture.results[0]
	if !first.Committed || !first.HTTPCommitted || first.PayloadReleased {
		t.Fatalf("first result not heartbeat-only committed: %+v", first)
	}
	if auth[0] != auth[1] || bytes.Contains(bodies[1], []byte("opaque")) {
		t.Fatalf("recovery identity/body = %v %s", auth, bodies[1])
	}
	if got, want := response.Body.String(), bufferedStreamHeartbeat+bufferedResponsesFramingSuccessFixture; got != want {
		t.Fatalf("downstream = %q, want %q", got, want)
	}
	if got := bufferedStreamReservedBytes.Load(); got != 0 {
		t.Fatalf("spool leaked %d bytes", got)
	}
}

func TestResponsesRecoveryHTTP(t *testing.T) {
	for _, tc := range []struct {
		name   string
		budget int
		code   string
		repeat bool
		want   int
	}{
		{"rejection then success", 3, "invalid_encrypted_content", false, 2},
		{"budget exhausted", 0, "invalid_encrypted_content", false, 1},
		{"generic error is not cleanup", 3, "invalid_parameter", false, 2},
		{"repeated rejection stops", 4, "invalid_encrypted_content", true, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			var bodies [][]byte
			var auth []string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				mu.Lock()
				bodies = append(bodies, body)
				auth = append(auth, r.Header.Get("Authorization"))
				n := len(bodies)
				mu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				if n == 1 || tc.repeat {
					w.WriteHeader(400)
					_, _ = w.Write([]byte(`{"error":{"type":"invalid_request_error","code":"` + tc.code + `","message":"rejected"}}`))
					return
				}
				_, _ = w.Write(storedResponse("recovered").Body)
			}))
			t.Cleanup(server.Close)
			captured := &capturingExecutionForwarder{delegate: newConcreteTestExecutionForwarder(t)}
			h, engine, _ := newContinuationFixture(t, captured)
			setContinuationChannel(t, h, channel.OpenAI, server.URL)
			h.manager.Current().Settings.RetryCount = tc.budget
			useAffinityRandomValues(h, 0, 0, 0)
			status := 200
			if tc.budget == 0 || tc.repeat {
				status = 400
			}
			serveContinuation(t, engine, "gl-client", recoveryHistory, status)
			mu.Lock()
			defer mu.Unlock()
			if len(bodies) != tc.want {
				t.Fatalf("attempts = %d, want %d", len(bodies), tc.want)
			}
			if tc.budget == 0 {
				return
			}
			if tc.code == "invalid_parameter" {
				if !bytes.Contains(bodies[1], []byte("opaque")) {
					t.Fatalf("generic error cleaned history: %s", bodies[1])
				}
				return
			}
			if auth[0] != auth[1] {
				t.Errorf("recovery changed identity: %v; evidence=%+v", auth, captured.results[0].ExecutionError)
			}
			if bytes.Contains(bodies[1], []byte("opaque")) || bytes.Contains(bodies[1], []byte("fc_old")) {
				t.Errorf("rejected history survived: %s", bodies[1])
			}
			if !bytes.Contains(bodies[1], []byte("call_keep")) || !bytes.Contains(bodies[1], []byte("result")) {
				t.Errorf("tool pairing lost: %s", bodies[1])
			}
		})
	}
}
