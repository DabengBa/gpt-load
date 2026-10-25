package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/state"
)

// Both sockets are real loopback WS connections: no scripted execution result
// grants replay safety or fabricates the callback's delivery boundary.
func TestWebsocketRecoveryMultiplexConcurrentLanes(t *testing.T) {
	firstReceived := make(chan struct{})
	var mu sync.Mutex
	var received []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		read := func() (string, string, bool) {
			_, body, err := conn.ReadMessage()
			if err != nil {
				return "", "", false
			}
			var req struct {
				StreamID string `json:"stream_id"`
			}
			_ = json.Unmarshal(body, &req)
			mu.Lock()
			received = append(received, string(body))
			mu.Unlock()
			return req.StreamID, string(body), true
		}
		lane, _, ok := read()
		if !ok || lane != "a" {
			return
		}
		close(firstReceived)
		lane, body, ok := read()
		if !ok || lane != "b" || !strings.Contains(body, "opaque_b") {
			return
		}
		// Lane b remains active while a rejects and replays on this socket.
		if conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"error","stream_id":"a","status":400,"error":{"code":"invalid_encrypted_content","message":"rejected history"}}`)) != nil {
			return
		}
		lane, body, ok = read()
		if !ok || lane != "a" || strings.Contains(body, "opaque_a") {
			return
		}
		_ = conn.WriteMessage(websocket.TextMessage, websocketCompleted("resp_b", "b"))
		_ = conn.WriteMessage(websocket.TextMessage, websocketCompleted("resp_a", "a"))
		// A later bound turn must not inherit a's earlier cleanup state. It
		// also retains its completed parent and its original lane.
		lane, body, ok = read()
		if !ok || lane != "b" || !strings.Contains(body, "opaque_a") || !strings.Contains(body, "resp_b") {
			return
		}
		_ = conn.WriteMessage(websocket.TextMessage, websocketCompleted("resp_b2", "b"))
		_, _, _ = conn.ReadMessage()
	}))
	t.Cleanup(upstream.Close)
	h, engine, input := websocketTestHandler(t, upstream.URL+"/v1", channel.OpenAI)
	input.SystemSettings = config.Settings{state.SettingRetryCount: 3}
	if _, err := h.manager.Publish(input); err != nil {
		t.Fatal(err)
	}
	sink := &recordingRequestLogSink{}
	h.requestLogSink = sink
	server := httptest.NewServer(engine)
	t.Cleanup(server.Close)
	conn := dialGatewayWebsocket(t, server.URL)
	t.Cleanup(func() { _ = conn.Close() })
	_ = conn.SetReadDeadline(time.Now().Add(3 * time.Second))
	request := func(lane, opaque, parent string) {
		t.Helper()
		value := map[string]any{"type": "response.create", "stream_id": lane, "model": "public", "store": false, "input": []any{map[string]any{"type": "reasoning", "encrypted_content": opaque, "summary": []any{map[string]any{"type": "summary_text", "text": "readable"}}}}}
		if parent != "" {
			value["previous_response_id"] = parent
		}
		if err := conn.WriteJSON(value); err != nil {
			t.Fatal(err)
		}
	}
	request("a", "opaque_a", "")
	select {
	case <-firstReceived:
	case <-time.After(3 * time.Second):
		t.Fatal("first lane not received")
	}
	request("b", "opaque_b", "")
	seen := make(map[string]int)
	for range 2 {
		_, body, err := conn.ReadMessage()
		if err != nil {
			t.Fatal(err)
		}
		var event struct{ Type, StreamID string }
		var fields map[string]json.RawMessage
		_ = json.Unmarshal(body, &fields)
		_ = json.Unmarshal(fields["type"], &event.Type)
		_ = json.Unmarshal(fields["stream_id"], &event.StreamID)
		if event.Type != "response.completed" {
			t.Fatalf("rejection leaked: %s", body)
		}
		seen[event.StreamID]++
	}
	if seen["a"] != 1 || seen["b"] != 1 {
		t.Fatalf("duplicate/missing output: %v", seen)
	}
	events := waitWebsocketLogs(t, sink, 2)
	attempts := 0
	for _, event := range events {
		attempts += len(event.Attempts)
	}
	if attempts != 3 {
		t.Fatalf("concurrent lane attempts=%d want=3", attempts)
	}
	request("b", "opaque_a", "resp_b")
	if _, body, err := conn.ReadMessage(); err != nil || !strings.Contains(string(body), "resp_b2") {
		t.Fatalf("bound parent/lane or turn-local state lost: body=%s err=%v", body, err)
	}
	waitWebsocketLogs(t, sink, 3)
	mu.Lock()
	defer mu.Unlock()
	if len(received) != 4 {
		t.Fatalf("received=%v want four creates", received)
	}
}

func TestWebsocketRecoveryLoopback(t *testing.T) {
	for _, tc := range []struct {
		name                                string
		multiplex, repeat, generic, visible bool
		retries, want                       int
	}{
		{name: "serial", retries: 3, want: 2},
		{name: "multiplex", multiplex: true, retries: 3, want: 2},
		{name: "repeat bounded", retries: 5, repeat: true, want: 2},
		{name: "multiplex repeat bounded", multiplex: true, retries: 5, repeat: true, want: 2},
		{name: "generic unchanged", multiplex: true, generic: true, retries: 3, want: 1},
		{name: "visible forbids replay", visible: true, retries: 3, want: 1},
		{name: "zero budget", want: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			var bodies, owners []string
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
				if err != nil {
					return
				}
				defer conn.Close()
				for {
					_, body, err := conn.ReadMessage()
					if err != nil {
						return
					}
					mu.Lock()
					bodies = append(bodies, string(body))
					owners = append(owners, r.Header.Get("Authorization"))
					n := len(bodies)
					mu.Unlock()
					lane := ""
					if tc.multiplex {
						lane = "a"
					}
					if n > 1 && !tc.repeat {
						_ = conn.WriteMessage(websocket.TextMessage, websocketCompleted("resp_ok", lane))
						continue
					}
					if tc.visible {
						_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"response.created","response":{"id":"resp_partial","object":"response","status":"in_progress"}}`))
					}
					code := "invalid_encrypted_content"
					if tc.generic {
						code = "invalid_parameter"
					}
					label := ""
					if lane != "" {
						label = `,"stream_id":"a"`
					}
					_ = conn.WriteMessage(websocket.TextMessage, []byte(fmt.Sprintf(`{"type":"error"%s,"status":400,"error":{"type":"invalid_request_error","code":%q,"message":"rejected history"}}`, label, code)))
				}
			}))
			t.Cleanup(upstream.Close)
			id := channel.CLIProxyAPI
			if tc.multiplex {
				id = channel.OpenAI
			}
			h, engine, input := websocketTestHandler(t, upstream.URL+"/v1", id)
			input.SystemSettings = config.Settings{state.SettingRetryCount: tc.retries}
			// A second healthy credential makes same-identity replay observable,
			// rather than an accidental consequence of a single candidate.
			backup := input.Groups[0]
			backup.ID, backup.Name = 2, "backup"
			input.Groups = append(input.Groups, backup)
			input.Credentials = append(input.Credentials, testCredentialConfig(2, 2))
			if _, err := h.manager.Publish(input); err != nil {
				t.Fatal(err)
			}
			if err := h.registry.(*state.CredentialRegistry).ReplaceCredentials([]state.CredentialEntry{
				testCredentialEntry(t, h.encryption, 1, 1, "upstream-key"),
				testCredentialEntry(t, h.encryption, 2, 2, "other-key"),
			}); err != nil {
				t.Fatal(err)
			}
			useAffinityRandomValues(h, 0, 0)
			sink := &recordingRequestLogSink{}

			h.requestLogSink = sink
			server := httptest.NewServer(engine)
			t.Cleanup(server.Close)
			conn := dialGatewayWebsocket(t, server.URL)
			t.Cleanup(func() { _ = conn.Close() })
			_ = conn.SetReadDeadline(time.Now().Add(3 * time.Second))
			request := map[string]any{"type": "response.create", "model": "public", "store": false, "input": []any{map[string]any{"type": "reasoning", "id": "rs_old", "encrypted_content": "opaque", "summary": []any{map[string]any{"type": "summary_text", "text": "keep me"}}}}}
			if tc.multiplex {
				request["stream_id"] = "a"
			}
			if err := conn.WriteJSON(request); err != nil {
				t.Fatal(err)
			}
			_, body, err := conn.ReadMessage()
			if err != nil {
				t.Fatal(err)
			}
			if tc.visible {
				if !strings.Contains(string(body), "response.created") {
					t.Fatalf("visible output lost: %s", body)
				}
				_, body, err = conn.ReadMessage()
				if err != nil {
					t.Fatal(err)
				}
			}
			events := waitWebsocketLogs(t, sink, 1)
			mu.Lock()
			defer mu.Unlock()
			if len(bodies) != tc.want || len(events[0].Attempts) != tc.want {
				t.Fatalf("attempts=%d logs=%d want=%d downstream=%s", len(bodies), len(events[0].Attempts), tc.want, body)
			}
			if tc.want == 2 {
				if owners[0] != owners[1] || owners[0] != "Bearer upstream-key" {
					t.Fatalf("identity changed: %v", owners)
				}
				if strings.Contains(bodies[1], "opaque") || strings.Contains(bodies[1], "rs_old") || !strings.Contains(bodies[1], "keep me") {
					t.Fatalf("unsafe/absent cleanup: %s", bodies[1])
				}
				if !events[0].Attempts[0].WillRetry || events[0].Attempts[0].Committed {
					t.Fatalf("retry delivery accounting: %+v", events[0].Attempts)
				}
			}
			var event struct {
				Type string `json:"type"`
			}
			_ = json.Unmarshal(body, &event)
			wantType := "error"
			if tc.want == 2 && !tc.repeat {
				wantType = "response.completed"
			}
			if event.Type != wantType {
				t.Fatalf("downstream=%s want type=%s", body, wantType)
			}
		})
	}
}

func TestWebsocketRecoveryBoundIdentityKeepsReconnect(t *testing.T) {
	testWebsocketBoundRetryableErrorRequestsClientReconnect(t,
		[]byte(`{"type":"error","status":400,"error":{"type":"invalid_request_error","code":"invalid_encrypted_content","message":"rejected history"}}`),
		execution.ErrorEvidence{Kind: execution.ErrorKindHTTP, StatusCode: http.StatusBadRequest,
			Code: "invalid_encrypted_content", OriginHint: execution.ErrorOriginUpstream}, "none")
}
