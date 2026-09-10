package embedded

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func wsTestSession(t *testing.T, target string, proxyURLs ...string) *CodexWSSession {
	t.Helper()
	proxyURL := "direct"
	if len(proxyURLs) != 0 {
		proxyURL = proxyURLs[0]
	}
	session, err := NewCodexWSSession(CodexWSSessionOptions{
		CredentialID: "ws-test", Credential: CodexCredential{
			Type: ProviderCodex, AccessToken: "test-access", RefreshToken: "test-refresh", AccountID: "test-account",
		}, ProxyURL: proxyURL, TurnTimeout: 3 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	// 生产入口固定官方端点；测试仅把已固定的执行地址指向本地假上游。
	session.auth.Attributes["base_url"] = target
	t.Cleanup(func() { _ = session.Close() })
	return session
}

func wsCompleted(id string) []byte {
	return []byte(fmt.Sprintf(`{"type":"response.completed","response":{"id":%q,"object":"response","status":"completed","store":false,"output":[],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}`, id))
}

func TestCodexWSSessionUsesEachTurnDeadline(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()
		for turn := 0; turn < 2; turn++ {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
			if turn == 1 {
				time.Sleep(100 * time.Millisecond)
			}
			if err := conn.WriteMessage(websocket.TextMessage, wsCompleted(fmt.Sprintf("resp_%d", turn))); err != nil {
				return
			}
		}
		_, _, _ = conn.ReadMessage()
	}))
	defer server.Close()
	session := wsTestSession(t, server.URL)
	session.options.TurnTimeout = 25 * time.Millisecond
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err := session.ExecuteTurn(ctx, json.RawMessage(`{"model":"gpt-5","input":"first"}`), nil); err != nil {
		t.Fatal(err)
	}
	result, err := session.ExecuteTurn(ctx, json.RawMessage(`{"model":"gpt-5","previous_response_id":"resp_0","input":"next"}`), nil)
	if err != nil || result.ResponseID != "resp_1" {
		t.Fatalf("per-turn deadline was capped by session default: result=%+v err=%v", result, err)
	}
}

func TestCodexWSSessionForwardsPreparedHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Codex-Turn-State") != "state-fixture" {
			t.Error("prepared turn header missing")
		}
		if r.Header.Get("Authorization") != "Bearer test-access" {
			t.Error("prepared header replaced credential")
		}
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		if _, _, err = conn.ReadMessage(); err == nil {
			_ = conn.WriteMessage(websocket.TextMessage, wsCompleted("resp_headers"))
		}
		_, _, _ = conn.ReadMessage()
	}))
	defer server.Close()
	session := wsTestSession(t, server.URL)
	session.options.Headers = http.Header{"X-Codex-Turn-State": {"state-fixture"}, "Authorization": {"Bearer untrusted"}}
	if _, err := session.ExecuteTurn(t.Context(), json.RawMessage(`{"model":"gpt-5","input":"hello"}`), nil); err != nil {
		t.Fatal(err)
	}
}

func TestCodexWSSessionDatesHandshakeHeadersBeforeGeneration(t *testing.T) {
	generated := make(chan time.Time, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, http.Header{"X-Codex-Primary-Reset-After-Seconds": {"60"}})
		if err != nil {
			return
		}
		defer conn.Close()
		if _, _, err = conn.ReadMessage(); err != nil {
			return
		}
		time.Sleep(50 * time.Millisecond)
		generated <- time.Now()
		_ = conn.WriteMessage(websocket.TextMessage, wsCompleted("resp_headers_time"))
		_, _, _ = conn.ReadMessage()
	}))
	defer server.Close()
	session := wsTestSession(t, server.URL)
	result, err := session.ExecuteTurn(t.Context(), json.RawMessage(`{"model":"gpt-5","input":"hello"}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.HeaderObservedAt.IsZero() || !result.HeaderObservedAt.Before(<-generated) {
		t.Fatal("handshake headers were dated at generation completion")
	}
}

func TestCodexWSSessionPrewarmUsesRealUpstreamState(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		for turn := 0; turn < 2; turn++ {
			var request map[string]any
			if conn.ReadJSON(&request) != nil {
				return
			}
			if turn == 0 && request["generate"] != false {
				t.Error("prewarm became a generation request")
			}
			if turn == 1 && request["previous_response_id"] != "resp_0" {
				t.Error("prewarm state was not continued")
			}
			if conn.WriteMessage(websocket.TextMessage, wsCompleted(fmt.Sprintf("resp_%d", turn))) != nil {
				return
			}
		}
		_, _, _ = conn.ReadMessage()
	}))
	defer server.Close()
	session := wsTestSession(t, server.URL)
	first, err := session.ExecuteTurn(t.Context(), json.RawMessage(`{"model":"gpt-5","input":"warm","generate":false}`), nil)
	if err != nil || first.ResponseID != "resp_0" {
		t.Fatalf("prewarm result=%+v err=%v", first, err)
	}
	if _, err = session.ExecuteTurn(t.Context(), json.RawMessage(`{"model":"gpt-5","input":"continue","previous_response_id":"resp_0"}`), nil); err != nil {
		t.Fatal(err)
	}
}

func TestCodexWSSessionContinuationAndIsolation(t *testing.T) {
	var connections, turns atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" || r.Header.Get("Authorization") != "Bearer test-access" {
			t.Error("unexpected request identity or path")
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, http.Header{"X-Codex-Test": {"handshake"}})
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()
		n := connections.Add(1)
		previous := ""
		for {
			_, body, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var request struct {
				Type     string `json:"type"`
				Previous string `json:"previous_response_id"`
				Store    bool   `json:"store"`
			}
			if err := json.Unmarshal(body, &request); err != nil {
				t.Error(err)
				return
			}
			if request.Type != "response.create" || request.Previous != previous || request.Store {
				t.Error("request lost continuation or stateless semantics")
				return
			}
			previous = fmt.Sprintf("resp_%d_%d", n, turns.Add(1))
			if err := conn.WriteMessage(websocket.TextMessage, wsCompleted(previous)); err != nil {
				return
			}
		}
	}))
	defer server.Close()
	session := wsTestSession(t, server.URL)
	events := 0
	consume := func(ctx context.Context, event json.RawMessage) error {
		if !json.Valid(event) {
			t.Error("event is not native JSON")
		}
		events++
		return nil
	}
	firstCtx, cancelFirst := context.WithCancel(context.Background())
	defer cancelFirst()
	first, err := session.ExecuteTurn(firstCtx, json.RawMessage(`{"model":"gpt-5","input":"hello","store":false}`), consume)
	if err != nil {
		t.Fatal(err)
	}
	if first.ResponseID == "" || first.Status != "completed" || !json.Valid(first.Usage) {
		t.Fatalf("missing terminal result: %+v", first)
	}
	cancelFirst() // 成功请求的 context 结束不应关闭已空闲的会话。
	second, err := session.ExecuteTurn(context.Background(), json.RawMessage(fmt.Sprintf(`{"model":"gpt-5","input":"continue","previous_response_id":%q}`, first.ResponseID)), consume)
	if err != nil {
		t.Fatal(err)
	}
	if second.ResponseID == first.ResponseID || events != 2 || connections.Load() != 1 {
		t.Fatal("turns did not reuse one connection")
	}
	if first.Headers.Get("X-Codex-Test") == "" || second.Headers.Get("X-Codex-Test") != "" {
		t.Fatal("handshake headers were lost or reused as fresh observation")
	}
	other := wsTestSession(t, server.URL)
	if _, err := other.ExecuteTurn(context.Background(), json.RawMessage(`{"model":"gpt-5","input":"another"}`), consume); err != nil {
		t.Fatal(err)
	}
	if connections.Load() != 2 {
		t.Fatal("independent sessions shared a connection")
	}
}

func TestCodexWSSessionRejectsInvalidOptions(t *testing.T) {
	valid := CodexCredential{Type: ProviderCodex, AccessToken: "a", RefreshToken: "r", AccountID: "acc"}
	cases := []struct {
		name    string
		options CodexWSSessionOptions
		code    string
	}{
		{"missing credential id", CodexWSSessionOptions{Credential: valid, ProxyURL: "direct"}, "invalid_session_options"},
		{"invalid credential", CodexWSSessionOptions{CredentialID: "id", ProxyURL: "direct"}, "invalid_session_options"},
		{"empty proxy", CodexWSSessionOptions{CredentialID: "id", Credential: valid}, "invalid_proxy"},
		{"proxy with path", CodexWSSessionOptions{CredentialID: "id", Credential: valid, ProxyURL: "http://127.0.0.1:8080/prefix"}, "invalid_proxy"},
		{"negative timeout", CodexWSSessionOptions{CredentialID: "id", Credential: valid, ProxyURL: "direct", TurnTimeout: -time.Second}, "invalid_session_options"},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			session, err := NewCodexWSSession(item.options)
			if session != nil {
				t.Fatal("session created from invalid options")
			}
			var failure *CodexWSError
			if !errors.As(err, &failure) || failure.Code != item.code || failure.DispatchState != CodexWSNotSent {
				t.Fatalf("unexpected rejection: %v", err)
			}
		})
	}
}

func TestCodexWSSessionCancelledBeforeSendIsNotSent(t *testing.T) {
	session := wsTestSession(t, "https://unused.invalid")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := session.ExecuteTurn(ctx, json.RawMessage(`{"model":"gpt-5","input":"hello"}`), nil)
	var failure *CodexWSError
	if !errors.As(err, &failure) || failure.Code != "canceled" || !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation contract lost: %v", err)
	}
	if result.DispatchState != CodexWSNotSent {
		t.Fatalf("cancelled turn claimed dispatch: %s", result.DispatchState)
	}
}

func TestCodexWSSessionClosedSessionIsNotReused(t *testing.T) {
	session := wsTestSession(t, "https://unused.invalid")
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	result, err := session.ExecuteTurn(context.Background(), json.RawMessage(`{"model":"gpt-5","input":"hello"}`), nil)
	var failure *CodexWSError
	if !errors.As(err, &failure) || failure.Code != "session_closed" || result.DispatchState != CodexWSNotSent {
		t.Fatalf("closed session reused: %v", err)
	}
}

func TestCodexWSSessionContinuationRequiresSession(t *testing.T) {
	session := wsTestSession(t, "https://unused.invalid")
	result, err := session.ExecuteTurn(context.Background(), json.RawMessage(`{"model":"gpt-5","input":"hello","previous_response_id":"resp_1"}`), nil)
	var failure *CodexWSError
	if !errors.As(err, &failure) || failure.Code != "continuation_requires_session" || result.DispatchState != CodexWSNotSent {
		t.Fatalf("continuation accepted without session: %v", err)
	}
}

func TestCodexWSSessionRejectsConcurrentTurn(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()
		if _, _, err := conn.ReadMessage(); err != nil {
			return
		}
		close(started)
		<-release
		_ = conn.WriteMessage(websocket.TextMessage, wsCompleted("resp_busy"))
	}))
	defer server.Close()
	session := wsTestSession(t, server.URL)
	done := make(chan error, 1)
	go func() {
		_, err := session.ExecuteTurn(context.Background(), json.RawMessage(`{"model":"gpt-5","input":"hello"}`), nil)
		done <- err
	}()
	<-started
	result, err := session.ExecuteTurn(context.Background(), json.RawMessage(`{"model":"gpt-5","input":"second"}`), nil)
	var failure *CodexWSError
	if !errors.As(err, &failure) || failure.Code != "session_busy" || result.DispatchState != CodexWSNotSent {
		t.Fatalf("concurrent turn accepted: %v", err)
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestCodexWSSessionTurnReusesOneConnection(t *testing.T) {
	var connections atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" || r.Header.Get("Authorization") != "Bearer test-access" {
			t.Error("unexpected request identity or path")
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, http.Header{"X-Codex-Test": {"handshake"}})
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()
		connections.Add(1)
		for index := 1; ; index++ {
			_, body, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var request struct {
				Type     string `json:"type"`
				Previous string `json:"previous_response_id"`
			}
			if err := json.Unmarshal(body, &request); err != nil {
				t.Error(err)
				return
			}
			if request.Type != "response.create" {
				t.Errorf("unexpected upstream request type %q", request.Type)
				return
			}
			if err := conn.WriteMessage(websocket.TextMessage, wsCompleted(fmt.Sprintf("resp_%d", index))); err != nil {
				return
			}
		}
	}))
	defer server.Close()
	session := wsTestSession(t, server.URL)
	events := 0
	consume := func(ctx context.Context, event json.RawMessage) error {
		if !json.Valid(event) {
			t.Error("event is not native JSON")
		}
		events++
		return nil
	}
	first, err := session.ExecuteTurn(context.Background(), json.RawMessage(`{"model":"gpt-5","input":"hello"}`), consume)
	if err != nil {
		t.Fatal(err)
	}
	if first.ResponseID == "" || first.Status != "completed" || !json.Valid(first.Usage) {
		t.Fatalf("missing terminal result: %+v", first)
	}
	if first.DispatchState != CodexWSMaybeSent || first.Headers.Get("X-Codex-Test") == "" {
		t.Fatalf("successful turn lost dispatch or handshake evidence: %+v", first)
	}
	second, err := session.ExecuteTurn(context.Background(), json.RawMessage(fmt.Sprintf(`{"model":"gpt-5","input":"continue","previous_response_id":%q}`, first.ResponseID)), consume)
	if err != nil {
		t.Fatal(err)
	}
	if second.ResponseID == first.ResponseID || events != 2 || connections.Load() != 1 {
		t.Fatal("turns did not reuse one connection")
	}
}

func TestCodexWSSessionDisconnectAfterSendIsMaybeSent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			t.Error(err)
			return
		}
		if _, _, err := conn.ReadMessage(); err != nil {
			return
		}
		_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"response.created","response":{"id":"resp_x","status":"in_progress"}}`))
		_ = conn.Close()
	}))
	defer server.Close()
	session := wsTestSession(t, server.URL)
	result, err := session.ExecuteTurn(context.Background(), json.RawMessage(`{"model":"gpt-5","input":"hello"}`), nil)
	if err == nil {
		t.Fatal("expected failure after upstream disconnect")
	}
	if result.DispatchState != CodexWSMaybeSent {
		t.Fatalf("post-send disconnect claimed not_sent: %s", result.DispatchState)
	}
	if _, next := session.ExecuteTurn(context.Background(), json.RawMessage(`{"model":"gpt-5","input":"hello"}`), nil); !errors.As(next, new(*CodexWSError)) {
		t.Fatal("session survived a post-send disconnect")
	}
}

func TestCodexWSErrorTextHidesUpstreamBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()
		if _, _, err := conn.ReadMessage(); err != nil {
			return
		}
		_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"error","error":{"code":"rate_limit_exceeded","message":"secret-upstream-body"}}`))
		_, _, _ = conn.ReadMessage()
	}))
	defer server.Close()
	session := wsTestSession(t, server.URL)
	result, err := session.ExecuteTurn(context.Background(), json.RawMessage(`{"model":"gpt-5","input":"hello"}`), nil)
	if result.DispatchState != CodexWSMaybeSent {
		t.Fatalf("upstream failure claimed not_sent: %s", result.DispatchState)
	}
	var failure *CodexWSError
	if !errors.As(err, &failure) {
		t.Fatalf("error contract lost: %v", err)
	}
	if failure.UpstreamCode != "rate_limit_exceeded" || strings.Contains(err.Error(), "secret-upstream-body") {
		t.Fatalf("error lost classification or leaked upstream body: %v", err)
	}
}

func TestCodexWSSessionFixedIdentity(t *testing.T) {
	const wantUA = "codex-tui/0.153.3 (Mac OS 26.5.1; arm64) iTerm.app/3.6.11 (codex-tui; 0.153.3)"
	for _, model := range []string{"gpt-6-astra", "gpt-5.6-luna"} {
		for _, test := range []struct {
			name    string
			headers http.Header
		}{
			{name: "absent"},
			{name: "supplied", headers: http.Header{"Version": {"9.9.9"}, "User-Agent": {"codex_cli_rs/0.200.0"}}},
			{name: "empty", headers: http.Header{"Version": {""}, "User-Agent": {""}}},
			{name: "case variants", headers: http.Header{"version": {"9.9.9"}, "user-agent": {"custom-client/1.0"}}},
		} {
			t.Run(model+"/"+test.name, func(t *testing.T) {
				var handshakes atomic.Int32
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					handshakes.Add(1)
					if r.Header.Get("Version") != "0.153.3" || r.Header.Get("User-Agent") != wantUA {
						t.Errorf("handshake identity: version=%q UA=%q", r.Header.Get("Version"), r.Header.Get("User-Agent"))
					}
					if r.Header.Get("Authorization") != "Bearer test-access" {
						t.Error("identity normalization changed credential")
					}
					conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
					if err != nil {
						return
					}
					defer conn.Close()
					for turn := 0; turn < 2; turn++ {
						if _, _, err := conn.ReadMessage(); err != nil {
							return
						}
						if err := conn.WriteMessage(websocket.TextMessage, wsCompleted(fmt.Sprintf("resp_fixed_%d", turn))); err != nil {
							return
						}
					}
					_, _, _ = conn.ReadMessage()
				}))
				defer server.Close()
				session := wsTestSession(t, server.URL)
				session.options.Headers = test.headers.Clone()
				for turn := 0; turn < 2; turn++ {
					payload := json.RawMessage(fmt.Sprintf(`{"model":%q,"input":"hello"}`, model))
					if turn == 1 {
						payload = json.RawMessage(fmt.Sprintf(`{"model":%q,"input":"next","previous_response_id":"resp_fixed_0"}`, model))
					}
					if _, err := session.ExecuteTurn(t.Context(), payload, nil); err != nil {
						t.Fatal(err)
					}
				}
				if handshakes.Load() != 1 {
					t.Errorf("handshakes = %d, want one reused connection", handshakes.Load())
				}
			})
		}
	}
}
