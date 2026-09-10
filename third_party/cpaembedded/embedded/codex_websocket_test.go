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
