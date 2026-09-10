package codex

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

func TestWSSessionIsExplicitAndHTTPExecutorIsUnchanged(t *testing.T) {
	session, err := NewWSSession(WSSessionOptions{
		CredentialID: "12", Credential: Credential{Type: Provider, AccessToken: "test-access", RefreshToken: "test-refresh", AccountID: "test-account"},
		ProxyURL: "direct",
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := session.Close(); err != nil {
			t.Error(err)
		}
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := session.ExecuteTurn(ctx, json.RawMessage(`{"model":"gpt-5","input":"hello"}`), nil)
	var failure *WSError
	if !errors.Is(err, context.Canceled) || !errors.As(err, &failure) || failure.Code != "canceled" || result.DispatchState != "not_sent" {
		t.Fatalf("cancellation contract lost: %v", err)
	}
	httpExecutor := NewExecutor().(*executor)
	if httpExecutor.bridge == nil {
		t.Fatal("existing HTTP executor changed")
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	closed, err := session.ExecuteTurn(context.Background(), json.RawMessage(`{"model":"gpt-5","input":"hello"}`), nil)
	if !errors.As(err, &failure) || failure.Code != "session_closed" || closed.DispatchState != "not_sent" {
		t.Fatalf("closed session reused: %v", err)
	}
}

func TestNewWSSessionRejectsUnresolvedProxy(t *testing.T) {
	credential := Credential{Type: Provider, AccessToken: "test-access", RefreshToken: "test-refresh", AccountID: "test-account"}
	for _, proxyURL := range []string{"", "environment", "http://127.0.0.1:8080/prefix"} {
		session, err := NewWSSession(WSSessionOptions{CredentialID: "12", Credential: credential, ProxyURL: proxyURL})
		if session != nil {
			t.Fatalf("session created for proxy %q", proxyURL)
		}
		var failure *WSError
		if !errors.As(err, &failure) || failure.Code != "invalid_proxy" || failure.DispatchState != "not_sent" {
			t.Fatalf("proxy %q accepted: %v", proxyURL, err)
		}
	}
}
