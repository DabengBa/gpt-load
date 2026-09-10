package embedded

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

type codexHeadersRoundTripperFunc func(*http.Request) (*http.Response, error)

func (fn codexHeadersRoundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}

func TestCodexHTTPFixedIdentityOnImagesAndWire(t *testing.T) {
	const defaultUA = "codex-tui/0.153.3 (Mac OS 26.5.1; arm64) iTerm.app/3.6.11 (codex-tui; 0.153.3)"
	const customUA = "codex-tui/0.200.0 (Mac OS 26.5.0; arm64)"
	for _, test := range []struct {
		name     string
		request  ExecuteRequest
		response string
		want     http.Header
	}{
		{
			name: "images keep fixed identity and explicit session",
			request: ExecuteRequest{
				Model: "gpt-image-2", Format: "openai-image", RequestPath: "/v1/images/generations",
				Payload:           []byte(`{"model":"gpt-image-2","prompt":"draw a circle"}`),
				Headers:           http.Header{"User-Agent": {customUA}, "Originator": {"custom-client"}, "Session_id": {"image-session"}},
				ConfiguredHeaders: []string{"User-Agent", "Originator"},
			},
			response: `{"created":1,"data":[{"b64_json":"aA=="}]}`,
			want:     http.Header{"User-Agent": {defaultUA}, "Originator": {"custom-client"}, "Version": {"0.153.3"}, "Session-Id": {"image-session"}},
		},
		{
			name: "empty version rules preserve fixed identity and empty originator",
			request: ExecuteRequest{
				Model: "gpt-5", Format: "openai-response", Payload: []byte(`{"model":"gpt-5","input":"hello"}`),
				Headers:           http.Header{"User-Agent": {customUA}, "Originator": {""}, "Version": {""}},
				ConfiguredHeaders: []string{"User-Agent", "Originator", "Version"},
			},
			response: "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5\",\"output\":[]}}\n\n",
			want:     http.Header{"User-Agent": {defaultUA}, "Originator": {""}, "Version": {"0.153.3"}},
		},
		{
			name: "removed identity remains fixed",
			request: ExecuteRequest{
				Model: "gpt-5", Format: "openai-response", Payload: []byte(`{"model":"gpt-5","input":"hello"}`),
				ConfiguredHeaders: []string{"User-Agent", "Originator", "Version"},
			},
			response: "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5\",\"output\":[]}}\n\n",
			want:     http.Header{"User-Agent": {defaultUA}, "Version": {"0.153.3"}},
		},
		{
			name: "image 2.5 uses existing direct image execution",
			request: ExecuteRequest{
				Model: "gpt-image-2.5", Format: "openai-image", RequestPath: "/v1/images/generations",
				Payload:           []byte(`{"model":"gpt-image-2.5","prompt":"draw a circle"}`),
				Headers:           http.Header{"User-Agent": {customUA}, "Version": {"9.9.9"}},
				ConfiguredHeaders: []string{"User-Agent", "Version"},
			},
			response: `{"created":1,"data":[{"b64_json":"aA=="}]}`,
			want:     http.Header{"User-Agent": {defaultUA}, "Originator": {"codex-tui"}, "Version": {"0.153.3"}},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			capturedHeaders := make(chan http.Header, 1)
			server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if test.request.Format == "openai-image" && r.URL.Path != "/backend-api/codex/images/generations" {
					t.Errorf("image path = %q", r.URL.Path)
				}
				capturedHeaders <- r.Header.Clone()
				if test.request.Format == "openai-image" {
					w.Header().Set("Content-Type", "application/json")
				} else {
					w.Header().Set("Content-Type", "text/event-stream")
				}
				if _, err := io.WriteString(w, test.response); err != nil {
					t.Errorf("write upstream response: %v", err)
				}
			}))
			defer server.Close()
			// dev 没有 BaseURL 通道，用转发到测试上游的 round tripper 改地址，
			// 保留真实 http.Transport，才能继续验证空 UA 不会被 Go 补回。
			upstream, err := url.Parse(server.URL)
			if err != nil {
				t.Fatalf("parse test upstream URL: %v", err)
			}
			transport := codexHeadersRoundTripperFunc(func(request *http.Request) (*http.Response, error) {
				redirected := request.Clone(request.Context())
				redirected.URL.Scheme = upstream.Scheme
				redirected.URL.Host = upstream.Host
				return server.Client().Transport.RoundTrip(redirected)
			})
			ctx := context.WithValue(t.Context(), "cliproxy.roundtripper", http.RoundTripper(transport))
			_, err = NewCodexHTTPExecutor().ExecuteCanonical(ctx, "credential-1", CodexCredential{
				Type: ProviderCodex, AccessToken: "access", RefreshToken: "refresh", AccountID: "account-1",
			}, test.request)
			if err != nil {
				t.Fatalf("ExecuteCanonical() error = %v", err)
			}
			captured := <-capturedHeaders
			for _, name := range []string{"User-Agent", "Originator", "Version", "Session-Id", "Session_id"} {
				_, present := captured[name]
				_, wantPresent := test.want[name]
				if present != wantPresent {
					t.Errorf("wire %s present = %t, want %t", name, present, wantPresent)
				}
				if got, want := captured.Get(name), test.want.Get(name); got != want {
					t.Errorf("wire %s = %q, want %q", name, got, want)
				}
			}
		})
	}
}
