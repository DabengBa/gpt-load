package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/dialect"
)

type timeoutHTTPForwarder struct {
	client *http.Client
	url    string
}

func (forwarder *timeoutHTTPForwarder) ForwardStream(ctx context.Context, input ForwardInput, _ http.ResponseWriter) UpstreamResult {
	return forwarder.Forward(ctx, input)
}

func (forwarder *timeoutHTTPForwarder) Forward(ctx context.Context, _ ForwardInput) UpstreamResult {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, forwarder.url, nil)
	if err != nil {
		return UpstreamResult{Err: err}
	}
	response, err := forwarder.client.Do(request)
	if response != nil {
		response.Body.Close()
	}
	return completeScriptedUpstreamResult(UpstreamResult{Err: err, RequestWritten: true})
}

func TestHandlerUpstreamTimeoutReturnsOpenAIError(t *testing.T) {
	for _, test := range []struct {
		name string
		path string
		body string
	}{
		{name: "completions", path: "/v1/chat/completions", body: `{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}`},
		{name: "completions_stream", path: "/v1/chat/completions", body: `{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}],"stream":true}`},
		{name: "responses", path: "/v1/responses", body: `{"model":"gpt-4o","input":"Hello"}`},
		{name: "responses_stream", path: "/v1/responses", body: `{"model":"gpt-4o","input":"Hello","stream":true}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			upstreamCanceled := make(chan struct{})
			upstream := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, request *http.Request) {
				<-request.Context().Done()
				close(upstreamCanceled)
			}))
			defer upstream.Close()
			forwarder := &timeoutHTTPForwarder{
				client: &http.Client{Timeout: 50 * time.Millisecond},
				url:    upstream.URL,
			}
			handler, _, _ := newHandlerForTest(t, forwarder, "sk-one")
			handler.dialects = dialect.NewSet(dialect.NewOpenAI(), dialect.NewOpenAIResponses())
			engine := gin.New()
			bindGatewayRoutesForTest(t, engine, handler)
			gateway := httptest.NewServer(engine)
			defer gateway.Close()
			request, err := http.NewRequest(http.MethodPost, gateway.URL+test.path, bytes.NewBufferString(test.body))
			if err != nil {
				t.Fatal(err)
			}
			request.Header.Set("Authorization", "Bearer gl-client")
			client := &http.Client{Timeout: 5 * time.Second}
			response, err := client.Do(request)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			if response.StatusCode != http.StatusRequestTimeout {
				t.Fatalf("status = %d, want 408", response.StatusCode)
			}
			if got := response.Header.Get("Content-Type"); got != "application/json; charset=utf-8" {
				t.Fatalf("Content-Type = %q, want JSON", got)
			}
			var body struct {
				Error struct {
					Message string `json:"message"`
					Type    string `json:"type"`
					Code    string `json:"code"`
				} `json:"error"`
			}
			if err := json.NewDecoder(response.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if body.Error.Message != "Gateway timeout: upstream request timed out" ||
				body.Error.Type != "timeout_error" || body.Error.Code != "upstream_timeout" {
				t.Fatalf("error = %+v", body.Error)
			}
			select {
			case <-upstreamCanceled:
			case <-time.After(time.Second):
				t.Fatal("upstream request was not canceled")
			}
		})
	}
}
