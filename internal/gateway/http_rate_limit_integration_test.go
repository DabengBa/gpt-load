package gateway

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
)

func TestConvertedHTTP429RetriesNextGatewayCandidate(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Errorf("path=%q", r.URL.Path)
		}
		if calls.Add(1) == 1 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"error":{"type":"rate_limit_error","code":"rate_limit_exceeded","message":"slow down"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, bufferedResponsesFramingSuccessFixture)
	}))
	defer upstream.Close()
	_, params := testChannelConfig(t, protocol.OpenAIResponses, upstream.URL)
	forwarder := &capturingExecutionForwarder{delegate: newConcreteTestExecutionForwarder(t)}
	sink := &recordingRequestLogSink{}
	engine, _ := newDialectGatewayEngineWithRequestLog(t, protocol.Anthropic, "gpt-4o", dialect.NewSet(dialect.NewAnthropic()), forwarder,
		config.Settings{state.SettingRetryCount: 2}, sink,
		dialectGatewayGroup{id: 1, name: "first", channelID: channel.OpenAI, params: params, apiKeys: []string{"sk-first"}},
		dialectGatewayGroup{id: 2, name: "second", channelID: channel.OpenAI, params: params, apiKeys: []string{"sk-second"}},
	)
	request := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"gpt-4o","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"hello"}]}`))
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)
	if calls.Load() != 2 || len(forwarder.results) != 2 {
		t.Fatalf("calls=%d attempts=%d body=%s", calls.Load(), len(forwarder.results), response.Body.String())
	}
	if response.Code != 200 || !strings.Contains(response.Body.String(), "message_stop") || strings.Contains(response.Body.String(), "slow down") {
		t.Errorf("response=%d %s", response.Code, response.Body.String())
	}
	if forwarder.results[0].PayloadReleased || !forwarder.results[1].PayloadReleased {
		t.Errorf("release boundary=%+v", forwarder.results)
	}
	events := sink.snapshot()
	if len(events) != 1 || len(events[0].Attempts) != 2 {
		t.Fatalf("events=%+v", events)
	}
	if events[0].Attempts[0].GroupID == events[0].Attempts[1].GroupID || events[0].Attempts[0].CredentialID == events[0].Attempts[1].CredentialID {
		t.Errorf("retry did not advance candidate: %+v", events[0].Attempts)
	}
}
