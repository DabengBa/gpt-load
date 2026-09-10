package gateway

import (
	"context"
	"errors"
	"io"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/state"
	"gpt-load/internal/telemetry"
)

// 上游以非 2xx 结束、下游只收到心跳时，attempt 既没有流级终止观测也不是正常结束。
// 旧路径把它记成 StreamEndCleanEOF，证据因此被 decisionEvidence 丢掉：请求日志记成功、
// 403 落进「无证据」分支终局、不换候选也不计凭据失败。
func TestHandlerBufferedUpstreamFailureKeepsEvidenceAndSwitchesCandidate(t *testing.T) {
	tests := []struct {
		name          string
		status        int
		hint          execution.FailureHint
		scope         execution.ErrorScope
		code          string
		wantRule      string
		wantErrorCode string
	}{
		{
			name:          "forbidden with invalid key body",
			status:        http.StatusForbidden,
			hint:          execution.FailureHintInvalidCredential,
			scope:         execution.ErrorScopeCredential,
			code:          "INVALID_API_KEY",
			wantRule:      "auth.invalid_credential",
			wantErrorCode: "upstream_invalid_key",
		},
		{
			name:          "service unavailable keeps the buffered replay",
			status:        http.StatusServiceUnavailable,
			hint:          execution.FailureHintHostError,
			scope:         execution.ErrorScopeGroup,
			code:          "",
			wantRule:      "buffered_stream.retry_before_release_upstream_status",
			wantErrorCode: "upstream_host_error",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			forwarder := &bufferedUpstreamFailureForwarder{
				status: test.status, hint: test.hint, scope: test.scope, code: test.code,
			}
			sink := &recordingRequestLogSink{}
			engine, handler, manager, _ := newRequestLogHandlerTestRuntime(
				t, forwarder, unlimitedAccessKeyRPMLimiter{}, sink, "sk-first", "sk-second",
			)
			publishHandlerPolicySettings(
				t, handler, manager, 2,
				config.Settings{state.SettingRetryCount: 2},
				config.Settings{state.SettingBufferedStream: true},
			)
			handler.newRandom = func() *rand.Rand { return rand.New(zeroSource{}) }

			request := httptest.NewRequest(
				http.MethodPost,
				"/v1/chat/completions",
				strings.NewReader(`{"model":"gpt-4o","stream":true}`),
			)
			request.Header.Set("Authorization", "Bearer gl-client")
			engine.ServeHTTP(httptest.NewRecorder(), request)

			if forwarder.calls != 2 {
				t.Fatalf("attempts = %d, want one switch inside the attempt budget", forwarder.calls)
			}
			if forwarder.groups[0] == forwarder.groups[1] {
				t.Fatalf("attempts = %v, want a second candidate", forwarder.groups)
			}

			events := sink.snapshot()
			if len(events) != 1 {
				t.Fatalf("request log events = %d, want one", len(events))
			}
			event := events[0]
			if event.Status == telemetry.RequestStatusSuccess {
				t.Fatalf("request log status = %q, want a failure: %#v", event.Status, event)
			}
			if event.ErrorCode == "" {
				t.Fatalf("request log error code is empty: %#v", event)
			}
			if len(event.Attempts) != 2 {
				t.Fatalf("logged attempts = %d, want two", len(event.Attempts))
			}
			first := event.Attempts[0]
			if first.RuleID != test.wantRule || !first.WillRetry {
				t.Fatalf("first attempt = %#v, want rule %q with a retry", first, test.wantRule)
			}
			if first.ErrorCode != test.wantErrorCode {
				t.Fatalf("first attempt error code = %q, want %q", first.ErrorCode, test.wantErrorCode)
			}
		})
	}
}

// 真实执行层 + 真实上游：buffered 请求在上游返回 403（凭据被拒）后必须换到下一个候选，
// 并把第二个候选的内容交给客户端，第一个候选的 payload 不得泄漏。
func TestBufferedStreamRecoversFromUpstreamForbiddenBeforeRelease(t *testing.T) {
	const successSSE = "data: {\"id\":\"ok\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"ok\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"

	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			writer.Header().Set("Content-Type", "application/json")
			writer.WriteHeader(http.StatusForbidden)
			_, _ = io.WriteString(writer, `{"code":"INVALID_API_KEY","message":"the key is rejected"}`)
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, successSSE)
		writer.(http.Flusher).Flush()
	}))
	defer upstream.Close()

	engine, _ := newStreamingGatewayEngine(t,
		streamGatewayGroup{id: 1, name: "forbidden-a", upstreamURL: upstream.URL, apiKey: "sk-a", bufferedStream: true},
		streamGatewayGroup{id: 2, name: "forbidden-b", upstreamURL: upstream.URL, apiKey: "sk-b", bufferedStream: true},
	)
	recorder := performStreamingRequest(engine)

	if calls.Load() != 2 {
		t.Fatalf(
			"upstream calls = %d, want two after a forbidden answer; body=%q",
			calls.Load(), recorder.Body.String(),
		)
	}
	if body := recorder.Body.String(); !strings.Contains(body, `"content":"ok"`) {
		t.Fatalf("body = %q, want the second candidate's stream", body)
	}
}

type bufferedUpstreamFailureForwarder struct {
	status int
	hint   execution.FailureHint
	scope  execution.ErrorScope
	code   string
	calls  int
	groups []uint
}

func (*bufferedUpstreamFailureForwarder) Forward(context.Context, ForwardInput) UpstreamResult {
	return UpstreamResult{Err: errors.New("unexpected unary forward")}
}

func (forwarder *bufferedUpstreamFailureForwarder) ForwardStream(
	_ context.Context,
	input ForwardInput,
	_ http.ResponseWriter,
) UpstreamResult {
	forwarder.calls++
	forwarder.groups = append(forwarder.groups, input.Group.ID)
	summary := "upstream rejected the attempt"
	return UpstreamResult{
		StatusCode:         forwarder.status,
		RequestWritten:     true,
		DispatchState:      execution.DispatchMaybeSent,
		Committed:          true,
		HTTPCommitted:      true,
		BufferedStream:     true,
		ClientVisibleBytes: int64(len(bufferedStreamHeartbeat)),
		ExecutionError: &execution.ErrorEvidence{
			Kind:       execution.ErrorKindHTTP,
			OriginHint: execution.ErrorOriginUpstream,
			ScopeHint:  forwarder.scope,
			Hint:       forwarder.hint,
			StatusCode: forwarder.status,
			Code:       forwarder.code,
			Summary:    summary,
		},
		ErrorSummary: summary,
	}
}
