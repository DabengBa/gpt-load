package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/platform/redact"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/testutil/encryptiontest"
)

func TestHandlerNativeResponsesBufferedFailureRetriesCandidate(t *testing.T) {
	const (
		failedEvent  = "event: response.failed\ndata: {\"type\":\"response.failed\",\"error\":{\"code\":\"internal_error\",\"message\":\"capacity exhausted\"},\"response\":{}}\n\n"
		createdEvent = "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_2\",\"status\":\"in_progress\",\"output\":[]}}\n\n"
		successEvent = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"recovered\"}\n\n" +
			"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_2\",\"status\":\"completed\",\"output\":[]}}\n\n"
	)

	for _, test := range []struct {
		name       string
		firstEvent string
		wantRetry  bool
		budget     int
	}{
		{name: "response failed", firstEvent: failedEvent, wantRetry: true, budget: 2},
		{name: "missing terminal EOF", firstEvent: createdEvent, wantRetry: true, budget: 2},
		{name: "budget one", firstEvent: failedEvent, budget: 1},
		{name: "budget two", firstEvent: failedEvent, wantRetry: true, budget: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			calls := make([]execution.AttemptSpec, 0, 2)
			executor := fakeExecutionExecutor{stream: func(
				_ context.Context,
				spec execution.AttemptSpec,
				sink execution.StreamSink,
			) execution.StreamResult {
				calls = append(calls, spec.Clone())
				if spec.ClientProtocol != protocol.OpenAIResponses ||
					spec.Operation != execution.OperationResponsesCreate ||
					spec.RouteMode != execution.RouteNative ||
					spec.Method != http.MethodPost || spec.Path != "/v1/responses" ||
					!bytes.Contains(spec.Body, []byte(`"store":false`)) {
					t.Fatalf("attempt spec = %#v", spec)
				}
				if err := sink(execution.StreamEvent{
					Sequence: 1, Kind: execution.StreamEventReady,
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": {"text/event-stream"}},
				}); err != nil {
					t.Fatalf("ready sink: %v", err)
				}
				event := successEvent
				if len(calls) == 1 {
					event = test.firstEvent
				}
				if err := sink(execution.StreamEvent{
					Sequence: 2, Kind: execution.StreamEventData, Data: []byte(event),
				}); err != nil {
					t.Fatalf("data sink: %v", err)
				}
				return execution.StreamResult{
					DispatchState: execution.DispatchMaybeSent, ResponseStarted: true,
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": {"text/event-stream"}},
				}
			}}

			forwarder := &capturingExecutionForwarder{delegate: NewExecutionForwarder(executor)}
			engine, _ := newDialectGatewayEngineWithSystemSettings(
				t, protocol.OpenAIResponses, "gpt-4o", dialect.NewSet(dialect.NewOpenAIResponses()),
				forwarder, config.Settings{state.SettingRetryCount: test.budget},
				dialectGatewayGroup{id: 1, name: "responses-first", channelID: channel.OpenAI, params: json.RawMessage(`{"base_url":"https://provider.example/v1"}`), apiKeys: []string{"sk-first"}},
				dialectGatewayGroup{id: 2, name: "responses-second", channelID: channel.OpenAI, params: json.RawMessage(`{"base_url":"https://provider.example/v1"}`), apiKeys: []string{"sk-second"}},
			)

			request := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
				`{"model":"gpt-4o","input":"hello","stream":true,"store":false}`,
			))
			request.Header.Set("Authorization", "Bearer gl-client")
			recorder := httptest.NewRecorder()
			engine.ServeHTTP(recorder, request)

			wantCalls := 1
			if test.wantRetry {
				wantCalls = 2
			}
			if len(calls) != wantCalls {
				t.Fatalf("attempt count = %d, want %d", len(calls), wantCalls)
			}
			if len(forwarder.inputs) != wantCalls || len(forwarder.results) != wantCalls {
				t.Fatalf("captured attempts = inputs:%d results:%d, want %d", len(forwarder.inputs), len(forwarder.results), wantCalls)
			}
			for index, input := range forwarder.inputs {
				if input.ClientProtocol != protocol.OpenAIResponses ||
					input.Operation != execution.OperationResponsesCreate ||
					input.RouteMode != execution.RouteNative ||
					input.Request.Method != http.MethodPost || input.Request.Path != "/v1/responses" ||
					!bufferedStreamReplayEligible(input.Request, input.Dialect) {
					t.Fatalf("forward input[%d] = %#v, want native replay-eligible Responses attempt", index, input)
				}
			}
			if test.wantRetry {
				first := forwarder.results[0]
				second := forwarder.results[1]
				if !first.HTTPCommitted || first.PayloadReleased || first.ClientVisibleBytes != int64(len(bufferedStreamHeartbeat)) {
					t.Fatalf("first buffered result = %#v, want heartbeat-only unreleased attempt", first)
				}
				if !second.HTTPCommitted || !second.PayloadReleased || second.ClientVisibleBytes <= int64(len(bufferedStreamHeartbeat)) {
					t.Fatalf("second buffered result = %#v, want released payload", second)
				}
			}
			if test.wantRetry && calls[0].Credential.ID == calls[1].Credential.ID {
				t.Fatalf("credentials = %d/%d, want a different candidate", calls[0].Credential.ID, calls[1].Credential.ID)
			}
			if test.wantRetry {
				body := recorder.Body.String()
				if body != bufferedStreamHeartbeat+successEvent || strings.Contains(body, "capacity exhausted") || strings.Contains(body, "buffered_stream_failed") {
					t.Fatalf("downstream body = %q, want only first heartbeat and second payload", body)
				}
			} else if !strings.Contains(recorder.Body.String(), "buffered_stream_failed") || strings.Contains(recorder.Body.String(), "recovered") {
				t.Fatalf("budget-limited body = %q, want buffered failure without a candidate payload", recorder.Body.String())
			}
		})
	}

	// A valid provider-incomplete terminal is a semantic result, not a transport
	// failure, and therefore must not use the buffered candidate retry gate.
	calls := 0
	executor := fakeExecutionExecutor{stream: func(
		_ context.Context,
		_ execution.AttemptSpec,
		sink execution.StreamSink,
	) execution.StreamResult {
		calls++
		if err := sink(execution.StreamEvent{Sequence: 1, Kind: execution.StreamEventReady,
			StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}}); err != nil {
			t.Fatalf("ready sink: %v", err)
		}
		if err := sink(execution.StreamEvent{Sequence: 2, Kind: execution.StreamEventData, Data: []byte(
			"event: response.incomplete\ndata: {\"type\":\"response.incomplete\",\"response\":{}}\n\n",
		)}); err != nil {
			t.Fatalf("incomplete sink: %v", err)
		}
		return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true,
			StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}}
	}}
	engine, _ := newDialectGatewayEngineWithSystemSettings(
		t, protocol.OpenAIResponses, "gpt-4o", dialect.NewSet(dialect.NewOpenAIResponses()),
		NewExecutionForwarder(executor), config.Settings{state.SettingRetryCount: 2},
		dialectGatewayGroup{id: 1, name: "responses-first", channelID: channel.OpenAI, params: json.RawMessage(`{"base_url":"https://provider.example/v1"}`), apiKeys: []string{"sk-first"}},
		dialectGatewayGroup{id: 2, name: "responses-second", channelID: channel.OpenAI, params: json.RawMessage(`{"base_url":"https://provider.example/v1"}`), apiKeys: []string{"sk-second"}},
	)
	request := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"gpt-4o","input":"hello","stream":true,"store":false}`,
	))
	request.Header.Set("Authorization", "Bearer gl-client")
	engine.ServeHTTP(httptest.NewRecorder(), request)
	if calls != 1 {
		t.Fatalf("incomplete attempts = %d, want one", calls)
	}
}

type capturingExecutionForwarder struct {
	delegate *ExecutionForwarder
	inputs   []ForwardInput
	results  []UpstreamResult
}

func (forwarder *capturingExecutionForwarder) Forward(ctx context.Context, input ForwardInput) UpstreamResult {
	forwarder.inputs = append(forwarder.inputs, input)
	result := forwarder.delegate.Forward(ctx, input)
	forwarder.results = append(forwarder.results, result)
	return result
}

func (forwarder *capturingExecutionForwarder) ForwardStream(
	ctx context.Context,
	input ForwardInput,
	downstream http.ResponseWriter,
) UpstreamResult {
	forwarder.inputs = append(forwarder.inputs, input)
	result := forwarder.delegate.ForwardStream(ctx, input, downstream)
	forwarder.results = append(forwarder.results, result)
	return result
}

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
				config.Settings{},
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
		streamGatewayGroup{id: 1, name: "forbidden-a", upstreamURL: upstream.URL, apiKey: "sk-a"},
		streamGatewayGroup{id: 2, name: "forbidden-b", upstreamURL: upstream.URL, apiKey: "sk-b"},
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

// newBufferedFallbackRuntime 构建两个 buffered 候选 + 真实执行层的 gateway，指向传入上游。
// groupSettings 会合并进每个分组的 Settings（例如自定义 header 规则）并返回请求日志 sink。
func newBufferedFallbackRuntime(
	t *testing.T,
	upstreamURL string,
	groupSettings config.Settings,
) (*gin.Engine, *recordingRequestLogSink) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	keyService := encryptiontest.Service(t, "buffered-fallback-summary-master-key")
	manager := state.NewManager()
	baseURL := testUpstreamBaseURL(upstreamURL, protocol.OpenAICompletions)
	channelID, params := testChannelConfig(t, protocol.OpenAICompletions, baseURL)
	groups := make([]state.GroupConfig, 0, 2)
	credentials := make([]state.CredentialConfig, 0, 2)
	entries := make([]state.CredentialEntry, 0, 2)
	for index := 1; index <= 2; index++ {
		id := uint(index)
		settings := config.Settings{}
		for key, value := range groupSettings {
			settings[key] = value
		}
		groups = append(groups, state.GroupConfig{
			ConnectionType: "api_key", ID: id, Name: fmt.Sprintf("openai-%d", index),
			ChannelID: channelID, Params: params,
			Models:   []state.ModelConfig{{ID: "gpt-4o"}},
			Settings: settings,
			Enabled:  true,
		})
		credentials = append(credentials, testCredentialConfig(id, id))
		entries = append(entries, testCredentialEntry(t, keyService, id, id, fmt.Sprintf("sk-%d", index)))
	}
	if _, err := manager.Publish(state.CompileInput{
		SystemSettings:  config.Settings{state.SettingRetryCount: 3},
		ChannelRegistry: channel.NewRegistry(), Groups: groups, Credentials: credentials,
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: keyService.Hash("gl-client"), Status: state.AccessKeyStatusActive,
		}},
	}); err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	registry := state.NewCredentialRegistry()
	if err := registry.ReplaceCredentials(entries); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}
	sink := &recordingRequestLogSink{}
	handler := NewHandler(
		manager, registry, keyService, newTestExecutionForwarder(t),
		dialect.NewSet(dialect.NewOpenAI()), health.NewStatsStore(), health.NewMutationCoordinator(),
		nil, nil, nil,
	)
	handler.requestLogSink = sink
	handler.newRequestID = func() (string, error) { return fixedRequestID, nil }
	handler.requestNow = newSteppingRequestClock()
	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)
	return engine, sink
}

func performBufferedFallbackRequest(t *testing.T, engine *gin.Engine) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4o","stream":true}`),
	)
	request.Header.Set("Authorization", "Bearer gl-client")
	recorder := httptest.NewRecorder()
	engine.ServeHTTP(recorder, request)
	return recorder
}

// 真实执行层 + 真实上游：buffered 请求在两个候选都被上游 429 拒绝、重试预算仍有剩余但已无
// 候选可换时，请求级摘要必须保留供应商短摘要，而不是只剩「没有候选」的泛化文案。
func TestHandlerBufferedCandidateExhaustionKeepsProviderSummary(t *testing.T) {
	const providerSummary = "Rate limit reached for gpt-4o in organization org-abc on tokens per min."
	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(writer, `{"error":{"message":"`+providerSummary+`","type":"rate_limit_exceeded","code":"rate_limit_exceeded"}}`)
	}))
	defer upstream.Close()

	engine, sink := newBufferedFallbackRuntime(t, upstream.URL, nil)
	recorder := performBufferedFallbackRequest(t, engine)

	if !strings.Contains(recorder.Body.String(), `"code":"no_available_candidate"`) {
		t.Fatalf("downstream body = %q, want the candidate fallback envelope", recorder.Body.String())
	}
	events := sink.snapshot()
	if len(events) != 1 {
		t.Fatalf("request log events = %d, want one", len(events))
	}
	event := events[0]
	if len(event.Attempts) != 2 {
		t.Fatalf("logged attempts = %d, want two", len(event.Attempts))
	}
	if event.ErrorSummary != providerSummary {
		t.Fatalf("request summary = %q, want the provider summary %q", event.ErrorSummary, providerSummary)
	}
	if event.ErrorCode != "no_available_candidate" {
		t.Fatalf("request error code = %q, want %q", event.ErrorCode, "no_available_candidate")
	}
	for index, attempt := range event.Attempts {
		if attempt.ErrorSummary != providerSummary {
			t.Fatalf("attempt[%d] summary = %q, want the provider summary", index, attempt.ErrorSummary)
		}
	}
	for _, surface := range []string{event.ErrorSummary, event.Attempts[0].ErrorSummary} {
		for _, forbidden := range []string{`{"error"`, `"code":"rate_limit_exceeded"`, `"type":"rate_limit_exceeded"`} {
			if strings.Contains(surface, forbidden) {
				t.Fatalf("summary %q retained raw provider JSON %q", surface, forbidden)
			}
		}
	}
}

// 真实执行层 + 真实上游：分组自定义 header 规则会被发送到上游；上游在供应商 message 里回显
// 该值时，attempt 级和请求级摘要都必须先按同一次选中的 HeaderRules secret 脱敏。
func TestHandlerBufferedCandidateExhaustionRedactsGroupHeaderRuleSecret(t *testing.T) {
	const tenantSecret = "tenant-alpha-9f3a2c"
	var applied atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		echoed := request.Header.Get("X-Tenant-Key")
		if echoed == tenantSecret {
			applied.Add(1)
		}
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(writer, `{"error":{"message":"tenant `+echoed+` was rate limited","type":"rate_limit_exceeded","code":"rate_limit_exceeded"}}`)
	}))
	defer upstream.Close()

	engine, sink := newBufferedFallbackRuntime(t, upstream.URL, config.Settings{
		state.SettingHeaderRules: map[string]any{
			"set": map[string]any{"X-Tenant-Key": tenantSecret},
		},
	})
	recorder := performBufferedFallbackRequest(t, engine)

	if !strings.Contains(recorder.Body.String(), `"code":"no_available_candidate"`) {
		t.Fatalf("downstream body = %q, want the candidate fallback envelope", recorder.Body.String())
	}
	if applied.Load() != 2 {
		t.Fatalf("upstream received the group header rule %d times, want 2", applied.Load())
	}
	events := sink.snapshot()
	if len(events) != 1 {
		t.Fatalf("request log events = %d, want one", len(events))
	}
	event := events[0]
	if len(event.Attempts) != 2 {
		t.Fatalf("logged attempts = %d, want two", len(event.Attempts))
	}
	surfaces := map[string]string{
		"request": event.ErrorSummary,
	}
	for index, attempt := range event.Attempts {
		surfaces[fmt.Sprintf("attempt[%d]", index)] = attempt.ErrorSummary
	}
	for label, surface := range surfaces {
		if strings.Contains(surface, tenantSecret) {
			t.Fatalf("%s summary leaked the group header rule secret: %q", label, surface)
		}
		if !strings.Contains(surface, redact.Placeholder) {
			t.Fatalf("%s summary = %q, want the redaction placeholder", label, surface)
		}
		if !strings.Contains(surface, "was rate limited") {
			t.Fatalf("%s summary = %q, want the provider message retained", label, surface)
		}
	}
}
