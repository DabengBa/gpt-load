package gateway

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

func TestSuccessfulPerformanceFaultsPreserveResponseAndReachCredentialBreaker(t *testing.T) {
	feedback := health.Feedback{
		Status: health.FeedbackStatusFaulty,
		Reason: "first_response_slow",
	}
	forwarder := performanceTestStreamForwarder(
		performanceTestStreamResult(feedback),
		performanceTestStreamResult(feedback),
		performanceTestStreamResult(feedback),
	)
	engine, handler, registry, stats := newStatsHandlerTestRuntime(t, forwarder, "sk-one")
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink

	for attempt := 1; attempt <= 3; attempt++ {
		request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(
			`{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"ping"}]}`,
		))
		request.Header.Set("Authorization", "Bearer gl-client")
		response := httptest.NewRecorder()
		engine.ServeHTTP(response, request)
		if response.Code != http.StatusOK || response.Body.String() != "data: {\"ok\":true}\n\n" {
			t.Fatalf("attempt %d response = %d %q, want unchanged successful provider response", attempt, response.Code, response.Body.String())
		}
		if got := stats.Snapshot(1, handler.now()).ConsecutiveFailure; got != uint64(attempt) {
			t.Fatalf("attempt %d consecutive failures = %d, want %d", attempt, got, attempt)
		}
	}
	if got := registry.BlacklistedCredentials(); len(got) != 1 || got[0].ID != 1 {
		t.Fatalf("blacklisted credentials = %#v, want credential 1 at the configured threshold", got)
	}
	events := sink.snapshot()
	if len(events) != 3 {
		t.Fatalf("request log events = %d, want 3", len(events))
	}
	for index, event := range events {
		if len(event.Attempts) != 1 || event.Attempts[0].Feedback.Status != health.FeedbackStatusFaulty {
			t.Fatalf("event %d attempts = %#v, want one faulty provider attempt", index, event.Attempts)
		}
	}

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(
		`{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"next"}]}`,
	))
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)
	if len(forwarder.streamInputs) != 3 {
		t.Fatalf("provider forwards after breaker = %d, want 3; next request must avoid the blacklisted credential", len(forwarder.streamInputs))
	}
}

func TestPerformanceFaultUsesConfiguredThresholdTwoAndExistingRelease(t *testing.T) {
	_, handler, registry, stats := newStatsHandlerTestRuntime(t, &scriptedForwarder{}, "sk-one")
	publishHandlerPolicySettings(t, handler, handler.manager, 1,
		config.Settings{state.SettingBlacklistReleaseSeconds: 60},
		config.Settings{state.SettingBlacklistThreshold: 2},
	)
	now := time.Date(2026, time.July, 17, 12, 0, 0, 0, time.UTC)
	group := handler.manager.Current().Groups[1]
	for count := 1; count <= 2; count++ {
		handler.applyPerformanceFeedbackFailure(group, 1, "e000000000001", http.StatusOK, now)
		if got := stats.Snapshot(1, now).ConsecutiveFailure; got != uint64(count) {
			t.Fatalf("performance failure count = %d, want %d", got, count)
		}
		if got := len(registry.BlacklistedCredentials()); got != count-1 {
			t.Fatalf("blacklisted count after %d faults = %d, want %d", count, got, count-1)
		}
	}
	if released, _ := registry.ReleaseExpiredBlacklists(now.Add(59 * time.Second)); released != 0 {
		t.Fatalf("premature blacklist release = %d, want 0", released)
	}
	if released, _ := registry.ReleaseExpiredBlacklists(now.Add(time.Minute), stats.ClearProblemState); released != 1 {
		t.Fatalf("scheduled blacklist release = %d, want 1", released)
	}
	if candidates := registry.CollectCredentialCandidates([]uint{1}, func(uint) bool { return false }, now.Add(time.Minute)); len(candidates) != 1 {
		t.Fatalf("released credential candidates = %#v, want credential eligible again", candidates)
	}
}

func TestProviderFeedbackRetryMeasurementIsAttemptLocal(t *testing.T) {
	clock := &providerFeedbackTestClock{now: time.Date(2026, time.July, 17, 12, 0, 0, 0, time.UTC)}
	success := performanceTestStreamResult(health.Feedback{})
	success.Usage = usage.Result{State: usage.StateComplete, Tokens: usage.Tokens{Output: 100}}
	forwarder := performanceTestStreamForwarder(
		UpstreamResult{StatusCode: http.StatusUnauthorized, Header: make(http.Header), Body: []byte(`{"error":{"code":"invalid_api_key"}}`)},
		success,
	)
	writePayload := forwarder.onStreamCall
	forwarder.onStreamCall = func(index int, writer http.ResponseWriter) {
		if index == 0 {
			clock.Advance(31 * time.Second)
		} else {
			clock.Advance(2 * time.Second)
		}
		forwarder.streamInputs[index].providerFeedback.observeProviderPayload()
		clock.Advance(time.Second)
		writePayload(index, writer)
	}
	engine, handler, _, _ := newStatsHandlerTestRuntime(t, forwarder, "sk-one", "sk-two")
	handler.now = clock.Now
	handler.requestNow = clock.Now
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(
		`{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"ping"}]}`,
	))
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)
	events := sink.snapshot()
	if response.Code != http.StatusOK || len(events) != 1 || len(events[0].Attempts) != 2 {
		t.Fatalf("retry result = %d, logs = %#v, want successful two-attempt request", response.Code, events)
	}
	first, second := events[0].Attempts[0].Feedback, events[0].Attempts[1].Feedback
	if first.Status != health.FeedbackStatusFaulty || first.Reason != "upstream_failure" || first.FirstResponseMs == nil || *first.FirstResponseMs != 31_000 ||
		second.Status != health.FeedbackStatusNormal || second.FirstResponseMs == nil || *second.FirstResponseMs != 2_000 {
		t.Fatalf("retry feedback = %#v then %#v, want independent 31s failure then 2s successful provider", first, second)
	}
}

func TestActualFailureThenSuccessfulPerformanceFaultShareCredentialStreak(t *testing.T) {
	forwarder := performanceTestStreamForwarder(
		UpstreamResult{StatusCode: http.StatusUnauthorized, Header: make(http.Header), Body: []byte(`{"error":{"code":"invalid_api_key"}}`)},
		performanceTestStreamResult(health.Feedback{Status: health.FeedbackStatusFaulty, Reason: "first_response_slow"}),
	)
	engine, handler, _, stats := newStatsHandlerTestRuntime(t, forwarder, "sk-one")
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	request := func() *httptest.ResponseRecorder {
		request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(
			`{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"ping"}]}`,
		))
		request.Header.Set("Authorization", "Bearer gl-client")
		response := httptest.NewRecorder()
		engine.ServeHTTP(response, request)
		return response
	}
	if response := request(); response.Code != http.StatusUnauthorized {
		t.Fatalf("actual provider failure response = %d, want %d", response.Code, http.StatusUnauthorized)
	}
	if got := stats.Snapshot(1, handler.now()).ConsecutiveFailure; got != 1 {
		t.Fatalf("failure streak after provider error = %d, want 1", got)
	}
	if response := request(); response.Code != http.StatusOK || response.Body.String() != "data: {\"ok\":true}\n\n" {
		t.Fatalf("faulty successful response = %d %q, want unchanged 200 response", response.Code, response.Body.String())
	}
	if got := stats.Snapshot(1, handler.now()).ConsecutiveFailure; got != 2 {
		t.Fatalf("failure streak after faulty success = %d, want 2", got)
	}
	events := sink.snapshot()
	if len(events) != 2 || events[0].Attempts[0].Feedback.Status != health.FeedbackStatusFaulty ||
		events[0].Attempts[0].Feedback.Reason != "upstream_failure" ||
		events[0].Attempts[0].FailureCategory != telemetry.FailureCategoryInvalidKey {
		t.Fatalf("provider failure feedback/category = %#v, want faulty feedback and the unchanged invalid_key category", events)
	}
}

func TestNormalSlowAndUnassessedSuccessResetPerformanceFaultStreak(t *testing.T) {
	statuses := []health.FeedbackStatus{
		health.FeedbackStatusNormal,
		health.FeedbackStatusSlow,
		health.FeedbackStatusUnassessed,
	}
	for _, status := range statuses {
		t.Run(string(status), func(t *testing.T) {
			fault := performanceTestStreamResult(health.Feedback{Status: health.FeedbackStatusFaulty, Reason: "first_response_slow"})
			success := performanceTestStreamResult(health.Feedback{Status: status})
			forwarder := performanceTestStreamForwarder(fault, success)
			engine, handler, _, stats := newStatsHandlerTestRuntime(t, forwarder, "sk-one")
			for range 2 {
				request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(
					`{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"ping"}]}`,
				))
				request.Header.Set("Authorization", "Bearer gl-client")
				response := httptest.NewRecorder()
				engine.ServeHTTP(response, request)
				if response.Code != http.StatusOK {
					t.Fatalf("response = %d, want %d", response.Code, http.StatusOK)
				}
			}
			if got := stats.Snapshot(1, handler.now()).ConsecutiveFailure; got != 0 {
				t.Fatalf("failure streak after %q success = %d, want reset", status, got)
			}
		})
	}
}

func performanceTestStreamResult(feedback health.Feedback) UpstreamResult {
	return UpstreamResult{
		DispatchState: execution.DispatchMaybeSent,
		StatusCode:    http.StatusOK, Header: make(http.Header), Committed: true,
		Stream: StreamObservation{EndReason: StreamEndCleanEOF}, Feedback: feedback,
	}
}

func performanceTestStreamForwarder(results ...UpstreamResult) *scriptedForwarder {
	return &scriptedForwarder{streamResults: results, onStreamCall: func(index int, writer http.ResponseWriter) {
		if index < len(results) && results[index].Committed {
			writer.WriteHeader(http.StatusOK)
			_, _ = writer.Write([]byte("data: {\"ok\":true}\n\n"))
		}
	}}
}

func TestProviderFeedbackMeasurementSubtractsSynchronousDownstreamWait(t *testing.T) {
	started := time.Date(2026, time.July, 17, 12, 0, 0, 0, time.UTC)
	now := started
	measurement := newProviderFeedbackMeasurement(func() time.Time { return now }, started)
	now = started.Add(2 * time.Second)
	measurement.observeProviderPayload()
	now = started.Add(5 * time.Second)
	writeStarted := measurement.beginDownstreamWrite(false)
	now = started.Add(9 * time.Second)
	measurement.endDownstreamWrite(writeStarted)
	now = started.Add(11 * time.Second)
	measurement.complete()
	result := UpstreamResult{
		StatusCode:    http.StatusOK,
		Usage:         usage.Result{Tokens: usage.Tokens{Output: 100}, State: usage.StateComplete},
		DispatchState: execution.DispatchMaybeSent,
	}
	decision := health.Decision{Category: health.FailureCategoryOK, Origin: execution.ErrorOriginUpstream}
	feedback := providerFeedbackForAttempt(result, decision, measurement, true)
	if feedback.Status != health.FeedbackStatusNormal || feedback.FirstResponseMs == nil || *feedback.FirstResponseMs != 2_000 || feedback.TokensPerSecond == nil || *feedback.TokensPerSecond != 20 {
		t.Fatalf("feedback with downstream delay = %#v, want 2s first response and 20 tokens/s", feedback)
	}
}

func TestProviderFeedbackDoesNotInferUnaryOrCanceledStreamPerformance(t *testing.T) {
	started := time.Date(2026, time.July, 17, 12, 0, 0, 0, time.UTC)
	now := started.Add(31 * time.Second)
	measurement := newProviderFeedbackMeasurement(func() time.Time { return now }, started)
	measurement.observeProviderPayload()
	measurement.complete()
	result := UpstreamResult{
		StatusCode: http.StatusOK,
		Usage:      usage.Result{Tokens: usage.Tokens{Output: 500}, State: usage.StateComplete},
	}
	success := health.Decision{Category: health.FailureCategoryOK, Origin: execution.ErrorOriginUpstream}
	if feedback := providerFeedbackForAttempt(result, success, measurement, false); feedback.Status != health.FeedbackStatusUnassessed {
		t.Fatalf("unary feedback = %#v, want unassessed", feedback)
	}
	preclassified := result
	preclassified.DispatchState = execution.DispatchMaybeSent
	preclassified.Feedback = health.Feedback{Status: health.FeedbackStatusFaulty, Reason: "first_response_slow"}
	if feedback := providerFeedbackForAttempt(preclassified, success, measurement, false); feedback.Status != health.FeedbackStatusUnassessed {
		t.Fatalf("preclassified unary feedback = %#v, want unassessed", feedback)
	}
	canceled := health.Decision{Category: health.FailureCategoryDownstreamCancel, Origin: execution.ErrorOriginDownstream}
	if feedback := providerFeedbackForAttempt(result, canceled, measurement, true); feedback.Status != health.FeedbackStatusUnassessed {
		t.Fatalf("canceled stream feedback = %#v, want unassessed", feedback)
	}
	providerFailure := result
	providerFailure.DispatchState = execution.DispatchMaybeSent
	providerFailure.ProviderErrorBeforeCommit = true
	providerDecision := health.Decision{Category: health.FailureCategoryClientError, Origin: execution.ErrorOriginUpstream}
	if feedback := providerFeedbackForAttempt(providerFailure, providerDecision, measurement, true); feedback.Status != health.FeedbackStatusFaulty || feedback.Reason != "upstream_failure" {
		t.Fatalf("upstream provider failure feedback = %#v, want faulty", feedback)
	}
	if feedback := providerFeedbackForAttempt(providerFailure, canceled, measurement, true); feedback.Status != health.FeedbackStatusUnassessed {
		t.Fatalf("provider error marker with downstream cancellation decision = %#v, want unassessed", feedback)
	}
	if isSuccessfulPerformanceFault(UpstreamResult{Feedback: health.Feedback{Status: health.FeedbackStatusFaulty, Reason: "first_response_slow"}}, success) {
		t.Fatal("not-sent performance observation counted as a credential failure")
	}
	for _, test := range []struct {
		name     string
		result   UpstreamResult
		decision health.Decision
	}{
		{name: "local", result: UpstreamResult{DispatchState: execution.DispatchLocal, ProviderErrorBeforeCommit: true}, decision: providerDecision},
		{name: "not sent", result: UpstreamResult{DispatchState: execution.DispatchNotSent, ProviderErrorBeforeCommit: true}, decision: providerDecision},
		{name: "internal", result: UpstreamResult{DispatchState: execution.DispatchMaybeSent, ProviderErrorBeforeCommit: true, Stream: StreamObservation{EndReason: StreamEndUpstreamTerminated}, ExecutionError: &execution.ErrorEvidence{Kind: execution.ErrorKindInternal, OriginHint: execution.ErrorOriginInternal}}, decision: providerDecision},
		{name: "client", result: UpstreamResult{DispatchState: execution.DispatchMaybeSent, ProviderErrorBeforeCommit: true}, decision: health.Decision{Category: health.FailureCategoryClientError, Origin: execution.ErrorOriginClient}},
		{name: "write failure", result: UpstreamResult{DispatchState: execution.DispatchMaybeSent, ProviderErrorBeforeCommit: true, Stream: StreamObservation{EndReason: StreamEndDownstreamWriteFailure}}, decision: canceled},
	} {
		t.Run(test.name, func(t *testing.T) {
			if feedback := providerFeedbackForAttempt(test.result, test.decision, measurement, true); feedback.Status != health.FeedbackStatusUnassessed {
				t.Fatalf("feedback = %#v, want unassessed", feedback)
			}
		})
	}
}

type providerFeedbackTestClock struct {
	mu  sync.Mutex
	now time.Time
}

func (clock *providerFeedbackTestClock) Now() time.Time {
	clock.mu.Lock()
	defer clock.mu.Unlock()
	return clock.now
}

func (clock *providerFeedbackTestClock) Advance(duration time.Duration) {
	clock.mu.Lock()
	clock.now = clock.now.Add(duration)
	clock.mu.Unlock()
}

type providerFeedbackTestWriter struct {
	header              http.Header
	clock               *providerFeedbackTestClock
	downstreamDelay     time.Duration
	providerPayloadOnly bool
}

func (writer *providerFeedbackTestWriter) Header() http.Header { return writer.header }
func (*providerFeedbackTestWriter) WriteHeader(int)            {}
func (writer *providerFeedbackTestWriter) Write(body []byte) (int, error) {
	if !writer.providerPayloadOnly || bytes.Contains(body, []byte("data:")) {
		writer.clock.Advance(writer.downstreamDelay)
	}
	return len(body), nil
}
func (*providerFeedbackTestWriter) FlushError() error { return nil }

func TestProviderFeedbackMeasuresSSEAndExcludesSlowDownstream(t *testing.T) {
	t.Run("normal SSE rate remains active", func(t *testing.T) {
		feedback := runSSEFeedbackTest(t, false, 0, 9, false)
		if feedback.Status != health.FeedbackStatusFaulty || feedback.Reason != "output_rate_faulty" {
			t.Fatalf("SSE feedback = %#v, want faulty at 9 tokens/s", feedback)
		}
	})
	t.Run("fast upstream with slow downstream stays normal", func(t *testing.T) {
		feedback := runSSEFeedbackTest(t, false, 4*time.Second, 60, true)
		if feedback.Status != health.FeedbackStatusNormal || feedback.TokensPerSecond == nil || *feedback.TokensPerSecond != 20 {
			t.Fatalf("SSE feedback = %#v, want normal at 20 upstream tokens/s", feedback)
		}
	})
}

func TestProviderFeedbackBufferedRateEndsBeforeClientRelease(t *testing.T) {
	feedback := runSSEFeedbackTest(t, true, 20*time.Second, 90, true)
	if feedback.Status != health.FeedbackStatusNormal || feedback.TokensPerSecond == nil || *feedback.TokensPerSecond != 30 {
		t.Fatalf("buffered feedback = %#v, want normal at 30 provider tokens/s", feedback)
	}
}

func TestProviderFeedbackMeasuresWebsocketTurnPayload(t *testing.T) {
	handler, engine, _ := websocketTestHandler(t, "http://127.0.0.1:1", channel.CLIProxyAPI)
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	handler.forwarder = websocketScriptForwarder{
		AttemptForwarder: handler.forwarder,
		open: func(context.Context, ForwardInput) (execution.WebsocketSession, execution.WebsocketResult) {
			session := &websocketScriptSession{done: make(chan struct{})}
			session.turn = func(ctx context.Context, _ []byte, emit func(context.Context, []byte) error) execution.WebsocketResult {
				response := []byte(`{"type":"response.completed","response":{"id":"resp_feedback","object":"response","status":"completed","model":"upstream","usage":{"input_tokens":1,"output_tokens":0,"total_tokens":1}}}`)
				if err := emit(ctx, response); err != nil {
					return execution.WebsocketResult{DispatchState: execution.DispatchMaybeSent, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindCanceled}}
				}
				return execution.WebsocketResult{DispatchState: execution.DispatchMaybeSent}
			}
			return session, execution.WebsocketResult{DispatchState: execution.DispatchNotSent}
		},
	}
	server := httptest.NewServer(engine)
	defer server.Close()
	connection := dialGatewayWebsocket(t, server.URL)
	defer connection.Close()
	if err := connection.WriteMessage(websocket.TextMessage, []byte(`{"type":"response.create","model":"public","input":"hello","store":false}`)); err != nil {
		t.Fatalf("write WebSocket turn: %v", err)
	}
	if _, _, err := connection.ReadMessage(); err != nil {
		t.Fatalf("read WebSocket response: %v", err)
	}
	events := waitWebsocketLogs(t, sink, 1)
	if len(events[0].Attempts) != 1 {
		t.Fatalf("WebSocket attempts = %#v, want one attempt", events[0].Attempts)
	}
	feedback := events[0].Attempts[0].Feedback
	if feedback.Status != health.FeedbackStatusNormal || feedback.FirstResponseMs == nil || feedback.TokensPerSecond != nil {
		t.Fatalf("WebSocket feedback = %#v, want measured first payload and no zero-token rate", feedback)
	}
}

func runSSEFeedbackTest(
	t *testing.T,
	buffered bool,
	downstreamDelay time.Duration,
	outputTokens int64,
	blockDownstream bool,
) health.Feedback {
	t.Helper()
	clock := &providerFeedbackTestClock{now: time.Date(2026, time.July, 17, 12, 0, 0, 0, time.UTC)}
	input := executionForwardInput()
	input.ClientProtocol = protocol.OpenAICompletions
	input.ObserveUsage = true
	input.BufferedStream = buffered
	input.providerFeedback = newProviderFeedbackMeasurement(clock.Now, clock.Now())
	var streamEventError error
	executor := fakeExecutionExecutor{stream: func(
		_ context.Context,
		_ execution.AttemptSpec,
		sink execution.StreamSink,
	) execution.StreamResult {
		if err := sink(execution.StreamEvent{
			Sequence: 1, Kind: execution.StreamEventReady, StatusCode: http.StatusOK,
			Header: http.Header{"Content-Type": {"text/event-stream"}},
		}); err != nil {
			streamEventError = err
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent}
		}
		if err := sink(execution.StreamEvent{Sequence: 2, Kind: execution.StreamEventData, Data: []byte(": upstream heartbeat\n\n")}); err != nil {
			streamEventError = err
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent}
		}
		clock.Advance(2 * time.Second)
		if err := sink(execution.StreamEvent{
			Sequence: 2, Kind: execution.StreamEventData,
			Data: []byte("data: {\"id\":\"chat_1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"one\"},\"finish_reason\":null}]}\n\n"),
		}); err != nil {
			streamEventError = err
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent}
		}
		clock.Advance(3 * time.Second)
		if err := sink(execution.StreamEvent{
			Sequence: 3, Kind: execution.StreamEventData,
			Data: []byte("data: {\"id\":\"chat_1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"),
		}); err != nil {
			streamEventError = err
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent}
		}
		if err := sink(execution.StreamEvent{Sequence: 4, Kind: execution.StreamEventData, Data: []byte("data: [DONE]\n\n")}); err != nil {
			streamEventError = err
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent}
		}
		return execution.StreamResult{
			DispatchState: execution.DispatchMaybeSent, ResponseStarted: true,
			StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}},
			Usage: &execution.UsageEvidence{Normalized: usage.Result{
				State: usage.StateComplete, Tokens: usage.Tokens{Output: outputTokens},
			}},
		}
	}}
	baseWriter := &providerFeedbackTestWriter{
		header: make(http.Header), clock: clock, downstreamDelay: downstreamDelay,
		providerPayloadOnly: buffered,
	}
	var writer http.ResponseWriter = baseWriter
	var blockingWriter *providerFeedbackBlockingWriter
	if blockDownstream {
		baseWriter.downstreamDelay = 0
		blockingWriter = &providerFeedbackBlockingWriter{
			providerFeedbackTestWriter: *baseWriter,
			entered:                    make(chan struct{}),
			release:                    make(chan struct{}),
		}
		t.Cleanup(func() { blockingWriter.releaseOnce.Do(func() { close(blockingWriter.release) }) })
		writer = blockingWriter
	}
	forwarder := NewExecutionForwarder(executor)
	forwarder.heartbeatInterval = time.Hour
	var result UpstreamResult
	if blockingWriter == nil {
		result = forwarder.ForwardStream(t.Context(), input, writer)
	} else {
		resultChannel := make(chan UpstreamResult, 1)
		go func() { resultChannel <- forwarder.ForwardStream(t.Context(), input, writer) }()
		select {
		case <-blockingWriter.entered:
		case <-time.After(5 * time.Second):
			t.Fatal("downstream writer did not block on provider payload")
		}
		clock.Advance(downstreamDelay)
		blockingWriter.releaseOnce.Do(func() { close(blockingWriter.release) })
		select {
		case result = <-resultChannel:
		case <-time.After(5 * time.Second):
			t.Fatal("stream forward did not finish after releasing downstream writer")
		}
	}
	if streamEventError != nil {
		t.Fatalf("executor stream event error = %v", streamEventError)
	}
	if result.Err != nil || result.Stream.EndReason != StreamEndCleanEOF {
		t.Fatalf("stream result = %#v, want clean successful stream", result)
	}
	decision := health.Decision{
		Category: health.FailureCategoryOK,
		Origin:   execution.ErrorOriginUpstream,
	}
	feedback := providerFeedbackForAttempt(result, decision, input.providerFeedback, true)
	if feedback.FirstResponseMs == nil || *feedback.FirstResponseMs != 2_000 {
		t.Fatalf("provider first payload = %#v, want 2000ms excluding headers and heartbeat", feedback.FirstResponseMs)
	}
	return feedback
}

type providerFeedbackBlockingWriter struct {
	providerFeedbackTestWriter
	entered     chan struct{}
	release     chan struct{}
	releaseOnce sync.Once
	blockOnce   sync.Once
}

func (writer *providerFeedbackBlockingWriter) Write(body []byte) (int, error) {
	if bytes.Contains(body, []byte("data:")) {
		writer.blockOnce.Do(func() {
			close(writer.entered)
			<-writer.release
		})
	}
	return writer.providerFeedbackTestWriter.Write(body)
}
