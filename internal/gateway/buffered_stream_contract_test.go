package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
)

func TestBufferedStreamChatRequiresFinishReasonBeforeDone(t *testing.T) {
	observer := newStreamEventObserver(dialect.NewOpenAI(), nil, true)
	if _, err := observer.classify(dialect.StreamEvent{Payload: []byte(`{"id":"chat_1","choices":[{"index":0,"finish_reason":null}]}`)}, false); err != nil {
		t.Fatal(err)
	}
	if _, err := observer.classify(dialect.StreamEvent{Payload: []byte(`[DONE]`)}, false); err == nil {
		t.Fatal("Chat [DONE] accepted after a choice without a finish_reason")
	}
}

func TestBufferedStreamChatRejectsInvalidFinishReason(t *testing.T) {
	observer := newStreamEventObserver(dialect.NewOpenAI(), nil, true)
	if _, err := observer.classify(dialect.StreamEvent{Payload: []byte(`{"id":"chat_1","choices":[{"index":0,"finish_reason":"unknown"}]}`)}, false); err == nil {
		t.Fatal("Chat accepted an invalid finish_reason")
	}
}

func TestBufferedStreamChatRequiresEveryChoiceToCloseBeforeDone(t *testing.T) {
	observer := newStreamEventObserver(dialect.NewOpenAI(), nil, true)
	if _, err := observer.classify(dialect.StreamEvent{Payload: []byte(`{"id":"chat_1","choices":[{"index":0,"finish_reason":"stop"},{"index":1,"finish_reason":null}]}`)}, false); err != nil {
		t.Fatal(err)
	}
	if _, err := observer.classify(dialect.StreamEvent{Payload: []byte(`[DONE]`)}, false); err == nil {
		t.Fatal("Chat [DONE] accepted while choice 1 was still open")
	}
}

func TestBufferedStreamRejectsEventsAfterTerminal(t *testing.T) {
	observer := newStreamEventObserver(dialect.NewAnthropic(), nil, true)
	if _, err := observer.classify(dialect.StreamEvent{Name: "message_stop", Payload: []byte(`{"type":"message_stop"}`)}, false); err != nil {
		t.Fatal(err)
	}
	if _, err := observer.classify(dialect.StreamEvent{Name: "ping", Payload: []byte(`{"type":"ping"}`)}, false); err == nil {
		t.Fatal("event after Anthropic message_stop was accepted")
	}
}

func TestBufferedStreamResponsesDefaultStoreIsNotReplayEligible(t *testing.T) {
	request := &dialect.ParsedRequest{
		Method: http.MethodPost,
		Path:   "/v1/responses",
		Body:   []byte(`{"model":"gpt-5","input":"hello"}`),
	}
	if bufferedStreamReplayEligible(request, dialect.NewOpenAIResponses()) {
		t.Fatal("Responses request with omitted store was replay-eligible")
	}
	for _, body := range []string{
		`{"model":"gpt-5","input":"hello","previous_response_id":"resp_1"}`,
		`{"model":"gpt-5","input":"hello","tools":[{"type":"computer_use_preview"}]}`,
		`{"model":"gpt-5","input":"hello","background":true,"store":false}`,
		`{"model":"gpt-5","input":"hello","prompt":{"id":"pmpt_1"},"store":false}`,
		`{"model":"gpt-5","input":[{"type":"item_reference","id":"item_1"}],"store":false}`,
		`{"model":"gpt-5","input":"hello","tools":[{"type":"file_search","vector_store_ids":["vs_1"]}],"store":false}`,
		`{"model":"gpt-5","input":"hello","prompt_cache_key":"cache_1","store":false}`,
	} {
		request.Body = []byte(body)
		if bufferedStreamReplayEligible(request, dialect.NewOpenAIResponses()) {
			t.Fatalf("Responses request %s was replay-eligible", body)
		}
	}
}

func TestBufferedStreamDoesNotCommitHeadersTwiceAcrossAttempts(t *testing.T) {
	var calls atomic.Int32
	executor := fakeExecutionExecutor{stream: func(
		_ context.Context,
		_ execution.AttemptSpec,
		sink execution.StreamSink,
	) execution.StreamResult {
		call := calls.Add(1)
		if err := sink(execution.StreamEvent{Sequence: 1, Kind: execution.StreamEventReady, StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}}); err != nil {
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindInternal, Summary: err.Error()}}
		}
		if call == 1 {
			if err := sink(execution.StreamEvent{Sequence: 2, Kind: execution.StreamEventData, Data: []byte(`data: {"choices":[{"index":0,"finish_reason":null}]}` + "\n\n")}); err != nil {
				return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindInternal, Summary: err.Error()}}
			}
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindInternal, Summary: "partial"}}
		}
		for _, data := range [][]byte{
			[]byte(`data: {"choices":[{"index":0,"finish_reason":"stop"}]}` + "\n\n"),
			[]byte("data: [DONE]\n\n"),
		} {
			if err := sink(execution.StreamEvent{Sequence: 3, Kind: execution.StreamEventData, Data: data}); err != nil {
				return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindInternal, Summary: err.Error()}}
			}
		}
		return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
	}}
	input := executionForwardInput()
	input.BufferedStream = true
	input.ClientProtocol = protocol.OpenAICompletions
	ctx := context.WithValue(context.Background(), bufferedStreamSessionContextKey{}, &bufferedStreamSession{})
	writer := &countingHeaderResponseWriter{ResponseWriter: httptest.NewRecorder()}
	first := NewExecutionForwarder(executor).ForwardStream(ctx, input, writer)
	if first.Stream.EndReason == StreamEndCleanEOF {
		t.Fatal("first partial attempt unexpectedly succeeded")
	}
	second := NewExecutionForwarder(executor).ForwardStream(ctx, input, writer)
	_ = second
	if writer.headers != 1 {
		t.Fatalf("WriteHeader calls = %d, want one owner commit", writer.headers)
	}
}

type countingHeaderResponseWriter struct {
	http.ResponseWriter
	headers int
}

func (writer *countingHeaderResponseWriter) WriteHeader(status int) {
	writer.headers++
	writer.ResponseWriter.WriteHeader(status)
}

func TestBufferedStreamTerminalErrorsKeepObservedResponsesID(t *testing.T) {
	for _, kind := range []execution.ErrorKind{execution.ErrorKindTimeout, execution.ErrorKindProvider, execution.ErrorKindHTTP, execution.ErrorKindInternal} {
		t.Run(fmt.Sprint(kind), func(t *testing.T) {
			executor := fakeExecutionExecutor{stream: func(_ context.Context, _ execution.AttemptSpec, sink execution.StreamSink) execution.StreamResult {
				if err := sink(execution.StreamEvent{Sequence: 1, Kind: execution.StreamEventReady, StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}}); err != nil {
					return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
				}
				if err := sink(execution.StreamEvent{Sequence: 2, Kind: execution.StreamEventData, Data: []byte("event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_keep\"}}\n\n")}); err != nil {
					t.Fatal(err)
				}
				return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Error: &execution.ErrorEvidence{Kind: kind, Summary: "terminal failure"}}
			}}
			input := responsesExecutionForwardInput()
			input.BufferedStream = true
			result := NewExecutionForwarder(executor).ForwardStream(context.Background(), input, httptest.NewRecorder())
			if result.Stream.ResponseID != "resp_keep" {
				t.Fatalf("response ID = %q, want resp_keep", result.Stream.ResponseID)
			}
		})
	}
}

func TestBufferedStreamResponsesFailedDispositionFails(t *testing.T) {
	observer := newStreamEventObserver(dialect.NewOpenAIResponses(), nil, true)
	if _, err := observer.classify(dialect.StreamEvent{
		Name:    "response.failed",
		Payload: []byte(`{"type":"response.failed","response":{"id":"resp_failed"}}`),
	}, false); err != nil {
		t.Fatal(err)
	}
	observation := observer.endObservation()
	if observation.EndReason == StreamEndCleanEOF {
		t.Fatalf("response.failed was accepted as clean EOF: %#v", observation)
	}
	if observation.ResponseID != "resp_failed" {
		t.Fatalf("response.failed response ID = %q, want resp_failed", observation.ResponseID)
	}
}

func TestBufferedStreamAnthropicRejectsDuplicateContentBlockStart(t *testing.T) {
	observer := newStreamEventObserver(dialect.NewAnthropic(), nil, true)
	start := dialect.StreamEvent{Name: "content_block_start", Payload: []byte(`{"type":"content_block_start","index":0}`)}
	if _, err := observer.classify(start, false); err != nil {
		t.Fatal(err)
	}
	if _, err := observer.classify(start, false); err == nil {
		t.Fatal("duplicate Anthropic content_block_start was accepted")
	}
}

func TestBufferedStreamResponsesReplayRejectsUnknownSemantics(t *testing.T) {
	request := &dialect.ParsedRequest{
		Method: http.MethodPost,
		Path:   "/v1/responses",
	}
	for _, body := range []string{
		`{"model":"gpt-5","input":"hello","store":false,"provider_state":"opaque"}`,
		`{"model":"gpt-5","input":"hello","store":false,"tools":[{"type":"function","name":"lookup","parameters":{},"provider_state":"opaque"}]}`,
	} {
		request.Body = []byte(body)
		if bufferedStreamReplayEligible(request, dialect.NewOpenAIResponses()) {
			t.Fatalf("Responses request with unknown semantics was replay-eligible: %s", body)
		}
	}
}

func TestBufferedStreamHeartbeatFailureKeepsResponsesID(t *testing.T) {
	started := make(chan struct{})
	executor := fakeExecutionExecutor{stream: func(ctx context.Context, _ execution.AttemptSpec, sink execution.StreamSink) execution.StreamResult {
		if err := sink(execution.StreamEvent{Sequence: 1, Kind: execution.StreamEventReady, StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}}); err != nil {
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
		}
		if err := sink(execution.StreamEvent{Sequence: 2, Kind: execution.StreamEventData, Data: []byte("event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_heartbeat\"}}\n\n")}); err != nil {
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
		}
		close(started)
		<-ctx.Done()
		return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindCanceled, Summary: "canceled"}}
	}}
	forwarder := NewExecutionForwarder(executor)
	forwarder.heartbeatInterval = time.Millisecond
	writer := &failAfterFirstWriteResponseWriter{header: make(http.Header)}
	resultCh := make(chan UpstreamResult, 1)
	input := responsesExecutionForwardInput()
	input.BufferedStream = true
	go func() { resultCh <- forwarder.ForwardStream(context.Background(), input, writer) }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("buffered Responses stream did not start")
	}
	result := <-resultCh
	if result.Stream.EndReason != StreamEndDownstreamWriteFailure {
		t.Fatalf("heartbeat failure result = %#v", result)
	}
	if result.Stream.ResponseID != "resp_heartbeat" {
		t.Fatalf("heartbeat failure response ID = %q, want resp_heartbeat", result.Stream.ResponseID)
	}
}

func TestBufferedStreamDeadlineBeforeReleaseKeepsResponsesID(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	executor := fakeExecutionExecutor{stream: func(_ context.Context, _ execution.AttemptSpec, sink execution.StreamSink) execution.StreamResult {
		for sequence, event := range []execution.StreamEvent{
			{Sequence: 1, Kind: execution.StreamEventReady, StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}},
			{Sequence: 2, Kind: execution.StreamEventData, Data: []byte("event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_deadline\"}}\n\n")},
			{Sequence: 3, Kind: execution.StreamEventData, Data: []byte("event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_deadline\"}}\n\n")},
		} {
			if err := sink(event); err != nil {
				t.Fatal(err)
			}
			_ = sequence
		}
		cancel()
		return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
	}}
	input := responsesExecutionForwardInput()
	input.BufferedStream = true
	result := NewExecutionForwarder(executor).ForwardStream(ctx, input, httptest.NewRecorder())
	if result.Stream.EndReason != StreamEndDownstreamWriteFailure {
		t.Fatalf("deadline-before-release result = %#v", result)
	}
	if result.Stream.ResponseID != "resp_deadline" {
		t.Fatalf("deadline-before-release response ID = %q, want resp_deadline", result.Stream.ResponseID)
	}
}

type failAfterFirstWriteResponseWriter struct {
	header http.Header
	mu     sync.Mutex
	writes int
}

func (writer *failAfterFirstWriteResponseWriter) Header() http.Header { return writer.header }
func (*failAfterFirstWriteResponseWriter) WriteHeader(int)            {}
func (*failAfterFirstWriteResponseWriter) FlushError() error          { return nil }
func (writer *failAfterFirstWriteResponseWriter) Write(data []byte) (int, error) {
	writer.mu.Lock()
	defer writer.mu.Unlock()
	writer.writes++
	if writer.writes > 1 {
		return 0, io.ErrClosedPipe
	}
	return len(data), nil
}

func TestHandlerBufferedStreamUsesOneTotalDeadlineAcrossAttempts(t *testing.T) {
	forwarder := &totalDeadlineStreamForwarder{}
	engine, _ := newConvertedFallbackHandlerTestRuntime(t, forwarder, config.Settings{
		state.SettingBufferedStream: true,
		state.SettingRequestTimeout: json.Number("1"),
		state.SettingRetryCount:     json.Number("1"),
	})
	request := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"model":"claude-client","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"stream":true}`))
	request.Header.Set("Authorization", "Bearer gl-client")
	started := time.Now()
	recorder := httptest.NewRecorder()
	engine.ServeHTTP(recorder, request)
	elapsed := time.Since(started)

	if forwarder.calls != 2 {
		t.Fatalf("buffered attempts = %d, want first attempt plus one attempt under the same deadline (deadlines=%v, body=%q)", forwarder.calls, forwarder.deadlines, recorder.Body.String())
	}
	if len(forwarder.deadlines) != 2 || !forwarder.deadlines[0].Equal(forwarder.deadlines[1]) {
		t.Fatalf("attempt deadlines = %#v, want one shared deadline", forwarder.deadlines)
	}
	if elapsed >= 1500*time.Millisecond {
		t.Fatalf("total buffered deadline was reset across attempts: elapsed=%s", elapsed)
	}
	if recorder.Body.Len() != 0 {
		t.Fatalf("deadline path exposed a body: %s", recorder.Body.String())
	}
}

type totalDeadlineStreamForwarder struct {
	calls     int
	deadlines []time.Time
}

func (*totalDeadlineStreamForwarder) Forward(context.Context, ForwardInput) UpstreamResult {
	return UpstreamResult{Err: errors.New("unexpected unary forward")}
}

func (forwarder *totalDeadlineStreamForwarder) ForwardStream(ctx context.Context, _ ForwardInput, _ http.ResponseWriter) UpstreamResult {
	forwarder.calls++
	if deadline, ok := ctx.Deadline(); ok {
		forwarder.deadlines = append(forwarder.deadlines, deadline)
	}
	if forwarder.calls == 1 {
		timer := time.NewTimer(100 * time.Millisecond)
		defer timer.Stop()
		select {
		case <-timer.C:
			return UpstreamResult{
				StatusCode: http.StatusOK, RequestWritten: true, DispatchState: execution.DispatchMaybeSent,
				Committed: true, HTTPCommitted: true,
				BufferedStream: true, ClientVisibleBytes: int64(len(bufferedStreamHeartbeat)),
				Stream: StreamObservation{EndReason: StreamEndUpstreamProtocolError},
			}
		case <-ctx.Done():
			return UpstreamResult{Err: ctx.Err(), RequestWritten: true}
		}
	}
	<-ctx.Done()
	return UpstreamResult{
		Err: ctx.Err(), RequestWritten: true, DispatchState: execution.DispatchMaybeSent, Committed: true, HTTPCommitted: true,
		BufferedStream: true, ClientVisibleBytes: int64(len(bufferedStreamHeartbeat)),
		Stream: StreamObservation{EndReason: StreamEndUpstreamProtocolError},
	}
}

func TestBufferedStreamSetsNoTransformAndDoesNotDuplicateHeaders(t *testing.T) {
	input := executionForwardInput()
	input.BufferedStream = true
	input.ClientProtocol = protocol.OpenAICompletions
	input.Request.Header = http.Header{
		"X-Trace":       {"one"},
		"Authorization": {"Bearer client"},
	}
	first, err := newExecutionAttemptSpec(input)
	if err != nil {
		t.Fatal(err)
	}
	second, err := newExecutionAttemptSpec(input)
	if err != nil {
		t.Fatal(err)
	}
	if got := second.Header.Values("X-Trace"); len(got) != 1 || got[0] != "one" {
		t.Fatalf("retry header values = %#v, want one value", got)
	}
	if first.Header.Values("Authorization") != nil || second.Header.Values("Authorization") != nil {
		t.Fatal("client authorization was forwarded to upstream")
	}
	if got := normalizeStreamResponseHeaders(http.Header{"Content-Type": {"text/event-stream"}}).Get("Cache-Control"); got != "no-cache, no-transform" {
		t.Fatalf("Cache-Control = %q, want no-cache, no-transform", got)
	}
}

func TestBufferedStreamTerminalErrorKeepsResponsesID(t *testing.T) {
	var result UpstreamResult
	executor := fakeExecutionExecutor{stream: func(
		_ context.Context,
		_ execution.AttemptSpec,
		sink execution.StreamSink,
	) execution.StreamResult {
		if err := sink(execution.StreamEvent{Sequence: 1, Kind: execution.StreamEventReady, StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}, UpstreamRequestID: "upstream-request"}); err != nil {
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindInternal, Summary: err.Error()}}
		}
		if err := sink(execution.StreamEvent{Sequence: 2, Kind: execution.StreamEventData, Data: []byte("event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_keep\"}}\n\n")}); err != nil {
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindInternal, Summary: err.Error()}}
		}
		return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
	}}
	input := responsesExecutionForwardInput()
	input.BufferedStream = true
	input.ClientProtocol = protocol.OpenAIResponses
	response := httptest.NewRecorder()
	result = NewExecutionForwarder(executor).ForwardStream(context.Background(), input, response)
	if result.Stream.EndReason == StreamEndCleanEOF {
		t.Fatal("missing Responses terminal event was accepted")
	}
	if result.Stream.ResponseID != "resp_keep" {
		t.Fatalf("terminal result response ID = %q, want resp_keep", result.Stream.ResponseID)
	}
}
