package embedded

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
)

type observerEvent struct {
	kind      string
	attemptID string
	body      []byte
	status    int
	headers   http.Header
	trailers  http.Header
	err       error
}

type recordingHTTPObserver struct {
	mu     sync.Mutex
	events []observerEvent
	done   chan struct{}
	once   sync.Once
}

type blockingHTTPObserver struct {
	*recordingHTTPObserver
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

func (observer *blockingHTTPObserver) ObserveRequest(attemptID string, request *http.Request) {
	observer.once.Do(func() { close(observer.started) })
	<-observer.release
	observer.recordingHTTPObserver.ObserveRequest(attemptID, request)
}

type panicHTTPObserver struct{}

func (panicHTTPObserver) ObserveRequest(string, *http.Request)     { panic("request callback") }
func (panicHTTPObserver) ObserveRequestBody(string, []byte)        { panic("request body callback") }
func (panicHTTPObserver) ObserveResponse(string, int, http.Header) { panic("response callback") }
func (panicHTTPObserver) ObserveResponseBody(string, []byte)       { panic("response body callback") }
func (panicHTTPObserver) ObserveResponseComplete(string, http.Header, error) {
	panic("completion callback")
}

func (observer *recordingHTTPObserver) ObserveRequest(attemptID string, request *http.Request) {
	observer.mu.Lock()
	defer observer.mu.Unlock()
	observer.events = append(observer.events, observerEvent{
		kind: "request", attemptID: attemptID, headers: request.Header.Clone(),
	})
}

func (observer *recordingHTTPObserver) ObserveRequestBody(attemptID string, body []byte) {
	observer.mu.Lock()
	defer observer.mu.Unlock()
	observer.events = append(observer.events, observerEvent{kind: "request-body", attemptID: attemptID, body: append([]byte(nil), body...)})
}

func (observer *recordingHTTPObserver) ObserveResponse(attemptID string, status int, headers http.Header) {
	observer.mu.Lock()
	defer observer.mu.Unlock()
	observer.events = append(observer.events, observerEvent{kind: "response", attemptID: attemptID, status: status, headers: headers.Clone()})
}

func (observer *recordingHTTPObserver) ObserveResponseBody(attemptID string, body []byte) {
	observer.mu.Lock()
	defer observer.mu.Unlock()
	observer.events = append(observer.events, observerEvent{kind: "response-body", attemptID: attemptID, body: append([]byte(nil), body...)})
}

func (observer *recordingHTTPObserver) ObserveResponseComplete(attemptID string, trailers http.Header, err error) {
	observer.mu.Lock()
	observer.events = append(observer.events, observerEvent{kind: "complete", attemptID: attemptID, trailers: trailers.Clone(), err: err})
	observer.mu.Unlock()
	observer.once.Do(func() { close(observer.done) })
}

func (observer *recordingHTTPObserver) wait(t *testing.T) {
	t.Helper()
	select {
	case <-observer.done:
	case <-time.After(time.Second):
		t.Fatal("observer did not receive completion")
	}
}

func (observer *recordingHTTPObserver) snapshot() []observerEvent {
	observer.mu.Lock()
	defer observer.mu.Unlock()
	result := make([]observerEvent, len(observer.events))
	copy(result, observer.events)
	return result
}

func newRecordingHTTPObserver() *recordingHTTPObserver {
	return &recordingHTTPObserver{done: make(chan struct{})}
}

func TestHTTPObserverReliablyDeliversLongStreamBehindSlowCallback(t *testing.T) {
	observer := &blockingHTTPObserver{
		recordingHTTPObserver: newRecordingHTTPObserver(),
		started:               make(chan struct{}),
		release:               make(chan struct{}),
	}
	requestBody := bytes.Repeat([]byte("r"), 128)
	responseBody := bytes.Repeat([]byte("s"), 128)
	request, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://upstream.invalid", &byteByByteReader{data: requestBody})
	if err != nil {
		t.Fatal(err)
	}
	response, err := observeRoundTrip(roundTripperFunc(func(request *http.Request) (*http.Response, error) {
		if got, readErr := io.ReadAll(request.Body); readErr != nil || !bytes.Equal(got, requestBody) {
			t.Fatalf("transport request body = %q, err = %v", got, readErr)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       &byteByByteReader{data: responseBody},
		}, nil
	}), request, observer, "slow-attempt")
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-observer.started:
	case <-time.After(time.Second):
		t.Fatal("observer did not start the blocking callback")
	}
	close(observer.release)
	if got, readErr := io.ReadAll(response.Body); readErr != nil || !bytes.Equal(got, responseBody) {
		t.Fatalf("response body = %q, err = %v", got, readErr)
	}
	if err := response.Body.Close(); err != nil {
		t.Fatal(err)
	}
	observer.wait(t)

	var requestBytes, responseBytes []byte
	for _, event := range observer.snapshot() {
		switch event.kind {
		case "request-body":
			requestBytes = append(requestBytes, event.body...)
		case "response-body":
			responseBytes = append(responseBytes, event.body...)
		}
	}
	if !bytes.Equal(requestBytes, requestBody) || !bytes.Equal(responseBytes, responseBody) {
		t.Fatalf("slow callback dropped bytes: request=%d/%d response=%d/%d", len(requestBytes), len(requestBody), len(responseBytes), len(responseBody))
	}
}

func TestHTTPObserverCancellationPreservesBytesReadAfterCancel(t *testing.T) {
	observer := &blockingHTTPObserver{
		recordingHTTPObserver: newRecordingHTTPObserver(),
		started:               make(chan struct{}),
		release:               make(chan struct{}),
	}
	ctx, cancel := context.WithCancel(t.Context())
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://upstream.invalid", nil)
	if err != nil {
		t.Fatal(err)
	}
	response, err := observeRoundTrip(roundTripperFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: &cancelAwareBody{ctx: ctx}}, nil
	}), request, observer, "cancel-attempt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := response.Body.Read(make([]byte, 16)); err != nil {
		t.Fatal(err)
	}
	cancel()
	if _, err := response.Body.Read(make([]byte, 32)); !errors.Is(err, context.Canceled) {
		t.Fatalf("second response read error = %v, want context.Canceled", err)
	}
	if err := response.Body.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-observer.started:
	case <-time.After(time.Second):
		t.Fatal("observer did not start the blocking callback")
	}
	close(observer.release)
	observer.wait(t)

	var body []byte
	var complete *observerEvent
	for _, event := range observer.snapshot() {
		switch event.kind {
		case "response-body":
			body = append(body, event.body...)
		case "complete":
			eventCopy := event
			complete = &eventCopy
		}
	}
	if string(body) != "firstafter-cancel" {
		t.Fatalf("captured response body = %q, want all bytes read after cancel", body)
	}
	if complete == nil || !errors.Is(complete.err, context.Canceled) {
		t.Fatalf("completion = %#v, want context.Canceled", complete)
	}
}

func TestHTTPObserverCallbackPanicStillClosesSink(t *testing.T) {
	sink := newObserverSink(panicHTTPObserver{}, "panic-attempt", nil)
	if !sink.emit(func() { sink.observer.ObserveRequestBody(sink.attemptID, []byte("body")) }) {
		t.Fatal("panic callback was not queued")
	}
	sink.complete(nil, nil)
	select {
	case <-sink.done:
	case <-time.After(time.Second):
		t.Fatal("observer sink did not close after completion callback panic")
	}
	if sink.emit(func() {}) {
		t.Fatal("observer sink accepted an event after completion")
	}
}

func TestHTTPObserverDoesNotCompleteOnContextCancelBeforeBodyLifecycleEnds(t *testing.T) {
	observer := newRecordingHTTPObserver()
	ctx, cancel := context.WithCancel(t.Context())
	sink := newObserverSink(observer, "cancel-lifecycle", ctx)
	cancel()
	select {
	case <-sink.done:
		t.Fatal("context cancellation completed observer before body lifecycle ended")
	case <-time.After(25 * time.Millisecond):
	}
	sink.complete(nil, context.Canceled)
	observer.wait(t)
}

func TestHTTPObserverRoundTripCapturesConsumedRequestAndResponseBytes(t *testing.T) {
	observer := newRecordingHTTPObserver()
	requestReader := io.MultiReader(strings.NewReader("request-"), strings.NewReader("body"))
	request, err := http.NewRequest(http.MethodPost, "https://upstream.invalid/v1/messages", requestReader)
	if err != nil {
		t.Fatal(err)
	}
	request = request.WithContext(context.WithValue(request.Context(), httpAttemptIDContextKey, "attempt-1"))
	request.Header.Set("Authorization", "Bearer secret")
	responseReader := io.MultiReader(strings.NewReader("response-"), strings.NewReader("body"))
	transport := roundTripperFunc(func(request *http.Request) (*http.Response, error) {
		if got, err := io.ReadAll(request.Body); err != nil || string(got) != "request-body" {
			t.Fatalf("transport request body = %q, err = %v", got, err)
		}
		return &http.Response{
			StatusCode: http.StatusCreated,
			Header:     http.Header{"X-Upstream": {"ok"}},
			Trailer:    http.Header{"X-Trailer": {"done"}},
			Body:       io.NopCloser(responseReader),
			Request:    request,
		}, nil
	})

	response, err := observeRoundTrip(transport, request, observer, "attempt-1")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.ReadAll(response.Body); err != nil {
		t.Fatal(err)
	}
	if err := response.Body.Close(); err != nil {
		t.Fatal(err)
	}
	observer.wait(t)

	events := observer.snapshot()
	if len(events) < 5 {
		t.Fatalf("observer events = %#v", events)
	}
	if events[0].kind != "request" || events[0].attemptID != "attempt-1" || events[0].headers.Get("Authorization") != "Bearer secret" {
		t.Fatalf("request event = %#v", events[0])
	}
	var requestBody, responseBody []byte
	for _, event := range events {
		switch event.kind {
		case "request-body":
			requestBody = append(requestBody, event.body...)
		case "response-body":
			responseBody = append(responseBody, event.body...)
		}
	}
	if string(requestBody) != "request-body" {
		t.Fatalf("request body events = %q", requestBody)
	}
	var responseEvent, complete *observerEvent
	for index := range events {
		event := &events[index]
		switch event.kind {
		case "response":
			responseEvent = event
		case "complete":
			complete = event
		}
	}
	if responseEvent == nil || responseEvent.status != http.StatusCreated || responseEvent.headers.Get("X-Upstream") != "ok" {
		t.Fatalf("response event = %#v", responseEvent)
	}
	if string(responseBody) != "response-body" {
		t.Fatalf("response body events = %q", responseBody)
	}
	if complete == nil || complete.trailers.Get("X-Trailer") != "done" || complete.err != nil {
		t.Fatalf("completion event = %#v", complete)
	}
}

func TestHTTPObserverResponseReadErrorAndCloseAreRecorded(t *testing.T) {
	readErr := errors.New("stream read failed")
	for _, test := range []struct {
		name     string
		body     io.ReadCloser
		wantErr  error
		read     bool
		wantBody string
	}{
		{name: "read error", body: &errorReadCloser{body: []byte("partial"), err: readErr}, wantErr: readErr, read: true, wantBody: "partial"},
		{name: "early close", body: io.NopCloser(strings.NewReader("unread")), wantErr: io.ErrClosedPipe, read: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			observer := newRecordingHTTPObserver()
			request := httptestNewRequestWithAttemptID("attempt-error")
			response, err := observeRoundTrip(roundTripperFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: test.body}, nil
			}), request, observer, "attempt-error")
			if err != nil {
				t.Fatal(err)
			}
			if test.read {
				body, actualReadErr := io.ReadAll(response.Body)
				if !errors.Is(actualReadErr, test.wantErr) || string(body) != test.wantBody {
					t.Fatalf("read body = %q, err = %v", body, actualReadErr)
				}
			}
			if err := response.Body.Close(); err != nil {
				t.Fatal(err)
			}
			observer.wait(t)
			events := observer.snapshot()
			complete := events[len(events)-1]
			if !errors.Is(complete.err, test.wantErr) {
				t.Fatalf("completion error = %v, want %v", complete.err, test.wantErr)
			}
		})
	}
}

func TestHTTPObserverIsOptionalAndDoesNotChangeTransportBehavior(t *testing.T) {
	request := httptestNewRequestWithAttemptID("")
	want := []byte("response")
	response, err := observeRoundTrip(roundTripperFunc(func(request *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(bytes.NewReader(want)), Request: request}, nil
	}), request, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	got, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("response body = %q, want %q", got, want)
	}
}

func TestHTTPObserverDoesNotClaimLocalGrokTokenCountAsUpstreamHTTP(t *testing.T) {
	observer := newRecordingHTTPObserver()
	ctx := context.WithValue(t.Context(), httpObserverContextKey, HTTPObserver(observer))
	ctx = context.WithValue(ctx, httpAttemptIDContextKey, "local-count")
	_, err := NewGrokHTTPExecutor().CountTokensCanonical(ctx, ExecuteRequest{
		Model: "grok-4.3", Format: "openai-response",
		Payload: []byte(`{"model":"grok-4.3","input":"hello"}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	if events := observer.snapshot(); len(events) != 0 {
		t.Fatalf("local token count emitted HTTP observer events: %#v", events)
	}
}

func TestSupportedExecutorContextsCarryHTTPObserver(t *testing.T) {
	observer := newRecordingHTTPObserver()
	ctx := context.WithValue(context.Background(), httpObserverContextKey, HTTPObserver(observer))
	ctx = context.WithValue(ctx, httpAttemptIDContextKey, "attempt-context")

	codex := NewCodexHTTPExecutor()
	codexContext := codex.executionContext(ctx, NewCodexAuth("id", CodexCredential{}, ""), nil, false)
	assertObserverContext(t, codexContext)

	claude, ok := NewClaudeHTTPExecutor().(*claudeHTTPExecutor)
	if !ok {
		t.Fatal("Claude executor type changed")
	}
	assertObserverContext(t, claude.executionContext(ctx, NewClaudeAuth("id", ClaudeCredential{}, ""), nil, false))

	grok, ok := NewGrokHTTPExecutor().(*grokHTTPExecutor)
	if !ok {
		t.Fatal("Grok executor type changed")
	}
	assertObserverContext(t, grok.executionContext(ctx, NewGrokAuth("id", GrokCredential{}, ""), nil))

	antigravity, ok := NewAntigravityHTTPExecutor().(*antigravityHTTPExecutor)
	if !ok {
		t.Fatal("Antigravity executor type changed")
	}
	antigravityContext, err := antigravity.executionContext(ctx, "id", "account", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	assertObserverContext(t, antigravityContext)
}

func assertObserverContext(t *testing.T, ctx context.Context) {
	t.Helper()
	transport, ok := ctx.Value("cliproxy.roundtripper").(noRedirectRoundTripper)
	if !ok || transport.observer == nil {
		t.Fatalf("execution context transport = %#v, observer missing", ctx.Value("cliproxy.roundtripper"))
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (fn roundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}

type byteByByteReader struct {
	data []byte
	pos  int
}

func (reader *byteByByteReader) Read(buffer []byte) (int, error) {
	if reader.pos == len(reader.data) {
		return 0, io.EOF
	}
	buffer[0] = reader.data[reader.pos]
	reader.pos++
	return 1, nil
}

func (reader *byteByByteReader) Close() error { return nil }

type cancelAwareBody struct {
	ctx  context.Context
	step int
}

func (body *cancelAwareBody) Read(buffer []byte) (int, error) {
	if body.step == 0 {
		body.step++
		copy(buffer, "first")
		return len("first"), nil
	}
	copy(buffer, "after-cancel")
	return len("after-cancel"), body.ctx.Err()
}

func (body *cancelAwareBody) Close() error { return nil }

type errorReadCloser struct {
	body []byte
	err  error
}

func (reader *errorReadCloser) Read(p []byte) (int, error) {
	if len(reader.body) == 0 {
		return 0, reader.err
	}
	n := copy(p, reader.body)
	reader.body = reader.body[n:]
	return n, nil
}

func (reader *errorReadCloser) Close() error { return nil }

func httptestNewRequestWithAttemptID(attemptID string) *http.Request {
	request, _ := http.NewRequest(http.MethodGet, "https://upstream.invalid", nil)
	return request.WithContext(context.WithValue(request.Context(), httpAttemptIDContextKey, attemptID))
}

var _ auth.ProviderExecutor = NewCodexHTTPExecutor()
