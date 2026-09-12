package utils

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
	"github.com/valyala/fasthttp/fasthttputil"
)

const testAttemptID = "attempt-1"

type responseRecord struct {
	status  int
	headers http.Header
}

type completionRecord struct {
	headers http.Header
	err     error
}

type terminationRecord struct {
	termination string
	detail      string
}

// recordingObserver captures every callback in order. It also implements the
// optional termination extension.
type recordingObserver struct {
	mu             sync.Mutex
	requests       []*http.Request
	requestBodies  [][]byte
	responses      []responseRecord
	responseBodies []byte
	completions    []completionRecord
	terminations   []terminationRecord
	panicMode      bool
}

func (o *recordingObserver) call(fn func()) {
	o.mu.Lock()
	panicMode := o.panicMode
	o.mu.Unlock()
	if panicMode {
		panic("observer failure")
	}
	fn()
}

func (o *recordingObserver) ObserveRequest(id string, request *http.Request) {
	if id != testAttemptID {
		panic("unexpected attempt id " + id)
	}
	o.call(func() {
		o.mu.Lock()
		defer o.mu.Unlock()
		o.requests = append(o.requests, request)
	})
}

func (o *recordingObserver) ObserveRequestBody(id string, data []byte) {
	o.call(func() {
		o.mu.Lock()
		defer o.mu.Unlock()
		o.requestBodies = append(o.requestBodies, bytes.Clone(data))
	})
}

func (o *recordingObserver) ObserveResponse(id string, status int, headers http.Header) {
	o.call(func() {
		o.mu.Lock()
		defer o.mu.Unlock()
		o.responses = append(o.responses, responseRecord{status: status, headers: headers.Clone()})
	})
}

func (o *recordingObserver) ObserveResponseBody(id string, data []byte) {
	o.call(func() {
		o.mu.Lock()
		defer o.mu.Unlock()
		o.responseBodies = append(o.responseBodies, data...)
	})
}

func (o *recordingObserver) ObserveResponseComplete(id string, headers http.Header, err error) {
	o.call(func() {
		o.mu.Lock()
		defer o.mu.Unlock()
		o.completions = append(o.completions, completionRecord{headers: headers, err: err})
	})
}

func (o *recordingObserver) ObserveResponseTermination(id string, termination string, detail string) {
	o.call(func() {
		o.mu.Lock()
		defer o.mu.Unlock()
		o.terminations = append(o.terminations, terminationRecord{termination: termination, detail: detail})
	})
}

func (o *recordingObserver) bodyBytes() []byte {
	o.mu.Lock()
	defer o.mu.Unlock()
	return bytes.Clone(o.responseBodies)
}

func (o *recordingObserver) firstCompletionErr() error {
	o.mu.Lock()
	defer o.mu.Unlock()
	if len(o.completions) == 0 {
		return nil
	}
	return o.completions[0].err
}

func (o *recordingObserver) completionCount() int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return len(o.completions)
}

func (o *recordingObserver) lastTermination() string {
	o.mu.Lock()
	defer o.mu.Unlock()
	if len(o.terminations) == 0 {
		return ""
	}
	return o.terminations[len(o.terminations)-1].termination
}

func (o *recordingObserver) requestBodyBytes() []byte {
	o.mu.Lock()
	defer o.mu.Unlock()
	var out []byte
	for _, part := range o.requestBodies {
		out = append(out, part...)
	}
	return out
}

func newObserverContext(observer HTTPObserver) *schemas.BifrostContext {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	ctx.SetValue(httpObserverContextKey, observer)
	ctx.SetValue(httpAttemptIDContextKey, testAttemptID)
	return ctx
}

func startServer(t *testing.T, handler fasthttp.RequestHandler) (*fasthttp.Client, func()) {
	t.Helper()
	listener := fasthttputil.NewInmemoryListener()
	server := &fasthttp.Server{Handler: handler}
	go func() { _ = server.Serve(listener) }()
	client := &fasthttp.Client{
		Dial: func(string) (net.Conn, error) { return listener.Dial() },
	}
	cleanup := func() {
		_ = server.Shutdown()
		_ = listener.Close()
	}
	return client, cleanup
}

func startTCPServer(t *testing.T, handler fasthttp.RequestHandler) (string, func()) {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	server := &fasthttp.Server{Handler: handler}
	go func() { _ = server.Serve(listener) }()
	return "http://" + listener.Addr().String(), func() {
		_ = server.Shutdown()
		_ = listener.Close()
	}
}

func TestMakeRequestWithContextUnaryObservesRequestAndBufferedResponse(t *testing.T) {
	requestBody := []byte(`{"hello":"world"}`)
	responseBody := []byte(`{"ok":true}`)
	client, cleanup := startServer(t, func(ctx *fasthttp.RequestCtx) {
		if !bytes.Equal(ctx.PostBody(), requestBody) {
			t.Errorf("server saw body %q", ctx.PostBody())
		}
		ctx.Response.Header.Set("X-Test", "yes")
		ctx.SetStatusCode(200)
		ctx.SetBody(responseBody)
	})
	defer cleanup()

	observer := &recordingObserver{}
	ctx := newObserverContext(observer)
	req := fasthttp.AcquireRequest()
	defer fasthttp.ReleaseRequest(req)
	req.Header.SetMethod(fasthttp.MethodPost)
	req.SetRequestURI("http://example.invalid/v1/chat")
	req.Header.Set("X-Request", "1")
	req.SetBody(requestBody)
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseResponse(resp)

	_, bifrostErr, wait := MakeRequestWithContext(ctx, client, req, resp)
	wait()
	if bifrostErr != nil {
		t.Fatalf("unexpected error: %v", bifrostErr)
	}

	if len(observer.requests) != 1 {
		t.Fatalf("expected 1 request event, got %d", len(observer.requests))
	}
	if got := observer.requests[0].Header.Get("X-Request"); got != "1" {
		t.Fatalf("request header not observed: %q", got)
	}
	if got := observer.requestBodyBytes(); !bytes.Equal(got, requestBody) {
		t.Fatalf("request body mismatch: %q", got)
	}
	if len(observer.responses) != 1 || observer.responses[0].status != 200 {
		t.Fatalf("response not observed: %#v", observer.responses)
	}
	if got := observer.responses[0].headers.Get("X-Test"); got != "yes" {
		t.Fatalf("response header not observed: %q", got)
	}
	if got := observer.bodyBytes(); !bytes.Equal(got, responseBody) {
		t.Fatalf("response body mismatch: %q", got)
	}
	if observer.completionCount() != 1 {
		t.Fatalf("expected exactly one completion, got %d", observer.completionCount())
	}
	if err := observer.firstCompletionErr(); err != nil {
		t.Fatalf("expected eof completion, got %v", err)
	}
	if got := observer.lastTermination(); got != TerminationEOF {
		t.Fatalf("expected eof termination, got %q", got)
	}
}

func TestStreamingSSEBytesArePreservedExactly(t *testing.T) {
	fixture := []byte("event: message\n" +
		"data: {\"a\":1}\n" +
		"\n" +
		": keep-alive comment\n" +
		"data: line1\r\ndata: line2\n\n" +
		"data: [DONE]\n\n")
	client, cleanup := startServer(t, func(ctx *fasthttp.RequestCtx) {
		ctx.Response.Header.SetContentType("text/event-stream")
		ctx.SetBody(fixture)
	})
	defer cleanup()

	observer := &recordingObserver{}
	ctx := newObserverContext(observer)
	req := fasthttp.AcquireRequest()
	defer fasthttp.ReleaseRequest(req)
	req.Header.SetMethod(fasthttp.MethodPost)
	req.SetRequestURI("http://example.invalid/v1/chat/stream")
	resp := fasthttp.AcquireResponse()
	resp.StreamBody = true

	if err := DoStreamingRequest(ctx, client, req, resp); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	reader, releaseGzip := DecompressStreamBody(ctx, resp)
	defer releaseGzip()
	got, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("read error: %v", err)
	}
	if !bytes.Equal(got, fixture) {
		t.Fatalf("parser-visible bytes changed\n got %q\nwant %q", got, fixture)
	}
	if !bytes.Equal(observer.bodyBytes(), fixture) {
		t.Fatalf("observed bytes changed\n got %q\nwant %q", observer.bodyBytes(), fixture)
	}
	ReleaseStreamingResponse(ctx, resp)
	if observer.completionCount() != 1 {
		t.Fatalf("expected exactly one completion, got %d", observer.completionCount())
	}
}

func TestReleaseStreamingResponseDrainsTailThroughObserver(t *testing.T) {
	fixture := []byte("data: one\n\ndata: [DONE]\n\n" + "tail-after-marker")
	client, cleanup := startServer(t, func(ctx *fasthttp.RequestCtx) {
		ctx.Response.Header.SetContentType("text/event-stream")
		ctx.SetBody(fixture)
	})
	defer cleanup()

	observer := &recordingObserver{}
	ctx := newObserverContext(observer)
	req := fasthttp.AcquireRequest()
	defer fasthttp.ReleaseRequest(req)
	req.Header.SetMethod(fasthttp.MethodPost)
	req.SetRequestURI("http://example.invalid/v1/chat/stream")
	resp := fasthttp.AcquireResponse()
	resp.StreamBody = true

	if err := DoStreamingRequest(ctx, client, req, resp); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	reader, releaseGzip := DecompressStreamBody(ctx, resp)
	// Simulate a parser that stops at the [DONE] marker: read exactly the
	// prefix, then stop touching the reader.
	markerEnd := bytes.Index(fixture, []byte("[DONE]\n\n")) + len("[DONE]\n\n")
	head := make([]byte, markerEnd)
	if _, err := io.ReadFull(reader, head); err != nil {
		t.Fatalf("read head: %v", err)
	}
	releaseGzip()

	ReleaseStreamingResponse(ctx, resp)

	if got := observer.bodyBytes(); !bytes.Equal(got, fixture) {
		t.Fatalf("drain tail was not observed exactly once\n got %q\nwant %q", got, fixture)
	}
	if observer.completionCount() != 1 {
		t.Fatalf("expected exactly one completion, got %d", observer.completionCount())
	}
	if got := observer.lastTermination(); got != TerminationEOF {
		t.Fatalf("expected eof termination after drain, got %q", got)
	}
}

func TestTruncatedStreamTerminatesWithErrorAndKeepsBytes(t *testing.T) {
	// A stream that dies mid-frame: the fixture has no terminal marker.
	fixture := []byte("data: {\"partial\":")
	client, cleanup := startServer(t, func(ctx *fasthttp.RequestCtx) {
		ctx.Response.Header.SetContentType("text/event-stream")
		ctx.Response.Header.SetContentLength(-1)
		ctx.SetBody(fixture)
	})
	defer cleanup()

	observer := &recordingObserver{}
	ctx := newObserverContext(observer)
	req := fasthttp.AcquireRequest()
	defer fasthttp.ReleaseRequest(req)
	req.Header.SetMethod(fasthttp.MethodPost)
	req.SetRequestURI("http://example.invalid/v1/chat/stream")
	resp := fasthttp.AcquireResponse()
	resp.StreamBody = true

	if err := DoStreamingRequest(ctx, client, req, resp); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	reader, releaseGzip := DecompressStreamBody(ctx, resp)
	got, err := io.ReadAll(reader)
	releaseGzip()
	if err != nil {
		t.Fatalf("read error: %v", err)
	}
	if !bytes.Equal(got, fixture) {
		t.Fatalf("truncated bytes changed: %q", got)
	}
	if !bytes.Equal(observer.bodyBytes(), fixture) {
		t.Fatalf("truncated bytes not captured: %q", observer.bodyBytes())
	}
	ReleaseStreamingResponse(ctx, resp)
	if observer.completionCount() != 1 {
		t.Fatalf("expected exactly one completion, got %d", observer.completionCount())
	}
}

func TestConnectionFailureIsLegalNoResponse(t *testing.T) {
	observer := &recordingObserver{}
	ctx := newObserverContext(observer)
	client := &fasthttp.Client{
		Dial: func(string) (net.Conn, error) { return nil, errors.New("dial refused") },
	}
	req := fasthttp.AcquireRequest()
	defer fasthttp.ReleaseRequest(req)
	req.Header.SetMethod(fasthttp.MethodPost)
	req.SetRequestURI("http://example.invalid/v1/chat")
	req.SetBody([]byte(`{"prompt":"x"}`))
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseResponse(resp)

	_, bifrostErr, wait := MakeRequestWithContext(ctx, client, req, resp)
	wait()
	if bifrostErr == nil {
		t.Fatal("expected transport error")
	}
	if len(observer.requests) != 1 {
		t.Fatalf("request must still be observed, got %d", len(observer.requests))
	}
	if got := observer.requestBodyBytes(); !bytes.Equal(got, []byte(`{"prompt":"x"}`)) {
		t.Fatalf("request body not observed: %q", got)
	}
	if len(observer.responses) != 0 {
		t.Fatalf("no response must be emitted, got %#v", observer.responses)
	}
	if observer.completionCount() != 1 {
		t.Fatalf("expected exactly one completion, got %d", observer.completionCount())
	}
	if observer.firstCompletionErr() == nil {
		t.Fatal("no-response completion must carry the transport error")
	}
	if got := observer.lastTermination(); got != TerminationNoResponse {
		t.Fatalf("expected no_response termination, got %q", got)
	}
}

func TestObserverFailureDoesNotChangeDataPlane(t *testing.T) {
	responseBody := []byte(`{"ok":true}`)
	client, cleanup := startServer(t, func(ctx *fasthttp.RequestCtx) {
		ctx.SetBody(responseBody)
	})
	defer cleanup()

	observer := &recordingObserver{panicMode: true}
	ctx := newObserverContext(observer)
	req := fasthttp.AcquireRequest()
	defer fasthttp.ReleaseRequest(req)
	req.Header.SetMethod(fasthttp.MethodPost)
	req.SetRequestURI("http://example.invalid/v1/chat")
	req.SetBody([]byte("body"))
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseResponse(resp)

	_, bifrostErr, wait := MakeRequestWithContext(ctx, client, req, resp)
	wait()
	if bifrostErr != nil {
		t.Fatalf("observer panic changed the result: %v", bifrostErr)
	}
	if got := resp.Body(); !bytes.Equal(got, responseBody) {
		t.Fatalf("body changed: %q", got)
	}
	if resp.StatusCode() != 200 {
		t.Fatalf("status changed: %d", resp.StatusCode())
	}
}

func TestRedirectChainOnlyInitialAndFinalAreObservable(t *testing.T) {
	// Pins the present-but-unreachable redirect helper (gpt-load plan decision
	// R-B, 2026-09-12). fasthttp runs the redirect loop inside unexported
	// doRequestFollowRedirects, so a Bifrost-only fork cannot emit per-hop
	// attempts. Both Bifrost call sites are unreachable from the current
	// gpt-load production route registry, so this is not an omitted production
	// attempt: the initial request and final response are observed, intermediate
	// 3xx hops are explicitly not claimed, and this test fixes that behavior so
	// it can never be silently upgraded to "covered".
	hops := 0
	client, cleanup := startServer(t, func(ctx *fasthttp.RequestCtx) {
		if string(ctx.Path()) == "/start" {
			hops++
			ctx.Response.Header.Set("Location", "/next")
			ctx.SetStatusCode(fasthttp.StatusFound)
			return
		}
		hops++
		ctx.SetStatusCode(200)
		ctx.SetBody([]byte("final"))
	})
	defer cleanup()

	observer := &recordingObserver{}
	ctx := newObserverContext(observer)
	req := fasthttp.AcquireRequest()
	defer fasthttp.ReleaseRequest(req)
	req.Header.SetMethod(fasthttp.MethodGet)
	req.SetRequestURI("http://example.invalid/start")
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseResponse(resp)

	_, bifrostErr, wait := MakeRequestWithContextFollowRedirects(ctx, client, req, resp, 3)
	wait()
	if bifrostErr != nil {
		t.Fatalf("unexpected error: %v", bifrostErr)
	}
	if hops != 2 {
		t.Fatalf("expected 2 real HTTP sends, got %d", hops)
	}
	if len(observer.responses) != 1 {
		t.Fatalf("B1: intermediate 3xx hops are not observable, got %d response events", len(observer.responses))
	}
	if observer.responses[0].status != 200 {
		t.Fatalf("only the final response is observable, got %d", observer.responses[0].status)
	}
	if got := observer.bodyBytes(); !bytes.Equal(got, []byte("final")) {
		t.Fatalf("final body mismatch: %q", got)
	}
	if observer.completionCount() != 1 {
		t.Fatalf("expected exactly one completion, got %d", observer.completionCount())
	}
}

func TestNoRedirectIsASingleAttempt(t *testing.T) {
	client, cleanup := startServer(t, func(ctx *fasthttp.RequestCtx) {
		ctx.SetBody([]byte("ok"))
	})
	defer cleanup()

	observer := &recordingObserver{}
	ctx := newObserverContext(observer)
	req := fasthttp.AcquireRequest()
	defer fasthttp.ReleaseRequest(req)
	req.Header.SetMethod(fasthttp.MethodGet)
	req.SetRequestURI("http://example.invalid/plain")
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseResponse(resp)

	_, bifrostErr, wait := MakeRequestWithContextFollowRedirects(ctx, client, req, resp, 3)
	wait()
	if bifrostErr != nil {
		t.Fatalf("unexpected error: %v", bifrostErr)
	}
	if len(observer.responses) != 1 || observer.completionCount() != 1 {
		t.Fatalf("expected one response and one completion, got %d/%d", len(observer.responses), observer.completionCount())
	}
}

func TestIdleTimeoutTerminationIsReported(t *testing.T) {
	baseURL, cleanup := startTCPServer(t, func(ctx *fasthttp.RequestCtx) {
		ctx.Response.Header.SetContentType("text/event-stream")
		ctx.SetBodyStreamWriter(func(w *bufio.Writer) {
			_, _ = w.Write([]byte("data: first\n\n"))
			_ = w.Flush()
			time.Sleep(500 * time.Millisecond)
		})
	})
	defer cleanup()

	observer := &recordingObserver{}
	ctx := newObserverContext(observer)
	client := &fasthttp.Client{}
	req := fasthttp.AcquireRequest()
	defer fasthttp.ReleaseRequest(req)
	req.Header.SetMethod(fasthttp.MethodPost)
	req.SetRequestURI(baseURL + "/v1/chat/stream")
	resp := fasthttp.AcquireResponse()
	resp.StreamBody = true

	if err := DoStreamingRequest(ctx, client, req, resp); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	reader, releaseGzip := DecompressStreamBody(ctx, resp)
	defer releaseGzip()
	rawStream := resp.BodyStream()
	reader, stopIdle := NewIdleTimeoutReader(reader, rawStream, 50*time.Millisecond, ctx)
	defer stopIdle()

	buffer := make([]byte, 1024)
	var err error
	for {
		_, err = reader.Read(buffer)
		if err != nil {
			break
		}
	}
	if !errors.Is(err, ErrStreamIdleTimeout) {
		t.Fatalf("expected idle timeout, got %v", err)
	}
	if observer.completionCount() != 1 {
		t.Fatalf("expected exactly one completion, got %d", observer.completionCount())
	}
	if got := observer.lastTermination(); got != TerminationIdleTimeout {
		t.Fatalf("expected idle_timeout termination, got %q", got)
	}
}

func TestCancellationTerminationIsReported(t *testing.T) {
	baseURL, cleanup := startTCPServer(t, func(ctx *fasthttp.RequestCtx) {
		ctx.Response.Header.SetContentType("text/event-stream")
		ctx.SetBodyStreamWriter(func(w *bufio.Writer) {
			_, _ = w.Write([]byte("data: first\n\n"))
			_ = w.Flush()
			time.Sleep(2 * time.Second)
		})
	})
	defer cleanup()

	observer := &recordingObserver{}
	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	defer cancel()
	ctx.SetValue(httpObserverContextKey, observer)
	ctx.SetValue(httpAttemptIDContextKey, testAttemptID)

	client := &fasthttp.Client{}
	req := fasthttp.AcquireRequest()
	defer fasthttp.ReleaseRequest(req)
	req.Header.SetMethod(fasthttp.MethodPost)
	req.SetRequestURI(baseURL + "/v1/chat/stream")
	resp := fasthttp.AcquireResponse()
	resp.StreamBody = true

	if err := DoStreamingRequest(ctx, client, req, resp); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	reader, releaseGzip := DecompressStreamBody(ctx, resp)
	defer releaseGzip()
	stopCancellation := SetupStreamCancellation(ctx, resp.BodyStream(), nil)
	defer stopCancellation()

	readErr := make(chan error, 1)
	go func() {
		buffer := make([]byte, 1024)
		for {
			if _, err := reader.Read(buffer); err != nil {
				readErr <- err
				return
			}
		}
	}()
	time.Sleep(30 * time.Millisecond)
	cancel()

	select {
	case err := <-readErr:
		if err == nil {
			t.Fatal("expected cancellation error")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("read did not unblock after cancellation")
	}
	// Completion is emitted by the read that observes the closed stream; wait
	// for it rather than assuming it already happened.
	deadline := time.Now().Add(2 * time.Second)
	for observer.completionCount() == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if observer.completionCount() != 1 {
		t.Fatalf("expected exactly one completion, got %d", observer.completionCount())
	}
	if got := observer.lastTermination(); got != TerminationCancel {
		t.Fatalf("expected cancel termination, got %q", got)
	}
}

// TestStreamTerminationClassification pins the termination labels for the
// non-EOF body endings without depending on transport-specific error values.
func TestStreamTerminationClassification(t *testing.T) {
	cases := []struct {
		name        string
		err         error
		termination string
	}{
		{"idle timeout", ErrStreamIdleTimeout, TerminationIdleTimeout},
		{"closed", ErrStreamClosed, TerminationClosed},
		{"cancel", context.Canceled, TerminationCancel},
		{"timeout", context.DeadlineExceeded, TerminationTimeout},
		{"read error", errors.New("boom"), TerminationReadError},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			observer := &recordingObserver{}
			ctx := newObserverContext(observer)
			state := newHTTPObserverState(ctx)
			if state == nil {
				t.Fatal("state must exist")
			}
			reader := state.wrapBody(&failingReader{err: tc.err})
			if _, err := io.ReadAll(reader); !errors.Is(err, tc.err) {
				t.Fatalf("error changed: %v", err)
			}
			if observer.completionCount() != 1 {
				t.Fatalf("expected exactly one completion, got %d", observer.completionCount())
			}
			if observer.firstCompletionErr() == nil {
				t.Fatal("completion must carry the read error")
			}
			if got := observer.lastTermination(); got != tc.termination {
				t.Fatalf("expected %q, got %q", tc.termination, got)
			}
		})
	}
}

type failingReader struct{ err error }

func (r *failingReader) Read([]byte) (int, error) { return 0, r.err }

func TestHTTPObserverWrapsNetHTTPBody(t *testing.T) {
	responseBody := []byte("net-http-body")
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer listener.Close()
	server := &fasthttp.Server{Handler: func(ctx *fasthttp.RequestCtx) {
		ctx.SetBody(responseBody)
	}}
	go func() { _ = server.Serve(listener) }()
	defer server.Shutdown()

	observer := &recordingObserver{}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	ctx.SetValue(httpObserverContextKey, observer)
	ctx.SetValue(httpAttemptIDContextKey, testAttemptID)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://%s/thing", listener.Addr()), nil)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	resp, err := DoHTTPRequest(http.DefaultClient, req)
	if err != nil {
		t.Fatalf("do request: %v", err)
	}
	got, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		t.Fatalf("read body: %v", readErr)
	}
	if closeErr := resp.Body.Close(); closeErr != nil {
		t.Fatalf("close body: %v", closeErr)
	}
	if !bytes.Equal(got, responseBody) {
		t.Fatalf("body changed: %q", got)
	}
	if !bytes.Equal(observer.bodyBytes(), responseBody) {
		t.Fatalf("observed body mismatch: %q", observer.bodyBytes())
	}
	if observer.completionCount() != 1 {
		t.Fatalf("expected exactly one completion, got %d", observer.completionCount())
	}
}

// waitForCompletion polls until the observer has at least n completions or the
// deadline elapses. Completion can be emitted by a teardown owner goroutine
// (cancellation / idle timer), so tests must not assume it is synchronous.
func waitForCompletion(t *testing.T, observer *recordingObserver, n int, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for observer.completionCount() < n && time.Now().Before(deadline) {
		time.Sleep(2 * time.Millisecond)
	}
}

// netHTTPPost opens a real net/http POST against a live fasthttp server so the
// request-body and response-body lifecycles are exercised end to end.
func netHTTPPost(t *testing.T, handler fasthttp.RequestHandler, requestBody []byte) (*recordingObserver, *http.Response, []byte) {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	server := &fasthttp.Server{Handler: handler}
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(func() {
		_ = server.Shutdown()
		_ = listener.Close()
	})

	observer := &recordingObserver{}
	ctx := newObserverContext(observer)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://%s/thing", listener.Addr()), bytes.NewReader(requestBody))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	resp, err := DoHTTPRequest(http.DefaultClient, req)
	if err != nil {
		t.Fatalf("do request: %v", err)
	}
	return observer, resp, requestBody
}

// Finding 1: the net/http request-body wrapper must only record request bytes.
// It must not complete the response state when net/http closes the request body
// (which it does right after writing it, before the response is consumed).
func TestNetHTTPPostSuccessCompletesOnResponseEOF(t *testing.T) {
	requestBody := []byte(`{"prompt":"hello world"}`)
	responseBody := []byte(`{"ok":true}`)

	observer, resp, _ := netHTTPPost(t, func(ctx *fasthttp.RequestCtx) {
		if !bytes.Equal(ctx.PostBody(), requestBody) {
			t.Errorf("server saw body %q", ctx.PostBody())
		}
		ctx.Response.Header.Set("X-Test", "yes")
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetBody(responseBody)
	}, requestBody)

	got, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		t.Fatalf("read body: %v", readErr)
	}
	if closeErr := resp.Body.Close(); closeErr != nil {
		t.Fatalf("close body: %v", closeErr)
	}

	if !bytes.Equal(observer.requestBodyBytes(), requestBody) {
		t.Fatalf("request body not observed on its own channel: %q", observer.requestBodyBytes())
	}
	if len(observer.responses) != 1 || observer.responses[0].status != fasthttp.StatusOK {
		t.Fatalf("response status not observed: %#v", observer.responses)
	}
	if got := observer.responses[0].headers.Get("X-Test"); got != "yes" {
		t.Fatalf("response header not observed: %q", got)
	}
	if !bytes.Equal(got, responseBody) || !bytes.Equal(observer.bodyBytes(), responseBody) {
		t.Fatalf("response body mismatch: read %q observed %q", got, observer.bodyBytes())
	}
	if observer.completionCount() != 1 {
		t.Fatalf("request-body Close must not complete the state; got %d completions (termination %q)",
			observer.completionCount(), observer.lastTermination())
	}
	if err := observer.firstCompletionErr(); err != nil {
		t.Fatalf("expected eof completion, got %v", err)
	}
	if got := observer.lastTermination(); got != TerminationEOF {
		t.Fatalf("expected eof termination, got %q", got)
	}
}

func TestNetHTTPResponseErrorLifecycle(t *testing.T) {
	requestBody := []byte(`{"prompt":"hello"}`)
	responseBody := []byte(`{"error":"boom"}`)

	observer, resp, _ := netHTTPPost(t, func(ctx *fasthttp.RequestCtx) {
		ctx.SetStatusCode(fasthttp.StatusInternalServerError)
		ctx.SetBody(responseBody)
	}, requestBody)

	got, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		t.Fatalf("read body: %v", readErr)
	}
	_ = resp.Body.Close()

	if len(observer.responses) != 1 || observer.responses[0].status != fasthttp.StatusInternalServerError {
		t.Fatalf("error status not observed: %#v", observer.responses)
	}
	if !bytes.Equal(got, responseBody) || !bytes.Equal(observer.bodyBytes(), responseBody) {
		t.Fatalf("error body mismatch: read %q observed %q", got, observer.bodyBytes())
	}
	if observer.completionCount() != 1 {
		t.Fatalf("expected exactly one completion, got %d", observer.completionCount())
	}
	if err := observer.firstCompletionErr(); err != nil {
		t.Fatalf("expected eof completion, got %v", err)
	}
	if got := observer.lastTermination(); got != TerminationEOF {
		t.Fatalf("expected eof termination, got %q", got)
	}
}

// TestNetHTTPSendErrorIsNoResponseAfterRequestBodyRead drives a raw TCP peer
// that consumes the request (so the request body really is read and observed)
// and then closes without responding: a legal no-response attempt.
func TestNetHTTPSendErrorIsNoResponseAfterRequestBodyRead(t *testing.T) {
	requestBody := []byte(`{"prompt":"send-error"}`)

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer listener.Close()
	serverDone := make(chan struct{})
	go func() {
		defer close(serverDone)
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		_ = conn.SetDeadline(time.Now().Add(3 * time.Second))
		var seen []byte
		buf := make([]byte, 1024)
		for !bytes.Contains(seen, requestBody) {
			n, readErr := conn.Read(buf)
			if n > 0 {
				seen = append(seen, buf[:n]...)
			}
			if readErr != nil {
				return
			}
		}
		// Body fully received: close without ever sending a response.
	}()

	observer := &recordingObserver{}
	ctx := newObserverContext(observer)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://%s/thing", listener.Addr()), bytes.NewReader(requestBody))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	resp, err := DoHTTPRequest(&http.Client{}, req)
	if err == nil {
		if resp != nil && resp.Body != nil {
			_ = resp.Body.Close()
		}
		t.Fatal("expected a send error")
	}
	if resp != nil {
		t.Fatalf("expected no response, got %#v", resp)
	}
	<-serverDone

	if !bytes.Equal(observer.requestBodyBytes(), requestBody) {
		t.Fatalf("request body not observed: %q", observer.requestBodyBytes())
	}
	if len(observer.responses) != 0 {
		t.Fatalf("no response must be emitted, got %#v", observer.responses)
	}
	if observer.completionCount() != 1 {
		t.Fatalf("expected exactly one no-response completion, got %d", observer.completionCount())
	}
	if observer.firstCompletionErr() == nil {
		t.Fatal("no-response completion must carry the transport error")
	}
	if got := observer.lastTermination(); got != TerminationNoResponse {
		t.Fatalf("expected no_response termination, got %q", got)
	}
}

// openNetHTTPStream sends a real net/http streaming request the way the
// Bedrock provider does (DoHTTPRequest, then teardown on resp.Body) and returns
// the observed response after the server has flushed one SSE chunk.
func openNetHTTPStream(t *testing.T, ctx *schemas.BifrostContext) *http.Response {
	t.Helper()
	baseURL, cleanup := startTCPServer(t, func(ctx *fasthttp.RequestCtx) {
		ctx.Response.Header.SetContentType("text/event-stream")
		ctx.SetBodyStreamWriter(func(w *bufio.Writer) {
			_, _ = w.Write(netHTTPStreamChunk)
			_ = w.Flush()
			time.Sleep(2 * time.Second)
		})
	})
	t.Cleanup(cleanup)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/chat/stream", nil)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	resp, err := DoHTTPRequest(&http.Client{}, req)
	if err != nil {
		t.Fatalf("do request: %v", err)
	}
	t.Cleanup(func() { _ = resp.Body.Close() })

	head := make([]byte, len(netHTTPStreamChunk))
	if _, err := io.ReadFull(resp.Body, head); err != nil {
		t.Fatalf("read head: %v", err)
	}
	if !bytes.Equal(head, netHTTPStreamChunk) {
		t.Fatalf("head mismatch: %q", head)
	}
	return resp
}

var netHTTPStreamChunk = []byte("data: first\n\n")

// Finding 2 for the net/http body: an early-returning parser plus a cancel must
// complete exactly once with an accurate, incomplete termination. net/http
// teardown closes resp.Body, so the response-body wrapper is the close owner.
func TestNetHTTPEarlyReturnCancelReportsIncompleteTermination(t *testing.T) {
	observer := &recordingObserver{}
	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	defer cancel()
	ctx.SetValue(httpObserverContextKey, observer)
	ctx.SetValue(httpAttemptIDContextKey, testAttemptID)

	resp := openNetHTTPStream(t, ctx)
	stopCancellation := SetupStreamCancellation(ctx, resp.Body, nil)
	defer stopCancellation()

	cancel()
	waitForCompletion(t, observer, 1, 2*time.Second)

	if got := observer.completionCount(); got != 1 {
		t.Fatalf("expected exactly one completion, got %d", got)
	}
	if observer.firstCompletionErr() == nil {
		t.Fatal("cancel termination must be marked incomplete (non-nil error)")
	}
	if got := observer.lastTermination(); got != TerminationCancel {
		t.Fatalf("expected cancel termination, got %q", got)
	}
	if got := observer.bodyBytes(); !bytes.Equal(got, netHTTPStreamChunk) {
		t.Fatalf("observed bytes changed: %q", got)
	}
}

func TestNetHTTPEarlyReturnDeadlineReportsIncompleteTermination(t *testing.T) {
	observer := &recordingObserver{}
	ctx, cancel := schemas.NewBifrostContextWithTimeout(context.Background(), 400*time.Millisecond)
	defer cancel()
	ctx.SetValue(httpObserverContextKey, observer)
	ctx.SetValue(httpAttemptIDContextKey, testAttemptID)

	resp := openNetHTTPStream(t, ctx)
	stopCancellation := SetupStreamCancellation(ctx, resp.Body, nil)
	defer stopCancellation()

	waitForCompletion(t, observer, 1, 3*time.Second)

	if got := observer.completionCount(); got != 1 {
		t.Fatalf("expected exactly one completion, got %d", got)
	}
	if observer.firstCompletionErr() == nil {
		t.Fatal("deadline termination must be marked incomplete (non-nil error)")
	}
	if got := observer.lastTermination(); got != TerminationTimeout {
		t.Fatalf("expected timeout termination, got %q", got)
	}
}

func TestNetHTTPEarlyReturnIdleTimeoutReportsIncompleteTermination(t *testing.T) {
	observer := &recordingObserver{}
	ctx := newObserverContext(observer)

	resp := openNetHTTPStream(t, ctx)
	reader, stopIdle := NewIdleTimeoutReader(resp.Body, resp.Body, 100*time.Millisecond, ctx)
	defer stopIdle()

	// Drain reads until the idle timer fires; the parser has long since stopped
	// reading, which is the early-return shape.
	buf := make([]byte, 64)
	var err error
	for err == nil {
		_, err = reader.Read(buf)
	}
	if !errors.Is(err, ErrStreamIdleTimeout) {
		t.Fatalf("expected idle timeout from read, got %v", err)
	}
	waitForCompletion(t, observer, 1, 2*time.Second)

	if got := observer.completionCount(); got != 1 {
		t.Fatalf("expected exactly one completion, got %d", got)
	}
	if observer.firstCompletionErr() == nil {
		t.Fatal("idle-timeout termination must be marked incomplete (non-nil error)")
	}
	if got := observer.lastTermination(); got != TerminationIdleTimeout {
		t.Fatalf("expected idle_timeout termination, got %q", got)
	}
}

// Finding 2 for the net/http body: a caller closing the response body before
// EOF must be reported as an incomplete "closed" termination, exactly once.
func TestNetHTTPEarlyCloseReportsIncompleteTermination(t *testing.T) {
	observer := &recordingObserver{}
	ctx := newObserverContext(observer)

	resp := openNetHTTPStream(t, ctx)
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("close body: %v", err)
	}

	if got := observer.completionCount(); got != 1 {
		t.Fatalf("expected exactly one completion, got %d", got)
	}
	if observer.firstCompletionErr() == nil {
		t.Fatal("early close must be marked incomplete (non-nil error)")
	}
	if got := observer.lastTermination(); got != TerminationClosed {
		t.Fatalf("expected closed termination, got %q", got)
	}
}

// streamedAttempt bundles the teardown handles of one streamed response so a
// test can reproduce the production defer order.
type streamedAttempt struct {
	resp             *fasthttp.Response
	reader           io.Reader
	rawStream        io.Reader
	releaseGzip      func()
	stopIdle         func()
	stopCancellation func()
}

func (a *streamedAttempt) teardown(ctx *schemas.BifrostContext) {
	a.stopCancellation()
	a.stopIdle()
	a.releaseGzip()
	ReleaseStreamingResponse(ctx, a.resp)
}

// openStreamedAttempt performs the same wiring the providers use: streaming
// send, decompress, idle-timeout wrapper, cancellation hookup.
func openStreamedAttempt(t *testing.T, ctx *schemas.BifrostContext, baseURL string, idleTimeout time.Duration) *streamedAttempt {
	t.Helper()
	client := &fasthttp.Client{}
	req := fasthttp.AcquireRequest()
	t.Cleanup(func() { fasthttp.ReleaseRequest(req) })
	req.Header.SetMethod(fasthttp.MethodPost)
	req.SetRequestURI(baseURL + "/v1/chat/stream")
	resp := fasthttp.AcquireResponse()
	resp.StreamBody = true
	if err := DoStreamingRequest(ctx, client, req, resp); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	reader, releaseGzip := DecompressStreamBody(ctx, resp)
	rawStream := resp.BodyStream()
	reader, stopIdle := NewIdleTimeoutReader(reader, rawStream, idleTimeout, ctx)
	stopCancellation := SetupStreamCancellation(ctx, rawStream, nil)
	return &streamedAttempt{
		resp:             resp,
		reader:           reader,
		rawStream:        rawStream,
		releaseGzip:      releaseGzip,
		stopIdle:         stopIdle,
		stopCancellation: stopCancellation,
	}
}

func startChunkThenHold(t *testing.T, chunk []byte, hold time.Duration) (string, func()) {
	t.Helper()
	return startTCPServer(t, func(ctx *fasthttp.RequestCtx) {
		ctx.Response.Header.SetContentType("text/event-stream")
		ctx.SetBodyStreamWriter(func(w *bufio.Writer) {
			_, _ = w.Write(chunk)
			_ = w.Flush()
			time.Sleep(hold)
		})
	})
}

// Finding 2: after the parser returns early there may be no pending Read. A
// cancel that closes the stream and causes ReleaseStreamingResponse to skip its
// drain must still emit exactly one incomplete termination.
func TestEarlyReturnCancelReportsIncompleteTermination(t *testing.T) {
	chunk := []byte("data: first\n\n")
	baseURL, cleanup := startChunkThenHold(t, chunk, 2*time.Second)
	defer cleanup()

	observer := &recordingObserver{}
	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	defer cancel()
	ctx.SetValue(httpObserverContextKey, observer)
	ctx.SetValue(httpAttemptIDContextKey, testAttemptID)

	attempt := openStreamedAttempt(t, ctx, baseURL, 5*time.Second)

	head := make([]byte, len(chunk))
	if _, err := io.ReadFull(attempt.reader, head); err != nil {
		t.Fatalf("read head: %v", err)
	}
	// Parser early return: no pending Read from here on.
	cancel()

	waitForCompletion(t, observer, 1, 2*time.Second)
	attempt.teardown(ctx)

	if got := observer.completionCount(); got != 1 {
		t.Fatalf("expected exactly one completion, got %d", got)
	}
	if observer.firstCompletionErr() == nil {
		t.Fatal("cancel termination must be marked incomplete (non-nil error)")
	}
	if got := observer.lastTermination(); got != TerminationCancel {
		t.Fatalf("expected cancel termination, got %q", got)
	}
	if got := observer.bodyBytes(); !bytes.Equal(got, chunk) {
		t.Fatalf("observed bytes changed: %q", got)
	}
}

// Finding 2: deadline exceeded and no pending Read.
func TestEarlyReturnDeadlineReportsIncompleteTermination(t *testing.T) {
	chunk := []byte("data: first\n\n")
	baseURL, cleanup := startChunkThenHold(t, chunk, 2*time.Second)
	defer cleanup()

	observer := &recordingObserver{}
	ctx, cancel := schemas.NewBifrostContextWithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	ctx.SetValue(httpObserverContextKey, observer)
	ctx.SetValue(httpAttemptIDContextKey, testAttemptID)

	attempt := openStreamedAttempt(t, ctx, baseURL, 5*time.Second)

	head := make([]byte, len(chunk))
	if _, err := io.ReadFull(attempt.reader, head); err != nil {
		t.Fatalf("read head: %v", err)
	}
	waitForCompletion(t, observer, 1, 3*time.Second)
	attempt.teardown(ctx)

	if got := observer.completionCount(); got != 1 {
		t.Fatalf("expected exactly one completion, got %d", got)
	}
	if observer.firstCompletionErr() == nil {
		t.Fatal("deadline termination must be marked incomplete (non-nil error)")
	}
	if got := observer.lastTermination(); got != TerminationTimeout {
		t.Fatalf("expected timeout termination, got %q", got)
	}
	if got := observer.bodyBytes(); !bytes.Equal(got, chunk) {
		t.Fatalf("observed bytes changed: %q", got)
	}
}

// Finding 2: idle timeout fires after the parser returned early and nobody is
// reading; the timer owns the close and must also own the completion.
func TestEarlyReturnIdleTimeoutReportsIncompleteTermination(t *testing.T) {
	chunk := []byte("data: first\n\n")
	baseURL, cleanup := startChunkThenHold(t, chunk, 2*time.Second)
	defer cleanup()

	observer := &recordingObserver{}
	ctx := newObserverContext(observer)

	attempt := openStreamedAttempt(t, ctx, baseURL, 100*time.Millisecond)

	head := make([]byte, len(chunk))
	if _, err := io.ReadFull(attempt.reader, head); err != nil {
		t.Fatalf("read head: %v", err)
	}
	waitForCompletion(t, observer, 1, 3*time.Second)
	attempt.teardown(ctx)

	if got := observer.completionCount(); got != 1 {
		t.Fatalf("expected exactly one completion, got %d", got)
	}
	if observer.firstCompletionErr() == nil {
		t.Fatal("idle-timeout termination must be marked incomplete (non-nil error)")
	}
	if got := observer.lastTermination(); got != TerminationIdleTimeout {
		t.Fatalf("expected idle_timeout termination, got %q", got)
	}
	if got := observer.bodyBytes(); !bytes.Equal(got, chunk) {
		t.Fatalf("observed bytes changed: %q", got)
	}
}

// Finding 2: LargeResponseReader.Close is another teardown owner. When another
// owner already marked the connection closed it skips the drain; it must still
// complete the unfinished state without touching the stream again.
func TestLargeResponseReaderCloseAfterOwnerStillCompletes(t *testing.T) {
	observer := &recordingObserver{}
	ctx := newObserverContext(observer)
	state := newHTTPObserverState(ctx)
	state.attachToContext(ctx)
	state.observeResponse(200, http.Header{})

	// Another owner already closed the stream.
	ctx.SetValue(schemas.BifrostContextKeyConnectionClosed, true)

	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseResponse(resp)
	resp.SetBodyStream(io.NopCloser(bytes.NewReader([]byte("unread"))), -1)

	closable := &LargeResponseReader{Reader: bytes.NewReader(nil), Resp: resp, ctx: ctx}
	if err := closable.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	if got := observer.completionCount(); got != 1 {
		t.Fatalf("expected exactly one completion, got %d", got)
	}
	if observer.firstCompletionErr() == nil {
		t.Fatal("closed termination must be marked incomplete (non-nil error)")
	}
	if got := observer.lastTermination(); got != TerminationClosed {
		t.Fatalf("expected closed termination, got %q", got)
	}
}

// Finding 3 helpers.
func openStreamedErrorResponse(t *testing.T, ctx *schemas.BifrostContext, handler fasthttp.RequestHandler) (*fasthttp.Response, func()) {
	t.Helper()
	baseURL, serverCleanup := startTCPServer(t, handler)
	client := &fasthttp.Client{}
	req := fasthttp.AcquireRequest()
	req.Header.SetMethod(fasthttp.MethodPost)
	req.SetRequestURI(baseURL + "/v1/chat/completions")
	req.SetBody([]byte(`{}`))
	resp := fasthttp.AcquireResponse()
	streamingClient := PrepareResponseStreaming(ctx, client, resp)
	_, bifrostErr, wait := MakeRequestWithContext(ctx, streamingClient, req, resp)
	wait()
	fasthttp.ReleaseRequest(req)
	if bifrostErr != nil {
		fasthttp.ReleaseResponse(resp)
		serverCleanup()
		t.Fatalf("send failed: %v", bifrostErr)
	}
	return resp, func() {
		fasthttp.ReleaseResponse(resp)
		serverCleanup()
	}
}

// Finding 3: a streamed error body larger than the 512 KiB parser cap must
// still be observed in full to the real transport termination, while the parser
// copy stays capped.
func TestMaterializeStreamErrorBodyBeyondParserCapKeepsRawComplete(t *testing.T) {
	const capBytes = 512 * 1024
	fixture := bytes.Repeat([]byte("x"), capBytes+4096)

	observer := &recordingObserver{}
	ctx := newObserverContext(observer)
	ctx.SetValue(schemas.BifrostContextKeyLargeResponseThreshold, int64(1))

	resp, cleanup := openStreamedErrorResponse(t, ctx, func(ctx *fasthttp.RequestCtx) {
		ctx.Response.Header.SetContentType("application/json")
		ctx.SetStatusCode(fasthttp.StatusBadRequest)
		ctx.SetBody(fixture)
	})
	defer cleanup()

	MaterializeStreamErrorBody(ctx, resp)

	if got := observer.bodyBytes(); !bytes.Equal(got, fixture) {
		t.Fatalf("raw capture truncated: got %d bytes want %d", len(got), len(fixture))
	}
	if observer.completionCount() != 1 {
		t.Fatalf("expected exactly one completion, got %d", observer.completionCount())
	}
	if err := observer.firstCompletionErr(); err != nil {
		t.Fatalf("expected eof completion, got %v", err)
	}
	if got := observer.lastTermination(); got != TerminationEOF {
		t.Fatalf("expected eof termination, got %q", got)
	}
	if got := resp.Body(); !bytes.Equal(got, fixture[:capBytes]) {
		t.Fatalf("parser copy changed: got %d bytes want %d", len(got), capBytes)
	}
}

func TestMaterializeStreamErrorBodyAtParserCapStillCompletes(t *testing.T) {
	const capBytes = 512 * 1024
	fixture := bytes.Repeat([]byte("y"), capBytes)

	observer := &recordingObserver{}
	ctx := newObserverContext(observer)
	ctx.SetValue(schemas.BifrostContextKeyLargeResponseThreshold, int64(1))

	resp, cleanup := openStreamedErrorResponse(t, ctx, func(ctx *fasthttp.RequestCtx) {
		ctx.Response.Header.SetContentType("application/json")
		ctx.SetStatusCode(fasthttp.StatusBadRequest)
		ctx.SetBody(fixture)
	})
	defer cleanup()

	MaterializeStreamErrorBody(ctx, resp)

	if got := observer.bodyBytes(); !bytes.Equal(got, fixture) {
		t.Fatalf("raw capture truncated: got %d bytes want %d", len(got), len(fixture))
	}
	if observer.completionCount() != 1 {
		t.Fatalf("expected exactly one completion, got %d", observer.completionCount())
	}
	if got := observer.lastTermination(); got != TerminationEOF {
		t.Fatalf("expected eof termination, got %q", got)
	}
	if got := resp.Body(); !bytes.Equal(got, fixture) {
		t.Fatalf("parser copy changed: got %d bytes want %d", len(got), len(fixture))
	}
}

func TestMaterializeStreamErrorBodyNormalEOFKeepsRawComplete(t *testing.T) {
	fixture := []byte(`{"error":{"message":"small error"}}`)

	observer := &recordingObserver{}
	ctx := newObserverContext(observer)
	ctx.SetValue(schemas.BifrostContextKeyLargeResponseThreshold, int64(1))

	resp, cleanup := openStreamedErrorResponse(t, ctx, func(ctx *fasthttp.RequestCtx) {
		ctx.Response.Header.SetContentType("application/json")
		ctx.SetStatusCode(fasthttp.StatusBadRequest)
		ctx.SetBody(fixture)
	})
	defer cleanup()

	MaterializeStreamErrorBody(ctx, resp)

	if got := observer.bodyBytes(); !bytes.Equal(got, fixture) {
		t.Fatalf("raw capture changed: got %q want %q", got, fixture)
	}
	if observer.completionCount() != 1 {
		t.Fatalf("expected exactly one completion, got %d", observer.completionCount())
	}
	if got := observer.lastTermination(); got != TerminationEOF {
		t.Fatalf("expected eof termination, got %q", got)
	}
}

// errorAfterReader delivers data first and then a fixed error.
type errorAfterReader struct {
	data []byte
	err  error
}

func (r *errorAfterReader) Read(p []byte) (int, error) {
	if len(r.data) > 0 {
		n := copy(p, r.data)
		r.data = r.data[n:]
		return n, nil
	}
	return 0, r.err
}

// Finding 3: a read error must be reported as an incomplete termination exactly
// once and must not be swallowed by the parser cap.
func TestMaterializeStreamErrorBodyReadErrorIsIncomplete(t *testing.T) {
	prefix := []byte(`{"error":"part`)
	readErr := errors.New("upstream read failed")

	observer := &recordingObserver{}
	ctx := newObserverContext(observer)
	ctx.SetValue(schemas.BifrostContextKeyLargeResponseThreshold, int64(1))
	state := newHTTPObserverState(ctx)
	state.attachToContext(ctx)

	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseResponse(resp)
	resp.SetBodyStream(&errorAfterReader{data: bytes.Clone(prefix), err: readErr}, -1)

	MaterializeStreamErrorBody(ctx, resp)

	if got := observer.bodyBytes(); !bytes.Equal(got, prefix) {
		t.Fatalf("raw capture changed: got %q want %q", got, prefix)
	}
	if observer.completionCount() != 1 {
		t.Fatalf("expected exactly one completion, got %d", observer.completionCount())
	}
	if observer.firstCompletionErr() == nil {
		t.Fatal("read-error termination must be marked incomplete (non-nil error)")
	}
}
