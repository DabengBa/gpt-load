package gemini

import (
	"bytes"
	"context"
	"net"
	"net/http"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// geminiRecordingObserver records the raw HTTP observation stream for the
// gemini streaming error path. It satisfies the fork's providers/utils
// HTTPObserver contract structurally and is attached with the same untyped
// string keys gpt-load uses, which is also a cross-package check of that
// contract.
type geminiRecordingObserver struct {
	mu              sync.Mutex
	statuses        []int
	responseHeaders []http.Header
	responseBodies  []byte
	completions     int
	completionErr   error
	terminations    []string
}

func (o *geminiRecordingObserver) ObserveRequest(string, *http.Request) {}

func (o *geminiRecordingObserver) ObserveRequestBody(string, []byte) {}

func (o *geminiRecordingObserver) ObserveResponse(_ string, status int, headers http.Header) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.statuses = append(o.statuses, status)
	o.responseHeaders = append(o.responseHeaders, headers.Clone())
}

func (o *geminiRecordingObserver) ObserveResponseBody(_ string, data []byte) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.responseBodies = append(o.responseBodies, data...)
}

func (o *geminiRecordingObserver) ObserveResponseComplete(_ string, _ http.Header, err error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.completions++
	o.completionErr = err
}

func (o *geminiRecordingObserver) ObserveResponseTermination(_ string, termination string, _ string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.terminations = append(o.terminations, termination)
}

func (o *geminiRecordingObserver) snapshot() ([]int, []http.Header, []byte, int, error, []string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	headers := make([]http.Header, 0, len(o.responseHeaders))
	for _, h := range o.responseHeaders {
		headers = append(headers, h.Clone())
	}
	return append([]int(nil), o.statuses...), headers, bytes.Clone(o.responseBodies), o.completions, o.completionErr, append([]string(nil), o.terminations...)
}

func TestGeminiStreamErrorBodyIsObserved(t *testing.T) {
	fixture := []byte(`{"error":{"code":400,"message":"bad request from gemini","status":"INVALID_ARGUMENT"}}`)

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	server := &fasthttp.Server{Handler: func(ctx *fasthttp.RequestCtx) {
		ctx.Response.Header.SetContentType("application/json")
		ctx.SetStatusCode(fasthttp.StatusBadRequest)
		ctx.SetBody(fixture)
	}}
	go func() { _ = server.Serve(listener) }()
	defer func() {
		_ = server.Shutdown()
		_ = listener.Close()
	}()

	observer := &geminiRecordingObserver{}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	ctx.SetValue("gpt-load.http-observer", observer)
	ctx.SetValue("gpt-load.http-attempt-id", "gemini-attempt")

	client := &fasthttp.Client{}
	_, bifrostErr := HandleGeminiChatCompletionStream(
		ctx,
		client,
		"http://"+listener.Addr().String()+"/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse",
		[]byte(`{"contents":[{"parts":[{"text":"hi"}]}]}`),
		map[string]string{"Content-Type": "application/json"},
		nil,
		30,
		false,
		false,
		schemas.Gemini,
		"gemini-2.0-flash",
		nil,
		nil,
		nil,
		nil,
	)
	if bifrostErr == nil {
		t.Fatal("expected an error for the 400 response")
	}

	statuses, responseHeaders, responseBodies, completions, completionErr, terminations := observer.snapshot()
	if len(statuses) != 1 || statuses[0] != fasthttp.StatusBadRequest {
		t.Fatalf("observed statuses = %v, want [400]", statuses)
	}
	if !bytes.Equal(responseBodies, fixture) {
		t.Fatalf("streaming error body was not observed from the raw reader\n got %q\nwant %q", responseBodies, fixture)
	}
	if completions != 1 {
		t.Fatalf("expected exactly one completion, got %d", completions)
	}
	if completionErr != nil {
		t.Fatalf("expected eof completion after a fully read error body, got %v", completionErr)
	}
	if len(terminations) != 1 || terminations[0] != "eof" {
		t.Fatalf("terminations = %v, want [eof]", terminations)
	}
	// Headers must be captured before any mutation: the recorded Content-Length is
	// the upstream one (the fixture length), not a recomputed value.
	if len(responseHeaders) != 1 {
		t.Fatalf("expected one recorded response header set, got %d", len(responseHeaders))
	}
	if got := responseHeaders[0].Get("Content-Length"); got != strconv.Itoa(len(fixture)) {
		t.Fatalf("recorded Content-Length = %q, want %d", got, len(fixture))
	}

	// Behaviour preservation: the caller still receives the parsed upstream error
	// built from the same body bytes, and raw-response enrichment is unchanged.
	if bifrostErr.Error == nil || bifrostErr.Error.Message != "bad request from gemini" {
		t.Fatalf("error message changed: %#v", bifrostErr.Error)
	}
	if bifrostErr.Error.Code == nil || *bifrostErr.Error.Code != "400" {
		t.Fatalf("error code changed: %#v", bifrostErr.Error)
	}
	if bifrostErr.ExtraFields.RawResponse != nil {
		t.Fatalf("raw response enrichment changed: %v", bifrostErr.ExtraFields.RawResponse)
	}

	if err := retryProbe(listener.Addr().String()); err != nil {
		t.Fatalf("probe: %v", err)
	}
}

// retryProbe is a tiny liveness helper that proves the test server accepts a
// second connection after the streamed error response (the stream must have
// been closed/released, not leaked).
func retryProbe(addr string) error {
	client := &fasthttp.Client{ReadTimeout: time.Second, WriteTimeout: time.Second}
	req := fasthttp.AcquireRequest()
	defer fasthttp.ReleaseRequest(req)
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseResponse(resp)
	req.SetRequestURI("http://" + addr + "/probe")
	req.Header.SetMethod(fasthttp.MethodGet)
	return client.DoTimeout(req, resp, time.Second)
}
