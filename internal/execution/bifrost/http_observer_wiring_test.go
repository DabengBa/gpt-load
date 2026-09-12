package bifrost

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"sync"
	"testing"

	"github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"

	"gpt-load/internal/execution"
)

// wiringObserver records the raw HTTP observation stream produced by the
// forked Bifrost core. It implements the root-module execution.HTTPObserver
// contract and the fork's optional termination extension, which is exactly how
// the gateway observer behaves.
type wiringObserver struct {
	mu             sync.Mutex
	requestBodies  []byte
	requestHeaders []string
	responseStatus []int
	responseBodies []byte
	completions    int
	completionErr  error
	terminations   []string
	responseEvents int
	attemptIDs     map[string]bool
}

func newWiringObserver() *wiringObserver {
	return &wiringObserver{attemptIDs: map[string]bool{}}
}

func (o *wiringObserver) record(id string) {
	o.attemptIDs[id] = true
}

func (o *wiringObserver) ObserveRequest(id string, request *http.Request) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.record(id)
	if request != nil {
		o.requestHeaders = append(o.requestHeaders, request.Header.Get("X-Wire"))
	}
}

func (o *wiringObserver) ObserveRequestBody(id string, data []byte) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.record(id)
	o.requestBodies = append(o.requestBodies, data...)
}

func (o *wiringObserver) ObserveResponse(id string, status int, headers http.Header) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.record(id)
	o.responseEvents++
	o.responseStatus = append(o.responseStatus, status)
}

func (o *wiringObserver) ObserveResponseBody(id string, data []byte) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.record(id)
	o.responseBodies = append(o.responseBodies, data...)
}

func (o *wiringObserver) ObserveResponseComplete(id string, headers http.Header, err error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.record(id)
	o.completions++
	o.completionErr = err
}

func (o *wiringObserver) ObserveResponseTermination(id string, termination string, detail string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.record(id)
	o.terminations = append(o.terminations, termination)
}

func (o *wiringObserver) snapshot() ([]byte, []byte, int, error, []string, map[string]bool) {
	o.mu.Lock()
	defer o.mu.Unlock()
	return bytes.Clone(o.requestBodies), bytes.Clone(o.responseBodies), o.completions, o.completionErr, append([]string(nil), o.terminations...), map[string]bool(o.attemptIDs)
}

func (o *wiringObserver) responseCount() int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.responseEvents
}

func startForkTestServer(t *testing.T, handler fasthttp.RequestHandler) (string, func()) {
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

// TestNewSDKContextExposesObserverUnderSharedStringKeys proves the adapter hands
// the gateway observer and attempt identity to the forked core using the exact
// untyped string keys the fork reads. If either key drifts, this fails.
func TestNewSDKContextExposesObserverUnderSharedStringKeys(t *testing.T) {
	observer := newWiringObserver()
	parent := execution.WithHTTPObserver(context.Background(), observer)
	parent = execution.WithHTTPAttemptID(parent, "attempt-42")

	runtime := &Runtime{}
	sdkContext := runtime.newSDKContext(parent, execution.AttemptSpec{AttemptID: "attempt-42"}, schemas.Key{})

	if got := sdkContext.Value("gpt-load.http-observer"); got == nil {
		t.Fatal("observer not propagated to the SDK context")
	}
	if got, _ := sdkContext.Value("gpt-load.http-attempt-id").(string); got != "attempt-42" {
		t.Fatalf("attempt id = %q, want attempt-42", got)
	}
}

// TestForkObservesThroughSDKContext drives the fork's real send helper with a
// context produced by the adapter, proving cross-module string-key lookup and
// attempt-ID attribution end to end.
func TestForkObservesThroughSDKContext(t *testing.T) {
	requestBody := []byte(`{"prompt":"ping"}`)
	responseBody := []byte(`{"pong":true}`)
	baseURL, cleanup := startForkTestServer(t, func(ctx *fasthttp.RequestCtx) {
		ctx.Response.Header.Set("X-Upstream", "fork-test")
		ctx.SetBody(responseBody)
	})
	defer cleanup()

	observer := newWiringObserver()
	parent := execution.WithHTTPObserver(context.Background(), observer)
	parent = execution.WithHTTPAttemptID(parent, "attempt-7")

	runtime := &Runtime{}
	sdkContext := runtime.newSDKContext(parent, execution.AttemptSpec{AttemptID: "attempt-7"}, schemas.Key{})

	client := &fasthttp.Client{}
	request := fasthttp.AcquireRequest()
	defer fasthttp.ReleaseRequest(request)
	request.Header.SetMethod(fasthttp.MethodPost)
	request.SetRequestURI(baseURL + "/v1/chat/completions")
	request.Header.Set("X-Wire", "present")
	request.SetBody(requestBody)
	response := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseResponse(response)

	_, bifrostErr, wait := utils.MakeRequestWithContext(sdkContext, client, request, response)
	wait()
	if bifrostErr != nil {
		t.Fatalf("unexpected error: %v", bifrostErr)
	}

	requestBytes, responseBytes, completions, completionErr, terminations, attemptIDs := observer.snapshot()
	if !bytes.Equal(requestBytes, requestBody) {
		t.Fatalf("request body = %q, want %q", requestBytes, requestBody)
	}
	if !bytes.Equal(responseBytes, responseBody) {
		t.Fatalf("response body = %q, want %q", responseBytes, responseBody)
	}
	if completions != 1 || completionErr != nil {
		t.Fatalf("completions=%d err=%v", completions, completionErr)
	}
	if len(terminations) != 1 || terminations[0] != utils.TerminationEOF {
		t.Fatalf("terminations = %v", terminations)
	}
	if !attemptIDs["attempt-7"] {
		t.Fatalf("attempt id not attributed: %v", attemptIDs)
	}
}

// TestForkStreamingSSEBytesAndDrainThroughSDKContext is the byte-fidelity gate:
// the parser reader and the observer must see the identical SSE fixture, and a
// drain after an early [DONE] stop must deliver the trailing bytes to the same
// observer exactly once.
func TestForkStreamingSSEBytesAndDrainThroughSDKContext(t *testing.T) {
	fixture := []byte("event: message\n" +
		"data: {\"delta\":\"a\"}\n\n" +
		"data: [DONE]\n\n" +
		"trailing-bytes-after-marker")
	baseURL, cleanup := startForkTestServer(t, func(ctx *fasthttp.RequestCtx) {
		ctx.Response.Header.SetContentType("text/event-stream")
		ctx.SetBodyStreamWriter(func(writer *bufio.Writer) {
			_, _ = writer.Write(fixture)
			_ = writer.Flush()
		})
	})
	defer cleanup()

	observer := newWiringObserver()
	parent := execution.WithHTTPObserver(context.Background(), observer)
	parent = execution.WithHTTPAttemptID(parent, "attempt-stream")

	runtime := &Runtime{}
	sdkContext := runtime.newSDKContext(parent, execution.AttemptSpec{AttemptID: "attempt-stream"}, schemas.Key{})

	client := &fasthttp.Client{}
	request := fasthttp.AcquireRequest()
	defer fasthttp.ReleaseRequest(request)
	request.Header.SetMethod(fasthttp.MethodPost)
	request.SetRequestURI(baseURL + "/v1/chat/completions")
	response := fasthttp.AcquireResponse()
	response.StreamBody = true

	if err := utils.DoStreamingRequest(sdkContext, client, request, response); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	reader, releaseGzip := utils.DecompressStreamBody(sdkContext, response)
	markerEnd := bytes.Index(fixture, []byte("[DONE]\n\n")) + len("[DONE]\n\n")
	head := make([]byte, markerEnd)
	if _, err := io.ReadFull(reader, head); err != nil {
		t.Fatalf("read head: %v", err)
	}
	releaseGzip()

	utils.ReleaseStreamingResponse(sdkContext, response)

	_, responseBytes, completions, completionErr, terminations, attemptIDs := observer.snapshot()
	if !bytes.Equal(responseBytes, fixture) {
		t.Fatalf("observed streaming bytes\n got %q\nwant %q", responseBytes, fixture)
	}
	if completions != 1 {
		t.Fatalf("expected exactly one completion, got %d", completions)
	}
	if completionErr != nil {
		t.Fatalf("expected eof completion, got %v", completionErr)
	}
	if len(terminations) != 1 || terminations[0] != utils.TerminationEOF {
		t.Fatalf("terminations = %v", terminations)
	}
	if !attemptIDs["attempt-stream"] {
		t.Fatalf("attempt id not attributed: %v", attemptIDs)
	}
}

// TestForkObservesConnectionFailureWithoutChangingError proves a legal
// no-response attempt is recorded (request bytes kept, response_received false,
// no_response termination) and the original provider error is returned
// untouched.
func TestForkObservesConnectionFailureWithoutChangingError(t *testing.T) {
	observer := newWiringObserver()
	parent := execution.WithHTTPObserver(context.Background(), observer)
	runtime := &Runtime{}
	sdkContext := runtime.newSDKContext(parent, execution.AttemptSpec{AttemptID: "attempt-fail"}, schemas.Key{})

	client := &fasthttp.Client{Dial: func(string) (net.Conn, error) { return nil, errors.New("dial refused") }}
	request := fasthttp.AcquireRequest()
	defer fasthttp.ReleaseRequest(request)
	request.Header.SetMethod(fasthttp.MethodPost)
	request.SetRequestURI("http://example.invalid/v1/chat")
	request.SetBody([]byte("payload"))
	response := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseResponse(response)

	_, bifrostErr, wait := utils.MakeRequestWithContext(sdkContext, client, request, response)
	wait()
	if bifrostErr == nil {
		t.Fatal("expected provider error")
	}

	requestBytes, responseBytes, completions, completionErr, terminations, _ := observer.snapshot()
	if !bytes.Equal(requestBytes, []byte("payload")) {
		t.Fatalf("request bytes lost: %q", requestBytes)
	}
	if len(responseBytes) != 0 {
		t.Fatalf("unexpected response bytes: %q", responseBytes)
	}
	if completions != 1 || completionErr == nil {
		t.Fatalf("completions=%d err=%v", completions, completionErr)
	}
	if len(terminations) != 1 || terminations[0] != utils.TerminationNoResponse {
		t.Fatalf("terminations = %v", terminations)
	}
	if events := observer.responseCount(); events != 0 {
		t.Fatalf("response events were fabricated: %d", events)
	}
}
