package utils

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/url"
	"sync"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// HTTPObserver is the raw request/response observer contract shared with the
// gpt-load root module (internal/execution.HTTPObserver) and its nested CPA
// module (third_party/cpaembedded/embedded.HTTPObserver). This fork cannot
// import either of them, so the interface is re-declared here with the exact
// same method set. Because Go interface satisfaction is structural, the
// concrete observer that gpt-load stores in the request context satisfies this
// interface too.
//
// The observer is purely observational: implementations must not block the
// transport, and any failure in an implementation must be isolated by the
// implementation itself (gpt-load already recovers and records observer
// failures). This fork additionally recovers panics around every callback so a
// broken observer can never change Bifrost data-plane behavior.
type HTTPObserver interface {
	ObserveRequest(attemptID string, request *http.Request)
	ObserveRequestBody(attemptID string, data []byte)
	ObserveResponse(attemptID string, status int, headers http.Header)
	ObserveResponseBody(attemptID string, data []byte)
	ObserveResponseComplete(attemptID string, headers http.Header, err error)
}

// HTTPObserverTermination is an optional extension. When the observer
// implements it, the fork reports how the observed body stream ended. The base
// five-method contract is unchanged, so the CPA adapter and any other existing
// observer keep working.
type HTTPObserverTermination interface {
	ObserveResponseTermination(attemptID string, termination string, detail string)
}

// Termination values reported through HTTPObserverTermination.
const (
	TerminationEOF            = "eof"
	TerminationClosed         = "closed"
	TerminationCancel         = "cancel"
	TerminationTimeout        = "timeout"
	TerminationIdleTimeout    = "idle_timeout"
	TerminationUnexpectedEOF  = "unexpected_eof"
	TerminationReadError      = "read_error"
	TerminationNoResponse     = "no_response"
	TerminationTransportError = "transport_error"
)

// String keys, not private types: gpt-load stores the observer under the
// untyped string constant execution.HTTPObserverContextKey and the attempt ID
// under execution.HTTPAttemptIDContextKey. Values stored with a `string` key
// are visible to a Value lookup that uses an equal `string` key, which is what
// the nested CPA module already relies on.
const (
	httpObserverContextKey  = "gpt-load.http-observer"
	httpAttemptIDContextKey = "gpt-load.http-attempt-id"
	// httpObserverStateContextKey carries the per-attempt observation handle
	// (not the observer itself) so that the send helper and every later read
	// site share one exactly-once completion guard.
	httpObserverStateContextKey = "gpt-load.http-observer-state"
)

// httpObserverState is the request-scoped observation handle for one real HTTP
// send. It is created by the send helper and attached to the
// *schemas.BifrostContext; every reader that delivers response body bytes
// afterwards (parser readers, error-body materialization, large-response
// readers, the release drain) looks the same handle up and emits through it.
//
// One state per send keeps: request/response events ordered, body bytes emitted
// once each, and exactly one completion even when a parser stops early and the
// release drain later reads the tail.
type httpObserverState struct {
	observer    HTTPObserver
	termination HTTPObserverTermination
	attemptID   string

	mu                 sync.Mutex
	responseHeaders    http.Header
	responseReceived   bool
	completed          bool
	pendingTermination string

	completeOnce sync.Once
}

// newHTTPObserverState returns a state bound to the observer and attempt ID
// carried by ctx, or nil when no observer is active (the overwhelmingly common
// case for direct SDK users and for tests that do not observe).
func newHTTPObserverState(ctx context.Context) *httpObserverState {
	if ctx == nil || typedNilContext(ctx) {
		return nil
	}
	observer, _ := ctx.Value(httpObserverContextKey).(HTTPObserver)
	if observer == nil {
		return nil
	}
	attemptID, _ := ctx.Value(httpAttemptIDContextKey).(string)
	state := &httpObserverState{observer: observer, attemptID: attemptID}
	if termination, ok := observer.(HTTPObserverTermination); ok {
		state.termination = termination
	}
	return state
}

// attachToContext makes the state discoverable by later read sites that receive
// the same *schemas.BifrostContext (or a context derived from it). It is a
// no-op when ctx is not the request-scoped Bifrost context, in which case only
// the send helper's own events are emitted (still correct, just no body
// wrappers downstream).
func (s *httpObserverState) attachToContext(ctx context.Context) {
	if s == nil || typedNilContext(ctx) {
		return
	}
	if bifrostContext, ok := ctx.(*schemas.BifrostContext); ok {
		bifrostContext.SetValue(httpObserverStateContextKey, s)
	}
}

// typedNilContext reports whether ctx is a nil *schemas.BifrostContext wrapped
// in a non-nil interface (a common call pattern that would otherwise panic on
// the first Value call).
func typedNilContext(ctx context.Context) bool {
	if bifrostContext, ok := ctx.(*schemas.BifrostContext); ok {
		return bifrostContext == nil
	}
	return false
}

// httpObserverStateFromContext returns the state attached by the send helper,
// if any. It is nil-safe so call sites can stay one-liners.
func httpObserverStateFromContext(ctx context.Context) *httpObserverState {
	if ctx == nil || typedNilContext(ctx) {
		return nil
	}
	if bifrostContext, ok := ctx.(*schemas.BifrostContext); ok {
		if state, ok := bifrostContext.Value(httpObserverStateContextKey).(*httpObserverState); ok {
			return state
		}
		return nil
	}
	state, _ := ctx.Value(httpObserverStateContextKey).(*httpObserverState)
	return state
}

// recoverObserverPanic isolates observer failures from the data plane. A
// panicking observer must never abort a provider request or change its error.
func recoverObserverPanic() {
	_ = recover()
}

func (s *httpObserverState) observeRequest(request *http.Request) {
	if s == nil || request == nil {
		return
	}
	defer recoverObserverPanic()
	s.observer.ObserveRequest(s.attemptID, request)
}

func (s *httpObserverState) observeRequestBody(data []byte) {
	if s == nil || len(data) == 0 {
		return
	}
	defer recoverObserverPanic()
	s.observer.ObserveRequestBody(s.attemptID, data)
}

func (s *httpObserverState) observeResponse(status int, headers http.Header) {
	if s == nil {
		return
	}
	s.mu.Lock()
	if s.responseReceived {
		s.mu.Unlock()
		return
	}
	s.responseReceived = true
	if headers == nil {
		headers = http.Header{}
	}
	s.responseHeaders = headers.Clone()
	s.mu.Unlock()
	defer recoverObserverPanic()
	s.observer.ObserveResponse(s.attemptID, status, headers)
}

func (s *httpObserverState) observeResponseBody(data []byte) {
	if s == nil || len(data) == 0 {
		return
	}
	defer recoverObserverPanic()
	s.observer.ObserveResponseBody(s.attemptID, data)
}

// complete emits the single completion event. Normal transport EOF is reported
// as a nil error (matching the CPA contract); every other termination is
// reported as an error together with its termination label. Headers are not
// repeated here: ObserveResponse already carried them, so re-sending them would
// duplicate them in the capture. Only trailers would be new, and the current
// contract has no trailer channel.
func (s *httpObserverState) complete(err error, termination string) {
	if s == nil {
		return
	}
	s.completeOnce.Do(func() {
		s.mu.Lock()
		s.completed = true
		s.mu.Unlock()
		if s.termination != nil {
			func() {
				defer recoverObserverPanic()
				s.termination.ObserveResponseTermination(s.attemptID, termination, errorDetail(err))
			}()
		}
		defer recoverObserverPanic()
		s.observer.ObserveResponseComplete(s.attemptID, nil, err)
	})
}

// completeIncomplete emits the single completion for an attempt that is torn
// down before its reader reached a natural end. err may be nil, in which case a
// representative error for the termination label is used so the completion is
// still marked incomplete. It is observation-only: it never reads, drains or
// closes the stream, so it cannot delay a close, change cancellation
// propagation or alter the provider data plane. It is a no-op once the state
// already completed.
func (s *httpObserverState) completeIncomplete(err error, termination string) {
	if s == nil {
		return
	}
	if err == nil {
		err = terminationError(termination)
	}
	s.complete(err, termination)
}

// completePending emits an incomplete completion for a stream whose close was
// already claimed by another teardown owner. The pending semantic termination
// (cancel / timeout / idle) is preferred, falling back to a generic closed.
func (s *httpObserverState) completePending() {
	if s == nil {
		return
	}
	s.mu.Lock()
	pending := s.pendingTermination
	s.mu.Unlock()
	if pending == "" {
		pending = TerminationClosed
	}
	s.completeIncomplete(nil, pending)
}

// terminationError returns a representative error for a teardown label so the
// completion is marked incomplete and distinguishable from a natural EOF.
func terminationError(termination string) error {
	switch termination {
	case TerminationCancel:
		return context.Canceled
	case TerminationTimeout:
		return context.DeadlineExceeded
	case TerminationIdleTimeout:
		return ErrStreamIdleTimeout
	case TerminationUnexpectedEOF:
		return io.ErrUnexpectedEOF
	case TerminationReadError:
		return errors.New("stream read error")
	default:
		return ErrStreamClosed
	}
}

func errorDetail(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// markPendingTermination records the semantic reason a stream is about to be
// torn down (idle timeout / cancellation). The raw transport error that follows
// is then labelled with that reason instead of a generic read_error, because the
// observer sits below the layer that performs the mapping.
func (s *httpObserverState) markPendingTermination(termination string) {
	if s == nil || termination == "" {
		return
	}
	s.mu.Lock()
	s.pendingTermination = termination
	s.mu.Unlock()
}

// classify converts a body-read error into the termination label for this
// attempt, preferring an explicit teardown reason recorded by the owner of the
// close (idle-timeout timer / cancellation).
func (s *httpObserverState) classify(err error) string {
	if err == nil || errors.Is(err, io.EOF) {
		return TerminationEOF
	}
	s.mu.Lock()
	pending := s.pendingTermination
	s.mu.Unlock()
	if pending != "" {
		return pending
	}
	return classifyStreamError(err)
}

// classifyStreamError maps a body-read error to a termination label without
// changing the error that flows back to the caller.
func classifyStreamError(err error) string {
	switch {
	case err == nil:
		return TerminationEOF
	case errors.Is(err, io.EOF):
		return TerminationEOF
	case errors.Is(err, ErrStreamIdleTimeout):
		return TerminationIdleTimeout
	case errors.Is(err, ErrStreamClosed):
		return TerminationClosed
	case errors.Is(err, io.ErrClosedPipe):
		return TerminationClosed
	case errors.Is(err, context.Canceled):
		return TerminationCancel
	case errors.Is(err, context.DeadlineExceeded):
		return TerminationTimeout
	case errors.Is(err, fasthttp.ErrTimeout):
		return TerminationTimeout
	case errors.Is(err, io.ErrUnexpectedEOF):
		return TerminationUnexpectedEOF
	default:
		return TerminationReadError
	}
}

// classifyTransportError maps a send-level (no body available) failure.
func classifyTransportError(err error) string {
	switch {
	case err == nil:
		return TerminationTransportError
	case errors.Is(err, context.Canceled):
		return TerminationCancel
	case errors.Is(err, context.DeadlineExceeded):
		return TerminationTimeout
	case errors.Is(err, fasthttp.ErrTimeout):
		return TerminationTimeout
	default:
		return TerminationTransportError
	}
}

// observeFasthttpSendResult records the outcome of a fasthttp send. On success
// the response status/headers are emitted before any Content-Encoding header is
// deleted downstream. When no response was produced (DNS, connect, TLS, send
// failure, timeout, cancellation) the attempt is completed as a legal
// "no response" rather than as a capture failure.
func (s *httpObserverState) observeFasthttpSendResult(resp *fasthttp.Response, err error) {
	if s == nil {
		return
	}
	if err != nil || resp == nil {
		var sendErr error = err
		if sendErr == nil {
			sendErr = io.ErrUnexpectedEOF
		}
		s.complete(sendErr, TerminationNoResponse)
		return
	}
	s.observeResponse(resp.StatusCode(), headerFromFasthttpResponse(&resp.Header))
}

// observeHTTPResponse records a net/http response status/headers. A nil
// response or a non-nil error is a legal no-response attempt.
func (s *httpObserverState) observeHTTPResponse(resp *http.Response, err error) {
	if s == nil {
		return
	}
	if err != nil || resp == nil {
		var sendErr error = err
		if sendErr == nil {
			sendErr = io.ErrUnexpectedEOF
		}
		s.complete(sendErr, TerminationNoResponse)
		return
	}
	s.observeResponse(resp.StatusCode, resp.Header)
}

// observeFasthttpBufferedBody emits a fully buffered fasthttp body. Streamed
// responses (resp.BodyStream() != nil) are deliberately skipped here: their
// bytes are delivered through the reader wrappers so that the parser and the
// release drain both observe the same stream exactly once.
func (s *httpObserverState) observeFasthttpBufferedBody(resp *fasthttp.Response) {
	if s == nil || resp == nil {
		return
	}
	if resp.BodyStream() != nil {
		return
	}
	body := resp.Body()
	if len(body) > 0 {
		s.observeResponseBody(body)
	}
	s.complete(nil, TerminationEOF)
}

// MaterializeObservedStreamBody drains a streamed fasthttp response body into
// resp through the request-scoped raw observer, mirroring fasthttp's
// Response.Body() semantics exactly: transfer framing was already removed by the
// transport, no Content-Encoding decoding and no size cap are applied, a read
// error becomes the body string, the stream is closed once, and the response
// header is left untouched. Buffered responses (BodyStream() == nil) are a
// no-op because the send helper already emitted their bytes.
//
// Call sites that read a streamed body with resp.Body() instead of a reader
// wrapper must call this first, otherwise those bytes bypass the observer.
func MaterializeObservedStreamBody(ctx *schemas.BifrostContext, resp *fasthttp.Response) {
	if resp == nil || resp.BodyStream() == nil {
		return
	}
	bodyStream := httpObserverStateFromContext(ctx).wrapBody(resp.BodyStream())
	body, readErr := io.ReadAll(bodyStream)
	// Close the real stream (never the wrapper) so fasthttp releases the pooled
	// reader/connection exactly once, matching Response.Body().
	resp.CloseBodyStream()
	if readErr != nil {
		body = []byte(readErr.Error())
	}
	resp.SetBody(body)
}

// wrapBody wraps a response body reader at the raw transport observation point:
// after the transport has removed transfer framing (chunked) and before any
// Content-Encoding decoding. The wrapper never implements io.Closer or
// fasthttp's CloseWithError so that closeBodyStream's io.Closer check keeps
// targeting the real body stream and idle-timeout teardown keeps propagating
// ErrStreamIdleTimeout.
func (s *httpObserverState) wrapBody(reader io.Reader) io.Reader {
	if s == nil || reader == nil {
		return reader
	}
	if _, ok := reader.(*observedBodyReader); ok {
		// Already observed (e.g. the same reader is wrapped twice along one
		// chain). Re-wrapping would double-emit every byte.
		return reader
	}
	return &observedBodyReader{state: s, inner: reader}
}

// wrapReadCloser wraps a net/http body. Unlike fasthttp, net/http teardown
// relies on io.Closer, so the wrapper forwards Close while still emitting the
// completion event.
func (s *httpObserverState) wrapReadCloser(reader io.ReadCloser) io.ReadCloser {
	if s == nil || reader == nil {
		return reader
	}
	if existing, ok := reader.(*observedBodyReadCloser); ok {
		return existing
	}
	return &observedBodyReadCloser{observedBodyReader: observedBodyReader{state: s, inner: reader}, closer: reader}
}

type observedBodyReader struct {
	state *httpObserverState
	inner io.Reader
}

func (r *observedBodyReader) Read(p []byte) (int, error) {
	n, err := r.inner.Read(p)
	if n > 0 {
		r.state.observeResponseBody(p[:n])
	}
	if err != nil {
		if errors.Is(err, io.EOF) {
			r.state.complete(nil, TerminationEOF)
		} else {
			r.state.complete(err, r.state.classify(err))
		}
	}
	return n, err
}

type observedBodyReadCloser struct {
	observedBodyReader
	closer io.Closer
}

func (r *observedBodyReadCloser) Close() error {
	err := r.closer.Close()
	// The EOF completion is sync.Once-guarded, so reaching here means the body
	// was closed before its natural end: report an incomplete termination. When a
	// teardown owner (cancellation / idle timer) already recorded the semantic
	// reason, prefer it over a generic closed. Observation-only: the close above
	// is unchanged and its error is still passed through to the caller.
	r.state.completeClosed(err)
	return err
}

// completeClosed emits the single completion for a net/http response body that
// was closed before EOF. It never reads, drains or closes anything itself, and
// it never changes the error Close returns.
func (s *httpObserverState) completeClosed(err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	pending := s.pendingTermination
	s.mu.Unlock()
	if pending != "" {
		s.completeIncomplete(err, pending)
		return
	}
	s.completeIncomplete(err, TerminationClosed)
}

// headerFromFasthttpResponse copies the transport-exposed response header.
func headerFromFasthttpResponse(header *fasthttp.ResponseHeader) http.Header {
	result := make(http.Header, header.Len())
	header.VisitAll(func(key, value []byte) {
		result.Add(string(key), string(value))
	})
	return result
}

// headerFromFasthttpRequest copies the transport-exposed request header.
func headerFromFasthttpRequest(header *fasthttp.RequestHeader) http.Header {
	result := make(http.Header, header.Len())
	header.VisitAll(func(key, value []byte) {
		result.Add(string(key), string(value))
	})
	return result
}

// fasthttpRequestToHTTP builds the *http.Request view of a fasthttp request for
// the observer contract. It carries method, URL, host and header values; it is
// not a wire-level representation.
func fasthttpRequestToHTTP(req *fasthttp.Request) *http.Request {
	if req == nil {
		return nil
	}
	target := &url.URL{Path: string(req.URI().Path())}
	if query := string(req.URI().QueryString()); query != "" {
		target.RawQuery = query
	}
	target.Scheme = string(req.URI().Scheme())
	target.Host = string(req.URI().Host())
	result := &http.Request{
		Method:     string(req.Header.Method()),
		URL:        target,
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		Header:     headerFromFasthttpRequest(&req.Header),
		Host:       string(req.URI().Host()),
	}
	return result
}

// observeFasthttpRequest emits the request line and the request body. All
// gpt-load-reachable requests are buffered (the streamed-request key is never
// set), so the buffered branch is the one that actually runs; the streamed
// branch observes the bytes as the transport pulls them without buffering.
func (s *httpObserverState) observeFasthttpRequest(req *fasthttp.Request) {
	if s == nil || req == nil {
		return
	}
	s.observeRequest(fasthttpRequestToHTTP(req))
	if !req.IsBodyStream() {
		s.observeRequestBody(req.Body())
		return
	}
	if bodyStream := req.BodyStream(); bodyStream != nil {
		req.SetBodyStream(s.wrapBody(bodyStream), req.Header.ContentLength())
	}
}

// observeHTTPRequest emits a net/http request and wraps its body so the bytes
// are observed as they are sent. The body is never buffered. The wrapper is
// request-only: net/http closes req.Body right after writing it, before the
// response is read, so its Close must never emit a response completion.
func (s *httpObserverState) observeHTTPRequest(req *http.Request) {
	if s == nil || req == nil {
		return
	}
	cloned := req.Clone(req.Context())
	cloned.Body = nil
	s.observeRequest(cloned)
	if req.Body != nil {
		if _, ok := req.Body.(*observedRequestBodyReadCloser); ok {
			return
		}
		req.Body = &observedRequestBodyReadCloser{state: s, inner: req.Body}
	}
}

// observedRequestBodyReadCloser observes request-body bytes as they are sent and
// forwards Close untouched. It deliberately does not complete the response
// state: the response lifecycle belongs to the response body reader or the
// transport error, never to net/http closing the request body.
type observedRequestBodyReadCloser struct {
	state *httpObserverState
	inner io.ReadCloser
}

func (r *observedRequestBodyReadCloser) Read(p []byte) (int, error) {
	n, err := r.inner.Read(p)
	if n > 0 {
		r.state.observeRequestBody(p[:n])
	}
	return n, err
}

func (r *observedRequestBodyReadCloser) Close() error {
	return r.inner.Close()
}
