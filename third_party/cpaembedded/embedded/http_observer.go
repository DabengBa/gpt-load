package embedded

import (
	"context"
	"errors"
	"io"
	"net/http"
	"sync"
)

// HTTPObserver mirrors the root execution contract without importing the root
// module. The nested CPA module reads the same context keys by value. Events
// are queued and delivered in order by the per-execution sink.
type HTTPObserver interface {
	ObserveRequest(string, *http.Request)
	ObserveRequestBody(string, []byte)
	ObserveResponse(string, int, http.Header)
	ObserveResponseBody(string, []byte)
	ObserveResponseComplete(string, http.Header, error)
}

const (
	httpObserverContextKey  = "gpt-load.http-observer"
	httpAttemptIDContextKey = "gpt-load.http-attempt-id"
)

type observerCallback func()

type observerSink struct {
	observer  HTTPObserver
	attemptID string

	mu        sync.Mutex
	queue     []observerCallback
	wake      chan struct{}
	accepting bool
	done      chan struct{}
	finish    sync.Once
}

func newObserverSink(observer HTTPObserver, attemptID string, _ context.Context) *observerSink {
	sink := &observerSink{
		observer: observer, attemptID: attemptID,
		wake: make(chan struct{}, 1), accepting: true, done: make(chan struct{}),
	}
	go sink.run()
	return sink
}

func (sink *observerSink) run() {
	for {
		select {
		case <-sink.wake:
			for {
				sink.mu.Lock()
				if len(sink.queue) == 0 {
					sink.mu.Unlock()
					break
				}
				event := sink.queue[0]
				sink.queue[0] = nil
				sink.queue = sink.queue[1:]
				sink.mu.Unlock()
				sink.invoke(event)
			}
		case <-sink.done:
			return
		}
	}
}

func (sink *observerSink) invoke(event observerCallback) {
	defer func() { _ = recover() }()
	event()
}

func (sink *observerSink) emit(event observerCallback) bool {
	if sink == nil || event == nil {
		return false
	}
	sink.mu.Lock()
	if !sink.accepting {
		sink.mu.Unlock()
		return false
	}
	sink.queue = append(sink.queue, event)
	sink.mu.Unlock()
	select {
	case sink.wake <- struct{}{}:
	default:
	}
	return true
}

func (sink *observerSink) complete(trailers http.Header, err error) {
	if sink == nil {
		return
	}
	trailers = trailers.Clone()
	sink.finish.Do(func() {
		sink.mu.Lock()
		sink.accepting = false
		sink.queue = append(sink.queue, func() {
			defer close(sink.done)
			sink.invoke(func() {
				sink.observer.ObserveResponseComplete(sink.attemptID, trailers, err)
			})
		})
		sink.mu.Unlock()
		select {
		case sink.wake <- struct{}{}:
		default:
		}
	})
}

type observedRequestBody struct {
	body io.ReadCloser
	sink *observerSink
}

func (body *observedRequestBody) Read(buffer []byte) (int, error) {
	count, err := body.body.Read(buffer)
	if count > 0 {
		captured := append([]byte(nil), buffer[:count]...)
		body.sink.emit(func() {
			body.sink.observer.ObserveRequestBody(body.sink.attemptID, captured)
		})
	}
	return count, err
}

func (body *observedRequestBody) Close() error { return body.body.Close() }

type observedResponseBody struct {
	body     io.ReadCloser
	trailers http.Header
	sink     *observerSink
	mu       sync.Mutex
	complete sync.Once
}

func (body *observedResponseBody) Read(buffer []byte) (int, error) {
	body.mu.Lock()
	defer body.mu.Unlock()
	count, err := body.body.Read(buffer)
	if count > 0 {
		captured := append([]byte(nil), buffer[:count]...)
		body.sink.emit(func() {
			body.sink.observer.ObserveResponseBody(body.sink.attemptID, captured)
		})
	}
	if err != nil {
		body.finish(err)
	}
	return count, err
}

func (body *observedResponseBody) Close() error {
	body.mu.Lock()
	defer body.mu.Unlock()
	err := body.body.Close()
	body.complete.Do(func() {
		completionErr := err
		if completionErr == nil {
			completionErr = io.ErrClosedPipe
		}
		body.sink.complete(body.trailers, completionErr)
	})
	return err
}

func (body *observedResponseBody) finish(err error) {
	if errors.Is(err, io.EOF) {
		err = nil
	}
	body.complete.Do(func() {
		body.sink.complete(body.trailers, err)
	})
}

func observeRoundTrip(base http.RoundTripper, request *http.Request, observer HTTPObserver, attemptID string) (*http.Response, error) {
	if base == nil {
		base = http.DefaultTransport
	}
	if observer == nil {
		return base.RoundTrip(request)
	}
	if attemptID == "" && request != nil && request.Context() != nil {
		attemptID, _ = request.Context().Value(httpAttemptIDContextKey).(string)
	}
	sink := newObserverSink(observer, attemptID, request.Context())
	if request.Body != nil {
		request.Body = &observedRequestBody{body: request.Body, sink: sink}
	}
	snapshot := cloneRequestForObserver(request)
	sink.emit(func() { sink.observer.ObserveRequest(sink.attemptID, snapshot) })
	response, err := base.RoundTrip(request)
	if err != nil {
		sink.complete(nil, err)
		return nil, err
	}
	if response == nil {
		sink.complete(nil, io.ErrUnexpectedEOF)
		return nil, io.ErrUnexpectedEOF
	}
	sink.emit(func() { sink.observer.ObserveResponse(sink.attemptID, response.StatusCode, response.Header.Clone()) })
	if response.Body == nil {
		response.Body = io.NopCloser(nilReader{})
	}
	response.Body = &observedResponseBody{body: response.Body, trailers: response.Trailer, sink: sink}
	return response, nil
}

func cloneRequestForObserver(request *http.Request) *http.Request {
	if request == nil {
		return nil
	}
	clone := request.Clone(request.Context())
	clone.Body = nil
	clone.GetBody = nil
	return clone
}

type nilReader struct{}

func (nilReader) Read([]byte) (int, error) { return 0, io.EOF }

func httpObserverFromContext(ctx context.Context) HTTPObserver {
	if ctx == nil {
		return nil
	}
	observer, _ := ctx.Value(httpObserverContextKey).(HTTPObserver)
	return observer
}

func httpAttemptIDFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	attemptID, _ := ctx.Value(httpAttemptIDContextKey).(string)
	return attemptID
}
