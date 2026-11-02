package gateway

import (
	"bytes"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"sync"
)

// captureHTTPObserver turns U003's ordered provider callbacks into store
// appends. The provider transport owns callback scheduling; Wait is the
// barrier before the attempt is terminally marked.
type captureHTTPObserver struct {
	attempt             CaptureAttempt
	expected            string
	mu                  sync.Mutex
	cond                *sync.Cond
	failure             error
	failureSink         func(error)
	active              int
	observations        int
	sourceStarted       bool
	complete            bool
	await               bool
	stopped             bool
	responseObserved    bool
	terminationObserved bool
}

func newCaptureHTTPObserver(attempt CaptureAttempt, expected ...string) *captureHTTPObserver {
	observer := &captureHTTPObserver{attempt: attempt}
	observer.cond = sync.NewCond(&observer.mu)
	if len(expected) > 0 {
		observer.expected = expected[0]
	}
	return observer
}

func (observer *captureHTTPObserver) setFailureSink(sink func(error)) {
	if observer == nil || sink == nil {
		return
	}
	observer.mu.Lock()
	observer.failureSink = sink
	failure := observer.failure
	observer.mu.Unlock()
	if failure != nil {
		notifyCaptureFailure(sink, failure)
	}
}

func notifyCaptureFailure(sink func(error), err error) {
	if sink == nil || err == nil {
		return
	}
	defer func() { _ = recover() }()
	sink(err)
}

// BeginHTTPObservation registers that a provider observation source started.
// AwaitCompletion is scoped to that source: without a registration no callback
// can ever arrive, so waiting for one would leak the forward barrier forever.
func (observer *captureHTTPObserver) BeginHTTPObservation(id string) {
	if observer == nil || observer.attempt == nil || id != observer.expected {
		return
	}
	observer.mu.Lock()
	if observer.stopped {
		observer.mu.Unlock()
		return
	}
	observer.observations++
	observer.sourceStarted = true
	observer.cond.Broadcast()
	observer.mu.Unlock()
}

// EndHTTPObservation releases one registered observation source.
func (observer *captureHTTPObserver) EndHTTPObservation(id string) {
	if observer == nil || id != observer.expected {
		return
	}
	observer.mu.Lock()
	if observer.observations > 0 {
		observer.observations--
	}
	observer.cond.Broadcast()
	observer.mu.Unlock()
}

func (observer *captureHTTPObserver) AwaitCompletion() {
	if observer == nil {
		return
	}
	observer.mu.Lock()
	observer.await = true
	observer.mu.Unlock()
}

func (observer *captureHTTPObserver) failLocked(err error) {
	if err != nil && observer.failure == nil {
		notifyCaptureFailure(observer.failureSink, err)
		observer.failure = err
	}
	observer.complete = true
	observer.cond.Broadcast()
}

func (observer *captureHTTPObserver) NoCompletionSource() {
	if observer == nil {
		return
	}
	observer.mu.Lock()
	if !observer.complete && !observer.stopped {
		observer.complete = true
	}
	observer.cond.Broadcast()
	observer.mu.Unlock()
}

func (observer *captureHTTPObserver) enter(id string) bool {
	if observer == nil || observer.attempt == nil {
		return false
	}
	observer.mu.Lock()
	if observer.expected != "" && id != observer.expected {
		observer.failLocked(fmt.Errorf("capture observer attempt mismatch: got %q want %q", id, observer.expected))
		observer.mu.Unlock()
		return false
	}
	if observer.stopped || observer.complete {
		observer.failLocked(errors.New("capture observer callback arrived after completion"))
		observer.mu.Unlock()
		return false
	}
	observer.active++
	observer.mu.Unlock()
	return true
}

func (observer *captureHTTPObserver) enterResponse(id string) bool {
	if observer == nil || observer.attempt == nil {
		return false
	}
	observer.mu.Lock()
	if observer.expected != "" && id != observer.expected {
		observer.failLocked(fmt.Errorf("capture observer attempt mismatch: got %q want %q", id, observer.expected))
		observer.mu.Unlock()
		return false
	}
	if observer.stopped || observer.complete {
		observer.failLocked(errors.New("capture observer callback arrived after completion"))
		observer.mu.Unlock()
		return false
	}
	observer.active++
	observer.responseObserved = true
	observer.mu.Unlock()
	return true
}

func (observer *captureHTTPObserver) enterTermination(id string) bool {
	if observer == nil || observer.attempt == nil {
		return false
	}
	observer.mu.Lock()
	if observer.expected != "" && id != observer.expected {
		observer.failLocked(fmt.Errorf("capture observer attempt mismatch: got %q want %q", id, observer.expected))
		observer.mu.Unlock()
		return false
	}
	if observer.stopped || observer.complete {
		observer.failLocked(errors.New("capture observer callback arrived after completion"))
		observer.mu.Unlock()
		return false
	}
	observer.active++
	observer.terminationObserved = true
	observer.mu.Unlock()
	return true
}

func (observer *captureHTTPObserver) inferredTermination(err error) (string, string, bool) {
	observer.mu.Lock()
	defer observer.mu.Unlock()
	if observer.terminationObserved {
		return "", "", false
	}
	observer.terminationObserved = true
	if err == nil {
		return "eof", "", true
	}
	if !observer.responseObserved {
		return "no_response", err.Error(), true
	}
	return "read_error", err.Error(), true
}

func (observer *captureHTTPObserver) enterCompletion(id string) bool {
	if observer == nil || observer.attempt == nil {
		return false
	}
	observer.mu.Lock()
	if observer.expected != "" && id != observer.expected {
		observer.failLocked(fmt.Errorf("capture observer attempt mismatch: got %q want %q", id, observer.expected))
		observer.mu.Unlock()
		return false
	}
	if observer.stopped || observer.complete {
		observer.failLocked(errors.New("capture observer completion arrived after completion"))
		observer.mu.Unlock()
		return false
	}
	observer.active++
	// Completion closes admission while already admitted callbacks drain.
	observer.complete = true
	observer.mu.Unlock()
	return true
}

func (observer *captureHTTPObserver) leave() {
	observer.mu.Lock()
	if observer.active > 0 {
		observer.active--
	}
	observer.cond.Broadcast()
	observer.mu.Unlock()
}

func (observer *captureHTTPObserver) recoverCallback() {
	if recovered := recover(); recovered != nil {
		observer.recordFailure(fmt.Errorf("capture observer callback panic: %v", recovered))
	}
}

func (observer *captureHTTPObserver) checkAttempt(id string) bool { return observer.enter(id) }

func (observer *captureHTTPObserver) recordFailure(err error) {
	if observer == nil || err == nil {
		return
	}
	observer.mu.Lock()
	if observer.failure != nil {
		observer.mu.Unlock()
		return
	}
	observer.failure = err
	sink := observer.failureSink
	observer.mu.Unlock()
	notifyCaptureFailure(sink, err)
}

func (observer *captureHTTPObserver) failureValue() error {
	if observer == nil {
		return nil
	}
	observer.mu.Lock()
	defer observer.mu.Unlock()
	return observer.failure
}

func (observer *captureHTTPObserver) ObserveRequest(id string, request *http.Request) {
	if !observer.checkAttempt(id) {
		return
	}
	defer observer.leave()
	defer observer.recoverCallback()
	if request != nil {
		if err := observer.attempt.AppendRequestHeaders(serializeHeaders(request.Header)); err != nil {
			observer.recordFailure(err)
		}
	}
}
func (observer *captureHTTPObserver) ObserveRequestBody(id string, data []byte) {
	if !observer.checkAttempt(id) {
		return
	}
	defer observer.leave()
	defer observer.recoverCallback()
	if len(data) > 0 {
		if err := observer.attempt.AppendRequestBody(bytes.Clone(data)); err != nil {
			observer.recordFailure(err)
		}
	}
}
func (observer *captureHTTPObserver) ObserveResponse(id string, status int, headers http.Header) {
	if !observer.enterResponse(id) {
		return
	}
	defer observer.leave()
	defer observer.recoverCallback()
	data := append([]byte("HTTP "+strconv.Itoa(status)+"\r\n"), serializeHeaders(headers)...)
	if err := observer.attempt.AppendResponseHeaders(data); err != nil {
		observer.recordFailure(err)
	}
}
func (observer *captureHTTPObserver) ObserveResponseBody(id string, data []byte) {
	if !observer.checkAttempt(id) {
		return
	}
	defer observer.leave()
	defer observer.recoverCallback()
	if len(data) > 0 {
		if err := observer.attempt.AppendResponseBody(bytes.Clone(data)); err != nil {
			observer.recordFailure(err)
		}
	}
}
func (observer *captureHTTPObserver) ObserveResponseComplete(id string, headers http.Header, err error) {
	if !observer.enterCompletion(id) {
		return
	}
	defer observer.leave()
	defer observer.recoverCallback()
	if termination, detail, inferred := observer.inferredTermination(err); inferred {
		if events, ok := observer.attempt.(captureResponseTerminationEvents); ok {
			if eventErr := events.RecordResponseTermination(termination, detail); eventErr != nil {
				observer.recordFailure(eventErr)
			}
		}
	}
	if len(headers) > 0 {
		if appendErr := observer.attempt.AppendResponseHeaders(serializeHeaders(headers)); appendErr != nil {
			observer.recordFailure(appendErr)
		}
	}
	if err != nil {
		observer.recordFailure(err)
		if events, ok := observer.attempt.(captureResponseEvents); ok {
			if eventErr := events.RecordResponseError(err); eventErr != nil {
				observer.recordFailure(eventErr)
			}
		}
	}
}

func (observer *captureHTTPObserver) ObserveResponseTermination(id, termination, detail string) {
	if !observer.enterTermination(id) {
		return
	}
	defer observer.leave()
	defer observer.recoverCallback()
	events, ok := observer.attempt.(captureResponseTerminationEvents)
	if !ok {
		return
	}
	if err := events.RecordResponseTermination(termination, detail); err != nil {
		observer.recordFailure(err)
	}
}

func (observer *captureHTTPObserver) Wait() {
	if observer == nil {
		return
	}
	observer.mu.Lock()
	// An open observation source may still deliver queued callbacks, and an
	// admitted callback is still running. Both must drain before terminal state;
	// the source closes after its final completion callback, so this can never
	// wait on a source that will not emit.
	for observer.observations != 0 || observer.active != 0 {
		observer.cond.Wait()
	}
	if observer.await && !observer.complete {
		// Nothing is in flight and every source this attempt started has ended,
		// yet no completion callback arrived. Only the callback can prove what
		// the provider exchanged, so the capture cannot be reported as a
		// complete exchange. Waiting any longer would block on a callback that
		// cannot arrive.
		detail := captureNoSourceNeverStarted
		if observer.sourceStarted {
			detail = captureNoSourceEndedIncomplete
		}
		err := errors.New("capture observer has no completion source: no response completion observed")
		observer.complete = true
		observer.cond.Broadcast()
		observer.mu.Unlock()
		observer.recordNoCompletionSource(detail, err)
		observer.mu.Lock()
		observer.failLocked(err)
	}
	observer.stopped = true
	observer.cond.Broadcast()
	observer.mu.Unlock()
}

// recordNoCompletionSource persists which kind of missing completion this
// attempt hit. It runs outside observer.mu because the store append is an
// arbitrary implementation.
func (observer *captureHTTPObserver) recordNoCompletionSource(detail string, cause error) {
	defer observer.recoverCallback()
	if termination, ok := observer.attempt.(captureResponseTerminationEvents); ok {
		if err := termination.RecordResponseTermination(
			captureTerminationNoSource,
			detail+": "+cause.Error(),
		); err != nil {
			observer.recordFailure(err)
		}
		return
	}
	observer.recordFailure(errors.Join(
		cause,
		errors.New("capture attempt cannot record missing observation source"),
	))
}
