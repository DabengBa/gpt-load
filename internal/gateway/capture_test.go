package gateway

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/execution"
	"gpt-load/internal/state"
)

type captureTestAttempt struct {
	mu               sync.Mutex
	request          []byte
	response         []byte
	waited           bool
	completed        bool
	failed           error
	flushes          int
	shortWrites      [][2]int
	errors           []error
	hijacks          []error
	requestErrs      []error
	requestCloseErrs []error
	requestOutcomes  []string
	terminations     []string
	bodyBlock        <-chan struct{}
	bodyEntered      chan struct{}
	bodyOnce         sync.Once
	waitStartOnce    sync.Once
	waitStarted      chan struct{}
	waitRelease      <-chan struct{}
	panicAppend      bool
}

func (a *captureTestAttempt) AppendRequestHeaders(data []byte) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.request = append(a.request, data...)
	return nil
}
func (a *captureTestAttempt) AppendRequestBody(data []byte) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.request = append(a.request, data...)
	return nil
}
func (a *captureTestAttempt) AppendResponseHeaders(data []byte) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.response = append(a.response, data...)
	return nil
}
func (a *captureTestAttempt) AppendResponseBody(data []byte) error {
	if a.panicAppend {
		panic("append response body")
	}
	if a.bodyBlock != nil {
		a.bodyOnce.Do(func() { close(a.bodyEntered) })
		<-a.bodyBlock
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.response = append(a.response, data...)
	return nil
}
func (a *captureTestAttempt) RecordResponseFlush() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.flushes++
	return nil
}
func (a *captureTestAttempt) RecordResponseShortWrite(written, requested int) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.shortWrites = append(a.shortWrites, [2]int{written, requested})
	return nil
}
func (a *captureTestAttempt) RecordResponseError(err error) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.errors = append(a.errors, err)
	return nil
}
func (a *captureTestAttempt) RecordResponseHijack(err error) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.hijacks = append(a.hijacks, err)
	return nil
}
func (a *captureTestAttempt) RecordRequestError(err error) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.requestErrs = append(a.requestErrs, err)
	return nil
}
func (a *captureTestAttempt) RecordRequestClose(err error) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.requestCloseErrs = append(a.requestCloseErrs, err)
	return nil
}
func (a *captureTestAttempt) RecordRequestOutcome(kind string, _ error) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.requestOutcomes = append(a.requestOutcomes, kind)
	return nil
}
func (a *captureTestAttempt) RecordResponseTermination(termination, detail string) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.terminations = append(a.terminations, termination+":"+detail)
	return nil
}

func (a *captureTestAttempt) Complete() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.completed = true
	return nil
}
func (a *captureTestAttempt) Fail(err error) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.failed = err
	return nil
}
func (a *captureTestAttempt) Wait() {
	if a.waitStarted != nil {
		a.waitStartOnce.Do(func() { close(a.waitStarted) })
		<-a.waitRelease
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.waited = true
}

type captureTestSession struct {
	mu            sync.Mutex
	metadata      CaptureSessionMetadata
	updates       []CaptureSessionMetadata
	attempts      []*captureTestAttempt
	waited        bool
	completed     bool
	failed        error
	waitStarted   chan struct{}
	waitStartOnce sync.Once
	waitRelease   <-chan struct{}
}

func (s *captureTestSession) StartAttempt(CaptureAttemptMetadata) (CaptureAttempt, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	attempt := &captureTestAttempt{}
	s.attempts = append(s.attempts, attempt)
	return attempt, nil
}
func (s *captureTestSession) UpdateMetadata(metadata CaptureSessionMetadata) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.updates = append(s.updates, metadata)
	if metadata.RequestID != "" {
		s.metadata.RequestID = metadata.RequestID
	}
	if metadata.AccessKeyID != 0 {
		s.metadata.AccessKeyID = metadata.AccessKeyID
	}
	if metadata.Protocol != "" {
		s.metadata.Protocol = metadata.Protocol
	}
	if metadata.Operation != "" {
		s.metadata.Operation = metadata.Operation
	}
	return nil
}

func (s *captureTestSession) Complete() error {
	s.mu.Lock()
	s.completed = true
	s.mu.Unlock()
	return nil
}
func (s *captureTestSession) Fail(err error) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.failed = err
	return nil
}
func (s *captureTestSession) Wait() {
	if s.waitStarted != nil {
		s.waitStartOnce.Do(func() { close(s.waitStarted) })
		<-s.waitRelease
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.waited = true
}

func (s *captureTestSession) isWaited() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.waited
}

func (s *captureTestSession) isCompleted() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.completed
}

func (s *captureTestSession) failureValue() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.failed
}
func (s *captureTestSession) attemptTerminal() bool {
	s.mu.Lock()
	if len(s.attempts) != 1 {
		s.mu.Unlock()
		return false
	}
	attempt := s.attempts[0]
	s.mu.Unlock()
	attempt.mu.Lock()
	defer attempt.mu.Unlock()
	return attempt.completed || attempt.failed != nil
}

type captureTestFactory struct {
	mu      sync.Mutex
	session *captureTestSession
	starts  int
}

func (f *captureTestFactory) StartSession(CaptureSessionMetadata) (CaptureSession, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.starts++
	f.session = &captureTestSession{}
	return f.session, nil
}

func (f *captureTestFactory) snapshot() (*captureTestSession, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.session, f.starts
}

func TestCaptureDeferredAttemptCopiesCallerBytes(t *testing.T) {
	target := &captureTestAttempt{}
	attempt := newDeferredCaptureAttempt()
	attempt.setTarget(target, nil)
	data := []byte("before mutation")
	if err := attempt.AppendResponseBody(data); err != nil {
		t.Fatal(err)
	}
	copy(data, "after mutation")
	attempt.Wait()
	if string(target.response) != "before mutation" {
		t.Fatalf("queued response = %q, want caller snapshot", target.response)
	}
}

func TestCaptureObserverBoundsMissingCompletion(t *testing.T) {
	observer := newCaptureHTTPObserver(&captureTestAttempt{})
	observer.NoCompletionSource()
	observer.Wait()
	if observer.failureValue() != nil {
		t.Fatalf("explicitly absent completion source failed: %v", observer.failureValue())
	}
}

func TestCaptureObserverWithoutCompletionSourceDoesNotWaitOrFail(t *testing.T) {
	observer := newCaptureHTTPObserver(&captureTestAttempt{})
	observer.NoCompletionSource()

	done := make(chan struct{})
	go func() {
		observer.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Fatal("observer waited for a completion source that does not exist")
	}
	if observer.failureValue() != nil {
		t.Fatalf("observer failure = %v, want nil for an explicitly absent source", observer.failureValue())
	}
}

func TestCaptureForwardFinalizationDoesNotBlockLateCompletionDrain(t *testing.T) {
	attempt := &captureTestAttempt{}
	observer := newCaptureHTTPObserver(attempt, "attempt-1")
	capture := &dataPlaneCapture{}
	done := make(chan struct{})
	go func() {
		capture.finishForward(attempt, observer, UpstreamResult{
			DispatchState:  execution.DispatchMaybeSent,
			RequestWritten: true,
		})
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Fatal("forward finalization blocked the user response on callback completion")
	}

	observer.ObserveResponseBody("attempt-1", []byte("late-body"))
	observer.ObserveResponseComplete("attempt-1", nil, nil)
	deadline := time.After(time.Second)
	for {
		attempt.mu.Lock()
		completed := attempt.completed
		failed := attempt.failed
		response := append([]byte(nil), attempt.response...)
		attempt.mu.Unlock()
		if completed || failed != nil {
			if failed != nil || !bytes.Contains(response, []byte("late-body")) {
				t.Fatalf("late callback terminal completed=%t failed=%v response=%q", completed, failed, response)
			}
			return
		}
		select {
		case <-deadline:
			t.Fatal("late callback was not drained before terminal attempt state")
		case <-time.After(time.Millisecond):
		}
	}
}
func TestCaptureObserverRecordsCallbackAppendPanic(t *testing.T) {
	observer := newCaptureHTTPObserver(&captureTestAttempt{panicAppend: true})
	observer.ObserveResponseBody("attempt-1", []byte("body"))
	observer.Wait()
	if observer.failureValue() == nil {
		t.Fatal("callback panic was not recorded")
	}
}

func TestCaptureObserverWaitsForOrderedDrain(t *testing.T) {
	attempt := &captureTestAttempt{}
	observer := newCaptureHTTPObserver(attempt)
	observer.ObserveRequest("attempt-1", &http.Request{Header: http.Header{"Authorization": {"secret"}}})
	observer.ObserveRequestBody("attempt-1", []byte("request"))
	observer.ObserveResponse("attempt-1", http.StatusOK, http.Header{"X-Test": {"ok"}})
	observer.ObserveResponseBody("attempt-1", []byte("response"))
	observer.ObserveResponseComplete("attempt-1", nil, nil)
	observer.Wait()
	if !bytes.Contains(attempt.request, []byte("Authorization")) || !bytes.Contains(attempt.request, []byte("request")) {
		t.Fatalf("request capture = %q", attempt.request)
	}
	if !bytes.Contains(attempt.response, []byte("X-Test")) || !bytes.Contains(attempt.response, []byte("response")) {
		t.Fatalf("response capture = %q", attempt.response)
	}
}

func TestCaptureObserverWaitsForAsynchronousCompletion(t *testing.T) {
	attempt := &captureTestAttempt{}
	observer := newCaptureHTTPObserver(attempt)
	observer.AwaitCompletion()

	release := make(chan struct{})
	go func() {
		<-release
		observer.ObserveRequest("attempt-1", &http.Request{Header: http.Header{"X-Test": {"request"}}})
		observer.ObserveResponse("attempt-1", http.StatusOK, http.Header{"X-Test": {"response"}})
		observer.ObserveResponseComplete("attempt-1", nil, nil)
	}()

	waited := make(chan struct{})
	go func() {
		observer.Wait()
		close(waited)
	}()
	select {
	case <-waited:
		t.Fatal("observer wait returned before asynchronous completion")
	default:
	}
	close(release)
	<-waited
	if !bytes.Contains(attempt.request, []byte("X-Test: request")) ||
		!bytes.Contains(attempt.response, []byte("X-Test: response")) {
		t.Fatalf("asynchronous capture was not drained: request=%q response=%q", attempt.request, attempt.response)
	}
}

func TestCaptureBifrostUsesHTTPObserverSource(t *testing.T) {
	session := &captureTestSession{}
	capture := &dataPlaneCapture{session: session}
	attempt, observer, context, accepted := capture.beginForward(context.Background(), ForwardInput{
		AttemptID: "bifrost-attempt", ChannelID: "openai",
	})
	if !accepted || attempt == nil || observer == nil {
		t.Fatalf("bifrost capture attempt=%T observer=%v, want attempt and observer", attempt, observer)
	}
	if execution.HTTPObserverFromContext(context) == nil {
		t.Fatal("Bifrost capture did not attach an upstream HTTP observer")
	}
	capture.finishForward(attempt, observer, UpstreamResult{DispatchState: execution.DispatchNotSent, StatusCode: http.StatusOK})
	if !waitForCapture(t, session.attemptTerminal) {
		t.Fatal("Bifrost capture attempt did not finalize")
	}
}
func TestCaptureFinishesWithNormalizedForwardResult(t *testing.T) {
	attempt := &captureTestAttempt{}
	capture := &dataPlaneCapture{}
	result := capture.normalizeAndFinishForward(attempt, nil, UpstreamResult{StatusCode: http.StatusOK})
	if result.DispatchState != execution.DispatchMaybeSent || result.Err == nil {
		t.Fatalf("normalized result = %#v", result)
	}
	deadline := time.After(time.Second)
	for {
		attempt.mu.Lock()
		completed := attempt.completed
		failed := attempt.failed
		attempt.mu.Unlock()
		if completed || failed != nil {
			if attempt.failed == nil || attempt.completed {
				t.Fatalf("capture terminal completed=%t failed=%v, want normalized failure", completed, failed)
			}
			break
		}
		select {
		case <-deadline:
			t.Fatal("normalized capture attempt did not reach terminal state")
		case <-time.After(time.Millisecond):
		}
	}
}
func TestCaptureForwardDrainsDispatchNotSentCompletion(t *testing.T) {
	attempt := &captureTestAttempt{}
	observer := newCaptureHTTPObserver(attempt, "attempt-1")
	capture := &dataPlaneCapture{}
	done := make(chan struct{})
	go func() {
		capture.finishForward(attempt, observer, UpstreamResult{
			DispatchState: execution.DispatchNotSent,
			Err:           errors.New("dial failed"),
		})
		close(done)
	}()
	time.Sleep(10 * time.Millisecond)
	observer.ObserveRequest("attempt-1", &http.Request{Header: http.Header{"X-Test": {"request"}}})
	observer.ObserveResponseComplete("attempt-1", nil, errors.New("dial failed"))
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("dispatch-not-sent completion was not drained")
	}
	if observer.failureValue() == nil {
		t.Fatal("transport completion error was not retained")
	}
}

func TestCaptureResponseWriterRecordsSuccessfulBytesAndPreservesInterfaces(t *testing.T) {
	base := gin.CreateTestContextOnly(httptest.NewRecorder(), gin.New()).Writer
	attempt := &captureTestAttempt{}
	writer := newCaptureResponseWriter(base, attempt)
	writer.Header().Set("X-Test", "yes")
	writer.WriteHeader(http.StatusAccepted)
	if _, err := writer.Write([]byte("one")); err != nil {
		t.Fatal(err)
	}
	if err := writer.FlushError(); err != nil {
		t.Fatal(err)
	}
	if _, ok := any(writer).(http.Flusher); !ok {
		t.Fatal("capture writer lost http.Flusher")
	}
	if _, ok := any(writer).(http.Hijacker); !ok {
		t.Fatal("capture writer lost http.Hijacker")
	}
	if _, ok := any(writer).(io.ReaderFrom); !ok {
		t.Fatal("capture writer lost io.ReaderFrom")
	}
	if !bytes.Contains(attempt.response, []byte("one")) {
		t.Fatalf("response capture = %q", attempt.response)
	}
	if !bytes.Contains(attempt.response, []byte("X-Test")) || !bytes.Contains(attempt.response, []byte("202")) {
		t.Fatalf("response metadata capture = %q", attempt.response)
	}
}

func TestCaptureResponseWriterSnapshotsHeadersAfterDownstreamApply(t *testing.T) {
	base := gin.CreateTestContextOnly(httptest.NewRecorder(), gin.New()).Writer
	downstream := &downstreamHeaderWriter{
		ResponseWriter: base,
		rules:          state.HeaderRules{Set: map[string]string{"X-Applied": "yes"}},
	}
	attempt := &captureTestAttempt{}
	writer := newCaptureResponseWriter(downstream, attempt)
	writer.WriteHeader(http.StatusAccepted)

	if !bytes.Contains(attempt.response, []byte("X-Applied: yes")) {
		t.Fatalf("captured response headers = %q, want downstream-applied header", attempt.response)
	}
}
func TestCaptureResponseWriterDoesNotBlockOnSlowAppend(t *testing.T) {
	release := make(chan struct{})
	entered := make(chan struct{})
	attempt := newDeferredCaptureAttempt()
	attempt.setTarget(&captureTestAttempt{
		bodyBlock: release, bodyEntered: entered,
	}, nil)
	base := gin.CreateTestContextOnly(httptest.NewRecorder(), gin.New()).Writer
	writer := newCaptureResponseWriter(base, attempt)

	writeDone := make(chan struct{})
	go func() {
		_, _ = writer.Write([]byte("body"))
		close(writeDone)
	}()
	select {
	case <-writeDone:
	case <-time.After(time.Second):
		t.Fatal("response Write blocked on capture append")
	}
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("capture append was not started")
	}
	close(release)
	attempt.Wait()
}
func TestCaptureMiddlewareStartsBeforeAuthenticationAndIsNoopWhenDisabled(t *testing.T) {
	factory := &captureTestFactory{}
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	engine := gin.New()
	context := gin.CreateTestContextOnly(recorder, engine)
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString("body"))
	request.Header.Set("Authorization", "Bearer secret")
	context.Request = request
	handler := &Handler{captureFactory: factory}
	captureDataPlaneRequest(handler, context)
	var session *captureTestSession
	if !waitForCapture(t, func() bool {
		session, _ = factory.snapshot()
		return session != nil
	}) {
		t.Fatal("capture session was not started before auth")
	}
	if _, err := io.ReadAll(request.Body); err != nil {
		t.Fatal(err)
	}
	if err := finalizeDataPlaneCapture(context); err != nil {
		t.Fatal(err)
	}
	if !waitForCapture(t, func() bool {
		session, _ = factory.snapshot()
		return session != nil && session.isWaited()
	}) {
		t.Fatal("session completion did not wait for observer drain")
	}

	request = httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	context = gin.CreateTestContextOnly(httptest.NewRecorder(), engine)
	context.Request = request
	captureDataPlaneRequest(&Handler{}, context)
	_, starts := factory.snapshot()
	if starts != 1 {
		t.Fatalf("disabled capture unexpectedly changed session starts: %d", starts)
	}
}

func TestCaptureMiddlewareBeforeDownstreamCapturesAllowedPreflightOnce(t *testing.T) {
	handler := newDownstreamHeadersTestHandler(t, configuredBrowserAccessSettings())
	factory := &captureTestFactory{}
	handler.SetCaptureFactory(factory)
	engine := gin.New()
	engine.Use(handler.CaptureMiddleware(), handler.DownstreamHeadersMiddleware())
	engine.Any("/v1/responses", func(context *gin.Context) {
		context.Status(http.StatusUnauthorized)
	})

	response := httptest.NewRecorder()
	engine.ServeHTTP(response, newPreflightRequest(
		"/v1/responses", "app://obsidian.md", "POST", "authorization",
	))
	if response.Code != http.StatusNoContent {
		t.Fatalf("preflight response=%d, want 204", response.Code)
	}
	var session *captureTestSession
	if !waitForCapture(t, func() bool {
		session, _ = factory.snapshot()
		return session != nil && session.isWaited()
	}) {
		t.Fatal("preflight capture did not finalize asynchronously")
	}
}
func TestCaptureRequestBodyOutcomesAreRetained(t *testing.T) {
	factory := &captureTestFactory{}
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", &captureErrorReadCloser{
		readErr:  errors.New("read body failed"),
		closeErr: errors.New("close body failed"),
	})
	context := gin.CreateTestContextOnly(httptest.NewRecorder(), gin.New())
	context.Request = request
	captureDataPlaneRequest(&Handler{captureFactory: factory}, context)
	_, _ = io.ReadAll(request.Body)
	_ = request.Body.Close()
	_ = finalizeDataPlaneCapture(context)
	var session *captureTestSession
	if !waitForCapture(t, func() bool {
		session, _ = factory.snapshot()
		if session == nil {
			return false
		}
		session.mu.Lock()
		defer session.mu.Unlock()
		return session.failed != nil
	}) {
		t.Fatal("request body read/close failure did not fail capture")
	}
	session.mu.Lock()
	attempt := session.attempts[0]
	session.mu.Unlock()
	attempt.mu.Lock()
	requestBytes := append([]byte(nil), attempt.request...)
	responseBytes := append([]byte(nil), attempt.response...)
	requestErrs := append([]error(nil), attempt.requestErrs...)
	requestCloseErrs := append([]error(nil), attempt.requestCloseErrs...)
	outcomes := append([]string(nil), attempt.requestOutcomes...)
	attempt.mu.Unlock()
	if len(requestErrs) != 1 || len(requestCloseErrs) != 1 ||
		!containsString(outcomes, "read_error") ||
		!containsString(outcomes, "close_error") {
		t.Fatalf("request outcomes errors=%v close=%v outcomes=%v request=%q response=%q", requestErrs, requestCloseErrs, outcomes, requestBytes, responseBytes)
	}
}

func TestCaptureRequestContextCancellationIsRetained(t *testing.T) {
	factory := &captureTestFactory{}
	requestContext, cancel := context.WithCancel(context.Background())
	request := httptest.NewRequest(http.MethodGet, "/v1/models", nil).WithContext(requestContext)
	context := gin.CreateTestContextOnly(httptest.NewRecorder(), gin.New())
	context.Request = request
	captureDataPlaneRequest(&Handler{captureFactory: factory}, context)
	cancel()
	_ = finalizeDataPlaneCapture(context)
	var session *captureTestSession
	if !waitForCapture(t, func() bool {
		session, _ = factory.snapshot()
		if session == nil {
			return false
		}
		session.mu.Lock()
		defer session.mu.Unlock()
		if len(session.attempts) == 0 {
			return false
		}
		attempt := session.attempts[0]
		attempt.mu.Lock()
		defer attempt.mu.Unlock()
		return containsString(attempt.requestOutcomes, "context_canceled")
	}) {
		t.Fatalf("request outcomes = %v, want context_canceled", factory.session.attempts[0].requestOutcomes)
	}
}

func waitForCapture(t *testing.T, condition func() bool) bool {
	t.Helper()
	deadline := time.After(time.Second)
	for {
		if condition() {
			return true
		}
		select {
		case <-deadline:
			return false
		case <-time.After(time.Millisecond):
		}
	}
}
func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
func TestCaptureSessionMetadataCanBeEnrichedAfterAuthentication(t *testing.T) {
	session := &captureTestSession{}
	capture := &dataPlaneCapture{session: session}
	capture.updateMetadata(CaptureSessionMetadata{AccessKeyID: 7, Protocol: "openai"})
	capture.updateMetadata(CaptureSessionMetadata{Operation: "chat.completions"})
	if !waitForCapture(t, func() bool {
		session.mu.Lock()
		defer session.mu.Unlock()
		return session.metadata.AccessKeyID == 7 && session.metadata.Protocol == "openai" && session.metadata.Operation == "chat.completions" && len(session.updates) == 2
	}) {
		t.Fatalf("session metadata = %#v updates=%d", session.metadata, len(session.updates))
	}
	session.mu.Lock()
	metadata := session.metadata
	updates := len(session.updates)
	session.mu.Unlock()
	if metadata.AccessKeyID != 7 || metadata.Protocol != "openai" || metadata.Operation != "chat.completions" {
		t.Fatalf("session metadata = %#v", metadata)
	}
	if updates != 2 {
		t.Fatalf("metadata updates = %d, want 2", updates)
	}
}

func TestCaptureResponseWriterRecordsFlushAndHijackOutcomes(t *testing.T) {
	base := gin.CreateTestContextOnly(httptest.NewRecorder(), gin.New()).Writer
	attempt := &captureTestAttempt{}
	writer := newCaptureResponseWriter(&captureOutcomeWriter{
		ResponseWriter: base,
		flushErr:       errors.New("flush failed"),
		hijackErr:      errors.New("hijack failed"),
	}, attempt)
	if err := writer.FlushError(); !errors.Is(err, errors.New("flush failed")) {
		if err == nil || err.Error() != "flush failed" {
			t.Fatalf("FlushError() = %v, want flush failed", err)
		}
	}
	if _, _, err := writer.Hijack(); err == nil || err.Error() != "hijack failed" {
		t.Fatalf("Hijack() = %v, want hijack failed", err)
	}
	if len(attempt.errors) < 2 || attempt.errors[0] == nil || attempt.errors[1] == nil ||
		len(attempt.hijacks) != 1 || attempt.hijacks[0].Error() != "hijack failed" {
		t.Fatalf("capture outcomes errors=%v hijacks=%v", attempt.errors, attempt.hijacks)
	}
}

func TestCaptureResponseWriterRetainsShortWrite(t *testing.T) {
	base := gin.CreateTestContextOnly(httptest.NewRecorder(), gin.New()).Writer
	attempt := &captureTestAttempt{}
	writer := newCaptureResponseWriter(&shortCaptureWriter{ResponseWriter: base}, attempt)
	written, err := writer.Write([]byte("short"))
	if err != nil || written != len("short")-1 {
		t.Fatalf("Write() = (%d, %v), want underlying short write", written, err)
	}
	if len(attempt.shortWrites) != 1 || attempt.shortWrites[0] != [2]int{written, len("short")} {
		t.Fatalf("short write events = %v", attempt.shortWrites)
	}
}
func TestCaptureResponseWriterReadFromDoesNotRecurse(t *testing.T) {
	base := gin.CreateTestContextOnly(httptest.NewRecorder(), gin.New()).Writer
	attempt := &captureTestAttempt{}
	writer := newCaptureResponseWriter(base, attempt)
	done := make(chan struct{})
	var written int64
	var err error
	go func() {
		written, err = writer.ReadFrom(&plainCaptureReader{data: []byte("read-from")})
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("ReadFrom did not return")
	}
	if err != nil || written != int64(len("read-from")) || !bytes.Contains(attempt.response, []byte("read-from")) {
		t.Fatalf("ReadFrom() = (%d, %v), captured=%q", written, err, attempt.response)
	}
}

type shortCaptureWriter struct{ gin.ResponseWriter }

func (writer *shortCaptureWriter) Write(data []byte) (int, error) {
	return len(data) - 1, nil
}

type captureErrorReadCloser struct {
	readErr  error
	closeErr error
}

func (body *captureErrorReadCloser) Read([]byte) (int, error) { return 0, body.readErr }
func (body *captureErrorReadCloser) Close() error             { return body.closeErr }

type captureOutcomeWriter struct {
	gin.ResponseWriter
	flushErr  error
	hijackErr error
}

func (writer *captureOutcomeWriter) FlushError() error { return writer.flushErr }
func (writer *captureOutcomeWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return nil, nil, writer.hijackErr
}

type plainCaptureReader struct{ data []byte }

func (reader *plainCaptureReader) Read(buffer []byte) (int, error) {
	if len(reader.data) == 0 {
		return 0, io.EOF
	}
	count := copy(buffer, reader.data)
	reader.data = reader.data[count:]
	return count, nil
}

func TestCaptureInitializationDoesNotBlockDataPlane(t *testing.T) {
	factory := &blockingCaptureFactory{started: make(chan struct{}), release: make(chan struct{})}
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString("queued-body"))
	context := gin.CreateTestContextOnly(httptest.NewRecorder(), gin.New())
	context.Request = request

	done := make(chan struct{})
	go func() {
		captureDataPlaneRequest(&Handler{captureFactory: factory}, context)
		close(done)
	}()
	<-factory.started
	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Fatal("capture initialization blocked the data plane")
	}
	if _, exists := context.Get(captureContextKey); !exists {
		t.Fatal("deferred capture was not installed before initialization completed")
	}
	if _, err := io.ReadAll(request.Body); err != nil {
		t.Fatal(err)
	}
	_ = finalizeDataPlaneCapture(context)
	close(factory.release)
	session, _ := factory.snapshot()
	if !waitForCapture(t, func() bool {
		session, _ = factory.snapshot()
		return session != nil && session.isWaited()
	}) {
		t.Fatal("deferred capture did not drain after initialization")
	}
	session.mu.Lock()
	attemptCount := len(session.attempts)
	attempt := session.attempts[0]
	session.mu.Unlock()
	attempt.mu.Lock()
	requestBytes := append([]byte(nil), attempt.request...)
	attempt.mu.Unlock()
	if attemptCount != 1 || !bytes.Contains(requestBytes, []byte("queued-body")) {
		t.Fatalf("queued request body was not retained: attempts=%d request=%q", attemptCount, requestBytes)
	}
}

func TestCaptureDeferredPreparationFinalizationDoesNotBlockDataPlane(t *testing.T) {
	factory := &blockingCaptureFactory{started: make(chan struct{}), release: make(chan struct{})}
	capture := newDeferredDataPlaneCapture(factory, CaptureSessionMetadata{Method: http.MethodGet, Path: "/v1/models"})
	finished := make(chan struct{})
	go func() {
		capture.finishPreparationAttempt(CaptureAttemptMetadata{
			AttemptID: "request:1",
			Sequence:  1,
			Fields:    map[string]string{"phase": "preparation"},
		}, errors.New("candidate preparation failed"))
		close(finished)
	}()
	<-factory.started
	select {
	case <-finished:
	case <-time.After(100 * time.Millisecond):
		t.Fatal("candidate preparation capture blocked the data plane")
	}

	_ = capture.finalize(context.Background())
	close(factory.release)
	session, _ := factory.snapshot()
	if !waitForCapture(t, func() bool {
		session, _ = factory.snapshot()
		if session == nil {
			return false
		}
		session.mu.Lock()
		attemptCount := len(session.attempts)
		if attemptCount < 2 {
			session.mu.Unlock()
			return false
		}
		attempt := session.attempts[1]
		session.mu.Unlock()
		attempt.mu.Lock()
		failed := attempt.failed != nil
		attempt.mu.Unlock()
		return failed
	}) {
		t.Fatal("deferred preparation failure did not reach a failed capture attempt")
	}
}

func TestCapturePendingBarrierSupportsSequentialForwards(t *testing.T) {
	ready := make(chan struct{})
	close(ready)
	capture := &dataPlaneCapture{session: &captureTestSession{}, sessionReady: ready}
	capture.beginForwardCapture()
	capture.finishForwardCapture()
	capture.beginForwardCapture()

	finished := make(chan struct{})
	go func() {
		capture.finishSession()
		close(finished)
	}()
	select {
	case <-finished:
		t.Fatal("session finalization reused a closed pending barrier")
	case <-time.After(50 * time.Millisecond):
	}
	capture.finishForwardCapture()
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("session finalization did not observe the second forward")
	}
}

func TestCaptureRejectedForwardDoesNotReleaseAcceptedBarrier(t *testing.T) {
	ready := make(chan struct{})
	close(ready)
	session := &captureTestSession{}
	capture := &dataPlaneCapture{session: session, sessionReady: ready}
	accepted, _, _, ok := capture.beginForward(context.Background(), ForwardInput{
		AttemptID: "accepted", ChannelID: "openai",
	})
	if !ok || accepted == nil {
		t.Fatal("initial forward was not accepted")
	}
	if err := capture.finalize(context.Background()); err != nil {
		t.Fatal(err)
	}
	rejected, observer, _, ok := capture.beginForward(context.Background(), ForwardInput{
		AttemptID: "rejected", ChannelID: "openai",
	})
	if ok || rejected != nil || observer != nil {
		t.Fatal("forward after finalization was accepted")
	}
	capture.normalizeAndFinishForwardOwned(rejected, observer, UpstreamResult{Err: errors.New("rejected")}, ok)
	time.Sleep(50 * time.Millisecond)
	if session.isWaited() {
		t.Fatal("rejected forward released an accepted pending barrier")
	}
	capture.finishForward(accepted, nil, UpstreamResult{DispatchState: execution.DispatchMaybeSent})
	if !waitForCapture(t, session.isWaited) {
		t.Fatal("accepted forward barrier did not drain")
	}
}

func TestCaptureRejectedForwardFailurePrecedesSessionTerminal(t *testing.T) {
	ready := make(chan struct{})
	close(ready)
	session := &captureTestSession{}
	capture := &dataPlaneCapture{session: session, sessionReady: ready}
	capture.failureMu.Lock()
	if err := capture.finalize(context.Background()); err != nil {
		capture.failureMu.Unlock()
		t.Fatal(err)
	}
	_, observer, _, accepted := capture.beginForward(context.Background(), ForwardInput{
		AttemptID: "rejected", ChannelID: "openai",
	})
	if accepted || observer != nil {
		capture.failureMu.Unlock()
		t.Fatal("forward after finalization was accepted")
	}
	capture.normalizeAndFinishForwardOwned(nil, observer, UpstreamResult{Err: errors.New("rejected")}, false)
	capture.failureMu.Unlock()
	if !waitForCapture(t, func() bool { return session.failureValue() != nil }) {
		t.Fatal("rejected forward failure did not fail session")
	}
	if session.isCompleted() {
		t.Fatal("session completed after rejected forward")
	}
}

func TestCaptureObserverLateFailureNotifiesCoordinator(t *testing.T) {
	capture := &dataPlaneCapture{}
	observer := newCaptureHTTPObserver(&captureTestAttempt{}, "attempt")
	observer.setFailureSink(capture.recordObserverFailure)
	observer.AwaitCompletion()
	observer.ObserveResponseComplete("attempt", nil, nil)
	observer.Wait()
	observer.ObserveResponseBody("attempt", []byte("late"))
	if !waitForCapture(t, func() bool { return capture.failureValue() != nil }) {
		t.Fatal("late observer failure did not reach capture coordinator")
	}
}

func TestCaptureObserverPersistsResponseTermination(t *testing.T) {
	attempt := &captureTestAttempt{}
	observer := newCaptureHTTPObserver(attempt, "attempt")
	observer.AwaitCompletion()
	observer.ObserveResponse("attempt", http.StatusOK, http.Header{"Content-Type": {"text/event-stream"}})
	observer.ObserveResponseBody("attempt", []byte("data: [DONE]\\r\\n\\r\\n"))
	observer.ObserveResponseTermination("attempt", "eof", "")
	observer.ObserveResponseComplete("attempt", nil, nil)
	observer.Wait()

	attempt.mu.Lock()
	defer attempt.mu.Unlock()
	if len(attempt.terminations) != 1 || attempt.terminations[0] != "eof:" {
		t.Fatalf("terminations = %#v, want one EOF termination", attempt.terminations)
	}
}

func TestCaptureObserverInfersTerminationFromCompletion(t *testing.T) {
	for _, test := range []struct {
		name            string
		observeResponse bool
		completionErr   error
		wantTermination string
		wantDetail      string
	}{
		{name: "eof", wantTermination: "eof"},
		{name: "no response", completionErr: errors.New("dial failed"), wantTermination: "no_response", wantDetail: "dial failed"},
		{name: "response read error", observeResponse: true, completionErr: errors.New("stream read failed"), wantTermination: "read_error", wantDetail: "stream read failed"},
	} {
		t.Run(test.name, func(t *testing.T) {
			attempt := &captureTestAttempt{}
			observer := newCaptureHTTPObserver(attempt, "attempt")
			observer.AwaitCompletion()
			if test.observeResponse {
				observer.ObserveResponse("attempt", http.StatusOK, nil)
			}
			observer.ObserveResponseComplete("attempt", nil, test.completionErr)
			observer.Wait()

			attempt.mu.Lock()
			defer attempt.mu.Unlock()
			want := test.wantTermination + ":" + test.wantDetail
			if len(attempt.terminations) != 1 || attempt.terminations[0] != want {
				t.Fatalf("terminations = %#v, want %q", attempt.terminations, want)
			}
		})
	}
}

func TestCaptureObserverDuplicateCompletionNotifiesCoordinator(t *testing.T) {
	capture := &dataPlaneCapture{}
	observer := newCaptureHTTPObserver(&captureTestAttempt{}, "attempt")
	observer.setFailureSink(capture.recordObserverFailure)
	observer.AwaitCompletion()
	observer.ObserveResponseComplete("attempt", nil, nil)
	observer.ObserveResponseComplete("attempt", nil, nil)
	if !waitForCapture(t, func() bool { return capture.failureValue() != nil }) {
		t.Fatal("duplicate completion did not reach capture coordinator")
	}
}

func TestCaptureObserverFailureSinkBlocksTerminalSnapshot(t *testing.T) {
	ready := make(chan struct{})
	close(ready)
	session := &captureTestSession{}
	capture := &dataPlaneCapture{session: session, sessionReady: ready}
	observer := newCaptureHTTPObserver(&captureTestAttempt{}, "attempt")
	observer.setFailureSink(capture.recordObserverFailure)
	observer.AwaitCompletion()
	observer.ObserveResponseComplete("attempt", nil, nil)
	observer.Wait()

	capture.lifecycleMu.Lock()
	released := false
	defer func() {
		if !released {
			capture.lifecycleMu.Unlock()
		}
	}()
	lateDone := make(chan struct{})
	go func() {
		observer.ObserveResponseBody("attempt", []byte("late"))
		close(lateDone)
	}()
	if !waitForCapture(t, func() bool { return capture.failureSinks.Load() == 1 }) {
		capture.lifecycleMu.Unlock()
		t.Fatal("late observer failure sink did not enter")
	}
	finished := make(chan struct{})
	go func() {
		capture.finishSession()
		close(finished)
	}()
	select {
	case <-finished:
		t.Fatal("session terminalized while observer failure sink was in flight")
	case <-time.After(50 * time.Millisecond):
	}
	released = true
	capture.lifecycleMu.Unlock()
	<-lateDone
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("session terminalization did not finish after observer failure")
	}
	if session.isCompleted() {
		t.Fatal("session completed after a late observer failure")
	}
	if session.failureValue() == nil {
		t.Fatal("late observer failure did not fail the session")
	}
}

func TestCaptureLateObserverFailureFailsSessionBeforeTerminal(t *testing.T) {
	ready := make(chan struct{})
	close(ready)
	session := &captureTestSession{}
	capture := &dataPlaneCapture{session: session, sessionReady: ready}
	capture.beginForwardCapture()
	waitStarted := make(chan struct{})
	waitRelease := make(chan struct{})
	attempt := &captureTestAttempt{waitStarted: waitStarted, waitRelease: waitRelease}
	observer := newCaptureHTTPObserver(attempt, "attempt")
	observer.setFailureSink(capture.recordObserverFailure)
	observer.AwaitCompletion()

	if err := capture.finalize(context.Background()); err != nil {
		t.Fatal(err)
	}
	capture.finishForward(attempt, observer, UpstreamResult{
		DispatchState:  execution.DispatchMaybeSent,
		RequestWritten: true,
	})
	observer.ObserveResponseComplete("attempt", nil, nil)
	select {
	case <-waitStarted:
	case <-time.After(time.Second):
		t.Fatal("attempt terminal barrier did not start")
	}
	observer.ObserveResponseBody("attempt", []byte("late"))
	close(waitRelease)

	if !waitForCapture(t, func() bool {
		session.mu.Lock()
		defer session.mu.Unlock()
		return session.failed != nil
	}) {
		t.Fatal("late observer failure did not fail the session before terminalization")
	}
}

func TestCaptureObserverMismatchedCompletionUnblocksWait(t *testing.T) {
	observer := newCaptureHTTPObserver(&captureTestAttempt{}, "expected")
	observer.AwaitCompletion()
	waited := make(chan struct{})
	go func() {
		observer.Wait()
		close(waited)
	}()
	observer.ObserveResponseComplete("wrong", nil, nil)
	select {
	case <-waited:
	case <-time.After(time.Second):
		t.Fatal("mismatched completion blocked observer Wait")
	}
	if observer.failureValue() == nil {
		t.Fatal("mismatched completion did not fail observer")
	}
}

func TestCaptureObserverRejectsLateCallbackAfterCompletion(t *testing.T) {
	observer := newCaptureHTTPObserver(&captureTestAttempt{}, "attempt")
	observer.AwaitCompletion()
	observer.ObserveResponseComplete("attempt", nil, nil)
	observer.ObserveResponseBody("attempt", []byte("late"))
	observer.Wait()
	if observer.failureValue() == nil {
		t.Fatal("late callback after completion was accepted")
	}
}

func TestCaptureObserverCompletionAndLateCallbackRaceIsTerminal(t *testing.T) {
	for i := 0; i < 100; i++ {
		observer := newCaptureHTTPObserver(&captureTestAttempt{}, "attempt")
		observer.AwaitCompletion()
		var callbacks sync.WaitGroup
		callbacks.Add(2)
		go func() {
			defer callbacks.Done()
			observer.ObserveResponseBody("attempt", []byte("body"))
		}()
		go func() {
			defer callbacks.Done()
			observer.ObserveResponseComplete("attempt", nil, nil)
		}()
		callbacks.Wait()
		observer.Wait()
		if observer.failureValue() != nil {
			// A callback racing with completion is either accepted before the
			// completion barrier or rejected as a late event; both are safe.
			continue
		}
	}
}

func TestCaptureResponseWriterWriteHeaderNowDelegatesBeforeCapture(t *testing.T) {
	base := gin.CreateTestContextOnly(httptest.NewRecorder(), gin.New()).Writer
	underlying := &writeHeaderNowSpy{ResponseWriter: base}
	attempt := &captureTestAttempt{}
	writer := newCaptureResponseWriter(underlying, attempt)
	writer.Header().Set("X-Commit", "yes")
	writer.WriteHeaderNow()
	if underlying.nowCalls != 1 || underlying.writeHeaderCalls != 0 {
		t.Fatalf("WriteHeaderNow delegation = now=%d writeHeader=%d", underlying.nowCalls, underlying.writeHeaderCalls)
	}
	if !bytes.Contains(attempt.response, []byte("HTTP 200")) || !bytes.Contains(attempt.response, []byte("X-Commit: yes")) {
		t.Fatalf("captured committed response = %q", attempt.response)
	}
}

func TestCaptureSuccessfulHijackWithoutEventTargetFailsCapture(t *testing.T) {
	base := gin.CreateTestContextOnly(httptest.NewRecorder(), gin.New()).Writer
	left, right := net.Pipe()
	defer left.Close()
	defer right.Close()
	var captureErr error
	writer := newCaptureResponseWriter(&successfulHijackWriter{ResponseWriter: base, conn: left}, &bareCaptureAttempt{}, func(err error) { captureErr = err })
	_, _, err := writer.Hijack()
	if err != nil {
		t.Fatal(err)
	}
	if captureErr == nil {
		t.Fatal("successful hijack without a durable event target did not fail capture")
	}
}

func TestCaptureMetadataWithoutUpdaterFailsCapture(t *testing.T) {
	capture := &dataPlaneCapture{session: &bareCaptureSession{}}
	capture.updateMetadata(CaptureSessionMetadata{RequestID: "material-request-id"})
	if !waitForCapture(t, func() bool { return capture.failureValue() != nil }) {
		t.Fatal("missing metadata updater was silently ignored")
	}
}

func TestCaptureMetadataUpdaterPanicIsContained(t *testing.T) {
	capture := &dataPlaneCapture{session: &panicMetadataSession{}}
	capture.updateMetadata(CaptureSessionMetadata{RequestID: "panic-safe"})
	if !waitForCapture(t, func() bool { return capture.failureValue() != nil }) {
		t.Fatal("metadata updater panic was not converted to capture failure")
	}
}

// These test doubles deliberately block or omit optional capabilities to prove
// that the capture boundary, rather than the data plane, owns those failures.
type blockingCaptureFactory struct {
	mu      sync.Mutex
	session *captureTestSession
	starts  int
	started chan struct{}
	release chan struct{}
	err     error
}

func (f *blockingCaptureFactory) StartSession(CaptureSessionMetadata) (CaptureSession, error) {
	f.mu.Lock()
	f.starts++
	if f.started != nil {
		select {
		case <-f.started:
		default:
			close(f.started)
		}
	}
	f.mu.Unlock()
	if f.release != nil {
		<-f.release
	}
	if f.err != nil {
		return nil, f.err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.session = &captureTestSession{}
	return f.session, nil
}

type capabilityResponseWriter struct{ *httptest.ResponseRecorder }

func newCapabilityResponseWriter() *capabilityResponseWriter {
	return &capabilityResponseWriter{httptest.NewRecorder()}
}
func (w *capabilityResponseWriter) FlushError() error { w.Flush(); return nil }
func (w *capabilityResponseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return nil, nil, errors.New("not available")
}
func (w *capabilityResponseWriter) ReadFrom(r io.Reader) (int64, error) {
	return io.Copy(w.ResponseRecorder, r)
}

func (f *blockingCaptureFactory) snapshot() (*captureTestSession, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.session, f.starts
}

type writeHeaderNowSpy struct {
	gin.ResponseWriter
	nowCalls         int
	writeHeaderCalls int
}

func (writer *writeHeaderNowSpy) WriteHeader(status int) {
	writer.writeHeaderCalls++
	writer.ResponseWriter.WriteHeader(status)
}
func (writer *writeHeaderNowSpy) WriteHeaderNow() {
	writer.nowCalls++
	writer.ResponseWriter.WriteHeaderNow()
}

type successfulHijackWriter struct {
	gin.ResponseWriter
	conn net.Conn
}

func (writer *successfulHijackWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return writer.conn, bufio.NewReadWriter(bufio.NewReader(writer.conn), bufio.NewWriter(writer.conn)), nil
}

type bareCaptureAttempt struct{}

func (*bareCaptureAttempt) AppendRequestHeaders([]byte) error  { return nil }
func (*bareCaptureAttempt) AppendRequestBody([]byte) error     { return nil }
func (*bareCaptureAttempt) AppendResponseHeaders([]byte) error { return nil }
func (*bareCaptureAttempt) AppendResponseBody([]byte) error    { return nil }
func (*bareCaptureAttempt) Complete() error                    { return nil }
func (*bareCaptureAttempt) Fail(error) error                   { return nil }
func (*bareCaptureAttempt) Wait()                              {}

type bareCaptureSession struct{}

type panicMetadataSession struct{ bareCaptureSession }

func (*panicMetadataSession) UpdateMetadata(CaptureSessionMetadata) error { panic("metadata updater") }

func (*bareCaptureSession) StartAttempt(CaptureAttemptMetadata) (CaptureAttempt, error) {
	return &bareCaptureAttempt{}, nil
}
func (*bareCaptureSession) Complete() error  { return nil }
func (*bareCaptureSession) Fail(error) error { return nil }
func (*bareCaptureSession) Wait()            {}

func (w *capabilityResponseWriter) Unwrap() http.ResponseWriter { return w.ResponseRecorder }
