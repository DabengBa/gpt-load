package gateway

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/execution"
)

const captureContextKey = "gpt-load.gateway.capture"

type CaptureSessionMetadata struct {
	RequestID   string
	AccessKeyID uint
	Protocol    string
	Operation   string
	Method      string
	Path        string
	Headers     http.Header
	Fields      map[string]string
}

type CaptureAttemptMetadata struct {
	AttemptID string
	Sequence  uint32
	Fields    map[string]string
}

// CaptureFactory is the injection boundary for the capture store. A nil
// factory keeps the data plane byte-for-byte unchanged and disabled.
type CaptureFactory interface {
	StartSession(CaptureSessionMetadata) (CaptureSession, error)
}

// CaptureSessionMetadataUpdater is an optional adapter boundary for metadata
// that is only known after authentication or request inspection. Implementations
// must treat non-zero fields as a patch and preserve existing raw capture data.
type CaptureSessionMetadataUpdater interface {
	UpdateMetadata(CaptureSessionMetadata) error
}

type CaptureSession interface {
	StartAttempt(CaptureAttemptMetadata) (CaptureAttempt, error)
	Complete() error
	Fail(error) error
	Wait()
}

type CaptureAttempt interface {
	AppendRequestHeaders([]byte) error
	AppendRequestBody([]byte) error
	AppendResponseHeaders([]byte) error
	AppendResponseBody([]byte) error
	Complete() error
	Fail(error) error
	Wait()
}

// Optional event methods let an adapter persist response events without
// forcing them into raw body bytes. U006's current store can ignore these.
type captureResponseEvents interface {
	RecordResponseFlush() error
	RecordResponseShortWrite(int, int) error
	RecordResponseError(error) error
}

type captureResponseTerminationEvents interface {
	RecordResponseTermination(string, string) error
}

// captureTerminationNoSource marks an attempt where no provider completion
// callback arrived, so its capture cannot be treated as a complete exchange.
const captureTerminationNoSource = "no_observation_source"

// The persisted detail distinguishes the two reasons a completion is missing.
// Neither one asserts anything about bytes the provider may have sent: only the
// absence of the callback that would prove what was exchanged.
const (
	captureNoSourceNeverStarted    = "provider observation source never started; no completion callback was possible"
	captureNoSourceEndedIncomplete = "provider observation source started and ended without a completion callback"
)

type captureResponseHijackEvents interface {
	RecordResponseHijack(error) error
}

type captureRequestEvents interface {
	RecordRequestError(error) error
	RecordRequestClose(error) error
	RecordRequestOutcome(string, error) error
}

type dataPlaneCapture struct {
	session          CaptureSession
	clientAttempt    CaptureAttempt
	factory          CaptureFactory
	initial          CaptureSessionMetadata
	requestContext   context.Context
	sessionReady     chan struct{}
	sessionMu        sync.Mutex
	initErr          error
	operationMu      sync.Mutex
	lifecycleMu      sync.Mutex
	finalizing       bool
	attemptOrderMu   sync.Mutex
	attemptOrderCond *sync.Cond
	nextAttemptOrder uint64
	runAttemptOrder  uint64
	metadataWG       sync.WaitGroup
	mu               sync.Mutex
	failure          error
	failureMu        sync.Mutex
	failureCond      *sync.Cond
	failureSinks     atomic.Int32
	terminalStarted  atomic.Bool
	pending          int
	pendingDone      chan struct{}
	finalizeOnce     sync.Once
}

func newDeferredDataPlaneCapture(
	factory CaptureFactory,
	metadata CaptureSessionMetadata,
	requestContext context.Context,
) *dataPlaneCapture {
	capture := &dataPlaneCapture{
		factory:        factory,
		initial:        metadata,
		requestContext: requestContext,
		sessionReady:   make(chan struct{}),
	}
	capture.clientAttempt = capture.startAttempt(CaptureAttemptMetadata{
		AttemptID: "client",
		Fields:    map[string]string{"kind": "client"},
	})
	go capture.initializeSession()
	return capture
}

func (capture *dataPlaneCapture) initializeSession() {
	session, err := safeStartSession(capture.factory, capture.initial)
	if err == nil && session == nil {
		err = errors.New("start capture session: nil session")
	}
	capture.sessionMu.Lock()
	capture.session = session
	capture.initErr = err
	close(capture.sessionReady)
	capture.sessionMu.Unlock()
	if err != nil {
		capture.recordFailure(fmt.Errorf("start capture session: %w", err))
	}
}

func (capture *dataPlaneCapture) ensureSessionReady() {
	if capture == nil {
		return
	}
	capture.sessionMu.Lock()
	if capture.sessionReady == nil {
		capture.sessionReady = make(chan struct{})
		if capture.session != nil {
			close(capture.sessionReady)
		}
	}
	ready := capture.sessionReady
	capture.sessionMu.Unlock()
	if ready != nil {
		<-ready
	}
}

func (capture *dataPlaneCapture) sessionValue() (CaptureSession, error) {
	if capture == nil {
		return nil, errors.New("capture is unavailable")
	}
	capture.ensureSessionReady()
	capture.sessionMu.Lock()
	defer capture.sessionMu.Unlock()
	return capture.session, capture.initErr
}

func (capture *dataPlaneCapture) captureEnabled() bool {
	if capture == nil {
		return false
	}
	if capture.factory != nil {
		return true
	}
	capture.sessionMu.Lock()
	defer capture.sessionMu.Unlock()
	return capture.session != nil || capture.sessionReady != nil
}

func (capture *dataPlaneCapture) beginPendingCapture() bool {
	if capture == nil {
		return false
	}
	capture.lifecycleMu.Lock()
	if capture.finalizing {
		capture.recordFailureLocked(errors.New("capture event arrived after finalization"))
		capture.lifecycleMu.Unlock()
		return false
	}
	capture.mu.Lock()
	if capture.pendingDone == nil {
		capture.pendingDone = make(chan struct{})
	}
	capture.pending++
	capture.mu.Unlock()
	capture.lifecycleMu.Unlock()
	return true
}

func (capture *dataPlaneCapture) beginForwardCapture() bool {
	return capture.beginPendingCapture()
}

func (capture *dataPlaneCapture) finishForwardCapture() {
	if capture == nil {
		return
	}
	capture.mu.Lock()
	if capture.pending > 0 {
		capture.pending--
		if capture.pending == 0 && capture.pendingDone != nil {
			pendingDone := capture.pendingDone
			capture.pendingDone = nil
			close(pendingDone)
		}
	}
	capture.mu.Unlock()
}

func (capture *dataPlaneCapture) recordFailure(err error) {
	if capture == nil || err == nil {
		return
	}
	capture.lifecycleMu.Lock()
	capture.recordFailureLocked(err)
	capture.lifecycleMu.Unlock()
}

func (capture *dataPlaneCapture) recordFailureLocked(err error) {
	if capture == nil || err == nil {
		return
	}
	capture.mu.Lock()
	defer capture.mu.Unlock()
	if capture.failure == nil {
		capture.failure = err
	}
}

func (capture *dataPlaneCapture) recordObserverFailure(err error) {
	if capture == nil || err == nil || !capture.beginObserverFailure() {
		return
	}
	defer capture.endObserverFailure()
	capture.recordFailure(err)
}

func (capture *dataPlaneCapture) beginObserverFailure() bool {
	if capture.terminalStarted.Load() {
		return false
	}
	capture.failureSinks.Add(1)
	if capture.terminalStarted.Load() {
		capture.endObserverFailure()
		return false
	}
	return true
}

func (capture *dataPlaneCapture) endObserverFailure() {
	capture.failureSinks.Add(-1)
	capture.failureMu.Lock()
	if capture.failureCond != nil {
		capture.failureCond.Broadcast()
	}
	capture.failureMu.Unlock()
}

func (capture *dataPlaneCapture) failureValue() error {
	if capture == nil {
		return nil
	}
	capture.mu.Lock()
	defer capture.mu.Unlock()
	return capture.failure
}

func (capture *dataPlaneCapture) recordRequestOutcome(kind string, err error) {
	if capture == nil || capture.clientAttempt == nil {
		return
	}
	if events, ok := capture.clientAttempt.(captureRequestEvents); ok {
		if eventErr := events.RecordRequestOutcome(kind, err); eventErr != nil {
			capture.recordFailure(fmt.Errorf("capture request outcome: %w", eventErr))
		}
	}
}

func (capture *dataPlaneCapture) recordRequestError(err error) {
	if capture == nil || err == nil {
		return
	}
	capture.recordFailure(err)
	if events, ok := capture.clientAttempt.(captureRequestEvents); ok {
		if eventErr := events.RecordRequestError(err); eventErr != nil {
			capture.recordFailure(fmt.Errorf("capture request error: %w", eventErr))
		}
	}
}

func (capture *dataPlaneCapture) recordRequestClose(err error) {
	if capture == nil || err == nil {
		return
	}
	capture.recordFailure(err)
	if events, ok := capture.clientAttempt.(captureRequestEvents); ok {
		if eventErr := events.RecordRequestClose(err); eventErr != nil {
			capture.recordFailure(fmt.Errorf("capture request close: %w", eventErr))
		}
	}
}

func (capture *dataPlaneCapture) updateMetadata(metadata CaptureSessionMetadata) {
	if capture == nil || metadataIsEmpty(metadata) || !capture.captureEnabled() {
		return
	}
	capture.lifecycleMu.Lock()
	if capture.finalizing {
		capture.recordFailureLocked(errors.New("capture metadata update arrived after finalization"))
		capture.lifecycleMu.Unlock()
		return
	}
	capture.metadataWG.Add(1)
	capture.lifecycleMu.Unlock()
	go func() {
		defer capture.metadataWG.Done()
		session, err := capture.sessionValue()
		if err != nil {
			capture.recordFailure(fmt.Errorf("update capture session metadata: %w", err))
			return
		}
		capture.operationMu.Lock()
		defer capture.operationMu.Unlock()
		updater, ok := session.(CaptureSessionMetadataUpdater)
		if !ok {
			capture.recordFailure(errors.New("capture session cannot persist metadata updates"))
			return
		}
		if err := safeUpdateMetadata(updater, metadata); err != nil {
			capture.recordFailure(fmt.Errorf("update capture session metadata: %w", err))
		}
	}()
}

func metadataIsEmpty(metadata CaptureSessionMetadata) bool {
	return metadata.RequestID == "" && metadata.AccessKeyID == 0 && metadata.Protocol == "" &&
		metadata.Operation == "" && metadata.Method == "" && metadata.Path == "" &&
		metadata.Headers == nil && len(metadata.Fields) == 0
}

func safeStartAttempt(session CaptureSession, metadata CaptureAttemptMetadata) (attempt CaptureAttempt, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("capture attempt start panic: %v", recovered)
		}
	}()
	return session.StartAttempt(metadata)
}
func (capture *dataPlaneCapture) startAttempt(metadata CaptureAttemptMetadata) CaptureAttempt {
	if capture == nil || !capture.captureEnabled() {
		return nil
	}
	deferred := newDeferredCaptureAttempt()
	capture.attemptOrderMu.Lock()
	if capture.attemptOrderCond == nil {
		capture.attemptOrderCond = sync.NewCond(&capture.attemptOrderMu)
	}
	order := capture.nextAttemptOrder
	capture.nextAttemptOrder++
	capture.attemptOrderMu.Unlock()
	go func() {
		capture.attemptOrderMu.Lock()
		for order != capture.runAttemptOrder {
			capture.attemptOrderCond.Wait()
		}
		capture.attemptOrderMu.Unlock()
		defer func() {
			capture.attemptOrderMu.Lock()
			capture.runAttemptOrder++
			capture.attemptOrderCond.Broadcast()
			capture.attemptOrderMu.Unlock()
		}()
		session, err := capture.sessionValue()
		if err == nil && session != nil {
			capture.operationMu.Lock()
			attempt, startErr := safeStartAttempt(session, metadata)
			capture.operationMu.Unlock()
			if startErr != nil {
				err = fmt.Errorf("start capture attempt: %w", startErr)
			} else if attempt == nil {
				err = errors.New("start capture attempt: nil attempt")
			} else {
				deferred.setTarget(attempt, nil)
				return
			}
		}
		if err == nil {
			err = errors.New("capture session is unavailable")
		}
		capture.recordFailure(err)
		deferred.setTarget(nil, err)
	}()
	return deferred
}

func (capture *dataPlaneCapture) finishPreparationAttempt(metadata CaptureAttemptMetadata, err error) {
	if capture == nil || !capture.captureEnabled() || !capture.beginPendingCapture() {
		return
	}
	attempt := capture.startAttempt(metadata)
	go func() {
		defer capture.finishForwardCapture()
		capture.finishAttempt(attempt, err)
	}()
}

func runCaptureEvent(event func(CaptureAttempt) error, target CaptureAttempt) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("capture callback panic: %v", recovered)
		}
	}()
	return event(target)
}

func (capture *dataPlaneCapture) finishAttempt(attempt CaptureAttempt, err error) {
	if attempt == nil {
		return
	}
	if waitErr := safeAttemptWait(attempt); waitErr != nil {
		capture.recordFailure(waitErr)
	}
	if failed, ok := attempt.(interface{ captureFailure() error }); ok {
		capture.recordFailure(failed.captureFailure())
	}
	if err == nil {
		err = capture.failureValue()
	}
	if err != nil {
		if failErr := safeAttemptFail(attempt, err); failErr != nil {
			capture.recordFailure(fmt.Errorf("fail capture attempt: %w", failErr))
		}
		return
	}
	if completeErr := safeAttemptComplete(attempt); completeErr != nil {
		capture.recordFailure(fmt.Errorf("complete capture attempt: %w", completeErr))
		if failErr := safeAttemptFail(attempt, completeErr); failErr != nil {
			capture.recordFailure(fmt.Errorf("fail capture attempt: %w", failErr))
		}
	}
}

func safeAttemptWait(attempt CaptureAttempt) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("capture attempt wait panic: %v", recovered)
		}
	}()
	attempt.Wait()
	return nil
}

func safeAttemptComplete(attempt CaptureAttempt) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("capture attempt completion panic: %v", recovered)
		}
	}()
	return attempt.Complete()
}

func safeAttemptFail(attempt CaptureAttempt, cause error) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("capture attempt failure panic: %v", recovered)
		}
	}()
	return attempt.Fail(cause)
}

func (capture *dataPlaneCapture) finalize(requestContext context.Context) error {
	if capture == nil || !capture.captureEnabled() {
		return nil
	}
	capture.finalizeOnce.Do(func() {
		capture.lifecycleMu.Lock()
		capture.finalizing = true
		capture.lifecycleMu.Unlock()
		if requestContext != nil && requestContext.Err() != nil {
			err := requestContext.Err()
			capture.recordRequestOutcome("context_canceled", err)
			capture.recordFailure(fmt.Errorf("capture request context: %w", err))
		}
		go capture.finishSession()
	})
	return nil
}

func (capture *dataPlaneCapture) finishSession() {
	session, initErr := capture.sessionValue()
	if initErr != nil {
		capture.recordFailure(fmt.Errorf("capture session initialization: %w", initErr))
	}
	for {
		capture.mu.Lock()
		pending := capture.pending
		pendingDone := capture.pendingDone
		capture.mu.Unlock()
		if pending == 0 {
			break
		}
		if pendingDone == nil {
			break
		}
		<-pendingDone
	}
	capture.metadataWG.Wait()
	capture.finishAttempt(capture.clientAttempt, nil)
	if session == nil {
		return
	}
	if waitErr := safeSessionWait(session); waitErr != nil {
		capture.recordFailure(waitErr)
	}
	capture.terminalStarted.Store(true)
	capture.failureMu.Lock()
	if capture.failureCond == nil {
		capture.failureCond = sync.NewCond(&capture.failureMu)
	}
	for capture.failureSinks.Load() != 0 {
		capture.failureCond.Wait()
	}
	capture.lifecycleMu.Lock()
	failure := capture.failureValue()
	if failure != nil {
		capture.lifecycleMu.Unlock()
		capture.failureMu.Unlock()
		if failErr := safeSessionFail(session, failure); failErr != nil {
			capture.recordFailure(fmt.Errorf("fail capture session: %w", failErr))
		}
		return
	}
	// Serialize the final failure snapshot with observer failure sinks. Once
	// session completion starts, callback admission has already been closed.
	completeErr := safeSessionComplete(session)
	capture.lifecycleMu.Unlock()
	capture.failureMu.Unlock()
	if completeErr != nil {
		capture.recordFailure(fmt.Errorf("complete capture session: %w", completeErr))
		_ = safeSessionFail(session, completeErr)
	}
}

func safeSessionWait(session CaptureSession) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("capture session wait panic: %v", recovered)
		}
	}()
	session.Wait()
	return nil
}
func safeSessionComplete(session CaptureSession) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("capture session completion panic: %v", recovered)
		}
	}()
	return session.Complete()
}

func safeSessionFail(session CaptureSession, cause error) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("capture session failure panic: %v", recovered)
		}
	}()
	return session.Fail(cause)
}

func safeUpdateMetadata(updater CaptureSessionMetadataUpdater, metadata CaptureSessionMetadata) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("capture metadata update panic: %v", recovered)
		}
	}()
	return updater.UpdateMetadata(metadata)
}

func safeStartSession(factory CaptureFactory, metadata CaptureSessionMetadata) (session CaptureSession, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("capture session start panic: %v", recovered)
		}
	}()
	return factory.StartSession(metadata)
}
func (handler *Handler) SetCaptureFactory(factory CaptureFactory) {
	if handler != nil {
		handler.captureFactory = factory
	}
}

func (handler *Handler) CaptureMiddleware() gin.HandlerFunc {
	return func(ginContext *gin.Context) {
		captureDataPlaneRequest(handler, ginContext)
		defer func() { _ = finalizeDataPlaneCapture(ginContext) }()
		if ginContext != nil {
			ginContext.Next()
		}
	}
}

func captureDataPlaneRequest(handler *Handler, ginContext *gin.Context) {
	if handler == nil || handler.captureFactory == nil || ginContext == nil || ginContext.Request == nil ||
		ginContext.Request.URL == nil || !isDataPlaneNamespacePath(ginContext.Request.URL.Path) {
		return
	}
	if _, exists := ginContext.Get(captureContextKey); exists {
		return
	}
	request := ginContext.Request
	metadata := CaptureSessionMetadata{Method: request.Method, Headers: request.Header.Clone()}
	if request.URL != nil {
		metadata.Path = request.URL.Path
	}
	capture := newDeferredDataPlaneCapture(handler.captureFactory, metadata, request.Context())
	// Install the coordinator before any asynchronous storage work so route
	// terminal hooks cannot create a second capture after an init failure.
	ginContext.Set(captureContextKey, capture)
	if capture.clientAttempt != nil {
		if appendErr := capture.clientAttempt.AppendRequestHeaders(serializeHeaders(request.Header)); appendErr != nil {
			capture.recordFailure(fmt.Errorf("capture request headers: %w", appendErr))
		}
	}
	if request.Body != nil {
		request.Body = &captureRequestBody{
			ReadCloser: request.Body,
			context:    request.Context(),
			observe: func(data []byte) {
				if capture.clientAttempt != nil {
					if err := capture.clientAttempt.AppendRequestBody(data); err != nil {
						capture.recordFailure(fmt.Errorf("capture request body: %w", err))
					}
				}
			},
			recordReadError:  capture.recordRequestError,
			recordCloseError: capture.recordRequestClose,
			recordFailure:    capture.recordFailure,
			recordOutcome:    capture.recordRequestOutcome,
		}
	}
	ginContext.Writer = newCaptureResponseWriter(
		ginContext.Writer,
		capture.clientAttempt,
		capture.recordFailure,
	)
}

func finalizeDataPlaneCapture(ginContext *gin.Context) error {
	if ginContext == nil {
		return nil
	}
	value, exists := ginContext.Get(captureContextKey)
	if !exists {
		return nil
	}
	capture, ok := value.(*dataPlaneCapture)
	if !ok || capture == nil {
		return nil
	}
	return capture.finalize(capture.requestContext)
}

func captureFromContext(ginContext *gin.Context) *dataPlaneCapture {
	if ginContext == nil {
		return nil
	}
	value, _ := ginContext.Get(captureContextKey)
	capture, _ := value.(*dataPlaneCapture)
	return capture
}

func (capture *dataPlaneCapture) beginForward(ctx context.Context, input ForwardInput) (CaptureAttempt, *captureHTTPObserver, context.Context, bool) {
	if ctx == nil {
		ctx = context.Background()
	}
	attemptContext := execution.WithHTTPAttemptID(ctx, input.AttemptID)
	if capture == nil || !capture.captureEnabled() {
		return nil, nil, attemptContext, false
	}
	if !capture.beginForwardCapture() {
		return nil, nil, attemptContext, false
	}
	attempt := capture.startAttempt(CaptureAttemptMetadata{
		AttemptID: input.AttemptID,
		Sequence:  input.AttemptSequence,
		Fields:    map[string]string{"kind": "forward"},
	})
	if attempt == nil {
		return nil, nil, attemptContext, true
	}
	observer := newCaptureHTTPObserver(attempt, input.AttemptID)
	observer.setFailureSink(capture.recordObserverFailure)
	if !captureObserverSourceAvailable(input) {
		return attempt, nil, attemptContext, true
	}
	return attempt, observer, execution.WithHTTPObserver(attemptContext, observer), true
}

func captureObserverSourceAvailable(input ForwardInput) bool {
	// Every registered channel that reaches beginForward uses an HTTP provider
	// adapter. WebSocket attempts use a separate path and never reach here.
	return input.ChannelID != ""
}
func (capture *dataPlaneCapture) normalizeAndFinishForward(
	attempt CaptureAttempt,
	observer *captureHTTPObserver,
	result UpstreamResult,
) UpstreamResult {
	return capture.normalizeAndFinishForwardOwned(attempt, observer, result, true)
}

func (capture *dataPlaneCapture) normalizeAndFinishForwardOwned(
	attempt CaptureAttempt,
	observer *captureHTTPObserver,
	result UpstreamResult,
	owned bool,
) UpstreamResult {
	result = normalizeUpstreamResultContract(result)
	if owned {
		capture.finishForward(attempt, observer, result)
	}
	return result
}

func (capture *dataPlaneCapture) finishForward(attempt CaptureAttempt, observer *captureHTTPObserver, result UpstreamResult) {
	if attempt == nil {
		capture.finishForwardCapture()
		return
	}
	if observer != nil {
		if captureShouldAwaitObserver(result) {
			observer.AwaitCompletion()
		} else {
			observer.NoCompletionSource()
		}
	}
	go func() {
		defer capture.finishForwardCapture()
		if observer != nil {
			observer.Wait()
			if observer.failureValue() != nil {
				// Capture failures affect only capture terminal state, never the
				// response returned by the data plane.
				capture.recordFailure(observer.failureValue())
			}
		}
		capture.finishAttempt(attempt, result.Err)
	}()
}

func captureShouldAwaitObserver(result UpstreamResult) bool {
	if result.DispatchState == execution.DispatchLocal || result.DispatchState == execution.DispatchNotSent {
		return false
	}
	if isConversionUnsupportedResult(result) {
		return false
	}
	return result.Err != nil || result.DispatchState == execution.DispatchMaybeSent ||
		result.RequestWritten || result.ResponseStarted || result.Committed || result.StatusCode != 0
}

func serializeHeaders(headers http.Header) []byte {
	if len(headers) == 0 {
		return nil
	}
	keys := make([]string, 0, len(headers))
	for key := range headers {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var output bytes.Buffer
	for _, key := range keys {
		for _, value := range headers[key] {
			output.WriteString(key)
			output.WriteString(": ")
			output.WriteString(value)
			output.WriteString("\r\n")
		}
	}
	return output.Bytes()
}
