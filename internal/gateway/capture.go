package gateway

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sort"
	"strconv"
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

func newDeferredDataPlaneCapture(factory CaptureFactory, metadata CaptureSessionMetadata) *dataPlaneCapture {
	capture := &dataPlaneCapture{
		factory:      factory,
		initial:      metadata,
		sessionReady: make(chan struct{}),
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

const (
	maxCaptureQueueEvents = 128
	maxCaptureBodyBytes   = 8 << 20
)

var (
	errCaptureQueueFull = errors.New("capture queue is full")
	errCaptureBodyLimit = errors.New("capture body byte limit exceeded")
)

// deferredCaptureAttempt owns caller bytes and drains them only after the
// storage attempt has been created. It keeps all storage work off the request.
type deferredCaptureAttempt struct {
	targetReady chan struct{}
	done        chan struct{}
	wake        chan struct{}
	mu          sync.Mutex
	target      CaptureAttempt
	queue       []func(CaptureAttempt) error
	bodyBytes   int64
	open        bool
	failure     error
	waitOnce    sync.Once
}

func newDeferredCaptureAttempt() *deferredCaptureAttempt {
	attempt := &deferredCaptureAttempt{
		targetReady: make(chan struct{}),
		done:        make(chan struct{}),
		wake:        make(chan struct{}, 1),
		open:        true,
	}
	go attempt.run()
	return attempt
}

func (attempt *deferredCaptureAttempt) setTarget(target CaptureAttempt, err error) {
	attempt.mu.Lock()
	if err != nil && attempt.failure == nil {
		attempt.failure = err
	}
	attempt.target = target
	close(attempt.targetReady)
	select {
	case attempt.wake <- struct{}{}:
	default:
	}
	attempt.mu.Unlock()
}

func (attempt *deferredCaptureAttempt) run() {
	<-attempt.targetReady
	for {
		attempt.mu.Lock()
		if len(attempt.queue) == 0 {
			if !attempt.open {
				close(attempt.done)
				attempt.mu.Unlock()
				return
			}
			attempt.mu.Unlock()
			<-attempt.wake
			continue
		}
		event := attempt.queue[0]
		attempt.queue[0] = nil
		attempt.queue = attempt.queue[1:]
		target := attempt.target
		attempt.mu.Unlock()
		if target == nil {
			continue
		}
		if err := runCaptureEvent(event, target); err != nil {
			attempt.mu.Lock()
			if attempt.failure == nil {
				attempt.failure = err
			}
			attempt.mu.Unlock()
		}
	}
}

func (attempt *deferredCaptureAttempt) enqueue(event func(CaptureAttempt) error) error {
	if attempt == nil || event == nil {
		return errors.New("capture attempt is unavailable")
	}
	attempt.mu.Lock()
	defer attempt.mu.Unlock()
	if !attempt.open {
		return errors.New("capture attempt is closed")
	}
	if len(attempt.queue) >= maxCaptureQueueEvents {
		return errCaptureQueueFull
	}
	attempt.queue = append(attempt.queue, event)
	select {
	case attempt.wake <- struct{}{}:
	default:
	}
	return nil
}

func (attempt *deferredCaptureAttempt) enqueueBytes(data []byte, countBody bool, event func(CaptureAttempt, []byte) error) error {
	if attempt == nil || event == nil {
		return errors.New("capture attempt is unavailable")
	}
	attempt.mu.Lock()
	defer attempt.mu.Unlock()
	if !attempt.open {
		return errors.New("capture attempt is closed")
	}
	if len(attempt.queue) >= maxCaptureQueueEvents {
		return errCaptureQueueFull
	}
	if countBody && attempt.bodyBytes+int64(len(data)) > maxCaptureBodyBytes {
		return errCaptureBodyLimit
	}
	snapshot := bytes.Clone(data)
	if countBody {
		attempt.bodyBytes += int64(len(snapshot))
	}
	attempt.queue = append(attempt.queue, func(target CaptureAttempt) error {
		return event(target, snapshot)
	})
	select {
	case attempt.wake <- struct{}{}:
	default:
	}
	return nil
}

func (attempt *deferredCaptureAttempt) AppendRequestHeaders(data []byte) error {
	return attempt.enqueueBytes(data, false, func(target CaptureAttempt, snapshot []byte) error {
		return target.AppendRequestHeaders(snapshot)
	})
}
func (attempt *deferredCaptureAttempt) AppendRequestBody(data []byte) error {
	return attempt.enqueueBytes(data, true, func(target CaptureAttempt, snapshot []byte) error {
		return target.AppendRequestBody(snapshot)
	})
}
func (attempt *deferredCaptureAttempt) AppendResponseHeaders(data []byte) error {
	return attempt.enqueueBytes(data, false, func(target CaptureAttempt, snapshot []byte) error {
		return target.AppendResponseHeaders(snapshot)
	})
}
func (attempt *deferredCaptureAttempt) AppendResponseBody(data []byte) error {
	return attempt.enqueueBytes(data, true, func(target CaptureAttempt, snapshot []byte) error {
		return target.AppendResponseBody(snapshot)
	})
}
func (attempt *deferredCaptureAttempt) RecordResponseFlush() error {
	return attempt.enqueue(func(target CaptureAttempt) error {
		events, ok := target.(captureResponseEvents)
		if !ok {
			return errors.New("capture target cannot persist response flush")
		}
		return events.RecordResponseFlush()
	})
}
func (attempt *deferredCaptureAttempt) RecordResponseShortWrite(written, requested int) error {
	return attempt.enqueue(func(target CaptureAttempt) error {
		events, ok := target.(captureResponseEvents)
		if !ok {
			return errors.New("capture target cannot persist response short write")
		}
		return events.RecordResponseShortWrite(written, requested)
	})
}
func (attempt *deferredCaptureAttempt) RecordResponseError(err error) error {
	return attempt.enqueue(func(target CaptureAttempt) error {
		events, ok := target.(captureResponseEvents)
		if !ok {
			return errors.New("capture target cannot persist response error")
		}
		return events.RecordResponseError(err)
	})
}
func (attempt *deferredCaptureAttempt) RecordResponseTermination(termination, detail string) error {
	return attempt.enqueue(func(target CaptureAttempt) error {
		events, ok := target.(captureResponseTerminationEvents)
		if !ok {
			return nil
		}
		return events.RecordResponseTermination(termination, detail)
	})
}
func (attempt *deferredCaptureAttempt) RecordResponseHijack(err error) error {
	return attempt.enqueue(func(target CaptureAttempt) error {
		events, ok := target.(captureResponseHijackEvents)
		if !ok {
			return errors.New("capture target cannot persist response hijack")
		}
		return events.RecordResponseHijack(err)
	})
}
func (attempt *deferredCaptureAttempt) RecordRequestError(err error) error {
	return attempt.enqueue(func(target CaptureAttempt) error {
		events, ok := target.(captureRequestEvents)
		if !ok {
			return errors.New("capture target cannot persist request error")
		}
		return events.RecordRequestError(err)
	})
}
func (attempt *deferredCaptureAttempt) RecordRequestClose(err error) error {
	return attempt.enqueue(func(target CaptureAttempt) error {
		events, ok := target.(captureRequestEvents)
		if !ok {
			return errors.New("capture target cannot persist request close")
		}
		return events.RecordRequestClose(err)
	})
}
func (attempt *deferredCaptureAttempt) RecordRequestOutcome(kind string, err error) error {
	return attempt.enqueue(func(target CaptureAttempt) error {
		events, ok := target.(captureRequestEvents)
		if !ok {
			return errors.New("capture target cannot persist request outcome")
		}
		return events.RecordRequestOutcome(kind, err)
	})
}
func (attempt *deferredCaptureAttempt) Wait() {
	if attempt == nil {
		return
	}
	attempt.waitOnce.Do(func() {
		attempt.mu.Lock()
		attempt.open = false
		targetReady := attempt.targetReady
		select {
		case attempt.wake <- struct{}{}:
		default:
		}
		attempt.mu.Unlock()
		<-targetReady
		<-attempt.done
		attempt.mu.Lock()
		target := attempt.target
		attempt.mu.Unlock()
		if target != nil {
			_ = safeAttemptWait(target)
		}
	})
}
func (attempt *deferredCaptureAttempt) Complete() error {
	attempt.Wait()
	attempt.mu.Lock()
	target := attempt.target
	attempt.mu.Unlock()
	if target == nil {
		return attempt.captureFailure()
	}
	return target.Complete()
}
func (attempt *deferredCaptureAttempt) Fail(err error) error {
	attempt.Wait()
	attempt.mu.Lock()
	if attempt.failure == nil && err != nil {
		attempt.failure = err
	}
	target := attempt.target
	attempt.mu.Unlock()
	if target == nil {
		return nil
	}
	return target.Fail(err)
}
func (attempt *deferredCaptureAttempt) captureFailure() error {
	if attempt == nil {
		return nil
	}
	attempt.mu.Lock()
	deferredErr := attempt.failure
	target := attempt.target
	attempt.mu.Unlock()
	if deferredErr != nil {
		return deferredErr
	}
	if failed, ok := target.(interface{ captureFailure() error }); ok {
		return failed.captureFailure()
	}
	return nil
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
	capture := newDeferredDataPlaneCapture(handler.captureFactory, metadata)
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
	var requestContext context.Context
	if ginContext.Request != nil {
		requestContext = ginContext.Request.Context()
	}
	return capture.finalize(requestContext)
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

type captureRequestBody struct {
	io.ReadCloser
	context          context.Context
	observe          func([]byte)
	recordReadError  func(error)
	recordCloseError func(error)
	recordFailure    func(error)
	recordOutcome    func(string, error)
	readOnce         sync.Once
	closeOnce        sync.Once
	cancelOnce       sync.Once
}

func (body *captureRequestBody) Read(p []byte) (int, error) {
	n, err := body.ReadCloser.Read(p)
	if n > 0 && body.observe != nil {
		body.observe(append([]byte(nil), p[:n]...))
	}
	if err != nil && !errors.Is(err, io.EOF) {
		body.readOnce.Do(func() {
			if body.recordReadError != nil {
				body.recordReadError(fmt.Errorf("request body read: %w", err))
			}
			if body.recordOutcome != nil {
				body.recordOutcome("read_error", err)
			}
		})
	}
	body.recordCancellation()
	return n, err
}

func (body *captureRequestBody) Close() error {
	err := body.ReadCloser.Close()
	if err != nil {
		body.closeOnce.Do(func() {
			if body.recordCloseError != nil {
				body.recordCloseError(fmt.Errorf("request body close: %w", err))
			}
			if body.recordOutcome != nil {
				body.recordOutcome("close_error", err)
			}
		})
	}
	body.recordCancellation()
	return err
}

func (body *captureRequestBody) recordCancellation() {
	if body.context == nil || body.context.Err() == nil {
		return
	}
	err := body.context.Err()
	body.cancelOnce.Do(func() {
		if body.recordFailure != nil {
			body.recordFailure(fmt.Errorf("request body context: %w", err))
		}
		if body.recordOutcome != nil {
			body.recordOutcome("context_canceled", err)
		}
	})
}

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
	for observer.active != 0 || (observer.await && !observer.complete) {
		observer.cond.Wait()
	}
	observer.stopped = true
	observer.cond.Broadcast()
	observer.mu.Unlock()
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

type captureResponseWriter struct {
	gin.ResponseWriter
	attempt   CaptureAttempt
	recordErr func(error)
	committed bool
	mu        sync.Mutex
}

func newCaptureResponseWriter(writer gin.ResponseWriter, attempt CaptureAttempt, recordErr ...func(error)) *captureResponseWriter {
	var onError func(error)
	if len(recordErr) > 0 {
		onError = recordErr[0]
	}
	return &captureResponseWriter{ResponseWriter: writer, attempt: attempt, recordErr: onError}
}

func (writer *captureResponseWriter) captureError(err error) {
	if writer != nil && writer.recordErr != nil && err != nil {
		writer.recordErr(err)
	}
}

func (writer *captureResponseWriter) commit(status int) {
	writer.mu.Lock()
	defer writer.mu.Unlock()
	if writer.committed {
		return
	}
	writer.committed = true
	if writer.attempt == nil {
		return
	}
	data := append([]byte("HTTP "+strconv.Itoa(status)+"\r\n"), serializeHeaders(writer.Header())...)
	if err := writer.attempt.AppendResponseHeaders(data); err != nil {
		writer.captureError(fmt.Errorf("capture response headers: %w", err))
	}
}

func (writer *captureResponseWriter) WriteHeader(statusCode int) {
	writer.ResponseWriter.WriteHeader(statusCode)
	writer.commit(statusCode)
}

func (writer *captureResponseWriter) WriteHeaderNow() {
	// Gin's writer owns commit semantics; observe the committed state only
	// after delegating, without synthesizing a second WriteHeader call.
	writer.ResponseWriter.WriteHeaderNow()
	status := writer.ResponseWriter.Status()
	if status == 0 {
		status = http.StatusOK
	}
	writer.commit(status)
}

func (writer *captureResponseWriter) committedValue() bool {
	writer.mu.Lock()
	defer writer.mu.Unlock()
	return writer.committed
}

func (writer *captureResponseWriter) Write(data []byte) (int, error) {
	if !writer.committedValue() {
		writer.WriteHeader(http.StatusOK)
	}
	n, err := writer.ResponseWriter.Write(data)
	if n > 0 && writer.attempt != nil {
		written := n
		if written > len(data) {
			written = len(data)
		}
		if appendErr := writer.attempt.AppendResponseBody(append([]byte(nil), data[:written]...)); appendErr != nil {
			writer.captureError(fmt.Errorf("capture response body: %w", appendErr))
		}
	}
	if events, ok := writer.attempt.(captureResponseEvents); ok {
		if n != len(data) {
			if eventErr := events.RecordResponseShortWrite(n, len(data)); eventErr != nil {
				writer.captureError(eventErr)
			}
		}
		if err != nil {
			if eventErr := events.RecordResponseError(err); eventErr != nil {
				writer.captureError(eventErr)
			}
		}
	}
	if n != len(data) {
		writer.captureError(fmt.Errorf("capture response short write: %w", io.ErrShortWrite))
	}
	if err != nil {
		writer.captureError(fmt.Errorf("capture response write: %w", err))
	}
	return n, err
}

func (writer *captureResponseWriter) WriteString(value string) (int, error) {
	return writer.Write([]byte(value))
}

func (writer *captureResponseWriter) Flush() {
	_ = writer.FlushError()
}

func (writer *captureResponseWriter) FlushError() error {
	if !writer.committedValue() {
		writer.WriteHeader(http.StatusOK)
	}
	err := http.NewResponseController(writer.ResponseWriter).Flush()
	if events, ok := writer.attempt.(captureResponseEvents); ok {
		if err == nil {
			if eventErr := events.RecordResponseFlush(); eventErr != nil {
				writer.captureError(eventErr)
			}
		} else {
			if eventErr := events.RecordResponseError(err); eventErr != nil {
				writer.captureError(eventErr)
			}
			writer.captureError(fmt.Errorf("capture response flush: %w", err))
		}
	}
	return err
}

func (writer *captureResponseWriter) ReadFrom(reader io.Reader) (int64, error) {
	buffer := make([]byte, 32*1024)
	var total int64
	for {
		read, readErr := reader.Read(buffer)
		if read > 0 {
			written, writeErr := writer.Write(buffer[:read])
			total += int64(written)
			if writeErr != nil {
				writer.captureError(fmt.Errorf("capture response write: %w", writeErr))
				return total, writeErr
			}
			if written != read {
				writer.captureError(fmt.Errorf("capture response short write: %w", io.ErrShortWrite))
				return total, io.ErrShortWrite
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				return total, nil
			}
			writer.captureError(fmt.Errorf("capture response read: %w", readErr))
			return total, readErr
		}
	}
}

func (writer *captureResponseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	hijacker, ok := writer.ResponseWriter.(http.Hijacker)
	if !ok {
		err := http.ErrNotSupported
		writer.recordHijackOutcome(err)
		return nil, nil, err
	}
	conn, buffered, err := hijacker.Hijack()
	writer.recordHijackOutcome(err)
	return conn, buffered, err
}

func (writer *captureResponseWriter) recordHijackOutcome(err error) {
	events, supported := writer.attempt.(captureResponseHijackEvents)
	if supported {
		if eventErr := events.RecordResponseHijack(err); eventErr != nil {
			writer.captureError(eventErr)
		}
	} else if err == nil {
		writer.captureError(errors.New("capture target cannot persist successful response hijack"))
	}
	if err != nil {
		if events, ok := writer.attempt.(captureResponseEvents); ok {
			if eventErr := events.RecordResponseError(err); eventErr != nil {
				writer.captureError(eventErr)
			}
		}
		writer.captureError(fmt.Errorf("capture response hijack: %w", err))
	}
}

func (writer *captureResponseWriter) CloseNotify() <-chan bool {
	if notifier, ok := writer.ResponseWriter.(http.CloseNotifier); ok {
		return notifier.CloseNotify()
	}
	return make(chan bool)
}

func (writer *captureResponseWriter) Pusher() http.Pusher {
	if pusher, ok := writer.ResponseWriter.(http.Pusher); ok {
		return pusher
	}
	return nil
}

func (writer *captureResponseWriter) Unwrap() http.ResponseWriter {
	return writer.ResponseWriter
}

var _ gin.ResponseWriter = (*captureResponseWriter)(nil)
var _ io.ReaderFrom = (*captureResponseWriter)(nil)
var _ http.Flusher = (*captureResponseWriter)(nil)
var _ http.Hijacker = (*captureResponseWriter)(nil)
