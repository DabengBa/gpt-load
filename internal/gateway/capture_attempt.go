package gateway

import (
	"bytes"
	"errors"
	"sync"
)

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
