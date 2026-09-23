//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package container

import (
	"bytes"
	"errors"
	"net/http"
	"sync"

	"gpt-load/internal/control"
	"gpt-load/internal/debugcapture"
	"gpt-load/internal/gateway"
)

// debugCaptureFactory adapts the database-backed capture store to the Gateway
// boundary. It also admits sessions to the runtime so shutdown can drain them
// before the shared database pool closes.
type debugCaptureFactory struct {
	store   *debugcapture.Store
	runtime *debugcapture.Runtime
}

func newDebugCaptureFactory(store *debugcapture.Store, runtime *debugcapture.Runtime) *debugCaptureFactory {
	if store == nil {
		return nil
	}
	return &debugCaptureFactory{store: store, runtime: runtime}
}

func (factory *debugCaptureFactory) StartSession(metadata gateway.CaptureSessionMetadata) (gateway.CaptureSession, error) {
	if factory == nil || factory.store == nil {
		return nil, errors.New("debug capture factory is unavailable")
	}
	var release func()
	if factory.runtime != nil {
		var admitted bool
		release, admitted = factory.runtime.AcquireSession()
		if !admitted {
			return nil, errors.New("debug capture runtime is stopping")
		}
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			if release != nil {
				release()
			}
			panic(recovered)
		}
	}()
	session, err := factory.store.StartSession(mapSessionMetadata(metadata))
	if err != nil {
		if release != nil {
			release()
		}
		return nil, err
	}
	return &debugCaptureSession{
		inner:   session,
		release: release,
	}, nil
}

func (factory *debugCaptureFactory) StartProbeCapture(metadata control.ProbeCaptureSessionMetadata) (control.ProbeCaptureSession, error) {
	session, err := factory.StartSession(gateway.CaptureSessionMetadata{
		RequestID: metadata.RequestID,
		Protocol:  metadata.Protocol,
		Operation: metadata.Operation,
		Fields:    metadata.Fields,
	})
	if err != nil {
		return nil, err
	}
	return &probeDebugCaptureSession{session: session}, nil
}

type debugCaptureSession struct {
	inner       *debugcapture.Session
	release     func()
	releaseOnce sync.Once
}

func (session *debugCaptureSession) StartAttempt(metadata gateway.CaptureAttemptMetadata) (gateway.CaptureAttempt, error) {
	if session == nil || session.inner == nil {
		return nil, errors.New("debug capture session is unavailable")
	}
	attempt, err := session.inner.StartAttempt(mapAttemptMetadata(metadata))
	if err != nil {
		return nil, err
	}
	return &debugCaptureAttempt{inner: attempt}, nil
}

func (session *debugCaptureSession) UpdateMetadata(metadata gateway.CaptureSessionMetadata) error {
	if session == nil || session.inner == nil {
		return errors.New("debug capture session is unavailable")
	}
	return session.inner.UpdateMetadata(mapSessionMetadata(metadata))
}

func (session *debugCaptureSession) Complete() error {
	if session == nil || session.inner == nil {
		return errors.New("debug capture session is unavailable")
	}
	if err := session.inner.Complete(); err != nil {
		return err
	}
	session.releaseAdmission()
	return nil
}

func (session *debugCaptureSession) Fail(cause error) error {
	if session == nil || session.inner == nil {
		return errors.New("debug capture session is unavailable")
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			session.releaseAdmission()
			panic(recovered)
		}
	}()
	err := session.inner.Fail(cause)
	session.releaseAdmission()
	return err
}

func (session *debugCaptureSession) Wait() {}

func (session *debugCaptureSession) releaseAdmission() {
	if session == nil || session.release == nil {
		return
	}
	session.releaseOnce.Do(session.release)
}

type debugCaptureAttempt struct {
	inner *debugcapture.Attempt
}

type probeDebugCaptureSession struct {
	session gateway.CaptureSession
}

func (session *probeDebugCaptureSession) StartProbeAttempt(metadata control.ProbeCaptureAttemptMetadata) (control.ProbeCaptureAttempt, error) {
	fields := make(map[string]string, len(metadata.Fields)+2)
	for key, value := range metadata.Fields {
		fields[key] = value
	}
	attempt, err := session.session.StartAttempt(gateway.CaptureAttemptMetadata{
		AttemptID: metadata.AttemptID,
		Sequence:  metadata.Sequence,
		Fields:    fields,
	})
	if err != nil {
		return nil, err
	}
	return &probeDebugCaptureAttempt{attempt: attempt}, nil
}

func (session *probeDebugCaptureSession) Complete() error      { return session.session.Complete() }
func (session *probeDebugCaptureSession) Fail(err error) error { return session.session.Fail(err) }

type probeDebugCaptureAttempt struct {
	attempt gateway.CaptureAttempt
}

func (attempt *probeDebugCaptureAttempt) AppendRequestHeaders(data []byte) error {
	return attempt.attempt.AppendRequestHeaders(data)
}
func (attempt *probeDebugCaptureAttempt) AppendRequestBody(data []byte) error {
	return attempt.attempt.AppendRequestBody(data)
}
func (attempt *probeDebugCaptureAttempt) AppendResponseHeaders(data []byte) error {
	return attempt.attempt.AppendResponseHeaders(data)
}
func (attempt *probeDebugCaptureAttempt) AppendResponseBody(data []byte) error {
	return attempt.attempt.AppendResponseBody(data)
}
func (attempt *probeDebugCaptureAttempt) RecordResponseTermination(kind, detail string) error {
	if events, ok := attempt.attempt.(interface {
		RecordResponseTermination(string, string) error
	}); ok {
		return events.RecordResponseTermination(kind, detail)
	}
	return nil
}
func (attempt *probeDebugCaptureAttempt) Complete() error      { return attempt.attempt.Complete() }
func (attempt *probeDebugCaptureAttempt) Fail(err error) error { return attempt.attempt.Fail(err) }

func (attempt *debugCaptureAttempt) AppendRequestHeaders(data []byte) error {
	return attempt.inner.AppendHeaders(debugcapture.DirectionRequest, data)
}

func (attempt *debugCaptureAttempt) AppendRequestBody(data []byte) error {
	return attempt.inner.AppendBodyPart(debugcapture.DirectionRequest, bytes.NewReader(data))
}

func (attempt *debugCaptureAttempt) AppendResponseHeaders(data []byte) error {
	return attempt.inner.AppendHeaders(debugcapture.DirectionResponse, data)
}

func (attempt *debugCaptureAttempt) AppendResponseBody(data []byte) error {
	return attempt.inner.AppendBodyPart(debugcapture.DirectionResponse, bytes.NewReader(data))
}

func (attempt *debugCaptureAttempt) RecordResponseFlush() error {
	return attempt.inner.RecordResponseFlush()
}

func (attempt *debugCaptureAttempt) RecordResponseShortWrite(written, requested int) error {
	return attempt.inner.RecordResponseShortWrite(written, requested)
}

func (attempt *debugCaptureAttempt) RecordResponseError(err error) error {
	return attempt.inner.RecordResponseError(err)
}

func (attempt *debugCaptureAttempt) RecordResponseTermination(termination, detail string) error {
	return attempt.inner.RecordResponseTermination(termination, detail)
}

func (attempt *debugCaptureAttempt) RecordResponseHijack(err error) error {
	return attempt.inner.RecordResponseHijack(err)
}

func (attempt *debugCaptureAttempt) RecordRequestError(err error) error {
	return attempt.inner.RecordRequestError(err)
}

func (attempt *debugCaptureAttempt) RecordRequestClose(err error) error {
	return attempt.inner.RecordRequestClose(err)
}

func (attempt *debugCaptureAttempt) RecordRequestOutcome(kind string, err error) error {
	return attempt.inner.RecordRequestOutcome(kind, err)
}

func (attempt *debugCaptureAttempt) Complete() error {
	return attempt.inner.Complete()
}

func (attempt *debugCaptureAttempt) Fail(err error) error {
	return attempt.inner.Fail(err)
}

func (attempt *debugCaptureAttempt) Wait() {}

func mapSessionMetadata(metadata gateway.CaptureSessionMetadata) debugcapture.SessionMetadata {
	fields := make(map[string]any, len(metadata.Fields)+3)
	for key, value := range metadata.Fields {
		fields[key] = value
	}
	if metadata.Method != "" {
		fields["client_method"] = metadata.Method
	}
	if metadata.Path != "" {
		fields["client_path"] = metadata.Path
	}
	if metadata.Headers != nil {
		fields["client_headers"] = cloneHeaderValues(metadata.Headers)
	}
	return debugcapture.SessionMetadata{
		RequestID:   metadata.RequestID,
		AccessKeyID: metadata.AccessKeyID,
		Protocol:    metadata.Protocol,
		Operation:   metadata.Operation,
		Fields:      fields,
	}
}

func mapAttemptMetadata(metadata gateway.CaptureAttemptMetadata) debugcapture.AttemptMetadata {
	fields := make(map[string]any, len(metadata.Fields)+2)
	for key, value := range metadata.Fields {
		fields[key] = value
	}
	fields["logical_attempt_id"] = metadata.AttemptID
	fields["logical_sequence"] = metadata.Sequence
	return debugcapture.AttemptMetadata{Fields: fields}
}

func cloneHeaderValues(headers http.Header) map[string][]string {
	result := make(map[string][]string, len(headers))
	for key, values := range headers {
		result[key] = append([]string(nil), values...)
	}
	return result
}

var _ gateway.CaptureFactory = (*debugCaptureFactory)(nil)
var _ gateway.CaptureSession = (*debugCaptureSession)(nil)
var _ gateway.CaptureSessionMetadataUpdater = (*debugCaptureSession)(nil)
var _ gateway.CaptureAttempt = (*debugCaptureAttempt)(nil)
var _ control.ProbeCaptureFactory = (*debugCaptureFactory)(nil)
var _ control.ProbeCaptureSession = (*probeDebugCaptureSession)(nil)
var _ control.ProbeCaptureAttempt = (*probeDebugCaptureAttempt)(nil)
