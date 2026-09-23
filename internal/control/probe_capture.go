package control

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"sync"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
)

// ProbeCaptureSessionMetadata identifies one manually executed provider probe.
type ProbeCaptureSessionMetadata struct {
	RequestID string
	Protocol  string
	Operation string
	Fields    map[string]string
}

// ProbeCaptureAttemptMetadata identifies one provider HTTP attempt within a probe.
type ProbeCaptureAttemptMetadata struct {
	AttemptID string
	Sequence  uint32
	Fields    map[string]string
}

// ProbeCaptureFactory is the control-owned adapter boundary for debug capture storage.
type ProbeCaptureFactory interface {
	StartProbeCapture(ProbeCaptureSessionMetadata) (ProbeCaptureSession, error)
}

// ProbeCaptureSession persists one probe and its ordered provider attempts.
type ProbeCaptureSession interface {
	StartProbeAttempt(ProbeCaptureAttemptMetadata) (ProbeCaptureAttempt, error)
	Complete() error
	Fail(error) error
}

// ProbeCaptureAttempt stores raw communication observed for one provider attempt.
type ProbeCaptureAttempt interface {
	AppendRequestHeaders([]byte) error
	AppendRequestBody([]byte) error
	AppendResponseHeaders([]byte) error
	AppendResponseBody([]byte) error
	RecordResponseTermination(string, string) error
	Complete() error
	Fail(error) error
}

func (service *Service) SetProbeCaptureFactory(factory ProbeCaptureFactory) {
	if service != nil {
		service.probeCaptures = factory
	}
}

func (probe *credentialProbeExecutor) startCapture(parent context.Context, spec execution.AttemptSpec) (context.Context, func(execution.AttemptResult)) {
	ctx := parent
	if ctx == nil {
		ctx = context.Background()
	}
	if probe.captureFactory == nil {
		return ctx, func(execution.AttemptResult) {}
	}
	provider, _ := probe.channels.ProviderKind(channel.ID(spec.ChannelID))
	session, err := probe.captureFactory.StartProbeCapture(ProbeCaptureSessionMetadata{
		RequestID: spec.RequestID,
		Protocol:  string(spec.ClientProtocol),
		Operation: string(spec.Operation),
		Fields: map[string]string{
			"provider": string(provider), "channel_id": spec.ChannelID,
			"model": spec.UpstreamModel, "route_mode": string(spec.RouteMode),
			"retry": "false", "fallback": "false",
		},
	})
	if err != nil || session == nil {
		return ctx, func(execution.AttemptResult) {}
	}
	attempt, err := session.StartProbeAttempt(ProbeCaptureAttemptMetadata{
		AttemptID: spec.AttemptID,
		Sequence:  spec.Sequence,
		Fields: map[string]string{
			"provider": string(provider), "channel_id": spec.ChannelID,
			"model": spec.UpstreamModel, "route_mode": string(spec.RouteMode),
			"retry": "false", "fallback": "false",
		},
	})
	if err != nil || attempt == nil {
		_ = session.Fail(err)
		return ctx, func(execution.AttemptResult) {}
	}
	observer := newProbeCaptureHTTPObserver(spec.AttemptID, attempt)
	ctx = execution.WithHTTPAttemptID(ctx, spec.AttemptID)
	ctx = execution.WithHTTPObserver(ctx, observer)
	return ctx, func(execution.AttemptResult) {
		if err := observer.SealAndDrain(); err != nil {
			_ = attempt.Fail(err)
			_ = session.Fail(err)
			return
		}
		if err := attempt.Complete(); err != nil {
			_ = attempt.Fail(err)
			_ = session.Fail(err)
			return
		}
		if err := session.Complete(); err != nil {
			_ = session.Fail(err)
		}
	}
}

type probeCaptureHTTPObserver struct {
	mu           sync.Mutex
	cond         *sync.Cond
	attemptID    string
	attempt      ProbeCaptureAttempt
	sealed       bool
	inFlight     int
	observations int
	writeErr     error
	terminated   bool
}

func newProbeCaptureHTTPObserver(attemptID string, attempt ProbeCaptureAttempt) *probeCaptureHTTPObserver {
	observer := &probeCaptureHTTPObserver{attemptID: attemptID, attempt: attempt}
	observer.cond = sync.NewCond(&observer.mu)
	return observer
}

func (observer *probeCaptureHTTPObserver) ObserveRequest(id string, request *http.Request) {
	if request == nil || !observer.begin(id) {
		return
	}
	defer observer.end()
	var headers bytes.Buffer
	if request.Host != "" {
		_, _ = fmt.Fprintf(&headers, "Host: %s\r\n", request.Host)
	}
	_ = request.Header.Write(&headers)
	observer.recordError(observer.attempt.AppendRequestHeaders(headers.Bytes()))
}

func (observer *probeCaptureHTTPObserver) ObserveRequestBody(id string, body []byte) {
	if !observer.begin(id) {
		return
	}
	defer observer.end()
	observer.recordError(observer.attempt.AppendRequestBody(body))
}

func (observer *probeCaptureHTTPObserver) ObserveResponse(id string, status int, headers http.Header) {
	if !observer.begin(id) {
		return
	}
	defer observer.end()
	var raw bytes.Buffer
	_, _ = fmt.Fprintf(&raw, "HTTP %d\r\n", status)
	_ = headers.Write(&raw)
	observer.recordError(observer.attempt.AppendResponseHeaders(raw.Bytes()))
}

func (observer *probeCaptureHTTPObserver) ObserveResponseBody(id string, body []byte) {
	if !observer.begin(id) {
		return
	}
	defer observer.end()
	observer.recordError(observer.attempt.AppendResponseBody(body))
}

func (observer *probeCaptureHTTPObserver) ObserveResponseTermination(id, kind, detail string) {
	if !observer.begin(id) {
		return
	}
	defer observer.end()
	observer.mu.Lock()
	observer.terminated = true
	observer.mu.Unlock()
	observer.recordError(observer.attempt.RecordResponseTermination(kind, detail))
}

func (observer *probeCaptureHTTPObserver) ObserveResponseComplete(id string, trailers http.Header, err error) {
	if !observer.begin(id) {
		return
	}
	defer observer.end()
	if len(trailers) > 0 {
		var raw bytes.Buffer
		_, _ = raw.WriteString("Trailers:\r\n")
		_ = trailers.Write(&raw)
		observer.recordError(observer.attempt.AppendResponseHeaders(raw.Bytes()))
	}
	observer.mu.Lock()
	terminated := observer.terminated
	observer.mu.Unlock()
	if !terminated && err != nil {
		kind := "read_error"
		if errors.Is(err, context.DeadlineExceeded) {
			kind = "timeout"
		} else if errors.Is(err, context.Canceled) {
			kind = "close"
		}
		observer.recordError(observer.attempt.RecordResponseTermination(kind, err.Error()))
	} else if !terminated {
		observer.recordError(observer.attempt.RecordResponseTermination("eof", ""))
	}
}

func (observer *probeCaptureHTTPObserver) recordError(err error) {
	if err == nil {
		return
	}
	observer.mu.Lock()
	observer.writeErr = errors.Join(observer.writeErr, err)
	observer.mu.Unlock()
}

func (observer *probeCaptureHTTPObserver) BeginHTTPObservation(id string) {
	if observer == nil {
		return
	}
	observer.mu.Lock()
	if !observer.sealed && observer.attempt != nil && id == observer.attemptID {
		observer.observations++
	}
	observer.mu.Unlock()
}

func (observer *probeCaptureHTTPObserver) EndHTTPObservation(id string) {
	if observer == nil {
		return
	}
	observer.mu.Lock()
	if id == observer.attemptID && observer.observations > 0 {
		observer.observations--
		observer.cond.Broadcast()
	}
	observer.mu.Unlock()
}

func (observer *probeCaptureHTTPObserver) begin(id string) bool {
	if observer == nil {
		return false
	}
	observer.mu.Lock()
	defer observer.mu.Unlock()
	if observer.attempt == nil || (observer.sealed && observer.observations == 0) || id != observer.attemptID {
		return false
	}
	observer.inFlight++
	return true
}

func (observer *probeCaptureHTTPObserver) end() {
	observer.mu.Lock()
	observer.inFlight--
	if observer.sealed && observer.inFlight == 0 && observer.observations == 0 {
		observer.cond.Broadcast()
	}
	observer.mu.Unlock()
}

func (observer *probeCaptureHTTPObserver) SealAndDrain() error {
	if observer == nil {
		return nil
	}
	observer.mu.Lock()
	observer.sealed = true
	for observer.inFlight > 0 || observer.observations > 0 {
		observer.cond.Wait()
	}
	err := observer.writeErr
	observer.mu.Unlock()
	return err
}

var _ execution.HTTPObserver = (*probeCaptureHTTPObserver)(nil)
var _ execution.HTTPObservationLifecycle = (*probeCaptureHTTPObserver)(nil)
