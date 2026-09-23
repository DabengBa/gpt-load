package control

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"gpt-load/internal/execution"
)

func TestCredentialProbeCaptureReturnsWhenObserverNeverStarts(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "probe-capture-no-observer-secret")
	credential := takeGroupCredential(t, fixture, groupID)
	capture := &probeCaptureTestFactory{}
	fixture.service.probeCaptures = capture
	want := failedCredentialProbeResult(http.StatusBadGateway, execution.ErrorKindProvider, "")
	fixture.service.executor = &probeCaptureTestExecutor{result: want}

	type probeOutcome struct {
		response CredentialProbeResponse
		err      error
	}
	finished := make(chan probeOutcome, 1)
	go func() {
		response, err := fixture.service.TestGroupCredential(t.Context(), groupID, credential.ID)
		finished <- probeOutcome{response: response, err: err}
	}()

	select {
	case got := <-finished:
		if got.err != nil {
			t.Fatal(got.err)
		}
		if got.response.Outcome != ProbeOutcomeInconclusive {
			t.Fatalf("probe outcome = %q, want inconclusive", got.response.Outcome)
		}
	case <-time.After(time.Second):
		t.Fatal("probe blocked waiting for an observer completion that never started")
	}

	if len(capture.sessions) != 1 || len(capture.sessions[0].attempts) != 1 {
		t.Fatalf("capture sessions/attempts = %d/%d, want 1/1", len(capture.sessions), len(capture.sessions[0].attempts))
	}
	attempt := capture.sessions[0].attempts[0]
	if !capture.sessions[0].completed || !attempt.completed {
		t.Fatalf("capture session/attempt completed = %t/%t, want true/true", capture.sessions[0].completed, attempt.completed)
	}
	if len(attempt.requestHeaders)+len(attempt.requestBody)+len(attempt.responseHeaders)+len(attempt.responseBody)+len(attempt.terminations) != 0 {
		t.Fatalf("capture claims provider communication without observer callbacks: %#v", attempt)
	}
}

func TestCredentialProbeCapturePreservesObservedProviderCommunication(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "probe-capture-secret")
	credential := takeGroupCredential(t, fixture, groupID)
	capture := &probeCaptureTestFactory{}
	fixture.service.probeCaptures = capture
	fixture.service.executor = &probeCaptureTestExecutor{
		result: failedCredentialProbeResult(http.StatusBadGateway, execution.ErrorKindProvider, ""),
		execute: func(ctx context.Context, spec execution.AttemptSpec) execution.AttemptResult {
			observer := execution.HTTPObserverFromContext(ctx)
			if observer == nil {
				t.Fatal("probe execution context has no HTTP observer")
			}
			observer.ObserveRequest(spec.AttemptID, &http.Request{
				Method: http.MethodPost,
				Header: http.Header{"Content-Type": {"application/json"}, "X-Probe": {"yes"}},
			})
			observer.ObserveRequestBody(spec.AttemptID, []byte(`{"prompt":`))
			observer.ObserveRequestBody(spec.AttemptID, []byte(`"probe"}`))
			observer.ObserveResponse(spec.AttemptID, http.StatusBadGateway, http.Header{"Content-Type": {"application/json"}})
			observer.ObserveResponseBody(spec.AttemptID, []byte(`{"error":`))
			observer.ObserveResponseBody(spec.AttemptID, []byte(`"malformed`))
			observeProbeTermination(t, observer, spec.AttemptID, "read_error", "unexpected EOF")
			observer.ObserveResponseComplete(spec.AttemptID, nil, errors.New("unexpected EOF"))
			return failedCredentialProbeResult(http.StatusBadGateway, execution.ErrorKindProvider, "")
		},
	}

	response, err := fixture.service.TestGroupCredential(t.Context(), groupID, credential.ID)
	if err != nil {
		t.Fatal(err)
	}
	if response.Outcome != ProbeOutcomeInconclusive {
		t.Fatalf("probe outcome = %q, want inconclusive", response.Outcome)
	}
	assertProbeCaptureRecorded(t, capture)
}

func TestModelProbeCapturePreservesObservedProviderCommunication(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "model-probe-capture-secret")
	capture := &probeCaptureTestFactory{}
	fixture.service.probeCaptures = capture
	fixture.service.executor = &probeCaptureTestExecutor{
		result: successfulCredentialProbeResult(),
		execute: func(ctx context.Context, spec execution.AttemptSpec) execution.AttemptResult {
			observer := execution.HTTPObserverFromContext(ctx)
			if observer == nil {
				t.Fatal("probe execution context has no HTTP observer")
			}
			observer.ObserveRequest(spec.AttemptID, &http.Request{Method: http.MethodPost, Header: http.Header{"X-Probe": {"model"}}})
			observer.ObserveRequestBody(spec.AttemptID, []byte("request"))
			observer.ObserveResponse(spec.AttemptID, http.StatusOK, http.Header{"Content-Type": {"text/event-stream"}})
			observer.ObserveResponseBody(spec.AttemptID, []byte("data: {\"partial\":"))
			observeProbeTermination(t, observer, spec.AttemptID, "eof", "")
			observer.ObserveResponseComplete(spec.AttemptID, nil, nil)
			return successfulCredentialProbeResult()
		},
	}

	response, err := fixture.service.ProbeGroupModels(t.Context(), ModelProbeRequest{
		Targets: []ModelProbeTargetRequest{{GroupID: groupID, Model: probeTestModel}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Results) != 1 || response.Results[0].LogID == nil {
		t.Fatalf("model probe result = %#v", response.Results)
	}
	assertProbeCaptureRecorded(t, capture)
}

func TestCredentialProbeCaptureWriteFailureDoesNotChangeProbeOutcome(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "probe-capture-write-failure-secret")
	credential := takeGroupCredential(t, fixture, groupID)
	capture := &probeCaptureTestFactory{failWrites: true}
	fixture.service.probeCaptures = capture
	fixture.service.executor = &probeCaptureTestExecutor{
		result: successfulCredentialProbeResult(),
		execute: func(ctx context.Context, spec execution.AttemptSpec) execution.AttemptResult {
			observer := execution.HTTPObserverFromContext(ctx)
			observer.ObserveRequest(spec.AttemptID, &http.Request{Method: http.MethodPost, Header: http.Header{}})
			observer.ObserveRequestBody(spec.AttemptID, []byte("request"))
			observer.ObserveResponse(spec.AttemptID, http.StatusOK, http.Header{})
			observer.ObserveResponseBody(spec.AttemptID, []byte("response"))
			observer.ObserveResponseComplete(spec.AttemptID, nil, nil)
			return successfulCredentialProbeResult()
		},
	}

	response, err := fixture.service.TestGroupCredential(t.Context(), groupID, credential.ID)
	if err != nil || response.Outcome != ProbeOutcomePassed {
		t.Fatalf("probe response = %#v, err = %v; capture write failures must not alter it", response, err)
	}
	assertProbeCaptureFailed(t, capture)
}

func TestCredentialProbeCaptureTerminationFailureDoesNotChangeProbeOutcome(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "probe-capture-termination-failure-secret")
	credential := takeGroupCredential(t, fixture, groupID)
	capture := &probeCaptureTestFactory{failTermination: true}
	fixture.service.probeCaptures = capture
	fixture.service.executor = &probeCaptureTestExecutor{
		execute: func(ctx context.Context, spec execution.AttemptSpec) execution.AttemptResult {
			execution.HTTPObserverFromContext(ctx).ObserveResponseComplete(spec.AttemptID, nil, nil)
			return successfulCredentialProbeResult()
		},
	}
	response, err := fixture.service.TestGroupCredential(t.Context(), groupID, credential.ID)
	if err != nil || response.Outcome != ProbeOutcomePassed {
		t.Fatalf("probe response = %#v, err = %v; want passed without executor error", response, err)
	}
	assertProbeCaptureFailed(t, capture)
}

func TestCredentialProbeCaptureDrainsQueuedObserverAfterExecute(t *testing.T) {
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "probe-capture-drain-secret")
	credential := takeGroupCredential(t, fixture, groupID)
	capture := &probeCaptureTestFactory{}
	fixture.service.probeCaptures = capture
	registered := make(chan struct{})
	release := make(chan struct{})
	callbackDone := make(chan struct{})
	fixture.service.executor = &probeCaptureTestExecutor{
		execute: func(ctx context.Context, spec execution.AttemptSpec) execution.AttemptResult {
			observer := execution.HTTPObserverFromContext(ctx)
			lifecycle, ok := observer.(execution.HTTPObservationLifecycle)
			if !ok {
				t.Error("probe observer lacks HTTP observation lifecycle")
				return successfulCredentialProbeResult()
			}
			lifecycle.BeginHTTPObservation(spec.AttemptID)
			close(registered)
			go func() {
				defer close(callbackDone)
				<-release
				observer.ObserveResponseBody(spec.AttemptID, []byte("queued raw response"))
				observer.ObserveResponseComplete(spec.AttemptID, nil, nil)
				lifecycle.EndHTTPObservation(spec.AttemptID)
			}()
			return successfulCredentialProbeResult()
		},
	}
	type outcome struct {
		response CredentialProbeResponse
		err      error
	}
	finished := make(chan outcome, 1)
	go func() {
		response, err := fixture.service.TestGroupCredential(t.Context(), groupID, credential.ID)
		finished <- outcome{response, err}
	}()
	select {
	case <-registered:
	case <-time.After(time.Second):
		t.Fatal("observation was not registered")
	}
	select {
	case <-finished:
		t.Fatal("probe completed while registered observer callback was still queued")
	case <-time.After(25 * time.Millisecond):
	}
	close(release)
	select {
	case got := <-finished:
		if got.err != nil || got.response.Outcome != ProbeOutcomePassed {
			t.Fatalf("probe response = %#v, err = %v", got.response, got.err)
		}
	case <-time.After(time.Second):
		t.Fatal("probe did not finish after observer drained")
	}
	<-callbackDone
	attempt := capture.sessions[0].attempts[0]
	if !attempt.completed || string(attempt.responseBody) != "queued raw response" || len(attempt.terminations) != 1 {
		t.Fatalf("drained capture = %#v", attempt)
	}
}

func assertProbeCaptureFailed(t *testing.T, capture *probeCaptureTestFactory) {
	t.Helper()
	if len(capture.sessions) != 1 || len(capture.sessions[0].attempts) != 1 {
		t.Fatalf("capture sessions = %#v, want one session and attempt", capture.sessions)
	}
	session := capture.sessions[0]
	attempt := session.attempts[0]
	if session.completed || attempt.completed || session.failed == nil || attempt.failed == nil {
		t.Fatalf("capture terminal state: session completed=%t failed=%v, attempt completed=%t failed=%v", session.completed, session.failed, attempt.completed, attempt.failed)
	}
}

func TestProbeCaptureHTTPObserverRecordsTerminationKinds(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		complete    error
		termination string
	}{
		{name: "eof", termination: "eof"},
		{name: "close", complete: context.Canceled, termination: "close"},
		{name: "timeout", complete: context.DeadlineExceeded, termination: "timeout"},
		{name: "read error", complete: errors.New("read failed"), termination: "read_error"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			attempt := &probeCaptureTestAttempt{}
			observer := newProbeCaptureHTTPObserver("attempt", attempt)
			observer.ObserveResponseComplete("attempt", nil, test.complete)
			observer.SealAndDrain()
			if len(attempt.terminations) != 1 || attempt.terminations[0] != test.termination {
				t.Fatalf("termination events = %#v, want %q", attempt.terminations, test.termination)
			}
		})
	}
	t.Run("explicit close", func(t *testing.T) {
		attempt := &probeCaptureTestAttempt{}
		observer := newProbeCaptureHTTPObserver("attempt", attempt)
		observeProbeTermination(t, observer, "attempt", "closed", "body closed")
		observer.ObserveResponseComplete("attempt", nil, context.Canceled)
		observer.SealAndDrain()
		if len(attempt.terminations) != 1 || attempt.terminations[0] != "closed" {
			t.Fatalf("termination events = %#v, want explicit close", attempt.terminations)
		}
	})
}

func TestProbeCaptureObserverDrainsAcceptedAsyncCallback(t *testing.T) {
	base := &probeCaptureTestAttempt{}
	attempt := &blockingProbeCaptureAttempt{
		probeCaptureTestAttempt: base,
		entered:                 make(chan struct{}),
		release:                 make(chan struct{}),
	}
	observer := newProbeCaptureHTTPObserver("attempt", attempt)
	callbackDone := make(chan struct{})
	go func() {
		defer close(callbackDone)
		observer.ObserveResponseBody("attempt", []byte("raw response"))
	}()
	<-attempt.entered

	sealed := make(chan struct{})
	go func() {
		observer.SealAndDrain()
		close(sealed)
	}()
	close(attempt.release)
	select {
	case <-sealed:
	case <-time.After(time.Second):
		t.Fatal("observer drain did not finish after accepted callback returned")
	}
	<-callbackDone
	if string(base.responseBody) != "raw response" {
		t.Fatalf("captured response body = %q, want raw response", base.responseBody)
	}
}

type blockingProbeCaptureAttempt struct {
	*probeCaptureTestAttempt
	entered chan struct{}
	release chan struct{}
}

func (attempt *blockingProbeCaptureAttempt) AppendResponseBody(data []byte) error {
	close(attempt.entered)
	<-attempt.release
	return attempt.probeCaptureTestAttempt.AppendResponseBody(data)
}

func observeProbeTermination(t *testing.T, observer execution.HTTPObserver, attemptID, kind, detail string) {
	t.Helper()
	terminationObserver, ok := observer.(interface {
		ObserveResponseTermination(string, string, string)
	})
	if !ok {
		t.Fatal("probe HTTP observer does not preserve response termination")
	}
	terminationObserver.ObserveResponseTermination(attemptID, kind, detail)
}

func assertProbeCaptureRecorded(t *testing.T, factory *probeCaptureTestFactory) {
	t.Helper()
	if len(factory.sessions) != 1 {
		t.Fatalf("capture sessions = %d, want one", len(factory.sessions))
	}
	session := factory.sessions[0]
	if session.metadata.RequestID == "" || session.metadata.Operation != string(execution.OperationProbe) ||
		session.metadata.Protocol == "" || session.metadata.Fields["model"] == "" {
		t.Fatalf("capture session metadata = %#v", session.metadata)
	}
	if !session.completed || len(session.attempts) != 1 {
		t.Fatalf("capture session completed=%t attempts=%d", session.completed, len(session.attempts))
	}
	attempt := session.attempts[0]
	if attempt.metadata.AttemptID == "" || attempt.metadata.Sequence != 1 ||
		attempt.metadata.Fields["provider"] == "" || attempt.metadata.Fields["model"] == "" {
		t.Fatalf("capture attempt metadata = %#v", attempt.metadata)
	}
	if string(attempt.requestHeaders) == "" || string(attempt.requestBody) == "" || string(attempt.responseHeaders) == "" ||
		string(attempt.responseBody) == "" || len(attempt.terminations) != 1 ||
		!attempt.completed {
		t.Fatalf("capture attempt = %#v", attempt)
	}
}

type probeCaptureTestExecutor struct {
	result  execution.AttemptResult
	execute func(context.Context, execution.AttemptSpec) execution.AttemptResult
}

func (executor *probeCaptureTestExecutor) Execute(ctx context.Context, spec execution.AttemptSpec) execution.AttemptResult {
	if executor.execute != nil {
		return executor.execute(ctx, spec)
	}
	return executor.result
}

func (*probeCaptureTestExecutor) ExecuteStream(context.Context, execution.AttemptSpec, execution.StreamSink) execution.StreamResult {
	panic("unexpected stream probe execution")
}

type probeCaptureTestFactory struct {
	sessions        []*probeCaptureTestSession
	failWrites      bool
	failTermination bool
}

func (factory *probeCaptureTestFactory) StartProbeCapture(metadata ProbeCaptureSessionMetadata) (ProbeCaptureSession, error) {
	session := &probeCaptureTestSession{metadata: metadata, failWrites: factory.failWrites, failTermination: factory.failTermination}
	factory.sessions = append(factory.sessions, session)
	return session, nil
}

type probeCaptureTestSession struct {
	metadata        ProbeCaptureSessionMetadata
	attempts        []*probeCaptureTestAttempt
	failWrites      bool
	failTermination bool
	completed       bool
	failed          error
}

func (session *probeCaptureTestSession) StartProbeAttempt(metadata ProbeCaptureAttemptMetadata) (ProbeCaptureAttempt, error) {
	attempt := &probeCaptureTestAttempt{metadata: metadata, failWrites: session.failWrites, failTermination: session.failTermination}
	session.attempts = append(session.attempts, attempt)
	return attempt, nil
}

func (session *probeCaptureTestSession) Complete() error {
	session.completed = true
	return nil
}

func (session *probeCaptureTestSession) Fail(err error) error { session.failed = err; return nil }

type probeCaptureTestAttempt struct {
	metadata        ProbeCaptureAttemptMetadata
	failWrites      bool
	failTermination bool
	requestHeaders  []byte
	requestBody     []byte
	responseHeaders []byte
	responseBody    []byte
	terminations    []string
	completed       bool
	failed          error
}

func (attempt *probeCaptureTestAttempt) AppendRequestHeaders(data []byte) error {
	if attempt.failWrites {
		return errors.New("injected capture write failure")
	}
	attempt.requestHeaders = append(attempt.requestHeaders, data...)
	return nil
}
func (attempt *probeCaptureTestAttempt) AppendRequestBody(data []byte) error {
	if attempt.failWrites {
		return errors.New("injected capture write failure")
	}
	attempt.requestBody = append(attempt.requestBody, data...)
	return nil
}
func (attempt *probeCaptureTestAttempt) AppendResponseHeaders(data []byte) error {
	if attempt.failWrites {
		return errors.New("injected capture write failure")
	}
	attempt.responseHeaders = append(attempt.responseHeaders, data...)
	return nil
}
func (attempt *probeCaptureTestAttempt) AppendResponseBody(data []byte) error {
	if attempt.failWrites {
		return errors.New("injected capture write failure")
	}
	attempt.responseBody = append(attempt.responseBody, data...)
	return nil
}
func (attempt *probeCaptureTestAttempt) RecordResponseTermination(kind, _ string) error {
	if attempt.failTermination {
		return errors.New("injected termination write failure")
	}
	attempt.terminations = append(attempt.terminations, kind)
	return nil
}
func (attempt *probeCaptureTestAttempt) Complete() error {
	attempt.completed = true
	return nil
}
func (attempt *probeCaptureTestAttempt) Fail(err error) error { attempt.failed = err; return nil }

var _ ProbeCaptureFactory = (*probeCaptureTestFactory)(nil)
var _ ProbeCaptureSession = (*probeCaptureTestSession)(nil)
var _ ProbeCaptureAttempt = (*probeCaptureTestAttempt)(nil)
