//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package container

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/control"
	"gpt-load/internal/debugcapture"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/state"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/testutil/encryptiontest"
	"gpt-load/internal/testutil/sqlitetest"
)

type modelTraceCaptureCall struct {
	spec       execution.AttemptSpec
	index      int
	observerID string
}

const modelTraceCaptureContextIndex = 3

// The source below observes an actual local HTTP round trip, not the result
// object's reconstructed body. The barrier guarantees concurrent challenges.
type modelTraceCaptureExecutor struct {
	url        string
	mode       string
	mu         sync.Mutex
	calls      []modelTraceCaptureCall
	challenges int
	ready      chan struct{}
}

func (e *modelTraceCaptureExecutor) Execute(ctx context.Context, spec execution.AttemptSpec) execution.AttemptResult {
	e.mu.Lock()
	index := modelTraceCaptureContextIndex
	if spec.PromptText != "Reply with the word hello." {
		index = e.challenges
		e.challenges++
		if e.challenges == 3 {
			close(e.ready)
		}
	}
	e.calls = append(e.calls, modelTraceCaptureCall{spec: spec, index: index, observerID: execution.HTTPAttemptIDFromContext(ctx)})
	e.mu.Unlock()
	select {
	case <-e.ready:
	case <-ctx.Done():
		return execution.AttemptResult{Error: &execution.ErrorEvidence{Kind: execution.ErrorKindCanceled}}
	}
	if e.mode == "missing" && index == 2 {
		return execution.AttemptResult{DispatchState: execution.DispatchMaybeSent, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindTransport}}
	}
	if e.mode == "incomplete" && index == 2 {
		observer := execution.HTTPObserverFromContext(ctx)
		if observer != nil {
			lifecycle := observer.(execution.HTTPObservationLifecycle)
			lifecycle.BeginHTTPObservation(spec.AttemptID)
			observer.ObserveRequestBody(spec.AttemptID, []byte("partial-request"))
			lifecycle.EndHTTPObservation(spec.AttemptID)
		}
		return execution.AttemptResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, GeneratedText: "1 2 3"}
	}
	requestCtx := ctx
	cancel := func() {}
	if e.mode == "cancel" || index == 2 {
		if e.mode == "cancel" {
			requestCtx, cancel = context.WithCancel(ctx)
			timer := time.AfterFunc(100*time.Millisecond, cancel)
			defer timer.Stop()
		} else {
			requestCtx, cancel = context.WithTimeout(ctx, 100*time.Millisecond)
		}
	}
	defer cancel()
	body := []byte(fmt.Sprintf("request-%d", index))
	req, _ := http.NewRequestWithContext(requestCtx, http.MethodPost, fmt.Sprintf("%s/%d?mode=%s", e.url, index, e.mode), bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/octet-stream")
	observer := execution.HTTPObserverFromContext(ctx)
	if observer != nil {
		if lifecycle, ok := observer.(execution.HTTPObservationLifecycle); ok {
			lifecycle.BeginHTTPObservation(spec.AttemptID)
			defer lifecycle.EndHTTPObservation(spec.AttemptID)
		}
		observer.ObserveRequest(spec.AttemptID, req)
		observer.ObserveRequestBody(spec.AttemptID, body)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		if observer != nil {
			observer.ObserveResponseComplete(spec.AttemptID, nil, err)
		}
		kind := execution.ErrorKindTimeout
		if e.mode == "cancel" {
			kind = execution.ErrorKindCanceled
		}
		return execution.AttemptResult{DispatchState: execution.DispatchMaybeSent, Error: &execution.ErrorEvidence{Kind: kind}}
	}
	defer resp.Body.Close()
	responseBody, readErr := io.ReadAll(resp.Body)
	if observer != nil {
		observer.ObserveResponse(spec.AttemptID, resp.StatusCode, resp.Header)
		observer.ObserveResponseBody(spec.AttemptID, responseBody)
		observer.ObserveResponseComplete(spec.AttemptID, resp.Trailer, readErr)
	}
	result := execution.AttemptResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: resp.StatusCode, Header: resp.Header, Body: responseBody}
	if resp.StatusCode == http.StatusOK {
		result.GeneratedText = "1 2 3"
	} else {
		result.Error = &execution.ErrorEvidence{Kind: execution.ErrorKindHTTP, StatusCode: resp.StatusCode}
	}
	return result
}

func (*modelTraceCaptureExecutor) ExecuteStream(context.Context, execution.AttemptSpec, execution.StreamSink) execution.StreamResult {
	panic("unexpected streaming request")
}

type modelTraceCaptureSink struct {
	mu     sync.Mutex
	events []telemetry.RequestEvent
}

type failingModelTraceCaptureFactory struct{}

func (failingModelTraceCaptureFactory) StartProbeCapture(control.ProbeCaptureSessionMetadata) (control.ProbeCaptureSession, error) {
	return nil, errors.New("synthetic capture failure")
}

func (s *modelTraceCaptureSink) Emit(event telemetry.RequestEvent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, event)
}

func TestModelTraceRealCaptureStore(t *testing.T) {
	t.Setenv("TMPDIR", t.TempDir())
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		if r.URL.Query().Get("mode") == "cancel" || r.URL.Path == "/2" {
			<-r.Context().Done()
			return
		}
		if r.URL.Path == "/1" {
			w.WriteHeader(http.StatusBadGateway)
		}
		_, _ = fmt.Fprintf(w, "response-%s", strings.TrimPrefix(r.URL.Path, "/"))
	}))
	t.Cleanup(server.Close)
	for _, mode := range []string{"mixed", "cancel", "missing", "incomplete", "capture_failure"} {
		t.Run(mode, func(t *testing.T) {
			db := sqlitetest.OpenMigrated(t)
			store, err := debugcapture.New(db)
			if err != nil {
				t.Fatal(err)
			}
			manager := state.NewManager()
			if _, err := manager.Publish(state.CompileInput{}); err != nil {
				t.Fatal(err)
			}
			registry := state.NewCredentialRegistry()
			executor := &modelTraceCaptureExecutor{url: server.URL, mode: mode, ready: make(chan struct{})}
			sink := &modelTraceCaptureSink{}
			service := control.NewService(db, manager, registry, nil, nil, nil, encryptiontest.Service(t, "local-test-key-material-2026"), executor, nil, nil, nil, health.NewStatsStore(), health.NewMutationCoordinator(), nil, nil, sink, channel.NewRegistry())
			if mode == "capture_failure" {
				service.SetProbeCaptureFactory(failingModelTraceCaptureFactory{})
			} else {
				service.SetProbeCaptureFactory(newDebugCaptureFactory(store, nil))
			}
			var create control.GroupCreateRequest
			if err := json.Unmarshal([]byte(`{"name":"capture-fixture","channel_id":"openai","params":{},"models":[{"id":"gpt-4o","alias_enabled":false}],"credential":"test-only"}`), &create); err != nil {
				t.Fatal(err)
			}
			group, err := service.CreateGroup(t.Context(), create)
			if err != nil {
				t.Fatal(err)
			}
			response, err := service.RunModelTraceProbe(t.Context(), control.ModelTraceRequest{GroupID: group.GroupID, Model: "gpt-4o"})
			if err != nil {
				t.Fatal(err)
			}
			if len(response.Challenges) != 3 || executor.challenges != 3 || len(executor.calls) != 4 || len(sink.events) != 4 {
				t.Fatalf("challenge result/call counts = %d/%d, total call/event counts = %d/%d", len(response.Challenges), executor.challenges, len(executor.calls), len(sink.events))
			}
			seen := map[string]bool{}
			for _, call := range executor.calls {
				if seen[call.spec.RequestID] || call.spec.RequestID == "" || (mode != "capture_failure" && call.observerID != call.spec.AttemptID) {
					t.Errorf("invalid or duplicate request/attempt correlation for index %d", call.index)
				}
				seen[call.spec.RequestID] = true
				logCount := 0
				for _, event := range sink.events {
					if event.RequestID == call.spec.RequestID {
						logCount++
						if event.Operation != execution.OperationModelTrace || len(event.Attempts) != 1 || event.Attempts[0].Sequence != 1 {
							t.Errorf("index %d: unexpected request log correlation", call.index)
						}
					}
				}
				if logCount != 1 {
					t.Errorf("index %d: request log count=%d", call.index, logCount)
				}
				var challenge *control.ModelTraceChallengeResponse
				for i := range response.Challenges {
					if response.Challenges[i].LogID != nil && *response.Challenges[i].LogID == call.spec.RequestID {
						challenge = &response.Challenges[i]
					}
				}
				wantStatus := http.StatusOK
				if mode == "cancel" || (call.index == 2 && mode != "incomplete") {
					wantStatus = 0
				} else if call.index == 1 {
					wantStatus = http.StatusBadGateway
				}
				if call.index == modelTraceCaptureContextIndex {
					check := response.ContextCheck
					if challenge != nil || check.LogID == nil || *check.LogID != call.spec.RequestID || string(check.Status) != "unavailable" {
						t.Error("context check should link to its own capture and be unavailable without input usage")
					}
				} else {
					if challenge == nil || challenge.StatusCode == nil || *challenge.StatusCode != wantStatus {
						t.Errorf("index %d: capture changed or lost provider status", call.index)
					}
					if challenge != nil && wantStatus == http.StatusOK && string(challenge.Outcome) != "succeeded" {
						t.Errorf("index %d: capture changed successful provider outcome", call.index)
					}
				}
				records, err := store.QuerySessions(debugcapture.SessionQuery{RequestID: call.spec.RequestID, Limit: 2})
				if mode == "capture_failure" {
					if err != nil || len(records) != 0 {
						t.Errorf("capture failure produced unexpected records: %d, err=%v", len(records), err)
					}
					continue
				}
				if err != nil || len(records) != 1 {
					t.Errorf("index %d: query sessions count=%d err=%v", call.index, len(records), err)
					continue
				}
				record := records[0]
				if len(record.Attempts) != 1 {
					t.Errorf("index %d: attempts=%d", call.index, len(record.Attempts))
					continue
				}
				attempt := record.Attempts[0]
				if attempt.Metadata.Fields["logical_attempt_id"] != call.spec.AttemptID || attempt.Sequence != 1 || record.Operation != string(execution.OperationModelTrace) {
					t.Errorf("index %d: incorrect capture metadata", call.index)
				}
				missing := (mode == "missing" || mode == "incomplete") && call.index == 2
				if missing {
					if record.State != debugcapture.StateFailed || attempt.State != debugcapture.StateFailed {
						t.Errorf("missing observer should fail capture: %s/%s", record.State, attempt.State)
					}
				} else if record.State != debugcapture.StateCompleted || attempt.State != debugcapture.StateCompleted {
					t.Errorf("index %d: capture state=%s/%s", call.index, record.State, attempt.State)
				}
				wantTermination := "eof"
				if mode == "cancel" {
					wantTermination = "close"
				} else if missing {
					wantTermination = "no_observation_source"
				} else if call.index == 2 {
					wantTermination = "timeout"
				}
				found := false
				for _, event := range attempt.Metadata.Events {
					if event.Kind == "response_termination" && event.Outcome == wantTermination {
						found = true
						if missing && ((mode == "missing" && !strings.Contains(event.Error, "never started")) || (mode == "incomplete" && !strings.Contains(event.Error, "started and ended"))) {
							t.Errorf("index %d: missing source diagnostic detail", call.index)
						}
					}
				}
				if !found {
					t.Errorf("index %d: missing termination %s", call.index, wantTermination)
				}
				if missing {
					if mode == "incomplete" {
						assertModelTracePart(t, store, record.ID, attempt.ID, debugcapture.DirectionRequest, []byte("partial-request"))
					}
					part, err := store.ReadPart(record.ID, attempt.ID, debugcapture.PartBody, debugcapture.DirectionResponse)
					if !errors.Is(err, debugcapture.ErrNotFound) || len(part) != 0 {
						t.Errorf("index %d: invented missing response body, err=%v", call.index, err)
					}
					continue
				}
				assertModelTracePart(t, store, record.ID, attempt.ID, debugcapture.DirectionRequest, []byte(fmt.Sprintf("request-%d", call.index)))
				headers, err := store.ReadPart(record.ID, attempt.ID, debugcapture.PartHeaders, debugcapture.DirectionRequest)
				if err != nil || !bytes.Contains(headers, []byte("Content-Type: application/octet-stream\r\n")) {
					t.Errorf("index %d: observed request headers missing, err=%v", call.index, err)
				}
				if mode != "cancel" && call.index != 2 {
					assertModelTracePart(t, store, record.ID, attempt.ID, debugcapture.DirectionResponse, []byte(fmt.Sprintf("response-%d", call.index)))
					headers, err := store.ReadPart(record.ID, attempt.ID, debugcapture.PartHeaders, debugcapture.DirectionResponse)
					if err != nil || !bytes.Contains(headers, []byte(fmt.Sprintf("HTTP %d\r\n", wantStatus))) {
						t.Errorf("index %d: observed response status missing, err=%v", call.index, err)
					}
				} else {
					part, err := store.ReadPart(record.ID, attempt.ID, debugcapture.PartBody, debugcapture.DirectionResponse)
					if !errors.Is(err, debugcapture.ErrNotFound) || len(part) != 0 {
						t.Errorf("index %d: invented response body after timeout/cancel, err=%v", call.index, err)
					}
				}
				var data bytes.Buffer
				if err := store.ExportZIP(record.ID, &data); err != nil {
					t.Fatal(err)
				}
				zr, err := zip.NewReader(bytes.NewReader(data.Bytes()), int64(data.Len()))
				if err != nil {
					t.Fatal(err)
				}
				foundBody := false
				foundResponse := false
				for _, f := range zr.File {
					rc, err := f.Open()
					if err != nil {
						t.Fatal(err)
					}
					b, err := io.ReadAll(rc)
					_ = rc.Close()
					if err != nil {
						t.Fatal(err)
					}
					if f.Name == fmt.Sprintf("attempts/%s/parts/request.body", attempt.ID) && bytes.Equal(b, []byte(fmt.Sprintf("request-%d", call.index))) {
						foundBody = true
					}
					if f.Name == fmt.Sprintf("attempts/%s/parts/response.body", attempt.ID) && bytes.Equal(b, []byte(fmt.Sprintf("response-%d", call.index))) {
						foundResponse = true
					}
				}
				if !foundBody {
					t.Errorf("index %d: ZIP lacks exact observed request body", call.index)
				}
				if mode != "cancel" && call.index != 2 && !foundResponse {
					t.Errorf("index %d: ZIP lacks exact observed response body", call.index)
				}
			}
			for _, challenge := range response.Challenges {
				if challenge.LogID == nil || !seen[*challenge.LogID] {
					t.Error("challenge log ID does not link to dispatched request")
				}
			}
		})
	}
}

func assertModelTracePart(t *testing.T, store *debugcapture.Store, session, attempt string, direction debugcapture.Direction, want []byte) {
	t.Helper()
	got, err := store.ReadPart(session, attempt, debugcapture.PartBody, direction)
	if err != nil || !bytes.Equal(got, want) {
		t.Errorf("%s raw body differs, err=%v", direction, err)
	}
}
