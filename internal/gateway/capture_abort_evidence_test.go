package gateway

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"

	"gpt-load/internal/execution"
)

// A fired first-token abort makes the real Bifrost transport return before it
// can emit any callback. The production capture observer must still finalize the
// attempt and record explicit missing-evidence, instead of waiting forever on a
// completion that the transport can no longer produce.
func TestCaptureObserverFinalizesFiredAttemptAbort(t *testing.T) {
	attempt := &captureTestAttempt{}
	observer := newCaptureHTTPObserver(attempt, "attempt-1")
	capture := &dataPlaneCapture{}

	bifrostContext := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	bifrostContext.SetValue(execution.HTTPObserverContextKey, execution.HTTPObserver(observer))
	bifrostContext.SetValue(execution.HTTPAttemptIDContextKey, "attempt-1")
	abort := utils.NewAttemptAbort(time.Minute)
	abort.Expire()
	bifrostContext.SetValue(schemas.BifrostContextKeyStreamAttemptAbort, abort)

	request, err := http.NewRequestWithContext(
		bifrostContext,
		http.MethodPost,
		"http://example.invalid/v1/chat",
		bytes.NewReader([]byte(`{}`)),
	)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	observer.AwaitCompletion()

	capture.beginForwardCapture()
	response, abortErr := utils.DoHTTPRequest(&http.Client{}, request)
	if response != nil {
		_ = response.Body.Close()
	}
	if !errors.Is(abortErr, utils.ErrStreamFirstTokenTimeout) {
		t.Fatalf("DoHTTPRequest error = %v, want the first-token timeout", abortErr)
	}
	capture.finishForward(attempt, observer, UpstreamResult{
		DispatchState:  execution.DispatchMaybeSent,
		RequestWritten: true,
		Err:            abortErr,
	})
	if !waitForCapture(t, func() bool {
		attempt.mu.Lock()
		defer attempt.mu.Unlock()
		return attempt.completed || attempt.failed != nil
	}) {
		capture.mu.Lock()
		pending := capture.pending
		capture.mu.Unlock()
		t.Fatalf("capture attempt never finalized after a fired attempt abort; pending=%d", pending)
	}
	if !waitForCapture(t, func() bool {
		capture.mu.Lock()
		defer capture.mu.Unlock()
		return capture.pending == 0
	}) {
		t.Fatal("capture finalization did not release the pending forward barrier")
	}
	attempt.mu.Lock()
	completed := attempt.completed
	failed := attempt.failed
	terminations := append([]string(nil), attempt.terminations...)
	attempt.mu.Unlock()
	if completed || failed == nil {
		t.Fatalf("attempt terminal completed=%t failed=%v, want retained abort evidence", completed, failed)
	}
	// The fired abort returns before the transport runs, so the fork completes
	// the source as a no-response attempt; the capture must carry that explicit
	// termination instead of looking like an observed response.
	if len(terminations) != 1 || !strings.HasPrefix(terminations[0], "no_response:") {
		t.Fatalf("attempt terminations = %v, want a single no_response evidence event", terminations)
	}
	if attempt.responseBodyValue() != nil || len(attempt.response) != 0 {
		t.Fatalf("attempt recorded transport bytes that were never observed: %q", attempt.response)
	}
}
