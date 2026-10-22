package gateway

import (
	"testing"
	"time"

	"gpt-load/internal/affinity"
	"gpt-load/internal/protocol"
	"gpt-load/internal/telemetry"
)

func TestRequestRecorderAffinityDiagnostics(t *testing.T) {
	for _, preferred := range []uint{0, 21, 99} {
		t.Run(string(rune('a'+preferred)), func(t *testing.T) {
			now := time.Now()
			sink := &recordingRequestLogSink{}
			recorder := newRequestRecorder(sink, fixedRequestID, now, 1, protocol.OpenAICompletions, func() time.Time { return now })
			dispatch := &requestDispatch{affinity: requestAffinity{
				preferredCredentialID: preferred,
				boundTarget: affinity.Target{
					GroupID: 1, CredentialID: preferred, IdentityGeneration: 1,
					EntryID: "e000000000001",
				},
			}}
			recorder.observeDispatch(dispatch)
			first := requestLogSelection(1, 21, "first")
			first.EntryID = "e000000000001"
			recorder.observeAffinityHitComparison(first)
			recorder.appendAttempt(requestLogSelection(11, 21, "first"), UpstreamResult{}, telemetry.FailureCategoryOK, telemetry.ActionTerminate, "", "", now, now)
			recorder.appendAttempt(requestLogSelection(11, 22, "retry"), UpstreamResult{}, telemetry.FailureCategoryOK, telemetry.ActionTerminate, "", "", now, now)
			recorder.emit()
			event := sink.snapshot()[0]
			if event.ResolvedAffinityCredentialID != preferred || event.FirstAttemptCredentialID != 21 || !event.AffinityHitAssignmentReached || event.AffinityHit != (preferred != 0 && preferred == 21) {
				t.Fatalf("affinity diagnostics = %+v", event)
			}
		})
	}
}

func TestRequestRecorderAffinityDiagnosticsBeforeAttempt(t *testing.T) {
	now := time.Now()
	sink := &recordingRequestLogSink{}
	recorder := newRequestRecorder(sink, fixedRequestID, now, 1, protocol.OpenAICompletions, func() time.Time { return now })
	recorder.observeDispatch(&requestDispatch{affinity: requestAffinity{
		preferredCredentialID: 21,
		boundTarget: affinity.Target{
			GroupID: 1, CredentialID: 21, IdentityGeneration: 1, EntryID: "e000000000001",
		},
	}})
	recorder.emit()
	event := sink.snapshot()[0]
	if event.ResolvedAffinityCredentialID != 21 || event.FirstAttemptCredentialID != 0 || event.AffinityHitAssignmentReached {
		t.Fatalf("pre-attempt diagnostics = %+v", event)
	}
}
