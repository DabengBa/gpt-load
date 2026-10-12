package requestlog

import (
	"context"
	"gpt-load/internal/platform/redact"
	"gpt-load/internal/storage/models"
	"testing"
)

func TestAffinityDiagnosticsPersistenceRoundTrip(t *testing.T) {
	db, _ := openRequestLogFileDB(t)
	event := testEvent("11111111-1111-4111-8111-111111111125")
	event.ResolvedAffinityCredentialID = 41
	event.FirstAttemptCredentialID = 42
	event.AffinityHitAssignmentReached = true
	service := emitRequestLogEvents(t, db, event)
	page, err := service.List(context.Background(), ListQuery{Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(page.Items) != 1 {
		t.Fatalf("list count = %d, want 1", len(page.Items))
	}
	got := page.Items[0]
	if got.ResolvedAffinityCredentialID != 41 || got.FirstAttemptCredentialID != 42 || !got.AffinityHitAssignmentReached {
		t.Fatalf("persisted diagnostics = %+v", got)
	}
}

func TestAffinityDiagnosticsMapAndDecode(t *testing.T) {
	event := testEvent("diagnostics")
	event.ResolvedAffinityCredentialID = 41
	event.FirstAttemptCredentialID = 42
	event.AffinityHitAssignmentReached = true
	row := mustMapEvent(t, redact.New(), event)
	records, err := decodeRequestLogRows([]models.RequestLog{row})
	if err != nil {
		t.Fatal(err)
	}
	got := records[0]
	if got.ResolvedAffinityCredentialID != 41 || got.FirstAttemptCredentialID != 42 || !got.AffinityHitAssignmentReached {
		t.Fatalf("decoded diagnostics = %+v", got)
	}
}
