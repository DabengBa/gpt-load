package requestlog

import (
	"context"
	"testing"
	"time"

	"gpt-load/internal/storage/models"
	"gpt-load/internal/telemetry"
)

func TestQueryGroupUsageUsesRollingHalfOpenWindow(t *testing.T) {
	db := openRequestLogQueryDB(t)
	service := newRequestLogTestService(db)
	now := time.Date(2026, time.July, 24, 10, 30, 0, 0, time.UTC)
	rows := []models.RequestLog{
		aggregationRow("00000000-0000-4000-8000-000000000001", now.Add(-24*time.Hour), 1, "old"),
		aggregationRow("00000000-0000-4000-8000-000000000002", now.Add(-23*time.Hour), 1, "inside-success"),
		aggregationRow("00000000-0000-4000-8000-000000000003", now.Add(-time.Hour), 1, "inside-error"),
		aggregationRow("00000000-0000-4000-8000-000000000004", now.Add(-time.Hour), 2, "other-group"),
		aggregationRow("00000000-0000-4000-8000-000000000005", now, 1, "future"),
	}
	rows[2].Status = string(telemetry.RequestStatusError)
	for _, row := range rows {
		if err := db.Create(&row).Error; err != nil {
			t.Fatalf("create request log %s: %v", row.ID, err)
		}
	}

	got, err := service.QueryGroupUsage(context.Background(), GroupUsageQuery{
		FromMS: now.Add(-24 * time.Hour).UnixMilli(),
		ToMS:   now.UnixMilli(),
	})
	if err != nil {
		t.Fatalf("QueryGroupUsage() error = %v", err)
	}
	if got[1].RequestCount != 3 || got[1].SuccessCount != 2 {
		t.Fatalf("group 1 usage = %#v, want 3 requests and 2 successes", got[1])
	}
	if got[2].RequestCount != 1 || got[2].SuccessCount != 1 {
		t.Fatalf("group 2 usage = %#v, want 1 request and 1 success", got[2])
	}
	if _, exists := got[0]; exists {
		t.Fatalf("unattributed group usage = %#v, want omitted", got[0])
	}
}
