package requestlog

import (
	"context"
	"math"
	"testing"
	"time"

	"gpt-load/internal/storage/models"
	"gpt-load/internal/telemetry"
)

func TestUsageLatencyCheckedAddsRejectOverflowWithoutMutation(t *testing.T) {
	for _, test := range []struct {
		name  string
		field string
	}{
		{name: "duration total", field: "duration_ms_total"},
		{name: "duration samples", field: "duration_sample_count"},
	} {
		t.Run(test.name, func(t *testing.T) {
			value := int64(math.MaxInt64)
			if err := checkedInt64Add(&value, 1, test.field); err == nil {
				t.Fatal("checkedInt64Add() error = nil, want overflow rejection")
			}
			if value != math.MaxInt64 {
				t.Fatalf("overflow mutated %s = %d", test.field, value)
			}
		})
	}
}
func TestUsageLatencyAndBreakdownContract(t *testing.T) {
	db := openRequestLogQueryDB(t)
	row := aggregationRow(
		aggregationRequestID(100),
		time.Date(2026, time.August, 8, 14, 0, 0, 0, time.UTC),
		7,
		"latency-model",
	)
	row.DurationMs = 0
	row.ChannelID = "channel-a"
	row.CredentialID = 11

	if err := (&gormBatchWriter{db: db}).WriteBatch(context.Background(), []models.RequestLog{row}); err != nil {
		t.Fatalf("WriteBatch() error = %v", err)
	}

	var journal models.UsageAggregationJournal
	if err := db.First(&journal).Error; err != nil {
		t.Fatal(err)
	}
	if journal.DurationMsTotal != 0 || journal.DurationSampleCount != 1 {
		t.Fatalf("journal latency = (%d, %d), want (0, 1)", journal.DurationMsTotal, journal.DurationSampleCount)
	}

	var stat models.UsageStat
	if err := db.First(&stat).Error; err != nil {
		t.Fatal(err)
	}
	if stat.DurationMsTotal != 0 || stat.DurationSampleCount != 1 {
		t.Fatalf("stat latency = (%d, %d), want (0, 1)", stat.DurationMsTotal, stat.DurationSampleCount)
	}

	report, err := newRequestLogTestService(db).QueryUsage(context.Background(), UsageQuery{
		FromMS:      row.CompletedAtMS - 3_600_000,
		ToMS:        row.CompletedAtMS + 3_600_000,
		Granularity: UsageGranularityHour,
	})
	if err != nil {
		t.Fatal(err)
	}
	if report.Breakdown.Scope != "admin" || len(report.Breakdown.Rows) != 1 ||
		report.Breakdown.Rows[0].GroupID == nil || *report.Breakdown.Rows[0].GroupID != 7 ||
		report.Breakdown.Rows[0].ChannelID == nil || *report.Breakdown.Rows[0].ChannelID != "channel-a" {
		t.Fatalf("admin breakdown = %#v", report.Breakdown)
	}
	if report.Breakdown.Total.DurationSampleCount != report.Summary.DurationSampleCount {
		t.Fatalf("breakdown total = %#v, summary = %#v", report.Breakdown.Total, report.Summary)
	}
}

func TestUsageLatencyCountsFailedRequests(t *testing.T) {
	db := openRequestLogQueryDB(t)
	row := aggregationRow(
		aggregationRequestID(102),
		time.Date(2026, time.August, 8, 17, 0, 0, 0, time.UTC),
		7,
		"failed-latency-model",
	)
	row.Status = string(telemetry.RequestStatusError)
	row.StatusCode = 503
	row.DurationMs = 29
	if err := (&gormBatchWriter{db: db}).WriteBatch(context.Background(), []models.RequestLog{row}); err != nil {
		t.Fatalf("WriteBatch() error = %v", err)
	}
	var stat models.UsageStat
	if err := db.First(&stat).Error; err != nil {
		t.Fatal(err)
	}
	if stat.RequestCount != 1 || stat.FailureCount != 1 || stat.DurationMsTotal != 29 || stat.DurationSampleCount != 1 {
		t.Fatalf("failed request latency aggregate = %+v", stat)
	}
}

func TestUsageLatencyIsIdempotentAcrossRequestLogReplay(t *testing.T) {
	db := openRequestLogQueryDB(t)
	row := aggregationRow(
		aggregationRequestID(101),
		time.Date(2026, time.August, 8, 16, 0, 0, 0, time.UTC),
		7,
		"replay-latency-model",
	)
	row.DurationMs = 37
	writer := &gormBatchWriter{db: db}
	for range 2 {
		if err := writer.WriteBatch(context.Background(), []models.RequestLog{row}); err != nil {
			t.Fatalf("WriteBatch() error = %v", err)
		}
	}
	var stat models.UsageStat
	if err := db.First(&stat).Error; err != nil {
		t.Fatal(err)
	}
	if stat.RequestCount != 1 || stat.DurationMsTotal != 37 || stat.DurationSampleCount != 1 {
		t.Fatalf("replayed stat = %#v, want one 37ms sample", stat)
	}
	var journal models.UsageAggregationJournal
	if err := db.First(&journal).Error; err != nil {
		t.Fatal(err)
	}
	if !journal.Applied || journal.DurationMsTotal != 37 || journal.DurationSampleCount != 1 {
		t.Fatalf("replayed journal = %#v", journal)
	}
}
