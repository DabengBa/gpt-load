package requestlog

import (
	"context"
	"testing"
	"time"

	"gpt-load/internal/storage/models"
)

func TestUsageBreakdownAggregatesByScopeAndMasksAccessKeyDimensions(t *testing.T) {
	db := openRequestLogQueryDB(t)

	start := time.Date(2026, time.August, 8, 15, 0, 0, 0, time.UTC)
	row := func(id, model, channel string, group, credential, access uint, requests int64) models.UsageStat {
		value := usageStat(start, group, model, requests)
		value.ID = 0
		value.ChannelID = channel
		value.CredentialID = credential
		value.AccessKeyID = access
		value.DurationMsTotal = requests * 10
		value.DurationSampleCount = requests
		return value
	}
	createUsageStats(t, db,
		row("a", "shared", "channel-a", 7, 1, 41, 2),
		row("b", "shared", "channel-a", 7, 2, 42, 3),
		row("c", "shared", "channel-a", 8, 1, 41, 4),
		row("d", "shared", "channel-b", 7, 1, 41, 5),
		row("e", "other", "channel-a", 7, 1, 41, 1),
	)
	query := UsageQuery{
		FromMS: start.UnixMilli(), ToMS: start.Add(time.Hour).UnixMilli(),
		Granularity: UsageGranularityHour,
	}
	admin, err := newRequestLogTestService(db).QueryUsage(context.Background(), query)
	if err != nil {
		t.Fatal(err)
	}
	if admin.Breakdown.Scope != "admin" || len(admin.Breakdown.Rows) != 4 {
		t.Fatalf("admin breakdown = %#v", admin.Breakdown)
	}
	var mergedAdmin *UsageBreakdownRow
	for index := range admin.Breakdown.Rows {
		candidate := &admin.Breakdown.Rows[index]
		if candidate.Model == "shared" && candidate.GroupID != nil && *candidate.GroupID == 7 &&
			candidate.ChannelID != nil && *candidate.ChannelID == "channel-a" {
			mergedAdmin = candidate
		}
	}
	if mergedAdmin == nil || mergedAdmin.RequestCount != 5 || mergedAdmin.DurationMsTotal != 50 ||
		mergedAdmin.DurationSampleCount != 5 {
		t.Fatalf("merged admin row = %#v", mergedAdmin)
	}
	if admin.Breakdown.Total != admin.Summary {
		t.Fatalf("admin total = %#v, summary = %#v", admin.Breakdown.Total, admin.Summary)
	}

	accessKeyID := uint(41)
	scoped, err := newRequestLogTestService(db).QueryUsage(context.Background(), UsageQuery{
		FromMS: start.UnixMilli(), ToMS: start.Add(time.Hour).UnixMilli(),
		Granularity: UsageGranularityHour, AccessKeyID: &accessKeyID,
	})
	if err != nil {
		t.Fatal(err)
	}
	if scoped.Breakdown.Scope != "access_key" || len(scoped.Breakdown.Rows) != 2 {
		t.Fatalf("access-key breakdown = %#v", scoped.Breakdown)
	}
	for _, breakdownRow := range scoped.Breakdown.Rows {
		if breakdownRow.GroupID != nil || breakdownRow.ChannelID != nil {
			t.Fatalf("access-key row exposes admin dimensions = %#v", breakdownRow)
		}
	}
	if scoped.Breakdown.Rows[0].Model != "other" || scoped.Breakdown.Rows[1].Model != "shared" ||
		scoped.Breakdown.Rows[1].RequestCount != 11 || scoped.Breakdown.Rows[1].DurationSampleCount != 11 {
		t.Fatalf("access-key rows = %#v", scoped.Breakdown.Rows)
	}
	if scoped.Breakdown.Total != scoped.Summary {
		t.Fatalf("access-key total = %#v, summary = %#v", scoped.Breakdown.Total, scoped.Summary)
	}
}

func TestUsageBreakdownPaginatesRowsWithoutChangingTotal(t *testing.T) {
	db := openRequestLogQueryDB(t)
	start := time.Date(2026, time.August, 8, 15, 0, 0, 0, time.UTC)
	rows := make([]models.UsageStat, 0, 25)
	for index := 0; index < 25; index++ {
		model := string(rune('a' + index))
		row := usageStat(start, 7, model, 1)
		row.ID = 0
		row.ChannelID = "channel-a"
		row.CredentialID = uint(index + 1)
		rows = append(rows, row)
	}
	createUsageStats(t, db, rows...)

	report, err := newRequestLogTestService(db).QueryUsage(context.Background(), UsageQuery{
		FromMS: start.UnixMilli(), ToMS: start.Add(time.Hour).UnixMilli(),
		Granularity: UsageGranularityHour, BreakdownPage: 2, BreakdownPageSize: 20,
	})
	if err != nil {
		t.Fatal(err)
	}
	if report.Breakdown.Pagination.Page != 2 || report.Breakdown.Pagination.PageSize != 20 ||
		report.Breakdown.Pagination.TotalItems != 25 || report.Breakdown.Pagination.TotalPages != 2 {
		t.Fatalf("breakdown pagination = %#v", report.Breakdown.Pagination)
	}
	if len(report.Breakdown.Rows) != 5 || report.Breakdown.Rows[0].Model != "u" ||
		report.Breakdown.Rows[4].Model != "y" {
		t.Fatalf("breakdown page rows = %#v", report.Breakdown.Rows)
	}
	if report.Breakdown.Total != report.Summary {
		t.Fatalf("breakdown total = %#v, summary = %#v", report.Breakdown.Total, report.Summary)
	}
}
