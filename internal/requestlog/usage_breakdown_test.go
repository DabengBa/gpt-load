package requestlog

import (
	"context"
	"fmt"
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
		Granularity: UsageGranularityHour, BreakdownSort: UsageBreakdownSortModel,
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
		BreakdownSort: UsageBreakdownSortModel,
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

func TestUsageBreakdownSortsGloballyAcrossPages(t *testing.T) {
	db := openRequestLogQueryDB(t)
	start := time.Date(2026, time.August, 8, 15, 0, 0, 0, time.UTC)
	rows := make([]models.UsageStat, 0, 21)
	for index := 1; index <= 21; index++ {
		model := fmt.Sprintf("model-%02d", index)
		row := usageStat(start, 7, model, 1)
		row.ID = 0
		row.ChannelID = "channel-a"
		row.CredentialID = uint(index)
		row.EstimatedCostNanoUSD = int64(index)
		rows = append(rows, row)
	}
	createUsageStats(t, db, rows...)

	service := newRequestLogTestService(db)
	query := UsageQuery{
		FromMS: start.UnixMilli(), ToMS: start.Add(time.Hour).UnixMilli(),
		Granularity: UsageGranularityHour, BreakdownPageSize: 20,
		BreakdownSort:          UsageBreakdownSortEstimatedCost,
		BreakdownSortDirection: UsageBreakdownSortDescending,
	}
	firstPage, err := service.QueryUsage(context.Background(), query)
	if err != nil {
		t.Fatal(err)
	}
	query.BreakdownPage = 2
	secondPage, err := service.QueryUsage(context.Background(), query)
	if err != nil {
		t.Fatal(err)
	}
	if firstPage.Breakdown.Rows[0].Model != "model-21" ||
		firstPage.Breakdown.Rows[len(firstPage.Breakdown.Rows)-1].Model != "model-02" ||
		len(secondPage.Breakdown.Rows) != 1 || secondPage.Breakdown.Rows[0].Model != "model-01" {
		t.Fatalf("sorted pages = %#v / %#v", firstPage.Breakdown.Rows, secondPage.Breakdown.Rows)
	}
}

func TestUsageBreakdownSortsByDisplayedIdentity(t *testing.T) {
	db := openRequestLogQueryDB(t)
	start := time.Date(2026, time.August, 8, 15, 0, 0, 0, time.UTC)
	for _, group := range []models.Group{
		{
			ID: 1, Name: "Zulu Group", ChannelID: "openai",
			Params: models.JSON(`{}`), Models: models.JSON(`[]`), Enabled: true,
		},
		{
			ID: 2, Name: "Alpha Group", ChannelID: "anthropic",
			Params: models.JSON(`{}`), Models: models.JSON(`[]`), Enabled: true,
		},
	} {
		if err := db.Create(&group).Error; err != nil {
			t.Fatalf("create group %d: %v", group.ID, err)
		}
	}
	rows := []models.UsageStat{
		usageStat(start, 1, "zulu-model", 1),
		usageStat(start, 2, "alpha-model", 1),
		usageStat(start, 99, "unknown-model", 1),
	}
	for index := range rows {
		rows[index].ID = 0
		rows[index].CredentialID = uint(index + 1)
	}
	rows[0].ChannelID, rows[1].ChannelID, rows[2].ChannelID = "openai", "anthropic", "legacy"
	createUsageStatsWithoutGroups(t, db, rows...)

	for _, test := range []struct {
		name      string
		sort      UsageBreakdownSort
		wantFirst string
	}{
		{name: "group name", sort: UsageBreakdownSortGroup, wantFirst: "unknown-model"},
		{name: "channel name", sort: UsageBreakdownSortChannel, wantFirst: "unknown-model"},
	} {
		t.Run(test.name, func(t *testing.T) {
			report, err := newRequestLogTestService(db).QueryUsage(context.Background(), UsageQuery{
				FromMS: start.UnixMilli(), ToMS: start.Add(time.Hour).UnixMilli(),
				Granularity: UsageGranularityHour, BreakdownPageSize: 20,
				BreakdownSort: test.sort, BreakdownSortDirection: UsageBreakdownSortAscending,
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(report.Breakdown.Rows) != 3 || report.Breakdown.Rows[0].Model != test.wantFirst {
				t.Fatalf("sorted rows = %#v, want %q first", report.Breakdown.Rows, test.wantFirst)
			}
		})
	}
}

func TestUsageBreakdownSortsByAggregateValues(t *testing.T) {
	db := openRequestLogQueryDB(t)
	start := time.Date(2026, time.August, 8, 15, 0, 0, 0, time.UTC)
	row := func(groupID uint, model string, requests, successes, durationTotal int64) models.UsageStat {
		value := usageStat(start, groupID, model, requests)
		value.ID = 0
		value.ChannelID = fmt.Sprintf("channel-%s", model)
		value.CredentialID = groupID
		value.SuccessCount = successes
		value.FailureCount = requests - successes
		value.DurationMsTotal = durationTotal
		value.DurationSampleCount = requests
		value.UncachedInputTokens = int64(groupID) * 10
		value.CacheReadTokens = int64(groupID) * 20
		value.CacheWrite5MTokens = int64(groupID) * 30
		value.CacheWrite1HTokens = int64(groupID) * 40
		value.CacheWriteUnknownTokens = int64(groupID) * 50
		value.OutputTokens = int64(groupID) * 60
		value.EstimatedCostNanoUSD = int64(groupID) * 100
		return value
	}
	createUsageStats(t, db,
		row(1, "alpha", 10, 5, 100),
		row(2, "bravo", 20, 14, 400),
		row(3, "charlie", 30, 27, 900),
	)

	tests := []struct {
		name      string
		sort      UsageBreakdownSort
		direction UsageBreakdownSortDirection
		first     string
		last      string
	}{
		{"model ascending", UsageBreakdownSortModel, UsageBreakdownSortAscending, "alpha", "charlie"},
		{"group ascending", UsageBreakdownSortGroup, UsageBreakdownSortAscending, "alpha", "charlie"},
		{"channel ascending", UsageBreakdownSortChannel, UsageBreakdownSortAscending, "alpha", "charlie"},
		{"requests descending", UsageBreakdownSortRequestCount, UsageBreakdownSortDescending, "charlie", "alpha"},
		{"success descending", UsageBreakdownSortSuccessCount, UsageBreakdownSortDescending, "charlie", "alpha"},
		{"failure ascending", UsageBreakdownSortFailureCount, UsageBreakdownSortAscending, "charlie", "bravo"},
		{"success rate ascending", UsageBreakdownSortSuccessRate, UsageBreakdownSortAscending, "alpha", "charlie"},
		{"latency ascending", UsageBreakdownSortAverageLatency, UsageBreakdownSortAscending, "alpha", "charlie"},
		{"uncached input ascending", UsageBreakdownSortUncachedInputTokens, UsageBreakdownSortAscending, "alpha", "charlie"},
		{"cache read ascending", UsageBreakdownSortCacheReadTokens, UsageBreakdownSortAscending, "alpha", "charlie"},
		{"cache write 5m ascending", UsageBreakdownSortCacheWrite5MTokens, UsageBreakdownSortAscending, "alpha", "charlie"},
		{"cache write 1h ascending", UsageBreakdownSortCacheWrite1HTokens, UsageBreakdownSortAscending, "alpha", "charlie"},
		{"cache write unknown ascending", UsageBreakdownSortCacheWriteUnknown, UsageBreakdownSortAscending, "alpha", "charlie"},
		{"output ascending", UsageBreakdownSortOutputTokens, UsageBreakdownSortAscending, "alpha", "charlie"},
		{"total tokens ascending", UsageBreakdownSortTotalTokens, UsageBreakdownSortAscending, "alpha", "charlie"},
		{"cost descending", UsageBreakdownSortEstimatedCost, UsageBreakdownSortDescending, "charlie", "alpha"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			report, err := newRequestLogTestService(db).QueryUsage(context.Background(), UsageQuery{
				FromMS: start.UnixMilli(), ToMS: start.Add(time.Hour).UnixMilli(),
				Granularity: UsageGranularityHour, BreakdownPageSize: 100,
				BreakdownSort: test.sort, BreakdownSortDirection: test.direction,
			})
			if err != nil {
				t.Fatal(err)
			}
			rows := report.Breakdown.Rows
			if len(rows) != 3 || rows[0].Model != test.first || rows[len(rows)-1].Model != test.last {
				t.Fatalf("sorted rows = %#v, want first/last %q/%q", rows, test.first, test.last)
			}
		})
	}
}

func TestUsageBreakdownUsesStableIdentityTieBreakers(t *testing.T) {
	db := openRequestLogQueryDB(t)
	start := time.Date(2026, time.August, 8, 15, 0, 0, 0, time.UTC)
	rows := make([]models.UsageStat, 0, 21)
	for index := 0; index < 21; index++ {
		groupID := uint(index/2 + 1)
		channelID := "channel-a"
		if index%2 == 0 {
			channelID = "channel-b"
		}
		if index == 20 {
			groupID, channelID = 11, "channel-a"
		}
		row := usageStat(start, groupID, "same", 1)
		row.ID = 0
		row.ChannelID = channelID
		row.CredentialID = uint(index + 1)
		row.EstimatedCostNanoUSD = 100
		rows = append(rows, row)
	}
	createUsageStats(t, db, rows...)

	service := newRequestLogTestService(db)
	query := UsageQuery{
		FromMS: start.UnixMilli(), ToMS: start.Add(time.Hour).UnixMilli(),
		Granularity: UsageGranularityHour, BreakdownPageSize: 20,
		BreakdownSort:          UsageBreakdownSortEstimatedCost,
		BreakdownSortDirection: UsageBreakdownSortDescending,
	}
	firstPage, err := service.QueryUsage(context.Background(), query)
	if err != nil {
		t.Fatal(err)
	}
	query.BreakdownPage = 2
	secondPage, err := service.QueryUsage(context.Background(), query)
	if err != nil {
		t.Fatal(err)
	}
	if len(firstPage.Breakdown.Rows) != 20 || len(secondPage.Breakdown.Rows) != 1 {
		t.Fatalf("tie-breaker page lengths = %d/%d", len(firstPage.Breakdown.Rows), len(secondPage.Breakdown.Rows))
	}
	first, second, twentieth, last := firstPage.Breakdown.Rows[0], firstPage.Breakdown.Rows[1], firstPage.Breakdown.Rows[19], secondPage.Breakdown.Rows[0]
	if first.Model != "same" || second.Model != "same" || twentieth.Model != "same" || last.Model != "same" ||
		first.GroupID == nil || second.GroupID == nil || twentieth.GroupID == nil || last.GroupID == nil ||
		*first.GroupID != 1 || *second.GroupID != 1 || *twentieth.GroupID != 10 || *last.GroupID != 11 ||
		first.ChannelID == nil || second.ChannelID == nil || twentieth.ChannelID == nil || last.ChannelID == nil ||
		*first.ChannelID != "channel-a" || *second.ChannelID != "channel-b" ||
		*twentieth.ChannelID != "channel-b" || *last.ChannelID != "channel-a" {
		t.Fatalf("tie-breaker rows = %#v / %#v", firstPage.Breakdown.Rows, secondPage.Breakdown.Rows)
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
		BreakdownSort: UsageBreakdownSortModel,
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
