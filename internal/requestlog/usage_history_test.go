package requestlog

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"gpt-load/internal/storage/models"
)

// Decode the exported report shape so the first red run proves the missing
// behavior instead of merely failing to compile against a new type.
func TestUsageBreakdownRequestHistory(t *testing.T) {
	db := openRequestLogQueryDB(t)
	start := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	fixtures := []models.UsageStat{
		{BucketStartMS: start.UnixMilli(), AccessKeyID: 41, GroupID: 7, ChannelID: "openai", CredentialID: 1, Model: "shared", ClientModel: "", RequestCount: 2, SuccessCount: 1, FailureCount: 1},
		{BucketStartMS: start.Add(time.Hour).UnixMilli(), AccessKeyID: 41, GroupID: 7, ChannelID: "openai", CredentialID: 1, Model: "shared", ClientModel: "", RequestCount: 1, SuccessCount: 1},
		{BucketStartMS: start.UnixMilli(), AccessKeyID: 41, GroupID: 8, ChannelID: "anthropic", CredentialID: 2, Model: "shared", ClientModel: "", RequestCount: 3, SuccessCount: 3},
		{BucketStartMS: start.UnixMilli(), AccessKeyID: 42, GroupID: 7, ChannelID: "openai", CredentialID: 1, Model: "shared", ClientModel: "", RequestCount: 9, SuccessCount: 9},
		{BucketStartMS: start.UnixMilli(), AccessKeyID: 41, GroupID: 7, ChannelID: "openai", CredentialID: 1, Model: "shared", ClientModel: "alias", RequestCount: 4, SuccessCount: 4},
	}
	createUsageStats(t, db, fixtures...)
	key := uint(41)
	query := UsageQuery{FromMS: start.UnixMilli(), ToMS: start.Add(6 * time.Hour).UnixMilli(), Granularity: UsageGranularityHour, BucketWidthMS: int64(3 * time.Hour / time.Millisecond), AccessKeyID: &key, BreakdownByClientModel: true, IncludeBreakdownHistory: true}
	report, err := newRequestLogTestService(db).QueryUsage(context.Background(), query)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Breakdown.Rows) != 2 {
		t.Fatalf("rows = %#v", report.Breakdown.Rows)
	}
	var decoded struct {
		History []struct {
			BucketStartMS int64
			RequestCount  int64
			SuccessCount  int64
		}
	}
	for _, row := range report.Breakdown.Rows {
		if row.ClientModel != "" {
			continue
		}
		encoded, err := json.Marshal(row)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(encoded, &decoded); err != nil {
			t.Fatal(err)
		}
		if len(decoded.History) != 1 || decoded.History[0].BucketStartMS != start.UnixMilli() || decoded.History[0].RequestCount != 6 || decoded.History[0].SuccessCount != 5 {
			t.Fatalf("history = %#v; want one 3h bucket with 5/6 scoped requests", decoded.History)
		}
		return
	}
	t.Fatal("missing unknown client model row")
}

func TestUsageBreakdownHistoryPageFiltersAndAttemptOnly(t *testing.T) {
	db := openRequestLogQueryDB(t)
	start := time.Date(2026, 8, 8, 15, 0, 0, 0, time.UTC)
	var stats []models.UsageStat
	for i := 0; i < 21; i++ {
		stats = append(stats, models.UsageStat{BucketStartMS: start.UnixMilli(), AccessKeyID: 1, GroupID: 7, ChannelID: "openai", CredentialID: 1, Model: fmt.Sprintf("model-%02d", i), RequestCount: 1, SuccessCount: 1})
	}
	stats = append(stats,
		models.UsageStat{BucketStartMS: start.Add(2 * time.Hour).UnixMilli(), AccessKeyID: 1, GroupID: 7, ChannelID: "openai", CredentialID: 1, Model: "model-20", RequestCount: 2, FailureCount: 2},
		models.UsageStat{BucketStartMS: start.UnixMilli(), AccessKeyID: 1, GroupID: 8, ChannelID: "other", CredentialID: 2, Model: "model-20", RequestCount: 7, SuccessCount: 7},
	)
	createUsageStats(t, db, stats...)
	if err := db.Create(&models.UsageAttemptStat{BucketStartMS: start.UnixMilli(), AccessKeyID: 1, GroupID: 7, ChannelID: "openai", CredentialID: 1, Model: "zz-attempt", AttemptCount: 1, FailureCount: 1}).Error; err != nil {
		t.Fatal(err)
	}
	credential := uint(1)
	query := UsageQuery{FromMS: start.UnixMilli(), ToMS: start.Add(3 * time.Hour).UnixMilli(), Granularity: UsageGranularityHour, BreakdownByClientModel: true, IncludeBreakdownHistory: true, CredentialID: &credential, BreakdownPage: 2, BreakdownPageSize: 20, BreakdownSort: UsageBreakdownSortModel, BreakdownSortDirection: UsageBreakdownSortAscending}
	report, err := newRequestLogTestService(db).QueryUsage(context.Background(), query)
	if err != nil {
		t.Fatal(err)
	}
	rows := report.Breakdown.Rows
	if len(rows) != 2 || rows[0].Model != "model-20" || len(rows[0].History) != 2 || rows[0].History[0].RequestCount != 1 || rows[0].History[1].RequestCount != 2 || rows[0].History[1].BucketStartMS != start.Add(2*time.Hour).UnixMilli() || rows[1].Model != "zz-attempt" || len(rows[1].History) != 0 {
		t.Fatalf("page histories = %#v", rows)
	}
	query.BreakdownByClientModel = false
	if _, err := newRequestLogTestService(db).QueryUsage(context.Background(), query); err != nil {
		t.Fatal(err)
	}
}

func TestUsageBreakdownHistoryAdminIdentityAndFilters(t *testing.T) {
	db := openRequestLogQueryDB(t)
	start := time.Date(2026, 8, 8, 15, 0, 0, 0, time.UTC)
	createUsageStats(t, db,
		models.UsageStat{BucketStartMS: start.UnixMilli(), AccessKeyID: 1, GroupID: 7, ChannelID: "openai", CredentialID: 1, Model: "shared", ClientModel: "", RequestCount: 2, SuccessCount: 2},
		models.UsageStat{BucketStartMS: start.UnixMilli(), AccessKeyID: 1, GroupID: 7, ChannelID: "openai", CredentialID: 1, Model: "shared", ClientModel: "alias", RequestCount: 3, SuccessCount: 3},
		models.UsageStat{BucketStartMS: start.UnixMilli(), AccessKeyID: 1, GroupID: 8, ChannelID: "anthropic", CredentialID: 2, Model: "shared", ClientModel: "", RequestCount: 5, SuccessCount: 5},
	)
	query := UsageQuery{FromMS: start.UnixMilli(), ToMS: start.Add(time.Hour).UnixMilli(), Granularity: UsageGranularityHour, BreakdownByClientModel: true, IncludeBreakdownHistory: true}
	service := newRequestLogTestService(db)
	report, err := service.QueryUsage(context.Background(), query)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Breakdown.Rows) != 3 {
		t.Fatalf("rows = %#v", report.Breakdown.Rows)
	}
	for _, row := range report.Breakdown.Rows {
		if len(row.History) != 1 || row.History[0].RequestCount != row.RequestCount {
			t.Fatalf("identity history = %#v", row)
		}
	}
	query.GroupID = new(uint)
	*query.GroupID = 7
	query.ChannelID = "openai"
	query.ClientModel = "alias"
	filtered, err := service.QueryUsage(context.Background(), query)
	if err != nil {
		t.Fatal(err)
	}
	if len(filtered.Breakdown.Rows) != 1 || filtered.Breakdown.Rows[0].History[0].RequestCount != 3 {
		t.Fatalf("filtered history = %#v", filtered.Breakdown.Rows)
	}
}
