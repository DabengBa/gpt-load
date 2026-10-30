package requestlog

import (
	"fmt"
	"testing"
	"time"

	"gpt-load/internal/platform/redact"
	migrationfiles "gpt-load/internal/storage/migrations"
	"gpt-load/internal/storage/models"
)

func TestUsageBreakdownByClientModel(t *testing.T) {
	db, _ := openRequestLogFileDB(t)
	first := testEvent("11111111-1111-4111-8111-111111111111")
	second := testEvent("22222222-2222-4222-8222-222222222222")
	first.CompletedAt = time.Date(2026, time.September, 22, 8, 30, 0, 0, time.UTC)
	second.CompletedAt = first.CompletedAt
	first.ClientModel, second.ClientModel = "alias-one", "alias-two"
	retry := first.Attempts[0]
	retry.Sequence = 2
	retry.UpstreamModel = "retry-upstream"
	first.Attempts = append(first.Attempts, retry)

	service := emitRequestLogEvents(t, db, first, second)
	start := first.CompletedAt.Truncate(time.Hour)
	query := UsageQuery{FromMS: start.UnixMilli(), ToMS: start.Add(time.Hour).UnixMilli(), Granularity: UsageGranularityHour}
	query.BreakdownByClientModel = true
	report, err := service.QueryUsage(t.Context(), query)
	if err != nil {
		t.Fatal(err)
	}
	if report.Summary.RequestCount != 2 || len(report.Breakdown.Rows) != 3 {
		t.Fatalf("report = %#v, want two aliases and one retry upstream", report)
	}
	query.ClientModel = "alias-one"
	report, err = service.QueryUsage(t.Context(), query)
	if err != nil {
		t.Fatal(err)
	}
	if report.Summary.RequestCount != 1 || len(report.Breakdown.Rows) != 2 {
		t.Fatalf("filtered report = %#v, want alias-one only", report)
	}
	for _, row := range report.Breakdown.Rows {
		if row.ClientModel != "alias-one" || row.AttemptCount != 1 {
			t.Fatalf("filtered attempt row = %#v", row)
		}
	}
	query.ClientModel = ""
	query.BreakdownByClientModel = false
	report, err = service.QueryUsage(t.Context(), query)
	if err != nil || len(report.Breakdown.Rows) != 2 {
		t.Fatalf("legacy grouping = %#v, error %v", report.Breakdown, err)
	}
	if err := db.Create(&models.UsageStat{BucketStartMS: start.UnixMilli(), AccessKeyID: 42, Model: "historical", RequestCount: 1, SuccessCount: 1}).Error; err != nil {
		t.Fatal(err)
	}
	query.BreakdownByClientModel = true
	report, err = service.QueryUsage(t.Context(), query)
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range report.Breakdown.Rows {
		if row.Model == "historical" && row.ClientModel == "" {
			return
		}
	}
	t.Fatalf("historical unknown model missing: %#v", report.Breakdown.Rows)
}

func TestUsageClientModelLegacyPendingJournalUpgradeReplay(t *testing.T) {
	db := openRequestLogQueryDB(t)
	event := testEvent(aggregationRequestID(300))
	event.ClientModel = "new-alias"
	row := mustMapEvent(t, redact.New(), event)
	requests, err := buildUsageAggregationJournals([]models.RequestLog{row})
	if err != nil || len(requests) != 1 {
		t.Fatalf("request journal: %#v, %v", requests, err)
	}
	attempts, err := buildUsageAttemptAggregationJournals([]models.RequestLog{row})
	if err != nil || len(attempts) != 1 {
		t.Fatalf("attempt journal: %#v, %v", attempts, err)
	}
	for _, sql := range []string{
		"DROP INDEX idx_usage_stats_identity", "DROP INDEX idx_usage_attempt_stats_identity",
		"ALTER TABLE usage_aggregation_journal DROP COLUMN client_model",
		"ALTER TABLE usage_attempt_aggregation_journals DROP COLUMN client_model",
		"ALTER TABLE usage_stats DROP COLUMN client_model",
		"ALTER TABLE usage_attempt_stats DROP COLUMN client_model",
		"CREATE UNIQUE INDEX idx_usage_stats_identity ON usage_stats (bucket_start_ms, access_key_id, channel_id, group_id, credential_id, model)",
		"CREATE UNIQUE INDEX idx_usage_attempt_stats_identity ON usage_attempt_stats (bucket_start_ms, access_key_id, channel_id, group_id, credential_id, model)",
	} {
		if err := db.Exec(sql).Error; err != nil {
			t.Fatalf("construct legacy schema %q: %v", sql, err)
		}
	}
	if err := db.Create(&row).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Omit("ClientModel").Create(&requests[0]).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Omit("ClientModel").Create(&attempts[0]).Error; err != nil {
		t.Fatal(err)
	}
	if err := migrationfiles.Up0028(db); err != nil {
		t.Fatalf("upgrade: %v", err)
	}
	writer := &gormBatchWriter{db: db}
	for range 2 {
		if err := writer.WriteBatch(t.Context(), []models.RequestLog{row}); err != nil {
			t.Fatalf("replay: %v", err)
		}
	}
	var stat models.UsageStat
	var attempt models.UsageAttemptStat
	if err := db.Take(&stat).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Take(&attempt).Error; err != nil {
		t.Fatal(err)
	}
	if stat.ClientModel != "" || stat.RequestCount != 1 || attempt.ClientModel != "" || attempt.AttemptCount != 1 {
		t.Fatalf("replayed legacy stats: request=%+v attempt=%+v", stat, attempt)
	}
	var requestJournal models.UsageAggregationJournal
	var attemptJournal models.UsageAttemptAggregationJournal
	if err := db.Take(&requestJournal).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Take(&attemptJournal).Error; err != nil {
		t.Fatal(err)
	}
	if !requestJournal.Applied || !attemptJournal.Applied {
		t.Fatalf("journals not applied: %+v %+v", requestJournal, attemptJournal)
	}
}

func TestUsageClientModelBreakdownPagination(t *testing.T) {
	db := openRequestLogQueryDB(t)
	start := time.Date(2026, time.September, 22, 8, 0, 0, 0, time.UTC)
	for i := range 25 {
		alias := fmt.Sprintf("alias-%02d", i)
		stat := models.UsageStat{BucketStartMS: start.UnixMilli(), AccessKeyID: 42, GroupID: 7, ChannelID: "openai", CredentialID: 8, Model: "upstream", ClientModel: alias, RequestCount: 1, SuccessCount: 1}
		attempt := models.UsageAttemptStat{BucketStartMS: start.UnixMilli(), AccessKeyID: 42, GroupID: 7, ChannelID: "openai", CredentialID: 8, Model: "upstream", ClientModel: alias, AttemptCount: 1}
		if err := db.Create(&stat).Error; err != nil {
			t.Fatal(err)
		}
		if err := db.Create(&attempt).Error; err != nil {
			t.Fatal(err)
		}
	}
	service := newRequestLogTestService(db)
	for page := 1; page <= 2; page++ {
		report, err := service.QueryUsage(t.Context(), UsageQuery{FromMS: start.UnixMilli(), ToMS: start.Add(time.Hour).UnixMilli(), Granularity: UsageGranularityHour, BreakdownByClientModel: true, BreakdownPage: page, BreakdownPageSize: 20})
		if err != nil {
			t.Fatal(err)
		}
		want := 20
		if page == 2 {
			want = 5
		}
		if len(report.Breakdown.Rows) != want || report.Breakdown.Pagination.TotalItems != 25 || report.Breakdown.Pagination.TotalPages != 2 || report.Breakdown.AttemptTotal.AttemptCount != 25 || report.Summary.RequestCount != 25 {
			t.Fatalf("page %d: %+v", page, report.Breakdown)
		}
		for i, row := range report.Breakdown.Rows {
			if row.ClientModel != fmt.Sprintf("alias-%02d", (page-1)*20+i) || row.RequestCount != 1 || row.AttemptCount != 1 {
				t.Fatalf("page %d row %d: %+v", page, i, row)
			}
		}
	}
}
