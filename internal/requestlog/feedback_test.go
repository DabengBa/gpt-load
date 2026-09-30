package requestlog

import (
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/platform/redact"
	"gpt-load/internal/storage/models"
	"gpt-load/internal/telemetry"
)

func TestFeedbackAttemptsPersistFinalAttemptAndAggregateIdempotently(t *testing.T) {
	db := openRequestLogQueryDB(t)
	completedAt := time.Date(2026, time.July, 24, 12, 0, 0, 0, time.UTC)
	firstResponseA := int64(31_000)
	rateA := 8.0
	firstResponseB := int64(400)
	rateB := 25.0

	retried := testEvent(aggregationRequestID(121))
	retried.CompletedAt = completedAt
	retried.UpstreamModel = "provider-b"
	retried.UpstreamReportedModel = "provider-b"
	retried.Attempts = []telemetry.Attempt{
		{
			Sequence: 1, GroupID: 7, GroupName: "first", ChannelID: channel.OpenAI,
			CredentialID: 11, Operation: execution.OperationChatCompletion, RouteMode: channel.RouteNative,
			UpstreamModel: "provider-a", DispatchState: execution.DispatchMaybeSent,
			StatusCode: 502, DurationMs: 30_000, FailureCategory: telemetry.FailureCategoryUpstreamHost,
			Action: telemetry.ActionRetry, WillRetry: true,
			Feedback: health.Feedback{
				Status: health.FeedbackStatusFaulty, Reason: "upstream_failure",
				FirstResponseMs: &firstResponseA, TokensPerSecond: &rateA,
			},
		},
		{
			Sequence: 2, GroupID: 7, GroupName: "final", ChannelID: channel.OpenAI,
			CredentialID: 12, Operation: execution.OperationChatCompletion, RouteMode: channel.RouteNative,
			UpstreamModel: "provider-b", DispatchState: execution.DispatchMaybeSent,
			StatusCode: 200, DurationMs: 500, FailureCategory: telemetry.FailureCategoryOK,
			Action: telemetry.ActionTerminate,
			Feedback: health.Feedback{
				Status:          health.FeedbackStatusNormal,
				FirstResponseMs: &firstResponseB, TokensPerSecond: &rateB,
			},
		},
	}
	retried.Usage.GroupID = 7
	retried.Usage.ChannelID = channel.OpenAI
	retried.Usage.CredentialID = 12
	retried.Usage.AttemptSequence = 2
	retried.Usage.Pricing.UpstreamModel = "provider-b"

	faultySuccess := feedbackTestEvent(aggregationRequestID(122), completedAt, "provider-c", 8, 13,
		health.Feedback{Status: health.FeedbackStatusFaulty, Reason: "output_rate_faulty"},
		telemetry.FailureCategoryOK)
	slowSuccess := feedbackTestEvent(aggregationRequestID(123), completedAt, "provider-d", 9, 14,
		health.Feedback{Status: health.FeedbackStatusSlow, Reason: "output_rate_slow"},
		telemetry.FailureCategoryOK)
	unassessed := feedbackTestEvent(aggregationRequestID(124), completedAt, "provider-e", 10, 15,
		health.Feedback{}, telemetry.FailureCategoryOK)

	rows := []models.RequestLog{
		mustMapEvent(t, redact.New(), retried),
		mustMapEvent(t, redact.New(), faultySuccess),
		mustMapEvent(t, redact.New(), slowSuccess),
		mustMapEvent(t, redact.New(), unassessed),
	}
	writer := &gormBatchWriter{db: db}
	if err := writer.WriteBatch(t.Context(), rows); err != nil {
		t.Fatalf("WriteBatch() error = %v", err)
	}
	if err := writer.WriteBatch(t.Context(), rows); err != nil {
		t.Fatalf("duplicate WriteBatch() error = %v", err)
	}

	detail, err := newRequestLogTestService(db).Get(t.Context(), retried.RequestID)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if len(detail.Attempts) != 2 || detail.Attempts[0].Feedback.Status != health.FeedbackStatusFaulty ||
		detail.Attempts[0].FailureCategory != telemetry.FailureCategoryUpstreamHost ||
		detail.Attempts[0].Feedback.FirstResponseMs == nil || *detail.Attempts[0].Feedback.FirstResponseMs != firstResponseA ||
		detail.Attempts[0].Feedback.TokensPerSecond == nil || *detail.Attempts[0].Feedback.TokensPerSecond != rateA ||
		detail.Attempts[1].Feedback.Status != health.FeedbackStatusNormal ||
		detail.Attempts[1].FailureCategory != telemetry.FailureCategoryOK {
		t.Fatalf("persisted retry attempts = %#v", detail.Attempts)
	}
	page, err := newRequestLogTestService(db).List(t.Context(), ListQuery{
		RequestID: retried.RequestID, Limit: 1,
	})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(page.Items) != 1 || page.Items[0].Attempts != nil ||
		page.Items[0].FinalAttemptFeedback.Status != health.FeedbackStatusNormal ||
		page.Items[0].FinalAttemptFeedback.FirstResponseMs == nil ||
		*page.Items[0].FinalAttemptFeedback.FirstResponseMs != firstResponseB {
		t.Fatalf("list item final feedback = %#v, want sequence 2 only", page.Items)
	}

	start := completedAt.Truncate(time.Hour)
	extraAttemptStats := make([]models.UsageAttemptStat, 20)
	for index := range extraAttemptStats {
		extraAttemptStats[index] = models.UsageAttemptStat{
			BucketStartMS: start.UnixMilli(), AccessKeyID: 42, ChannelID: string(channel.OpenAI),
			GroupID: 20, CredentialID: uint(100 + index), Model: "page-" + string(rune('a'+index)),
			AttemptCount: 1,
		}
	}
	if err := db.Create(&extraAttemptStats).Error; err != nil {
		t.Fatalf("create historical unassessed attempt stats: %v", err)
	}
	service := newRequestLogTestService(db)
	admin, err := service.QueryUsage(t.Context(), UsageQuery{
		FromMS: start.UnixMilli(), ToMS: start.Add(time.Hour).UnixMilli(),
		Granularity: UsageGranularityHour, BreakdownPageSize: 20,
	})
	if err != nil {
		t.Fatalf("admin QueryUsage() error = %v", err)
	}
	if admin.Breakdown.AttemptTotal.AttemptCount != 25 ||
		admin.Breakdown.AttemptTotal.AttemptFailureCount != 1 ||
		admin.Breakdown.AttemptTotal.NormalAttemptCount != 1 ||
		admin.Breakdown.AttemptTotal.SlowAttemptCount != 1 ||
		admin.Breakdown.AttemptTotal.FaultyAttemptCount != 2 {
		t.Fatalf("admin attempt total = %+v, want attempts=25 failures=1 normal/slow/faulty=1/1/2", admin.Breakdown.AttemptTotal)
	}
	pagedAdmin, err := service.QueryUsage(t.Context(), UsageQuery{
		FromMS: start.UnixMilli(), ToMS: start.Add(time.Hour).UnixMilli(),
		Granularity: UsageGranularityHour, BreakdownPage: 2, BreakdownPageSize: 20,
	})
	if err != nil {
		t.Fatalf("paged admin QueryUsage() error = %v", err)
	}
	if len(pagedAdmin.Breakdown.Rows) != 5 || pagedAdmin.Breakdown.Pagination.TotalItems != 25 ||
		pagedAdmin.Breakdown.Pagination.TotalPages != 2 ||
		pagedAdmin.Breakdown.AttemptTotal != admin.Breakdown.AttemptTotal {
		t.Fatalf("paged admin attempt breakdown = %+v, want five rows and unchanged full-range totals", pagedAdmin.Breakdown)
	}
	var faultySuccessRow *UsageBreakdownRow
	for index := range pagedAdmin.Breakdown.Rows {
		if pagedAdmin.Breakdown.Rows[index].Model == "provider-c" {
			faultySuccessRow = &pagedAdmin.Breakdown.Rows[index]
		}
	}
	if faultySuccessRow == nil || faultySuccessRow.AttemptFailureCount != 0 ||
		faultySuccessRow.FaultyAttemptCount != 1 || faultySuccessRow.AttemptCount != 1 {
		t.Fatalf("faulty successful attempt aggregate = %+v, want separate feedback and actual failure counts", faultySuccessRow)
	}

	groupID := uint(8)
	filtered, err := service.QueryUsage(t.Context(), UsageQuery{
		FromMS: start.UnixMilli(), ToMS: start.Add(time.Hour).UnixMilli(),
		Granularity: UsageGranularityHour, GroupID: &groupID, BreakdownPageSize: 20,
	})
	if err != nil {
		t.Fatalf("filtered QueryUsage() error = %v", err)
	}
	if filtered.Breakdown.AttemptTotal.AttemptCount != 1 ||
		filtered.Breakdown.AttemptTotal.FaultyAttemptCount != 1 {
		t.Fatalf("group-filtered attempt total = %+v, want only provider-c", filtered.Breakdown.AttemptTotal)
	}

	accessKeyID := uint(42)
	accessKey, err := service.QueryUsage(t.Context(), UsageQuery{
		FromMS: start.UnixMilli(), ToMS: start.Add(time.Hour).UnixMilli(),
		Granularity: UsageGranularityHour, AccessKeyID: &accessKeyID, BreakdownPageSize: 20,
	})
	if err != nil {
		t.Fatalf("access-key QueryUsage() error = %v", err)
	}
	if accessKey.Breakdown.Scope != "access_key" || len(accessKey.Breakdown.Rows) != 20 ||
		accessKey.Breakdown.Pagination.TotalItems != 25 || accessKey.Breakdown.Pagination.TotalPages != 2 ||
		accessKey.Breakdown.AttemptTotal != admin.Breakdown.AttemptTotal {
		t.Fatalf("access-key attempt breakdown = %+v", accessKey.Breakdown)
	}
	for _, row := range accessKey.Breakdown.Rows {
		if row.GroupID != nil || row.ChannelID != nil {
			t.Fatalf("access-key feedback row exposes admin dimensions: %+v", row)
		}
	}
}

func TestFeedbackFinalAttemptWithoutUsageIdentityPersistsToListAndGet(t *testing.T) {
	db := openRequestLogQueryDB(t)
	event := testEvent(aggregationRequestID(125))
	event.Status = telemetry.RequestStatusError
	event.StatusCode = 502
	event.Attempts[0].StatusCode = 502
	event.Attempts[0].FailureCategory = telemetry.FailureCategoryUpstreamHost
	event.Attempts[0].DispatchState = execution.DispatchMaybeSent
	event.Attempts[0].Feedback = health.Feedback{
		Status: health.FeedbackStatusFaulty, Reason: "upstream_failure",
	}
	event.Usage.GroupID = 0
	event.Usage.ChannelID = ""
	event.Usage.CredentialID = 0
	event.Usage.AttemptSequence = 0
	event.Usage.Pricing.UpstreamModel = ""

	row := mustMapEvent(t, redact.New(), event)
	if row.GroupID != 0 || row.ChannelID != "" || row.CredentialID != 0 ||
		row.AttemptRows[0].GroupID == 0 || row.AttemptRows[0].CredentialID == 0 {
		t.Fatalf("request and attempt identity fixture = request %d/%q/%d, attempt %d/%d",
			row.GroupID, row.ChannelID, row.CredentialID,
			row.AttemptRows[0].GroupID, row.AttemptRows[0].CredentialID)
	}
	writer := &gormBatchWriter{db: db}
	if err := writer.WriteBatch(t.Context(), []models.RequestLog{row}); err != nil {
		t.Fatalf("WriteBatch() error = %v", err)
	}

	service := newRequestLogTestService(db)
	detail, err := service.Get(t.Context(), event.RequestID)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if len(detail.Attempts) != 1 || detail.Attempts[0].Feedback.Status != health.FeedbackStatusFaulty ||
		detail.FinalAttemptFeedback.Status != health.FeedbackStatusFaulty ||
		detail.FinalAttemptFeedback.Reason != "upstream_failure" {
		t.Fatalf("Get() final feedback = %+v, persisted attempt = %+v", detail.FinalAttemptFeedback, detail.Attempts)
	}

	page, err := service.List(t.Context(), ListQuery{RequestID: event.RequestID, Limit: 1})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(page.Items) != 1 || page.Items[0].FinalAttemptFeedback.Status != health.FeedbackStatusFaulty ||
		page.Items[0].FinalAttemptFeedback.Reason != "upstream_failure" {
		t.Fatalf("List() final feedback = %+v, want faulty upstream_failure", page.Items)
	}
}

func feedbackTestEvent(
	id string,
	completedAt time.Time,
	model string,
	groupID, credentialID uint,
	feedback health.Feedback,
	failureCategory telemetry.FailureCategory,
) telemetry.RequestEvent {
	event := testEvent(id)
	event.CompletedAt = completedAt
	event.UpstreamModel = model
	event.UpstreamReportedModel = model
	event.Attempts[0].GroupID = groupID
	event.Attempts[0].CredentialID = credentialID
	event.Attempts[0].UpstreamModel = model
	event.Attempts[0].Feedback = feedback
	event.Attempts[0].FailureCategory = failureCategory
	event.Usage.GroupID = groupID
	event.Usage.CredentialID = credentialID
	event.Usage.AttemptSequence = 1
	event.Usage.Pricing.UpstreamModel = model
	return event
}
