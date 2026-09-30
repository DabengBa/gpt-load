package control

import (
	"encoding/json/v2"
	"math"
	"strings"
	"testing"

	"gpt-load/internal/health"
	"gpt-load/internal/pricing"
	"gpt-load/internal/protocol"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

func TestFeedbackRequestLogAPIProjectsFinalListAttemptAndEveryDetailAttempt(t *testing.T) {
	firstResponse := int64(31_000)
	firstRate := 8.0
	finalResponse := int64(400)
	finalRate := 25.0
	record := requestlog.Record{
		RequestID:           "00000000-0000-4000-8000-000000000951",
		Protocol:            protocol.OpenAICompletions,
		ModelConsistency:    telemetry.ModelConsistencyNotApplicable,
		Status:              telemetry.RequestStatusSuccess,
		UsageState:          usage.StateNotApplicable,
		CostState:           pricing.CostStateNotApplicable,
		PricingCompleteness: pricing.CompletenessNotApplicable,
		FinalAttemptFeedback: health.Feedback{
			Status:          health.FeedbackStatusNormal,
			FirstResponseMs: &finalResponse, TokensPerSecond: &finalRate,
		},
		Attempts: []requestlog.Attempt{
			{
				Sequence: 1, FailureCategory: telemetry.FailureCategoryUpstreamHost,
				Feedback: health.Feedback{
					Status: health.FeedbackStatusFaulty, Reason: "upstream_failure",
					FirstResponseMs: &firstResponse, TokensPerSecond: &firstRate,
				},
			},
			{
				Sequence: 2, FailureCategory: telemetry.FailureCategoryOK,
				Feedback: health.Feedback{
					Status:          health.FeedbackStatusNormal,
					FirstResponseMs: &finalResponse, TokensPerSecond: &finalRate,
				},
			},
		},
	}
	page, err := mapRequestLogListResponse(requestlog.Page{Items: []requestlog.Record{record}}, nil)
	if err != nil {
		t.Fatalf("mapRequestLogListResponse() error = %v", err)
	}
	if len(page.Items) != 1 || page.Items[0].FeedbackStatus == nil ||
		*page.Items[0].FeedbackStatus != health.FeedbackStatusNormal ||
		page.Items[0].FeedbackReason != nil || page.Items[0].ProviderFirstResponseMs == nil ||
		*page.Items[0].ProviderFirstResponseMs != finalResponse ||
		page.Items[0].ProviderTokensPerSecond == nil || *page.Items[0].ProviderTokensPerSecond != finalRate {
		t.Fatalf("list feedback = %#v, want final sequence 2", page.Items)
	}
	detail, err := mapRequestLogDetailResponse(record, nil)
	if err != nil {
		t.Fatalf("mapRequestLogDetailResponse() error = %v", err)
	}
	if len(detail.Attempts) != 2 || detail.Attempts[0].FeedbackStatus == nil ||
		*detail.Attempts[0].FeedbackStatus != health.FeedbackStatusFaulty ||
		detail.Attempts[0].FeedbackReason == nil || *detail.Attempts[0].FeedbackReason != "upstream_failure" ||
		detail.Attempts[0].ProviderFirstResponseMs == nil ||
		*detail.Attempts[0].ProviderFirstResponseMs != firstResponse ||
		detail.Attempts[1].FeedbackStatus == nil ||
		*detail.Attempts[1].FeedbackStatus != health.FeedbackStatusNormal {
		t.Fatalf("detail attempt feedback = %#v", detail.Attempts)
	}
	raw, err := json.Marshal(detail)
	if err != nil {
		t.Fatalf("marshal request log detail: %v", err)
	}
	if !strings.Contains(string(raw), `"provider_first_response_ms":31000`) ||
		!strings.Contains(string(raw), `"feedback_status":"normal"`) ||
		strings.Contains(string(raw), `"feedback_reason":""`) {
		t.Fatalf("request log detail JSON = %s", raw)
	}
}

func TestFeedbackUsageBreakdownAPIValidatesAndProjectsThreeCounts(t *testing.T) {
	summary := requestlog.UsageAggregate{RequestCount: 1, SuccessCount: 1}
	breakdown := requestlog.UsageBreakdown{
		Scope: "access_key",
		Rows: []requestlog.UsageBreakdownRow{{
			Model:          "model-a",
			UsageAggregate: summary,
			UsageAttemptAggregate: requestlog.UsageAttemptAggregate{
				AttemptCount: 5, AttemptFailureCount: 1,
				NormalAttemptCount: 1, SlowAttemptCount: 1, FaultyAttemptCount: 2,
			},
		}},
		Total: summary,
		AttemptTotal: requestlog.UsageAttemptAggregate{
			AttemptCount: 5, AttemptFailureCount: 1,
			NormalAttemptCount: 1, SlowAttemptCount: 1, FaultyAttemptCount: 2,
		},
		Pagination: requestlog.UsagePagination{Page: 1, PageSize: 20, TotalItems: 1, TotalPages: 1},
	}
	response, err := mapUsageBreakdown(breakdown, true, mustMapUsageAggregateForTest(t, summary))
	if err != nil {
		t.Fatalf("mapUsageBreakdown() error = %v", err)
	}
	encoded, err := json.Marshal(response)
	if err != nil {
		t.Fatalf("marshal usage breakdown: %v", err)
	}
	for _, expected := range []string{
		`"normal_attempt_count":1`, `"slow_attempt_count":1`, `"faulty_attempt_count":2`,
	} {
		if !strings.Contains(string(encoded), expected) {
			t.Fatalf("usage breakdown JSON = %s, missing %s", encoded, expected)
		}
	}
	for _, invalid := range []requestlog.UsageAttemptAggregate{
		{AttemptCount: 2, NormalAttemptCount: 1, SlowAttemptCount: 1, FaultyAttemptCount: 1},
		{AttemptCount: 1, AttemptFailureCount: 2},
		{AttemptCount: 1, FaultyAttemptCount: -1},
	} {
		if _, err := mapUsageAttemptAggregate(invalid); err == nil {
			t.Errorf("mapUsageAttemptAggregate(%+v) error = nil, want invalid count rejection", invalid)
		}
	}
}

func TestFeedbackRequestLogAPIRejectsNonFiniteTokenRateAndLeavesUnassessedNull(t *testing.T) {
	record := requestlog.Record{
		Protocol:             protocol.OpenAICompletions,
		ModelConsistency:     telemetry.ModelConsistencyNotApplicable,
		Status:               telemetry.RequestStatusSuccess,
		UsageState:           usage.StateNotApplicable,
		CostState:            pricing.CostStateNotApplicable,
		PricingCompleteness:  pricing.CompletenessNotApplicable,
		FinalAttemptFeedback: health.Feedback{TokensPerSecond: new(math.Inf(1))},
	}
	if _, err := mapRequestLogListResponse(requestlog.Page{Items: []requestlog.Record{record}}, nil); err == nil {
		t.Fatal("mapRequestLogListResponse() error = nil, want non-finite provider token rate rejection")
	}
	record.FinalAttemptFeedback = health.Feedback{}
	page, err := mapRequestLogListResponse(requestlog.Page{Items: []requestlog.Record{record}}, nil)
	if err != nil {
		t.Fatalf("mapRequestLogListResponse() with unassessed feedback error = %v", err)
	}
	if page.Items[0].FeedbackStatus != nil || page.Items[0].FeedbackReason != nil ||
		page.Items[0].ProviderFirstResponseMs != nil || page.Items[0].ProviderTokensPerSecond != nil {
		t.Fatalf("unassessed list feedback = %#v, want null fields", page.Items[0])
	}
}
