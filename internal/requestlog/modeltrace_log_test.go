package requestlog

import (
	"context"
	"testing"
	"time"

	"gorm.io/gorm"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/pricing"
	"gpt-load/internal/protocol"
	"gpt-load/internal/storage/models"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

// 指纹挑战的日志行契约：每条挑战是一次正式模型请求，
// 因此它必须像普通请求一样落库、可计费、可聚合，
// 只有「归因结果」不进入 request log 之外的决策路径。
const (
	fingerprintModel        = "gpt-4o"
	fingerprintGroupID      = 41
	fingerprintCredentialID = 57
)

type fingerprintAttemptFixture struct {
	statusCode      int
	dispatchState   execution.DispatchState
	failureCategory telemetry.FailureCategory
	failureOrigin   execution.ErrorOrigin
	failureScope    execution.ErrorScope
	errorCode       string
	usage           usage.Result
	cost            pricing.CostState
	completeness    pricing.Completeness
	costNanoUSD     int64
}

func fingerprintEvent(id string, attempt fingerprintAttemptFixture) telemetry.RequestEvent {
	completedAt := time.Date(2026, time.November, 3, 9, 0, 0, 0, time.UTC)
	event := telemetry.RequestEvent{
		RequestID:        id,
		CompletedAt:      completedAt,
		AccessKeyID:      0,
		Protocol:         protocol.OpenAICompletions,
		Operation:        execution.OperationModelTrace,
		ClientModel:      fingerprintModel,
		UpstreamModel:    fingerprintModel,
		ModelConsistency: telemetry.ModelConsistencyUnknown,
		Status:           telemetry.RequestStatusSuccess,
		StatusCode:       attempt.statusCode,
		DurationMs:       3210,
	}
	if attempt.errorCode != "" {
		event.Status = telemetry.RequestStatusError
		event.ModelConsistency = telemetry.ModelConsistencyNotApplicable
		event.ErrorCode = attempt.errorCode
		event.ErrorSummary = attempt.errorCode
	}
	event.Attempts = []telemetry.Attempt{{
		Sequence:        1,
		CompletedAt:     completedAt,
		GroupID:         fingerprintGroupID,
		GroupName:       "fingerprint-main",
		ChannelID:       channel.OpenAI,
		CredentialID:    fingerprintCredentialID,
		Operation:       execution.OperationModelTrace,
		RouteMode:       channel.RouteNative,
		UpstreamModel:   fingerprintModel,
		DispatchState:   attempt.dispatchState,
		ResponseStarted: true,
		StatusCode:      attempt.statusCode,
		DurationMs:      3200,
		FailureCategory: attempt.failureCategory,
		FailureOrigin:   attempt.failureOrigin,
		FailureScope:    attempt.failureScope,
		Action:          telemetry.ActionTerminate,
		Effect:          telemetry.EffectNone,
		ErrorCode:       attempt.errorCode,
		ErrorSummary:    attempt.errorCode,
		Usage:           attempt.usage,
	}}
	event.Usage = telemetry.UsageObservation{
		GroupID:         fingerprintGroupID,
		ChannelID:       channel.OpenAI,
		CredentialID:    fingerprintCredentialID,
		AttemptSequence: 1,
		Result:          attempt.usage,
		Pricing: telemetry.PricingObservation{
			UpstreamModel:        fingerprintModel,
			CostState:            string(attempt.cost),
			PricingCompleteness:  string(attempt.completeness),
			EstimatedCostNanoUSD: attempt.costNanoUSD,
		},
	}
	return event
}

func TestFingerprintChallengeRowsAreFormalRequests(t *testing.T) {
	t.Parallel()
	const (
		successID = "66666666-6666-4666-8666-666666666661"
		failureID = "66666666-6666-4666-8666-666666666662"
	)
	db, _ := openRequestLogFileDB(t)
	emitRequestLogEvents(t, db,
		fingerprintEvent(successID, fingerprintAttemptFixture{
			statusCode:      200,
			dispatchState:   execution.DispatchMaybeSent,
			failureCategory: telemetry.FailureCategoryOK,
			usage: usage.Result{
				State:  usage.StateComplete,
				Tokens: usage.Tokens{UncachedInput: 340, Output: 312},
			},
			cost:         pricing.CostStatePriced,
			completeness: pricing.CompletenessComplete,
			costNanoUSD:  1_250_000,
		}),
		fingerprintEvent(failureID, fingerprintAttemptFixture{
			statusCode:      429,
			dispatchState:   execution.DispatchMaybeSent,
			failureCategory: telemetry.FailureCategoryRateLimited,
			failureOrigin:   execution.ErrorOriginUpstream,
			failureScope:    execution.ErrorScopeCredential,
			errorCode:       "rate_limited",
			usage:           usage.Result{State: usage.StateNotApplicable},
			cost:            pricing.CostStateNotApplicable,
			completeness:    pricing.CompletenessNotApplicable,
		}),
	)

	tests := []struct {
		name            string
		id              string
		wantStatus      string
		wantUsageState  string
		wantUsageOutput int64
		wantCostState   string
		wantCostNanoUSD int64
		wantErrorCode   string
		wantCategory    string
	}{
		{
			name: "succeeded challenge is billable", id: successID,
			wantStatus:      string(telemetry.RequestStatusSuccess),
			wantUsageState:  string(usage.StateComplete),
			wantUsageOutput: 312,
			wantCostState:   string(pricing.CostStatePriced),
			wantCostNanoUSD: 1_250_000,
			wantCategory:    string(telemetry.FailureCategoryOK),
		},
		{
			name: "failed challenge keeps its own evidence", id: failureID,
			wantStatus:     string(telemetry.RequestStatusError),
			wantUsageState: string(usage.StateNotApplicable),
			wantCostState:  string(pricing.CostStateNotApplicable),
			wantErrorCode:  "rate_limited",
			wantCategory:   string(telemetry.FailureCategoryRateLimited),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			row := mustProbeRow(t, db, test.id)
			if row.Operation != string(execution.OperationModelTrace) {
				t.Fatalf("operation = %q, want %q", row.Operation, execution.OperationModelTrace)
			}
			if row.AccessKeyID != 0 {
				t.Fatalf("access_key_id = %d, want 0", row.AccessKeyID)
			}
			if row.Status != test.wantStatus || row.UsageState != test.wantUsageState {
				t.Fatalf("status/usage_state = %q/%q, want %q/%q",
					row.Status, row.UsageState, test.wantStatus, test.wantUsageState)
			}
			if row.OutputTokens != test.wantUsageOutput {
				t.Fatalf("output_tokens = %d, want %d", row.OutputTokens, test.wantUsageOutput)
			}
			if row.CostState != test.wantCostState || row.EstimatedCostNanoUSD != test.wantCostNanoUSD {
				t.Fatalf("cost_state/cost = %q/%d, want %q/%d",
					row.CostState, row.EstimatedCostNanoUSD, test.wantCostState, test.wantCostNanoUSD)
			}
			if row.ErrorCode != test.wantErrorCode {
				t.Fatalf("error_code = %q, want %q", row.ErrorCode, test.wantErrorCode)
			}
			if row.AttemptCount != 1 {
				t.Fatalf("attempt_count = %d, want 1 (one challenge is one attempt)", row.AttemptCount)
			}
			attempts := mustProbeAttempts(t, db, test.id)
			if len(attempts) != 1 {
				t.Fatalf("attempt rows = %d, want 1", len(attempts))
			}
			attempt := attempts[0]
			if attempt.Operation != string(execution.OperationModelTrace) ||
				attempt.FailureCategory != test.wantCategory ||
				attempt.Action != string(telemetry.ActionTerminate) ||
				attempt.Effect != string(telemetry.EffectNone) {
				t.Fatalf("attempt row = %#v", attempt)
			}
			if attempt.GroupID != fingerprintGroupID || attempt.CredentialID != fingerprintCredentialID {
				t.Fatalf("attempt attribution = %d/%d", attempt.GroupID, attempt.CredentialID)
			}
		})
	}
}

// A fingerprint challenge is a real model use, so unlike a control-plane probe
// it must contribute to usage aggregation and to the group/credential views.
func TestFingerprintChallengeUsageIsAggregated(t *testing.T) {
	t.Parallel()
	const (
		challengeA = "77777777-7777-4777-8777-777777777771"
		challengeB = "77777777-7777-4777-8777-777777777772"
		challengeC = "77777777-7777-4777-8777-777777777773"
	)
	db, _ := openRequestLogFileDB(t)
	completedAt := time.Date(2026, time.November, 3, 9, 0, 0, 0, time.UTC)
	events := make([]telemetry.RequestEvent, 0, 3)
	for _, id := range []string{challengeA, challengeB, challengeC} {
		event := fingerprintEvent(id, fingerprintAttemptFixture{
			statusCode:      200,
			dispatchState:   execution.DispatchMaybeSent,
			failureCategory: telemetry.FailureCategoryOK,
			usage: usage.Result{
				State:  usage.StateComplete,
				Tokens: usage.Tokens{UncachedInput: 340, Output: 312},
			},
			cost:         pricing.CostStatePriced,
			completeness: pricing.CompletenessComplete,
			costNanoUSD:  1_250_000,
		})
		event.CompletedAt = completedAt
		events = append(events, event)
	}
	service := emitRequestLogEvents(t, db, events...)

	var journals int64
	if err := db.Model(&models.UsageAggregationJournal{}).
		Where("request_id IN ?", []string{challengeA, challengeB, challengeC}).
		Count(&journals).Error; err != nil {
		t.Fatalf("query usage journals: %v", err)
	}
	if journals != 3 {
		t.Fatalf("usage journals = %d, want one per challenge", journals)
	}

	groupUsage, err := service.QueryGroupUsage(context.Background(), GroupUsageQuery{
		FromMS: completedAt.Add(-time.Hour).UnixMilli(),
		ToMS:   completedAt.Add(time.Hour).UnixMilli(),
	})
	if err != nil {
		t.Fatalf("QueryGroupUsage() error = %v", err)
	}
	group := groupUsage[fingerprintGroupID]
	if group.RequestCount != 3 || group.SuccessCount != 3 {
		t.Fatalf("group usage = %#v, want three formal challenge requests", group)
	}

	activity, err := service.QueryCredentialActivity(context.Background(), CredentialActivityQuery{
		CredentialIDs: []uint{fingerprintCredentialID},
		FromMS:        completedAt.Add(-time.Hour).UnixMilli(),
		ToMS:          completedAt.Add(time.Hour).UnixMilli(),
	})
	if err != nil {
		t.Fatalf("QueryCredentialActivity() error = %v", err)
	}
	if value := activity[fingerprintCredentialID]; value.SuccessCount != 3 {
		t.Fatalf("credential activity = %#v, want three successes", value)
	}
}

// Fingerprint attribution never feeds health decisions: the log rows record the
// attempt's own evidence, and the operation is not a probe, so nothing in the
// control-plane probe exclusion set applies to it.
func TestFingerprintOperationIsNotAProbeObservation(t *testing.T) {
	t.Parallel()
	db, _ := openRequestLogFileDB(t)
	const id = "88888888-8888-4888-8888-888888888881"
	emitRequestLogEvents(t, db, fingerprintEvent(id, fingerprintAttemptFixture{
		statusCode:      200,
		dispatchState:   execution.DispatchMaybeSent,
		failureCategory: telemetry.FailureCategoryOK,
		usage: usage.Result{
			State:  usage.StateComplete,
			Tokens: usage.Tokens{UncachedInput: 12, Output: 7},
		},
		cost:         pricing.CostStatePriced,
		completeness: pricing.CompletenessComplete,
		costNanoUSD:  900,
	}))

	var counted int64
	if err := withCountedModelTrace(db, &counted); err != nil {
		t.Fatalf("count fingerprint rows: %v", err)
	}
	if counted != 1 {
		t.Fatalf("counted fingerprint rows = %d, want 1", counted)
	}
}

// withCountedModelTrace proves the shared traffic filter only excludes the
// probe operation, never the fingerprint challenge operation.
func withCountedModelTrace(db *gorm.DB, target *int64) error {
	return withoutControlPlaneObservations(db.Model(&models.RequestLog{})).
		Where("operation = ?", string(execution.OperationModelTrace)).
		Count(target).Error
}
