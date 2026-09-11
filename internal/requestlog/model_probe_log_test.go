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

// 模型测活的日志行契约（design docs/design/model-probe.md §3 D3）。
// 这些断言必须走真实 Service → worker → mapEvent → sqlite：control 包的 fixture
// 只注入记录型 sink，既不经过 mapEvent 也不触发 DB CHECK，证明不了 R3/R4。
const (
	probeModel        = "gpt-4o"
	probeGroupID      = 7
	probeCredentialID = 33
)

type probeAttemptFixture struct {
	sequence        int
	statusCode      int
	dispatchState   execution.DispatchState
	failureCategory telemetry.FailureCategory
	failureOrigin   execution.ErrorOrigin
	failureScope    execution.ErrorScope
}

// probeEvent 按 §3 D3 的映射表构造一条 operation=probe 事件：
// passed → success + model_consistency=unknown；failed/inconclusive → error + 归零。
func probeEvent(
	id string,
	passed bool,
	reason string,
	attempts []probeAttemptFixture,
) telemetry.RequestEvent {
	event := telemetry.RequestEvent{
		RequestID:             id,
		CompletedAt:           time.Date(2026, time.September, 11, 9, 0, 0, 0, time.UTC),
		AccessKeyID:           0,
		Protocol:              protocol.OpenAICompletions,
		Operation:             execution.OperationProbe,
		ClientModel:           probeModel,
		UpstreamModel:         probeModel,
		UpstreamReportedModel: "",
		ModelConsistency:      telemetry.ModelConsistencyUnknown,
		Status:                telemetry.RequestStatusSuccess,
		StatusCode:            200,
		DurationMs:            412,
	}
	if !passed {
		event.Status = telemetry.RequestStatusError
		event.ModelConsistency = telemetry.ModelConsistencyNotApplicable
		event.ErrorCode = reason
		event.StatusCode = attempts[len(attempts)-1].statusCode
	}
	event.Attempts = make([]telemetry.Attempt, 0, len(attempts))
	for _, attempt := range attempts {
		event.Attempts = append(event.Attempts, telemetry.Attempt{
			Sequence:        attempt.sequence,
			CompletedAt:     event.CompletedAt,
			GroupID:         probeGroupID,
			GroupName:       "openai-main",
			ChannelID:       channel.OpenAI,
			CredentialID:    probeCredentialID,
			Operation:       execution.OperationProbe,
			RouteMode:       channel.RouteNative,
			UpstreamModel:   probeModel,
			DispatchState:   attempt.dispatchState,
			StatusCode:      attempt.statusCode,
			DurationMs:      400,
			FailureCategory: attempt.failureCategory,
			FailureOrigin:   attempt.failureOrigin,
			FailureScope:    attempt.failureScope,
			Action:          telemetry.ActionTerminate,
			Effect:          telemetry.EffectNone,
			ErrorCode:       reason,
		})
	}
	event.Usage = telemetry.UsageObservation{
		GroupID:         probeGroupID,
		ChannelID:       channel.OpenAI,
		CredentialID:    probeCredentialID,
		AttemptSequence: attempts[len(attempts)-1].sequence,
		Result:          usage.Result{State: usage.StateNotApplicable},
		Pricing: telemetry.PricingObservation{
			UpstreamModel:       probeModel,
			CostState:           string(pricing.CostStateNotApplicable),
			PricingCompleteness: string(pricing.CompletenessNotApplicable),
		},
	}
	return event
}

// billableEvent 是普通数据面行：必须照常进入用量与凭据统计。
func billableEvent(id string) telemetry.RequestEvent {
	event := testEvent(id)
	event.Usage.Result = usage.Result{
		State:  usage.StateComplete,
		Tokens: usage.Tokens{UncachedInput: 10, Output: 5},
	}
	event.Usage.Pricing = telemetry.PricingObservation{
		UpstreamModel:        "upstream-model",
		CostState:            string(pricing.CostStatePriced),
		PricingCompleteness:  string(pricing.CompletenessComplete),
		EstimatedCostNanoUSD: 250_000_000,
	}
	return event
}

func emitRequestLogEvents(t *testing.T, db *gorm.DB, events ...telemetry.RequestEvent) *Service {
	t.Helper()
	service := newRequestLogTestService(db)
	if err := service.Start(); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	for _, event := range events {
		service.Emit(event)
	}
	if err := service.Stop(context.Background()); err != nil {
		t.Fatalf("Stop() error = %v", err)
	}
	stats := service.Stats()
	if stats.DroppedPersistFailedTotal != 0 {
		t.Fatalf("日志行被丢弃（mapEvent/写入失败）= %d, want 0", stats.DroppedPersistFailedTotal)
	}
	if stats.PersistedTotal != uint64(len(events)) {
		t.Fatalf("PersistedTotal = %d, want %d", stats.PersistedTotal, len(events))
	}
	return service
}

func mustProbeRow(t *testing.T, db *gorm.DB, id string) models.RequestLog {
	t.Helper()
	var row models.RequestLog
	if err := db.Where("id = ?", id).Take(&row).Error; err != nil {
		t.Fatalf("probe 日志行 %q 不存在: %v", id, err)
	}
	return row
}

func mustProbeAttempts(t *testing.T, db *gorm.DB, id string) []models.RequestLogAttempt {
	t.Helper()
	var rows []models.RequestLogAttempt
	if err := db.Where("request_id = ?", id).Order("sequence ASC").Find(&rows).Error; err != nil {
		t.Fatalf("probe attempt 行 %q 查询失败: %v", id, err)
	}
	return rows
}

func TestModelProbeLog(t *testing.T) {
	t.Parallel()
	const (
		passedID   = "11111111-1111-4111-8111-111111111111"
		failedID   = "11111111-1111-4111-8111-111111111112"
		fallbackID = "11111111-1111-4111-8111-111111111113"
	)
	db, _ := openRequestLogFileDB(t)
	emitRequestLogEvents(t, db,
		probeEvent(passedID, true, "", []probeAttemptFixture{{
			sequence: 1, statusCode: 200,
			dispatchState:   execution.DispatchMaybeSent,
			failureCategory: telemetry.FailureCategoryOK,
		}}),
		probeEvent(failedID, false, "invalid_credential", []probeAttemptFixture{{
			sequence: 1, statusCode: 401,
			dispatchState:   execution.DispatchMaybeSent,
			failureCategory: telemetry.FailureCategoryInvalidKey,
			failureOrigin:   execution.ErrorOriginUpstream,
			failureScope:    execution.ErrorScopeCredential,
		}}),
		probeEvent(fallbackID, false, "probe_incompatible", []probeAttemptFixture{
			{
				sequence: 1, statusCode: 500,
				dispatchState:   execution.DispatchMaybeSent,
				failureCategory: telemetry.FailureCategoryUpstreamHost,
				failureOrigin:   execution.ErrorOriginUpstream,
				failureScope:    execution.ErrorScopeModel,
			},
			{
				sequence: 2, statusCode: 400,
				dispatchState:   execution.DispatchNotSent,
				failureCategory: telemetry.FailureCategoryClientError,
				failureOrigin:   execution.ErrorOriginClient,
				failureScope:    execution.ErrorScopeRequest,
			},
		}),
	)

	tests := []struct {
		name                 string
		id                   string
		wantStatus           string
		wantModelConsistency string
		wantErrorCode        string
		wantAttemptCount     int
		wantFailureCategory  string
		wantDispatchState    string
	}{
		{
			name: "passed", id: passedID,
			wantStatus:           string(telemetry.RequestStatusSuccess),
			wantModelConsistency: string(telemetry.ModelConsistencyUnknown),
			wantAttemptCount:     1,
			wantFailureCategory:  string(telemetry.FailureCategoryOK),
			wantDispatchState:    string(execution.DispatchMaybeSent),
		},
		{
			name: "failed", id: failedID,
			wantStatus:           string(telemetry.RequestStatusError),
			wantModelConsistency: string(telemetry.ModelConsistencyNotApplicable),
			wantErrorCode:        "invalid_credential",
			wantAttemptCount:     1,
			wantFailureCategory:  string(telemetry.FailureCategoryInvalidKey),
			wantDispatchState:    string(execution.DispatchMaybeSent),
		},
		{
			name: "protocol fallback keeps every executed attempt", id: fallbackID,
			wantStatus:           string(telemetry.RequestStatusError),
			wantModelConsistency: string(telemetry.ModelConsistencyNotApplicable),
			wantErrorCode:        "probe_incompatible",
			wantAttemptCount:     2,
			wantFailureCategory:  string(telemetry.FailureCategoryClientError),
			wantDispatchState:    string(execution.DispatchNotSent),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			row := mustProbeRow(t, db, test.id)
			if row.Operation != string(execution.OperationProbe) {
				t.Fatalf("operation = %q, want probe", row.Operation)
			}
			if row.AccessKeyID != 0 {
				t.Fatalf("access_key_id = %d, want 0（控制面流量）", row.AccessKeyID)
			}
			if row.Status != test.wantStatus || row.ModelConsistency != test.wantModelConsistency {
				t.Fatalf("status/model_consistency = %q/%q, want %q/%q",
					row.Status, row.ModelConsistency, test.wantStatus, test.wantModelConsistency)
			}
			if row.AttemptCount != test.wantAttemptCount {
				t.Fatalf("attempt_count = %d, want %d", row.AttemptCount, test.wantAttemptCount)
			}
			if row.UsageState != string(usage.StateNotApplicable) ||
				row.CostState != string(pricing.CostStateNotApplicable) ||
				row.PricingCompleteness != string(pricing.CompletenessNotApplicable) ||
				row.EstimatedCostNanoUSD != 0 {
				t.Fatalf("用量/成本状态 = %q/%q/%q/%d, want not_applicable 且 0 成本",
					row.UsageState, row.CostState, row.PricingCompleteness, row.EstimatedCostNanoUSD)
			}
			if row.ErrorCode != test.wantErrorCode {
				t.Fatalf("error_code = %q, want %q", row.ErrorCode, test.wantErrorCode)
			}
			if row.GroupID != probeGroupID || row.CredentialID != probeCredentialID ||
				row.UpstreamModel != probeModel {
				t.Fatalf("归因/模型 = %d/%d/%q, want %d/%d/%q",
					row.GroupID, row.CredentialID, row.UpstreamModel,
					probeGroupID, probeCredentialID, probeModel)
			}

			attempts := mustProbeAttempts(t, db, test.id)
			if len(attempts) != test.wantAttemptCount {
				t.Fatalf("attempt 行数 = %d, want %d", len(attempts), test.wantAttemptCount)
			}
			for index, attempt := range attempts {
				wantSequence := index + 1
				if attempt.Sequence != wantSequence {
					t.Fatalf("attempt sequence = %d, want %d（连续）", attempt.Sequence, wantSequence)
				}
				if attempt.Action != string(telemetry.ActionTerminate) ||
					attempt.Effect != string(telemetry.EffectNone) {
					t.Fatalf("attempt action/effect = %q/%q, want terminate/none",
						attempt.Action, attempt.Effect)
				}
				if attempt.Operation != string(execution.OperationProbe) ||
					attempt.GroupID != probeGroupID ||
					attempt.CredentialID != probeCredentialID ||
					attempt.GroupName == "" {
					t.Fatalf("attempt 归因不完整: %#v", attempt)
				}
			}
			last := attempts[len(attempts)-1]
			if last.FailureCategory != test.wantFailureCategory ||
				last.DispatchState != test.wantDispatchState {
				t.Fatalf("末次 attempt failure_category/dispatch_state = %q/%q, want %q/%q",
					last.FailureCategory, last.DispatchState,
					test.wantFailureCategory, test.wantDispatchState)
			}
		})
	}
}

func TestProbeUsageIsolation(t *testing.T) {
	t.Parallel()
	const (
		probeID  = "22222222-2222-4222-8222-222222222222"
		normalID = "33333333-3333-4333-8333-333333333333"
	)
	db, _ := openRequestLogFileDB(t)
	// Both events share one window so the group-usage assertion has a positive
	// control inside its own range: the normal row must still be counted.
	windowStart := time.Date(2026, time.September, 11, 8, 0, 0, 0, time.UTC)
	normalEvent := billableEvent(normalID)
	normalEvent.CompletedAt = windowStart.Add(50 * time.Minute)
	service := emitRequestLogEvents(t, db,
		probeEvent(probeID, true, "", []probeAttemptFixture{{
			sequence: 1, statusCode: 200,
			dispatchState:   execution.DispatchMaybeSent,
			failureCategory: telemetry.FailureCategoryOK,
		}}),
		normalEvent,
	)

	var journals int64
	if err := db.Model(&models.UsageAggregationJournal{}).
		Where("request_id = ?", probeID).Count(&journals).Error; err != nil {
		t.Fatalf("查询用量聚合 journal: %v", err)
	}
	if journals != 0 {
		t.Fatalf("probe 行产生了用量聚合 journal = %d, want 0", journals)
	}

	var statRows []models.UsageStat
	if err := db.Find(&statRows).Error; err != nil {
		t.Fatalf("查询 usage_stats: %v", err)
	}
	if len(statRows) != 1 || statRows[0].Model != "upstream-model" || statRows[0].RequestCount != 1 {
		t.Fatalf("usage_stats = %#v, want 只有普通行贡献（model=upstream-model, request_count=1）", statRows)
	}

	var attemptStats []models.CredentialAttemptStat
	if err := db.Find(&attemptStats).Error; err != nil {
		t.Fatalf("查询 credential_attempt_stats: %v", err)
	}
	if len(attemptStats) != 1 || attemptStats[0].CredentialID != 8 || attemptStats[0].SuccessCount != 1 {
		t.Fatalf("credential_attempt_stats = %#v, want 只有普通行凭据 8 计一次成功", attemptStats)
	}

	var logRows int64
	if err := db.Model(&models.RequestLog{}).
		Where("id IN ?", []string{probeID, normalID}).Count(&logRows).Error; err != nil {
		t.Fatalf("查询 request_logs: %v", err)
	}
	if logRows != 2 {
		t.Fatalf("request_logs = %d, want 2（隔离不影响日志本身落库）", logRows)
	}

	// 调度中心的分组 24h 请求数/成功率直接读 request_logs（QueryGroupUsage，
	// 与用量聚合 journal 无关），所以 probe 行必须在查询侧排除，
	// 而不是只靠“不产出 journal”。
	groupUsage, err := service.QueryGroupUsage(context.Background(), GroupUsageQuery{
		FromMS: time.Date(2026, time.September, 11, 8, 0, 0, 0, time.UTC).UnixMilli(),
		ToMS:   time.Date(2026, time.September, 11, 10, 0, 0, 0, time.UTC).UnixMilli(),
	})
	if err != nil {
		t.Fatalf("QueryGroupUsage() error = %v", err)
	}
	group, exists := groupUsage[probeGroupID]
	if !exists {
		t.Fatal("normal row missing from group usage: the negative assertion lost its positive control")
	}
	if group.RequestCount != 1 || group.SuccessCount != 1 {
		t.Fatalf("group usage = %#v, want exactly the normal row (probe rows must not count)", group)
	}
}

// Probe rows must also stay out of the credential views, which the aggregation
// tables cannot guarantee on their own: whole-hour segments read
// CredentialAttemptStat / usage_stats (always clean), but the residual sub-hour
// and boundary segments read request_log_attempts / request_logs directly.
func TestProbeCredentialActivityIsolation(t *testing.T) {
	t.Parallel()
	const (
		probeID  = "44444444-4444-4444-8444-444444444444"
		normalID = "55555555-5555-4555-8555-555555555555"
	)
	db, _ := openRequestLogFileDB(t)
	// Residual hour segment below is [08:45Z, 09:00Z); both events fall inside it,
	// so the whole-hour segments are empty and the outcome depends entirely on the
	// direct reads filtering the control-plane operation.
	residual := time.Date(2026, time.September, 11, 8, 50, 0, 0, time.UTC)
	probe := probeEvent(probeID, true, "", []probeAttemptFixture{{
		sequence: 1, statusCode: 200,
		dispatchState:   execution.DispatchMaybeSent,
		failureCategory: telemetry.FailureCategoryOK,
	}})
	probe.CompletedAt = residual
	probe.Attempts[0].CompletedAt = residual
	normal := billableEvent(normalID)
	normal.CompletedAt = residual
	service := emitRequestLogEvents(t, db, probe, normal)

	fromMS := residual.Add(-5 * time.Minute).UnixMilli()
	toMS := residual.Add(30 * time.Minute).UnixMilli()
	activity, err := service.QueryCredentialActivity(context.Background(), CredentialActivityQuery{
		CredentialIDs: []uint{probeCredentialID, 8},
		FromMS:        fromMS,
		ToMS:          toMS,
	})
	if err != nil {
		t.Fatalf("QueryCredentialActivity() error = %v", err)
	}
	if value := activity[probeCredentialID]; value.SuccessCount != 0 || value.FailureCount != 0 {
		t.Fatalf("probe attempt leaked into the 24h credential success/failure counts: %#v", value)
	}
	if value := activity[8]; value.SuccessCount != 1 || value.FailureCount != 0 {
		t.Fatalf("normal credential activity = %#v, want one success (positive control)", value)
	}

	probeWindow, err := service.QueryCredentialWindowUsage(
		context.Background(),
		CredentialWindowUsageQuery{
			CredentialID: probeCredentialID,
			FromMS:       fromMS,
			ToMS:         toMS,
			Source:       CredentialWindowUsageSourceHourlyStats,
		},
	)
	if err != nil {
		t.Fatalf("QueryCredentialWindowUsage() error = %v", err)
	}
	if probeWindow.RequestCount != 0 || probeWindow.SuccessCount != 0 || probeWindow.LastUsedAtMS != nil {
		t.Fatalf("probe row leaked into the credential window usage: %#v", probeWindow)
	}
	normalWindow, err := service.QueryCredentialWindowUsage(
		context.Background(),
		CredentialWindowUsageQuery{
			CredentialID: 8,
			FromMS:       fromMS,
			ToMS:         toMS,
			Source:       CredentialWindowUsageSourceHourlyStats,
		},
	)
	if err != nil {
		t.Fatalf("QueryCredentialWindowUsage() error = %v", err)
	}
	if normalWindow.RequestCount != 1 || normalWindow.LastUsedAtMS == nil {
		t.Fatalf("normal credential window usage = %#v, want one request (positive control)", normalWindow)
	}
}
