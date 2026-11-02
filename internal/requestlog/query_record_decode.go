package requestlog

import (
	"fmt"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/pricing"
	"gpt-load/internal/protocol"
	"gpt-load/internal/reasoning"
	"gpt-load/internal/storage/models"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

func decodeAccessKeyRef(accessKeyID uint) AccessKeyRef {
	// access_key_id = 0 是控制面观察（模型测活、凭据测活）：这一行不属于任何访问密钥。
	// 不能沿用“先标记已删除、批量补名时再降级”的默认值，否则这类行会被渲染成
	// “已删除 · #0”，把“没有访问密钥”误报成“引用的密钥已消失”。
	if accessKeyID == 0 {
		return AccessKeyRef{}
	}
	return AccessKeyRef{ID: accessKeyID, Deleted: true}
}

func decodeRequestLogRows(rows []models.RequestLog) ([]Record, error) {
	records := make([]Record, 0, len(rows))
	for _, row := range rows {
		if err := validateStoredModelObservation(row); err != nil {
			return nil, err
		}
		if err := validateRequestLogUsageCost(row); err != nil {
			return nil, err
		}
		storedAffinitySource, storedAffinityState := NormalizeAffinityObservation(row.AffinitySource, row.AffinityState)
		records = append(records, Record{
			RequestID:                    row.ID,
			CompletedAtMS:                row.CompletedAtMS,
			AccessKey:                    decodeAccessKeyRef(row.AccessKeyID),
			Protocol:                     protocol.Protocol(row.Protocol),
			Operation:                    execution.Operation(row.Operation),
			ClientModel:                  row.ClientModel,
			UpstreamModel:                row.UpstreamModel,
			UpstreamReportedModel:        row.UpstreamReportedModel,
			ModelConsistency:             telemetry.ModelConsistency(row.ModelConsistency),
			Status:                       telemetry.RequestStatus(row.Status),
			StatusCode:                   row.StatusCode,
			Stream:                       row.Stream,
			FirstResponseMs:              row.FirstResponseMs,
			DurationMs:                   row.DurationMs,
			AttemptCount:                 row.AttemptCount,
			ErrorCode:                    row.ErrorCode,
			ErrorSummary:                 row.ErrorSummary,
			AffinityHit:                  row.AffinityHit,
			ResolvedAffinityCredentialID: row.ResolvedAffinityCredentialID,
			FirstAttemptCredentialID:     row.FirstAttemptCredentialID,
			AffinityHitAssignmentReached: row.AffinityHitAssignmentReached,
			ContinuityHit:                row.ContinuityHit,
			AffinityKey:                  NormalizeAffinityKey(row.AffinityKey),
			AffinitySource:               storedAffinitySource,
			AffinityState:                storedAffinityState,
			Reasoning: reasoning.Config{
				Mode:         row.ReasoningMode,
				Effort:       row.ReasoningEffort,
				BudgetTokens: row.ReasoningBudgetTokens,
			},
			Attempts:                nil,
			GroupID:                 row.GroupID,
			ChannelID:               channel.ID(row.ChannelID),
			CredentialID:            row.CredentialID,
			UsageState:              usage.State(row.UsageState),
			CostState:               pricing.CostState(row.CostState),
			PricingCompleteness:     pricing.Completeness(row.PricingCompleteness),
			UncachedInputTokens:     row.UncachedInputTokens,
			CacheReadTokens:         row.CacheReadTokens,
			CacheWrite5MTokens:      row.CacheWrite5MTokens,
			CacheWrite1HTokens:      row.CacheWrite1HTokens,
			CacheWriteUnknownTokens: row.CacheWriteUnknownTokens,
			OutputTokens:            row.OutputTokens,
			EstimatedCostNanoUSD:    row.EstimatedCostNanoUSD,
		})
	}
	return records, nil
}

func validateStoredModelObservation(row models.RequestLog) error {
	successfulModeledRequest := row.Status == string(telemetry.RequestStatusSuccess) &&
		row.UpstreamModel != ""
	reported := row.UpstreamReportedModel != ""

	switch telemetry.ModelConsistency(row.ModelConsistency) {
	case telemetry.ModelConsistencyNotApplicable:
		if reported || successfulModeledRequest {
			return fmt.Errorf("decode request log model consistency: invalid not-applicable observation")
		}
	case telemetry.ModelConsistencyUnknown:
		if !successfulModeledRequest || reported {
			return fmt.Errorf("decode request log model consistency: invalid unknown observation")
		}
	case telemetry.ModelConsistencyMatch, telemetry.ModelConsistencyMismatch:
		if !successfulModeledRequest || !reported {
			return fmt.Errorf("decode request log model consistency: invalid observed model")
		}
	default:
		return fmt.Errorf(
			"decode request log model consistency: invalid state %q",
			row.ModelConsistency,
		)
	}
	return nil
}

func validateRequestLogUsageCost(row models.RequestLog) error {
	if err := validateFrozenPricingState(
		usage.State(row.UsageState),
		pricing.CostState(row.CostState),
		pricing.Completeness(row.PricingCompleteness),
		row.EstimatedCostNanoUSD,
	); err != nil {
		return fmt.Errorf("decode request log usage/cost: %w", err)
	}
	for _, value := range []int64{
		row.UncachedInputTokens,
		row.CacheReadTokens,
		row.CacheWrite5MTokens,
		row.CacheWrite1HTokens,
		row.CacheWriteUnknownTokens,
		row.OutputTokens,
	} {
		if value < 0 {
			return fmt.Errorf("decode request log usage tokens: negative value")
		}
	}
	if _, ok := usage.CheckedTotal(usage.Tokens{
		UncachedInput:     row.UncachedInputTokens,
		CacheRead:         row.CacheReadTokens,
		CacheWrite5M:      row.CacheWrite5MTokens,
		CacheWrite1H:      row.CacheWrite1HTokens,
		CacheWriteUnknown: row.CacheWriteUnknownTokens,
		Output:            row.OutputTokens,
	}); !ok {
		return fmt.Errorf("decode request log usage tokens: total overflow")
	}
	return nil
}
