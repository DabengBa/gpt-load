package requestlog

import (
	"context"
	"encoding/json"
	"fmt"

	"gorm.io/gorm"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/pricing"
	"gpt-load/internal/protocol"
	"gpt-load/internal/reasoning"
	"gpt-load/internal/storage/models"
	"gpt-load/internal/telemetry"
)

func (service *Service) Get(ctx context.Context, requestID string) (Record, error) {
	var record Record
	err := service.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var row models.RequestLog
		if err := tx.Where("id = ?", requestID).First(&row).Error; err != nil {
			return fmt.Errorf("query request log detail: %w", err)
		}
		records, err := decodeRequestLogRows([]models.RequestLog{row})
		if err != nil {
			return err
		}
		if err := loadAccessKeyRefs(ctx, tx, records); err != nil {
			return err
		}

		var attemptRows []models.RequestLogAttempt
		if err := tx.Where("request_id = ?", requestID).
			Order("sequence ASC").
			Find(&attemptRows).Error; err != nil {
			return fmt.Errorf("query request log attempts: %w", err)
		}
		attempts, err := decodeAttemptRows(attemptRows)
		if err != nil {
			return err
		}
		records[0].Attempts = attempts
		if len(attempts) > 0 {
			records[0].FinalAttemptDurationMs = new(attempts[len(attempts)-1].DurationMs)
		}
		records[0].FinalAttemptFeedback = finalAttemptFeedback(attempts)
		records[0].RouteMode = finalRouteMode(records[0], attempts)
		records[0].UpstreamProtocol = finalUpstreamProtocol(records[0], attempts)
		receipt := finalPricingReceipt(records[0], attempts)
		records[0].PricingMode = pricingModeForReceipt(receipt)
		records[0].ContextThresholdTokens = contextThresholdForReceipt(receipt)
		record = records[0]
		return nil
	})
	if err != nil {
		return Record{}, err
	}
	return record, nil
}

func decodeAttemptRows(rows []models.RequestLogAttempt) ([]Attempt, error) {
	attempts := make([]Attempt, 0, len(rows))
	for _, row := range rows {
		upstreamProtocol := protocol.Protocol(row.UpstreamProtocol)
		if upstreamProtocol != "" && !upstreamProtocol.Valid() {
			return nil, fmt.Errorf("decode request log attempt %d: invalid upstream protocol", row.Sequence)
		}
		receipt, err := decodeAttemptPricingReceipt(row)
		if err != nil {
			return nil, err
		}
		feedback, err := decodeAttemptFeedback(row)
		if err != nil {
			return nil, err
		}
		attempts = append(attempts, Attempt{
			Sequence:          row.Sequence,
			GroupID:           row.GroupID,
			GroupName:         row.GroupName,
			ChannelID:         channel.ID(row.ChannelID),
			CredentialID:      row.CredentialID,
			Operation:         execution.Operation(row.Operation),
			RouteMode:         channel.RouteMode(row.RouteMode),
			UpstreamModel:     row.UpstreamModel,
			UpstreamRequestID: row.UpstreamRequestID,
			DispatchState:     execution.DispatchState(row.DispatchState),
			ResponseStarted:   row.ResponseStarted,
			UpstreamProtocol:  upstreamProtocol,
			Reasoning: reasoning.Config{
				Mode:         row.ReasoningMode,
				Effort:       row.ReasoningEffort,
				BudgetTokens: row.ReasoningBudgetTokens,
			},
			StatusCode:      row.StatusCode,
			DurationMs:      row.DurationMs,
			Feedback:        feedback,
			FailureCategory: telemetry.FailureCategory(row.FailureCategory),
			FailureOrigin:   execution.ErrorOrigin(row.FailureOrigin),
			FailureScope:    execution.ErrorScope(row.FailureScope),
			RetryDirective:  telemetry.RetryDirective(row.RetryDirective),
			Effect:          telemetry.Effect(row.Effect),
			RuleID:          row.RuleID,
			Action:          telemetry.Action(row.Action),
			WillRetry:       row.WillRetry,
			ErrorCode:       row.ErrorCode,
			ErrorSummary:    row.ErrorSummary,
			Committed:       row.Committed,
			PricingReceipt:  receipt,
		})
	}
	return attempts, nil
}

func decodeAttemptFeedback(row models.RequestLogAttempt) (health.Feedback, error) {
	feedback := health.Feedback{
		Status:          health.FeedbackStatus(row.FeedbackStatus),
		Reason:          row.FeedbackReason,
		FirstResponseMs: row.ProviderFirstResponseMs,
		TokensPerSecond: row.ProviderTokensPerSecond,
	}
	if err := validateAttemptFeedback(feedback); err != nil {
		return health.Feedback{}, fmt.Errorf("decode request log attempt %d feedback: %w", row.Sequence, err)
	}
	return feedback, nil
}

func decodeAttemptPricingReceipt(row models.RequestLogAttempt) (*pricing.Receipt, error) {
	if len(row.PricingReceipt) == 0 || string(row.PricingReceipt) == "null" {
		return nil, nil
	}
	var decoded pricing.Receipt
	if err := json.Unmarshal(row.PricingReceipt, &decoded); err != nil {
		return nil, fmt.Errorf("decode request log pricing receipt: %w", err)
	}
	if err := pricing.ValidateReceipt(decoded); err != nil {
		return nil, fmt.Errorf("decode request log pricing receipt: %w", err)
	}
	if decoded.Rule != (pricing.ReceiptRule{
		ChannelID: row.ChannelID,
		ModelID:   row.UpstreamModel,
	}) {
		return nil, fmt.Errorf("decode request log pricing receipt: channel identity does not match attempt")
	}
	return &decoded, nil
}
