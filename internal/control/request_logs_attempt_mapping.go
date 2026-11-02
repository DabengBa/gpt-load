package control

import (
	"fmt"
	"strconv"

	"gpt-load/internal/channel"
	"gpt-load/internal/pricing"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/telemetry"
)

func mapRequestLogDetailResponse(
	record requestlog.Record,
	credentialLabels map[uint]string,
) (requestLogDetailResponse, error) {
	usageCost, err := mapRequestLogUsageCost(record)
	if err != nil {
		return requestLogDetailResponse{}, err
	}
	item, err := mapRequestLogItemResponse(record, usageCost, credentialLabels)
	if err != nil {
		return requestLogDetailResponse{}, err
	}
	attempts := make([]requestLogAttemptResponse, 0, len(record.Attempts))
	for _, attempt := range record.Attempts {
		mapped, err := mapRequestLogAttempt(attempt, credentialLabels)
		if err != nil {
			return requestLogDetailResponse{}, err
		}
		attempts = append(attempts, mapped)
	}
	return requestLogDetailResponse{
		requestLogItemResponse: item,
		Attempts:               attempts,
	}, nil
}

func mapRequestLogAttempt(
	attempt requestlog.Attempt,
	credentialLabels map[uint]string,
) (requestLogAttemptResponse, error) {
	channelID, err := nullableRequestLogChannelID(attempt.ChannelID)
	if err != nil {
		return requestLogAttemptResponse{}, fmt.Errorf("map request log attempt: %w", err)
	}
	credentialID, err := nullableRequestLogID(attempt.CredentialID, "credential")
	if err != nil {
		return requestLogAttemptResponse{}, fmt.Errorf("map request log attempt: %w", err)
	}
	operation, err := nullableRequestLogOperation(attempt.Operation)
	if err != nil {
		return requestLogAttemptResponse{}, fmt.Errorf("map request log attempt: %w", err)
	}
	routeMode, err := nullableRequestLogRouteMode(attempt.RouteMode)
	if err != nil {
		return requestLogAttemptResponse{}, fmt.Errorf("map request log attempt: %w", err)
	}
	dispatchState, err := nullableRequestLogDispatchState(attempt.DispatchState)
	if err != nil {
		return requestLogAttemptResponse{}, fmt.Errorf("map request log attempt: %w", err)
	}
	upstreamProtocol, err := nullableRequestLogUpstreamProtocol(attempt.UpstreamProtocol)
	if err != nil {
		return requestLogAttemptResponse{}, fmt.Errorf("map request log attempt: %w", err)
	}
	failureOrigin, err := nullableRequestLogFailureOrigin(attempt.FailureOrigin)
	if err != nil {
		return requestLogAttemptResponse{}, fmt.Errorf("map request log attempt: %w", err)
	}
	failureScope, err := nullableRequestLogFailureScope(attempt.FailureScope)
	if err != nil {
		return requestLogAttemptResponse{}, fmt.Errorf("map request log attempt: %w", err)
	}
	retryDirective, err := nullableRequestLogRetryDirective(attempt.RetryDirective)
	if err != nil {
		return requestLogAttemptResponse{}, fmt.Errorf("map request log attempt: %w", err)
	}
	effect, err := nullableRequestLogEffect(attempt.Effect)
	if err != nil {
		return requestLogAttemptResponse{}, fmt.Errorf("map request log attempt: %w", err)
	}
	receipt, err := mapRequestLogPricingReceipt(attempt.PricingReceipt)
	if err != nil {
		return requestLogAttemptResponse{}, err
	}
	feedback, err := mapRequestLogFeedback(attempt.Feedback)
	if err != nil {
		return requestLogAttemptResponse{}, err
	}
	return requestLogAttemptResponse{
		Sequence:                attempt.Sequence,
		GroupID:                 attempt.GroupID,
		GroupName:               attempt.GroupName,
		ChannelID:               channelID,
		CredentialID:            credentialID,
		CredentialName:          credentialLabelFor(credentialLabels, credentialID),
		Operation:               operation,
		RouteMode:               routeMode,
		UpstreamModel:           nullableRequestLogModel(attempt.UpstreamModel),
		UpstreamRequestID:       nullableRequestLogModel(attempt.UpstreamRequestID),
		DispatchState:           dispatchState,
		ResponseStarted:         attempt.ResponseStarted,
		UpstreamProtocol:        upstreamProtocol,
		Reasoning:               mapRequestLogReasoningConfig(attempt.Reasoning),
		StatusCode:              attempt.StatusCode,
		DurationMs:              attempt.DurationMs,
		FeedbackStatus:          feedback.FeedbackStatus,
		FeedbackReason:          feedback.FeedbackReason,
		ProviderFirstResponseMs: feedback.ProviderFirstResponseMs,
		ProviderTokensPerSecond: feedback.ProviderTokensPerSecond,
		FailureCategory:         attempt.FailureCategory,
		FailureOrigin:           failureOrigin,
		FailureScope:            failureScope,
		RetryDirective:          retryDirective,
		Effect:                  effect,
		RuleID:                  nullableRequestLogModel(attempt.RuleID),
		Action:                  requestLogAttemptAction(attempt.Action),
		WillRetry:               attempt.WillRetry,
		ErrorCode:               attempt.ErrorCode,
		ErrorSummary:            attempt.ErrorSummary,
		Committed:               attempt.Committed,
		PricingReceipt:          receipt,
	}, nil
}

func requestLogAttemptAction(action telemetry.Action) string {
	return string(action)
}

func mapRequestLogPricingReceipt(
	receipt *pricing.Receipt,
) (*requestLogPricingReceiptResponse, error) {
	if receipt == nil {
		return nil, nil
	}
	if err := pricing.ValidateReceipt(*receipt); err != nil {
		return nil, fmt.Errorf("map request log pricing receipt: %w", err)
	}
	result := &requestLogPricingReceiptResponse{
		SchemaVersion: receipt.SchemaVersion,
		Method:        receipt.Method,
		MethodVersion: receipt.MethodVersion,
		Currency:      receipt.Currency,
		PricingMode:   receipt.PricingMode,
		Rule:          requestLogPricingIdentityResponse{ModelID: receipt.Rule.ModelID},
		LineItems:     make([]requestLogPricingLineResponse, 0, len(receipt.LineItems)),
		TotalNanoUSD:  strconv.FormatInt(receipt.TotalNanoUSD, 10),
	}
	channelID := channel.ID(receipt.Rule.ChannelID)
	if _, ok := requestLogChannelRegistry.Get(channelID); !ok {
		return nil, fmt.Errorf("map request log pricing receipt: unknown channel ID")
	}
	result.Rule.ChannelID = new(string(channelID))
	if receipt.ContextThresholdTokens != nil {
		value := strconv.FormatInt(*receipt.ContextThresholdTokens, 10)
		result.ContextThresholdTokens = &value
	}
	for _, line := range receipt.LineItems {
		mapped := requestLogPricingLineResponse{
			Code:     line.Code,
			Quantity: strconv.FormatInt(line.Quantity, 10),
			Multiplier: requestLogPricingMultiplierResponse{
				Numerator:   strconv.FormatInt(line.Multiplier.Numerator, 10),
				Denominator: strconv.FormatInt(line.Multiplier.Denominator, 10),
			},
			State: line.State,
		}
		if line.RateNanoUSDPerMillion != nil {
			value := strconv.FormatInt(*line.RateNanoUSDPerMillion, 10)
			mapped.RateNanoUSDPerMillion = &value
		}
		if line.AmountNanoUSD != nil {
			value := strconv.FormatInt(*line.AmountNanoUSD, 10)
			mapped.AmountNanoUSD = &value
		}
		result.LineItems = append(result.LineItems, mapped)
	}
	return result, nil
}
