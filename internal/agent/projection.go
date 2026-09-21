package agent

import (
	"fmt"
	"strconv"
	"strings"

	"gpt-load/internal/debugcapture"
	"gpt-load/internal/execution"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/usage"
)

func projectRequestSummary(record requestlog.Record) (RequestSummary, error) {
	if err := requestlog.ValidateUsageCostState(
		record.UsageState,
		record.CostState,
		record.PricingCompleteness,
		record.EstimatedCostNanoUSD,
	); err != nil {
		return RequestSummary{}, fmt.Errorf("project request usage/cost: %w", err)
	}
	totalTokens, ok := usage.CheckedTotal(usage.Tokens{
		UncachedInput:     record.UncachedInputTokens,
		CacheRead:         record.CacheReadTokens,
		CacheWrite5M:      record.CacheWrite5MTokens,
		CacheWrite1H:      record.CacheWrite1HTokens,
		CacheWriteUnknown: record.CacheWriteUnknownTokens,
		Output:            record.OutputTokens,
	})
	if !ok {
		return RequestSummary{}, fmt.Errorf("project request tokens: overflow")
	}
	for _, value := range []int64{
		record.UncachedInputTokens,
		record.CacheReadTokens,
		record.CacheWrite5MTokens,
		record.CacheWrite1HTokens,
		record.CacheWriteUnknownTokens,
		record.OutputTokens,
		record.EstimatedCostNanoUSD,
	} {
		if value < 0 {
			return RequestSummary{}, fmt.Errorf("project request: negative accounting value")
		}
	}
	if err := validateSafeMilliseconds(record.CompletedAtMS); err != nil {
		return RequestSummary{}, fmt.Errorf("project request completed_at_ms: %w", err)
	}
	return RequestSummary{
		RequestID:       record.RequestID,
		CompletedAtMS:   record.CompletedAtMS,
		SemanticStatus:  string(record.Status),
		HTTPStatusCode:  record.StatusCode,
		Protocol:        optionalRequestLogString(string(record.Protocol)),
		Operation:       nullableOperation(record.Operation),
		ClientModel:     optionalRequestLogString(record.ClientModel),
		UpstreamModel:   optionalRequestLogString(record.UpstreamModel),
		Stream:          record.Stream,
		FirstResponseMs: record.FirstResponseMs,
		DurationMs:      record.DurationMs,
		AttemptCount:    record.AttemptCount,
		ErrorCode:       record.ErrorCode,
		AccessKeyID:     optionalRequestLogID(record.AccessKey.ID),
		Usage: UsageTotalsView{
			State:                   string(record.UsageState),
			UncachedInputTokens:     strconv.FormatInt(record.UncachedInputTokens, 10),
			CacheReadTokens:         strconv.FormatInt(record.CacheReadTokens, 10),
			CacheWrite5MTokens:      strconv.FormatInt(record.CacheWrite5MTokens, 10),
			CacheWrite1HTokens:      strconv.FormatInt(record.CacheWrite1HTokens, 10),
			CacheWriteUnknownTokens: strconv.FormatInt(record.CacheWriteUnknownTokens, 10),
			OutputTokens:            strconv.FormatInt(record.OutputTokens, 10),
			TotalTokens:             strconv.FormatInt(totalTokens, 10),
		},
		Cost: CostView{
			State:            string(record.CostState),
			Completeness:     string(record.PricingCompleteness),
			EstimatedNanoUSD: strconv.FormatInt(record.EstimatedCostNanoUSD, 10),
		},
		ProviderAttempts: len(record.Attempts),
	}, nil
}

func projectAttempt(
	requestID string,
	attempt requestlog.Attempt,
	scrub func(string) string,
) AttemptView {
	return AttemptView{
		AttemptRef:        logicalAttemptRef(requestID, attempt.Sequence),
		Sequence:          attempt.Sequence,
		GroupID:           optionalRequestLogID(attempt.GroupID),
		CredentialID:      optionalRequestLogID(attempt.CredentialID),
		ChannelID:         optionalRequestLogString(string(attempt.ChannelID)),
		Operation:         nullableOperation(attempt.Operation),
		RouteMode:         optionalRequestLogString(string(attempt.RouteMode)),
		UpstreamModel:     optionalRequestLogString(attempt.UpstreamModel),
		UpstreamRequestID: optionalRequestLogString(attempt.UpstreamRequestID),
		DispatchState:     optionalRequestLogString(string(attempt.DispatchState)),
		ResponseStarted:   attempt.ResponseStarted,
		Protocol:          optionalRequestLogString(string(attempt.UpstreamProtocol)),
		StatusCode:        attempt.StatusCode,
		DurationMs:        attempt.DurationMs,
		FailureCategory:   string(attempt.FailureCategory),
		FailureOrigin:     optionalRequestLogString(string(attempt.FailureOrigin)),
		FailureScope:      optionalRequestLogString(string(attempt.FailureScope)),
		RetryDirective:    optionalRequestLogString(string(attempt.RetryDirective)),
		Effect:            optionalRequestLogString(string(attempt.Effect)),
		Action:            string(attempt.Action),
		WillRetry:         attempt.WillRetry,
		ErrorCode:         attempt.ErrorCode,
		ErrorSummary:      scrub(attempt.ErrorSummary),
		Committed:         attempt.Committed,
	}
}

func nullableOperation(value execution.Operation) *string {
	if value == "" || !value.Valid() {
		return nil
	}
	text := string(value)
	return &text
}

func projectCapture(
	requestID string,
	session debugcapture.SessionRecord,
	validAttemptSequences map[int]struct{},
) CaptureView {
	attempts := make([]CaptureAttemptView, 0, len(session.Attempts))
	for _, attempt := range session.Attempts {
		ref, sequence, linked := captureLogicalAttempt(
			attempt.Metadata.Fields,
			requestID,
			validAttemptSequences,
		)
		linkState := AttemptLinkStateUnlinked
		if linked {
			linkState = AttemptLinkStateLinked
		}
		view := CaptureAttemptView{
			CaptureAttemptID:       attempt.ID,
			StorageSequence:        attempt.Sequence,
			State:                  string(attempt.State),
			StartedAtMS:            attempt.StartedAt.UnixMilli(),
			LogicalAttemptRef:      ref,
			LogicalAttemptSequence: sequence,
			LinkState:              linkState,
		}
		if !attempt.CompletedAt.IsZero() {
			completedAtMS := attempt.CompletedAt.UnixMilli()
			view.CompletedAtMS = &completedAtMS
		}
		attempts = append(attempts, view)
	}
	return CaptureView{
		CaptureID:    session.ID,
		State:        string(session.State),
		Complete:     session.State == debugcapture.StateCompleted,
		CreatedAtMS:  session.CreatedAt.UnixMilli(),
		ExpiresAtMS:  session.ExpiresAt.UnixMilli(),
		Protocol:     optionalRequestLogString(session.Protocol),
		Operation:    optionalRequestLogString(session.Operation),
		ErrorPresent: session.Error != "",
		Attempts:     attempts,
	}
}

// captureLogicalAttempt extracts the explicit logical attempt reference from
// capture attempt metadata. The capture store's own sequence is never used as
// a substitute for the request-log attempt sequence.
func captureLogicalAttempt(
	fields map[string]any,
	requestID string,
	validAttemptSequences map[int]struct{},
) (*string, *int, bool) {
	raw, exists := fields["logical_attempt_id"]
	if !exists {
		return nil, nil, false
	}
	value, ok := raw.(string)
	if !ok {
		return nil, nil, false
	}
	prefix := requestID + ":"
	if !strings.HasPrefix(value, prefix) {
		return nil, nil, false
	}
	parsed, err := strconv.Atoi(strings.TrimPrefix(value, prefix))
	if err != nil || parsed < 1 {
		return nil, nil, false
	}
	if _, exists := validAttemptSequences[parsed]; !exists {
		return nil, nil, false
	}
	ref := value
	sequence := parsed
	return &ref, &sequence, true
}

// logicalAttemptRef builds the canonical logical attempt reference used by
// both request-log attempts and capture metadata.
func logicalAttemptRef(requestID string, sequence int) string {
	return requestID + ":" + strconv.Itoa(sequence)
}

func unavailableEvidence(reasonState string) EvidenceView {
	return EvidenceView{
		SchemaVersion:     SchemaVersion,
		State:             reasonState,
		RawContentExposed: false,
		Captures:          []CaptureView{},
	}
}
