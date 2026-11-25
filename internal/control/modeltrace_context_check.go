package control

import (
	"context"
	"net/http"

	"gpt-load/internal/execution"
	"gpt-load/internal/usage"
)

const (
	modelTraceContextPrompt = "Reply with the word hello."
	// The fixed text is six tokens in both o200k_base and cl100k_base.
	// Protocol framing and other tokenizer differences fall within the tolerance.
	modelTraceContextExpectedTokens int64 = 6
	modelTraceContextTolerance      int64 = 24
	modelTraceContextOutputTokens         = 256
)

type modelTraceContextStatus string

const (
	modelTraceContextNoAnomaly   modelTraceContextStatus = "no_anomaly"
	modelTraceContextSuspected   modelTraceContextStatus = "suspected_injection"
	modelTraceContextUnavailable modelTraceContextStatus = "unavailable"
)

// ModelTraceContextCheck compares a standalone user message with reported usage.
// It is diagnostic only and is not an input to the fingerprint detector.
type ModelTraceContextCheck struct {
	Status              modelTraceContextStatus `json:"status"`
	ExpectedInputTokens *int64                  `json:"expected_input_tokens"`
	ReportedInputTokens *int64                  `json:"reported_input_tokens"`
	Delta               *int64                  `json:"delta"`
	LogID               *string                 `json:"log_id"`
}

func (s *Service) runModelTraceContextCheck(ctx context.Context, run *modelTraceExecutionContext) ModelTraceContextCheck {
	executed, err := s.executeModelTraceRequest(ctx, run, modelTraceContextPrompt, modelTraceContextOutputTokens)
	result := evaluateModelTraceContextCheck(executed.attempt)
	if err == nil {
		result.LogID = s.emitModelTraceRequestLog(run, executed)
	}
	return result
}

func evaluateModelTraceContextCheck(attempt execution.AttemptResult) ModelTraceContextCheck {
	expected := modelTraceContextExpectedTokens
	result := ModelTraceContextCheck{Status: modelTraceContextUnavailable, ExpectedInputTokens: &expected}
	if attempt.Error != nil || attempt.StatusCode < http.StatusOK || attempt.StatusCode >= http.StatusMultipleChoices ||
		attempt.Usage == nil || attempt.Usage.Normalized.State != usage.StateComplete {
		return result
	}
	tokens := attempt.Usage.Normalized.Tokens
	// Include cached input but not output. An absent/invalid input count is not a
	// clean result. Only return counts representable exactly in the browser.
	tokens.Output = 0
	reported, ok := usage.CheckedTotal(tokens)
	if !ok || reported == 0 || reported > 1<<53-1 {
		return result
	}
	delta := reported - expected
	result.ReportedInputTokens, result.Delta = &reported, &delta
	result.Status = modelTraceContextNoAnomaly
	if delta > modelTraceContextTolerance {
		result.Status = modelTraceContextSuspected
	}
	return result
}
