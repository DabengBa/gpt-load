package gateway

import (
	"gpt-load/internal/codexrecovery"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
)

// responsesRecoverySignal only inspects an actual upstream rejection. Error
// summaries on successful/local results are not authority to rewrite history.
func responsesRecoverySignal(result UpstreamResult) codexrecovery.Signal {
	if result.DispatchState == execution.DispatchLocal || result.DispatchState == execution.DispatchNotSent ||
		(result.StatusCode < 400 && !result.ProviderErrorBeforeCommit && result.ExecutionError == nil) {
		return ""
	}
	if result.ExecutionError != nil {
		// Upstream HTTP 400 request rejections may carry client origin: that
		// describes fault attribution, not a locally generated response.
		if result.ExecutionError.Kind == execution.ErrorKindInternal || result.ExecutionError.OriginHint == execution.ErrorOriginInternal {
			return ""
		}
		if signal := codexrecovery.ClassifyParts(result.ExecutionError.Code, result.ExecutionError.Summary); signal != "" {
			return signal
		}
	}
	if signal := codexrecovery.Classify(string(result.ClassificationBody)); signal != "" {
		return signal
	}
	return codexrecovery.Classify(string(result.Body))
}

// A recognized rejection in an HTTP-200 SSE envelope is not a generic stream
// interruption/host fault. The explicit rejection proves processing was refused.
// Qualify only the existing generic buffered SSE rule; never override a safety,
// cooldown, or other policy decision. The wire status/evidence remain untouched.
func responsesBufferedRecoveryDecision(result UpstreamResult, decision health.Decision, eligible bool) health.Decision {
	if eligible && result.BufferedStream && result.HTTPCommitted && result.Committed &&
		!result.PayloadReleased && result.ClientVisibleBytes > 0 && result.DispatchState == execution.DispatchMaybeSent &&
		decision.RuleID == "buffered_stream.retry_before_release" && responsesRecoverySignal(result) != "" {
		decision.Category = health.FailureCategoryClientError
		decision.Scope = execution.ErrorScopeRequest
		decision.Effect = health.EffectNone
		decision.RuleID = "responses.history_rejected_before_release"
	}
	return decision
}
