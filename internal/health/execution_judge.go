package health

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	"gpt-load/internal/execution"
)

// ExecutionAttempt is the provider-neutral evidence used for one health and
// retry decision after an executor call.
type ExecutionAttempt struct {
	DispatchState       execution.DispatchState
	StatusCode          int
	Header              http.Header
	Evidence            *execution.ErrorEvidence
	DownstreamCommitted bool
	// HTTPCommitted is true once the gateway has sent response headers or a
	// heartbeat. It is deliberately independent from payload release.
	HTTPCommitted bool
	// PayloadReleased is irreversible: a true value means real upstream
	// payload bytes were made visible to the client.
	PayloadReleased bool
	// ClientVisibleBytes includes heartbeats and released payload bytes.
	ClientVisibleBytes int64
	// BufferedStream identifies the explicit buffered-stream retry contract.
	BufferedStream bool
	DownstreamErr  error
	Now            time.Time
}

// JudgeExecution decides the complete GPT-Load retry and runtime effect for one
// executor result.
func JudgeExecution(attempt ExecutionAttempt, decisionContext DecisionContext) Decision {
	decisionContext = normalizeDecisionContext(decisionContext)
	if errors.Is(attempt.DownstreamErr, context.Canceled) ||
		errors.Is(attempt.DownstreamErr, context.DeadlineExceeded) {
		return decision(
			FailureCategoryDownstreamCancel,
			execution.ErrorOriginDownstream,
			execution.ErrorScopeRequest,
			RetryNone,
			EffectNone,
			"safety.downstream_cancel",
		)
	}
	if attempt.DownstreamErr != nil {
		return decision(
			FailureCategoryAmbiguous,
			execution.ErrorOriginDownstream,
			execution.ErrorScopeRequest,
			RetryNone,
			EffectNone,
			"safety.downstream_write_failed",
		)
	}
	if attempt.DownstreamCommitted && attempt.Evidence == nil && isSuccessStatus(attempt.StatusCode) {
		return decision(
			FailureCategoryOK,
			execution.ErrorOriginUpstream,
			"",
			RetryNone,
			EffectNone,
			"success.committed_stream",
		)
	}

	if attempt.Evidence == nil {
		if attempt.DispatchState.Valid() && isSuccessStatus(attempt.StatusCode) {
			return decision(
				FailureCategoryOK,
				execution.ErrorOriginUpstream,
				"",
				RetryNone,
				EffectNone,
				"success.upstream_response",
			)
		}
		if transientUpstreamStatus(attempt.statusCode()) {
			// The upstream answered with a status the gateway treats as
			// transient but exposed no classifiable evidence. The candidate is
			// suspected, not proven: count the failure against its credential so
			// a candidate that keeps answering this way reaches the blacklist
			// threshold and is recovered by the validation probe, and switch
			// candidate for this request instead of failing it.
			return constrainCommittedDecision(decision(
				FailureCategoryAmbiguous,
				originForDispatch(attempt.DispatchState),
				execution.ErrorScopeCredential,
				RetryNextCandidate,
				EffectRecordCredentialFailure,
				"fallback.missing_evidence_retry",
			), attempt)
		}
		return decision(
			FailureCategoryAmbiguous,
			originForDispatch(attempt.DispatchState),
			"",
			RetryNone,
			EffectNone,
			"fallback.missing_evidence",
		)
	}

	if attempt.Evidence.Kind == execution.ErrorKindCanceled {
		return decision(
			FailureCategoryDownstreamCancel,
			execution.ErrorOriginDownstream,
			execution.ErrorScopeRequest,
			RetryNone,
			EffectNone,
			"safety.execution_canceled",
		)
	}
	if attempt.DispatchState == execution.DispatchNotSent {
		if result, ok := candidatePreparationDecision(attempt.Evidence); ok {
			return result
		}
		if attempt.Evidence.Hint == execution.FailureHintRefreshRequired {
			return authenticationDecision(attempt, decisionContext)
		}
		if attempt.Evidence.Hint == execution.FailureHintRefreshUnavailable {
			return refreshTemporarilyUnavailableDecision(attempt, decisionContext)
		}
		if attempt.Evidence.Hint == execution.FailureHintReauthorizationRequired {
			return decision(
				FailureCategoryAuthenticationRequired,
				execution.ErrorOriginUpstream,
				execution.ErrorScopeCredential,
				RetryNextCandidate,
				EffectNone,
				"auth.reauthorization_required",
			)
		}
		switch attempt.Evidence.Kind {
		case execution.ErrorKindTransport, execution.ErrorKindTimeout:
			return decision(
				FailureCategoryUpstreamHostError,
				execution.ErrorOriginUpstream,
				execution.ErrorScopeGroup,
				RetryNextCandidate,
				EffectSkipGroup,
				"transport.not_sent",
			)
		case execution.ErrorKindConversionUnsupported:
			return decision(
				FailureCategoryConversionUnsupported,
				execution.ErrorOriginInternal,
				execution.ErrorScopeGroup,
				RetryNextCandidate,
				EffectSkipGroup,
				"conversion.not_sent",
			)
		case execution.ErrorKindInvalidRequest:
			return decision(
				FailureCategoryClientError,
				originForEvidence(attempt.Evidence),
				scopeOrDefault(attempt.Evidence.ScopeHint, execution.ErrorScopeRequest),
				RetryNone,
				EffectNone,
				"request.invalid_before_dispatch",
			)
		default:
			return decision(
				FailureCategoryAmbiguous,
				originForEvidence(attempt.Evidence),
				attempt.Evidence.ScopeHint,
				RetryNone,
				EffectNone,
				"fallback.not_sent",
			)
		}
	}
	if attempt.DispatchState != execution.DispatchMaybeSent {
		return decision(
			FailureCategoryAmbiguous,
			originForEvidence(attempt.Evidence),
			attempt.Evidence.ScopeHint,
			RetryNone,
			EffectNone,
			"fallback.invalid_dispatch_state",
		)
	}
	if !attempt.ResponseStarted() &&
		(attempt.Evidence.Kind == execution.ErrorKindTransport || attempt.Evidence.Kind == execution.ErrorKindTimeout) {
		if attempt.BufferedStream && attempt.HTTPCommitted && !attempt.PayloadReleased &&
			attempt.ClientVisibleBytes > 0 && decisionContext.BufferedReplayEligible {
			return decision(
				FailureCategoryAmbiguous,
				execution.ErrorOriginUpstream,
				execution.ErrorScopeGroup,
				RetryNextCandidate,
				EffectSkipGroup,
				"buffered_stream.retry_before_release_unknown",
			)
		}
		return decision(
			FailureCategoryAmbiguous,
			execution.ErrorOriginUpstream,
			attempt.Evidence.ScopeHint,
			RetryNone,
			EffectNone,
			"transport.outcome_unknown",
		)
	}

	if attempt.StatusCode >= http.StatusContinue && attempt.StatusCode < http.StatusOK {
		return decision(
			FailureCategoryAmbiguous,
			execution.ErrorOriginUpstream,
			execution.ErrorScopeRequest,
			RetryNone,
			EffectNone,
			"safety.final_informational_response",
		)
	}
	category := classifyExecutionEvidence(attempt)
	result := decisionForExecutionCategory(category, attempt, decisionContext)
	result = bufferedStreamRetryDecision(result, attempt, decisionContext)
	if attempt.Evidence.ReplaySafety == execution.ReplaySafetyUnknown &&
		result.RuleID == "fallback.ambiguous" {
		result.RuleID = "safety.replay_unknown"
	}
	result = constrainOperationReplay(result, attempt, decisionContext)
	return constrainCommittedDecision(result, attempt)
}

func refreshTemporarilyUnavailableDecision(attempt ExecutionAttempt, decisionContext DecisionContext) Decision {
	category := FailureCategoryUpstreamHostError
	ruleID := RuleID("auth.refresh_temporarily_unavailable")
	if attempt.Evidence.StatusCode == http.StatusTooManyRequests {
		category = FailureCategoryRateLimited
		ruleID = "auth.refresh_rate_limited"
	}
	cooldown := decisionContext.DefaultRateLimitCooldown
	if cooldown <= 0 {
		cooldown = time.Minute
	}
	if attempt.Evidence.RetryAfter > 0 {
		cooldown = attempt.Evidence.RetryAfter
	}
	result := decision(
		category,
		execution.ErrorOriginUpstream,
		execution.ErrorScopeCredential,
		RetryNextCandidate,
		EffectCooldownCredential,
		ruleID,
	)
	result.CooldownUntil = attempt.Now.Add(cooldown)
	return result
}

func candidatePreparationDecision(evidence *execution.ErrorEvidence) (Decision, bool) {
	if evidence == nil || evidence.OriginHint != execution.ErrorOriginInternal {
		return Decision{}, false
	}
	effect := EffectNone
	switch evidence.Code {
	case "credential_decrypt_failed",
		"credential_normalization_failed":
		if evidence.ScopeHint != execution.ErrorScopeCredential {
			return Decision{}, false
		}
	case "group_proxy_prepare_failed":
		if evidence.ScopeHint != execution.ErrorScopeGroup {
			return Decision{}, false
		}
		effect = EffectSkipGroup
	default:
		return Decision{}, false
	}
	return decision(
		FailureCategoryAmbiguous,
		execution.ErrorOriginInternal,
		evidence.ScopeHint,
		RetryNextCandidate,
		effect,
		RuleID("candidate.")+RuleID(evidence.Code),
	), true
}

func (attempt ExecutionAttempt) ResponseStarted() bool {
	return attempt.StatusCode != 0 || attempt.Evidence != nil && attempt.Evidence.StatusCode != 0
}

// statusCode resolves the upstream HTTP status of one attempt, preferring the
// status the gateway recorded over the one carried by the evidence.
func (attempt ExecutionAttempt) statusCode() int {
	if attempt.StatusCode != 0 {
		return attempt.StatusCode
	}
	if attempt.Evidence != nil {
		return attempt.Evidence.StatusCode
	}
	return 0
}

func classifyExecutionEvidence(attempt ExecutionAttempt) FailureCategory {
	statusCode := attempt.statusCode()
	markers := ""
	if attempt.Evidence != nil {
		if statusCode == http.StatusUnauthorized && attempt.Evidence.ReplaySafety == execution.ReplaySafetyUnknown &&
			attempt.Evidence.Hint == "" {
			return FailureCategoryAmbiguous
		}
		if structuredUnsupportedModelClientError(statusCode, attempt.Evidence) {
			return FailureCategoryClientError
		}
		switch attempt.Evidence.Hint {
		case execution.FailureHintInvalidCredential:
			return FailureCategoryInvalidKey
		case execution.FailureHintRefreshRequired,
			execution.FailureHintReauthorizationRequired:
			return FailureCategoryAuthenticationRequired
		case execution.FailureHintRateLimited:
			return FailureCategoryRateLimited
		case execution.FailureHintRequestRejected:
			return FailureCategoryClientError
		case execution.FailureHintCandidateUnavailable:
			return FailureCategoryModelUnavailable
		case execution.FailureHintModelUnavailable:
			return FailureCategoryModelUnavailable
		case execution.FailureHintHostError:
			return FailureCategoryUpstreamHostError
		}
		markers = strings.ToLower(strings.Join([]string{
			attempt.Evidence.Type,
			attempt.Evidence.Code,
			attempt.Evidence.Summary,
		}, " "))
	}

	switch {
	case statusCode == http.StatusTooManyRequests || containsAny(markers,
		"rate_limit", "rate limit", "too_many_requests", "quota_exceeded",
		"resource_exhausted", "throttl"):
		return FailureCategoryRateLimited
	case statusCode == http.StatusUnauthorized:
		return FailureCategoryInvalidKey
	case containsAny(markers,
		"model_not_found", "model not found", "model_not_available",
		"model unavailable", "deployment_not_found", "unsupported_model",
		"model_access_denied", "model permission denied", "model not authorized"):
		return FailureCategoryModelUnavailable
	case containsAny(markers,
		"invalid_api_key", "api_key_invalid", "authentication_error",
		"authentication failed", "invalid credential", "api key not valid"):
		return FailureCategoryInvalidKey
	case statusCode >= http.StatusInternalServerError && statusCode <= 599:
		return FailureCategoryUpstreamHostError
	case statusCode >= http.StatusBadRequest && statusCode <= 499:
		return FailureCategoryClientError
	case attempt.Evidence != nil && attempt.Evidence.Kind == execution.ErrorKindInvalidRequest:
		return FailureCategoryClientError
	default:
		return FailureCategoryAmbiguous
	}
}

func structuredUnsupportedModelClientError(statusCode int, evidence *execution.ErrorEvidence) bool {
	if evidence == nil || statusCode != http.StatusBadRequest {
		return false
	}
	for _, value := range []string{evidence.Type, evidence.Code} {
		normalized := strings.ToLower(strings.TrimSpace(value))
		if normalized == "unsupported_model" || normalized == "unsupported-model" {
			return true
		}
	}
	return false
}

func decisionForExecutionCategory(
	category FailureCategory,
	attempt ExecutionAttempt,
	decisionContext DecisionContext,
) Decision {
	if transientCapacity, ok := transientCapacityDecision(attempt); ok {
		return transientCapacity
	}
	origin := originForEvidence(attempt.Evidence)
	scope := attempt.Evidence.ScopeHint
	if attempt.Evidence != nil && attempt.Evidence.Hint == execution.FailureHintCandidateUnavailable {
		if attempt.Evidence.ReplaySafety == execution.ReplaySafetyUnknown {
			return decision(category, origin, scope, RetryNone, EffectNone, "safety.replay_unknown")
		}
		return decision(
			category,
			origin,
			scopeOrDefault(scope, execution.ErrorScopeModel),
			RetryNextCandidate,
			EffectNone,
			"candidate.unavailable",
		)
	}
	switch category {
	case FailureCategoryRateLimited:
		return rateLimitDecision(attempt, decisionContext)
	case FailureCategoryModelUnavailable:
		retry := retryUnlessExplicitlyUnknown(attempt.Evidence)
		if decisionContext.Operation.ReplayPolicy() == execution.ReplayPolicyRequireRejectedBeforeProcessing {
			ruleID := RuleID("images.model_unavailable")
			if decisionContext.Operation == execution.OperationEmbeddingsCreate {
				ruleID = "embeddings.model_unavailable"
			}
			if decisionContext.Operation == execution.OperationRerank {
				ruleID = "rerank.model_unavailable"
			}
			return decision(
				category,
				origin,
				scopeOrDefault(scope, execution.ErrorScopeModel),
				retry,
				EffectNone,
				ruleID,
			)
		}
		result := decision(
			category,
			origin,
			scopeOrDefault(scope, execution.ErrorScopeModel),
			retry,
			EffectCooldownCredential,
			"model.unavailable",
		)
		result.CooldownUntil = attempt.Now.Add(time.Hour)
		return result
	case FailureCategoryInvalidKey:
		return decision(
			category,
			origin,
			scopeOrDefault(scope, execution.ErrorScopeCredential),
			retryUnlessExplicitlyUnknown(attempt.Evidence),
			EffectRecordCredentialFailure,
			"auth.invalid_credential",
		)
	case FailureCategoryAuthenticationRequired:
		return authenticationDecision(attempt, decisionContext)
	case FailureCategoryUpstreamHostError:
		retry := RetryNone
		ruleID := RuleID("upstream.host_error.replay_unsafe")
		if attempt.Evidence.ReplaySafety == execution.ReplaySafetyRejectedBeforeProcessing {
			retry = RetryNextCandidate
			ruleID = "upstream.host_error.rejected_before_processing"
		} else if attempt.Evidence.ReplaySafety != execution.ReplaySafetyUnknown &&
			requestMayReplayAfterResponse(decisionContext) {
			retry = RetryNextCandidate
			ruleID = "upstream.host_error.read_only"
		}
		return decision(
			category,
			origin,
			scopeOrDefault(scope, execution.ErrorScopeGroup),
			retry,
			EffectSkipGroup,
			ruleID,
		)
	case FailureCategoryConversionUnsupported:
		return decision(
			category,
			execution.ErrorOriginInternal,
			scopeOrDefault(scope, execution.ErrorScopeGroup),
			RetryNone,
			EffectSkipGroup,
			"conversion.unsupported",
		)
	case FailureCategoryClientError:
		return decision(
			category,
			origin,
			scopeOrDefault(scope, execution.ErrorScopeRequest),
			RetryNone,
			EffectNone,
			"fallback.http_client_error",
		)
	case FailureCategoryOK:
		return decision(category, origin, scope, RetryNone, EffectNone, "success.upstream_response")
	default:
		return decision(category, origin, scope, RetryNone, EffectNone, ambiguousRuleID(attempt.Evidence))
	}
}

func transientCapacityDecision(attempt ExecutionAttempt) (Decision, bool) {
	evidence := attempt.Evidence
	if evidence == nil || evidence.ReplaySafety != execution.ReplaySafetyRejectedBeforeProcessing ||
		originForEvidence(evidence) != execution.ErrorOriginUpstream ||
		(evidence.Kind != execution.ErrorKindHTTP && evidence.Kind != execution.ErrorKindProvider) {
		return Decision{}, false
	}
	codeValue := strings.ToLower(strings.TrimSpace(evidence.Code))
	overloaded := codeValue == "server_is_overloaded"
	rateLimited := codeValue == "rate_limit_exceeded"
	if overloaded == rateLimited {
		return Decision{}, false
	}
	category := FailureCategoryUpstreamHostError
	if rateLimited {
		category = FailureCategoryRateLimited
	}
	return decision(
		category,
		execution.ErrorOriginUpstream,
		evidence.ScopeHint,
		RetryNextCandidate,
		EffectNone,
		"candidate.transient_capacity",
	), true
}

func bufferedStreamRetryDecision(
	result Decision,
	attempt ExecutionAttempt,
	decisionContext DecisionContext,
) Decision {
	if !attempt.BufferedStream || !attempt.HTTPCommitted || attempt.PayloadReleased ||
		attempt.ClientVisibleBytes <= 0 || !decisionContext.BufferedReplayEligible ||
		attempt.Evidence == nil || result.Retry != RetryNone {
		return result
	}
	// Only failures produced by the buffered collector may use this gate. In
	// particular, a legal provider incomplete result is not a transient retry.
	switch attempt.Evidence.Code {
	case "upstream_stream_terminated", "upstream_stream_idle_timeout", "upstream_protocol_error":
		return decision(
			FailureCategoryAmbiguous,
			execution.ErrorOriginUpstream,
			execution.ErrorScopeGroup,
			RetryNextCandidate,
			EffectSkipGroup,
			"buffered_stream.retry_before_release",
		)
	}
	if transientUpstreamStatus(attempt.statusCode()) {
		// The upstream answered with a status the gateway treats as transient and
		// released no payload: this attempt may still switch candidate. Category,
		// scope and effect stay with the evidence, so a classified host error keeps
		// skipping its group instead of degrading to an unclassified fallback.
		result.Retry = RetryNextCandidate
		result.RuleID = "buffered_stream.retry_before_release_upstream_status"
		return result
	}
	// 上游以 2xx 状态开始流但未释放任何 payload 就失败（SSE 错误事件、连接中断等）：
	// 心跳已提交、payload 未释放，换候选是安全的。这类证据没有可重试的状态码，
	// 只能靠证据码识别（run finding: upstream 200 后无内容被误判为终局）。
	// 客户端错误（4xx）仍保持终局：请求本身无效，换候选无意义。
	if isSuccessStatus(attempt.statusCode()) {
		switch attempt.Evidence.Code {
		case "upstream_error", "upstream_sse_error":
			return decision(
				FailureCategoryAmbiguous,
				execution.ErrorOriginUpstream,
				execution.ErrorScopeGroup,
				RetryNextCandidate,
				EffectSkipGroup,
				"buffered_stream.retry_before_release",
			)
		}
	}
	return result
}

func constrainOperationReplay(
	result Decision,
	attempt ExecutionAttempt,
	decisionContext DecisionContext,
) Decision {
	if result.Retry == RetryNone ||
		attempt.DispatchState != execution.DispatchMaybeSent ||
		decisionContext.Operation.ReplayPolicy() != execution.ReplayPolicyRequireRejectedBeforeProcessing {
		return result
	}
	if attempt.Evidence != nil &&
		attempt.Evidence.ReplaySafety == execution.ReplaySafetyRejectedBeforeProcessing {
		return result
	}
	result.Retry = RetryNone
	result.RuleID = "safety.operation_replay_unsafe"
	return result
}

func ambiguousRuleID(evidence *execution.ErrorEvidence) RuleID {
	if evidence == nil {
		return "fallback.ambiguous"
	}
	switch evidence.Code {
	case "upstream_sse_error":
		return "stream.provider_error"
	case "upstream_stream_terminated":
		return "stream.upstream_terminated"
	case "upstream_protocol_error":
		return "stream.protocol_error"
	case "upstream_stream_idle_timeout":
		return "stream.idle_timeout"
	case "upstream_response_incomplete":
		return "stream.provider_incomplete"
	default:
		return "fallback.ambiguous"
	}
}

// transientUpstreamStatus reports whether an upstream status is transient:
// either the attempt carried no classifiable evidence and the status alone
// grants a candidate switch, or the attempt failed with that status before any
// payload release. Anything else stays final because no candidate was proven bad.
func transientUpstreamStatus(statusCode int) bool {
	if statusCode == http.StatusRequestTimeout || statusCode == http.StatusTooManyRequests {
		return true
	}
	return statusCode >= http.StatusInternalServerError && statusCode < 600
}

func normalizeDecisionContext(value DecisionContext) DecisionContext {
	if value.DefaultRateLimitCooldown <= 0 {
		value.DefaultRateLimitCooldown = time.Minute
	}
	return value
}

func decision(
	category FailureCategory,
	origin execution.ErrorOrigin,
	scope execution.ErrorScope,
	retry RetryDirective,
	effect Effect,
	ruleID RuleID,
) Decision {
	return Decision{
		Category: category,
		Origin:   origin,
		Scope:    scope,
		Retry:    retry,
		Effect:   effect,
		RuleID:   ruleID,
	}
}

func rateLimitDecision(attempt ExecutionAttempt, decisionContext DecisionContext) Decision {
	scope := attempt.Evidence.ScopeHint
	retry := retryUnlessExplicitlyUnknown(attempt.Evidence)
	if scope == execution.ErrorScopeRequest || scope == execution.ErrorScopeModel {
		return decision(
			FailureCategoryRateLimited,
			originForEvidence(attempt.Evidence),
			scope,
			RetryNone,
			EffectNone,
			"rate_limit.scoped",
		)
	}
	effect := EffectCooldownCredential
	ruleID := RuleID("legacy.http_429_credential_cooldown")
	if scope == execution.ErrorScopeCredential {
		ruleID = "rate_limit.credential.default_cooldown"
	}
	result := decision(
		FailureCategoryRateLimited,
		originForEvidence(attempt.Evidence),
		scope,
		retry,
		effect,
		ruleID,
	)
	if attempt.Evidence.RetryAfter > 0 {
		result.CooldownUntil = attempt.Now.Add(attempt.Evidence.RetryAfter)
		result.RuleID = "rate_limit.retry_after"
		return result
	}
	header := attempt.Header
	if len(header) == 0 {
		header = attempt.Evidence.Header
	}
	if until, ok := ParseRateLimitReset(header, attempt.Now); ok {
		result.CooldownUntil = until
		result.RuleID = "rate_limit.reset_header"
		return result
	}
	result.CooldownUntil = attempt.Now.Add(decisionContext.DefaultRateLimitCooldown)
	return result
}

func authenticationDecision(attempt ExecutionAttempt, decisionContext DecisionContext) Decision {
	hint := attempt.Evidence.Hint
	if attempt.DispatchState == execution.DispatchMaybeSent &&
		attempt.Evidence.ReplaySafety != execution.ReplaySafetyRejectedBeforeProcessing {
		return decision(
			FailureCategoryAuthenticationRequired,
			execution.ErrorOriginUpstream,
			execution.ErrorScopeCredential,
			RetryNone,
			EffectNone,
			"auth.replay_unsafe",
		)
	}
	if hint == execution.FailureHintRefreshRequired {
		if decisionContext.CredentialRefreshable &&
			attempt.Evidence.ReplaySafety == execution.ReplaySafetyRejectedBeforeProcessing {
			return decision(
				FailureCategoryAuthenticationRequired,
				execution.ErrorOriginUpstream,
				execution.ErrorScopeCredential,
				RetryRefreshCredential,
				EffectNone,
				"auth.refresh_required",
			)
		}
		return decision(
			FailureCategoryAuthenticationRequired,
			execution.ErrorOriginUpstream,
			execution.ErrorScopeCredential,
			RetryNextCandidate,
			EffectNone,
			"auth.refresh_unavailable",
		)
	}
	return decision(
		FailureCategoryAuthenticationRequired,
		execution.ErrorOriginUpstream,
		execution.ErrorScopeCredential,
		RetryNextCandidate,
		EffectNone,
		"auth.reauthorization_required",
	)
}

func constrainCommittedDecision(result Decision, attempt ExecutionAttempt) Decision {
	if attempt.BufferedStream && attempt.HTTPCommitted && !attempt.PayloadReleased &&
		attempt.ClientVisibleBytes > 0 {
		// A heartbeat commits HTTP but does not expose provider payload. Keep the
		// explicit buffered retry decision instead of treating Committed as final.
		return result
	}
	if !attempt.DownstreamCommitted && !attempt.PayloadReleased {
		return result
	}
	originalRetry := result.Retry
	originalEffect := result.Effect
	result.Retry = RetryNone
	switch result.Effect {
	case EffectCooldownCredential, EffectRecordCredentialFailure:
		if !trustedCommittedCredentialEffect(result, attempt.Evidence) {
			result.Effect = EffectNone
			result.CooldownUntil = time.Time{}
		}
	default:
		result.Effect = EffectNone
		result.CooldownUntil = time.Time{}
	}
	if result.Category == FailureCategoryRateLimited && result.Effect == EffectCooldownCredential {
		result.RuleID = "rate_limit.credential.committed"
	} else if originalRetry != result.Retry || originalEffect != result.Effect {
		result.RuleID = "safety.committed"
	}
	return result
}

func trustedCommittedCredentialEffect(
	result Decision,
	evidence *execution.ErrorEvidence,
) bool {
	if evidence == nil || result.Scope != execution.ErrorScopeCredential {
		return false
	}
	if evidence.ScopeHint == execution.ErrorScopeCredential {
		return true
	}
	return result.Effect == EffectRecordCredentialFailure &&
		evidence.Hint == execution.FailureHintInvalidCredential
}

func retryUnlessExplicitlyUnknown(evidence *execution.ErrorEvidence) RetryDirective {
	if evidence != nil && evidence.ReplaySafety == execution.ReplaySafetyUnknown {
		return RetryNone
	}
	return RetryNextCandidate
}

func requestMayReplayAfterResponse(value DecisionContext) bool {
	if strings.EqualFold(value.Method, http.MethodGet) || strings.EqualFold(value.Method, http.MethodHead) {
		return true
	}
	switch value.Operation {
	case execution.OperationResponsesRetrieve,
		execution.OperationResponsesInputItems,
		execution.OperationResponsesInputTokens,
		execution.OperationCountTokens,
		execution.OperationListModels:
		return true
	default:
		return false
	}
}

func originForEvidence(evidence *execution.ErrorEvidence) execution.ErrorOrigin {
	if evidence == nil {
		return ""
	}
	if evidence.OriginHint != "" {
		return evidence.OriginHint
	}
	switch evidence.Kind {
	case execution.ErrorKindTransport,
		execution.ErrorKindTimeout,
		execution.ErrorKindHTTP,
		execution.ErrorKindProvider:
		return execution.ErrorOriginUpstream
	case execution.ErrorKindCanceled:
		return execution.ErrorOriginDownstream
	case execution.ErrorKindConversionUnsupported,
		execution.ErrorKindInternal:
		return execution.ErrorOriginInternal
	case execution.ErrorKindInvalidRequest:
		return execution.ErrorOriginClient
	default:
		return ""
	}
}

func originForDispatch(dispatch execution.DispatchState) execution.ErrorOrigin {
	if dispatch == execution.DispatchLocal {
		return execution.ErrorOriginInternal
	}
	if dispatch == execution.DispatchMaybeSent || dispatch == execution.DispatchNotSent {
		return execution.ErrorOriginUpstream
	}
	return ""
}

func scopeOrDefault(value, fallback execution.ErrorScope) execution.ErrorScope {
	if value != "" {
		return value
	}
	return fallback
}

func isSuccessStatus(statusCode int) bool {
	return statusCode >= http.StatusOK && statusCode < http.StatusMultipleChoices
}

func containsAny(value string, markers ...string) bool {
	for _, marker := range markers {
		if strings.Contains(value, marker) {
			return true
		}
	}
	return false
}
