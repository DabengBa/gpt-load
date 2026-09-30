package gateway

import (
	"sync"
	"time"

	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/state"
	"gpt-load/internal/usage"
)

type providerFeedbackMeasurement struct {
	mu                      sync.Mutex
	now                     func() time.Time
	attemptStartedAt        time.Time
	firstPayloadAt          time.Time
	completedAt             time.Time
	downstreamWriteDuration time.Duration
	firstDownstreamDuration time.Duration
}

func newProviderFeedbackMeasurement(now func() time.Time, attemptStartedAt time.Time) *providerFeedbackMeasurement {
	if now == nil {
		now = time.Now
	}
	return &providerFeedbackMeasurement{now: now, attemptStartedAt: attemptStartedAt}
}

func (measurement *providerFeedbackMeasurement) observeProviderPayload() {
	if measurement == nil {
		return
	}
	at := measurement.now()
	measurement.mu.Lock()
	if measurement.firstPayloadAt.IsZero() {
		measurement.firstPayloadAt = at
		measurement.firstDownstreamDuration = measurement.downstreamWriteDuration
	}
	measurement.mu.Unlock()
}

func (measurement *providerFeedbackMeasurement) beginDownstreamWrite(buffered bool) time.Time {
	if measurement == nil || buffered {
		return time.Time{}
	}
	return measurement.now()
}

func (measurement *providerFeedbackMeasurement) endDownstreamWrite(startedAt time.Time) {
	if measurement == nil || startedAt.IsZero() {
		return
	}
	duration := measurement.now().Sub(startedAt)
	if duration <= 0 {
		return
	}
	measurement.mu.Lock()
	measurement.downstreamWriteDuration += duration
	measurement.mu.Unlock()
}

func (measurement *providerFeedbackMeasurement) complete() {
	if measurement == nil {
		return
	}
	measurement.mu.Lock()
	if measurement.completedAt.IsZero() {
		measurement.completedAt = measurement.now()
	}
	measurement.mu.Unlock()
}

func (measurement *providerFeedbackMeasurement) observation() (*int64, *int64) {
	if measurement == nil {
		return nil, nil
	}
	measurement.mu.Lock()
	defer measurement.mu.Unlock()
	if measurement.firstPayloadAt.IsZero() {
		return nil, nil
	}
	firstResponseMS := (measurement.firstPayloadAt.Sub(measurement.attemptStartedAt) - measurement.firstDownstreamDuration).Milliseconds()
	if firstResponseMS < 0 {
		return nil, nil
	}
	firstResponse := new(firstResponseMS)
	if measurement.completedAt.IsZero() {
		return firstResponse, nil
	}
	generation := measurement.completedAt.Sub(measurement.firstPayloadAt) -
		(measurement.downstreamWriteDuration - measurement.firstDownstreamDuration)
	generationMS := generation.Milliseconds()
	if generationMS <= 0 {
		return firstResponse, nil
	}
	return firstResponse, new(generationMS)
}

func providerFeedbackForAttempt(
	result UpstreamResult,
	decision health.Decision,
	measurement *providerFeedbackMeasurement,
	stream bool,
) health.Feedback {
	eligible := stream && result.DispatchState == execution.DispatchMaybeSent && result.Err == nil &&
		decision.Category == health.FailureCategoryOK && decision.Origin == execution.ErrorOriginUpstream
	if eligible && result.Feedback.Status != health.FeedbackStatusUnassessed {
		return result.Feedback
	}
	firstResponseMS, generationMS := measurement.observation()
	var outputTokens *int64
	if (result.Usage.State == usage.StateComplete || result.Usage.State == usage.StatePartial) &&
		result.Usage.Tokens.Output > 0 {
		outputTokens = new(result.Usage.Tokens.Output)
	}
	return health.ClassifyFeedback(health.FeedbackObservation{
		Eligible:        eligible,
		ProviderFailed:  isProviderFeedbackFailure(result, decision),
		FirstResponseMs: firstResponseMS,
		GenerationMs:    generationMS,
		OutputTokens:    outputTokens,
	})
}

func isProviderFeedbackFailure(result UpstreamResult, decision health.Decision) bool {
	if result.DispatchState != execution.DispatchMaybeSent ||
		decision.Origin != execution.ErrorOriginUpstream || decision.Category == health.FailureCategoryOK {
		return false
	}
	switch result.Stream.EndReason {
	case StreamEndDownstreamWriteFailure, StreamEndClientCanceled, StreamEndServerShutdown:
		return false
	}
	if evidence := result.ExecutionError; evidence != nil {
		if evidence.OriginHint != "" && evidence.OriginHint != execution.ErrorOriginUpstream {
			return false
		}
		switch evidence.Kind {
		case execution.ErrorKindCanceled, execution.ErrorKindInvalidRequest,
			execution.ErrorKindConversionUnsupported, execution.ErrorKindInternal:
			return false
		}
		if evidence.OriginHint == execution.ErrorOriginUpstream &&
			(evidence.Kind == execution.ErrorKindProvider || evidence.Kind == execution.ErrorKindTransport ||
				evidence.Kind == execution.ErrorKindTimeout || evidence.ScopeHint == execution.ErrorScopeCredential ||
				evidence.ScopeHint == execution.ErrorScopeModel || evidence.ScopeHint == execution.ErrorScopeGroup) {
			return true
		}
	}
	switch result.Stream.EndReason {
	case StreamEndSSEError, StreamEndUpstreamTerminated, StreamEndIdleTimeout,
		StreamEndProviderIncomplete, StreamEndUpstreamFailure, StreamEndUpstreamProtocolError:
		return true
	}
	if result.ProviderErrorBeforeCommit {
		return true
	}
	if decision.Category == health.FailureCategoryConversionUnsupported {
		return false
	}
	if decision.Category == health.FailureCategoryClientError {
		return result.ExecutionError != nil && result.ExecutionError.Kind == execution.ErrorKindProvider
	}
	return true
}

func isSuccessfulPerformanceFault(result UpstreamResult, decision health.Decision) bool {
	return result.Feedback.Status == health.FeedbackStatusFaulty &&
		result.Feedback.Reason != "upstream_failure" &&
		result.DispatchState == execution.DispatchMaybeSent &&
		decision.Category == health.FailureCategoryOK &&
		decision.Origin == execution.ErrorOriginUpstream &&
		result.Err == nil
}

func (handler *Handler) applyPerformanceFeedbackFailure(
	group state.GroupView,
	credentialID uint,
	entryID string,
	statusCode int,
	attemptNow time.Time,
) {
	handler.applyGroupDecisionEffectForEntry(
		group,
		credentialID,
		0,
		entryID,
		health.Decision{
			Category: health.FailureCategoryUpstreamHostError,
			Origin:   execution.ErrorOriginUpstream,
			Scope:    execution.ErrorScopeCredential,
			Retry:    health.RetryNone,
			Effect:   health.EffectRecordCredentialFailure,
			RuleID:   "feedback.performance_fault",
		},
		statusCode,
		attemptNow,
	)
}
