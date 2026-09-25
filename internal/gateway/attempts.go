package gateway

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"

	"gpt-load/internal/accessquota"
	"gpt-load/internal/connection"
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/platform/utils"
	"gpt-load/internal/reasoning"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
	subscriptionruntime "gpt-load/internal/subscription/runtime"
)

// deferredAttempt records one retryable outcome that may still become the
// terminal answer when every candidate is exhausted.
type deferredAttempt struct {
	result        UpstreamResult
	decision      health.Decision
	upstreamModel string
	attemptIndex  int
}

// deferredAttempts groups the four fallback buckets with their terminal
// precedence: provider error > response > transport > conversion.
type deferredAttempts struct {
	response      *deferredAttempt
	transport     *deferredAttempt
	conversion    *deferredAttempt
	providerError *deferredAttempt
}

// credentialRefreshRetry preserves one explicit provider-auth retry on the
// same selected credential.
type credentialRefreshRetry struct {
	selection scheduler.Selection
	ref       state.CredentialRef
}

// preparedRequest is the per-candidate rewrite result: an applied override
// body plus refreshed attempt-local observations.
type preparedRequest struct {
	request               *dialect.ParsedRequest
	observations          dialect.RequestMetadata
	observationsAvailable bool
	err                   error
}

// preparedRequestCache is the per-request prepared-request cache. It keeps at
// most one candidate's rewrite so switching groups releases the old body
// instead of accumulating full request bodies across retries.
type preparedRequestCache struct {
	dialect       dialect.Dialect
	parsed        *dialect.ParsedRequest
	metadata      dialect.RequestMetadata
	externalModel string
	operation     execution.Operation

	groupID       uint
	entryID       string
	upstreamModel string
	cached        *preparedRequest
}

func (cache *preparedRequestCache) get(selection scheduler.Selection) preparedRequest {
	upstreamModel := optionalModelValue(selection.UpstreamModelID)
	if cache.cached != nil &&
		cache.groupID == selection.GroupID &&
		cache.entryID == selection.EntryID &&
		cache.upstreamModel == upstreamModel {
		return *cache.cached
	}
	cache.cached = nil
	cache.groupID = selection.GroupID
	cache.entryID = selection.EntryID
	cache.upstreamModel = upstreamModel
	prepared := preparedRequest{
		request: cache.parsed, observations: cache.metadata, observationsAvailable: true,
	}
	if cache.operation == execution.OperationWebSearch {
		return prepared
	}
	body, applied, err := selection.Group.ParameterOverrides.Apply(
		cache.dialect.Protocol(),
		cache.metadata.Operation,
		cache.externalModel,
		cache.parsed.Body,
	)
	if err != nil {
		prepared.err = err
		cache.cached = &prepared
		return prepared
	}
	entryEffort := ""
	for _, model := range selection.Group.Models {
		if selection.EntryID != "" && model.EntryID == selection.EntryID {
			entryEffort = model.ReasoningEffort
			break
		}
	}
	effort, source, resolveErr := reasoning.ResolveEffort(entryEffort, cache.metadata.Reasoning.Effort)
	if resolveErr != nil {
		prepared.err = resolveErr
		cache.cached = &prepared
		return prepared
	}
	if dialect.SupportsReasoningEffortOverride(cache.dialect.Protocol(), cache.metadata.Operation) &&
		effort != "" && source != reasoning.EffortSourceClient {
		var effortApplied bool
		body, effortApplied, err = dialect.SetReasoningEffort(body, effort, cache.dialect.Protocol())
		if err != nil {
			prepared.err = err
			cache.cached = &prepared
			return prepared
		}
		applied = applied || effortApplied
	}
	if !applied {
		cache.cached = &prepared
		return prepared
	}
	if int64(len(body)) > maxRequestBodyBytes {
		prepared.err = errRequestTooLarge
		cache.cached = &prepared
		return prepared
	}
	request := &dialect.ParsedRequest{
		Method: cache.parsed.Method, Path: cache.parsed.Path, RawQuery: cache.parsed.RawQuery,
		Header: cache.parsed.Header.Clone(), Body: body,
	}
	// Routing metadata is frozen from the original client request. Refresh only
	// attempt-local observations, and never block dispatch when observation fails.
	// An unavailable observation stays unknown instead of reusing client values.
	prepared.observations = dialect.RequestMetadata{ObserveUsage: cache.metadata.ObserveUsage}
	prepared.observationsAvailable = false
	if metadata, inspectErr := cache.dialect.InspectRequest(request); inspectErr == nil {
		prepared.observations = dialect.RequestMetadata{
			ObserveUsage:     cache.metadata.ObserveUsage,
			PricingMode:      metadata.PricingMode,
			UsageDiagnostics: metadata.UsageDiagnostics,
			Reasoning:        metadata.Reasoning,
		}
		prepared.observationsAvailable = true
	}
	prepared.request = request
	cache.cached = &prepared
	return prepared
}

// attemptLoop carries the mutable orchestration state for one request's
// candidate iteration. Frozen inputs come from the admission dispatch; the
// mutable fields replace what used to be free-floating closure variables.
type attemptLoop struct {
	handler    *Handler
	ginContext *gin.Context
	iterator   *scheduler.Iterator
	limit      int

	dispatch       *requestDispatch
	recorder       *requestRecorder
	quotaAdmission *requestAccessQuotaAdmission

	requestContext         context.Context
	stream                 bool
	operation              execution.Operation
	buffered               bool
	bufferedReplayEligible bool
	bufferedTimeoutFrozen  bool

	attemptSequence  int
	forwardAttempts  int
	lastAttemptIndex int
	deferred         deferredAttempts

	refreshRetry          *credentialRefreshRetry
	authRefreshReplayUsed bool

	prepared *preparedRequestCache

	loggedOverrideFailures   map[uint]struct{}
	parameterOverrideFailure *reason
}

func newAttemptLoop(
	handler *Handler,
	ginContext *gin.Context,
	iterator *scheduler.Iterator,
	limit int,
	dispatch *requestDispatch,
	recorder *requestRecorder,
) *attemptLoop {
	requestContext := ginContext.Request.Context()
	buffered := dispatch.metadata.Stream && dispatch.streamMode == streamDeliveryBuffered
	loop := &attemptLoop{
		handler:                handler,
		ginContext:             ginContext,
		iterator:               iterator,
		limit:                  limit,
		dispatch:               dispatch,
		recorder:               recorder,
		quotaAdmission:         dispatch.quotaAdmission,
		requestContext:         requestContext,
		stream:                 dispatch.metadata.Stream,
		operation:              dispatch.metadata.Operation,
		buffered:               buffered,
		lastAttemptIndex:       -1,
		loggedOverrideFailures: make(map[uint]struct{}),
	}
	if buffered {
		loop.bufferedReplayEligible = bufferedStreamReplayEligible(dispatch.parsed, dispatch.dialect)
		loop.requestContext = context.WithValue(
			requestContext,
			bufferedStreamSessionContextKey{},
			&bufferedStreamSession{},
		)
	}
	loop.prepared = &preparedRequestCache{
		dialect:       dispatch.dialect,
		parsed:        dispatch.parsed,
		metadata:      dispatch.metadata,
		externalModel: dispatch.model,
		operation:     loop.operation,
	}
	return loop
}

func (loop *attemptLoop) decisionContextForSelection(selection scheduler.Selection) health.DecisionContext {
	defaultRateLimitCooldown := fixedCooldown
	credentialRefreshable := false
	if connection.Normalize(selection.Group.ConnectionType) == connection.Subscription {
		defaultRateLimitCooldown = subscriptionruntime.DefaultRefreshFailureCooldown
		credentialRefreshable = true
	}
	method := ""
	if loop.dispatch.parsed != nil {
		method = loop.dispatch.parsed.Method
	}
	return health.DecisionContext{
		DefaultRateLimitCooldown: defaultRateLimitCooldown,
		CredentialRefreshable:    credentialRefreshable,
		Method:                   method,
		Operation:                loop.operation,
		BufferedReplayEligible:   loop.buffered && loop.bufferedReplayEligible,
	}
}

func (loop *attemptLoop) recordCandidatePreparationFailure(
	selection scheduler.Selection,
	attemptObservations dialect.RequestMetadata,
	attemptObservationsAvailable bool,
	code string,
	summary string,
	scope execution.ErrorScope,
) bool {
	handler := loop.handler
	recorder := loop.recorder
	ginContext := loop.ginContext
	loop.attemptSequence++
	if loop.attemptSequence == 1 {
		if loop.dispatch.metadata.PreviousResponseID != "" {
			recorder.setContinuityHit(true)
		} else if loop.dispatch.affinity.preferredCredentialID != 0 &&
			selection.CredentialID == loop.dispatch.affinity.preferredCredentialID {
			recorder.setAffinityHit(true)
		}
	}
	updateDebugHeaders(ginContext.Writer.Header(), selection.Group.Name, loop.attemptSequence)
	if recorder != nil {
		recorder.freezeNextAttemptPricing(
			handler.freezeAttemptPricing(
				selection,
				attemptObservations,
				attemptObservationsAvailable,
				recorder.accessKeyMultiplier,
			),
		)
	}
	attemptStarted := recorder.beforeForward()
	evidence := execution.ErrorEvidence{
		Kind: execution.ErrorKindInternal, OriginHint: execution.ErrorOriginInternal,
		ScopeHint: scope, Code: code, Summary: summary,
	}
	result := UpstreamResult{
		Err:           fmt.Errorf("%w: candidate preparation failed", ErrUpstreamProtocol),
		DispatchState: execution.DispatchNotSent, ExecutionError: &evidence,
		ErrorSummary: summary,
	}
	capture := captureFromContext(ginContext)
	executionRequestID := "untracked"
	if recorder != nil && recorder.requestID != "" {
		executionRequestID = recorder.requestID
	}
	capture.finishPreparationAttempt(CaptureAttemptMetadata{
		AttemptID: executionRequestID + ":" + strconv.Itoa(loop.attemptSequence),
		Sequence:  uint32(loop.attemptSequence),
		Fields: map[string]string{
			"kind":  "forward",
			"phase": "preparation",
			"error": summary,
		},
	}, result.Err)
	attemptCompleted := time.Time{}
	if recorder != nil {
		attemptCompleted = recorder.now()
	}
	attemptNow := handler.now()
	decision := judgeUpstreamResult(
		result,
		attemptNow,
		loop.decisionContextForSelection(selection),
	)
	recordedAttempt := recorder.appendDecisionAttempt(
		selection, result, decision, code, summary, attemptStarted, attemptCompleted,
	)
	loop.lastAttemptIndex = recordedAttempt
	handler.applyGroupDecisionEffectForEntry(selection.Group, selection.CredentialID, 0, selection.EntryID, decision, 0, attemptNow)
	if decision.Effect == health.EffectSkipGroup {
		loop.iterator.SkipGroup(selection.GroupID)
	}
	loop.deferred.transport = &deferredAttempt{
		result: result, decision: decision,
		upstreamModel: optionalModelValue(selection.UpstreamModelID),
		attemptIndex:  recordedAttempt,
	}
	if decision.Retry != health.RetryNone {
		recorder.retryIfAnotherForward(recordedAttempt)
	}
	return decision.Retry != health.RetryNone
}

func (loop *attemptLoop) recordCaptureCandidatePreparationFailure(groupName string, code string, summary string) {
	recorder := loop.recorder
	ginContext := loop.ginContext
	loop.attemptSequence++
	updateDebugHeaders(ginContext.Writer.Header(), groupName, loop.attemptSequence)
	capture := captureFromContext(ginContext)
	executionRequestID := "untracked"
	if recorder != nil && recorder.requestID != "" {
		executionRequestID = recorder.requestID
	}
	evidence := execution.ErrorEvidence{
		Kind: execution.ErrorKindInternal, OriginHint: execution.ErrorOriginInternal,
		ScopeHint: execution.ErrorScopeRequest, Code: code, Summary: summary,
	}
	result := UpstreamResult{
		Err:           fmt.Errorf("%w: candidate preparation failed", ErrUpstreamProtocol),
		DispatchState: execution.DispatchNotSent, ExecutionError: &evidence,
		ErrorSummary: summary,
	}
	capture.finishPreparationAttempt(CaptureAttemptMetadata{
		AttemptID: executionRequestID + ":" + strconv.Itoa(loop.attemptSequence),
		Sequence:  uint32(loop.attemptSequence),
		Fields: map[string]string{
			"kind":  "forward",
			"phase": "preparation",
			"error": summary,
		},
	}, result.Err)
}

// candidateStep is the outcome of pulling the next candidate: yield a
// selection, skip this iteration (mimicking the original `continue`), or stop
// the loop entirely (mimicking `break`).
type candidateStep uint8

const (
	candidateYield candidateStep = iota
	candidateSkip
	candidateExhausted
)

// nextCandidate returns the next selection, honoring the credential-refresh
// replay slot before consulting the iterator.
func (loop *attemptLoop) nextCandidate() (scheduler.Selection, state.CredentialRef, bool, candidateStep) {
	handler := loop.handler
	if loop.refreshRetry != nil {
		selection := loop.refreshRetry.selection
		currentRef, exists := handler.registry.CredentialRef(selection.CredentialID)
		if !exists || currentRef.GroupID != selection.GroupID ||
			currentRef.IdentityGeneration != loop.refreshRetry.ref.IdentityGeneration {
			loop.refreshRetry = nil
			return scheduler.Selection{}, state.CredentialRef{}, false, candidateSkip
		}
		loop.authRefreshReplayUsed = true
		forceRefresh := currentRef.Version <= loop.refreshRetry.ref.Version
		loop.refreshRetry = nil
		return selection, currentRef, forceRefresh, candidateYield
	}
	selection, err := loop.iterator.Next()
	if err != nil {
		return scheduler.Selection{}, state.CredentialRef{}, false, candidateExhausted
	}
	candidateRef, allowed := loop.dispatch.allowedCredentialRefs[selection.CredentialID]
	if !allowed || candidateRef.GroupID != selection.GroupID {
		return scheduler.Selection{}, state.CredentialRef{}, false, candidateSkip
	}
	return selection, candidateRef, false, candidateYield
}

// admitQuota performs the deferred access-quota admission that must happen
// after a candidate is selected but before the first forward. It returns false
// when the request was already completed on the wire.
func (loop *attemptLoop) admitQuota() bool {
	handler := loop.handler
	ginContext := loop.ginContext
	quotaAdmission := loop.quotaAdmission
	if quotaAdmission == nil || quotaAdmission.admitted || handler.accessQuota == nil {
		return true
	}
	var ticket accessquota.Ticket
	var decision accessquota.Decision
	if quotaAdmission.snapshot == nil {
		ticket, decision = handler.accessQuota.Admit(
			quotaAdmission.accessKeyID,
			handler.quotaNow(),
		)
	} else {
		var current bool
		ticket, decision, current = handler.admitAccessQuotaForSnapshot(
			quotaAdmission.snapshot,
			quotaAdmission.accessKeyID,
			handler.quotaNow(),
		)
		if !current {
			handler.completeConfigurationChanged(ginContext, loop.recorder)
			return false
		}
	}
	if !decision.Allowed {
		handler.completeAccessQuotaReason(ginContext, loop.recorder, decision)
		return false
	}
	quotaAdmission.ticket = ticket
	quotaAdmission.admitted = true
	return true
}

func (loop *attemptLoop) run() {
	handler := loop.handler
	ginContext := loop.ginContext
	recorder := loop.recorder
	dispatch := loop.dispatch

	for loop.forwardAttempts < loop.limit {
		if ginContext.Request.Context().Err() != nil || loop.requestContext.Err() != nil {
			recorder.completeCanceled(loop.requestContext, 0, loop.lastAttemptIndex)
			return
		}
		selection, ref, forceCredentialRefresh, step := loop.nextCandidate()
		if step == candidateExhausted {
			break
		}
		if step == candidateSkip {
			continue
		}
		encrypted, active := handler.registry.ActiveEncryptedCredentialDataIfMatch(ref)
		if !active {
			continue
		}
		prepared := loop.prepared.get(selection)
		if prepared.err != nil {
			code := "parameter_override_failed"
			summary := "Parameter override could not be applied."
			if errors.Is(prepared.err, errRequestTooLarge) {
				code = "parameter_override_request_too_large"
				summary = "Parameter override produced a request that is too large."
				if loop.parameterOverrideFailure == nil {
					loop.parameterOverrideFailure = &reasonRequestTooLarge
				}
			} else {
				if loop.parameterOverrideFailure == nil {
					loop.parameterOverrideFailure = &reasonParameterOverrideUnavailable
				}
			}
			loop.recordCaptureCandidatePreparationFailure(selection.Group.Name, code, summary)
			if _, logged := loop.loggedOverrideFailures[selection.GroupID]; !logged {
				utils.LogPlaneBestEffort(
					handler.logger,
					logrus.WarnLevel,
					utils.LogPlaneData,
					logrus.Fields{"group_id": selection.GroupID, "reason": "apply_failed"},
					"Parameter override failed for an upstream Group",
				)
				loop.loggedOverrideFailures[selection.GroupID] = struct{}{}
			}
			loop.iterator.SkipGroup(selection.GroupID)
			continue
		}
		attemptObservations := prepared.observations
		attemptObservationsAvailable := prepared.observationsAvailable
		decryptedCredential, err := handler.encryption.Decrypt(encrypted)
		if err != nil {
			if !loop.recordCandidatePreparationFailure(
				selection,
				attemptObservations,
				attemptObservationsAvailable,
				"credential_decrypt_failed",
				"Stored credential could not be decrypted.",
				execution.ErrorScopeCredential,
			) {
				break
			}
			continue
		}
		normalizedCredential, err := normalizeChannelCredential(
			handler.channels,
			handler.subscriptions,
			selection.ChannelID,
			selection.Group.ConnectionType,
			decryptedCredential,
		)
		if err != nil {
			if !loop.recordCandidatePreparationFailure(
				selection,
				attemptObservations,
				attemptObservationsAvailable,
				"credential_normalization_failed",
				"Stored credential could not be prepared.",
				execution.ErrorScopeCredential,
			) {
				break
			}
			continue
		}
		effectiveProxy, proxyFingerprint, err := resolveAttemptProxy(
			handler.encryption,
			selection.Group.Proxy,
		)
		if err != nil {
			if !loop.recordCandidatePreparationFailure(
				selection,
				attemptObservations,
				attemptObservationsAvailable,
				"group_proxy_prepare_failed",
				"Group proxy configuration could not be prepared.",
				execution.ErrorScopeGroup,
			) {
				break
			}
			continue
		}
		if !loop.admitQuota() {
			return
		}

		loop.attemptSequence++
		loop.forwardAttempts++
		if loop.attemptSequence == 1 {
			if dispatch.metadata.PreviousResponseID != "" {
				recorder.setContinuityHit(true)
			} else if dispatch.affinity.preferredCredentialID != 0 &&
				selection.CredentialID == dispatch.affinity.preferredCredentialID {
				recorder.setAffinityHit(true)
			}
		}
		updateDebugHeaders(ginContext.Writer.Header(), selection.Group.Name, loop.attemptSequence)
		if loop.buffered && !loop.bufferedTimeoutFrozen {
			loop.bufferedTimeoutFrozen = true
			if timeout := selection.Group.Timeouts.Request; timeout > 0 {
				var cancel context.CancelFunc
				loop.requestContext, cancel = context.WithTimeout(loop.requestContext, timeout)
				defer cancel()
			}
		}
		executionRequestID := "untracked"
		if recorder != nil && recorder.requestID != "" {
			executionRequestID = recorder.requestID
		}
		input := ForwardInput{
			attemptTarget: attemptTarget{
				Group:             selection.Group,
				APIKey:            normalizedCredential.apiKey,
				CredentialSecrets: normalizedCredential.secrets,
				ChannelID:         string(selection.ChannelID),
				UpstreamModelID:   optionalModelValue(selection.UpstreamModelID),
				TargetConfig:      selection.ResolvedTarget.TargetConfig,
				Credential: execution.NewCredentialSnapshot(
					selection.CredentialID,
					ref.Version,
					ref.IdentityGeneration,
					normalizedCredential.payload,
				),
				Proxy:                  effectiveProxy,
				ProxyFingerprint:       proxyFingerprint,
				ForceCredentialRefresh: forceCredentialRefresh,
			},
			attemptIdentity: attemptIdentity{
				RequestID:       executionRequestID,
				AttemptID:       executionRequestID + ":" + strconv.Itoa(loop.attemptSequence),
				AttemptSequence: uint32(loop.attemptSequence),
			},
			preparedRoute: preparedRoute{
				Request:                  prepared.request,
				ExternalModel:            dispatch.model,
				Operation:                dispatch.metadata.Operation,
				RouteRequirement:         dispatch.metadata.RouteRequirement,
				ResponsesStorePreference: dispatch.metadata.ResponsesStorePreference,
				ResponsesStoreDowngraded: selection.ResponsesStoreDowngraded,
				RouteMode:                execution.RouteMode(selection.RouteMode),
			},
			attemptEffects: attemptEffects{
				Dialect:        dispatch.dialect,
				ClientProtocol: dispatch.dialect.Protocol(),
				ObserveUsage:   attemptObservations.ObserveUsage,
				BufferedStream: loop.buffered,
				ContinuityKey:  dispatch.affinity.continuityKey,
				OnResponse:     handler.responseBindingObserver(recorder.accessKeyID, selection, ref, prepared.request),
				OnFirstResponse: func() {
					recorder.recordFirstResponse()
				},
			},
		}
		if recorder != nil {
			recorder.freezeNextAttemptPricing(
				handler.freezeAttemptPricing(
					selection,
					attemptObservations,
					attemptObservationsAvailable,
					recorder.accessKeyMultiplier,
				),
			)
		}
		attemptStarted := recorder.beforeForward()
		var result UpstreamResult
		capture := captureFromContext(ginContext)
		captureAttempt, captureObserver, attemptContext, captureStarted := capture.beginForward(
			loop.requestContext,
			input,
		)
		dispatch.affinity.markBoundAttempt(selection, ref)
		func() {
			if loop.stream {
				result = handler.forwarder.ForwardStream(attemptContext, input, ginContext.Writer)
			} else {
				result = handler.forwarder.Forward(attemptContext, input)
			}
		}()
		result = capture.normalizeAndFinishForwardOwned(captureAttempt, captureObserver, result, captureStarted)
		if !loop.stream && result.HasResponse() && !result.ProviderErrorBeforeCommit &&
			result.DispatchState != execution.DispatchLocal &&
			result.StatusCode >= http.StatusOK && result.StatusCode < http.StatusMultipleChoices {
			if input.OnResponse != nil {
				if err := input.OnResponse(result.Body); err != nil {
					result.Err = err
					result.ExecutionError = &execution.ErrorEvidence{
						Kind: execution.ErrorKindInternal, OriginHint: execution.ErrorOriginInternal,
						ScopeHint: execution.ErrorScopeRequest, Code: "response_binding_conflict",
						Summary: "Response ownership could not be recorded.", ReplaySafety: execution.ReplaySafetyUnknown,
					}
				}
			}
		}
		attemptCompleted := time.Time{}
		if recorder != nil {
			attemptCompleted = recorder.now()
		}
		requestCanceled := ginContext.Request.Context().Err() != nil || loop.requestContext.Err() != nil
		if loop.stream && result.Committed && result.Stream.EndReason == StreamEndNone {
			if result.Err == nil && result.ExecutionError != nil {
				// 上游以错误结束但没走到任何流级终止观测（buffered 心跳已提交、payload 未释放）。
				// 记成正常结束会丢掉执行层证据：裁决退化成「无证据」、请求日志把失败记成成功。
				result.Stream = streamTerminalObservationWithResponseID(
					StreamEndUpstreamFailure,
					result.Stream.ResponseID,
				)
			} else {
				result.Stream = prioritizeStreamObservation(
					ginContext.Request.Context(),
					result.Err,
					result.Stream,
				)
			}
		}
		attemptNow := handler.now()
		resultForDecision := result
		if requestCanceled {
			if loop.requestContext.Err() != nil {
				resultForDecision.Err = loop.requestContext.Err()
			} else {
				resultForDecision.Err = ginContext.Request.Context().Err()
			}
		}
		decision := judgeUpstreamResult(
			resultForDecision,
			attemptNow,
			loop.decisionContextForSelection(selection),
		)
		dispatch.affinity.markBoundProviderFailure(selection, ref, resultForDecision.DispatchState, decision)
		if loop.stream && result.BufferedStream && result.HTTPCommitted && !result.PayloadReleased {
			// A heartbeat has committed HTTP, but no provider payload is visible;
			// this is the only committed state in which an explicit buffered retry
			// may proceed. Committed is intentionally left true.
			recordedAttempt := recorder.recordStreamAttempt(
				selection, normalizedCredential.secrets, result, decision, attemptStarted, attemptCompleted,
			)
			loop.lastAttemptIndex = recordedAttempt
			handler.applyGroupDecisionEffectForEntry(
				selection.Group, selection.CredentialID,
				refreshCooldownCredentialVersion(result, ref.Version),
				selection.EntryID, decision, result.StatusCode, attemptNow,
			)
			if decision.Effect == health.EffectSkipGroup {
				loop.iterator.SkipGroup(selection.GroupID)
			}
			if requestCanceled {
				recorder.completeCanceled(ginContext.Request.Context(), 0, recordedAttempt)
				return
			}
			if decision.Retry != health.RetryNone && loop.forwardAttempts < loop.limit {
				recorder.retryIfAnotherForward(recordedAttempt)
				continue
			}
			recorder.completeStream(result, optionalModelValue(selection.UpstreamModelID), recordedAttempt)
			if result.Stream.EndReason != StreamEndDownstreamWriteFailure &&
				result.Stream.EndReason != StreamEndClientCanceled &&
				result.Stream.EndReason != StreamEndServerShutdown &&
				!errors.Is(result.Err, context.Canceled) {
				if err := handler.writeBufferedStreamError(ginContext, dispatch.dialect.Protocol(), result.Stream.ResponseID); err != nil {
					handler.completeWriteTerminal(ginContext, recorder, result.StatusCode)
				}
			}
			return
		}
		if result.Committed {
			if recorder != nil {
				recordedAttempt := recorder.recordStreamAttempt(
					selection, normalizedCredential.secrets, result, decision, attemptStarted, attemptCompleted,
				)
				recorder.completeStream(result, optionalModelValue(selection.UpstreamModelID), recordedAttempt)
			}
			handler.applyGroupDecisionEffectForEntry(
				selection.Group,
				selection.CredentialID,
				0,
				selection.EntryID,
				decision,
				result.StatusCode,
				attemptNow,
			)
			if loop.stream && result.Stream.EndReason == StreamEndCleanEOF {
				handler.recordCredentialSuccess(selection.CredentialID, attemptNow)
				handler.recordEntrySuccess(selection.GroupID, selection.EntryID, selection.CredentialID)
				if dispatch.metadata.PreviousResponseID == "" {
					handler.recordAffinitySuccess(ginContext.Request.Context(), dispatch.affinity, selection, ref)
				}
			}
			return
		}
		if requestCanceled {
			if recorder != nil {
				recordedAttempt := recorder.recordAttempt(
					selection,
					normalizedCredential.secrets,
					result,
					decision,
					attemptStarted,
					attemptCompleted,
				)
				recorder.completeCanceled(ginContext.Request.Context(), 0, recordedAttempt)
			}
			return
		}
		if loop.operation != execution.OperationWebSearch && !loop.stream && result.DispatchState != execution.DispatchLocal &&
			!result.ProviderErrorBeforeCommit && result.HasResponse() &&
			result.StatusCode >= http.StatusOK &&
			result.StatusCode < http.StatusMultipleChoices &&
			!(result.ExecutionError != nil && isUpstreamDiagnosticErrorCode(result.ExecutionError.Code)) {
			handler.recordCredentialSuccess(selection.CredentialID, attemptNow)
			handler.recordEntrySuccess(selection.GroupID, selection.EntryID, selection.CredentialID)
		}
		recordedAttempt := recorder.recordAttempt(
			selection, normalizedCredential.secrets, result, decision, attemptStarted, attemptCompleted,
		)
		loop.lastAttemptIndex = recordedAttempt
		handler.applyGroupDecisionEffectForEntry(
			selection.Group,
			selection.CredentialID,
			refreshCooldownCredentialVersion(result, ref.Version),
			selection.EntryID,
			decision,
			result.StatusCode,
			attemptNow,
		)
		if decision.Retry == health.RetryRefreshCredential &&
			!loop.authRefreshReplayUsed && loop.forwardAttempts < loop.limit {
			loop.refreshRetry = &credentialRefreshRetry{selection: selection, ref: ref}
		}
		if decision.Effect == health.EffectSkipGroup {
			loop.iterator.SkipGroup(selection.GroupID)
		}
		if result.StatusCode >= http.StatusContinue && result.StatusCode < http.StatusOK {
			recorder.completeTransport(
				reasonUpstreamProtocol,
				optionalModelValue(selection.UpstreamModelID),
				recordedAttempt,
			)
			if err := handler.writeReason(ginContext, reasonUpstreamProtocol); err != nil {
				handler.completeWriteTerminal(ginContext, recorder, reasonUpstreamProtocol.Status)
			}
			return
		}
		if result.ProviderErrorBeforeCommit {
			if decision.Retry != health.RetryNone {
				loop.deferred.providerError = &deferredAttempt{
					result:        result,
					decision:      decision,
					upstreamModel: optionalModelValue(selection.UpstreamModelID),
					attemptIndex:  recordedAttempt,
				}
				recorder.retryIfAnotherForward(recordedAttempt)
				continue
			}
			recorder.completeProviderError(
				result,
				optionalModelValue(selection.UpstreamModelID),
				recordedAttempt,
			)
			if err := handler.writeReason(ginContext, reasonUpstreamProtocol); err != nil {
				handler.completeWriteTerminal(ginContext, recorder, reasonUpstreamProtocol.Status)
			}
			return
		}
		if result.HasResponse() {
			loop.deferred.response = &deferredAttempt{
				result: result, decision: decision, upstreamModel: optionalModelValue(selection.UpstreamModelID), attemptIndex: recordedAttempt,
			}
			if decision.Retry != health.RetryNone {
				recorder.retryIfAnotherForward(recordedAttempt)
				continue
			}
			recorder.completeResponse(result, decision, optionalModelValue(selection.UpstreamModelID), recordedAttempt)
			if err := handler.writeUpstreamResponse(ginContext, result); err != nil {
				handler.completeWriteTerminal(ginContext, recorder, result.StatusCode)
				return
			}
			if result.DispatchState != execution.DispatchLocal &&
				result.StatusCode >= http.StatusOK && result.StatusCode < http.StatusMultipleChoices &&
				dispatch.metadata.PreviousResponseID == "" &&
				!(result.ExecutionError != nil && isUpstreamDiagnosticErrorCode(result.ExecutionError.Code)) {
				handler.recordAffinitySuccess(ginContext.Request.Context(), dispatch.affinity, selection, ref)
			}
			return
		}
		if errors.Is(result.Err, context.Canceled) {
			recorder.completeCanceled(ginContext.Request.Context(), 0, recordedAttempt)
			return
		}
		if decision.Retry != health.RetryNone {
			deferred := &deferredAttempt{
				result: result, decision: decision, upstreamModel: optionalModelValue(selection.UpstreamModelID), attemptIndex: recordedAttempt,
			}
			if isConversionUnsupportedResult(result) {
				loop.deferred.conversion = deferred
			} else {
				loop.deferred.transport = deferred
			}
			recorder.retryIfAnotherForward(recordedAttempt)
			continue
		}
		value := transportReason(result)
		recorder.completeTransport(value, optionalModelValue(selection.UpstreamModelID), recordedAttempt)
		if err := handler.writeReason(ginContext, value); err != nil {
			handler.completeWriteTerminal(ginContext, recorder, value.Status)
		}
		return
	}

	loop.completeExhausted()
}

// completeExhausted resolves the terminal answer once every candidate is
// spent: deferred fallbacks in precedence order, then the static rejection
// reasons.
func (loop *attemptLoop) completeExhausted() {
	handler := loop.handler
	ginContext := loop.ginContext
	recorder := loop.recorder

	if loop.deferred.providerError != nil {
		recorder.completeProviderError(
			loop.deferred.providerError.result,
			loop.deferred.providerError.upstreamModel,
			loop.deferred.providerError.attemptIndex,
		)
		if err := handler.writeReason(ginContext, reasonUpstreamProtocol); err != nil {
			handler.completeWriteTerminal(ginContext, recorder, reasonUpstreamProtocol.Status)
		}
		return
	}
	if loop.deferred.response != nil {
		recorder.completeResponse(
			loop.deferred.response.result,
			loop.deferred.response.decision,
			loop.deferred.response.upstreamModel,
			loop.deferred.response.attemptIndex,
		)
		if err := handler.writeUpstreamResponse(ginContext, loop.deferred.response.result); err != nil {
			handler.completeWriteTerminal(
				ginContext,
				recorder,
				loop.deferred.response.result.StatusCode,
			)
			return
		}
		return
	}
	if loop.deferred.transport != nil {
		value := transportReason(loop.deferred.transport.result)
		recorder.completeTransport(value, loop.deferred.transport.upstreamModel, loop.deferred.transport.attemptIndex)
		if err := handler.writeReason(ginContext, value); err != nil {
			handler.completeWriteTerminal(ginContext, recorder, value.Status)
		}
		return
	}
	if loop.deferred.conversion != nil {
		value := reasonProtocolConversionUnsupported
		recorder.completeTransport(value, loop.deferred.conversion.upstreamModel, loop.deferred.conversion.attemptIndex)
		if err := handler.writeReason(ginContext, value); err != nil {
			handler.completeWriteTerminal(ginContext, recorder, value.Status)
		}
		return
	}
	if loop.iterator.StaticReason() == scheduler.ReasonModelRequiredByFilter {
		handler.completeReason(ginContext, recorder, reasonModelRequiredByFilter)
		return
	}
	if loop.parameterOverrideFailure != nil && loop.lastAttemptIndex < 0 {
		handler.completeReason(ginContext, recorder, *loop.parameterOverrideFailure)
		return
	}
	handler.completeReason(ginContext, recorder, reasonNoCandidate)
}
