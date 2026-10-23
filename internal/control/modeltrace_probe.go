package control

/*
ModelTrace 指纹检测把「模型归因」做成一次正式模型使用：

1. 从独立部署的 ModelTrace 取得固定数量的挑战；
2. 每条挑战经分组实际路由、协议转换和凭据执行一次真实模型请求；
3. 每条挑战无论成败都写一条正式 request log（operation = modeltrace），
   usage 按正式请求规则计量；
4. 把所有可用输出一次性提交 /api/analyze 得到归因结果。

归因结果只作为独立诊断事实返回，不参与健康、路由、熔断、重试或亲和决策；
本文件也不调用任何健康恢复入口。
*/

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
	"gpt-load/internal/platform/utils"
	"gpt-load/internal/pricing"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

// modelTraceChallengeOutputTokens bounds one challenge answer. The bank's
// largest expected count is 332 integers, which fits comfortably below this
// budget while staying a bounded, code-owned value.
const modelTraceChallengeOutputTokens = 2048
const modelTraceRunDeadline = 6 * time.Minute

var modelTraceInFlight sync.Map
var modelTraceSlots = make(chan struct{}, 4)

type ModelTraceChallengeOutcome string

const (
	modelTraceChallengeSucceeded ModelTraceChallengeOutcome = "succeeded"
	modelTraceChallengeFailed    ModelTraceChallengeOutcome = "failed"
)

// ModelTraceFailure is the bounded reason vocabulary of a detection that could
// not produce an attribution. It is separate from per-challenge outcomes.
type ModelTraceFailure string

const (
	modelTraceFailureNoOutputs           ModelTraceFailure = "no_usable_outputs"
	modelTraceFailureAnalysisRejected    ModelTraceFailure = "analysis_rejected"
	modelTraceFailureAnalysisUnavailable ModelTraceFailure = "analysis_unavailable"
)

// ModelTraceRequest targets exactly one (group, model) route.
type ModelTraceRequest struct {
	GroupID uint   `json:"group_id"`
	Model   string `json:"model"`
}

// ModelTraceServiceDescription is the safe identity of the configured
// deployment. The token is never part of it.
type ModelTraceServiceDescription struct {
	BaseURL string `json:"base_url"`
	Enabled bool   `json:"enabled"`
}

// ModelTraceChallengeResponse is one executed challenge.
type ModelTraceChallengeResponse struct {
	ID            string                     `json:"id"`
	ExpectedCount int                        `json:"expected_count"`
	Outcome       ModelTraceChallengeOutcome `json:"outcome"`
	Reason        *ProbeReason               `json:"reason"`
	LogID         *string                    `json:"log_id"`
	LatencyMS     *int64                     `json:"latency_ms"`
	StatusCode    *int                       `json:"status_code"`
}

// ModelTraceProbeResponse is the complete detection result. Analyses and
// challenges are independent: a failed analysis keeps every challenge row.
type ModelTraceProbeResponse struct {
	GroupID    uint                          `json:"group_id"`
	GroupName  string                        `json:"group_name"`
	Model      string                        `json:"model"`
	Service    *ModelTraceServiceDescription `json:"service"`
	Challenges []ModelTraceChallengeResponse `json:"challenges"`
	Analysis   *modelTraceAnalysis           `json:"analysis"`
	Failure    *ModelTraceFailure            `json:"failure"`
	TestedAtMS int64                         `json:"tested_at_ms"`
}

func validateModelTraceRequest(request ModelTraceRequest) error {
	if request.GroupID == 0 || !validUsageModel(request.Model) {
		return app_errors.ErrBadRequest
	}
	return nil
}

type modelTraceChallengeTarget struct {
	protocol        protocol.Protocol
	routeMode       channel.RouteMode
	model           string
	maxOutputTokens int
}

// modelTraceTarget resolves the declared generative route of one model. It
// shares the channel's probe contract, so a channel without a generated-text
// route reports the target as unsupported instead of burning a request.
func modelTraceTarget(group state.GroupView, model string) (modelTraceChallengeTarget, bool) {
	if group.ChannelID == "" || group.ResolvedTarget.ChannelID != group.ChannelID ||
		!group.ResolvedTarget.ProviderKind.Valid() || !groupHasModel(group, model) {
		return modelTraceChallengeTarget{}, false
	}
	clientProtocol, routeMode, ok := group.ResolvedTarget.ModelTraceRoute(model)
	if !ok {
		return modelTraceChallengeTarget{}, false
	}
	return modelTraceChallengeTarget{
		protocol:        clientProtocol,
		routeMode:       routeMode,
		model:           model,
		maxOutputTokens: modelTraceChallengeOutputTokens,
	}, true
}

// RunModelTraceProbe executes one fingerprint detection. Configuration and
// target problems fail the call before any upstream request; everything after
// the first dispatch is reported per challenge.
func (s *Service) RunModelTraceProbe(
	ctx context.Context,
	request ModelTraceRequest,
) (ModelTraceProbeResponse, error) {
	if err := validateModelTraceRequest(request); err != nil {
		return ModelTraceProbeResponse{}, err
	}
	key := fmt.Sprintf("%d:%s", request.GroupID, request.Model)
	if _, loaded := modelTraceInFlight.LoadOrStore(key, struct{}{}); loaded {
		return ModelTraceProbeResponse{}, app_errors.ErrDuplicateResource
	}
	defer modelTraceInFlight.Delete(key)
	select {
	case modelTraceSlots <- struct{}{}:
		defer func() { <-modelTraceSlots }()
	case <-ctx.Done():
		return ModelTraceProbeResponse{}, ctx.Err()
	}
	ctx, cancel := context.WithTimeout(ctx, modelTraceRunDeadline)
	defer cancel()
	cfg, err := normalizedModelTraceConfig(loadModelTraceConfig())
	if err != nil {
		return ModelTraceProbeResponse{}, err
	}
	if s == nil {
		return ModelTraceProbeResponse{}, app_errors.ErrInternalServer
	}
	group, target, credential, err := s.captureModelTraceTarget(ctx, request)
	if err != nil {
		return ModelTraceProbeResponse{}, err
	}
	client := newModelTraceClient(cfg)
	challenges, err := client.challenges(ctx)
	if err != nil {
		return ModelTraceProbeResponse{}, err
	}
	executed := s.executeModelTraceChallenges(ctx, group, target, credential, challenges)
	outcome := ModelTraceProbeResponse{
		GroupID:    group.ID,
		GroupName:  group.Name,
		Model:      target.model,
		Service:    &ModelTraceServiceDescription{BaseURL: cfg.BaseURL, Enabled: true},
		Challenges: executed.challenges,
		TestedAtMS: probeTestedAtMS(s.now().UTC()),
	}
	outputs := modelTraceAcceptedOutputs(executed, challenges)
	if len(outputs) == 0 {
		failure := modelTraceFailureNoOutputs
		outcome.Failure = &failure
		logModelTraceProbe(outcome)
		return outcome, nil
	}
	analysis, err := client.analyze(ctx, outputs)
	if err != nil {
		failure := modelTraceFailureAnalysisUnavailable
		if errors.Is(err, app_errors.ErrModelTraceAnalysisRejected) {
			failure = modelTraceFailureAnalysisRejected
		}
		outcome.Failure = &failure
		logModelTraceProbe(outcome)
		return outcome, nil
	}
	outcome.Analysis = analysis
	logModelTraceProbe(outcome)
	return outcome, nil
}

func (s *Service) captureModelTraceTarget(
	ctx context.Context,
	request ModelTraceRequest,
) (state.GroupView, modelTraceChallengeTarget, state.CredentialRef, error) {
	observation, err := s.captureRuntimeObservation()
	if err != nil {
		return state.GroupView{}, modelTraceChallengeTarget{}, state.CredentialRef{}, err
	}
	group, exists := observation.snapshot.LookupGroup(request.GroupID)
	if !exists {
		return state.GroupView{}, modelTraceChallengeTarget{}, state.CredentialRef{}, app_errors.ErrBadRequest
	}
	target, supported := modelTraceTarget(group, request.Model)
	if !supported {
		return state.GroupView{}, modelTraceChallengeTarget{}, state.CredentialRef{}, app_errors.ErrBadRequest
	}
	entry, found := s.schedulableProbeCredential(group.ID, observation.observedAt)
	if !found {
		return state.GroupView{}, modelTraceChallengeTarget{}, state.CredentialRef{}, app_errors.ErrNoActiveCredential
	}
	return group, target, credentialProbeRef(entry), nil
}

// modelTraceExecutionContext is the shared state of one detection run.
type modelTraceExecutionContext struct {
	group      state.GroupView
	target     modelTraceChallengeTarget
	credential state.CredentialRef
	// priceTable is frozen once, before any challenge is dispatched, so a price
	// publication that lands mid-run cannot re-price a request that was already
	// sent. It mirrors the gateway, which freezes the table before forwarding.
	priceTable *pricing.Table
	challenges []ModelTraceChallengeResponse
	outputs    []modelTraceOutput
}

// modelTraceChallengeRequest retains one executed challenge for the log pass.
type modelTraceChallengeRequest struct {
	response    ModelTraceChallengeResponse
	requestID   string
	attempt     execution.AttemptResult
	sequence    uint32
	completedAt time.Time
	duration    time.Duration
}

func (s *Service) executeModelTraceChallenges(
	ctx context.Context,
	group state.GroupView,
	target modelTraceChallengeTarget,
	credential state.CredentialRef,
	challenges []modelTraceChallengeDTO,
) modelTraceExecutionContext {
	run := modelTraceExecutionContext{
		group:      group,
		target:     target,
		credential: credential,
		priceTable: s.frozenModelTracePriceTable(),
		challenges: make([]ModelTraceChallengeResponse, len(challenges)),
		outputs:    make([]modelTraceOutput, len(challenges)),
	}
	var waitGroup sync.WaitGroup
	for index, challenge := range challenges {
		waitGroup.Add(1)
		go func(index int, challenge modelTraceChallengeDTO) {
			defer waitGroup.Done()
			s.runModelTraceChallenge(ctx, &run, index, challenge)
		}(index, challenge)
	}
	waitGroup.Wait()
	return run
}

func (s *Service) runModelTraceChallenge(
	ctx context.Context,
	run *modelTraceExecutionContext,
	index int,
	challenge modelTraceChallengeDTO,
) {
	response := ModelTraceChallengeResponse{
		ID: challenge.ID, ExpectedCount: challenge.ExpectedCount,
	}
	executed, err := s.executeModelTraceChallenge(ctx, run, challenge)
	if err != nil {
		response.Outcome = modelTraceChallengeFailed
		response.Reason = modelTraceReason(ProbeReasonUnknown)
		run.challenges[index] = response
		return
	}
	response.LogID = s.emitModelTraceRequestLog(run, challenge, executed)
	statusCode := executed.attempt.StatusCode
	latency := max(executed.duration.Milliseconds(), 0)
	response.StatusCode = &statusCode
	response.LatencyMS = &latency
	if answer := strings.TrimSpace(executed.attempt.GeneratedText); answer != "" {
		response.Outcome = modelTraceChallengeSucceeded
		run.outputs[index] = modelTraceOutput{
			PromptID: challenge.ID, ExpectedCount: challenge.ExpectedCount, Text: answer,
		}
	} else {
		response.Outcome = modelTraceChallengeFailed
		response.Reason = modelTraceChallengeFailureReason(executed.attempt)
	}
	run.challenges[index] = response
}

// executeModelTraceChallenge dispatches exactly one challenge as a formal
// attempt. The attempt is validated by the execution contract, so a malformed
// challenge cannot silently widen what reaches an upstream.
func (s *Service) executeModelTraceChallenge(
	ctx context.Context,
	run *modelTraceExecutionContext,
	challenge modelTraceChallengeDTO,
) (modelTraceChallengeRequest, error) {
	if s == nil || run == nil || s.executor == nil || s.channelRegistry == nil || s.encryption == nil {
		return modelTraceChallengeRequest{}, app_errors.ErrInternalServer
	}
	if err := ctx.Err(); err != nil {
		return modelTraceChallengeRequest{}, err
	}
	plaintext, err := s.encryption.Decrypt(run.credential.EncryptedValue)
	if err != nil {
		return modelTraceChallengeRequest{}, app_errors.ErrInternalServer
	}
	credential, err := normalizeStoredCredential(s.channelRegistry, run.group.ChannelID, plaintext)
	plaintext = ""
	if err != nil {
		return modelTraceChallengeRequest{}, app_errors.ErrInternalServer
	}
	apiKey, _ := credential.Value("api_key")
	proxy, proxyFingerprint, err := probeAttemptProxy(run.group.Proxy)
	if err != nil {
		return modelTraceChallengeRequest{}, app_errors.ErrInternalServer
	}
	requestID, err := newOperationID(cryptorand.Reader)
	if err != nil {
		return modelTraceChallengeRequest{}, app_errors.ErrInternalServer
	}
	attemptID, err := newOperationID(cryptorand.Reader)
	if err != nil {
		return modelTraceChallengeRequest{}, app_errors.ErrInternalServer
	}
	spec := execution.NewAttemptSpec(execution.AttemptSpec{
		RequestID: requestID, AttemptID: attemptID, Sequence: 1,
		ChannelID:      string(run.group.ChannelID),
		RouteMode:      execution.RouteMode(run.target.routeMode),
		ClientProtocol: run.target.protocol,
		Operation:      execution.OperationModelTrace,
		ClientModel:    run.target.model, UpstreamModel: run.target.model,
		Header:            applyControlHeaderRules(run.group.HeaderRules, apiKey),
		ConfiguredHeaders: run.group.HeaderRules.ConfiguredNames(),
		TargetConfig:      run.group.ResolvedTarget.TargetConfig,
		Timeouts:          executionTimeouts(run.group.Timeouts),
		Credential: execution.NewCredentialSnapshot(
			run.credential.ID, modelTraceCredentialVersion(run.credential), run.credential.IdentityGeneration,
			credential.CanonicalJSON(),
		),
		ProbeMaxOutputTokens: run.target.maxOutputTokens,
		PromptText:           challenge.Prompt,
		Proxy:                proxy,
		ProxyFingerprint:     proxyFingerprint,
	})
	if err := spec.Validate(); err != nil {
		return modelTraceChallengeRequest{}, app_errors.ErrInternalServer
	}
	startedAt := s.now()
	attempt := s.executor.Execute(ctx, spec)
	completedAt := s.now().UTC()
	return modelTraceChallengeRequest{
		requestID: requestID, attempt: attempt, sequence: spec.Sequence,
		completedAt: completedAt, duration: max(completedAt.Sub(startedAt), 0),
	}, nil
}

// modelTraceCredentialVersion normalizes the credential generation the same way
// a probe does, so the secret-free snapshot always carries a usable identity.
func modelTraceCredentialVersion(ref state.CredentialRef) uint64 {
	if ref.Version == 0 {
		return 1
	}
	return ref.Version
}

// modelTraceChallengeFailureReason classifies a challenge that produced no
// answer. It reports why the answer is missing; it never authorizes a health,
// routing, or retry decision.
func modelTraceChallengeFailureReason(result execution.AttemptResult) *ProbeReason {
	evidence := classifyModelTraceEvidence(result)
	return evidence
}

func classifyModelTraceEvidence(result execution.AttemptResult) *ProbeReason {
	if result.Validate() != nil {
		return modelTraceReason(ProbeReasonUnknown)
	}
	if result.Error == nil {
		if result.StatusCode >= http.StatusOK && result.StatusCode < http.StatusMultipleChoices {
			if strings.TrimSpace(result.GeneratedText) != "" {
				return nil
			}
			if result.GeneratedTextInvalid {
				return modelTraceReason(ProbeReasonInvalidResponse)
			}
			return modelTraceReason(ProbeReasonNoAnswer)
		}
		return modelTraceReason(ProbeReasonUnknown)
	}
	decision := health.JudgeExecution(health.ExecutionAttempt{
		DispatchState: result.DispatchState,
		StatusCode:    result.StatusCode,
		Header:        result.Header,
		Evidence:      result.Error,
		Now:           time.Now(),
	}, health.DecisionContext{Operation: execution.OperationModelTrace})
	switch decision.Category {
	case health.FailureCategoryInvalidKey:
		return modelTraceReason(ProbeReasonInvalidCredential)
	case health.FailureCategoryBilling:
		return modelTraceReason(ProbeReasonInsufficientBalance)
	case health.FailureCategoryModelUnavailable:
		return modelTraceReason(ProbeReasonModelUnavailable)
	case health.FailureCategoryRateLimited:
		return modelTraceReason(ProbeReasonRateLimited)
	}
	switch result.Error.Kind {
	case execution.ErrorKindTimeout:
		return modelTraceReason(ProbeReasonTimeout)
	case execution.ErrorKindConversionUnsupported, execution.ErrorKindInvalidRequest:
		return modelTraceReason(ProbeReasonIncompatible)
	}
	return modelTraceReason(ProbeReasonUpstreamError)
}

// emitModelTraceRequestLog writes the formal request-log row of one challenge.
// The row is a real model request: usage is priced exactly like traffic, the
// outcome carries the attempt's own failure category, and only the attribution
// itself stays outside the log. The write keeps the log pipeline's best-effort
// semantics and never changes the response.
func (s *Service) emitModelTraceRequestLog(
	run *modelTraceExecutionContext,
	challenge modelTraceChallengeDTO,
	executed modelTraceChallengeRequest,
) *string {
	if s == nil || s.requestLogSink == nil || executed.requestID == "" {
		return nil
	}
	s.requestLogSink.Emit(buildModelTraceRequestEvent(run, challenge, executed, run.priceTable, s.now().UTC()))
	logID := executed.requestID
	return &logID
}

func buildModelTraceRequestEvent(
	run *modelTraceExecutionContext,
	challenge modelTraceChallengeDTO,
	executed modelTraceChallengeRequest,
	priceTable *pricing.Table,
	completedAt time.Time,
) telemetry.RequestEvent {
	result := executed.attempt
	reason := classifyModelTraceEvidence(result)
	succeeded := strings.TrimSpace(result.GeneratedText) != "" && result.Error == nil &&
		result.StatusCode >= http.StatusOK && result.StatusCode < http.StatusMultipleChoices
	event := telemetry.RequestEvent{
		RequestID:     executed.requestID,
		CompletedAt:   completedAt,
		AccessKeyID:   0,
		Protocol:      run.target.protocol,
		Operation:     execution.OperationModelTrace,
		ClientModel:   run.target.model,
		UpstreamModel: run.target.model,
		Status:        telemetry.RequestStatusSuccess,
		StatusCode:    result.StatusCode,
		DurationMs:    max(executed.duration.Milliseconds(), 0),
	}
	if succeeded {
		// The gateway never reads a reported model on this path, so "unknown" is
		// the honest observation; the zero value would be rejected by the mapper.
		event.ModelConsistency = telemetry.ModelConsistencyUnknown
	} else {
		event.Status = telemetry.RequestStatusError
		event.ModelConsistency = telemetry.ModelConsistencyNotApplicable
		event.ErrorCode = probeReasonValue(reason)
	}
	decision := health.Decision{}
	if result.Error != nil {
		decision = health.JudgeExecution(health.ExecutionAttempt{
			DispatchState: result.DispatchState,
			StatusCode:    result.StatusCode,
			Header:        result.Header,
			Evidence:      result.Error,
			Now:           completedAt,
		}, health.DecisionContext{Operation: execution.OperationModelTrace})
	}
	failureCategory := telemetry.FailureCategoryOK
	if !succeeded {
		failureCategory = telemetry.FailureCategoryFromHealth(decision.Category)
	}
	summary := probeAttemptErrorSummary(result, run.group.HeaderRules)
	event.Attempts = []telemetry.Attempt{{
		Sequence:          int(executed.sequence),
		CompletedAt:       executed.completedAt,
		GroupID:           run.group.ID,
		GroupName:         run.group.Name,
		ChannelID:         run.group.ChannelID,
		CredentialID:      run.credential.ID,
		Operation:         execution.OperationModelTrace,
		RouteMode:         run.target.routeMode,
		UpstreamModel:     run.target.model,
		UpstreamRequestID: result.UpstreamRequestID,
		DispatchState:     result.DispatchState,
		ResponseStarted:   result.ResponseStarted,
		UpstreamProtocol:  result.UpstreamProtocol,
		StatusCode:        result.StatusCode,
		DurationMs:        max(executed.duration.Milliseconds(), 0),
		FailureCategory:   failureCategory,
		FailureOrigin:     decision.Origin,
		FailureScope:      decision.Scope,
		// A challenge is a single formal model request: it never reroutes and it
		// never applies a health effect.
		Action:       telemetry.ActionTerminate,
		Effect:       telemetry.EffectNone,
		ErrorCode:    "",
		ErrorSummary: summary,
	}}
	if !succeeded {
		event.Attempts[0].ErrorCode = probeReasonValue(reason)
	}
	if event.Status != telemetry.RequestStatusSuccess {
		event.ErrorSummary = summary
		if event.ErrorSummary == "" {
			event.ErrorSummary = event.ErrorCode
		}
	}
	event.Usage = modelTraceUsageObservation(run, executed, result, priceTable)
	return event
}

// modelTraceUsageObservation freezes the challenge usage like traffic: a
// dispatch that reported usage is priced against the table frozen before the
// challenge was sent, while a challenge with no usable usage is recorded as
// missing rather than silently becoming free.
func modelTraceUsageObservation(
	run *modelTraceExecutionContext,
	executed modelTraceChallengeRequest,
	result execution.AttemptResult,
	priceTable *pricing.Table,
) telemetry.UsageObservation {
	observation := telemetry.UsageObservation{
		Result:          usage.Result{State: usage.StateNotApplicable},
		GroupID:         run.group.ID,
		ChannelID:       run.group.ChannelID,
		CredentialID:    run.credential.ID,
		AttemptSequence: int(executed.sequence),
		Pricing: telemetry.PricingObservation{
			UpstreamModel:       run.target.model,
			CostState:           string(pricing.CostStateNotApplicable),
			PricingCompleteness: string(pricing.CompletenessNotApplicable),
		},
	}
	if result.DispatchState == execution.DispatchLocal || result.DispatchState == execution.DispatchNotSent {
		return observation
	}
	captured := usage.Result{}
	if result.Usage != nil {
		captured = result.Usage.Normalized
	}
	// A failed request can still have valid provider usage (for example a
	// truncated answer or a provider error after token generation). Preserve it
	// for formal accounting; only an absent/invalid usage observation becomes
	// missing. The attribution outcome must never decide whether tokens count.
	if !modelTraceUsableUsage(captured) {
		captured = usage.Result{State: usage.StateMissing}
	}
	observation.Result = captured
	observation.Pricing = telemetry.PricingObservation{
		UpstreamModel:       run.target.model,
		CostState:           string(pricing.CostStateUnpriced),
		PricingCompleteness: string(pricing.CompletenessUnavailable),
	}
	if priceTable == nil {
		return observation
	}
	quote, receipt := priceTable.QuoteForModeWithReceipt(
		pricing.Identity{ChannelID: string(run.group.ChannelID), ModelID: run.target.model},
		captured,
		pricing.ModeStandard,
	)
	observation.Pricing.CostState = string(quote.State)
	observation.Pricing.PricingCompleteness = string(quote.Completeness)
	observation.Pricing.EstimatedCostNanoUSD = int64(quote.EstimatedCostNanoUSD)
	if receipt != nil {
		if encoded, err := json.Marshal(receipt); err == nil {
			observation.Pricing.ReceiptJSON = string(encoded)
		}
	}
	return observation
}

// modelTraceAcceptedOutputs keeps only challenges that produced an answer, in
// challenge order. The analysis calibration therefore sees the real number of
// usable outputs instead of padded or partial text.
func modelTraceAcceptedOutputs(
	executed modelTraceExecutionContext,
	challenges []modelTraceChallengeDTO,
) []modelTraceOutput {
	outputs := make([]modelTraceOutput, 0, len(challenges))
	for index, challenge := range challenges {
		if index >= len(executed.outputs) {
			break
		}
		output := executed.outputs[index]
		if strings.TrimSpace(output.Text) == "" {
			continue
		}
		output.PromptID = challenge.ID
		output.ExpectedCount = challenge.ExpectedCount
		outputs = append(outputs, output)
	}
	return outputs
}

func logModelTraceProbe(outcome ModelTraceProbeResponse) {
	failure := ""
	if outcome.Failure != nil {
		failure = string(*outcome.Failure)
	}
	succeeded := 0
	for _, challenge := range outcome.Challenges {
		if challenge.Outcome == modelTraceChallengeSucceeded {
			succeeded++
		}
	}
	utils.LogPlaneBestEffort(
		logrus.StandardLogger(),
		logrus.InfoLevel,
		utils.LogPlaneControl,
		logrus.Fields{
			"event":             "modeltrace_probe_completed",
			"group_id":          outcome.GroupID,
			"model":             outcome.Model,
			"challenges":        len(outcome.Challenges),
			"challenges_passed": succeeded,
			"failure":           failure,
			"prediction":        modelTracePrediction(outcome.Analysis),
			"probability":       modelTraceProbability(outcome.Analysis),
			"used_outputs":      modelTraceUsedOutputs(outcome.Analysis),
		},
		"ModelTrace fingerprint probe completed",
	)
}

func modelTracePrediction(analysis *modelTraceAnalysis) string {
	if analysis == nil {
		return ""
	}
	return analysis.Prediction
}

func modelTraceProbability(analysis *modelTraceAnalysis) float64 {
	if analysis == nil {
		return 0
	}
	return analysis.Probability
}

func modelTraceUsedOutputs(analysis *modelTraceAnalysis) int {
	if analysis == nil {
		return 0
	}
	return analysis.UsedOutputs
}

func (server *Server) handleModelTraceProbe(c *gin.Context) {
	var request ModelTraceRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "modeltrace_probe", mapControlJSONError(err))
		return
	}
	result, err := server.service.RunModelTraceProbe(c.Request.Context(), request)
	if err != nil {
		writeServiceError(c, "modeltrace_probe", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

// modelTraceUsableUsage reports whether a challenge result carries usage that
// can be priced. It mirrors the control-plane usage contract without importing
// the gateway's private helper.
func modelTraceUsableUsage(result usage.Result) bool {
	tokens := result.Tokens
	if tokens.UncachedInput < 0 || tokens.CacheRead < 0 || tokens.CacheWrite5M < 0 ||
		tokens.CacheWrite1H < 0 || tokens.CacheWriteUnknown < 0 || tokens.Output < 0 {
		return false
	}
	if _, ok := usage.CheckedTotal(tokens); !ok {
		return false
	}
	switch result.State {
	case usage.StateComplete, usage.StatePartial:
		return true
	case usage.StateMissing:
		return tokens == (usage.Tokens{})
	default:
		return false
	}
}

// modelTraceReason returns a stable pointer to one probe reason value. It is a
// production-local helper; tests have their own probeReasonOf.
func modelTraceReason(reason ProbeReason) *ProbeReason {
	value := reason
	return &value
}

// frozenModelTracePriceTable captures the currently published immutable table.
// A missing table is frozen as nil, which prices the challenge as unpriced.
func (s *Service) frozenModelTracePriceTable() *pricing.Table {
	if s == nil || s.priceRuntime == nil {
		return nil
	}
	return s.priceRuntime.Load()
}
