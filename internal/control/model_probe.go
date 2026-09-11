package control

import (
	"context"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
	"gpt-load/internal/pricing"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

const (
	modelProbeMaxTargets  = 64
	modelProbeConcurrency = 4
)

// ModelProbeTargetRequest is one (group, model) probe target.
type ModelProbeTargetRequest struct {
	GroupID uint   `json:"group_id"`
	Model   string `json:"model"`
}

// ModelProbeRequest probes 1..modelProbeMaxTargets targets through one endpoint:
// batch probing is the same primitive with more targets, not a second mechanism.
type ModelProbeRequest struct {
	Targets []ModelProbeTargetRequest `json:"targets"`
}

// ModelProbeResultResponse is one probe outcome. Fields that only exist after a
// dispatch stay null for targets that never executed, and LogID is non-null
// exactly when an upstream attempt was executed.
type ModelProbeResultResponse struct {
	GroupID         uint               `json:"group_id"`
	GroupName       string             `json:"group_name"`
	Model           string             `json:"model"`
	Outcome         ProbeOutcome       `json:"outcome"`
	Reason          *ProbeReason       `json:"reason"`
	Protocol        *protocol.Protocol `json:"protocol"`
	RouteMode       *channel.RouteMode `json:"route_mode"`
	StatusCode      *int               `json:"status_code"`
	LatencyMS       *int64             `json:"latency_ms"`
	CredentialID    *uint              `json:"credential_id"`
	CredentialLabel *string            `json:"credential_label"`
	LogID           *string            `json:"log_id"`
	TestedAtMS      int64              `json:"tested_at_ms"`
}

type ModelProbeResponse struct {
	Results []ModelProbeResultResponse `json:"results"`
}

func validateModelProbeRequest(request ModelProbeRequest) error {
	if len(request.Targets) == 0 || len(request.Targets) > modelProbeMaxTargets {
		return app_errors.ErrBadRequest
	}
	for _, target := range request.Targets {
		// validUsageModel is the same shape the request-log mapper accepts, so a
		// target that passes validation can never be rejected later by mapEvent.
		if target.GroupID == 0 || !validUsageModel(target.Model) {
			return app_errors.ErrBadRequest
		}
	}
	return nil
}

type modelProbeKey struct {
	groupID uint
	model   string
}

func dedupeModelProbeTargets(targets []ModelProbeTargetRequest) []ModelProbeTargetRequest {
	seen := make(map[modelProbeKey]struct{}, len(targets))
	deduped := make([]ModelProbeTargetRequest, 0, len(targets))
	for _, target := range targets {
		key := modelProbeKey{groupID: target.GroupID, model: target.Model}
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		deduped = append(deduped, target)
	}
	return deduped
}

func (service *Service) ProbeGroupModels(
	ctx context.Context,
	request ModelProbeRequest,
) (ModelProbeResponse, error) {
	if err := validateModelProbeRequest(request); err != nil {
		return ModelProbeResponse{}, err
	}
	observation, err := service.captureRuntimeObservation()
	if err != nil {
		return ModelProbeResponse{}, err
	}
	probe := newCredentialProbeExecutor(service.encryption, service.channelRegistry, service.executor)
	deduped := dedupeModelProbeTargets(request.Targets)
	results := make([]ModelProbeResultResponse, len(deduped))
	// Acquire before launching so the bound also caps live goroutines, and wait
	// so results can be assembled in deduped input order.
	semaphore := make(chan struct{}, modelProbeConcurrency)
	var waitGroup sync.WaitGroup
	for index, target := range deduped {
		semaphore <- struct{}{}
		waitGroup.Add(1)
		go func(index int, target ModelProbeTargetRequest) {
			defer waitGroup.Done()
			defer func() { <-semaphore }()
			results[index] = service.probeModelTarget(ctx, probe, observation, target)
		}(index, target)
	}
	waitGroup.Wait()
	return ModelProbeResponse{Results: results}, nil
}

// probeModelTarget never fails the batch: every target problem becomes that
// target's own reason, and only a target that reached the executor gets a log id.
func (service *Service) probeModelTarget(
	ctx context.Context,
	probe *credentialProbeExecutor,
	observation runtimeObservation,
	target ModelProbeTargetRequest,
) ModelProbeResultResponse {
	result := ModelProbeResultResponse{
		GroupID:    target.GroupID,
		Model:      target.Model,
		TestedAtMS: probeTestedAtMS(observation.observedAt),
	}
	group, exists := observation.snapshot.Groups[target.GroupID]
	if !exists {
		return probeWithoutExecution(result, ProbeReasonTargetUnavailable)
	}
	result.GroupName = group.Name
	if !groupHasModel(group, target.Model) {
		return probeWithoutExecution(result, ProbeReasonTargetUnavailable)
	}
	probeTarget, supported := buildGroupProbeTarget(group, target.Model)
	if !supported {
		return probeWithoutExecution(result, ProbeReasonIncompatible)
	}
	routeMode, routeSupported := group.ResolvedTarget.ModeForModel(
		probeTarget.protocol,
		execution.OperationProbe,
		target.Model,
	)
	if !routeSupported {
		return probeWithoutExecution(result, ProbeReasonIncompatible)
	}
	result.Protocol = &probeTarget.protocol
	result.RouteMode = &routeMode
	entry, found := service.schedulableProbeCredential(group.ID, observation.observedAt)
	if !found {
		return probeWithoutExecution(result, ProbeReasonNoSchedulableCredential)
	}
	if err := ctx.Err(); err != nil {
		return probeWithoutExecution(result, ProbeReasonUnknown)
	}
	executed, err := probe.Probe(ctx, group, probeTarget, credentialProbeRef(entry))
	if err != nil {
		return probeWithoutExecution(result, ProbeReasonUnknown)
	}
	evidence := classifyCredentialProbeEvidence(executed.result)
	statusCode := executed.result.StatusCode
	latencyMS := max(executed.latency.Milliseconds(), 0)
	credentialID := entry.ID
	result.Outcome = evidence.outcome
	result.Reason = evidence.reason
	result.StatusCode = &statusCode
	result.LatencyMS = &latencyMS
	result.CredentialID = &credentialID
	if label, known := service.CredentialLabels([]uint{entry.ID})[entry.ID]; known && label != "" {
		result.CredentialLabel = &label
	}
	result.LogID = service.emitProbeRequestLog(probeLogObservation{
		group:       group,
		model:       target.Model,
		credential:  credentialProbeRef(entry),
		executed:    executed,
		evidence:    evidence,
		completedAt: service.now().UTC(),
	})
	return result
}

// probeWithoutExecution records a target problem. Nothing was dispatched, so the
// honest outcome is "inconclusive" (no verdict was observed) and no log id is
// claimed.
func probeWithoutExecution(
	result ModelProbeResultResponse,
	reason ProbeReason,
) ModelProbeResultResponse {
	value := reason
	result.Outcome = ProbeOutcomeInconclusive
	result.Reason = &value
	return result
}

func groupHasModel(group state.GroupView, model string) bool {
	for _, configured := range group.Models {
		if configured.ID == model {
			return true
		}
	}
	return false
}

// schedulableProbeCredential picks the smallest-id schedulable credential of one
// group. Selection is deterministic on purpose: a probe reports whether this group
// can currently serve with an available credential, not a sampled traffic share.
// The candidate list only carries identity, so the entry is read again to obtain
// the material a probe execution needs.
func (service *Service) schedulableProbeCredential(
	groupID uint,
	observedAt time.Time,
) (state.CredentialEntry, bool) {
	if service == nil || service.registry == nil {
		return state.CredentialEntry{}, false
	}
	candidates := service.registry.CollectCredentialCandidates([]uint{groupID}, nil, observedAt)
	if len(candidates) == 0 {
		return state.CredentialEntry{}, false
	}
	entries, err := service.registry.SnapshotGroupCredentialEntriesExact(
		groupID,
		[]uint{candidates[0].ID},
	)
	if err != nil || len(entries) != 1 {
		return state.CredentialEntry{}, false
	}
	return entries[0], true
}

func probeTestedAtMS(observedAt time.Time) int64 {
	value, err := safeEpochMilliseconds(observedAt)
	if err != nil {
		return 0
	}
	return value
}

// probeLogObservation is one completed probe handed to the request-log pipeline.
type probeLogObservation struct {
	group       state.GroupView
	model       string
	credential  state.CredentialRef
	executed    credentialProbeExecution
	evidence    credentialProbeEvidence
	completedAt time.Time
}

// emitProbeRequestLog writes one operation=probe row and returns its primary key.
// The write inherits the log pipeline's best-effort semantics; the id is only
// claimed when a sink is wired and an attempt actually executed.
func (service *Service) emitProbeRequestLog(observation probeLogObservation) *string {
	if service == nil || service.requestLogSink == nil {
		return nil
	}
	if observation.executed.requestID == "" || len(observation.executed.attempts) == 0 {
		return nil
	}
	service.requestLogSink.Emit(buildProbeRequestEvent(observation))
	logID := observation.executed.requestID
	return &logID
}

// buildProbeRequestEvent maps a probe to the frozen request-log contract. Probe
// rows are control-plane observations: they carry no access key, no usage and no
// cost, while every executed protocol keeps its own attempt row.
func buildProbeRequestEvent(observation probeLogObservation) telemetry.RequestEvent {
	executed := observation.executed
	event := telemetry.RequestEvent{
		RequestID:     executed.requestID,
		CompletedAt:   observation.completedAt,
		AccessKeyID:   0,
		Protocol:      executed.protocol,
		Operation:     execution.OperationProbe,
		ClientModel:   observation.model,
		UpstreamModel: observation.model,
		Status:        telemetry.RequestStatusSuccess,
		StatusCode:    executed.result.StatusCode,
		DurationMs:    max(executed.latency.Milliseconds(), 0),
	}
	if observation.evidence.outcome == ProbeOutcomePassed {
		// A probe never reads a reported model, so "unknown" is the honest
		// observation; the zero value would be rejected by the log mapper.
		event.ModelConsistency = telemetry.ModelConsistencyUnknown
	} else {
		event.Status = telemetry.RequestStatusError
		event.ModelConsistency = telemetry.ModelConsistencyNotApplicable
		event.ErrorCode = probeReasonValue(observation.evidence.reason)
		event.ErrorSummary = event.ErrorCode
	}
	attempts := make([]telemetry.Attempt, 0, len(executed.attempts))
	for _, attempt := range executed.attempts {
		attemptEvidence := classifyCredentialProbeEvidence(attempt.result)
		attempts = append(attempts, telemetry.Attempt{
			Sequence:          int(attempt.sequence),
			CompletedAt:       attempt.completedAt,
			GroupID:           observation.group.ID,
			GroupName:         observation.group.Name,
			ChannelID:         observation.group.ChannelID,
			CredentialID:      observation.credential.ID,
			Operation:         execution.OperationProbe,
			RouteMode:         attempt.routeMode,
			UpstreamModel:     observation.model,
			UpstreamRequestID: attempt.result.UpstreamRequestID,
			DispatchState:     attempt.result.DispatchState,
			ResponseStarted:   attempt.result.ResponseStarted,
			UpstreamProtocol:  attempt.result.UpstreamProtocol,
			StatusCode:        attempt.result.StatusCode,
			DurationMs:        max(attempt.duration.Milliseconds(), 0),
			FailureCategory:   probeAttemptFailureCategory(attemptEvidence),
			FailureOrigin:     attemptEvidence.decision.Origin,
			FailureScope:      attemptEvidence.decision.Scope,
			Action:            telemetry.ActionTerminate,
			Effect:            telemetry.EffectNone,
			ErrorCode:         probeReasonValue(attemptEvidence.reason),
		})
	}
	event.Attempts = attempts
	returned := executed.attempts[len(executed.attempts)-1]
	event.Usage = telemetry.UsageObservation{
		GroupID:         observation.group.ID,
		ChannelID:       observation.group.ChannelID,
		CredentialID:    observation.credential.ID,
		AttemptSequence: int(returned.sequence),
		Result:          usage.Result{State: usage.StateNotApplicable},
		Pricing: telemetry.PricingObservation{
			UpstreamModel:       observation.model,
			CostState:           string(pricing.CostStateNotApplicable),
			PricingCompleteness: string(pricing.CompletenessNotApplicable),
		},
	}
	return event
}

// probeAttemptFailureCategory maps one attempt judgement onto the closed
// request-log failure category set. Unjudged attempts fall back to the zero
// health category, which is "ambiguous".
func probeAttemptFailureCategory(evidence credentialProbeEvidence) telemetry.FailureCategory {
	if evidence.outcome == ProbeOutcomePassed {
		return telemetry.FailureCategoryOK
	}
	return telemetry.FailureCategoryFromHealth(evidence.decision.Category)
}

func probeReasonValue(reason *ProbeReason) string {
	if reason == nil {
		return ""
	}
	return string(*reason)
}

func (server *Server) handleModelProbe(c *gin.Context) {
	var request ModelProbeRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "probe_model", mapControlJSONError(err))
		return
	}
	result, err := server.service.ProbeGroupModels(c.Request.Context(), request)
	if err != nil {
		writeServiceError(c, "probe_model", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}
