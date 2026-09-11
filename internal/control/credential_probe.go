package control

import (
	"context"
	cryptorand "crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"gorm.io/gorm"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/platform/epochms"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/utils"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/storage/models"
)

type ProbeOutcome string

const (
	ProbeOutcomePassed       ProbeOutcome = "passed"
	ProbeOutcomeFailed       ProbeOutcome = "failed"
	ProbeOutcomeInconclusive ProbeOutcome = "inconclusive"
)

// ProbeReason is the shared probe reason vocabulary. Execution outcomes come from
// classifyCredentialProbeEvidence; the two target-level reasons are only produced
// by the model probe, which resolves (group, model) before it can execute.
type ProbeReason string

const (
	ProbeReasonInvalidCredential ProbeReason = "invalid_credential"
	ProbeReasonModelUnavailable  ProbeReason = "model_unavailable"
	ProbeReasonRateLimited       ProbeReason = "rate_limited"
	ProbeReasonTimeout           ProbeReason = "timeout"
	ProbeReasonUpstreamError     ProbeReason = "upstream_error"
	ProbeReasonIncompatible      ProbeReason = "probe_incompatible"
	ProbeReasonUnknown           ProbeReason = "unknown"

	ProbeReasonTargetUnavailable       ProbeReason = "target_unavailable"
	ProbeReasonNoSchedulableCredential ProbeReason = "no_schedulable_credential"
)

type CredentialProbeResponse struct {
	Outcome      ProbeOutcome      `json:"outcome"`
	Model        string            `json:"model"`
	Protocol     protocol.Protocol `json:"protocol"`
	LatencyMS    int64             `json:"latency_ms"`
	Reason       *ProbeReason      `json:"reason"`
	CanRestore   bool              `json:"can_restore"`
	RestoreProof *string           `json:"restore_proof"`
	LogID        *string           `json:"log_id"`
	TestedAtMS   int64             `json:"tested_at_ms"`
}

type CredentialProbeRestoreRequest struct {
	RestoreProof string `json:"restore_proof"`
}

const credentialProbeRestoreProofDomain = "gpt-load/control/credential-probe-restore/v1"

type credentialProbeRestoreProofPayload struct {
	CredentialID       uint   `json:"credential_id"`
	CooldownUntil      string `json:"cooldown_until"`
	EncryptedValue     string `json:"encrypted_value"`
	FailureGeneration  uint64 `json:"failure_generation"`
	Fingerprint        string `json:"fingerprint"`
	GroupID            uint   `json:"group_id"`
	IdentityGeneration uint64 `json:"identity_generation"`
	TargetSignature    string `json:"target_signature"`
	Version            uint64 `json:"version"`
}

type credentialProbeCredential struct {
	ref           state.CredentialRef
	cooldownUntil time.Time
}

// credentialProbeExecution is one completed probe execution: the returned attempt
// plus every attempt executed before it. A protocol fallback runs the executor
// more than once, and the durable request log must record one row per executed
// protocol instead of silently keeping only the last one.
type credentialProbeExecution struct {
	requestID string
	result    execution.AttemptResult
	latency   time.Duration
	protocol  protocol.Protocol
	attempts  []credentialProbeAttempt
}

type credentialProbeAttempt struct {
	sequence    uint32
	routeMode   channel.RouteMode
	completedAt time.Time
	duration    time.Duration
	result      execution.AttemptResult
}

type credentialProbeExecutor struct {
	decryptor credentialDecryptor
	channels  *channel.Registry
	executor  execution.Executor
	now       func() time.Time
}

type credentialProbeFailure struct {
	stage string
	cause error
}

func (failure *credentialProbeFailure) Error() string {
	return "credential probe preparation failed at " + failure.stage
}

func (failure *credentialProbeFailure) Unwrap() error {
	return failure.cause
}

func newCredentialProbeExecutor(
	decryptor credentialDecryptor,
	channels *channel.Registry,
	executor execution.Executor,
) *credentialProbeExecutor {
	return &credentialProbeExecutor{
		decryptor: decryptor,
		channels:  channels,
		executor:  executor,
		now:       time.Now,
	}
}

func (probe *credentialProbeExecutor) Probe(
	ctx context.Context,
	group state.GroupView,
	target groupValidationTarget,
	ref state.CredentialRef,
) (credentialProbeExecution, error) {
	if probe == nil || probe.executor == nil || probe.channels == nil {
		return credentialProbeExecution{}, newCredentialProbeFailure(
			"missing_executor",
			app_errors.ErrInternalServer,
		)
	}
	if probe.decryptor == nil {
		return credentialProbeExecution{}, newCredentialProbeFailure(
			"decrypt",
			app_errors.ErrInternalServer,
		)
	}
	if err := ctx.Err(); err != nil {
		return credentialProbeExecution{}, err
	}
	plaintext, err := probe.decryptor.Decrypt(ref.EncryptedValue)
	if err != nil {
		return credentialProbeExecution{}, newCredentialProbeFailure(
			"decrypt",
			app_errors.ErrInternalServer,
		)
	}
	credential, err := normalizeStoredCredential(probe.channels, group.ChannelID, plaintext)
	plaintext = ""
	if err != nil {
		return credentialProbeExecution{}, newCredentialProbeFailure(
			"credential",
			app_errors.ErrInternalServer,
		)
	}
	apiKey, _ := credential.Value("api_key")
	proxy, proxyFingerprint, err := validationAttemptProxy(group.Proxy)
	if err != nil {
		return credentialProbeExecution{}, newCredentialProbeFailure(
			"proxy",
			app_errors.ErrInternalServer,
		)
	}
	requestID, err := newOperationID(cryptorand.Reader)
	if err != nil {
		return credentialProbeExecution{}, newCredentialProbeFailure(
			"request_identity",
			app_errors.ErrInternalServer,
		)
	}
	version := ref.Version
	if version == 0 {
		version = 1
	}
	generation := ref.IdentityGeneration
	if generation == 0 {
		generation = 1
	}
	probeProtocols := []protocol.Protocol{target.protocol}
	probeProtocols = append(probeProtocols, target.fallbackProtocols...)
	attempts := make([]credentialProbeAttempt, 0, len(probeProtocols))
	startedAt := probe.now()
	for index, probeProtocol := range probeProtocols {
		routeMode, supported := group.ResolvedTarget.ModeForModel(
			probeProtocol,
			execution.OperationProbe,
			target.model,
		)
		if !supported {
			return credentialProbeExecution{}, newCredentialProbeFailure(
				"request",
				app_errors.ErrValidation,
			)
		}
		attemptID, attemptErr := newOperationID(cryptorand.Reader)
		if attemptErr != nil {
			return credentialProbeExecution{}, newCredentialProbeFailure(
				"attempt_identity",
				app_errors.ErrInternalServer,
			)
		}
		spec := execution.NewAttemptSpec(execution.AttemptSpec{
			RequestID: requestID, AttemptID: attemptID, Sequence: uint32(index + 1),
			ChannelID: string(group.ChannelID),
			RouteMode: execution.RouteMode(routeMode), ClientProtocol: probeProtocol,
			Operation: execution.OperationProbe, ClientModel: target.model, UpstreamModel: target.model,
			Header:            applyControlHeaderRules(group.HeaderRules, apiKey),
			ConfiguredHeaders: group.HeaderRules.ConfiguredNames(),
			TargetConfig:      group.ResolvedTarget.TargetConfig,
			Timeouts:          executionTimeouts(group.Timeouts),
			Credential: execution.NewCredentialSnapshot(
				ref.ID,
				version,
				generation,
				credential.CanonicalJSON(),
			),
			Proxy: proxy, ProxyFingerprint: proxyFingerprint,
		})
		if err := spec.Validate(); err != nil {
			return credentialProbeExecution{}, newCredentialProbeFailure(
				"request",
				app_errors.ErrValidation,
			)
		}
		attemptStartedAt := probe.now()
		result := probe.executor.Execute(ctx, spec)
		if err := ctx.Err(); err != nil {
			return credentialProbeExecution{}, err
		}
		attempts = append(attempts, credentialProbeAttempt{
			sequence:    spec.Sequence,
			routeMode:   channel.RouteMode(routeMode),
			completedAt: probe.now().UTC(),
			duration:    max(probe.now().Sub(attemptStartedAt), 0),
			result:      result,
		})
		latency := max(probe.now().Sub(startedAt), 0)
		executed := credentialProbeExecution{
			requestID: requestID,
			result:    result,
			latency:   latency,
			protocol:  probeProtocol,
			attempts:  attempts,
		}
		if credentialProbePassed(result) || index+1 == len(probeProtocols) ||
			!validationProbeNeedsProtocolFallback(result) {
			return executed, nil
		}
	}
	return credentialProbeExecution{}, newCredentialProbeFailure(
		"probe",
		app_errors.ErrInternalServer,
	)
}

func newCredentialProbeFailure(stage string, cause error) error {
	return &credentialProbeFailure{stage: stage, cause: cause}
}

func credentialProbeFailureStage(err error) string {
	var failure *credentialProbeFailure
	if errors.As(err, &failure) && failure.stage != "" {
		return failure.stage
	}
	return "probe"
}

func credentialProbePassed(result execution.AttemptResult) bool {
	return result.Validate() == nil && result.Error == nil &&
		result.StatusCode >= http.StatusOK &&
		result.StatusCode < http.StatusMultipleChoices
}

// credentialProbeEvidence is one probe judgement: the reported outcome and reason
// plus the health decision behind them. The response needs outcome/reason and the
// durable log needs the decision (category/origin/scope), so both read a single
// judgement instead of judging twice and drifting apart.
type credentialProbeEvidence struct {
	outcome  ProbeOutcome
	reason   *ProbeReason
	decision health.Decision
}

func classifyCredentialProbeResult(result execution.AttemptResult) (
	ProbeOutcome,
	*ProbeReason,
) {
	evidence := classifyCredentialProbeEvidence(result)
	return evidence.outcome, evidence.reason
}

func classifyCredentialProbeEvidence(result execution.AttemptResult) credentialProbeEvidence {
	if result.Validate() != nil {
		return credentialProbeEvidenceWith(
			ProbeOutcomeInconclusive,
			ProbeReasonUnknown,
			health.Decision{},
		)
	}
	if credentialProbePassed(result) {
		return credentialProbeEvidence{outcome: ProbeOutcomePassed}
	}
	if result.Error == nil {
		return credentialProbeEvidenceWith(
			ProbeOutcomeInconclusive,
			ProbeReasonUnknown,
			health.Decision{},
		)
	}
	decision := health.JudgeExecution(health.ExecutionAttempt{
		DispatchState: result.DispatchState,
		StatusCode:    result.StatusCode,
		Header:        result.Header,
		Evidence:      result.Error,
		Now:           time.Now(),
	}, health.DecisionContext{Operation: execution.OperationProbe})
	switch decision.Category {
	case health.FailureCategoryInvalidKey:
		return credentialProbeEvidenceWith(
			ProbeOutcomeFailed,
			ProbeReasonInvalidCredential,
			decision,
		)
	case health.FailureCategoryModelUnavailable:
		return credentialProbeEvidenceWith(
			ProbeOutcomeFailed,
			ProbeReasonModelUnavailable,
			decision,
		)
	case health.FailureCategoryRateLimited:
		return credentialProbeEvidenceWith(
			ProbeOutcomeInconclusive,
			ProbeReasonRateLimited,
			decision,
		)
	}
	switch {
	case result.Error.Kind == execution.ErrorKindTimeout:
		return credentialProbeEvidenceWith(
			ProbeOutcomeInconclusive,
			ProbeReasonTimeout,
			decision,
		)
	case result.Error.Kind == execution.ErrorKindConversionUnsupported ||
		result.Error.Kind == execution.ErrorKindInvalidRequest:
		return credentialProbeEvidenceWith(
			ProbeOutcomeInconclusive,
			ProbeReasonIncompatible,
			decision,
		)
	case decision.Category == health.FailureCategoryUpstreamHostError ||
		result.Error.Kind == execution.ErrorKindTransport ||
		result.Error.Kind == execution.ErrorKindHTTP ||
		result.Error.Kind == execution.ErrorKindProvider:
		return credentialProbeEvidenceWith(
			ProbeOutcomeInconclusive,
			ProbeReasonUpstreamError,
			decision,
		)
	}
	return credentialProbeEvidenceWith(
		ProbeOutcomeInconclusive,
		ProbeReasonUnknown,
		decision,
	)
}

func credentialProbeEvidenceWith(
	outcome ProbeOutcome,
	reason ProbeReason,
	decision health.Decision,
) credentialProbeEvidence {
	value := reason
	return credentialProbeEvidence{outcome: outcome, reason: &value, decision: decision}
}

func (s *Service) TestGroupCredential(
	ctx context.Context,
	groupID uint,
	credentialID uint,
) (CredentialProbeResponse, error) {
	if groupID == 0 || credentialID == 0 {
		return CredentialProbeResponse{}, app_errors.ErrBadRequest
	}
	group, target, credential, err := s.captureCredentialProbe(ctx, groupID, credentialID)
	if err != nil {
		return CredentialProbeResponse{}, err
	}
	probe := newCredentialProbeExecutor(s.encryption, s.channelRegistry, s.executor)
	executed, err := probe.Probe(ctx, group, target, credential.ref)
	if err != nil {
		return CredentialProbeResponse{}, err
	}
	evidence := classifyCredentialProbeEvidence(executed.result)
	completedAt := s.now().UTC()
	testedAt, err := epochms.FromTime(completedAt)
	if err != nil {
		return CredentialProbeResponse{}, fmt.Errorf(
			"encode credential probe time: %w",
			app_errors.ErrInternalServer,
		)
	}
	response := CredentialProbeResponse{
		Outcome: evidence.outcome, Model: target.model, Protocol: executed.protocol,
		LatencyMS:  max(executed.latency.Milliseconds(), 0),
		Reason:     evidence.reason,
		TestedAtMS: testedAt,
	}
	if evidence.outcome == ProbeOutcomePassed {
		response.RestoreProof = s.currentCredentialProbeRestoreProof(credential, target.signature)
		response.CanRestore = response.RestoreProof != nil
	}
	response.LogID = s.emitProbeRequestLog(probeLogObservation{
		group:       group,
		model:       target.model,
		credential:  credential.ref,
		executed:    executed,
		evidence:    evidence,
		completedAt: completedAt,
	})
	logCredentialProbe(credential.ref, response)
	return response, nil
}

func (s *Service) RestoreTestedGroupCredential(
	ctx context.Context,
	groupID uint,
	credentialID uint,
	restoreProof string,
) (CredentialItemResponse, error) {
	if groupID == 0 || credentialID == 0 {
		return CredentialItemResponse{}, app_errors.ErrBadRequest
	}
	restoreProof = strings.TrimSpace(restoreProof)
	if restoreProof == "" {
		return CredentialItemResponse{}, app_errors.ErrValidation
	}
	return s.restoreGroupCredential(ctx, groupID, credentialID, restoreProof)
}

func (s *Service) captureCredentialProbe(
	ctx context.Context,
	groupID uint,
	credentialID uint,
) (state.GroupView, groupValidationTarget, credentialProbeCredential, error) {
	if s == nil || s.db == nil || s.manager == nil || s.registry == nil {
		return state.GroupView{}, groupValidationTarget{}, credentialProbeCredential{}, app_errors.ErrInternalServer
	}
	s.writeMu.RLock()
	defer s.writeMu.RUnlock()
	groupRow, err := loadGroupRow(s.db.WithContext(ctx), groupID)
	if err != nil {
		return state.GroupView{}, groupValidationTarget{}, credentialProbeCredential{}, err
	}
	if normalizeGroupConnectionType(groupRow.ConnectionType) == models.ConnectionTypeSubscription {
		return state.GroupView{}, groupValidationTarget{}, credentialProbeCredential{}, app_errors.ErrForbidden
	}
	var credentialRow models.Credential
	if err := s.db.WithContext(ctx).
		Where("id = ? AND group_id = ?", credentialID, groupID).
		Take(&credentialRow).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return state.GroupView{}, groupValidationTarget{}, credentialProbeCredential{}, credentialNotFoundError()
		}
		return state.GroupView{}, groupValidationTarget{}, credentialProbeCredential{}, app_errors.ParseDBError(err)
	}
	view, exists := findRuntimeCredential(s.registry.Snapshot(), credentialID)
	if err := validateCredentialRuntimeRow(groupRow, credentialRow, view, exists); err != nil {
		return state.GroupView{}, groupValidationTarget{}, credentialProbeCredential{}, err
	}
	entries, err := s.registry.SnapshotGroupCredentialEntriesExact(groupID, []uint{credentialID})
	if err != nil || len(entries) != 1 {
		return state.GroupView{}, groupValidationTarget{}, credentialProbeCredential{}, dbRegistryMismatch(
			mismatchMissingRegistry,
			groupID,
			credentialID,
		)
	}
	entry := entries[0]
	if entry.Fingerprint != credentialRow.Fingerprint ||
		entry.EncryptedValue != credentialRow.Data {
		return state.GroupView{}, groupValidationTarget{}, credentialProbeCredential{}, dbRegistryMismatch(
			mismatchIdentity,
			groupID,
			credentialID,
		)
	}
	snapshot := s.manager.Current()
	if snapshot == nil {
		return state.GroupView{}, groupValidationTarget{}, credentialProbeCredential{}, app_errors.ErrInternalServer
	}
	group, exists := snapshot.Groups[groupID]
	if !exists {
		return state.GroupView{}, groupValidationTarget{}, credentialProbeCredential{}, dbRegistryMismatch(
			mismatchMissingRegistry,
			groupID,
			credentialID,
		)
	}
	target, valid := buildGroupValidationTarget(group)
	if !valid {
		return state.GroupView{}, groupValidationTarget{}, credentialProbeCredential{}, app_errors.ErrValidation
	}
	return group, target, credentialProbeCredentialFromEntry(entry), nil
}

func credentialProbeRef(entry state.CredentialEntry) state.CredentialRef {
	return state.CredentialRef{
		ID: entry.ID, GroupID: entry.GroupID,
		Version: entry.Version, IdentityGeneration: entry.IdentityGeneration,
		Fingerprint: entry.Fingerprint, EncryptedValue: entry.EncryptedValue,
		FailureGeneration: entry.FailureGeneration,
	}
}

func credentialProbeCredentialFromEntry(entry state.CredentialEntry) credentialProbeCredential {
	return credentialProbeCredential{
		ref:           credentialProbeRef(entry),
		cooldownUntil: entry.CooldownUntil,
	}
}

func credentialProbeCooldownIdentity(value time.Time) string {
	if value.IsZero() {
		return ""
	}
	return value.UTC().Format(time.RFC3339Nano)
}

func (s *Service) currentCredentialProbeRestoreProof(
	tested credentialProbeCredential,
	testedSignature groupValidationSignature,
) *string {
	if s == nil || s.manager == nil || s.registry == nil || s.encryption == nil || s.mutations == nil {
		return nil
	}
	var proof *string
	s.manager.WithCurrentSnapshot(func(snapshot *state.ConfigSnapshot) bool {
		if snapshot == nil {
			return false
		}
		group, exists := snapshot.Groups[tested.ref.GroupID]
		if !exists {
			return false
		}
		currentTarget, valid := buildGroupValidationTarget(group)
		if !valid || currentTarget.signature != testedSignature {
			return false
		}
		capture := func() {
			entries, err := s.registry.SnapshotGroupCredentialEntriesExact(
				tested.ref.GroupID,
				[]uint{tested.ref.ID},
			)
			if err != nil || len(entries) != 1 {
				return
			}
			current := entries[0]
			currentCredential := credentialProbeCredentialFromEntry(current)
			if !current.Blacklisted ||
				currentCredential.ref != tested.ref ||
				!currentCredential.cooldownUntil.Equal(tested.cooldownUntil) {
				return
			}
			value, ok := s.credentialProbeRestoreProof(tested, testedSignature)
			if ok {
				proof = &value
			}
		}
		s.mutations.Do(tested.ref.ID, capture)
		return proof != nil
	})
	return proof
}

func (s *Service) credentialProbeRestoreProof(
	credential credentialProbeCredential,
	targetSignature groupValidationSignature,
) (string, bool) {
	if s == nil || s.encryption == nil {
		return "", false
	}
	ref := credential.ref
	payload, err := json.Marshal(credentialProbeRestoreProofPayload{
		CredentialID: ref.ID, GroupID: ref.GroupID,
		CooldownUntil: credentialProbeCooldownIdentity(credential.cooldownUntil),
		Version:       ref.Version, IdentityGeneration: ref.IdentityGeneration,
		Fingerprint: ref.Fingerprint, EncryptedValue: ref.EncryptedValue,
		FailureGeneration: ref.FailureGeneration,
		TargetSignature:   hex.EncodeToString(targetSignature[:]),
	})
	if err != nil {
		return "", false
	}
	proof := s.encryption.Hash(credentialProbeRestoreProofDomain + "\n" + string(payload))
	return proof, proof != ""
}

func (s *Service) credentialProbeRestoreProofMatches(
	entry state.CredentialEntry,
	targetSignature groupValidationSignature,
	expected string,
) bool {
	if !entry.Blacklisted {
		return false
	}
	current, ok := s.credentialProbeRestoreProof(
		credentialProbeCredentialFromEntry(entry),
		targetSignature,
	)
	if !ok {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(current), []byte(expected)) == 1
}

func logCredentialProbe(ref state.CredentialRef, response CredentialProbeResponse) {
	reason := ""
	if response.Reason != nil {
		reason = string(*response.Reason)
	}
	utils.LogPlaneBestEffort(
		logrus.StandardLogger(),
		logrus.InfoLevel,
		utils.LogPlaneControl,
		logrus.Fields{
			"event":         "credential_probe_completed",
			"credential_id": ref.ID,
			"group_id":      ref.GroupID,
			"outcome":       response.Outcome,
			"reason":        reason,
			"model":         response.Model,
			"protocol":      response.Protocol,
			"latency_ms":    response.LatencyMS,
		},
		"Credential probe completed",
	)
}
