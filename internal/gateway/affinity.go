package gateway

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"gpt-load/internal/affinity"
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/platform/utils"
	"gpt-load/internal/protocol"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
	"gpt-load/internal/telemetry"
)

const (
	affinityBindingWriteTimeout = 2 * time.Second
	affinityBindingLockStripes  = 64
)

type requestAffinity struct {
	key                         affinity.Key
	displayKey                  string
	policy                      affinity.DurablePolicy
	observation                 affinity.Observation
	boundTarget                 affinity.Target
	boundAttempted              bool
	boundProviderFailed         bool
	preferredHealthBypassed     bool
	preferredCredentialID       uint
	preferredIdentityGeneration uint64
	continuityKey               string
	source                      telemetry.AffinitySource
	state                       telemetry.AffinityState
	err                         error
}

func affinityTargetForSelection(selection scheduler.Selection, ref state.CredentialRef) affinity.Target {
	upstreamModel := ""
	if selection.UpstreamModelID != nil {
		upstreamModel = *selection.UpstreamModelID
	}
	return affinity.Target{
		GroupID:            selection.GroupID,
		CredentialID:       selection.CredentialID,
		IdentityGeneration: ref.IdentityGeneration,
		EntryID:            selection.EntryID,
		UpstreamModelID:    upstreamModel,
	}
}

func (request *requestAffinity) markBoundAttempt(selection scheduler.Selection, ref state.CredentialRef) {
	if request == nil || !request.boundTarget.Valid() {
		return
	}
	if affinityTargetForSelection(selection, ref) == request.boundTarget {
		request.boundAttempted = true
	}
}

func (request *requestAffinity) markBoundProviderFailure(
	selection scheduler.Selection,
	ref state.CredentialRef,
	dispatchState execution.DispatchState,
	decision health.Decision,
) {
	if request == nil || !request.boundAttempted || !request.boundTarget.Valid() ||
		dispatchState == execution.DispatchLocal {
		return
	}
	if affinityTargetForSelection(selection, ref) != request.boundTarget {
		return
	}
	if decision.Origin == execution.ErrorOriginUpstream && decision.Retry != health.RetryNone {
		request.boundProviderFailed = true
	}
}

func (request requestAffinity) mayRecordSuccess(selection scheduler.Selection, ref state.CredentialRef) bool {
	if request.boundTarget.Valid() && !request.observation.Found() {
		return false
	}
	if !request.boundTarget.Valid() {
		return true
	}
	if affinityTargetForSelection(selection, ref) == request.boundTarget {
		return true
	}
	return request.preferredHealthBypassed || (request.boundAttempted && request.boundProviderFailed)
}

func affinityResultSuccessful(result UpstreamResult, decision health.Decision) bool {
	return result.Err == nil && result.ExecutionError == nil && !result.ProviderErrorBeforeCommit &&
		result.DispatchState != execution.DispatchLocal && result.DispatchState != execution.DispatchNotSent &&
		result.StatusCode >= 200 && result.StatusCode < 300 &&
		decision.Category == health.FailureCategoryOK && decision.Origin == execution.ErrorOriginUpstream &&
		result.Feedback.Status != health.FeedbackStatusFaulty
}

func (handler *Handler) affinitySelectionAvailable(
	selection scheduler.Selection,
	ref state.CredentialRef,
) bool {
	current, exists := handler.registry.CredentialRef(selection.CredentialID)
	if !exists || current.ID != ref.ID || current.GroupID != selection.GroupID ||
		current.IdentityGeneration != ref.IdentityGeneration {
		return false
	}

	if _, ready := handler.registry.ActiveEncryptedCredentialDataIfMatch(current); !ready {
		return false
	}
	now := handler.now()
	candidates := handler.registry.CollectCredentialCandidates([]uint{selection.GroupID}, nil, now)
	available := false
	for _, candidate := range candidates {
		if candidate.ID == ref.ID && candidate.IdentityGeneration == ref.IdentityGeneration {
			available = true
			break
		}
	}
	if !available {
		return false
	}
	if runtime, ok := handler.registry.(state.EntryRuntimeSource); ok {
		entry, _ := runtime.EntryRuntime(state.RouteEntryKey{GroupID: selection.GroupID, EntryID: selection.EntryID}, now)
		return entry.RuntimeState(now) == state.EntryRuntimeAvailable
	}
	return false
}

func affinityBindingLockStripe(key affinity.Key) int {
	var sum uint
	for index := range key {
		sum += uint(key[index])
	}
	return int(sum % affinityBindingLockStripes)
}

func (handler *Handler) affinityBindingLock(key affinity.Key) *sync.Mutex {
	return &handler.affinityBindingLocks[affinityBindingLockStripe(key)]
}

// affinitySignal 优先选择显式 prompt 缓存键，仅在无显式键时回退到推断的
// prompt 前缀，匹配 U001 键合约。
func affinitySignal(
	metadata dialect.RequestMetadata,
) (affinity.SignalType, telemetry.AffinitySource, []byte, bool) {
	if metadata.PromptCacheKey != "" {
		return affinity.SignalPromptCacheKey,
			telemetry.AffinitySourcePromptCacheKey,
			[]byte(metadata.PromptCacheKey),
			true
	}
	if len(metadata.AffinityPrefix) > 0 {
		return affinity.SignalPromptPrefix,
			telemetry.AffinitySourcePromptPrefix,
			metadata.AffinityPrefix,
			true
	}
	return "", telemetry.AffinitySourceNone, nil, false
}

func hasAffinityEnabledCandidate(
	snapshot *state.ConfigSnapshot,
	allowedCredentialRefs map[uint]state.CredentialRef,
) bool {
	if snapshot == nil {
		return false
	}
	for _, ref := range allowedCredentialRefs {
		group, exists := snapshot.Groups[ref.GroupID]
		if exists && group.AffinityEnabled {
			return true
		}
	}
	return false
}

// resolveRequestAffinity derives the request's affinity key and resolves the
// bound credential through the hot cache first and the durable store second.
// A durable lookup or decode failure is returned in requestAffinity.err so the
// caller can fail closed instead of falling back with an unknown binding.
func (handler *Handler) resolveRequestAffinity(
	ctx context.Context,
	snapshot *state.ConfigSnapshot,
	accessKeyID uint,
	clientProtocol protocol.Protocol,
	clientModel string,
	operation execution.Operation,
	metadata dialect.RequestMetadata,
	allowedCredentialRefs map[uint]state.CredentialRef,
) requestAffinity {
	if handler == nil || snapshot == nil {
		return requestAffinity{state: telemetry.AffinityStateCacheUnavailable}
	}
	signalType, source, signalValue, hasSignal := affinitySignal(metadata)
	result := requestAffinity{
		source: source,
		state:  telemetry.AffinityStateNoSignal,
		policy: affinity.DurablePolicy{
			TTL:      snapshot.Settings.AffinityTTL,
			Capacity: snapshot.Settings.AffinityCapacity,
		},
	}
	if len(metadata.AffinityPrefix) > 0 {
		continuity := affinity.DeriveKey(
			handler.encryption,
			accessKeyID,
			clientProtocol,
			clientModel,
			operation,
			affinity.SignalPromptPrefix,
			metadata.AffinityPrefix,
		)
		result.continuityKey = string(continuity)
	}
	if !hasSignal {
		return result
	}
	key := affinity.DeriveKey(
		handler.encryption,
		accessKeyID,
		clientProtocol,
		clientModel,
		operation,
		signalType,
		signalValue,
	)
	result.displayKey = affinity.DisplayKey(key)
	if handler.affinityCache == nil ||
		!handler.affinityCache.Configure(
			snapshot.Revision,
			snapshot.Settings.AffinityCapacity,
			snapshot.Settings.AffinityTTL,
		) {
		result.state = telemetry.AffinityStateCacheUnavailable
		return result
	}
	if !key.Valid() {
		result.state = telemetry.AffinityStateCacheUnavailable
		return result
	}
	result.key = key

	observation := handler.affinityCache.Lookup(key)
	target, hasTarget := affinity.Target{}, false
	if observation.Found() {
		target, hasTarget = observation.Target, true
	} else if handler.affinityStore != nil && hasAffinityEnabledCandidate(snapshot, allowedCredentialRefs) {
		lock := handler.affinityBindingLock(key)
		lock.Lock()
		stored, found, err := handler.affinityStore.Lookup(ctx, key, result.policy)
		if err == nil && found {
			// The hot cache is evictable: a durable hit refills it so this
			// request and later ones keep the bound target.
			handler.affinityCache.RecordSuccess(key, observation, stored)
			observation = handler.affinityCache.Lookup(key)
			if observation.Found() {
				target = observation.Target
			} else {
				target = stored
			}
			hasTarget = true
		}
		lock.Unlock()
		if err != nil {
			result.state = telemetry.AffinityStateCacheUnavailable
			result.err = fmt.Errorf("lookup affinity binding: %w", err)
			return result
		}
	}
	result.observation = observation
	if hasTarget {
		result.boundTarget = target
	}
	if !hasTarget {
		result.state = telemetry.AffinityStateCacheMiss
		return result
	}
	group, exists := snapshot.Groups[target.GroupID]
	if !exists {
		result.state = telemetry.AffinityStateTargetUnavailable
		return result
	}
	if !group.AffinityEnabled {
		result.state = telemetry.AffinityStateGroupDisabled
		return result
	}
	// Configuration removal or a model edit invalidates the remembered route.
	// Inspect the full snapshot, not request-filtered candidates: temporary
	// permission, weight, health or capacity exclusions must preserve a binding.
	routeExists := false
	for _, model := range group.Models {
		if model.EntryID == target.EntryID && strings.TrimSpace(model.ID) == target.UpstreamModelID &&
			(state.ExternalModelName(model.ID, model.Alias) == clientModel || model.TestAlias == clientModel) {
			routeExists = true
			break
		}
	}
	if !routeExists {
		result.boundTarget = affinity.Target{}
		result.state = telemetry.AffinityStateTargetUnavailable
		return result
	}
	ref, allowed := allowedCredentialRefs[target.CredentialID]
	if !allowed || ref.GroupID != target.GroupID ||
		ref.IdentityGeneration != target.IdentityGeneration {
		result.state = telemetry.AffinityStateTargetUnavailable
		return result
	}
	result.preferredCredentialID = target.CredentialID
	// The bypass permission belongs to the observed binding identity, not the
	// credential ID: a same-ID replacement must re-earn it with fresh health.
	result.preferredIdentityGeneration = target.IdentityGeneration
	result.state = telemetry.AffinityStateHit
	return result
}

// recordAffinitySuccess mirrors one successful provider selection into the hot
// cache and, when a durable store is wired, persists it. A store failure keeps
// the delivered response untouched, preserves the hot-cache result, and is
// reported on the data plane without the raw affinity key.
func (handler *Handler) recordAffinitySuccess(
	ctx context.Context,
	request requestAffinity,
	selection scheduler.Selection,
	ref state.CredentialRef,
) {
	if handler == nil || handler.affinityCache == nil || !request.key.Valid() ||
		!selection.Group.AffinityEnabled || !request.mayRecordSuccess(selection, ref) {
		return
	}
	target := affinityTargetForSelection(selection, ref)
	lock := handler.affinityBindingLock(request.key)
	lock.Lock()
	defer lock.Unlock()
	if !handler.affinitySelectionAvailable(selection, ref) {
		return
	}
	updated := handler.affinityCache.RecordSuccess(request.key, request.observation, target)
	if handler.affinityStore == nil || !updated {
		return
	}
	writeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), affinityBindingWriteTimeout)
	defer cancel()
	if err := handler.affinityStore.Upsert(writeCtx, request.key, target, request.policy); err != nil {
		utils.LogPlaneBestEffort(
			handler.logger,
			logrus.ErrorLevel,
			utils.LogPlaneData,
			logrus.Fields{
				"event":         "affinity_binding_persist_failed",
				"affinity_key":  request.displayKey,
				"group_id":      target.GroupID,
				"credential_id": target.CredentialID,
				"error":         err,
			},
			"Affinity binding persistence failed",
		)
	}
}
