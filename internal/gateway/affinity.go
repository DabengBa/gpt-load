package gateway

import (
	"gpt-load/internal/affinity"
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
	"gpt-load/internal/telemetry"
)

type requestAffinity struct {
	key                   affinity.Key
	observation           affinity.Observation
	preferredCredentialID uint
	continuityKey         string
	source                telemetry.AffinitySource
	state                 telemetry.AffinityState
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

func (handler *Handler) resolveRequestAffinity(
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
	result := requestAffinity{source: source, state: telemetry.AffinityStateNoSignal}
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
	// 连续性键即使在软亲和缓存本身不可用时也用于 provider 私有重放范围限定。
	result.continuityKey = string(key)
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
	result.observation = observation
	if !observation.Found() {
		result.state = telemetry.AffinityStateCacheMiss
		return result
	}
	target := observation.Target
	group, exists := snapshot.Groups[target.GroupID]
	if !exists {
		result.state = telemetry.AffinityStateTargetUnavailable
		return result
	}
	if !group.AffinityEnabled {
		result.state = telemetry.AffinityStateGroupDisabled
		return result
	}
	ref, allowed := allowedCredentialRefs[target.CredentialID]
	if !allowed || ref.GroupID != target.GroupID ||
		ref.IdentityGeneration != target.IdentityGeneration {
		result.state = telemetry.AffinityStateTargetUnavailable
		return result
	}
	result.preferredCredentialID = target.CredentialID
	result.state = telemetry.AffinityStateHit
	return result
}

func (handler *Handler) recordAffinitySuccess(
	request requestAffinity,
	selection scheduler.Selection,
	ref state.CredentialRef,
) {
	if handler == nil || handler.affinityCache == nil || !request.key.Valid() ||
		!selection.Group.AffinityEnabled {
		return
	}
	handler.affinityCache.RecordSuccess(
		request.key,
		request.observation,
		affinity.Target{
			GroupID: selection.GroupID, CredentialID: selection.CredentialID,
			IdentityGeneration: ref.IdentityGeneration,
		},
	)
}
