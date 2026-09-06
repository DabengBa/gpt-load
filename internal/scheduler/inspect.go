package scheduler

import (
	"errors"
	"fmt"
	"sort"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
)

var ErrInconsistentSnapshot = errors.New("inconsistent scheduler snapshot")

type ReasonCode string

const (
	ReasonAccessKeyDisabled         ReasonCode = "access_key_disabled"
	ReasonAccessKeyExpired          ReasonCode = "access_key_expired"
	ReasonProtocolFiltered          ReasonCode = "protocol_filtered"
	ReasonModelFiltered             ReasonCode = "model_filtered"
	ReasonModelRequiredByFilter     ReasonCode = "model_required_by_filter"
	ReasonOperationUnsupported      ReasonCode = "operation_unsupported"
	ReasonNativeRouteRequired       ReasonCode = "native_route_required"
	ReasonNoRouteTarget             ReasonCode = "no_route_target"
	ReasonGroupDisabled             ReasonCode = "group_disabled"
	ReasonGroupFiltered             ReasonCode = "group_filtered"
	ReasonNoAvailableGroup          ReasonCode = "no_available_group"
	ReasonNoCredentials             ReasonCode = "no_credentials"
	ReasonGroupWeightZero           ReasonCode = "group_weight_zero"
	ReasonEntryWeightZero           ReasonCode = "entry_weight_zero"
	ReasonCredentialDisabled        ReasonCode = "credential_disabled"
	ReasonCredentialAuthUnavailable ReasonCode = "credential_auth_unavailable"
	ReasonCredentialBlacklisted     ReasonCode = "credential_blacklisted"
	ReasonCredentialCooldown        ReasonCode = "credential_cooldown"
	ReasonCredentialWeightZero      ReasonCode = "credential_weight_zero"
	ReasonCredentialNotAllowed      ReasonCode = "credential_not_allowed"
	ReasonNoAvailableCredential     ReasonCode = "no_available_credential"
	ReasonEntryBlacklisted          ReasonCode = "entry_blacklisted"
	ReasonEntryCooldown             ReasonCode = "entry_cooldown"
	ReasonTierDemoted               ReasonCode = "tier_demoted"
)

type Inspection struct {
	ClientProtocol   protocol.Protocol
	Operation        execution.Operation
	RouteRequirement execution.RouteRequirement
	ExternalModel    *string
	Routable         bool
	Reason           ReasonCode
	Groups           []GroupInspection
}

type GroupInspection struct {
	GroupID                   uint
	GroupName                 string
	ChannelID                 channel.ID
	RouteMode                 channel.RouteMode
	RouteRequirementSatisfied bool
	EntryID                   string
	UpstreamModelID           *string
	WeightManual              *int
	EntryWeight               int
	Priority                  int
	EntryCooldownUntil        time.Time
	Included                  bool
	Routable                  bool
	Reason                    ReasonCode
	// EffectiveShare is the entry's expected traffic share inside the P1 tier
	// (priority 1) under the current snapshot, access key and availability
	// constraints. Fallback tiers (priority ≥ 2) stay out of the P1
	// normalization and always carry share 0.
	EffectiveShare float64
	Credentials    []CredentialInspection
}

type CredentialInspection struct {
	CredentialID    uint
	Available       bool
	Reason          ReasonCode
	WeightManual    *int
	WeightAuto      int
	EffectiveWeight int64
	CooldownUntil   time.Time
}

// CredentialRuntimeView is the scheduler's neutral view of runtime health.
type CredentialRuntimeView = state.CredentialRuntimeView

type targetDecision struct {
	target                   state.RouteTarget
	group                    state.GroupCatalogView
	requirementOK            bool
	responsesStoreDowngraded bool
	included                 bool
	reason                   ReasonCode
}

// targetEntryKey de-duplicates route targets of one group and upstream model
// when multi-mapping groups contribute several entries (design §5.1).
type targetEntryKey struct {
	groupID         uint
	upstreamModelID string
}

func cloneWeight(weight *int) *int {
	if weight == nil {
		return nil
	}
	value := *weight
	return &value
}

func evaluateTargets(
	snapshot *state.ConfigSnapshot,
	index state.ExecutionCandidateIndex,
	query normalizedQuery,
) ([]targetDecision, ReasonCode, error) {
	if snapshot == nil {
		return nil, "", fmt.Errorf("%w: nil ConfigSnapshot", ErrInconsistentSnapshot)
	}
	if query.accessKey.Status == state.AccessKeyStatusDisabled {
		return []targetDecision{}, ReasonAccessKeyDisabled, nil
	}
	if len(query.accessKey.Filters.Protocols) > 0 {
		if _, allowed := query.accessKey.Filters.Protocols[query.clientProtocol]; !allowed {
			return []targetDecision{}, ReasonProtocolFiltered, nil
		}
	}
	if len(query.accessKey.Filters.Models) > 0 {
		if query.externalModel == nil {
			return []targetDecision{}, ReasonModelRequiredByFilter, nil
		}
		if _, allowed := query.accessKey.Filters.Models[*query.externalModel]; !allowed {
			return []targetDecision{}, ReasonModelFiltered, nil
		}
	}
	if !query.clientProtocol.Valid() || !query.operation.Valid() ||
		!query.routeRequirement.Valid() || !query.responsesStorePreference.Valid() {
		return []targetDecision{}, ReasonOperationUnsupported, nil
	}
	byOperation := index[query.clientProtocol]
	if len(byOperation) == 0 {
		return []targetDecision{}, ReasonNoRouteTarget, nil
	}
	byModel, operationSupported := byOperation[query.operation]
	if !operationSupported {
		return []targetDecision{}, ReasonOperationUnsupported, nil
	}
	modelKey := state.NoModelRouteKey
	if query.externalModel != nil {
		modelKey = *query.externalModel
	}
	routes := byModel[modelKey]
	if len(routes) == 0 {
		return []targetDecision{}, ReasonNoRouteTarget, nil
	}

	decisions := make([]targetDecision, 0, len(routes))
	// 设计 §5.1:同一分组的同一对外名可产生多个条目 target(多映射),
	// 去重键扩展为 (GroupID, UpstreamModelID);资源类 target 的上游模型为空串,
	// 每分组仍恰保留一个。
	seenEntries := make(map[targetEntryKey]struct{}, len(routes))
	included := 0
	for _, route := range routes {
		entryKey := targetEntryKey{groupID: route.GroupID, upstreamModelID: route.UpstreamModelID}
		if _, duplicate := seenEntries[entryKey]; duplicate {
			continue
		}
		seenEntries[entryKey] = struct{}{}
		group, exists := snapshot.GroupCatalog[route.GroupID]
		if !exists {
			return nil, "", fmt.Errorf(
				"%w: route target group %d missing from catalog",
				ErrInconsistentSnapshot,
				route.GroupID,
			)
		}
		requirementOK, storeDowngraded, requirementReason := routeRequirementSatisfied(query, route)
		decision := targetDecision{
			target: cloneRouteTarget(route), group: group,
			requirementOK: requirementOK, responsesStoreDowngraded: storeDowngraded,
			included: true,
		}
		groupFiltered := false
		if len(query.accessKey.Filters.Groups) > 0 {
			_, allowed := query.accessKey.Filters.Groups[route.GroupID]
			groupFiltered = !allowed
		}
		switch {
		case !requirementOK:
			decision.included = false
			decision.reason = requirementReason
		case !group.Enabled:
			decision.included = false
			decision.reason = ReasonGroupDisabled
		case groupFiltered:
			decision.included = false
			decision.reason = ReasonGroupFiltered
		}
		if decision.included {
			included++
		}
		decisions = append(decisions, decision)
	}
	if included > 0 {
		return decisions, "", nil
	}
	reason := decisions[0].reason
	for _, decision := range decisions[1:] {
		if decision.reason != reason {
			return decisions, ReasonNoAvailableGroup, nil
		}
	}
	return decisions, reason, nil
}

func routeRequirementSatisfied(
	query normalizedQuery,
	route state.RouteTarget,
) (bool, bool, ReasonCode) {
	if !query.routeRequirement.Allows(execution.RouteMode(route.Mode)) {
		return false, false, ReasonNativeRouteRequired
	}
	if query.operation != execution.OperationResponsesCreate {
		return true, false, ""
	}
	if query.routeRequirement.Normalize() == execution.RouteRequirementNative {
		if route.ResolvedTarget.SupportsResponsesLifecycle() {
			return true, false, ""
		}
		return false, false, ReasonNativeRouteRequired
	}
	if query.responsesStorePreference != execution.ResponsesStorePreferencePreferStored {
		return true, false, ""
	}
	handling := route.ResolvedTarget.ResponsesStoreHandling(
		protocol.OpenAIResponses,
		execution.OperationResponsesCreate,
	)
	switch handling {
	case channel.ResponsesStoreHandlingUpstreamManaged:
		return true, false, ""
	case channel.ResponsesStoreHandlingStateless:
		return true, true, ""
	default:
		return false, false, ReasonOperationUnsupported
	}
}

func accessKeyAllowsGroup(accessKey state.AccessKeyView, groupID uint) bool {
	if len(accessKey.Filters.Groups) == 0 {
		return true
	}
	_, allowed := accessKey.Filters.Groups[groupID]
	return allowed
}

func normalizedAutoWeight(weight int) int {
	if weight == 0 {
		return state.DefaultWeight
	}
	return weight
}

// combinedWeight is the single source of scheduling share for a
// (group, entry, credential) triple: 组权重 × 条目权重 × 密钥权重
// (design §5.1). Any factor ≤ 0 yields 0 so the triple never joins a
// weighted pool; callers must treat 0 as excluded. Unset group and
// credential weights fall back to the state defaults, and the entry weight
// arrives already normalized by snapshot compilation.
func combinedWeight(
	groupManual *int,
	entryWeight int,
	credentialManual *int,
	credentialAuto int,
) int64 {
	groupWeight := state.DefaultWeight
	if groupManual != nil {
		groupWeight = *groupManual
	}
	credentialWeight := normalizedAutoWeight(credentialAuto)
	if credentialManual != nil {
		credentialWeight = *credentialManual
	}
	if groupWeight <= 0 || entryWeight <= 0 || credentialWeight <= 0 {
		return 0
	}
	return int64(groupWeight) * int64(entryWeight) * int64(credentialWeight)
}

// effectiveWeight preserves the pre-route-entry helper contract for existing
// scheduler tests and compatibility callers. Route-entry scheduling uses
// combinedWeight directly with the entry factor supplied by the snapshot.
func effectiveWeight(groupManual, credentialManual *int, credentialAuto int) int64 {
	return combinedWeight(groupManual, 1, credentialManual, credentialAuto)
}

func inspectCredential(
	group state.GroupCatalogView,
	entryWeight int,
	credential CredentialRuntimeView,
	allowedCredentialIDs map[uint]struct{},
	now time.Time,
) CredentialInspection {
	result := CredentialInspection{
		CredentialID: credential.ID,
		WeightManual: cloneWeight(credential.WeightManual),
		WeightAuto:   normalizedAutoWeight(credential.WeightAuto),
	}
	if group.WeightManual != nil && *group.WeightManual == 0 {
		result.Reason = ReasonGroupWeightZero
		return result
	}
	if allowedCredentialIDs != nil {
		if _, allowed := allowedCredentialIDs[credential.ID]; !allowed {
			result.Reason = ReasonCredentialNotAllowed
			return result
		}
	}
	if credential.Status != state.CredentialStatusActive {
		result.Reason = ReasonCredentialDisabled
		return result
	}
	if !credential.AuthReady() {
		result.Reason = ReasonCredentialAuthUnavailable
		return result
	}
	if credential.WeightManual != nil && *credential.WeightManual == 0 {
		result.Reason = ReasonCredentialWeightZero
		return result
	}
	switch credential.RuntimeState(now) {
	case state.CredentialRuntimeBlacklisted:
		result.Reason = ReasonCredentialBlacklisted
	case state.CredentialRuntimeCooldown:
		result.Reason = ReasonCredentialCooldown
		result.CooldownUntil = credential.CooldownUntil
	default:
		result.Available = true
		result.EffectiveWeight = combinedWeight(
			group.WeightManual,
			entryWeight,
			credential.WeightManual,
			credential.WeightAuto,
		)
	}
	return result
}

// Inspect preserves the existing credential-only inspection contract.
func Inspect(
	snapshot *state.ConfigSnapshot,
	credentials []CredentialRuntimeView,
	query Query,
	now time.Time,
) (Inspection, error) {
	return InspectWithEntryRuntime(snapshot, credentials, nil, query, now)
}

// InspectWithEntryRuntime inspects route targets together with optional
// in-memory route-entry health. Entry state is keyed by group and route-entry
// identity (design §2), never by credential, so one model failure cannot hide
// sibling entries.
func InspectWithEntryRuntime(
	snapshot *state.ConfigSnapshot,
	credentials []CredentialRuntimeView,
	entryRuntime []state.EntryRuntimeView,
	query Query,
	now time.Time,
) (Inspection, error) {
	normalized := normalizeQuery(query)
	result := Inspection{
		ClientProtocol:   normalized.clientProtocol,
		Operation:        normalized.operation,
		RouteRequirement: normalized.routeRequirement,
		ExternalModel:    cloneString(normalized.externalModel),
		Groups:           []GroupInspection{},
	}
	if snapshot == nil {
		return Inspection{}, fmt.Errorf("%w: nil ConfigSnapshot", ErrInconsistentSnapshot)
	}
	if normalized.accessKey.Status == state.AccessKeyStatusDisabled {
		result.Reason = ReasonAccessKeyDisabled
		return result, nil
	}

	entryRuntimeByKey := make(map[state.RouteEntryKey]state.EntryRuntimeView, len(entryRuntime))
	for _, entry := range entryRuntime {
		if entry.Key.GroupID == 0 || entry.Key.EntryID == "" {
			continue
		}
		entryRuntimeByKey[entry.Key] = entry
	}
	credentialsByGroup := make(map[uint][]CredentialRuntimeView)
	for _, credential := range credentials {
		if _, exists := snapshot.GroupCatalog[credential.GroupID]; !exists {
			return Inspection{}, fmt.Errorf(
				"%w: registry credential %d group %d missing from catalog",
				ErrInconsistentSnapshot,
				credential.ID,
				credential.GroupID,
			)
		}
		cloned := credential
		cloned.WeightManual = cloneWeight(credential.WeightManual)
		credentialsByGroup[credential.GroupID] = append(credentialsByGroup[credential.GroupID], cloned)
	}
	for groupID := range credentialsByGroup {
		sort.Slice(credentialsByGroup[groupID], func(i, j int) bool {
			return credentialsByGroup[groupID][i].ID < credentialsByGroup[groupID][j].ID
		})
	}

	decisions, staticReason, err := evaluateTargets(
		snapshot,
		snapshot.ExecutionRouteCatalog,
		normalized,
	)
	if err != nil {
		return Inspection{}, err
	}
	for _, decision := range decisions {
		entryKey := state.RouteEntryKey{
			GroupID: decision.target.GroupID,
			EntryID: decision.target.EntryID,
		}
		entryState, entryHasRuntime := entryRuntimeByKey[entryKey]
		groupResult := GroupInspection{
			GroupID:                   decision.group.ID,
			GroupName:                 decision.group.Name,
			ChannelID:                 decision.target.ResolvedTarget.ChannelID,
			RouteMode:                 decision.target.Mode,
			RouteRequirementSatisfied: decision.requirementOK,
			EntryID:                   decision.target.EntryID,
			UpstreamModelID:           optionalModel(decision.target.UpstreamModelID),
			WeightManual:              cloneWeight(decision.group.WeightManual),
			EntryWeight:               decision.target.EntryWeight,
			Priority:                  decision.target.Priority,
			Included:                  decision.included,
			Reason:                    decision.reason,
			Credentials:               []CredentialInspection{},
		}
		if entryHasRuntime && entryState.RuntimeState(now) == state.EntryRuntimeCooldown {
			groupResult.EntryCooldownUntil = entryState.CooldownUntil
		}
		if !decision.included {
			result.Groups = append(result.Groups, groupResult)
			continue
		}
		groupCredentials := credentialsByGroup[decision.group.ID]
		groupWeightZero := decision.group.WeightManual != nil &&
			*decision.group.WeightManual == 0
		entryUnavailable := entryHasRuntime &&
			entryState.RuntimeState(now) != state.EntryRuntimeAvailable
		for _, credential := range groupCredentials {
			credentialResult := inspectCredential(
				decision.group,
				decision.target.EntryWeight,
				credential,
				normalized.allowedCredentialIDs,
				now,
			)
			groupResult.Credentials = append(groupResult.Credentials, credentialResult)
		}
		if !entryUnavailable {
			for _, credential := range groupResult.Credentials {
				if credential.Available && credential.EffectiveWeight > 0 {
					groupResult.Routable = true
					break
				}
			}
		}
		switch {
		case groupWeightZero:
			groupResult.Reason = ReasonGroupWeightZero
		case decision.target.EntryWeight <= 0:
			groupResult.Reason = ReasonEntryWeightZero
		case entryUnavailable && entryState.Blacklisted:
			groupResult.Reason = ReasonEntryBlacklisted
		case entryUnavailable:
			groupResult.Reason = ReasonEntryCooldown
		case len(groupCredentials) == 0:
			groupResult.Reason = ReasonNoCredentials
		case !groupResult.Routable:
			groupResult.Reason = ReasonNoAvailableCredential
		}
		if groupResult.Routable {
			result.Routable = true
		}
		result.Groups = append(result.Groups, groupResult)
	}

	minimumRoutableTier := 0
	for _, group := range result.Groups {
		if !group.Routable || group.Priority <= 0 {
			continue
		}
		if minimumRoutableTier == 0 || group.Priority < minimumRoutableTier {
			minimumRoutableTier = group.Priority
		}
	}
	if minimumRoutableTier > 0 {
		for index := range result.Groups {
			group := &result.Groups[index]
			if group.Routable && group.Priority > minimumRoutableTier && group.Reason == "" {
				group.Reason = ReasonTierDemoted
			}
		}
	}
	applyEffectiveShares(result.Groups)

	if result.Routable {
		return result, nil
	}
	if staticReason != "" {
		result.Reason = staticReason
	} else {
		result.Reason = ReasonNoAvailableCredential
	}
	return result, nil
}

// applyEffectiveShares normalizes combined weights inside the lowest currently
// routable priority tier. Rows in other tiers or rows that are not currently
// routable contribute nothing and receive share 0, mirroring tier-first
// selection.
func applyEffectiveShares(groups []GroupInspection) {
	activeTier := 0
	for i := range groups {
		group := &groups[i]
		if !group.Routable || group.Priority <= 0 {
			continue
		}
		var mass int64
		for _, credential := range group.Credentials {
			if credential.Available {
				mass += credential.EffectiveWeight
			}
		}
		if mass > 0 && (activeTier == 0 || group.Priority < activeTier) {
			activeTier = group.Priority
		}
	}
	if activeTier == 0 {
		return
	}
	var activeTierTotal int64
	for i := range groups {
		group := &groups[i]
		if group.Priority != activeTier || !group.Routable {
			continue
		}
		for _, credential := range group.Credentials {
			if credential.Available {
				activeTierTotal += credential.EffectiveWeight
			}
		}
	}
	if activeTierTotal <= 0 {
		return
	}
	for i := range groups {
		group := &groups[i]
		if group.Priority != activeTier || !group.Routable {
			continue
		}
		var mass int64
		for _, credential := range group.Credentials {
			if credential.Available {
				mass += credential.EffectiveWeight
			}
		}
		group.EffectiveShare = float64(mass) / float64(activeTierTotal)
	}
}
