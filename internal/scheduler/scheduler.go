// Package scheduler selects channel targets and credentials without IO or persistence access.
package scheduler

import (
	"errors"
	"math/rand"
	"sort"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
)

var ErrExhausted = errors.New("scheduler exhausted")

type CredentialSource interface {
	CollectCredentialCandidates(groupIDs []uint, excluded func(uint) bool, now time.Time) []state.CredentialMeta
}

type Query struct {
	ClientProtocol           protocol.Protocol
	Operation                execution.Operation
	RouteRequirement         execution.RouteRequirement
	ResponsesStorePreference execution.ResponsesStorePreference
	ExternalModel            *string
	AccessKey                state.AccessKeyView
	AllowedCredentialIDs     map[uint]struct{}
	PreferredCredentialID    uint
}

type Selection struct {
	CredentialID             uint
	GroupID                  uint
	ChannelID                channel.ID
	ResolvedTarget           channel.ResolvedTarget
	RouteMode                channel.RouteMode
	UpstreamModelID          *string
	Group                    state.GroupView
	ResponsesStoreDowngraded bool
}

type candidateTarget struct {
	target                   state.RouteTarget
	group                    state.GroupView
	responsesStoreDowngraded bool
}

// candidateKey is the retry-dedup dimension of the scheduler: one tried pair
// is a (credential, upstream model) combination, so the same credential stays
// retryable on a different route entry while a failed pair never repeats
// (design §5.3).
type candidateKey struct {
	credentialID  uint
	upstreamModel string
}

// weightedCandidate is one schedulable (group, entry, credential) triple with
// its combined weight.
type weightedCandidate struct {
	credential state.CredentialMeta
	target     candidateTarget
	weight     int64
}

// candidatePool holds the static route targets of one store-handling class,
// grouped by route mode and entry priority tier. Targets keep snapshot order
// (Priority, GroupID, UpstreamModelID) inside every tier; a group contributes
// one target per route entry, and groupIDsByMode stays de-duplicated.
type candidatePool struct {
	tierTargetsByMode map[channel.RouteMode]map[int][]candidateTarget
	groupIDsByMode    map[channel.RouteMode][]uint
}

type Iterator struct {
	credentials           CredentialSource
	random                *rand.Rand
	regular               candidatePool
	storeDowngraded       candidatePool
	tiers                 []int
	allowedCredentialIDs  map[uint]struct{}
	preferredCredentialID uint
	tried                 map[candidateKey]struct{}
	skippedGroups         map[uint]struct{}
	staticReason          ReasonCode
	now                   func() time.Time
}

type normalizedQuery struct {
	clientProtocol           protocol.Protocol
	operation                execution.Operation
	routeRequirement         execution.RouteRequirement
	responsesStorePreference execution.ResponsesStorePreference
	externalModel            *string
	accessKey                state.AccessKeyView
	allowedCredentialIDs     map[uint]struct{}
}

func New(snapshot *state.ConfigSnapshot, credentials CredentialSource, query Query, random *rand.Rand) *Iterator {
	return newWithClock(snapshot, credentials, query, random, time.Now)
}

// CandidateGroupIDsForQuery returns the frozen credential-capture scope for a
// fully classified execution query. Multi-mapping groups contribute one entry
// per group: the returned IDs are de-duplicated while keeping snapshot order.
func CandidateGroupIDsForQuery(snapshot *state.ConfigSnapshot, query Query) []uint {
	if snapshot == nil {
		return nil
	}
	decisions, _, err := evaluateTargets(
		snapshot,
		snapshot.ExecutionCandidates,
		normalizeQuery(query),
	)
	if err != nil {
		return []uint{}
	}
	groupIDs := make([]uint, 0, len(decisions))
	seenGroups := make(map[uint]struct{}, len(decisions))
	for _, decision := range decisions {
		if !decision.included {
			continue
		}
		if _, duplicate := seenGroups[decision.target.GroupID]; duplicate {
			continue
		}
		seenGroups[decision.target.GroupID] = struct{}{}
		groupIDs = append(groupIDs, decision.target.GroupID)
	}
	return groupIDs
}

func newWithClock(
	snapshot *state.ConfigSnapshot,
	credentials CredentialSource,
	query Query,
	random *rand.Rand,
	now func() time.Time,
) *Iterator {
	iterator := &Iterator{
		credentials:           credentials,
		random:                random,
		regular:               newCandidatePool(),
		storeDowngraded:       newCandidatePool(),
		allowedCredentialIDs:  cloneAllowedCredentialIDs(query),
		preferredCredentialID: query.PreferredCredentialID,
		tried:                 make(map[candidateKey]struct{}),
		skippedGroups:         make(map[uint]struct{}),
		now:                   now,
	}
	targets, staticReason := filterTargetsWithReason(snapshot, query)
	iterator.staticReason = staticReason
	// 设计 §5.1:条目权重为 0 的条目保留在索引中,但不进入调度候选池
	// (组合权重任一因子 ≤ 0 剔除);全部条目都被剔除时以静态原因码说明。
	for _, target := range targets {
		if target.target.EntryWeight <= 0 {
			continue
		}
		pool := &iterator.regular
		if target.responsesStoreDowngraded {
			pool = &iterator.storeDowngraded
		}
		pool.add(target)
	}
	iterator.tiers = collectTiers(&iterator.regular, &iterator.storeDowngraded)
	if len(iterator.tiers) == 0 && len(targets) > 0 && iterator.staticReason == "" {
		iterator.staticReason = ReasonEntryWeightZero
	}
	return iterator
}

func newCandidatePool() candidatePool {
	return candidatePool{
		tierTargetsByMode: make(map[channel.RouteMode]map[int][]candidateTarget),
		groupIDsByMode:    make(map[channel.RouteMode][]uint),
	}
}

// add registers one route-entry target under its route mode and priority tier
// while keeping the per-mode group list de-duplicated (design §5.1).
func (pool *candidatePool) add(target candidateTarget) {
	mode := target.target.Mode
	tier := target.target.Priority
	if pool.tierTargetsByMode[mode] == nil {
		pool.tierTargetsByMode[mode] = make(map[int][]candidateTarget)
	}
	pool.tierTargetsByMode[mode][tier] = append(pool.tierTargetsByMode[mode][tier], target)
	for _, groupID := range pool.groupIDsByMode[mode] {
		if groupID == target.target.GroupID {
			return
		}
	}
	pool.groupIDsByMode[mode] = append(pool.groupIDsByMode[mode], target.target.GroupID)
}

// collectTiers returns the ascending priority tiers present in either pool
// (design §5.2: Iterator 构建时按 Priority 分层).
func collectTiers(pools ...*candidatePool) []int {
	tierSet := make(map[int]struct{})
	for _, pool := range pools {
		for _, byTier := range pool.tierTargetsByMode {
			for tier := range byTier {
				tierSet[tier] = struct{}{}
			}
		}
	}
	tiers := make([]int, 0, len(tierSet))
	for tier := range tierSet {
		tiers = append(tiers, tier)
	}
	sort.Ints(tiers)
	return tiers
}

func (iterator *Iterator) StaticReason() ReasonCode {
	if iterator == nil {
		return ""
	}
	return iterator.staticReason
}

func cloneAllowedCredentialIDs(query Query) map[uint]struct{} {
	source := query.AllowedCredentialIDs
	if source == nil {
		return nil
	}
	cloned := make(map[uint]struct{}, len(source))
	for credentialID := range source {
		cloned[credentialID] = struct{}{}
	}
	return cloned
}

func (iterator *Iterator) SkipGroup(groupID uint) {
	if iterator == nil || groupID == 0 {
		return
	}
	if iterator.skippedGroups == nil {
		iterator.skippedGroups = make(map[uint]struct{})
	}
	iterator.skippedGroups[groupID] = struct{}{}
}

// weightedTierPool builds the weighted (group, entry, credential) triples of
// one priority tier inside a pool and route mode. Credentials are collected
// live so registry changes between Next calls are honored; tried pairs,
// skipped groups, frozen allowed credentials, and non-positive combined
// weights are excluded (design §5.1–§5.3).
func (iterator *Iterator) weightedTierPool(
	candidates *candidatePool,
	mode channel.RouteMode,
	tier int,
	now time.Time,
) ([]weightedCandidate, int64) {
	if iterator == nil || iterator.credentials == nil {
		return nil, 0
	}
	targets := candidates.tierTargetsByMode[mode][tier]
	if len(targets) == 0 {
		return nil, 0
	}
	collected := iterator.credentials.CollectCredentialCandidates(
		candidates.groupIDsByMode[mode],
		nil,
		now,
	)
	if len(collected) == 0 {
		return nil, 0
	}
	credentialsByGroup := make(map[uint][]state.CredentialMeta, len(collected))
	for _, credential := range collected {
		credentialsByGroup[credential.GroupID] = append(credentialsByGroup[credential.GroupID], credential)
	}
	weighted := make([]weightedCandidate, 0, len(targets))
	for _, target := range targets {
		for _, credential := range credentialsByGroup[target.target.GroupID] {
			if iterator.allowedCredentialIDs != nil {
				if _, allowed := iterator.allowedCredentialIDs[credential.ID]; !allowed {
					continue
				}
			}
			if _, skipped := iterator.skippedGroups[target.target.GroupID]; skipped {
				continue
			}
			key := candidateKey{
				credentialID:  credential.ID,
				upstreamModel: target.target.UpstreamModelID,
			}
			if _, triedPair := iterator.tried[key]; triedPair {
				continue
			}
			weight := combinedWeight(
				target.group.WeightManual,
				target.target.EntryWeight,
				credential.WeightManual,
				credential.WeightAuto,
			)
			if weight <= 0 {
				continue
			}
			weighted = append(weighted, weightedCandidate{
				credential: credential,
				target:     target,
				weight:     weight,
			})
		}
	}
	var total int64
	for _, candidate := range weighted {
		total += candidate.weight
	}
	return weighted, total
}

func (iterator *Iterator) Next() (Selection, error) {
	if iterator == nil || iterator.random == nil || iterator.now == nil {
		return Selection{}, ErrExhausted
	}
	now := iterator.now()
	// 优先级分层(设计 §5.2):只在当前最高可用层内挑选,层耗尽才降级;
	// 层内保持既有的 store 降级与路由模式偏好顺序。
	for _, tier := range iterator.tiers {
		for _, pool := range []*candidatePool{&iterator.regular, &iterator.storeDowngraded} {
			for _, mode := range []channel.RouteMode{channel.RouteNative, channel.RouteConverted} {
				weighted, total := iterator.weightedTierPool(pool, mode, tier, now)
				if total <= 0 {
					continue
				}

				selected, preferred := preferredCandidate(
					weighted,
					iterator.preferredCredentialID,
				)
				if !preferred {
					ticket := iterator.random.Int63n(total)
					selected = weighted[len(weighted)-1]
					for _, candidate := range weighted {
						if ticket < candidate.weight {
							selected = candidate
							break
						}
						ticket -= candidate.weight
					}
				}
				iterator.tried[candidateKey{
					credentialID:  selected.credential.ID,
					upstreamModel: selected.target.target.UpstreamModelID,
				}] = struct{}{}
				return newSelection(selected.credential, selected.target), nil
			}
		}
	}
	return Selection{}, ErrExhausted
}

// preferredCandidate resolves the session-affinity credential inside the
// current tier bucket: the affinity hit pins the credential dimension of the
// triple, while the entry follows the frozen target order (design §5.3).
func preferredCandidate(
	weighted []weightedCandidate,
	credentialID uint,
) (weightedCandidate, bool) {
	if credentialID == 0 {
		return weightedCandidate{}, false
	}
	for _, candidate := range weighted {
		if candidate.credential.ID == credentialID {
			return candidate, true
		}
	}
	return weightedCandidate{}, false
}

func filterTargetsWithReason(
	snapshot *state.ConfigSnapshot,
	query Query,
) ([]candidateTarget, ReasonCode) {
	if snapshot == nil {
		return nil, ""
	}
	decisions, staticReason, err := evaluateTargets(
		snapshot,
		snapshot.ExecutionCandidates,
		normalizeQuery(query),
	)
	if err != nil {
		return nil, ""
	}
	targets := make([]candidateTarget, 0, len(decisions))
	for _, decision := range decisions {
		if !decision.included {
			continue
		}
		group, exists := snapshot.Groups[decision.target.GroupID]
		if !exists {
			continue
		}
		targets = append(targets, candidateTarget{
			target:                   cloneRouteTarget(decision.target),
			group:                    group,
			responsesStoreDowngraded: decision.responsesStoreDowngraded,
		})
	}
	return targets, staticReason
}

func normalizeQuery(query Query) normalizedQuery {
	clientProtocol := query.ClientProtocol
	operation := query.Operation
	if operation == "" {
		if clientProtocol == protocol.OpenAIImages {
			// Images endpoints always select generate or edit explicitly. Keep an
			// omitted operation invalid instead of silently changing the action.
		} else if clientProtocol == protocol.OpenAIResponses {
			if query.ExternalModel == nil {
				operation = execution.OperationResponsesRetrieve
			} else {
				operation = execution.OperationResponsesCreate
			}
		} else {
			operation = execution.OperationChatCompletion
		}
	}
	return normalizedQuery{
		clientProtocol:           clientProtocol,
		operation:                operation,
		routeRequirement:         query.RouteRequirement.Normalize(),
		responsesStorePreference: query.ResponsesStorePreference,
		externalModel:            cloneString(query.ExternalModel),
		accessKey:                query.AccessKey,
		allowedCredentialIDs:     cloneAllowedCredentialIDs(query),
	}
}

func newSelection(credential state.CredentialMeta, target candidateTarget) Selection {
	upstreamModelID := optionalModel(target.target.UpstreamModelID)
	resolvedTarget := target.target.ResolvedTarget
	resolvedTarget.TargetConfig = append([]byte(nil), resolvedTarget.TargetConfig...)
	return Selection{
		CredentialID:             credential.ID,
		GroupID:                  credential.GroupID,
		ChannelID:                resolvedTarget.ChannelID,
		ResolvedTarget:           resolvedTarget,
		RouteMode:                target.target.Mode,
		UpstreamModelID:          upstreamModelID,
		Group:                    cloneGroupView(target.group),
		ResponsesStoreDowngraded: target.responsesStoreDowngraded,
	}
}

func optionalModel(value string) *string {
	if value == "" {
		return nil
	}
	return cloneString(&value)
}

func cloneRouteTarget(target state.RouteTarget) state.RouteTarget {
	target.ResolvedTarget.TargetConfig = append([]byte(nil), target.ResolvedTarget.TargetConfig...)
	return target
}

func cloneGroupView(group state.GroupView) state.GroupView {
	group.Params = append([]byte(nil), group.Params...)
	group.ClientProtocols = append([]protocol.Protocol(nil), group.ClientProtocols...)
	group.Models = append([]state.ModelConfig(nil), group.Models...)
	group.WeightManual = cloneWeight(group.WeightManual)
	group.HeaderRules.Set = cloneStringMap(group.HeaderRules.Set)
	group.HeaderRules.Remove = append([]string(nil), group.HeaderRules.Remove...)
	group.ResolvedTarget.TargetConfig = append([]byte(nil), group.ResolvedTarget.TargetConfig...)
	return group
}

func cloneStringMap(source map[string]string) map[string]string {
	if source == nil {
		return nil
	}
	cloned := make(map[string]string, len(source))
	for key, value := range source {
		cloned[key] = value
	}
	return cloned
}

func cloneString(value *string) *string {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
