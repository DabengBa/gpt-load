// Package state owns immutable runtime configuration snapshots.
package state

import (
	"encoding/json"
	"fmt"
	"net/netip"
	"sort"
	"strings"
	"time"

	"gpt-load/internal/accessquota"
	"gpt-load/internal/channel"
	"gpt-load/internal/connection"
	"gpt-load/internal/execution"
	"gpt-load/internal/outboundproxy"
	"gpt-load/internal/parameteroverride"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/protocol"
)

const maxSafeAccessKeyEpochMS = int64(9_007_199_254_740_991)

type CompileInput struct {
	SystemSettings   config.Settings
	ChannelRegistry  *channel.Registry
	Groups           []GroupConfig
	Credentials      []CredentialConfig
	AccessKeys       []AccessKeyConfig
	GlobalProxy      *outboundproxy.Config
	EnvironmentProxy *outboundproxy.Config
}

type GroupConfig struct {
	ID              uint
	Name            string
	ChannelID       channel.ID
	ConnectionType  string
	Params          json.RawMessage
	ValidationModel string
	Models          []ModelConfig
	Settings        config.Settings
	WeightManual    *int
	Enabled         bool
	Proxy           *outboundproxy.Config
}

// CredentialConfig contains only non-secret credential metadata required to
// validate a runtime configuration publication.
type CredentialConfig struct {
	ID                 uint
	GroupID            uint
	Status             CredentialStatus
	WeightManual       *int
	Version            uint64
	IdentityGeneration uint64
	Fingerprint        string
}

// ModelConfig is one group model route entry. Weight and Priority are
// optional: nil keeps the design defaults (weight 1, priority 1); weight 0
// retains the entry but excludes it from traffic splitting.
type ModelConfig struct {
	ID             string
	Alias          string
	EntryID        string
	Weight         *int
	Priority       *int
	CircuitBreaker *EntryCircuitBreaker
}

type AccessKeyConfig struct {
	ID               uint
	Name             string
	KeyHash          string
	KeySuffix        string
	Status           AccessKeyStatus
	Filters          FilterSet
	ExpiresAtMS      *int64
	AllowedPeerCIDRs []netip.Prefix
	RPMLimit         int64
	CostLimitRules   []accessquota.Rule
}

type AccessKeyStatus string

const (
	AccessKeyStatusActive   AccessKeyStatus = "active"
	AccessKeyStatusDisabled AccessKeyStatus = "disabled"
)

type FilterSet struct {
	Groups    map[uint]struct{}
	Protocols map[protocol.Protocol]struct{}
	Models    map[string]struct{}
}

type RouteTarget struct {
	GroupID         uint
	UpstreamModelID string
	EntryID         string
	Mode            channel.RouteMode
	ResolvedTarget  channel.ResolvedTarget
	EntryWeight     int // 条目权重,nil 归一为 1;0 保留条目但不参与分流
	Priority        int // 条目优先级,nil 归一为 1
}

// NoModelRouteKey identifies operations whose upstream resource ID, rather
// than a model, determines the target after affinity resolution.
const NoModelRouteKey = ""

// ExecutionCandidateIndex indexes targets by client protocol, logical
// operation, and external model. Resource operations use NoModelRouteKey.
type ExecutionCandidateIndex map[protocol.Protocol]map[execution.Operation]map[string][]RouteTarget

type TimeoutConfig struct {
	FirstByte  time.Duration
	Request    time.Duration
	StreamIdle time.Duration
}

type HeaderRules struct {
	Set    map[string]string
	Remove []string
}

type GroupView struct {
	ID                  uint
	Name                string
	ChannelID           channel.ID
	ConnectionType      string
	Params              json.RawMessage
	ResolvedTarget      channel.ResolvedTarget
	ValidationModel     string
	ClientProtocols     []protocol.Protocol
	Models              []ModelConfig
	Timeouts            TimeoutConfig
	HeaderRules         HeaderRules
	RetryCount          int
	BlacklistThreshold  int
	AffinityEnabled     bool
	WeightManual        *int
	Proxy               outboundproxy.Effective
	ParameterOverrides  parameteroverride.Rules
	ModelBreakerByEntry map[uint]map[string]*EntryCircuitBreaker
}

type GroupCatalogView struct {
	ID           uint
	Name         string
	Enabled      bool
	WeightManual *int
	Models       []ModelConfig
}

type AccessKeyView struct {
	ID               uint
	Name             string
	KeySuffix        string
	Status           AccessKeyStatus
	Filters          FilterSet
	ExpiresAtMS      *int64
	AllowedPeerCIDRs []netip.Prefix
	RPMLimit         int64
	CostLimitRules   []accessquota.Rule
}

type ConfigSnapshot struct {
	Revision              uint64
	Settings              RuntimeSettings
	ExecutionCandidates   ExecutionCandidateIndex
	ExecutionRouteCatalog ExecutionCandidateIndex
	Groups                map[uint]GroupView
	AccessKeysByHash      map[string]AccessKeyView
	GroupCatalog          map[uint]GroupCatalogView
	AccessKeysByID        map[uint]AccessKeyView
	GlobalProxy           outboundproxy.Effective
}

func Compile(input CompileInput) (*ConfigSnapshot, error) {
	if err := validateCompileInput(input); err != nil {
		return nil, err
	}
	runtimeSettings, err := ResolveRuntimeSettings(input.SystemSettings)
	if err != nil {
		return nil, err
	}
	globalProxy, err := outboundproxy.Resolve(nil, nil, input.GlobalProxy, input.EnvironmentProxy)
	if err != nil {
		return nil, fmt.Errorf("compile global proxy: %w", err)
	}

	snapshot := &ConfigSnapshot{
		Settings:              runtimeSettings,
		ExecutionCandidates:   make(ExecutionCandidateIndex),
		ExecutionRouteCatalog: make(ExecutionCandidateIndex),
		Groups:                make(map[uint]GroupView),
		AccessKeysByHash:      make(map[string]AccessKeyView),
		GroupCatalog:          make(map[uint]GroupCatalogView),
		AccessKeysByID:        make(map[uint]AccessKeyView),
		GlobalProxy:           globalProxy,
	}

	for _, group := range input.Groups {
		catalogView := GroupCatalogView{
			ID: group.ID, Name: group.Name, Enabled: group.Enabled,
			WeightManual: cloneWeight(group.WeightManual),
			Models:       cloneModelConfigs(group.Models),
		}
		snapshot.GroupCatalog[group.ID] = catalogView
		if err := appendExecutionTargets(snapshot.ExecutionRouteCatalog, input.ChannelRegistry, group); err != nil {
			return nil, err
		}
		resolved, err := ResolveGroupRuntimeSettings(runtimeSettings, group.Settings)
		if err != nil {
			return nil, fmt.Errorf("compile group %d settings: %w", group.ID, err)
		}
		groupProxy, err := outboundproxy.Resolve(nil, group.Proxy, input.GlobalProxy, input.EnvironmentProxy)
		if err != nil {
			return nil, fmt.Errorf("compile group %d proxy: %w", group.ID, err)
		}
		if !group.Enabled {
			continue
		}

		view := GroupView{
			ID:                  group.ID,
			Name:                group.Name,
			ValidationModel:     strings.TrimSpace(group.ValidationModel),
			Models:              cloneModelConfigs(group.Models),
			Timeouts:            resolved.Timeouts,
			HeaderRules:         resolved.HeaderRules,
			RetryCount:          resolved.RetryCount,
			BlacklistThreshold:  resolved.BlacklistThreshold,
			AffinityEnabled:     resolved.AffinityEnabled,
			WeightManual:        cloneWeight(group.WeightManual),
			ConnectionType:      connection.Normalize(group.ConnectionType),
			Proxy:               groupProxy,
			ParameterOverrides:  resolved.ParameterOverrides,
			ModelBreakerByEntry: make(map[uint]map[string]*EntryCircuitBreaker),
		}
		for _, model := range group.Models {
			if model.CircuitBreaker == nil {
				continue
			}
			if view.ModelBreakerByEntry[group.ID] == nil {
				view.ModelBreakerByEntry[group.ID] = make(map[string]*EntryCircuitBreaker)
			}
			external := ExternalModelName(model.ID, model.Alias)
			entryID := routeEntryIdentity(external, model)
			view.ModelBreakerByEntry[group.ID][entryID] = cloneEntryCircuitBreaker(model.CircuitBreaker)
		}
		params, err := input.ChannelRegistry.ValidateParams(group.ChannelID, group.Params)
		if err != nil {
			return nil, fmt.Errorf("compile group %d params: %w", group.ID, err)
		}
		target, err := input.ChannelRegistry.Resolve(group.ChannelID, group.Params)
		if err != nil {
			return nil, fmt.Errorf("compile group %d channel: %w", group.ID, err)
		}
		descriptor, _ := input.ChannelRegistry.Get(group.ChannelID)
		view.ChannelID = group.ChannelID
		view.Params = params.CanonicalJSON()
		view.ResolvedTarget = cloneResolvedTarget(target)
		view.ClientProtocols = append([]protocol.Protocol(nil), descriptor.ClientProtocols...)
		if err := appendExecutionTargets(snapshot.ExecutionCandidates, input.ChannelRegistry, group); err != nil {
			return nil, err
		}
		snapshot.Groups[group.ID] = view
	}

	for _, accessKey := range input.AccessKeys {
		snapshot.AccessKeysByID[accessKey.ID] = newAccessKeyView(accessKey)
		if accessKey.Status == AccessKeyStatusActive {
			snapshot.AccessKeysByHash[accessKey.KeyHash] = newAccessKeyView(accessKey)
		}
	}

	sortExecutionRouteIndex(snapshot.ExecutionCandidates)
	sortExecutionRouteIndex(snapshot.ExecutionRouteCatalog)
	return snapshot, nil
}

func newAccessKeyView(input AccessKeyConfig) AccessKeyView {
	rules := append([]accessquota.Rule(nil), input.CostLimitRules...)
	sort.Slice(rules, func(i, j int) bool {
		if rules[i].Kind != rules[j].Kind {
			return rules[i].Kind == accessquota.KindTotal
		}
		if rules[i].PeriodSeconds != rules[j].PeriodSeconds {
			return rules[i].PeriodSeconds < rules[j].PeriodSeconds
		}
		return rules[i].ID < rules[j].ID
	})
	return AccessKeyView{
		ID: input.ID, Name: input.Name, Status: input.Status,
		KeySuffix:        input.KeySuffix,
		Filters:          cloneFilterSet(input.Filters),
		ExpiresAtMS:      cloneAccessKeyExpiry(input.ExpiresAtMS),
		AllowedPeerCIDRs: cloneAllowedPeerCIDRs(input.AllowedPeerCIDRs),
		RPMLimit:         input.RPMLimit,
		CostLimitRules:   rules,
	}
}

// AccessQuotaDefinitions returns a caller-owned rules map for runtime reconciliation.
func (snapshot *ConfigSnapshot) AccessQuotaDefinitions() map[uint][]accessquota.Rule {
	definitions := make(map[uint][]accessquota.Rule)
	if snapshot == nil {
		return definitions
	}
	for accessKeyID, view := range snapshot.AccessKeysByID {
		if len(view.CostLimitRules) == 0 {
			continue
		}
		definitions[accessKeyID] = append([]accessquota.Rule(nil), view.CostLimitRules...)
	}
	return definitions
}

func appendExecutionTargets(
	index ExecutionCandidateIndex,
	registry *channel.Registry,
	group GroupConfig,
) error {
	target, err := registry.Resolve(group.ChannelID, group.Params)
	if err != nil {
		return fmt.Errorf("compile group %d channel: %w", group.ID, err)
	}
	descriptor, ok := registry.Get(group.ChannelID)
	if !ok {
		return fmt.Errorf("compile group %d channel: unknown channel %q", group.ID, group.ChannelID)
	}
	// 模型配置是分组进入数据面调度的统一门槛；无模型资源请求也不能绕过。
	if len(group.Models) == 0 {
		return nil
	}
	for _, clientProtocol := range descriptor.ClientProtocols {
		for _, operation := range target.Operations(clientProtocol) {
			if operation == execution.OperationListModels || operation == execution.OperationProbe {
				continue
			}
			mode, ok := target.Mode(clientProtocol, operation)
			if !ok {
				return fmt.Errorf("compile group %d channel has no route mode for %q/%q", group.ID, clientProtocol, operation)
			}
			switch operation {
			case execution.OperationResponsesRetrieve,
				execution.OperationResponsesDelete,
				execution.OperationResponsesCancel,
				execution.OperationResponsesInputItems:
				appendExecutionTarget(index, clientProtocol, operation, NoModelRouteKey, RouteTarget{
					GroupID: group.ID, Mode: mode, ResolvedTarget: cloneResolvedTarget(target),
					EntryWeight: 1, Priority: 1,
				})
			case execution.OperationResponsesPassthrough:
				appendExecutionTarget(index, clientProtocol, operation, NoModelRouteKey, RouteTarget{
					GroupID: group.ID, Mode: mode, ResolvedTarget: cloneResolvedTarget(target),
					EntryWeight: 1, Priority: 1,
				})
				fallthrough
			case execution.OperationChatCompletion,
				execution.OperationResponsesCreate,
				execution.OperationResponsesCompact,
				execution.OperationResponsesInputTokens,
				execution.OperationCountTokens,
				execution.OperationImagesGenerate,
				execution.OperationImagesEdit,
				execution.OperationEmbeddingsCreate:
				// 每个模型条目产生一个 target:同一分组同一对外名可以产生多个
				// target(多映射,V1 只约束 (对外名, 上游模型) 组合唯一)。
				for _, model := range group.Models {
					modelMode, supported := target.ModeForModel(clientProtocol, operation, model.ID)
					if !supported {
						return fmt.Errorf("compile group %d channel has no route mode for %q/%q model %q", group.ID, clientProtocol, operation, model.ID)
					}
					external := ExternalModelName(model.ID, model.Alias)
					appendExecutionTarget(index, clientProtocol, operation, external, RouteTarget{
						GroupID: group.ID, UpstreamModelID: strings.TrimSpace(model.ID),
						Mode: modelMode, ResolvedTarget: cloneResolvedTarget(target),
						EntryWeight: normalizeRouteEntryValue(model.Weight),
						Priority:    normalizeRouteEntryValue(model.Priority),
						EntryID:     routeEntryIdentity(external, model),
					})
				}
			default:
				return fmt.Errorf("compile group %d channel has unsupported routable operation %q", group.ID, operation)
			}
		}
	}
	return nil
}

func appendExecutionTarget(
	index ExecutionCandidateIndex,
	clientProtocol protocol.Protocol,
	operation execution.Operation,
	externalModel string,
	target RouteTarget,
) {
	if index[clientProtocol] == nil {
		index[clientProtocol] = make(map[execution.Operation]map[string][]RouteTarget)
	}
	if index[clientProtocol][operation] == nil {
		index[clientProtocol][operation] = make(map[string][]RouteTarget)
	}
	index[clientProtocol][operation][externalModel] = append(
		index[clientProtocol][operation][externalModel],
		target,
	)
}

func routeEntryIdentity(external string, model ModelConfig) string {
	if model.EntryID != "" {
		return model.EntryID
	}
	return "derived:" + external + "#" + strings.TrimSpace(model.ID)
}

// back to the design default of 1 (weight and priority, design §3). An
// explicit 0 weight survives normalization: the target stays indexed but is
// excluded from traffic splitting by the scheduler (design §4).
func normalizeRouteEntryValue(value *int) int {
	if value == nil {
		return 1
	}
	return *value
}

func cloneModelConfigs(models []ModelConfig) []ModelConfig {
	if models == nil {
		return nil
	}
	cloned := make([]ModelConfig, len(models))
	for index, model := range models {
		cloned[index] = ModelConfig{
			ID: model.ID, Alias: model.Alias, EntryID: model.EntryID,
			Weight: cloneWeight(model.Weight), Priority: cloneWeight(model.Priority),
			CircuitBreaker: cloneEntryCircuitBreaker(model.CircuitBreaker),
		}
	}
	return cloned
}

func cloneResolvedTarget(target channel.ResolvedTarget) channel.ResolvedTarget {
	target.TargetConfig = append(json.RawMessage(nil), target.TargetConfig...)
	return target
}

func sortExecutionRouteIndex(index ExecutionCandidateIndex) {
	for _, byOperation := range index {
		for _, byModel := range byOperation {
			for model := range byModel {
				// 稳定排序:同键 target 保持编译期条目顺序,保证快照可复现。
				sort.SliceStable(byModel[model], func(i, j int) bool {
					left, right := byModel[model][i], byModel[model][j]
					if left.Priority != right.Priority {
						return left.Priority < right.Priority
					}
					if left.GroupID != right.GroupID {
						return left.GroupID < right.GroupID
					}
					return left.UpstreamModelID < right.UpstreamModelID
				})
			}
		}
	}
}

func validateCompileInput(input CompileInput) error {
	groupIDs := make(map[uint]struct{}, len(input.Groups))
	for _, group := range input.Groups {
		if group.ID == 0 {
			return fmt.Errorf("group id is required")
		}
		if _, duplicate := groupIDs[group.ID]; duplicate {
			return fmt.Errorf("duplicate group id %d", group.ID)
		}
		groupIDs[group.ID] = struct{}{}
		if input.ChannelRegistry == nil {
			return fmt.Errorf("group %d channel registry is required", group.ID)
		}
		if group.ChannelID == "" {
			return fmt.Errorf("group %d channel id is required", group.ID)
		}
		if _, ok := input.ChannelRegistry.Get(group.ChannelID); !ok {
			return fmt.Errorf("group %d has unknown channel %q", group.ID, group.ChannelID)
		}
		connectionType := connection.Normalize(group.ConnectionType)
		if !input.ChannelRegistry.SupportsConnectionType(group.ChannelID, connectionType) {
			return fmt.Errorf("group %d channel %q does not support connection type %q", group.ID, group.ChannelID, connectionType)
		}
		if _, err := input.ChannelRegistry.Resolve(group.ChannelID, group.Params); err != nil {
			return fmt.Errorf("group %d channel %q: %w", group.ID, group.ChannelID, err)
		}
		if err := validateManualWeight(fmt.Sprintf("group %d", group.ID), group.WeightManual); err != nil {
			return err
		}
		// 路由条目规则 V1–V4(设计 §3)在编译期作为最终防线再次执行;
		// 违规拒绝发布,错误信息携带分组与模型名。
		if err := ValidateModelRouteEntries(fmt.Sprintf("group %d", group.ID), group.Models); err != nil {
			return err
		}
	}

	credentialIDs := make(map[uint]struct{}, len(input.Credentials))
	for _, credential := range input.Credentials {
		if credential.ID == 0 {
			return fmt.Errorf("credential id is required")
		}
		if _, duplicate := credentialIDs[credential.ID]; duplicate {
			return fmt.Errorf("duplicate credential id %d", credential.ID)
		}
		credentialIDs[credential.ID] = struct{}{}
		if credential.GroupID == 0 {
			return fmt.Errorf("credential %d group id is required", credential.ID)
		}
		if _, ok := groupIDs[credential.GroupID]; !ok {
			return fmt.Errorf("credential %d belongs to unknown group %d", credential.ID, credential.GroupID)
		}
		switch credential.Status {
		case CredentialStatusActive, CredentialStatusDisabled:
		default:
			return fmt.Errorf("credential %d has invalid status %q", credential.ID, credential.Status)
		}
		if err := validateManualWeight(fmt.Sprintf("credential %d", credential.ID), credential.WeightManual); err != nil {
			return err
		}
		if credential.Version == 0 {
			return fmt.Errorf("credential %d version is required", credential.ID)
		}
		if credential.IdentityGeneration == 0 {
			return fmt.Errorf("credential %d identity generation is required", credential.ID)
		}
		if strings.TrimSpace(credential.Fingerprint) == "" {
			return fmt.Errorf("credential %d fingerprint is required", credential.ID)
		}
	}

	accessKeyIDs := make(map[uint]struct{}, len(input.AccessKeys))
	hashes := make(map[string]struct{}, len(input.AccessKeys))
	quotaDefinitions := make(map[uint][]accessquota.Rule)
	for _, accessKey := range input.AccessKeys {
		if accessKey.ID == 0 {
			return fmt.Errorf("access key id is required")
		}
		if _, duplicate := accessKeyIDs[accessKey.ID]; duplicate {
			return fmt.Errorf("duplicate access key id %d", accessKey.ID)
		}
		accessKeyIDs[accessKey.ID] = struct{}{}
		if accessKey.RPMLimit < 0 {
			return fmt.Errorf("access key %d rpm limit must not be negative", accessKey.ID)
		}
		if accessKey.ExpiresAtMS != nil &&
			(*accessKey.ExpiresAtMS < 0 || *accessKey.ExpiresAtMS > maxSafeAccessKeyEpochMS) {
			return fmt.Errorf("access key %d expiry must be a safe millisecond value", accessKey.ID)
		}
		if err := validateAllowedPeerCIDRs(accessKey.ID, accessKey.AllowedPeerCIDRs); err != nil {
			return err
		}
		switch accessKey.Status {
		case AccessKeyStatusActive, AccessKeyStatusDisabled:
		default:
			return fmt.Errorf("access key %d has invalid status %q", accessKey.ID, accessKey.Status)
		}
		if strings.TrimSpace(accessKey.KeyHash) == "" {
			return fmt.Errorf("access key %d key hash is required", accessKey.ID)
		}
		if _, duplicate := hashes[accessKey.KeyHash]; duplicate {
			return fmt.Errorf("duplicate access key hash %q", accessKey.KeyHash)
		}
		hashes[accessKey.KeyHash] = struct{}{}
		if err := validateFilterSet(accessKey.ID, accessKey.Filters); err != nil {
			return err
		}
		if len(accessKey.CostLimitRules) > 0 {
			quotaDefinitions[accessKey.ID] = accessKey.CostLimitRules
		}
	}
	if err := accessquota.ValidateDefinitions(quotaDefinitions); err != nil {
		return fmt.Errorf("validate access key cost limit rules: %w", err)
	}
	return nil
}

func validateFilterSet(accessKeyID uint, filters FilterSet) error {
	for p := range filters.Protocols {
		if !p.Valid() {
			return fmt.Errorf("access key %d filter has invalid protocol %q", accessKeyID, p)
		}
	}
	for model := range filters.Models {
		if strings.TrimSpace(model) == "" {
			return fmt.Errorf("access key %d filter model is required", accessKeyID)
		}
	}
	return nil
}

func cloneFilterSet(source FilterSet) FilterSet {
	cloned := FilterSet{}
	if source.Groups != nil {
		cloned.Groups = make(map[uint]struct{}, len(source.Groups))
		for id := range source.Groups {
			cloned.Groups[id] = struct{}{}
		}
	}
	if source.Protocols != nil {
		cloned.Protocols = make(map[protocol.Protocol]struct{}, len(source.Protocols))
		for p := range source.Protocols {
			cloned.Protocols[p] = struct{}{}
		}
	}
	if source.Models != nil {
		cloned.Models = make(map[string]struct{}, len(source.Models))
		for model := range source.Models {
			cloned.Models[model] = struct{}{}
		}
	}
	return cloned
}

func validateAllowedPeerCIDRs(accessKeyID uint, prefixes []netip.Prefix) error {
	if len(prefixes) > 64 {
		return fmt.Errorf("access key %d allowed peer CIDR count exceeds limit", accessKeyID)
	}
	seen := make(map[netip.Prefix]struct{}, len(prefixes))
	for _, prefix := range prefixes {
		if !prefix.IsValid() || prefix.Addr().Zone() != "" || prefix.Addr().Is4In6() || prefix != prefix.Masked() {
			return fmt.Errorf("access key %d has invalid allowed peer CIDR", accessKeyID)
		}
		if _, duplicate := seen[prefix]; duplicate {
			return fmt.Errorf("access key %d has duplicate allowed peer CIDR", accessKeyID)
		}
		seen[prefix] = struct{}{}
	}
	return nil
}

func cloneAccessKeyExpiry(source *int64) *int64 {
	if source == nil {
		return nil
	}
	cloned := *source
	return &cloned
}

func cloneAllowedPeerCIDRs(source []netip.Prefix) []netip.Prefix {
	if source == nil {
		return nil
	}
	return append(make([]netip.Prefix, 0, len(source)), source...)
}
