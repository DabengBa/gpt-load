// Central model-route schedule management API (design
// docs/design/model-route-central-scheduling.md §5): cross-group aggregation
// by external model name, per-context detail with entry-level circuit breaker
// state, transactional tri-state PATCH with optimistic snapshot revision, and
// per-entry runtime recovery.
package control

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/protocol"
	"gpt-load/internal/reasoning"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
)

// scheduleBreakerSourceDefault marks a breaker parameter resolved from the
// built-in behavior instead of an explicit entry override (§5.3).
const scheduleBreakerSourceDefault = "default"

// scheduleBreakerSourceEntry marks a breaker parameter explicitly configured
// on the route entry (§5.3).
const scheduleBreakerSourceEntry = "entry"

// scheduleDefaultModelCooldownSeconds is the effective model-level cooldown
// when the entry does not override it: the judge's model-unavailable default
// of one hour (internal/health/execution_judge.go).
const scheduleDefaultModelCooldownSeconds = 3600

// modelRouteScheduleRevisionConflict rejects a schedule PATCH whose optimistic
// snapshot_revision no longer matches the published snapshot (§5.5).
var modelRouteScheduleRevisionConflict = &app_errors.APIError{
	HTTPStatus: http.StatusConflict,
	Code:       "MODEL_ROUTE_SCHEDULE_REVISION_CONFLICT",
	Message:    "Snapshot revision changed, reload and retry",
}

var modelRouteScheduleRuntimeConflict = &app_errors.APIError{
	HTTPStatus: http.StatusConflict,
	Code:       "MODEL_ROUTE_SCHEDULE_RUNTIME_CONFLICT",
	Message:    "Route entry runtime changed, reload and retry",
}

type modelRouteScheduleIndexItem struct {
	ExternalModel         string              `json:"external_model"`
	Protocol              protocol.Protocol   `json:"protocol"`
	Operation             execution.Operation `json:"operation"`
	CandidateCount        int                 `json:"candidate_count"`
	GroupCount            int                 `json:"group_count"`
	CooledCandidates      int                 `json:"cooled_candidates"`
	BlacklistedCandidates int                 `json:"blacklisted_candidates"`
}

type modelRouteScheduleIndexResponse struct {
	Items []modelRouteScheduleIndexItem `json:"items"`
}

type modelRouteScheduleDetailRequest struct {
	Protocol      protocol.Protocol
	ExternalModel string
	Operation     execution.Operation
	AccessKeyID   uint
}

type scheduleBreakerParameterView struct {
	BlacklistThreshold *int `json:"blacklist_threshold"`
	CooldownSeconds    *int `json:"cooldown_seconds"`
}

type scheduleBreakerSourcesView struct {
	BlacklistThreshold string `json:"blacklist_threshold"`
	CooldownSeconds    string `json:"cooldown_seconds"`
}

// scheduleBreakerView is the §5.3 response triple: configured (explicit entry
// override, null when absent), effective (resolved values) and per-parameter
// entry|default sources.
type scheduleBreakerView struct {
	Configured scheduleBreakerParameterView `json:"configured"`
	Effective  scheduleBreakerParameterView `json:"effective"`
	Sources    scheduleBreakerSourcesView   `json:"sources"`
}

type scheduleEntryRuntimeView struct {
	State                state.EntryRuntimeState `json:"state"`
	CooldownUntilMS      *int64                  `json:"cooldown_until_ms"`
	BlacklistReleaseAtMS *int64                  `json:"blacklist_release_at_ms"`
	FailureCount         int                     `json:"failure_count"`
	FailureVersion       uint64                  `json:"failure_version"`
}

type modelRouteScheduleEntryResponse struct {
	EntryID  string `json:"entry_id"`
	ModelID  string `json:"model_id"`
	Alias    string `json:"alias"`
	Weight   int    `json:"weight"`
	Priority int    `json:"priority"`
	// ConcurrencyLimit is the entry's in-flight cap; 0 means unlimited.
	ConcurrencyLimit int                              `json:"concurrency_limit"`
	CircuitBreaker   scheduleBreakerView              `json:"circuit_breaker"`
	Reasoning        scheduleReasoningView            `json:"reasoning"`
	Runtime          scheduleEntryRuntimeView         `json:"runtime"`
	Enabled          bool                             `json:"enabled"`
	Included         bool                             `json:"included"`
	Routable         bool                             `json:"routable"`
	ReasonCode       *scheduler.ReasonCode            `json:"reason_code"`
	ConfiguredShare  float64                          `json:"configured_share"`
	EffectiveShare   float64                          `json:"effective_share"`
	Credentials      []routeInspectCredentialResponse `json:"credentials"`
}

type modelRouteScheduleGroupResponse struct {
	GroupID      uint                              `json:"group_id"`
	GroupName    string                            `json:"group_name"`
	ChannelID    channel.ID                        `json:"channel_id"`
	Enabled      bool                              `json:"enabled"`
	RequestCount int64                             `json:"request_count"`
	SuccessRate  float64                           `json:"success_rate"`
	Entries      []modelRouteScheduleEntryResponse `json:"entries"`
}

type scheduleReasoningView struct {
	Configured *string                `json:"configured"`
	Effective  *string                `json:"effective"`
	Source     reasoning.EffortSource `json:"source"`
}

type modelRouteScheduleDetailResponse struct {
	ObservedAtMS     int64                             `json:"observed_at_ms"`
	SnapshotRevision uint64                            `json:"snapshot_revision"`
	ExternalModel    *string                           `json:"external_model"`
	Protocol         protocol.Protocol                 `json:"protocol"`
	Operation        execution.Operation               `json:"operation"`
	RouteRequirement execution.RouteRequirement        `json:"route_requirement"`
	AccessKey        *routeInspectAccessKeyResponse    `json:"access_key,omitempty"`
	Routable         bool                              `json:"routable"`
	ReasonCode       *scheduler.ReasonCode             `json:"reason_code"`
	Groups           []modelRouteScheduleGroupResponse `json:"groups"`
}

func scheduleRouteRequirement(
	clientProtocol protocol.Protocol,
	operation execution.Operation,
	defaultRequirement execution.RouteRequirement,
) execution.RouteRequirement {
	if clientProtocol == protocol.OpenAIResponses {
		switch operation {
		case execution.OperationResponsesRetrieve,
			execution.OperationResponsesDelete,
			execution.OperationResponsesCancel,
			execution.OperationResponsesInputItems,
			execution.OperationResponsesPassthrough:
			return execution.RouteRequirementNative
		}
	}
	return defaultRequirement
}

func preferScheduleContext(candidate protocol.Protocol, current protocol.Protocol) bool {
	order := map[protocol.Protocol]int{
		protocol.OpenAICompletions: 0,
		protocol.OpenAIResponses:   1,
		protocol.Anthropic:         2,
		protocol.Gemini:            3,
		protocol.OpenAIImages:      4,
		protocol.OpenAIEmbeddings:  5,
	}
	candidateRank, candidateKnown := order[candidate]
	currentRank, currentKnown := order[current]
	if candidateKnown != currentKnown {
		return candidateKnown
	}
	if candidateKnown && candidateRank != currentRank {
		return candidateRank < currentRank
	}
	return string(candidate) < string(current)
}

// GetModelRouteScheduleIndex aggregates the real route candidates of the
// current snapshot across groups by external model name (§5.1). Targets are
// deduplicated per (group, entry) so the operations of one protocol do not
// multiply the counts, and runtime badges come from the entry runtime only.
func (s *Service) GetModelRouteScheduleIndex() (modelRouteScheduleIndexResponse, error) {
	observation, err := s.captureRuntimeObservation()
	if err != nil {
		return modelRouteScheduleIndexResponse{}, err
	}
	runtimeByKey := scheduleRuntimeByKey(s.registry.EntryRuntimeSnapshot())
	// Test aliases select one provider/model for client requests; they are not dispatch-center contexts.
	testAliases := make(map[string]struct{})
	for _, group := range observation.snapshot.GroupCatalog {
		for _, model := range group.Models {
			if model.TestAlias != "" {
				testAliases[model.TestAlias] = struct{}{}
			}
		}
	}

	type scheduleCandidateKey struct {
		groupID uint
		entryID string
	}
	type scheduleIndexContext struct {
		protocol  protocol.Protocol
		operation execution.Operation
	}
	type scheduleIndexAccumulator struct {
		candidates map[scheduleCandidateKey]struct{}
		groups     map[uint]struct{}
		cooled     int
		blacklist  int
	}
	items := make(map[string]map[scheduleIndexContext]*scheduleIndexAccumulator)
	for protocolKey, byOperation := range observation.snapshot.ExecutionRouteCatalog {
		for operationKey, byModel := range byOperation {
			for external, targets := range byModel {
				if _, hidden := testAliases[external]; hidden {
					continue
				}
				if external == state.NoModelRouteKey || len(targets) == 0 {
					continue
				}
				contexts, exists := items[external]
				if !exists {
					contexts = make(map[scheduleIndexContext]*scheduleIndexAccumulator)
					items[external] = contexts
				}
				context := scheduleIndexContext{protocol: protocolKey, operation: operationKey}
				accumulator, exists := contexts[context]
				if !exists {
					accumulator = &scheduleIndexAccumulator{
						candidates: make(map[scheduleCandidateKey]struct{}),
						groups:     make(map[uint]struct{}),
					}
					contexts[context] = accumulator
				}
				for _, target := range targets {
					key := scheduleCandidateKey{groupID: target.GroupID, entryID: target.EntryID}
					if _, duplicate := accumulator.candidates[key]; duplicate {
						continue
					}
					accumulator.candidates[key] = struct{}{}
					accumulator.groups[target.GroupID] = struct{}{}
					view, hasRuntime := runtimeByKey[state.RouteEntryKey{
						GroupID: target.GroupID, EntryID: target.EntryID,
					}]
					if !hasRuntime {
						continue
					}
					switch view.RuntimeState(observation.observedAt) {
					case state.EntryRuntimeCooldown:
						accumulator.cooled++
					case state.EntryRuntimeBlacklisted:
						accumulator.blacklist++
					}
				}
			}
		}
	}

	result := modelRouteScheduleIndexResponse{
		Items: make([]modelRouteScheduleIndexItem, 0, len(items)),
	}
	for external, contexts := range items {
		var selectedContext scheduleIndexContext
		var selected *scheduleIndexAccumulator
		for context, accumulator := range contexts {
			if selected == nil || preferScheduleContext(context.protocol, selectedContext.protocol) ||
				(context.protocol == selectedContext.protocol && string(context.operation) < string(selectedContext.operation)) {
				selectedContext = context
				selected = accumulator
			}
		}
		result.Items = append(result.Items, modelRouteScheduleIndexItem{
			ExternalModel:         external,
			Protocol:              selectedContext.protocol,
			Operation:             selectedContext.operation,
			CandidateCount:        len(selected.candidates),
			GroupCount:            len(selected.groups),
			CooledCandidates:      selected.cooled,
			BlacklistedCandidates: selected.blacklist,
		})
	}
	sort.Slice(result.Items, func(i, j int) bool {
		return result.Items[i].ExternalModel < result.Items[j].ExternalModel
	})
	return result, nil
}

func validateModelRouteScheduleDetailRequest(request modelRouteScheduleDetailRequest) error {
	if !request.Protocol.DataPlaneEnabled() ||
		!validUsageModel(request.ExternalModel) {
		return app_errors.ErrValidation
	}
	if request.Operation != "" && !request.Operation.Valid() {
		return app_errors.ErrValidation
	}
	if _, err := dialect.InspectStandardRequest(request.Protocol, request.ExternalModel); err != nil {
		return app_errors.ErrValidation
	}
	return nil
}

// parseModelRouteScheduleDetailQuery reads the §5.2 detail query context:
// required protocol/external_model plus the optional access_key_id scope and
// the optional operation that defaults to the protocol's standard operation.
func parseModelRouteScheduleDetailQuery(c *gin.Context) (modelRouteScheduleDetailRequest, error) {
	request := modelRouteScheduleDetailRequest{
		Protocol:      protocol.Protocol(strings.TrimSpace(c.Query("protocol"))),
		ExternalModel: c.Query("external_model"),
		Operation:     execution.Operation(strings.TrimSpace(c.Query("operation"))),
	}
	if rawAccessKeyID := strings.TrimSpace(c.Query("access_key_id")); rawAccessKeyID != "" {
		parsed, err := parseCanonicalSafePlatformUint(rawAccessKeyID)
		if err != nil || parsed == 0 {
			return modelRouteScheduleDetailRequest{}, app_errors.ErrValidation
		}
		request.AccessKeyID = parsed
	}
	if err := validateModelRouteScheduleDetailRequest(request); err != nil {
		return modelRouteScheduleDetailRequest{}, err
	}
	return request, nil
}

// GetModelRouteScheduleDetail returns the real inspection candidates for one
// external model under a protocol/operation context (§5.2), each row carrying
// the entry identity, the §5.3 breaker triple and the entry runtime state.
// An optional access_key_id scopes the inspection through that key's filters;
// when omitted the dispatch center observes the unfiltered candidate table.
func (s *Service) GetModelRouteScheduleDetail(
	request modelRouteScheduleDetailRequest,
) (modelRouteScheduleDetailResponse, error) {
	if err := validateModelRouteScheduleDetailRequest(request); err != nil {
		return modelRouteScheduleDetailResponse{}, err
	}
	metadata, err := dialect.InspectStandardRequest(request.Protocol, request.ExternalModel)
	if err != nil || metadata.Model == nil {
		return modelRouteScheduleDetailResponse{}, app_errors.ErrValidation
	}
	observation, err := s.captureRuntimeObservation()
	if err != nil {
		return modelRouteScheduleDetailResponse{}, err
	}
	// Access-key scope is optional: the dispatch center sends none, so the
	// inspection runs against an unfiltered active key view.
	accessKey := state.AccessKeyView{Status: state.AccessKeyStatusActive}
	if request.AccessKeyID != 0 {
		scoped, exists := observation.snapshot.AccessKeysByID[request.AccessKeyID]
		if !exists {
			return modelRouteScheduleDetailResponse{}, app_errors.ErrResourceNotFound
		}
		accessKey = scoped
	}
	operation := metadata.Operation
	if request.Operation != "" {
		operation = request.Operation
	}
	routeRequirement := scheduleRouteRequirement(request.Protocol, operation, metadata.RouteRequirement)
	entryRuntime := s.registry.EntryRuntimeSnapshot()
	groupUsage, err := s.queryModelRouteScheduleGroupUsage(observation.observedAt)
	if err != nil {
		return modelRouteScheduleDetailResponse{}, err
	}
	explanation := scheduler.Inspection{
		ClientProtocol:   request.Protocol,
		Operation:        operation,
		RouteRequirement: routeRequirement,
		ExternalModel:    cloneRouteModel(metadata.Model),
		Reason:           scheduler.ReasonAccessKeyExpired,
		Groups:           []scheduler.GroupInspection{},
	}
	if accessKey.ExpiresAtMS == nil ||
		observation.observedAt.UnixMilli() < *accessKey.ExpiresAtMS {
		explanation, err = scheduler.InspectWithEntryRuntime(
			observation.snapshot,
			observation.keys,
			entryRuntime,
			scheduler.Query{
				ClientProtocol:   request.Protocol,
				Operation:        operation,
				RouteRequirement: routeRequirement,
				ExternalModel:    cloneRouteModel(metadata.Model),
				AccessKey:        accessKey,
			},
			observation.observedAt,
		)
		if err != nil {
			if errors.Is(err, scheduler.ErrInconsistentSnapshot) {
				return modelRouteScheduleDetailResponse{}, fmt.Errorf(
					"inspect model route schedule: %w",
					app_errors.ErrInternalServer,
				)
			}
			return modelRouteScheduleDetailResponse{}, err
		}
	}
	return mapModelRouteScheduleDetail(
		observation, request, accessKey, explanation, entryRuntime, groupUsage,
	)
}

func mapModelRouteScheduleDetail(
	observation runtimeObservation,
	request modelRouteScheduleDetailRequest,
	accessKey state.AccessKeyView,
	explanation scheduler.Inspection,
	entryRuntime []state.EntryRuntimeView,
	groupUsage map[uint]requestlog.GroupUsage,
) (modelRouteScheduleDetailResponse, error) {
	observedAtMS, err := safeEpochMilliseconds(observation.observedAt)
	if err != nil {
		return modelRouteScheduleDetailResponse{}, fmt.Errorf(
			"map model route schedule observed_at_ms: %w", err,
		)
	}
	configurations := scheduleEntryConfigurations(observation.snapshot)
	runtimeByKey := scheduleRuntimeByKey(entryRuntime)
	result := modelRouteScheduleDetailResponse{
		ObservedAtMS:     observedAtMS,
		SnapshotRevision: observation.snapshot.Revision,
		ExternalModel:    cloneRouteModel(explanation.ExternalModel),
		Protocol:         request.Protocol,
		Operation:        explanation.Operation,
		RouteRequirement: explanation.RouteRequirement,
		Routable:         explanation.Routable,
		ReasonCode:       optionalReason(explanation.Reason),
		Groups:           []modelRouteScheduleGroupResponse{},
	}
	if request.AccessKeyID != 0 {
		result.AccessKey = &routeInspectAccessKeyResponse{
			ID: accessKey.ID, Name: accessKey.Name, Status: accessKey.Status,
		}
	}
	groupIndex := make(map[uint]int, len(explanation.Groups))
	for _, group := range explanation.Groups {
		if _, exists := groupIndex[group.GroupID]; exists {
			continue
		}
		catalog, exists := observation.snapshot.GroupCatalog[group.GroupID]
		if !exists {
			return modelRouteScheduleDetailResponse{}, fmt.Errorf(
				"map model route schedule group %d: %w",
				group.GroupID, app_errors.ErrInternalServer,
			)
		}
		usage := groupUsage[group.GroupID]
		successRate := float64(0)
		if usage.RequestCount > 0 {
			successRate = float64(usage.SuccessCount) / float64(usage.RequestCount)
		}
		result.Groups = append(result.Groups, modelRouteScheduleGroupResponse{
			GroupID: group.GroupID, GroupName: group.GroupName, ChannelID: group.ChannelID,
			Enabled: catalog.Enabled, RequestCount: usage.RequestCount, SuccessRate: successRate,
			Entries: []modelRouteScheduleEntryResponse{},
		})
		groupIndex[group.GroupID] = len(result.Groups) - 1
	}
	// Entries are appended below after the group metadata is initialized.
	configuredShares := configuredEntryShares(explanation.Groups)
	for index, group := range explanation.Groups {
		credentials, err := mapScheduleCredentials(group)
		if err != nil {
			return modelRouteScheduleDetailResponse{}, err
		}
		entryCooldownUntilMS, err := optionalSafeEpochMilliseconds(group.EntryCooldownUntil)
		if err != nil {
			return modelRouteScheduleDetailResponse{}, fmt.Errorf(
				"map model route schedule entry_cooldown_until_ms: %w", err,
			)
		}
		// Disabled groups have no GroupView in the snapshot, so their alias
		// and configured breaker are unavailable; model_id comes from the
		// inspection row which always carries the upstream model.
		configuration := configurations[group.GroupID][group.EntryID]
		upstreamModel := ""
		if group.UpstreamModelID != nil {
			upstreamModel = *group.UpstreamModelID
		}
		runtimeView, err := scheduleEntryRuntime(
			runtimeByKey, group, entryCooldownUntilMS, observation.observedAt,
		)
		if err != nil {
			return modelRouteScheduleDetailResponse{}, err
		}
		entry := modelRouteScheduleEntryResponse{
			EntryID:  group.EntryID,
			ModelID:  upstreamModel,
			Alias:    configuration.alias,
			Enabled:  configuration.enabled,
			Weight:   group.EntryWeight,
			Priority: group.Priority,
			CircuitBreaker: scheduleBreakerViewFromConfiguration(
				configuration.circuitBreaker,
			),
			ConcurrencyLimit: configuration.concurrencyLimit,
			Runtime:          runtimeView,
			Included:         group.Included,
			Routable:         group.Routable,
			ReasonCode:       optionalReason(group.Reason),
			ConfiguredShare:  configuredShares[index],
			EffectiveShare:   group.EffectiveShare,
			Credentials:      credentials,
		}
		entry.Reasoning, err = scheduleReasoningProjection(configuration)
		if err != nil {
			return modelRouteScheduleDetailResponse{}, fmt.Errorf("map model route schedule reasoning: %w", app_errors.ErrInternalServer)
		}
		position := groupIndex[group.GroupID]
		result.Groups[position].Entries = append(result.Groups[position].Entries, entry)
	}
	return result, nil
}

func (s *Service) queryModelRouteScheduleGroupUsage(observedAt time.Time) (map[uint]requestlog.GroupUsage, error) {
	result := make(map[uint]requestlog.GroupUsage)
	reader, ok := s.requestLogs.(requestlog.GroupUsageReader)
	if !ok || reader == nil {
		return result, nil
	}
	toMS, err := safeEpochMilliseconds(observedAt)
	if err != nil {
		return nil, fmt.Errorf("query model route schedule group usage end: %w", err)
	}
	fromMS, err := safeEpochMilliseconds(observedAt.Add(-24 * time.Hour))
	if err != nil {
		return nil, fmt.Errorf("query model route schedule group usage start: %w", err)
	}
	usage, err := reader.QueryGroupUsage(context.Background(), requestlog.GroupUsageQuery{
		FromMS: fromMS,
		ToMS:   toMS,
	})
	if err != nil {
		return nil, fmt.Errorf("query model route schedule group usage: %w", err)
	}
	for groupID, value := range usage {
		if groupID == 0 || value.RequestCount < 0 || value.RequestCount > maxSafeInteger ||
			value.SuccessCount < 0 || value.SuccessCount > maxSafeInteger ||
			value.SuccessCount > value.RequestCount {
			return nil, fmt.Errorf("query model route schedule group usage: invalid aggregate")
		}
		result[groupID] = value
	}
	return result, nil
}

func mapScheduleCredentials(
	group scheduler.GroupInspection,
) ([]routeInspectCredentialResponse, error) {
	credentials := make([]routeInspectCredentialResponse, 0, len(group.Credentials))
	for _, credential := range group.Credentials {
		cooldownUntilMS, err := optionalSafeEpochMilliseconds(credential.CooldownUntil)
		if err != nil {
			return nil, fmt.Errorf("map model route schedule cooldown_until_ms: %w", err)
		}
		credentials = append(credentials, routeInspectCredentialResponse{
			CredentialID:    credential.CredentialID,
			Available:       credential.Available,
			ReasonCode:      optionalReason(credential.Reason),
			CooldownUntilMS: cooldownUntilMS,
		})
	}
	return credentials, nil
}

func scheduleRuntimeByKey(
	views []state.EntryRuntimeView,
) map[state.RouteEntryKey]state.EntryRuntimeView {
	result := make(map[state.RouteEntryKey]state.EntryRuntimeView, len(views))
	for _, view := range views {
		if view.Key.GroupID == 0 || view.Key.EntryID == "" {
			continue
		}
		result[view.Key] = view
	}
	return result
}

func scheduleEntryRuntime(
	runtimeByKey map[state.RouteEntryKey]state.EntryRuntimeView,
	group scheduler.GroupInspection,
	entryCooldownUntilMS *int64,
	observedAt time.Time,
) (scheduleEntryRuntimeView, error) {
	result := scheduleEntryRuntimeView{
		State:           state.EntryRuntimeAvailable,
		CooldownUntilMS: entryCooldownUntilMS,
	}
	view, exists := runtimeByKey[state.RouteEntryKey{
		GroupID: group.GroupID, EntryID: group.EntryID,
	}]
	if !exists {
		return result, nil
	}
	blacklistReleaseAtMS, err := optionalSafeEpochMilliseconds(view.BlacklistReleaseAt)
	if err != nil {
		return scheduleEntryRuntimeView{}, fmt.Errorf(
			"map model route schedule blacklist_release_at_ms: %w", err,
		)
	}
	result.FailureCount = view.FailureCount
	result.FailureVersion = view.FailureVersion
	result.BlacklistReleaseAtMS = blacklistReleaseAtMS
	result.State = view.RuntimeState(observedAt)
	return result, nil
}

type scheduleEntryConfiguration struct {
	alias            string
	enabled          bool
	circuitBreaker   *state.EntryCircuitBreaker
	reasoningEffort  string
	concurrencyLimit int
}

// scheduleEntryConfigurations indexes every snapshot catalog group's models by
// route entry identity (real entry_id or the derived identity of design §2.2-2)
// so detail rows surface persisted alias and breaker configuration even when a
// group is disabled and therefore absent from snapshot.Groups.
func scheduleEntryConfigurations(
	snapshot *state.ConfigSnapshot,
) map[uint]map[string]scheduleEntryConfiguration {
	result := make(map[uint]map[string]scheduleEntryConfiguration)
	if snapshot == nil {
		return result
	}
	for groupID, group := range snapshot.GroupCatalog {
		for _, model := range group.Models {
			upstream := strings.TrimSpace(model.ID)
			external := state.ExternalModelName(upstream, model.Alias)
			identity := model.EntryID
			if identity == "" {
				identity = "derived:" + external + "#" + upstream
			}
			if result[groupID] == nil {
				result[groupID] = make(map[string]scheduleEntryConfiguration)
			}
			result[groupID][identity] = scheduleEntryConfiguration{
				alias:            model.Alias,
				enabled:          model.Enabled == nil || *model.Enabled,
				circuitBreaker:   cloneEntryCircuitBreaker(model.CircuitBreaker),
				reasoningEffort:  model.ReasoningEffort,
				concurrencyLimit: model.ConcurrencyLimit,
			}
		}
	}
	return result
}

func optionalString(value string) *string {
	if value == "" {
		return nil
	}
	return new(value)
}

func scheduleReasoningProjection(configuration scheduleEntryConfiguration) (scheduleReasoningView, error) {
	effective, source, err := reasoning.ResolveEffort(configuration.reasoningEffort, "")
	if err != nil {
		return scheduleReasoningView{}, err
	}
	return scheduleReasoningView{
		Configured: optionalString(configuration.reasoningEffort),
		Effective:  optionalString(effective),
		Source:     source,
	}, nil
}

// scheduleBreakerViewFromConfiguration renders the §5.3 triple: no default
// threshold exists (unset means the entry never counts failures) while
// cooldown falls back to the judge's one-hour default.
func scheduleBreakerViewFromConfiguration(
	configured *state.EntryCircuitBreaker,
) scheduleBreakerView {
	defaultCooldown := scheduleDefaultModelCooldownSeconds
	view := scheduleBreakerView{
		Configured: scheduleBreakerParameterView{},
		Effective: scheduleBreakerParameterView{
			CooldownSeconds: &defaultCooldown,
		},
		Sources: scheduleBreakerSourcesView{
			BlacklistThreshold: scheduleBreakerSourceDefault,
			CooldownSeconds:    scheduleBreakerSourceDefault,
		},
	}
	if configured == nil {
		return view
	}
	if configured.BlacklistThreshold != nil {
		threshold := *configured.BlacklistThreshold
		view.Configured.BlacklistThreshold = &threshold
		view.Effective.BlacklistThreshold = &threshold
		view.Sources.BlacklistThreshold = scheduleBreakerSourceEntry
	}
	if configured.CooldownSeconds != nil {
		seconds := *configured.CooldownSeconds
		view.Configured.CooldownSeconds = &seconds
		view.Effective.CooldownSeconds = &seconds
		view.Sources.CooldownSeconds = scheduleBreakerSourceEntry
	}
	return view
}

type modelRouteScheduleRecoverRequest struct {
	GroupID        uint    `json:"group_id"`
	EntryID        string  `json:"entry_id"`
	FailureVersion *uint64 `json:"failure_version"`
}

type modelRouteScheduleRecoverResponse struct {
	GroupID uint                     `json:"group_id"`
	EntryID string                   `json:"entry_id"`
	Runtime scheduleEntryRuntimeView `json:"runtime"`
}

// RecoverModelRouteScheduleEntry clears the in-memory failure runtime (count,
// blacklist, cooldown) of one (group_id, entry_id) route entry (§5.6).
// Credential runtime state is not touched.
func (s *Service) RecoverModelRouteScheduleEntry(
	request modelRouteScheduleRecoverRequest,
) (modelRouteScheduleRecoverResponse, error) {
	entryID := strings.TrimSpace(request.EntryID)
	if request.GroupID == 0 || entryID == "" {
		return modelRouteScheduleRecoverResponse{}, app_errors.ErrValidation
	}
	if request.FailureVersion == nil {
		return modelRouteScheduleRecoverResponse{}, modelRouteScheduleRuntimeConflict
	}
	if !s.registry.RecoverEntryIfVersionMatch(
		request.GroupID, entryID, *request.FailureVersion,
	) {
		return modelRouteScheduleRecoverResponse{}, modelRouteScheduleRuntimeConflict
	}
	now := s.now().UTC()
	view, _ := s.registry.EntryRuntime(state.RouteEntryKey{
		GroupID: request.GroupID, EntryID: entryID,
	}, now)
	cooldownUntilMS, err := optionalSafeEpochMilliseconds(view.CooldownUntil)
	if err != nil {
		return modelRouteScheduleRecoverResponse{}, fmt.Errorf(
			"map model route schedule recovery cooldown: %w", err,
		)
	}
	blacklistReleaseAtMS, err := optionalSafeEpochMilliseconds(view.BlacklistReleaseAt)
	if err != nil {
		return modelRouteScheduleRecoverResponse{}, fmt.Errorf(
			"map model route schedule recovery blacklist_release_at_ms: %w", err,
		)
	}
	return modelRouteScheduleRecoverResponse{
		GroupID: request.GroupID,
		EntryID: entryID,
		Runtime: scheduleEntryRuntimeView{
			State:                view.RuntimeState(now),
			CooldownUntilMS:      cooldownUntilMS,
			BlacklistReleaseAtMS: blacklistReleaseAtMS,
			FailureCount:         view.FailureCount,
			FailureVersion:       view.FailureVersion,
		},
	}, nil
}
