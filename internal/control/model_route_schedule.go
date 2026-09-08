// Central model-route schedule management API (design
// docs/design/model-route-central-scheduling.md §5): cross-group aggregation
// by external model name, per-context detail with entry-level circuit breaker
// state, transactional tri-state PATCH with optimistic snapshot revision, and
// per-entry runtime recovery.
package control

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"gorm.io/gorm"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/protocol"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
	"gpt-load/internal/storage/models"
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

type modelRouteScheduleIndexItem struct {
	ExternalModel         string `json:"external_model"`
	CandidateCount        int    `json:"candidate_count"`
	GroupCount            int    `json:"group_count"`
	HasFallback           bool   `json:"has_fallback"`
	CooledCandidates      int    `json:"cooled_candidates"`
	BlacklistedCandidates int    `json:"blacklisted_candidates"`
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
	State           state.EntryRuntimeState `json:"state"`
	CooldownUntilMS *int64                  `json:"cooldown_until_ms"`
	FailureCount    int                     `json:"failure_count"`
}

type modelRouteScheduleEntryResponse struct {
	EntryID        string                           `json:"entry_id"`
	ModelID        string                           `json:"model_id"`
	Alias          string                           `json:"alias"`
	WeightManual   *int                             `json:"weight_manual"`
	Weight         int                              `json:"weight"`
	Priority       int                              `json:"priority"`
	Fallback       bool                             `json:"fallback"`
	CircuitBreaker scheduleBreakerView              `json:"circuit_breaker"`
	Runtime        scheduleEntryRuntimeView         `json:"runtime"`
	Included       bool                             `json:"included"`
	Routable       bool                             `json:"routable"`
	ReasonCode     *scheduler.ReasonCode            `json:"reason_code"`
	EffectiveShare float64                          `json:"effective_share"`
	Credentials    []routeInspectCredentialResponse `json:"credentials"`
}

type modelRouteScheduleGroupResponse struct {
	GroupID     uint                              `json:"group_id"`
	GroupName   string                            `json:"group_name"`
	ChannelID   channel.ID                        `json:"channel_id"`
	GroupWeight *int                              `json:"group_weight"`
	Entries     []modelRouteScheduleEntryResponse `json:"entries"`
}

type modelRouteScheduleDetailResponse struct {
	ObservedAtMS     int64                             `json:"observed_at_ms"`
	SnapshotRevision uint64                            `json:"snapshot_revision"`
	RouteStrategy    state.RouteStrategy               `json:"route_strategy"`
	ExternalModel    *string                           `json:"external_model"`
	Protocol         protocol.Protocol                 `json:"protocol"`
	Operation        execution.Operation               `json:"operation"`
	RouteRequirement execution.RouteRequirement        `json:"route_requirement"`
	AccessKey        routeInspectAccessKeyResponse     `json:"access_key"`
	Routable         bool                              `json:"routable"`
	ReasonCode       *scheduler.ReasonCode             `json:"reason_code"`
	Groups           []modelRouteScheduleGroupResponse `json:"groups"`
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

	type scheduleCandidateKey struct {
		groupID uint
		entryID string
	}
	type scheduleIndexAccumulator struct {
		candidates  map[scheduleCandidateKey]struct{}
		groups      map[uint]struct{}
		hasFallback bool
		cooled      int
		blacklist   int
	}
	items := make(map[string]*scheduleIndexAccumulator)
	for _, byOperation := range observation.snapshot.ExecutionRouteCatalog {
		for _, byModel := range byOperation {
			for external, targets := range byModel {
				if external == state.NoModelRouteKey || len(targets) == 0 {
					continue
				}
				accumulator, exists := items[external]
				if !exists {
					accumulator = &scheduleIndexAccumulator{
						candidates: make(map[scheduleCandidateKey]struct{}),
						groups:     make(map[uint]struct{}),
					}
					items[external] = accumulator
				}
				for _, target := range targets {
					key := scheduleCandidateKey{groupID: target.GroupID, entryID: target.EntryID}
					if _, duplicate := accumulator.candidates[key]; duplicate {
						continue
					}
					accumulator.candidates[key] = struct{}{}
					accumulator.groups[target.GroupID] = struct{}{}
					if target.Priority > 1 {
						accumulator.hasFallback = true
					}
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
	for external, accumulator := range items {
		result.Items = append(result.Items, modelRouteScheduleIndexItem{
			ExternalModel:         external,
			CandidateCount:        len(accumulator.candidates),
			GroupCount:            len(accumulator.groups),
			HasFallback:           accumulator.hasFallback,
			CooledCandidates:      accumulator.cooled,
			BlacklistedCandidates: accumulator.blacklist,
		})
	}
	sort.Slice(result.Items, func(i, j int) bool {
		return result.Items[i].ExternalModel < result.Items[j].ExternalModel
	})
	return result, nil
}

func validateModelRouteScheduleDetailRequest(request modelRouteScheduleDetailRequest) error {
	if !request.Protocol.DataPlaneEnabled() ||
		request.AccessKeyID == 0 ||
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
// required protocol/external_model/access_key_id plus the optional operation
// that defaults to the protocol's standard operation.
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
// external model under a protocol/operation/access-key context (§5.2), each
// row carrying the entry identity, the §5.3 breaker triple and the entry
// runtime state.
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
	accessKey, exists := observation.snapshot.AccessKeysByID[request.AccessKeyID]
	if !exists {
		return modelRouteScheduleDetailResponse{}, app_errors.ErrResourceNotFound
	}
	operation := metadata.Operation
	if request.Operation != "" {
		operation = request.Operation
	}
	entryRuntime := s.registry.EntryRuntimeSnapshot()
	explanation := scheduler.Inspection{
		ClientProtocol:   request.Protocol,
		Operation:        operation,
		RouteRequirement: metadata.RouteRequirement,
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
				RouteRequirement: metadata.RouteRequirement,
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
		observation, request, accessKey, explanation, entryRuntime,
	)
}

func mapModelRouteScheduleDetail(
	observation runtimeObservation,
	request modelRouteScheduleDetailRequest,
	accessKey state.AccessKeyView,
	explanation scheduler.Inspection,
	entryRuntime []state.EntryRuntimeView,
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
		RouteStrategy:    observation.snapshot.Settings.RouteStrategy,
		ExternalModel:    cloneRouteModel(explanation.ExternalModel),
		Protocol:         request.Protocol,
		Operation:        explanation.Operation,
		RouteRequirement: explanation.RouteRequirement,
		AccessKey: routeInspectAccessKeyResponse{
			ID: accessKey.ID, Name: accessKey.Name, Status: accessKey.Status,
		},
		Routable:   explanation.Routable,
		ReasonCode: optionalReason(explanation.Reason),
		Groups:     []modelRouteScheduleGroupResponse{},
	}
	groupIndex := make(map[uint]int, len(explanation.Groups))
	for _, group := range explanation.Groups {
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
		entry := modelRouteScheduleEntryResponse{
			EntryID:      group.EntryID,
			ModelID:      upstreamModel,
			Alias:        configuration.alias,
			WeightManual: cloneInt(configuration.weightManual),
			Weight:       group.EntryWeight,
			Priority:     group.Priority,
			Fallback:     group.Priority > 1,
			CircuitBreaker: scheduleBreakerViewFromConfiguration(
				configuration.circuitBreaker,
			),
			Runtime: scheduleEntryRuntime(
				runtimeByKey, group, entryCooldownUntilMS, observation.observedAt,
			),
			Included:       group.Included,
			Routable:       group.Routable,
			ReasonCode:     optionalReason(group.Reason),
			EffectiveShare: group.EffectiveShare,
			Credentials:    credentials,
		}
		position, exists := groupIndex[group.GroupID]
		if !exists {
			position = len(result.Groups)
			groupIndex[group.GroupID] = position
			result.Groups = append(result.Groups, modelRouteScheduleGroupResponse{
				GroupID:     group.GroupID,
				GroupName:   group.GroupName,
				ChannelID:   group.ChannelID,
				GroupWeight: cloneInt(group.WeightManual),
				Entries:     []modelRouteScheduleEntryResponse{},
			})
		}
		result.Groups[position].Entries = append(result.Groups[position].Entries, entry)
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
			WeightManual:    cloneInt(credential.WeightManual),
			WeightAuto:      credential.WeightAuto,
			EffectiveWeight: credential.EffectiveWeight,
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
) scheduleEntryRuntimeView {
	result := scheduleEntryRuntimeView{
		State:           state.EntryRuntimeAvailable,
		CooldownUntilMS: entryCooldownUntilMS,
	}
	view, exists := runtimeByKey[state.RouteEntryKey{
		GroupID: group.GroupID, EntryID: group.EntryID,
	}]
	if exists {
		result.FailureCount = view.FailureCount
		result.State = view.RuntimeState(observedAt)
	}
	return result
}

type scheduleEntryConfiguration struct {
	alias          string
	upstreamModel  string
	weightManual   *int
	circuitBreaker *state.EntryCircuitBreaker
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
				alias:          model.Alias,
				upstreamModel:  upstream,
				weightManual:   cloneInt(model.Weight),
				circuitBreaker: cloneEntryCircuitBreaker(model.CircuitBreaker),
			}
		}
	}
	return result
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

// scheduleBreakerPatchField decodes the tri-state circuit_breaker patch field
// of §5.5: absent = unchanged, null = clear the whole override, object =
// per-parameter tri-state merge ({} leaves both parameters unchanged).
type scheduleBreakerPatchField struct {
	Set                bool
	Null               bool
	BlacklistThreshold *optionalField[int]
	CooldownSeconds    *optionalField[int]
}

func (field *scheduleBreakerPatchField) UnmarshalJSON(data []byte) error {
	if field == nil {
		return fmt.Errorf("circuit_breaker patch receiver is nil")
	}
	field.Set = true
	field.Null = false
	field.BlacklistThreshold = nil
	field.CooldownSeconds = nil
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		field.Null = true
		return nil
	}
	var wire struct {
		BlacklistThreshold json.RawMessage `json:"blacklist_threshold"`
		CooldownSeconds    json.RawMessage `json:"cooldown_seconds"`
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&wire); err != nil {
		return err
	}
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return fmt.Errorf("circuit_breaker contains multiple JSON values")
		}
		return err
	}
	// json.RawMessage (not *optionalField) keeps explicit nulls decodable:
	// encoding/json nils out a pointer field without invoking its
	// UnmarshalJSON, making null indistinguishable from an absent key.
	blacklistThreshold, err := decodeScheduleBreakerParameter(wire.BlacklistThreshold)
	if err != nil {
		return err
	}
	cooldownSeconds, err := decodeScheduleBreakerParameter(wire.CooldownSeconds)
	if err != nil {
		return err
	}
	field.BlacklistThreshold = blacklistThreshold
	field.CooldownSeconds = cooldownSeconds
	return nil
}

func decodeScheduleBreakerParameter(
	raw json.RawMessage,
) (*optionalField[int], error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var value optionalField[int]
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, err
	}
	return &value, nil
}

type modelRouteSchedulePatchUpdate struct {
	GroupID        uint                      `json:"group_id"`
	EntryID        string                    `json:"entry_id"`
	Weight         optionalField[int]        `json:"weight"`
	Priority       optionalField[int]        `json:"priority"`
	CircuitBreaker scheduleBreakerPatchField `json:"circuit_breaker"`
}

type modelRouteSchedulePatchRequest struct {
	SnapshotRevision *uint64                         `json:"snapshot_revision"`
	Protocol         protocol.Protocol               `json:"protocol"`
	ExternalModel    string                          `json:"external_model"`
	Operation        string                          `json:"operation"`
	AccessKeyID      uint                            `json:"access_key_id"`
	Updates          []modelRouteSchedulePatchUpdate `json:"updates"`
}

type modelRouteSchedulePatchResponse struct {
	SnapshotRevisionNew uint64                            `json:"snapshot_revision_new"`
	Detail              *modelRouteScheduleDetailResponse `json:"detail"`
}

type scheduleEntryNotFoundData struct {
	GroupID uint   `json:"group_id"`
	EntryID string `json:"entry_id"`
}

type scheduleValidationData struct {
	GroupID uint   `json:"group_id"`
	Message string `json:"message"`
}

// UpdateModelRouteSchedule applies the field-level tri-state patch of §5.5:
// every update is merged and validated (V1–V7) before a single database
// transaction updates all affected groups and exactly one snapshot is
// published; any failure rejects the whole request with zero writes.
func (s *Service) UpdateModelRouteSchedule(
	ctx context.Context,
	request modelRouteSchedulePatchRequest,
) (modelRouteSchedulePatchResponse, error) {
	if request.SnapshotRevision == nil || len(request.Updates) == 0 {
		return modelRouteSchedulePatchResponse{}, app_errors.ErrValidation
	}
	if err := validateScheduleDetailEchoContext(request); err != nil {
		return modelRouteSchedulePatchResponse{}, err
	}
	updatesByGroup := make(map[uint][]modelRouteSchedulePatchUpdate)
	for _, update := range request.Updates {
		if update.GroupID == 0 || strings.TrimSpace(update.EntryID) == "" {
			return modelRouteSchedulePatchResponse{}, app_errors.ErrValidation
		}
		updatesByGroup[update.GroupID] = append(updatesByGroup[update.GroupID], update)
	}
	groupIDs := make([]uint, 0, len(updatesByGroup))
	for groupID := range updatesByGroup {
		groupIDs = append(groupIDs, groupID)
	}
	sort.Slice(groupIDs, func(i, j int) bool { return groupIDs[i] < groupIDs[j] })

	if _, err := s.writeGroupConfig(ctx, func(tx *gorm.DB) error {
		current := s.manager.Current()
		if current == nil || current.Revision != *request.SnapshotRevision {
			return modelRouteScheduleRevisionConflict
		}
		for _, groupID := range groupIDs {
			if err := applyModelRouteScheduleGroupPatch(tx, groupID, updatesByGroup[groupID]); err != nil {
				return err
			}
		}
		return nil
	}, nil); err != nil {
		return modelRouteSchedulePatchResponse{}, err
	}

	revision := uint64(0)
	if current := s.manager.Current(); current != nil {
		revision = current.Revision
	}
	if request.Protocol == "" || request.ExternalModel == "" || request.AccessKeyID == 0 {
		return modelRouteSchedulePatchResponse{SnapshotRevisionNew: revision}, nil
	}
	detail, err := s.scheduleDetailAfterPatch(request)
	if err != nil {
		// The patch itself is committed and published; a detail echo that
		// cannot be rendered (e.g. the context access key vanished) must not
		// turn the successful mutation into a client-visible failure.
		return modelRouteSchedulePatchResponse{SnapshotRevisionNew: revision}, nil
	}
	return modelRouteSchedulePatchResponse{
		SnapshotRevisionNew: revision,
		Detail:              &detail,
	}, nil
}

// validateScheduleDetailEchoContext requires a complete detail echo context
// when the patch request carries any part of one; an incomplete context would
// silently drop the response detail.
func validateScheduleDetailEchoContext(request modelRouteSchedulePatchRequest) error {
	partial := request.Protocol != "" || request.ExternalModel != "" || request.AccessKeyID != 0
	if !partial {
		return nil
	}
	if request.Protocol == "" || request.ExternalModel == "" || request.AccessKeyID == 0 {
		return app_errors.ErrValidation
	}
	return nil
}

func applyModelRouteScheduleGroupPatch(
	tx *gorm.DB,
	groupID uint,
	updates []modelRouteSchedulePatchUpdate,
) error {
	group, err := loadGroupRow(tx, groupID)
	if err != nil {
		return err
	}
	var entries []groupModelEntry
	if err := decodeGroupDiscoveryJSON(group.Models, &entries); err != nil {
		return fmt.Errorf("decode group %d models: %w", groupID, app_errors.ErrInternalServer)
	}
	entryIndex := make(map[string]int, len(entries))
	for index := range entries {
		if entries[index].EntryID != "" {
			entryIndex[entries[index].EntryID] = index
		}
	}
	for _, update := range updates {
		entryID := strings.TrimSpace(update.EntryID)
		index, exists := entryIndex[entryID]
		if !exists {
			return app_errors.NewAPIErrorWithData(
				app_errors.ErrValidation,
				scheduleEntryNotFoundData{GroupID: groupID, EntryID: entryID},
			)
		}
		applyModelRouteScheduleUpdateFields(&entries[index], update)
	}
	configurations := make([]state.ModelConfig, 0, len(entries))
	for _, entry := range entries {
		configurations = append(configurations, entry.toModelConfig())
	}
	// Validate every merged entry (V1–V7) before any row is written so a
	// rejected request cannot leave a half-applied batch behind.
	if err := state.ValidateModelRouteEntries(
		fmt.Sprintf("group %d", groupID), configurations,
	); err != nil {
		return app_errors.NewAPIErrorWithData(app_errors.ErrValidation, scheduleValidationData{
			GroupID: groupID, Message: err.Error(),
		})
	}
	encoded, err := json.Marshal(entries)
	if err != nil {
		return fmt.Errorf("encode group %d models: %w", groupID, err)
	}
	if err := tx.Model(&models.Group{}).
		Where("id = ?", groupID).
		Update("models", models.JSON(encoded)).Error; err != nil {
		return app_errors.ParseDBError(err)
	}
	return nil
}

func applyModelRouteScheduleUpdateFields(
	entry *groupModelEntry,
	update modelRouteSchedulePatchUpdate,
) {
	if update.Weight.Set {
		if update.Weight.Null {
			entry.Weight = nil
		} else {
			value := update.Weight.Value
			entry.Weight = &value
		}
	}
	if update.Priority.Set {
		if update.Priority.Null {
			entry.Priority = nil
		} else {
			value := update.Priority.Value
			entry.Priority = &value
		}
	}
	if !update.CircuitBreaker.Set {
		return
	}
	if update.CircuitBreaker.Null {
		entry.CircuitBreaker = nil
		return
	}
	breaker := cloneEntryCircuitBreaker(entry.CircuitBreaker)
	if breaker == nil {
		breaker = &state.EntryCircuitBreaker{}
	}
	if threshold := update.CircuitBreaker.BlacklistThreshold; threshold != nil {
		if threshold.Null {
			breaker.BlacklistThreshold = nil
		} else {
			value := threshold.Value
			breaker.BlacklistThreshold = &value
		}
	}
	if seconds := update.CircuitBreaker.CooldownSeconds; seconds != nil {
		if seconds.Null {
			breaker.CooldownSeconds = nil
		} else {
			value := seconds.Value
			breaker.CooldownSeconds = &value
		}
	}
	if breaker.BlacklistThreshold == nil && breaker.CooldownSeconds == nil {
		entry.CircuitBreaker = nil
		return
	}
	entry.CircuitBreaker = breaker
}

// scheduleDetailAfterPatch re-renders the §5.2 detail from the freshly
// published snapshot when the patch request carried a detail context.
func (s *Service) scheduleDetailAfterPatch(
	request modelRouteSchedulePatchRequest,
) (modelRouteScheduleDetailResponse, error) {
	if request.Protocol == "" || request.ExternalModel == "" || request.AccessKeyID == 0 {
		return modelRouteScheduleDetailResponse{}, nil
	}
	detailRequest := modelRouteScheduleDetailRequest{
		Protocol:      request.Protocol,
		ExternalModel: request.ExternalModel,
		AccessKeyID:   request.AccessKeyID,
	}
	if request.Operation != "" {
		detailRequest.Operation = execution.Operation(request.Operation)
	}
	return s.GetModelRouteScheduleDetail(detailRequest)
}

type modelRouteScheduleRecoverRequest struct {
	GroupID uint   `json:"group_id"`
	EntryID string `json:"entry_id"`
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
	if !s.registry.RecoverEntryForEntry(request.GroupID, entryID) {
		return modelRouteScheduleRecoverResponse{}, app_errors.ErrInternalServer
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
	return modelRouteScheduleRecoverResponse{
		GroupID: request.GroupID,
		EntryID: entryID,
		Runtime: scheduleEntryRuntimeView{
			State:           view.RuntimeState(now),
			CooldownUntilMS: cooldownUntilMS,
			FailureCount:    view.FailureCount,
		},
	}, nil
}
