package control

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"strings"

	"gorm.io/gorm"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/protocol"
	"gpt-load/internal/reasoning"
	"gpt-load/internal/state"
	"gpt-load/internal/storage/models"
)

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
	GroupID         uint                      `json:"group_id"`
	EntryID         string                    `json:"entry_id"`
	Weight          optionalField[int]        `json:"weight"`
	Priority        optionalField[int]        `json:"priority"`
	Enabled         optionalField[bool]       `json:"enabled"`
	CircuitBreaker  scheduleBreakerPatchField `json:"circuit_breaker"`
	ReasoningEffort optionalField[string]     `json:"reasoning_effort"`
	// ConcurrencyLimit absent keeps the stored cap; null is rejected.
	ConcurrencyLimit optionalField[int] `json:"concurrency_limit"`
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
	slices.Sort(groupIDs)

	if _, err := s.writeGroupConfig(ctx, func(tx *gorm.DB) error {
		current := s.manager.Current()
		if current == nil || current.Revision != *request.SnapshotRevision {
			return modelRouteScheduleRevisionConflict
		}
		for _, groupID := range groupIDs {
			if err := s.applyModelRouteScheduleGroupPatch(tx, groupID, updatesByGroup[groupID]); err != nil {
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
	if request.Protocol == "" || request.ExternalModel == "" {
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
// silently drop the response detail. access_key_id stays optional: the echo
// renders the same unscoped detail the dispatch center displays.
func validateScheduleDetailEchoContext(request modelRouteSchedulePatchRequest) error {
	partial := request.Protocol != "" || request.ExternalModel != "" || request.AccessKeyID != 0
	if !partial {
		return nil
	}
	if request.Protocol == "" || request.ExternalModel == "" {
		return app_errors.ErrValidation
	}
	return nil
}

func (s *Service) applyModelRouteScheduleGroupPatch(
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
		if err := applyModelRouteScheduleUpdateFields(&entries[index], update); err != nil {
			return err
		}
	}
	providerKind, ok := s.channelRegistry.ProviderKind(channel.ID(group.ChannelID))
	if !ok {
		return app_errors.ErrValidation
	}
	configurations := make([]state.ModelConfig, 0, len(entries))
	for _, entry := range entries {
		if (providerKind == channel.ProviderGemini || providerKind == channel.ProviderGoogleVertex) && entry.ReasoningEffort == "none" {
			return app_errors.NewAPIErrorWithData(app_errors.ErrValidation, scheduleValidationData{
				GroupID: groupID, Message: "Gemini routes cannot configure reasoning effort none",
			})
		}
		_, _, err := reasoning.ResolveEffort(entry.ReasoningEffort, "")
		if err != nil {
			return app_errors.ErrValidation
		}
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
) error {
	if update.ReasoningEffort.Set {
		if update.ReasoningEffort.Null {
			entry.ReasoningEffort = ""
		} else {
			value, err := canonicalScheduleReasoningEffort(update.ReasoningEffort.Value)
			if err != nil {
				return err
			}
			entry.ReasoningEffort = value
		}
	}
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
	if update.ConcurrencyLimit.Set {
		if update.ConcurrencyLimit.Null {
			return app_errors.ErrValidation
		}
		if err := validateConcurrencyLimit(update.ConcurrencyLimit.Value); err != nil {
			return app_errors.ErrValidation
		}
		entry.ConcurrencyLimit = update.ConcurrencyLimit.Value
	}
	if update.Enabled.Set {
		if update.Enabled.Null {
			return app_errors.ErrValidation
		}
		value := update.Enabled.Value
		entry.Enabled = &value
	}
	if !update.CircuitBreaker.Set {
		return nil
	}
	if update.CircuitBreaker.Null {
		entry.CircuitBreaker = nil
		return nil
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
		return nil
	}
	entry.CircuitBreaker = breaker
	return nil
}

func canonicalScheduleReasoningEffort(value string) (string, error) {
	if strings.TrimSpace(value) == "" {
		return "", app_errors.ErrValidation
	}
	canonical, _, err := reasoning.ResolveEffort(value, "")
	if err != nil {
		return "", app_errors.ErrValidation
	}
	return canonical, nil
}

// scheduleDetailAfterPatch re-renders the §5.2 detail from the freshly
// published snapshot when the patch request carried a detail context.
func (s *Service) scheduleDetailAfterPatch(
	request modelRouteSchedulePatchRequest,
) (modelRouteScheduleDetailResponse, error) {
	if request.Protocol == "" || request.ExternalModel == "" {
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
