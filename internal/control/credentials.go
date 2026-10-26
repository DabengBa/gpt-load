package control

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"time"

	"gorm.io/gorm"

	"gpt-load/internal/channel"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/state"
	stateloader "gpt-load/internal/state/loader"
	"gpt-load/internal/storage/models"
)

type CredentialImportRequest struct {
	Credentials string `json:"credential"`
}

type CredentialImportResult struct {
	GroupID      uint `json:"group_id"`
	CredentialID uint `json:"credential_id"`
}

type CredentialUpdateRequest struct {
	Credentials optionalField[string] `json:"credential"`
}

type GroupCredentialResponse struct {
	Credential *CredentialItemResponse `json:"credential"`
}

func (s *Service) currentGroupCredentialID(ctx context.Context, groupID uint) (uint, error) {
	capture, err := s.captureCredentials(ctx, groupID)
	if err != nil {
		return 0, err
	}
	if len(capture.rows) > 1 {
		return 0, app_errors.ErrSingleCredentialRequired
	}
	if _, err := validateCredentialCapture(capture); err != nil {
		return 0, err
	}
	if len(capture.rows) == 0 {
		return 0, nil
	}
	return capture.rows[0].ID, nil
}

func (s *Service) GetGroupCredential(ctx context.Context, groupID uint) (GroupCredentialResponse, error) {
	if groupID == 0 {
		return GroupCredentialResponse{}, app_errors.ErrBadRequest
	}
	id, err := s.currentGroupCredentialID(ctx, groupID)
	if err != nil || id == 0 {
		return GroupCredentialResponse{}, err
	}
	detail, err := s.loadCredentialItem(ctx, groupID, id)
	if err != nil {
		return GroupCredentialResponse{}, err
	}
	items := []CredentialItemResponse{detail}
	// enrichCredentialActivities fills the slice elements in place; the summary
	// lookup never fails the read, so a failed lookup leaves it empty.
	s.enrichCredentialActivities(ctx, items)
	detail = items[0]
	return GroupCredentialResponse{Credential: &detail}, nil
}

type CredentialRevealResult struct {
	CredentialID uint            `json:"credential_id"`
	Credential   json.RawMessage `json:"credential"`
	RevealedAtMS int64           `json:"revealed_at_ms"`
}

type CredentialItemResponse struct {
	CredentialID            uint                       `json:"credential_id"`
	SecretVersion           uint64                     `json:"secret_version"`
	Mask                    string                     `json:"mask"`
	EffectiveStatus         string                     `json:"effective_status"`
	RecentSuccessCount      uint64                     `json:"recent_success_count"`
	RecentFailureCount      uint64                     `json:"recent_failure_count"`
	ConsecutiveFailureCount uint64                     `json:"consecutive_failure_count"`
	LastFailureCategory     string                     `json:"last_failure_category"`
	LastFailureSummary      string                     `json:"last_failure_summary,omitempty"`
	LastStatusCode          *int                       `json:"last_status_code"`
	CooldownUntilMS         *int64                     `json:"cooldown_until_ms"`
	Recovery                CredentialRecoveryResponse `json:"recovery"`
}

type CredentialRecoveryResponse struct {
	Mode      string `json:"mode"`
	Automatic bool   `json:"automatic"`
	AtMS      *int64 `json:"at_ms"`
}

type credentialCapture struct {
	group      models.Group
	rows       []models.Credential
	views      []state.CredentialRuntimeView
	observedAt time.Time
}

type credentialObservation struct {
	group      models.Group
	rows       []models.Credential
	runtime    map[uint]state.CredentialRuntimeView
	observedAt time.Time
}

func (s *Service) ImportGroupCredentials(
	ctx context.Context,
	groupID uint,
	request CredentialImportRequest,
) (CredentialImportResult, error) {
	if groupID == 0 {
		return CredentialImportResult{}, app_errors.ErrValidation
	}
	result := CredentialImportResult{GroupID: groupID}
	var entries []state.CredentialEntry
	err := s.writeCredentialConfig(ctx, groupID, 0, func(tx *gorm.DB) error {
		mutation, mutationEntries, err := s.importGroupCredentialsMutation(
			ctx, tx, groupID, request.Credentials,
		)
		if err != nil {
			return err
		}
		result = mutation
		entries = mutationEntries
		return nil
	}, func() error {
		_, err := s.reconcileRegistryGroup(groupID, entries)
		return err
	})
	if err != nil {
		return CredentialImportResult{}, err
	}
	return result, nil
}

// importGroupCredentialsMutation is the single mutation core shared by the
// plain writeCredentialConfig entry and the idempotent Mutate seam.
func (s *Service) importGroupCredentialsMutation(
	ctx context.Context,
	tx *gorm.DB,
	groupID uint,
	rawCredentials string,
) (CredentialImportResult, []state.CredentialEntry, error) {
	group, err := loadGroupRow(tx, groupID)
	if err != nil {
		return CredentialImportResult{}, nil, err
	}
	if group.ChannelID == "" {
		return CredentialImportResult{}, nil, app_errors.ErrValidation
	}
	var count int64
	if err := tx.Model(&models.Credential{}).Where("group_id = ?", groupID).Count(&count).Error; err != nil {
		return CredentialImportResult{}, nil, app_errors.ParseDBError(err)
	}
	if count != 0 {
		return CredentialImportResult{}, nil, app_errors.ErrSingleCredentialRequired
	}
	normalized, err := s.normalizeCredentials(channel.ID(group.ChannelID), rawCredentials)
	if err != nil {
		return CredentialImportResult{}, nil, err
	}
	if len(normalized.candidates) != 1 || normalized.duplicateLines != 0 {
		return CredentialImportResult{}, nil, app_errors.ErrSingleCredentialRequired
	}
	added, duplicated, err := s.persistCredentials(tx, groupID, normalized)
	if err != nil {
		return CredentialImportResult{}, nil, err
	}
	entries, err := stateloader.BuildGroupCredentialEntries(ctx, tx, groupID)
	if err != nil {
		return CredentialImportResult{}, nil, err
	}
	if err := state.ValidateCredentialEntries(entries); err != nil {
		return CredentialImportResult{}, nil, err
	}
	if added != 1 || duplicated != 0 || len(entries) != 1 {
		return CredentialImportResult{}, nil, app_errors.ErrSingleCredentialRequired
	}
	return CredentialImportResult{GroupID: groupID, CredentialID: entries[0].ID}, entries, nil
}

func (s *Service) captureCredentials(ctx context.Context, groupID uint) (credentialCapture, error) {
	s.writeMu.RLock()
	var group models.Group
	groupErr := s.db.WithContext(ctx).Where("id = ?", groupID).Take(&group).Error
	rows := make([]models.Credential, 0)
	var rowsErr error
	if groupErr == nil {
		rowsErr = s.db.WithContext(ctx).Where("group_id = ?", groupID).Order("id ASC").Find(&rows).Error
	}
	var views []state.CredentialRuntimeView
	var observedAt time.Time
	if groupErr == nil && rowsErr == nil {
		views = s.registry.Snapshot()
		observedAt = s.now().UTC()
	}
	s.writeMu.RUnlock()
	if groupErr != nil {
		if errors.Is(groupErr, gorm.ErrRecordNotFound) {
			return credentialCapture{}, groupNotFoundError()
		}
		return credentialCapture{}, app_errors.ParseDBError(groupErr)
	}
	if group.ChannelID == "" {
		return credentialCapture{}, app_errors.ErrValidation
	}
	if rowsErr != nil {
		return credentialCapture{}, app_errors.ParseDBError(rowsErr)
	}

	return credentialCapture{group: group, rows: rows, views: views, observedAt: observedAt}, nil
}

func credentialIDs(rows []models.Credential) []uint {
	ids := make([]uint, len(rows))
	for index := range rows {
		ids[index] = rows[index].ID
	}
	return ids
}

func validateCredentialCapture(capture credentialCapture) (credentialObservation, error) {
	groupID := capture.group.ID
	byID := make(map[uint]state.CredentialRuntimeView, len(capture.views))
	for _, view := range capture.views {
		byID[view.ID] = view
	}
	persisted := make(map[uint]struct{}, len(capture.rows))
	for _, row := range capture.rows {
		view, exists := byID[row.ID]
		if err := validateCredentialRuntimeRow(capture.group, row, view, exists); err != nil {
			return credentialObservation{}, err
		}
		persisted[row.ID] = struct{}{}
	}
	for _, view := range capture.views {
		if view.GroupID == groupID {
			if _, exists := persisted[view.ID]; !exists {
				return credentialObservation{}, dbRegistryMismatch(mismatchExtraRegistry, groupID, view.ID)
			}
		}
	}
	return credentialObservation{group: capture.group, rows: capture.rows, runtime: byID, observedAt: capture.observedAt}, nil
}

func (s *Service) decodeCredential(group models.Group, row models.Credential) (json.RawMessage, string, error) {
	plaintext, err := s.encryption.Decrypt(row.Data)
	if err != nil {
		return nil, "", fmt.Errorf("decrypt credential %d: %w", row.ID, app_errors.ErrInternalServer)
	}

	validated, err := normalizeStoredCredential(s.channelRegistry, channel.ID(group.ChannelID), plaintext)
	plaintext = ""
	if err != nil {
		return nil, "", fmt.Errorf("validate credential %d: %w", row.ID, app_errors.ErrInternalServer)
	}
	apiKey, _ := validated.Value("api_key")
	return validated.CanonicalJSON(), apiKey, nil
}
