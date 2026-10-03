package control

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"gorm.io/gorm"

	"gpt-load/internal/channel"
	"gpt-load/internal/health"
	"gpt-load/internal/platform/epochms"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/state"
	"gpt-load/internal/storage/models"
)

func normalizeCredentialUpdate(request CredentialUpdateRequest) (string, error) {
	if !request.Credentials.Set || request.Credentials.Null || strings.TrimSpace(request.Credentials.Value) == "" {
		return "", app_errors.ErrBadRequest
	}
	return strings.TrimSpace(request.Credentials.Value), nil
}

func nextCredentialUpdatedAtMS(now time.Time, previous int64) (int64, error) {
	nowMS, err := epochms.FromTime(now)
	if err != nil {
		return 0, err
	}
	if nowMS < 1 {
		nowMS = 1
	}
	if nowMS <= previous {
		if previous == math.MaxInt64 {
			return 0, fmt.Errorf("credential version exhausted")
		}
		nowMS = previous + 1
	}
	return nowMS, nil
}

func findRuntimeCredential(
	views []state.CredentialRuntimeView,
	credentialID uint,
) (state.CredentialRuntimeView, bool) {
	for _, view := range views {
		if view.ID == credentialID {
			return view, true
		}
	}
	return state.CredentialRuntimeView{}, false
}

func (s *Service) RevealGroupCredential(
	ctx context.Context,
	groupID uint,
	credentialID uint,
) (CredentialRevealResult, error) {
	if groupID == 0 || credentialID == 0 {
		return CredentialRevealResult{}, app_errors.ErrBadRequest
	}
	s.writeMu.RLock()
	defer s.writeMu.RUnlock()
	group, err := loadGroupRow(s.db.WithContext(ctx), groupID)
	if err != nil {
		return CredentialRevealResult{}, err
	}
	if group.ChannelID == "" {
		return CredentialRevealResult{}, app_errors.ErrValidation
	}
	if normalizeGroupConnectionType(group.ConnectionType) == models.ConnectionTypeSubscription {
		return CredentialRevealResult{}, app_errors.ErrForbidden
	}
	var row models.Credential
	if err := s.db.WithContext(ctx).Select("id", "group_id", "data", "fingerprint", "identity_fingerprint", "secret_version", "auth_state", "updated_at_ms").
		Where("id = ? AND group_id = ?", credentialID, groupID).Take(&row).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return CredentialRevealResult{}, credentialNotFoundError()
		}
		return CredentialRevealResult{}, app_errors.ParseDBError(err)
	}
	credential, _, err := s.decodeCredential(group, row)
	if err != nil {
		return CredentialRevealResult{}, err
	}
	revealedAtMS, err := safeEpochMilliseconds(s.now())
	if err != nil {
		return CredentialRevealResult{}, app_errors.ErrInternalServer
	}
	return CredentialRevealResult{
		CredentialID: row.ID, Credential: append([]byte(nil), credential...), RevealedAtMS: revealedAtMS,
	}, nil
}

func (s *Service) UpdateGroupCredential(
	ctx context.Context,
	groupID uint,
	credentialID uint,
	request CredentialUpdateRequest,
) (CredentialItemResponse, error) {
	if groupID == 0 || credentialID == 0 {
		return CredentialItemResponse{}, app_errors.ErrBadRequest
	}
	credentials, err := normalizeCredentialUpdate(request)
	if err != nil {
		return CredentialItemResponse{}, err
	}
	var committed models.Credential
	var committedGroup models.Group
	var nextFingerprint string
	err = s.writeCredentialConfig(ctx, groupID, credentialID, func(tx *gorm.DB) error {
		group, err := loadGroupRow(tx, groupID)
		if err != nil {
			return err
		}
		if group.ChannelID == "" {
			return app_errors.ErrValidation
		}
		committedGroup = group
		if normalizeGroupConnectionType(group.ConnectionType) != models.ConnectionTypeAPIKey {
			return app_errors.ErrValidation
		}
		if err := tx.Where("id = ? AND group_id = ?", credentialID, groupID).Take(&committed).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return credentialNotFoundError()
			}
			return app_errors.ParseDBError(err)
		}
		view, exists := findRuntimeCredential(s.registry.Snapshot(), credentialID)
		if err := validateCredentialRuntimeRow(group, committed, view, exists); err != nil {
			return err
		}
		normalized, err := s.normalizeCredentials(channel.ID(group.ChannelID), credentials)
		if err != nil || len(normalized.candidates) != 1 || normalized.duplicateLines != 0 {
			return app_errors.ErrValidation
		}
		candidate := normalized.candidates[0]
		ciphertext, err := s.encryption.Encrypt(string(candidate.canonical))
		if err != nil {
			return app_errors.ErrInternalServer
		}
		updatedAtMS, err := nextCredentialUpdatedAtMS(s.now(), committed.UpdatedAtMS)
		if err != nil {
			return app_errors.ErrInternalServer
		}
		committed.SecretVersion++
		if committed.SecretVersion == 0 {
			return app_errors.ErrInternalServer
		}
		committed.Data = ciphertext
		committed.Fingerprint = candidate.fingerprint
		committed.IdentityFingerprint = candidate.fingerprint
		committed.AuthState = models.CredentialAuthStateReady
		committed.AuthErrorCode = ""
		committed.UpdatedAtMS = updatedAtMS
		nextFingerprint = candidate.fingerprint
		updates := map[string]any{
			"data": ciphertext, "fingerprint": candidate.fingerprint,
			"identity_fingerprint": candidate.fingerprint,
			"secret_version":       committed.SecretVersion,
			"auth_state":           committed.AuthState, "auth_error_code": "",
			"updated_at_ms": updatedAtMS,
		}
		if err := tx.Model(&models.Credential{}).Where("id = ? AND group_id = ?", credentialID, groupID).
			Updates(updates).Error; err != nil {
			return app_errors.ParseDBError(err)
		}
		return nil
	}, func() error {
		entries, snapshotErr := s.registry.SnapshotGroupCredentialEntriesExact(groupID, []uint{credentialID})
		if snapshotErr != nil {
			return dbRegistryMismatch(mismatchMissingRegistry, groupID, credentialID)
		}
		entry := entries[0]
		entry.AuthState = state.CredentialAuthStateReady
		entry.Version = groupCollectionCredentialVersion(committed.SecretVersion)
		entry.IdentityGeneration = groupCollectionCredentialIdentity(
			committed.IdentityFingerprint,
			committedGroup,
		)
		entry.Fingerprint = nextFingerprint
		entry.EncryptedValue = committed.Data
		return s.registry.RestoreGroupCredentialEntriesExact(groupID, []state.CredentialEntry{entry})
	})
	if err != nil {
		return CredentialItemResponse{}, err
	}
	return s.loadCredentialItem(ctx, groupID, credentialID)
}

func (s *Service) DeleteGroupCredential(ctx context.Context, groupID, credentialID uint) error {
	if groupID == 0 || credentialID == 0 {
		return app_errors.ErrBadRequest
	}
	return s.writeCredentialConfig(ctx, groupID, credentialID, func(tx *gorm.DB) error {
		group, err := loadGroupRow(tx, groupID)
		if err != nil {
			return err
		}
		if group.ChannelID == "" {
			return app_errors.ErrValidation
		}
		var row models.Credential
		if err := tx.Where("id = ? AND group_id = ?", credentialID, groupID).Take(&row).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return credentialNotFoundError()
			}
			return app_errors.ParseDBError(err)
		}
		view, exists := findRuntimeCredential(s.registry.Snapshot(), credentialID)
		if err := validateCredentialRuntimeRow(group, row, view, exists); err != nil {
			return err
		}
		if err := tx.Delete(&row).Error; err != nil {
			return app_errors.ParseDBError(err)
		}
		return nil
	}, func() error {
		if !s.registry.RemoveCredential(credentialID) {
			return dbRegistryMismatch(mismatchMissingRegistry, groupID, credentialID)
		}
		s.stats.Reset(credentialID)
		s.retireCredentialRuntime(credentialID)
		return nil
	})
}

// RestoreGroupCredential repairs live runtime health state (cooldown,
// blacklist) without committing config. It is exempt from the operation
// recovery barrier by design — it writes no committed state and may be needed
// exactly while recovery is pending — but the registry/stats mutation runs
// inside the credential mutation coordinator so data-plane failure recording
// cannot interleave with the restore.
func (s *Service) RestoreGroupCredential(
	ctx context.Context,
	groupID uint,
	credentialID uint,
) (CredentialItemResponse, error) {
	if groupID == 0 || credentialID == 0 {
		return CredentialItemResponse{}, app_errors.ErrBadRequest
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	group, err := loadGroupRow(s.db.WithContext(ctx), groupID)
	if err != nil {
		return CredentialItemResponse{}, err
	}
	if group.ChannelID == "" {
		return CredentialItemResponse{}, app_errors.ErrValidation
	}
	var row models.Credential
	if err := s.db.WithContext(ctx).
		Where("id = ? AND group_id = ?", credentialID, groupID).
		Take(&row).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return CredentialItemResponse{}, credentialNotFoundError()
		}
		return CredentialItemResponse{}, app_errors.ParseDBError(err)
	}
	observedAt := s.now().UTC()
	var restored state.CredentialRuntimeView
	var applyErr error
	if err := s.doCredentialMutations([]uint{credentialID}, func() {
		view, exists := findRuntimeCredential(s.registry.Snapshot(), credentialID)
		if err := validateCredentialRuntimeRow(group, row, view, exists); err != nil {
			applyErr = err
			return
		}
		bucket := classifyHealthKey(
			state.GroupCatalogView{ID: group.ID, Name: group.Name, Enabled: group.Enabled},
			view,
			observedAt,
		)
		if bucket != healthBucketCooldown && bucket != healthBucketBlacklisted {
			applyErr = app_errors.ErrInvalidCredentialState
			return
		}
		if !s.registry.RestoreRuntimeState(credentialID) {
			applyErr = dbRegistryMismatch(mismatchMissingRegistry, groupID, credentialID)
			return
		}
		if s.stats != nil {
			s.stats.ClearProblemState(credentialID)
		}
		view, exists = findRuntimeCredential(s.registry.Snapshot(), credentialID)
		if !exists {
			applyErr = dbRegistryMismatch(mismatchMissingRegistry, groupID, credentialID)
			return
		}
		restored = view
	}); err != nil {
		return CredentialItemResponse{}, err
	}
	if applyErr != nil {
		return CredentialItemResponse{}, applyErr
	}
	return s.mapCredentialItem(ctx, row, restored, group, s.stats.Snapshot(credentialID, observedAt), observedAt)
}

func validateCredentialRuntimeRow(
	group models.Group,
	row models.Credential,
	view state.CredentialRuntimeView,
	exists bool,
) error {
	groupID := group.ID
	if !exists {
		return dbRegistryMismatch(mismatchMissingRegistry, groupID, row.ID)
	}
	if view.GroupID != groupID {
		return dbRegistryMismatch(mismatchGroupID, groupID, row.ID)
	}
	if view.AuthState != normalizeRuntimeCredentialAuthState(row.AuthState) {
		return dbRegistryMismatch(mismatchStatus, groupID, row.ID)
	}
	if view.Version != groupCollectionCredentialVersion(row.SecretVersion) ||
		view.IdentityGeneration != groupCollectionCredentialIdentity(row.IdentityFingerprint, group) {
		return dbRegistryMismatch(mismatchIdentity, groupID, row.ID)
	}
	return nil
}

func (s *Service) loadCredentialItem(ctx context.Context, groupID, credentialID uint) (CredentialItemResponse, error) {
	capture, err := s.captureCredentials(ctx, groupID)
	if err != nil {
		return CredentialItemResponse{}, err
	}
	observation, err := validateCredentialCapture(capture)
	if err != nil {
		return CredentialItemResponse{}, err
	}
	for _, row := range observation.rows {
		if row.ID == credentialID {
			return s.mapCredentialItem(ctx, row, observation.runtime[credentialID], observation.group,
				s.stats.Snapshot(credentialID, observation.observedAt), observation.observedAt)
		}
	}
	return CredentialItemResponse{}, credentialNotFoundError()
}

func (s *Service) mapCredentialItem(
	ctx context.Context,
	row models.Credential,
	view state.CredentialRuntimeView,
	group models.Group,
	stats health.CredentialStats,
	observedAt time.Time,
) (CredentialItemResponse, error) {
	canonical, identity, err := s.decodeCredential(group, row)
	if err != nil {
		return CredentialItemResponse{}, err
	}
	mask, account, err := s.credentialPresentation(group, row, canonical, identity)
	if err != nil {
		return CredentialItemResponse{}, err
	}
	bucket := classifyHealthKey(state.GroupCatalogView{ID: group.ID, Name: group.Name, Enabled: group.Enabled}, view, observedAt)
	item, err := mapCredentialRuntimeItem(mask, row.ID, view, bucket, stats, observedAt)
	if err != nil {
		return CredentialItemResponse{}, err
	}
	item.ConnectionType = string(normalizeGroupConnectionType(group.ConnectionType))
	item.SecretVersion = row.SecretVersion
	item.AuthState = string(row.AuthState)
	item.Account = account
	if normalizeGroupConnectionType(group.ConnectionType) == models.ConnectionTypeSubscription {
		var observation models.CredentialObservation
		result := s.db.WithContext(ctx).Take(&observation, "credential_id = ?", row.ID)
		if result.Error != nil && !errors.Is(result.Error, gorm.ErrRecordNotFound) {
			return CredentialItemResponse{}, app_errors.ParseDBError(result.Error)
		}
		item.Observation = presentCredentialObservation(observation, row.IdentityFingerprint)
	}
	return item, nil
}
