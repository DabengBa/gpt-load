package control

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
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
		if err != nil || len(normalized.candidates) != 1 {
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

func (s *Service) RestoreGroupCredential(
	ctx context.Context,
	groupID uint,
	credentialID uint,
) (CredentialItemResponse, error) {
	return s.restoreGroupCredential(ctx, groupID, credentialID, "")
}

func (s *Service) restoreGroupCredential(
	ctx context.Context,
	groupID uint,
	credentialID uint,
	restoreProof string,
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
	if restoreProof != "" &&
		normalizeGroupConnectionType(group.ConnectionType) == models.ConnectionTypeSubscription {
		return CredentialItemResponse{}, app_errors.ErrForbidden
	}
	var row models.Credential
	if err := s.db.WithContext(ctx).Where("id = ? AND group_id = ?", credentialID, groupID).Take(&row).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return CredentialItemResponse{}, credentialNotFoundError()
		}
		return CredentialItemResponse{}, app_errors.ParseDBError(err)
	}
	view, exists := findRuntimeCredential(s.registry.Snapshot(), credentialID)
	if err := validateCredentialRuntimeRow(group, row, view, exists); err != nil {
		return CredentialItemResponse{}, err
	}
	groupView := state.GroupCatalogView{ID: group.ID, Name: group.Name, Enabled: group.Enabled}
	var (
		observedAt time.Time
		restoreErr error
	)
	restore := func(targetSignature *groupValidationSignature) {
		observedAt = s.now().UTC()
		var testedCredential *credentialProbeCredential
		current, exists := findRuntimeCredential(s.registry.Snapshot(), credentialID)
		if !exists {
			restoreErr = dbRegistryMismatch(mismatchMissingRegistry, groupID, credentialID)
			return
		}
		if targetSignature == nil {
			bucket := classifyHealthKey(groupView, current, observedAt)
			if bucket != healthBucketCooldown && bucket != healthBucketBlacklisted {
				restoreErr = app_errors.ErrInvalidCredentialState
				return
			}
		} else {
			entries, snapshotErr := s.registry.SnapshotGroupCredentialEntriesExact(
				groupID,
				[]uint{credentialID},
			)
			if snapshotErr != nil || len(entries) != 1 {
				restoreErr = dbRegistryMismatch(mismatchMissingRegistry, groupID, credentialID)
				return
			}
			if !s.credentialProbeRestoreProofMatches(entries[0], *targetSignature, restoreProof) {
				restoreErr = app_errors.ErrCredentialVersionConflict
				return
			}
			credential := credentialProbeCredentialFromEntry(entries[0])
			testedCredential = &credential
		}
		stats := s.stats.Snapshot(credentialID, observedAt)
		stats.ConsecutiveFailure = 0
		stats.ConsecutiveProblem = 0
		stats.LastFailureCategory = 0
		stats.LastStatusCode = 0
		if targetSignature == nil {
			if !s.registry.RestoreRuntimeState(credentialID) {
				restoreErr = dbRegistryMismatch(mismatchMissingRegistry, groupID, credentialID)
				return
			}
		} else {
			if testedCredential == nil || !s.registry.RestoreRuntimeStateIfMatch(
				testedCredential.ref,
				testedCredential.cooldownUntil,
			) {
				restoreErr = app_errors.ErrCredentialVersionConflict
				return
			}
		}
		s.stats.ClearProblemState(credentialID)
	}
	coordinateRestore := func(targetSignature *groupValidationSignature) {
		if s.mutations == nil {
			restore(targetSignature)
		} else {
			s.mutations.Do(credentialID, func() { restore(targetSignature) })
		}
	}
	if restoreProof == "" {
		coordinateRestore(nil)
	} else {
		if s.manager == nil || s.mutations == nil {
			return CredentialItemResponse{}, app_errors.ErrInternalServer
		}
		matched := s.manager.WithCurrentSnapshot(func(snapshot *state.ConfigSnapshot) bool {
			if snapshot == nil {
				return false
			}
			currentGroup, exists := snapshot.Groups[groupID]
			if !exists {
				restoreErr = app_errors.ErrCredentialVersionConflict
				return false
			}
			currentTarget, valid := buildGroupValidationTarget(currentGroup)
			if !valid {
				restoreErr = app_errors.ErrCredentialVersionConflict
				return false
			}
			coordinateRestore(&currentTarget.signature)
			return restoreErr == nil
		})
		if !matched && restoreErr == nil {
			return CredentialItemResponse{}, app_errors.ErrInternalServer
		}
	}
	if restoreErr != nil {
		return CredentialItemResponse{}, restoreErr
	}
	view, exists = findRuntimeCredential(s.registry.Snapshot(), credentialID)
	if !exists {
		return CredentialItemResponse{}, dbRegistryMismatch(mismatchMissingRegistry, groupID, credentialID)
	}
	return s.mapCredentialItem(ctx, row, view, group, s.stats.Snapshot(credentialID, observedAt), observedAt)
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

func normalizeCredentialBatchRequest(request CredentialBatchRequest) ([]uint, error) {
	if request.Action != CredentialBatchDelete ||
		len(request.CredentialIDs) < 1 || len(request.CredentialIDs) > 100 {
		return nil, app_errors.ErrValidation
	}
	ids := append([]uint(nil), request.CredentialIDs...)
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	for index, id := range ids {
		if id == 0 || index > 0 && id == ids[index-1] {
			return nil, app_errors.ErrValidation
		}
	}
	return ids, nil
}

func (s *Service) BatchGroupCredentials(
	ctx context.Context,
	groupID uint,
	request CredentialBatchRequest,
) (CredentialBatchResponse, error) {
	if groupID == 0 {
		return CredentialBatchResponse{}, app_errors.ErrBadRequest
	}
	ids, err := normalizeCredentialBatchRequest(request)
	if err != nil {
		return CredentialBatchResponse{}, err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	group, err := loadGroupRow(s.db.WithContext(ctx), groupID)
	if err != nil {
		return CredentialBatchResponse{}, err
	}
	var rows []models.Credential
	if err := s.db.WithContext(ctx).Where("group_id = ? AND id IN ?", groupID, ids).Find(&rows).Error; err != nil {
		return CredentialBatchResponse{}, app_errors.ParseDBError(err)
	}
	if len(rows) != len(ids) {
		return CredentialBatchResponse{}, credentialNotFoundError()
	}
	for _, row := range rows {
		view, exists := findRuntimeCredential(s.registry.Snapshot(), row.ID)
		if err := validateCredentialRuntimeRow(group, row, view, exists); err != nil {
			return CredentialBatchResponse{}, err
		}
	}
	if err := s.withControlTransaction(ctx, func(tx *gorm.DB) error {
		result := tx.Where("group_id = ? AND id IN ?", groupID, ids).Delete(&models.Credential{})
		if result.Error != nil {
			return app_errors.ParseDBError(result.Error)
		}
		if result.RowsAffected != int64(len(ids)) {
			return fmt.Errorf("batch credential rows affected = %d, want %d: %w", result.RowsAffected, len(ids), app_errors.ErrDatabase)
		}
		return nil
	}); err != nil {
		return CredentialBatchResponse{}, err
	}
	if err := s.registry.RemoveGroupCredentials(groupID, ids); err != nil {
		return CredentialBatchResponse{}, err
	}
	for _, id := range ids {
		s.stats.Reset(id)
		s.retireCredentialRuntime(id)
	}
	return CredentialBatchResponse{
		AffectedCredentialIDs: ids,
		Summary:               summarizeGroupRuntimeCredentials(group, s.registry.Snapshot(), s.now().UTC()),
	}, nil
}

func summarizeGroupRuntimeCredentials(
	group models.Group,
	views []state.CredentialRuntimeView,
	observedAt time.Time,
) CredentialSummaryResponse {
	summary := CredentialSummaryResponse{}
	groupView := state.GroupCatalogView{ID: group.ID, Name: group.Name, Enabled: group.Enabled}
	for _, view := range views {
		if view.GroupID != group.ID {
			continue
		}
		summary.Total++
		switch classifyHealthKey(groupView, view, observedAt) {
		case healthBucketAvailable:
			summary.Available++
		case healthBucketCooldown:
			summary.Cooldown++
		case healthBucketBlacklisted:
			summary.Blacklisted++
		case healthBucketDisabled:
			summary.Disabled++
		}
	}
	return summary
}
