package control

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"

	"gorm.io/gorm"

	"gpt-load/internal/channel"
	"gpt-load/internal/platform/canonicaljson"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/state"
	stateloader "gpt-load/internal/state/loader"
	"gpt-load/internal/storage/models"
)

type CredentialConnectRequest struct {
	StagedCredentialID string `json:"staged_credential_id"`
}

// ConnectGroupCredentialsIdempotent consumes a subscription stage exactly once
// while allowing the same HTTP operation to recover after a lost response.
func (s *Service) ConnectGroupCredentialsIdempotent(
	ctx context.Context,
	idempotencyKey string,
	groupID uint,
	expectedID uint,
	stageID string,
) (CredentialImportResult, error) {
	normalized := strings.TrimSpace(stageID)
	if groupID == 0 || normalized == "" {
		return CredentialImportResult{}, app_errors.ErrValidation
	}
	canonicalBody, err := canonicalIdempotencyBody(struct {
		StagedCredentialID   string `json:"staged_credential_id"`
		ExpectedCredentialID uint   `json:"expected_credential_id"`
	}{
		StagedCredentialID: normalized, ExpectedCredentialID: expectedID,
	})
	if err != nil {
		return CredentialImportResult{}, app_errors.ErrInternalServer
	}
	resourceIdentity := "group:" + strconv.FormatUint(uint64(groupID), 10)
	digest, err := buildIdempotencyDigest(idempotencyDigestInput{
		Version: 1, Method: "POST", OperationKind: operationKindCredentialImport,
		PathTemplate:    "/api/groups/:group_id/credential/connect",
		ResourceLocator: resourceIdentity, AuthScopeID: idempotencyAuthScopeID,
		CanonicalBody: canonicalBody,
	})
	if err != nil {
		return CredentialImportResult{}, app_errors.ErrInternalServer
	}
	operationResult, err := s.executeIdempotentOperation(ctx, idempotentOperationInput{
		IdempotencyKey: idempotencyKey, DigestVersion: 1, RequestDigest: digest.Digest,
		Kind: operationKindCredentialImport,
		Mutate: func(tx *gorm.DB) (idempotentMutationResult, error) {
			result, entries, err := s.connectGroupCredentialsMutation(ctx, tx, groupID, expectedID, normalized)
			if err != nil {
				return idempotentMutationResult{}, err
			}
			if err := state.ValidateCredentialEntries(entries); err != nil {
				return idempotentMutationResult{}, err
			}
			canonicalResult, err := canonicaljson.Marshal(result)
			if err != nil {
				return idempotentMutationResult{}, app_errors.ErrInternalServer
			}
			return idempotentMutationResult{
				ResourceIdentity: resourceIdentity,
				CanonicalResult:  canonicalResult,
			}, nil
		},
	})
	if err != nil {
		return CredentialImportResult{}, err
	}
	var result CredentialImportResult
	if err := json.Unmarshal(operationResult.CanonicalResult, &result); err != nil {
		return CredentialImportResult{}, app_errors.ErrInternalServer
	}
	return result, nil
}

// ConnectGroupCredentials promotes a ready subscription stage into an existing
// subscription Group without changing the API-key text import contract.
func (s *Service) ConnectGroupCredentials(
	ctx context.Context,
	groupID uint,
	expectedID uint,
	stageID string,
) (CredentialImportResult, error) {
	normalized := strings.TrimSpace(stageID)
	if groupID == 0 || normalized == "" {
		return CredentialImportResult{}, app_errors.ErrValidation
	}
	result := CredentialImportResult{GroupID: groupID}
	var entries []state.CredentialEntry
	var err error
	err = s.writeCredentialConfig(ctx, groupID, expectedID, func(tx *gorm.DB) error {
		result, entries, err = s.connectGroupCredentialsMutation(ctx, tx, groupID, expectedID, normalized)
		if err != nil {
			return err
		}
		return state.ValidateCredentialEntries(entries)
	}, func() error {
		_, reconcileErr := s.reconcileRegistryGroup(groupID, entries)
		return reconcileErr
	})
	if err != nil {
		return CredentialImportResult{}, err
	}
	return result, nil
}

func (s *Service) connectGroupCredentialsMutation(
	ctx context.Context,
	tx *gorm.DB,
	groupID uint,
	expectedID uint,
	stageID string,
) (CredentialImportResult, []state.CredentialEntry, error) {
	group, err := loadGroupRow(tx, groupID)
	if err != nil {
		return CredentialImportResult{}, nil, err
	}
	if normalizeGroupConnectionType(group.ConnectionType) != models.ConnectionTypeSubscription {
		return CredentialImportResult{}, nil, app_errors.ErrValidation
	}
	var current []models.Credential
	if err := tx.Where("group_id = ?", groupID).Find(&current).Error; err != nil {
		return CredentialImportResult{}, nil, app_errors.ParseDBError(err)
	}
	if len(current) > 1 || (len(current) == 0 && expectedID != 0) || (len(current) == 1 && current[0].ID != expectedID) {
		return CredentialImportResult{}, nil, app_errors.ErrCredentialVersionConflict
	}
	if err := s.validateCredentialConnection(
		tx, group.ID, channel.ID(group.ChannelID), group.ConnectionType, stageID,
	); err != nil {
		return CredentialImportResult{}, nil, err
	}
	err = s.consumeCredentialStage(
		tx, group.ID, channel.ID(group.ChannelID), group.ConnectionType, stageID,
	)
	if err != nil {
		return CredentialImportResult{}, nil, err
	}
	entries, err := stateloader.BuildGroupCredentialEntries(ctx, tx, groupID)
	if err != nil {
		return CredentialImportResult{}, nil, err
	}
	if len(entries) != 1 {
		return CredentialImportResult{}, nil, app_errors.ErrDuplicateCredentialIdentity
	}
	return CredentialImportResult{GroupID: groupID, CredentialID: entries[0].ID}, entries, nil
}

func (s *Service) validateCredentialConnection(
	tx *gorm.DB,
	groupID uint,
	channelID channel.ID,
	connectionType models.ConnectionType,
	stageID string,
) error {
	stage, err := s.loadConsumableCredentialStage(tx, channelID, connectionType, stageID, true)
	if err != nil {
		return err
	}

	var existingRows []models.Credential
	if err := tx.Where("group_id = ?", groupID).Order("id ASC").Find(&existingRows).Error; err != nil {
		return app_errors.ParseDBError(err)
	}
	if len(existingRows) == 0 {
		return nil
	}
	if len(existingRows) != 1 {
		return app_errors.ErrDuplicateCredentialIdentity
	}
	existing := existingRows[0]
	if existing.IdentityFingerprint != stage.IdentityFingerprint {
		return app_errors.ErrDuplicateCredentialIdentity
	}
	switch existing.AuthState {
	case models.CredentialAuthStateReauthorizationRequired,
		models.CredentialAuthStateOutcomeUnknown:
		return nil
	default:
		return app_errors.ErrDuplicateCredentialIdentity
	}
}
