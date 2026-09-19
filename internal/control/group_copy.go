package control

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"

	"gorm.io/gorm"

	"gpt-load/internal/catalog"
	"gpt-load/internal/platform/canonicaljson"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/state"
	stateloader "gpt-load/internal/state/loader"
	"gpt-load/internal/storage/models"
)

type groupCopyDigestBody struct {
	SourceGroupID uint `json:"source_group_id"`
}

func (s *Service) CopyGroupIdempotent(
	ctx context.Context,
	idempotencyKey string,
	sourceGroupID uint,
) (GroupCreateResult, error) {
	if sourceGroupID == 0 {
		return GroupCreateResult{}, app_errors.ErrValidation
	}
	canonicalBody, err := canonicalIdempotencyBody(groupCopyDigestBody{SourceGroupID: sourceGroupID})
	if err != nil {
		return GroupCreateResult{}, app_errors.ErrInternalServer
	}
	sourceIdentity := "group:" + strconv.FormatUint(uint64(sourceGroupID), 10)
	digest, err := buildIdempotencyDigest(idempotencyDigestInput{
		Version:         1,
		Method:          "POST",
		OperationKind:   operationKindGroupCopy,
		PathTemplate:    "/api/groups/:group_id/copy",
		ResourceLocator: sourceIdentity,
		AuthScopeID:     idempotencyAuthScopeID,
		CanonicalBody:   canonicalBody,
	})
	if err != nil {
		return GroupCreateResult{}, app_errors.ErrInternalServer
	}
	var catalogSnapshot *catalog.Snapshot

	operationResult, err := s.executeIdempotentOperation(ctx, idempotentOperationInput{
		IdempotencyKey: idempotencyKey,
		DigestVersion:  1,
		RequestDigest:  digest.Digest,
		Kind:           operationKindGroupCopy,
		PrepareMutation: func() {
			if s.catalogRuntime != nil {
				catalogSnapshot = s.catalogRuntime.Load()
			}
		},
		Mutate: func(tx *gorm.DB) (idempotentMutationResult, error) {
			var source models.Group
			if err := tx.Take(&source, sourceGroupID).Error; err != nil {
				if errors.Is(err, gorm.ErrRecordNotFound) {
					return idempotentMutationResult{}, groupNotFoundError()
				}
				return idempotentMutationResult{}, app_errors.ParseDBError(err)
			}
			name, err := resolveGroupCopyName(tx, source.Name)
			if err != nil {
				return idempotentMutationResult{}, err
			}
			clone := models.Group{
				PriceMultiplierMicros: cloneOptionalInt64(source.PriceMultiplierMicros),
				Name:                  name,
				ChannelID:             source.ChannelID,
				ConnectionType:        source.ConnectionType,
				Params:                append(models.JSON(nil), source.Params...),
				ProviderURL:           cloneString(source.ProviderURL),
				Models:                append(models.JSON(nil), source.Models...),
				Overrides:             append(models.JSON(nil), source.Overrides...),
				ProxyConfig:           cloneString(source.ProxyConfig),
				Enabled:               source.Enabled,
			}
			if err := tx.Create(&clone).Error; err != nil {
				return idempotentMutationResult{}, app_errors.ParseDBError(err)
			}
			if !source.Enabled {
				// enabled 带 default:true,显式列更新才能保留禁用态。
				if err := tx.Model(&clone).Update("enabled", false).Error; err != nil {
					return idempotentMutationResult{}, app_errors.ParseDBError(err)
				}
				clone.Enabled = false
			}
			var credentials []models.Credential
			if err := tx.Where("group_id = ?", source.ID).Order("id ASC").Find(&credentials).Error; err != nil {
				return idempotentMutationResult{}, app_errors.ParseDBError(err)
			}
			for index := range credentials {
				credential := credentials[index]
				credential.ID = 0
				credential.GroupID = clone.ID
				credential.Group = nil
				credential.CreatedAtMS = 0
				credential.UpdatedAtMS = 0
				if credential.AuthState == models.CredentialAuthStateRefreshing {
					credential.AuthState = models.CredentialAuthStateOutcomeUnknown
					credential.AuthErrorCode = "refresh_interrupted"
				}
				if err := tx.Create(&credential).Error; err != nil {
					return idempotentMutationResult{}, app_errors.ParseDBError(err)
				}
			}
			entries, err := stateloader.BuildGroupCredentialEntries(ctx, tx, clone.ID)
			if err != nil {
				return idempotentMutationResult{}, err
			}
			if err := state.ValidateCredentialEntries(entries); err != nil {
				return idempotentMutationResult{}, err
			}
			if err := reconcileReferencedPrices(tx, catalogSnapshot); err != nil {
				return idempotentMutationResult{}, err
			}
			input, err := stateloader.BuildCompileInputWithProxy(
				ctx, tx, s.encryption, s.environmentProxy, s.channelRegistry,
			)
			if err != nil {
				return idempotentMutationResult{}, err
			}
			if _, err := state.Compile(input); err != nil {
				return idempotentMutationResult{}, err
			}
			if _, err := loadPriceTable(ctx, tx); err != nil {
				return idempotentMutationResult{}, err
			}
			result := GroupCreateResult{
				GroupID:          clone.ID,
				GroupName:        clone.Name,
				CredentialsAdded: len(credentials),
			}
			canonicalResult, err := canonicaljson.Marshal(result)
			if err != nil {
				return idempotentMutationResult{}, app_errors.ErrInternalServer
			}
			return idempotentMutationResult{
				ResourceIdentity: "group:" + strconv.FormatUint(uint64(clone.ID), 10),
				CanonicalResult:  canonicalResult,
			}, nil
		},
	})
	if err != nil {
		return GroupCreateResult{}, err
	}
	var result GroupCreateResult
	if err := json.Unmarshal(operationResult.CanonicalResult, &result); err != nil {
		return GroupCreateResult{}, app_errors.ErrInternalServer
	}
	if !operationResult.Replayed && s.catalogSync != nil {
		s.catalogSync.RequestGroupSync()
	}
	return result, nil
}

func resolveGroupCopyName(tx *gorm.DB, sourceName string) (string, error) {
	base := strings.TrimSpace(truncateGroupNameBytes(sourceName, 240))
	if base == "" {
		base = "group"
	}
	for suffix := 1; ; suffix++ {
		candidate := base + "-copy"
		if suffix > 1 {
			candidate = fmt.Sprintf("%s-copy-%d", base, suffix)
		}
		var count int64
		if err := tx.Model(&models.Group{}).Where("name = ?", candidate).Count(&count).Error; err != nil {
			return "", app_errors.ParseDBError(err)
		}
		if count == 0 {
			return candidate, nil
		}
	}
}

func truncateGroupNameBytes(name string, maxBytes int) string {
	if len(name) <= maxBytes {
		return name
	}
	truncated := name[:maxBytes]
	for !utf8.ValidString(truncated) {
		truncated = truncated[:len(truncated)-1]
	}
	return truncated
}
