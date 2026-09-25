package control

import (
	"context"
	"encoding/json"
	"strconv"

	"github.com/sirupsen/logrus"
	"gorm.io/gorm"

	"gpt-load/internal/catalog"
	"gpt-load/internal/channel"
	"gpt-load/internal/outboundproxy"
	"gpt-load/internal/platform/canonicaljson"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/utils"
	"gpt-load/internal/state"
	stateloader "gpt-load/internal/state/loader"
	"gpt-load/internal/storage/models"
)

type groupCreateDigestBody struct {
	PriceMultiplier     string                `json:"price_multiplier,omitempty"`
	Name                *string               `json:"name"`
	ChannelID           channel.ID            `json:"channel_id"`
	ConnectionType      models.ConnectionType `json:"connection_type"`
	Params              json.RawMessage       `json:"params"`
	ProviderURL         *string               `json:"provider_url,omitempty"`
	Models              []GroupModel          `json:"models"`
	Credentials         []string              `json:"credentials"`
	StagedCredentialIDs []string              `json:"staged_credential_ids,omitempty"`
	ConfirmSameTarget   bool                  `json:"confirm_same_target,omitempty"`
	Proxy               *outboundproxy.Config `json:"proxy,omitempty"`
}

type credentialImportDigestBody struct {
	Credentials []string `json:"credentials"`
}

func digestGroupModels(values []GroupModel) []GroupModel {
	result := append([]GroupModel(nil), values...)
	for index := range result {
		result[index].TestAlias = ""
	}
	return result
}

func (s *Service) CreateGroupIdempotent(
	ctx context.Context,
	idempotencyKey string,
	request GroupCreateRequest,
) (GroupCreateResult, error) {
	normalized, err := s.normalizeGroupCreate(ctx, request)
	if err != nil {
		return GroupCreateResult{}, err
	}
	credentialLines := []string(nil)
	if normalized.connectionType == models.ConnectionTypeAPIKey {
		credentialLines, err = normalizeIdempotencyKeyLines(request.Credentials)
		if err != nil {
			return GroupCreateResult{}, err
		}
	}
	digestBody := groupCreateDigestBody{
		PriceMultiplier:     priceMultiplierDigest(normalized.priceMultiplier),
		Name:                normalized.explicitName,
		ChannelID:           normalized.channelID,
		ConnectionType:      normalized.connectionType,
		Params:              append(json.RawMessage(nil), normalized.params...),
		ProviderURL:         cloneString(normalized.providerURL),
		Models:              digestGroupModels(normalized.models),
		Credentials:         credentialLines,
		StagedCredentialIDs: append([]string(nil), normalized.stagedCredentialIDs...),
		ConfirmSameTarget:   normalized.confirmSameTarget,
		Proxy:               normalized.proxy,
	}
	canonicalBody, err := canonicalIdempotencyBody(digestBody)
	if err != nil {
		return GroupCreateResult{}, app_errors.ErrInternalServer
	}
	digest, err := buildIdempotencyDigest(idempotencyDigestInput{
		Version:         1,
		Method:          "POST",
		OperationKind:   operationKindGroupCreate,
		PathTemplate:    "/api/groups",
		ResourceLocator: "new",
		AuthScopeID:     idempotencyAuthScopeID,
		CanonicalBody:   canonicalBody,
	})
	if err != nil {
		return GroupCreateResult{}, app_errors.ErrInternalServer
	}
	if isLiteralPrivateHost(normalized.hostname) {
		utils.LogPlaneBestEffort(
			logrus.StandardLogger(),
			logrus.WarnLevel,
			utils.LogPlaneControl,
			logrus.Fields{"host": normalized.hostname},
			"Creating channel group with a private or local host",
		)
	}
	var catalogSnapshot *catalog.Snapshot

	operationResult, err := s.executeIdempotentOperation(ctx, idempotentOperationInput{
		IdempotencyKey: idempotencyKey,
		DigestVersion:  1,
		RequestDigest:  digest.Digest,
		Kind:           operationKindGroupCreate,
		PrepareMutation: func() {
			if s.catalogRuntime != nil {
				catalogSnapshot = s.catalogRuntime.Load()
			}
		},
		Mutate: func(tx *gorm.DB) (idempotentMutationResult, error) {
			mutation, err := s.mutateCreateGroup(ctx, tx, normalized)
			if err != nil {
				return idempotentMutationResult{}, err
			}
			if err := reconcileReferencedPrices(tx, catalogSnapshot); err != nil {
				return idempotentMutationResult{}, err
			}
			input, err := stateloader.BuildCompileInput(ctx, tx, s.channelRegistry)
			if err != nil {
				return idempotentMutationResult{}, err
			}
			if _, err := state.Compile(input); err != nil {
				return idempotentMutationResult{}, err
			}
			if _, err := loadPriceTable(ctx, tx); err != nil {
				return idempotentMutationResult{}, err
			}
			canonicalResult, err := canonicaljson.Marshal(mutation.result)
			if err != nil {
				return idempotentMutationResult{}, app_errors.ErrInternalServer
			}
			return idempotentMutationResult{
				ResourceIdentity: "group:" + strconv.FormatUint(uint64(mutation.result.GroupID), 10),
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
	if !operationResult.Replayed && len(normalized.models) > 0 && s.catalogSync != nil {
		s.catalogSync.RequestGroupSync()
	}
	return result, nil
}

func (s *Service) ImportGroupCredentialsIdempotent(
	ctx context.Context,
	idempotencyKey string,
	groupID uint,
	request CredentialImportRequest,
) (CredentialImportResult, error) {
	if groupID == 0 {
		return CredentialImportResult{}, app_errors.ErrValidation
	}
	credentialLines, err := normalizeIdempotencyKeyLines(request.Credentials)
	if err != nil {
		return CredentialImportResult{}, err
	}
	canonicalBody, err := canonicalIdempotencyBody(credentialImportDigestBody{Credentials: credentialLines})
	if err != nil {
		return CredentialImportResult{}, app_errors.ErrInternalServer
	}
	resourceIdentity := "group:" + strconv.FormatUint(uint64(groupID), 10)
	digest, err := buildIdempotencyDigest(idempotencyDigestInput{
		Version:         1,
		Method:          "POST",
		OperationKind:   operationKindCredentialImport,
		PathTemplate:    "/api/groups/:group_id/credentials/import",
		ResourceLocator: resourceIdentity,
		AuthScopeID:     idempotencyAuthScopeID,
		CanonicalBody:   canonicalBody,
	})
	if err != nil {
		return CredentialImportResult{}, app_errors.ErrInternalServer
	}

	operationResult, err := s.executeIdempotentOperation(ctx, idempotentOperationInput{
		IdempotencyKey: idempotencyKey,
		DigestVersion:  1,
		RequestDigest:  digest.Digest,
		Kind:           operationKindCredentialImport,
		Mutate: func(tx *gorm.DB) (idempotentMutationResult, error) {
			result, _, err := s.importGroupCredentialsMutation(
				ctx, tx, groupID, request.Credentials,
			)
			if err != nil {
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
