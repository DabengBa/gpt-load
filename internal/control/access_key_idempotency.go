package control

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"gorm.io/gorm"

	"gpt-load/internal/platform/canonicaljson"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	stateloader "gpt-load/internal/state/loader"
)

type accessKeyFilterDigestBody struct {
	Groups       []uint              `json:"groups"`
	Protocols    []protocol.Protocol `json:"protocols"`
	Models       []string            `json:"models"`
	AllowedCIDRs []string            `json:"allowed_cidrs,omitempty"`
}

type accessKeyCreateDigestBody struct {
	PriceMultiplier string                          `json:"price_multiplier,omitempty"`
	Name            string                          `json:"name"`
	Status          *state.AccessKeyStatus          `json:"status,omitempty"`
	Filters         accessKeyFilterDigestBody       `json:"filters"`
	RPMLimit        int64                           `json:"rpm_limit"`
	CostLimitRules  []AccessKeyCostLimitRuleRequest `json:"cost_limit_rules,omitempty"`
	ExpiresAtMS     *int64                          `json:"expires_at_ms,omitempty"`
}

func (s *Service) CreateAccessKeyIdempotent(
	ctx context.Context,
	idempotencyKey string,
	request AccessKeyCreateRequest,
) (AccessKeyCreateResult, error) {
	normalized, err := normalizeAccessKeyCreateRequest(request)
	if err != nil {
		return AccessKeyCreateResult{}, err
	}
	var digestStatus *state.AccessKeyStatus
	if normalized.status != state.AccessKeyStatusActive {
		digestStatus = &normalized.status
	}
	canonicalBody, err := canonicalIdempotencyBody(accessKeyCreateDigestBody{
		PriceMultiplier: priceMultiplierDigest(normalized.priceMultiplier),
		Name:            normalized.name, Status: digestStatus,
		Filters:        canonicalAccessKeyFilterSet(normalized.filters),
		RPMLimit:       normalized.rpmLimit,
		CostLimitRules: costLimitRuleRequestsForDigest(normalized.costLimitRules),
		ExpiresAtMS:    normalized.expiresAtMS,
	})
	if err != nil {
		return AccessKeyCreateResult{}, app_errors.ErrInternalServer
	}
	digest, err := buildIdempotencyDigest(idempotencyDigestInput{
		Version:         1,
		Method:          "POST",
		OperationKind:   operationKindAccessKeyCreate,
		PathTemplate:    "/api/access-keys",
		ResourceLocator: "new",
		AuthScopeID:     idempotencyAuthScopeID,
		CanonicalBody:   canonicalBody,
	})
	if err != nil {
		return AccessKeyCreateResult{}, app_errors.ErrInternalServer
	}

	var operationStartedAt time.Time
	operationResult, err := s.executeIdempotentOperation(ctx, idempotentOperationInput{
		IdempotencyKey: idempotencyKey,
		DigestVersion:  1,
		RequestDigest:  digest.Digest,
		Kind:           operationKindAccessKeyCreate,
		PrepareMutation: func() {
			operationStartedAt = s.now()
		},
		Mutate: func(tx *gorm.DB) (idempotentMutationResult, error) {
			mutation, err := s.mutateCreateAccessKey(tx, normalized, operationStartedAt)
			if err != nil {
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
			canonicalResult, err := canonicaljson.Marshal(mutation.metadata)
			if err != nil {
				return idempotentMutationResult{}, fmt.Errorf(
					"encode AccessKey operation result: %w",
					app_errors.ErrInternalServer,
				)
			}
			return idempotentMutationResult{
				ResourceIdentity: fmt.Sprintf("access-key:%d", mutation.metadata.ID),
				CanonicalResult:  canonicalResult,
				Ephemeral:        mutation.plaintext,
			}, nil
		},
	})
	if err != nil {
		return AccessKeyCreateResult{}, err
	}
	var metadata AccessKeyMetadata
	if err := json.Unmarshal(operationResult.CanonicalResult, &metadata); err != nil {
		return AccessKeyCreateResult{}, app_errors.ErrInternalServer
	}
	if metadata.PriceMultiplier == "" {
		metadata.PriceMultiplier = "1"
	}
	if metadata.CostLimitRules == nil {
		// Pre-0002 idempotency results did not carry this additive field. Preserve
		// replay compatibility while keeping the current wire contract array-shaped.
		metadata.CostLimitRules = []AccessKeyCostLimitRule{}
	}
	if metadata.Filters.AllowedCIDRs == nil {
		// Pre-0007 operation results did not carry this additive field. Preserve
		// replay compatibility while keeping the current wire contract array-shaped.
		metadata.Filters.AllowedCIDRs = []string{}
	}
	result := AccessKeyCreateResult{
		AccessKeyMetadata: metadata,
		Replayed:          operationResult.Replayed,
	}
	if plaintext, ok := operationResult.Ephemeral.(string); ok &&
		!operationResult.Replayed {
		result.Key = plaintext
	}
	return result, nil
}

func canonicalAccessKeyFilterSet(filters AccessKeyFilters) accessKeyFilterDigestBody {
	result := accessKeyFilterDigestBody{
		Groups:       append([]uint(nil), filters.Groups...),
		Protocols:    append([]protocol.Protocol(nil), filters.Protocols...),
		Models:       append([]string(nil), filters.Models...),
		AllowedCIDRs: append([]string(nil), filters.AllowedCIDRs...),
	}
	sort.Slice(result.Groups, func(left, right int) bool {
		return result.Groups[left] < result.Groups[right]
	})
	sort.Slice(result.Protocols, func(left, right int) bool {
		return string(result.Protocols[left]) < string(result.Protocols[right])
	})
	sort.Strings(result.Models)
	sort.Strings(result.AllowedCIDRs)
	return result
}
