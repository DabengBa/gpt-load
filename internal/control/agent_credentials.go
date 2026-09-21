package control

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"

	"gpt-load/internal/agent"
	"gpt-load/internal/platform/canonicaljson"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
)

// AgentCredentialCreateRequest is the admin-authenticated create request for
// one independent Agent credential.
type AgentCredentialCreateRequest struct {
	Name        string   `json:"name"`
	Scopes      []string `json:"scopes"`
	ExpiresAtMS *int64   `json:"expires_at_ms"`
}

// AgentCredentialCreateResponse returns the safe metadata and, exactly once,
// the plaintext secret.
type AgentCredentialCreateResponse struct {
	agent.CredentialMetadata
	Secret      string `json:"secret,omitempty"`
	Replayed    bool   `json:"replayed"`
	OperationID string `json:"operation_id"`
}

// AgentCredentialListResponse is the credential ledger read model.
type AgentCredentialListResponse struct {
	Items []agent.CredentialMetadata `json:"items"`
}

type agentCredentialCreateDigestBody struct {
	Name        string        `json:"name"`
	Scopes      []agent.Scope `json:"scopes"`
	ExpiresAtMS *int64        `json:"expires_at_ms,omitempty"`
}

// SetAgentCredentialStore wires the Agent credential ledger into the control
// plane. It is optional so existing constructions keep working; when it is not
// wired the management routes fail closed.
func (service *Service) SetAgentCredentialStore(store *agent.CredentialStore) {
	if service != nil {
		service.agentCredentials = store
	}
}

// CreateAgentCredentialIdempotent creates one Agent credential under the
// shared idempotency ledger. The plaintext secret is returned only on the
// first execution; replays return metadata only.
func (service *Service) CreateAgentCredentialIdempotent(
	ctx context.Context,
	idempotencyKey string,
	request AgentCredentialCreateRequest,
) (AgentCredentialCreateResponse, error) {
	if service == nil || service.agentCredentials == nil {
		return AgentCredentialCreateResponse{}, app_errors.ErrInternalServer
	}
	name, err := normalizeAccessKeyName(request.Name)
	if err != nil {
		return AgentCredentialCreateResponse{}, app_errors.ErrValidation
	}
	scopes, err := agent.NormalizeScopes(request.Scopes)
	if err != nil {
		return AgentCredentialCreateResponse{}, app_errors.ErrValidation
	}
	if err := validateOptionalExpiresAtMS(request.ExpiresAtMS); err != nil {
		return AgentCredentialCreateResponse{}, err
	}
	canonicalBody, err := canonicalIdempotencyBody(agentCredentialCreateDigestBody{
		Name: name, Scopes: scopes, ExpiresAtMS: request.ExpiresAtMS,
	})
	if err != nil {
		return AgentCredentialCreateResponse{}, app_errors.ErrInternalServer
	}
	digest, err := buildIdempotencyDigest(idempotencyDigestInput{
		Version:         1,
		Method:          "POST",
		OperationKind:   operationKindAgentCredentialCreate,
		PathTemplate:    "/api/agent-credentials",
		ResourceLocator: "new",
		AuthScopeID:     idempotencyAuthScopeID,
		CanonicalBody:   canonicalBody,
	})
	if err != nil {
		return AgentCredentialCreateResponse{}, app_errors.ErrInternalServer
	}
	var operationStartedAt time.Time
	operationResult, err := service.executeIdempotentOperation(ctx, idempotentOperationInput{
		IdempotencyKey: idempotencyKey,
		DigestVersion:  1,
		RequestDigest:  digest.Digest,
		Kind:           operationKindAgentCredentialCreate,
		PrepareMutation: func() {
			operationStartedAt = service.now()
		},
		Mutate: func(tx *gorm.DB) (idempotentMutationResult, error) {
			if err := validateFutureExpiresAtMS(request.ExpiresAtMS, operationStartedAt); err != nil {
				return idempotentMutationResult{}, err
			}
			metadata, secret, err := service.agentCredentials.CreateInTx(tx, agent.CredentialCreateInput{
				Name:        name,
				Scopes:      scopes,
				ExpiresAtMS: request.ExpiresAtMS,
			})
			if err != nil {
				return idempotentMutationResult{}, err
			}
			canonicalResult, err := canonicaljson.Marshal(metadata)
			if err != nil {
				return idempotentMutationResult{}, fmt.Errorf(
					"encode Agent credential operation result: %w",
					app_errors.ErrInternalServer,
				)
			}
			return idempotentMutationResult{
				ResourceIdentity: fmt.Sprintf("agent-credential:%d", metadata.ID),
				CanonicalResult:  canonicalResult,
				Ephemeral:        secret,
			}, nil
		},
	})
	if err != nil {
		return AgentCredentialCreateResponse{}, err
	}
	var metadata agent.CredentialMetadata
	if err := json.Unmarshal(operationResult.CanonicalResult, &metadata); err != nil {
		return AgentCredentialCreateResponse{}, app_errors.ErrInternalServer
	}
	result := AgentCredentialCreateResponse{
		CredentialMetadata: metadata,
		Replayed:           operationResult.Replayed,
		OperationID:        operationResult.OperationID,
	}
	if secret, ok := operationResult.Ephemeral.(string); ok && !operationResult.Replayed {
		result.Secret = secret
	}
	return result, nil
}

// ListAgentCredentials returns every credential's safe metadata.
func (service *Service) ListAgentCredentials(ctx context.Context) (AgentCredentialListResponse, error) {
	if service == nil || service.agentCredentials == nil {
		return AgentCredentialListResponse{}, app_errors.ErrInternalServer
	}
	items, err := service.agentCredentials.List(ctx)
	if err != nil {
		return AgentCredentialListResponse{}, err
	}
	return AgentCredentialListResponse{Items: items}, nil
}

// DisableAgentCredential revokes one credential. Disabling is idempotent and
// takes effect for the next authenticated Agent request.
func (service *Service) DisableAgentCredential(
	ctx context.Context,
	id uint,
) (agent.CredentialMetadata, error) {
	if service == nil || service.agentCredentials == nil {
		return agent.CredentialMetadata{}, app_errors.ErrInternalServer
	}
	return service.agentCredentials.Disable(ctx, id)
}

func (s *Server) handleCreateAgentCredential(c *gin.Context) {
	idempotencyKey, ok := requiredIdempotencyKey(c, "create_agent_credential")
	if !ok {
		return
	}
	var request AgentCredentialCreateRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "create_agent_credential", mapControlJSONError(err))
		return
	}
	result, err := s.service.CreateAgentCredentialIdempotent(
		c.Request.Context(),
		idempotencyKey,
		request,
	)
	if err != nil {
		writeServiceError(c, "create_agent_credential", err)
		return
	}
	setMutationResourceLocator(c, fmt.Sprintf("agent-credential:%d", result.ID))
	setSecretResponseHeaders(c)
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleListAgentCredentials(c *gin.Context) {
	result, err := s.service.ListAgentCredentials(c.Request.Context())
	if err != nil {
		writeServiceError(c, "list_agent_credentials", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleDisableAgentCredential(c *gin.Context) {
	id, ok := agentCredentialID(c)
	if !ok {
		return
	}
	result, err := s.service.DisableAgentCredential(c.Request.Context(), id)
	if err != nil {
		writeServiceError(c, "disable_agent_credential", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func agentCredentialID(c *gin.Context) (uint, bool) {
	parsed, err := strconv.ParseUint(c.Param("id"), 10, strconv.IntSize)
	if err != nil || parsed == 0 {
		writeServiceError(c, "agent_credential_id", app_errors.ErrBadRequest)
		return 0, false
	}
	return uint(parsed), true
}
