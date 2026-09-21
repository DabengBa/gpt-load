package agent

import (
	"errors"
	"net/http"

	"github.com/gin-gonic/gin"

	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
)

func (server *Server) handleCreateChangeProposal(c *gin.Context) {
	if !requireScope(c, ScopeChangesPropose) {
		return
	}
	principal, ok := CurrentPrincipal(c)
	if !ok {
		writeAgentServiceError(c, "agent_create_change_proposal", app_errors.ErrUnauthorized)
		return
	}
	var request CreateChangeProposalInput
	if err := bindStrictAgentJSON(c, &request); err != nil {
		writeAgentServiceError(c, "agent_create_change_proposal", mapAgentJSONError(err))
		return
	}
	service, err := server.changeProposalService()
	if err != nil {
		writeAgentServiceError(c, "agent_create_change_proposal", err)
		return
	}
	result, err := service.CreateChangeProposal(c.Request.Context(), principal, request)
	if err != nil {
		writeAgentServiceError(c, "agent_create_change_proposal", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (server *Server) handleGetChangeProposal(c *gin.Context) {
	if !requireScope(c, ScopeDiagnosticsRead) {
		return
	}
	proposalID, ok := normalizeProposalID(c.Param(proposalIDRouteParam))
	if !ok {
		writeAgentServiceError(c, "agent_get_change_proposal", app_errors.ErrBadRequest)
		return
	}
	service, err := server.changeProposalService()
	if err != nil {
		writeAgentServiceError(c, "agent_get_change_proposal", err)
		return
	}
	result, err := service.GetChangeProposal(c.Request.Context(), proposalID)
	if err != nil {
		writeAgentServiceError(c, "agent_get_change_proposal", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (server *Server) handleApplyChangeProposal(c *gin.Context) {
	if !requireScope(c, ScopeChangesApply) {
		return
	}
	principal, ok := CurrentPrincipal(c)
	if !ok {
		writeAgentServiceError(c, "agent_apply_change_proposal", app_errors.ErrUnauthorized)
		return
	}
	proposalID, ok := normalizeProposalID(c.Param(proposalIDRouteParam))
	if !ok {
		writeAgentServiceError(c, "agent_apply_change_proposal", app_errors.ErrBadRequest)
		return
	}
	idempotencyKey, ok := requiredAgentIdempotencyKey(c)
	if !ok {
		return
	}
	service, err := server.changeProposalService()
	if err != nil {
		writeAgentServiceError(c, "agent_apply_change_proposal", err)
		return
	}
	result, err := service.ApplyChangeProposal(
		c.Request.Context(), principal, proposalID, idempotencyKey,
	)
	if err != nil {
		writeAgentServiceError(c, "agent_apply_change_proposal", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (server *Server) handleGetControlOperation(c *gin.Context) {
	if !requireScope(c, ScopeDiagnosticsRead) {
		return
	}
	operationID, ok := normalizeProposalID(c.Param(operationIDRouteParam))
	if !ok {
		writeAgentServiceError(c, "agent_get_control_operation", app_errors.ErrBadRequest)
		return
	}
	service, err := server.changeProposalService()
	if err != nil {
		writeAgentServiceError(c, "agent_get_control_operation", err)
		return
	}
	result, err := service.GetControlOperation(c.Request.Context(), operationID)
	if err != nil {
		writeAgentServiceError(c, "agent_get_control_operation", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func mapAgentJSONError(err error) error {
	if err == nil {
		return nil
	}
	var maxBytesError *http.MaxBytesError
	if errors.As(err, &maxBytesError) {
		return app_errors.ErrRequestTooLarge
	}
	return app_errors.ErrInvalidJSON
}

func requiredAgentIdempotencyKey(c *gin.Context) (string, bool) {
	values := c.Request.Header.Values("Idempotency-Key")
	if len(values) == 0 || len(values) == 1 && values[0] == "" {
		writeAgentServiceError(c, "agent_apply_change_proposal", app_errors.ErrIdempotencyKeyRequired)
		return "", false
	}
	if len(values) != 1 || !canonicalLowercaseUUIDv4.MatchString(values[0]) {
		writeAgentServiceError(c, "agent_apply_change_proposal", app_errors.ErrInvalidIdempotencyKey)
		return "", false
	}
	return values[0], true
}
