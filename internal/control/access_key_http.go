package control

import (
	"fmt"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/platform/response"
)

func (s *Server) handleCreateAccessKey(c *gin.Context) {
	idempotencyKey, ok := requiredIdempotencyKey(c, "create_access_key")
	if !ok {
		return
	}
	var request AccessKeyCreateRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "create_access_key", mapControlJSONError(err))
		return
	}
	result, err := s.service.CreateAccessKeyIdempotent(c.Request.Context(), idempotencyKey, request)
	if err != nil {
		writeServiceError(c, "create_access_key", err)
		return
	}
	setMutationResourceLocator(c, fmt.Sprintf("access-key:%d", result.ID))
	setSecretResponseHeaders(c)
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleListAccessKeyOptions(c *gin.Context) {
	result, err := s.service.ListAccessKeyOptions(c.Request.Context())
	if err != nil {
		writeServiceError(c, "list_access_key_options", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleRevealAccessKey(c *gin.Context) {
	id, ok := accessKeyID(c)
	if !ok {
		return
	}
	result, err := s.service.RevealAccessKey(c.Request.Context(), id)
	if err != nil {
		writeServiceError(c, "reveal_access_key", err)
		return
	}
	setSecretResponseHeaders(c)
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleRotateAccessKey(c *gin.Context) {
	id, ok := accessKeyID(c)
	if !ok {
		return
	}
	idempotencyKey, ok := requiredIdempotencyKey(c, "rotate_access_key")
	if !ok {
		return
	}
	if err := bindOptionalEmptyJSONObject(c); err != nil {
		writeServiceError(c, "rotate_access_key", mapControlJSONError(err))
		return
	}
	result, err := s.service.RotateAccessKeyIdempotent(c.Request.Context(), idempotencyKey, id)
	if err != nil {
		writeServiceError(c, "rotate_access_key", err)
		return
	}
	setMutationResourceLocator(c, fmt.Sprintf("access-key:%d", result.ID))
	setSecretResponseHeaders(c)
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleUpdateAccessKey(c *gin.Context) {
	id, ok := accessKeyID(c)
	if !ok {
		return
	}
	var request AccessKeyUpdateRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "update_access_key", mapControlJSONError(err))
		return
	}
	result, err := s.service.UpdateAccessKey(c.Request.Context(), id, request)
	if err != nil {
		writeServiceError(c, "update_access_key", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleResetAccessKeyCostLimits(c *gin.Context) {
	id, ok := accessKeyID(c)
	if !ok {
		return
	}
	var request AccessKeyCostLimitResetRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "reset_access_key_cost_limits", mapControlJSONError(err))
		return
	}
	if err := s.service.ResetAccessKeyCostLimitRules(c.Request.Context(), id, request.RuleIDs); err != nil {
		writeServiceError(c, "reset_access_key_cost_limits", err)
		return
	}
	response.SuccessI18n(c, "common.success", nil)
}

func (s *Server) handleDeleteAccessKey(c *gin.Context) {
	id, ok := accessKeyID(c)
	if !ok {
		return
	}
	if err := s.service.DeleteAccessKey(c.Request.Context(), id); err != nil {
		writeServiceError(c, "delete_access_key", err)
		return
	}
	response.SuccessI18n(c, "common.success", nil)
}
