package control

import (
	"fmt"
	"strconv"

	"github.com/gin-gonic/gin"

	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
)

func (s *Server) handleGetGroupSummary(c *gin.Context) {
	id, ok := groupID(c, "get_group_summary")
	if !ok {
		return
	}
	result, err := s.service.GetGroupSummary(c.Request.Context(), id)
	if err != nil {
		writeServiceError(c, "get_group_summary", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleGetGroupSettings(c *gin.Context) {
	id, ok := groupID(c, "get_group_settings")
	if !ok {
		return
	}
	result, err := s.service.GetGroupSettings(c.Request.Context(), id)
	if err != nil {
		writeServiceError(c, "get_group_settings", err)
		return
	}
	response.SuccessI18n(c, "common.success", groupSettingsHTTPProjection(result, wantsSiteInfo(c)))
}

func (s *Server) handleCreateGroup(c *gin.Context) {
	idempotencyKey, ok := requiredIdempotencyKey(c, "create_group")
	if !ok {
		return
	}
	var request GroupCreateRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "create_group", mapControlJSONError(err))
		return
	}
	result, err := s.service.CreateGroupIdempotent(c.Request.Context(), idempotencyKey, request)
	if err != nil {
		writeServiceError(c, "create_group", err)
		return
	}
	setMutationResourceLocator(c, fmt.Sprintf("group:%d", result.GroupID))
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleCopyGroup(c *gin.Context) {
	idempotencyKey, ok := requiredIdempotencyKey(c, "copy_group")
	if !ok {
		return
	}
	id, ok := groupID(c, "copy_group")
	if !ok {
		return
	}
	if err := bindOptionalEmptyJSONObject(c); err != nil {
		writeServiceError(c, "copy_group", mapControlJSONError(err))
		return
	}
	result, err := s.service.CopyGroupIdempotent(c.Request.Context(), idempotencyKey, id)
	if err != nil {
		writeServiceError(c, "copy_group", err)
		return
	}
	setMutationResourceLocator(c, fmt.Sprintf("group:%d", result.GroupID))
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleUpdateGroupSettings(c *gin.Context) {
	id, ok := groupID(c, "update_group_settings")
	if !ok {
		return
	}
	var request GroupSettingsUpdateRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "update_group_settings", mapControlJSONError(err))
		return
	}
	result, err := s.service.UpdateGroupSettings(c.Request.Context(), id, request)
	if err != nil {
		writeServiceError(c, "update_group_settings", err)
		return
	}
	response.SuccessI18n(c, "common.success", groupSettingsHTTPProjection(result, wantsSiteInfo(c)))
}

func wantsSiteInfo(c *gin.Context) bool {
	values := c.Request.URL.Query()["include"]
	return len(values) == 1 && values[0] == "site_info"
}

func groupSettingsHTTPProjection(result GroupSettingsResponse, include bool) any {
	projected := struct {
		GroupSettingsResponse
		SiteAccess *GroupSiteAccessResponse `json:"site_access,omitempty"`
	}{GroupSettingsResponse: result}
	if include {
		projected.SiteAccess = &result.SiteAccess
	}
	return projected
}

func (s *Server) handleUpdateGroupModels(c *gin.Context) {
	id, ok := groupID(c, "update_group_models")
	if !ok {
		return
	}
	var request GroupModelsUpdateRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "update_group_models", mapControlJSONError(err))
		return
	}
	result, err := s.service.UpdateGroupModels(c.Request.Context(), id, request)
	if err != nil {
		writeServiceError(c, "update_group_models", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleGetGroupModels(c *gin.Context) {
	id, ok := groupID(c, "get_group_models")
	if !ok {
		return
	}
	result, err := s.service.GetGroupModels(c.Request.Context(), id)
	if err != nil {
		writeServiceError(c, "get_group_models", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleDeleteGroup(c *gin.Context) {
	id, ok := groupID(c, "delete_group")
	if !ok {
		return
	}
	if err := s.service.DeleteGroup(c.Request.Context(), id); err != nil {
		writeServiceError(c, "delete_group", err)
		return
	}
	response.SuccessI18n(c, "common.success", nil)
}

func (s *Server) handleGetGroupCredential(c *gin.Context) {
	groupID, ok := groupID(c, "get_group_credential")
	if !ok {
		return
	}
	result, err := s.service.GetGroupCredential(c.Request.Context(), groupID)
	if err != nil {
		writeServiceError(c, "get_group_credential", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) resolveGroupCredential(c *gin.Context) {
	expectedID, err := strconv.ParseUint(c.GetHeader("X-Credential-ID"), 10, strconv.IntSize)
	if err != nil || expectedID == 0 {
		writeServiceError(c, "resolve_group_credential", app_errors.ErrValidation)
		c.Abort()
		return
	}
	id, ok := groupID(c, "resolve_group_credential")
	if !ok {
		c.Abort()
		return
	}
	credentialID, err := s.service.currentGroupCredentialID(c.Request.Context(), id)
	if err == nil && uint(expectedID) != credentialID {
		err = app_errors.ErrCredentialVersionConflict
	}
	if err == nil && credentialID == 0 {
		err = credentialNotFoundError()
	}
	if err != nil {
		writeServiceError(c, "resolve_group_credential", err)
		c.Abort()
		return
	}
	// Pin the ID before auditing and mutation so a concurrent replacement is rejected
	// by the existing group-and-ID checks, never silently retargeted.
	c.Params = append(c.Params, gin.Param{Key: "credential_id", Value: strconv.FormatUint(uint64(credentialID), 10)})
}

func (s *Server) handleRevealGroupCredential(c *gin.Context) {
	groupID, ok := groupID(c, "reveal_group_credential")
	if !ok {
		return
	}
	credentialID, ok := credentialID(c, "reveal_group_credential")
	if !ok {
		return
	}
	if err := bindOptionalEmptyJSONObject(c); err != nil {
		writeServiceError(c, "reveal_group_credential", mapControlJSONError(err))
		return
	}
	result, err := s.service.RevealGroupCredential(c.Request.Context(), groupID, credentialID)
	if err != nil {
		writeServiceError(c, "reveal_group_credential", err)
		return
	}
	setSecretResponseHeaders(c)
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleUpdateGroupCredential(c *gin.Context) {
	groupID, ok := groupID(c, "update_group_credential")
	if !ok {
		return
	}
	credentialID, ok := credentialID(c, "update_group_credential")
	if !ok {
		return
	}
	var request CredentialUpdateRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "update_group_credential", mapControlJSONError(err))
		return
	}
	result, err := s.service.UpdateGroupCredential(c.Request.Context(), groupID, credentialID, request)
	if err != nil {
		writeServiceError(c, "update_group_credential", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleDeleteGroupCredential(c *gin.Context) {
	groupID, ok := groupID(c, "delete_group_credential")
	if !ok {
		return
	}
	credentialID, ok := credentialID(c, "delete_group_credential")
	if !ok {
		return
	}
	if err := bindOptionalEmptyJSONObject(c); err != nil {
		writeServiceError(c, "delete_group_credential", mapControlJSONError(err))
		return
	}
	if err := s.service.DeleteGroupCredential(c.Request.Context(), groupID, credentialID); err != nil {
		writeServiceError(c, "delete_group_credential", err)
		return
	}
	response.SuccessI18n(c, "common.success", nil)
}

func (s *Server) handleRestoreGroupCredential(c *gin.Context) {
	groupID, ok := groupID(c, "restore_group_credential")
	if !ok {
		return
	}
	credentialID, ok := credentialID(c, "restore_group_credential")
	if !ok {
		return
	}
	if err := bindOptionalEmptyJSONObject(c); err != nil {
		writeServiceError(c, "restore_group_credential", mapControlJSONError(err))
		return
	}
	result, err := s.service.RestoreGroupCredential(c.Request.Context(), groupID, credentialID)
	if err != nil {
		writeServiceError(c, "restore_group_credential", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleTestGroupCredential(c *gin.Context) {
	groupID, ok := groupID(c, "test_group_credential")
	if !ok {
		return
	}
	credentialID, ok := credentialID(c, "test_group_credential")
	if !ok {
		return
	}
	if err := bindOptionalEmptyJSONObject(c); err != nil {
		writeServiceError(c, "test_group_credential", mapControlJSONError(err))
		return
	}
	result, err := s.service.TestGroupCredential(c.Request.Context(), groupID, credentialID)
	if err != nil {
		writeServiceError(c, "test_group_credential", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleImportGroupCredentials(c *gin.Context) {
	id, ok := groupID(c, "import_group_credentials")
	if !ok {
		return
	}
	idempotencyKey, ok := requiredIdempotencyKey(c, "import_group_credentials")
	if !ok {
		return
	}
	var request CredentialImportRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "import_group_credentials", mapControlJSONError(err))
		return
	}
	result, err := s.service.ImportGroupCredentialsIdempotent(c.Request.Context(), idempotencyKey, id, request)
	if err != nil {
		writeServiceError(c, "import_group_credentials", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleDiscoverGroupModels(c *gin.Context) {
	id, ok := groupID(c, "discover_group_models")
	if !ok {
		return
	}
	if err := bindOptionalEmptyJSONObject(c); err != nil {
		writeServiceError(c, "discover_group_models", mapControlJSONError(err))
		return
	}
	result, err := s.service.DiscoverGroupModels(c.Request.Context(), id)
	if err != nil {
		writeServiceError(c, "discover_group_models", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleDiscoverModels(c *gin.Context) {
	var request ModelDiscoveryRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeDiscoveryError(c, mapControlJSONError(err))
		return
	}
	result, err := s.service.DiscoverModels(c.Request.Context(), request)
	if err != nil {
		writeDiscoveryError(c, err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}
