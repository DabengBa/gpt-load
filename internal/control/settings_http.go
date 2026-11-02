package control

import (
	"github.com/gin-gonic/gin"

	"gpt-load/internal/platform/i18n"
	"gpt-load/internal/platform/response"
)

func (s *Server) handleGetSettings(c *gin.Context) {
	result, err := s.service.GetSettings(c.Request.Context())
	if err != nil {
		writeServiceError(c, "get_settings", err)
		return
	}
	writeSettingsResponse(c, result)
}

func (s *Server) handleUpdateSettings(c *gin.Context) {
	var request SettingsUpdateRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "update_settings", mapControlJSONError(err))
		return
	}
	result, err := s.service.UpdateSettings(c.Request.Context(), request)
	if err != nil {
		writeServiceError(c, "update_settings", err)
		return
	}
	writeSettingsResponse(c, result)
}

func writeSettingsResponse(c *gin.Context, settings SettingsResponse) {
	c.Header("Cache-Control", "no-store")
	c.Header("Pragma", "no-cache")
	c.Header("Content-Language", i18n.GetLanguageFromContext(c))
	c.Header("Vary", "Accept-Language")
	response.SuccessI18n(c, "common.success", settings)
}
