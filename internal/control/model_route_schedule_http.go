package control

import (
	"github.com/gin-gonic/gin"

	"gpt-load/internal/platform/response"
)

func (s *Server) handleModelRouteScheduleIndex(c *gin.Context) {
	result, err := s.service.GetModelRouteScheduleIndex()
	if err != nil {
		writeServiceError(c, "list_model_route_schedule", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleModelRouteScheduleDetail(c *gin.Context) {
	request, apiErr := parseModelRouteScheduleDetailQuery(c)
	if apiErr != nil {
		writeServiceError(c, "get_model_route_schedule_detail", apiErr)
		return
	}
	result, err := s.service.GetModelRouteScheduleDetail(request)
	if err != nil {
		writeServiceError(c, "get_model_route_schedule_detail", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleUpdateModelRouteSchedule(c *gin.Context) {
	var request modelRouteSchedulePatchRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "update_model_route_schedule", mapControlJSONError(err))
		return
	}
	result, err := s.service.UpdateModelRouteSchedule(c.Request.Context(), request)
	if err != nil {
		writeServiceError(c, "update_model_route_schedule", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleRecoverModelRouteScheduleEntry(c *gin.Context) {
	var request modelRouteScheduleRecoverRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "recover_model_route_schedule", mapControlJSONError(err))
		return
	}
	result, err := s.service.RecoverModelRouteScheduleEntry(request)
	if err != nil {
		writeServiceError(c, "recover_model_route_schedule", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}
