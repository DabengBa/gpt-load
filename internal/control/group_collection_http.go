package control

import (
	"github.com/gin-gonic/gin"

	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
	"gpt-load/internal/storage/models"
)

const (
	groupCollectionDefaultPage     int64 = 1
	groupCollectionDefaultPageSize int64 = 100
	groupCollectionMaxPageSize     int64 = 100
	groupCollectionMaxQueryRunes         = 200
)

func (s *Server) handleListGroupCollection(c *gin.Context) {
	query, apiErr := parseGroupCollectionQuery(
		c.Request.URL.RawQuery,
		c.Request.URL.ForceQuery,
	)
	if apiErr != nil {
		writeServiceError(c, "list_groups", apiErr)
		return
	}
	result, err := s.service.ListGroupCollection(c.Request.Context(), query)
	if err != nil {
		writeServiceError(c, "list_groups", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleListGroupOptions(c *gin.Context) {
	if !requireEmptyQuery(c, "list_group_options") {
		return
	}
	result, err := s.service.ListGroupOptions(c.Request.Context())
	if err != nil {
		writeServiceError(c, "list_group_options", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func parseGroupCollectionQuery(
	rawQuery string,
	forceQuery bool,
) (GroupCollectionQuery, *app_errors.APIError) {
	query := GroupCollectionQuery{
		Sort:     GroupCollectionSortRecent,
		Page:     groupCollectionDefaultPage,
		PageSize: groupCollectionDefaultPageSize,
	}
	values, apiErr := parseCollectionQueryValues(
		rawQuery, forceQuery, "q", "status", "connection_type", "sort", "page", "page_size",
	)
	if apiErr != nil {
		return GroupCollectionQuery{}, apiErr
	}

	text, ok := collectionQueryText(values, "q", groupCollectionMaxQueryRunes)
	if !ok {
		return GroupCollectionQuery{}, app_errors.ErrBadRequest
	}
	query.Query = text
	if entries, exists := values["status"]; exists {
		status, ok := parseGroupCollectionStatus(entries[0])
		if !ok {
			return GroupCollectionQuery{}, app_errors.ErrBadRequest
		}
		query.Status = status
	}
	if entries, exists := values["connection_type"]; exists {
		connectionType, ok := parseGroupCollectionConnectionType(entries[0])
		if !ok {
			return GroupCollectionQuery{}, app_errors.ErrBadRequest
		}
		query.ConnectionType = connectionType
	}
	if entries, exists := values["sort"]; exists {
		sortValue := GroupCollectionSort(entries[0])
		switch sortValue {
		case GroupCollectionSortRecent,
			GroupCollectionSortStatus,
			GroupCollectionSortName,
			GroupCollectionSortCredentials,
			GroupCollectionSortCreated:
			query.Sort = sortValue
		default:
			return GroupCollectionQuery{}, app_errors.ErrBadRequest
		}
	}
	if entries, exists := values["page"]; exists {
		page, ok := parseCollectionPositiveInt64(entries[0])
		if !ok {
			return GroupCollectionQuery{}, app_errors.ErrBadRequest
		}
		query.Page = page
	}
	if entries, exists := values["page_size"]; exists {
		pageSize, ok := parseCollectionPositiveInt64(entries[0])
		if !ok || pageSize > groupCollectionMaxPageSize {
			return GroupCollectionQuery{}, app_errors.ErrBadRequest
		}
		query.PageSize = pageSize
	}
	return query, nil
}

func parseGroupCollectionStatus(
	value string,
) (*GroupCollectionStatus, bool) {
	status := GroupCollectionStatus(value)
	switch status {
	case GroupCollectionStatusAvailable,
		GroupCollectionStatusUnavailable,
		GroupCollectionStatusDisabled:
		return &status, true
	default:
		return nil, false
	}
}

func parseGroupCollectionConnectionType(
	value string,
) (*models.ConnectionType, bool) {
	connectionType := models.ConnectionType(value)
	switch connectionType {
	case models.ConnectionTypeAPIKey, models.ConnectionTypeSubscription:
		return &connectionType, true
	default:
		return nil, false
	}
}
