package control

import (
	"github.com/gin-gonic/gin"

	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
	"gpt-load/internal/state"
)

const (
	accessKeyCollectionMaxPageSize   int64 = 100
	accessKeyCollectionMaxQueryRunes       = 200
)

func (s *Server) handleListAccessKeyCollection(c *gin.Context) {
	query, apiErr := parseAccessKeyCollectionQuery(
		c.Request.URL.RawQuery,
		c.Request.URL.ForceQuery,
	)
	if apiErr != nil {
		writeServiceError(c, "list_access_keys", apiErr)
		return
	}
	result, err := s.service.ListAccessKeyCollection(c.Request.Context(), query)
	if err != nil {
		writeServiceError(c, "list_access_keys", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func parseAccessKeyCollectionQuery(
	rawQuery string,
	forceQuery bool,
) (AccessKeyCollectionQuery, *app_errors.APIError) {
	query := AccessKeyCollectionQuery{
		Page:     defaultAccessKeyCollectionPage,
		PageSize: defaultAccessKeyCollectionPageSize,
	}
	values, apiErr := parseCollectionQueryValues(
		rawQuery, forceQuery, "q", "status", "page", "page_size",
	)
	if apiErr != nil {
		return AccessKeyCollectionQuery{}, apiErr
	}

	text, ok := collectionQueryText(values, "q", accessKeyCollectionMaxQueryRunes)
	if !ok {
		return AccessKeyCollectionQuery{}, app_errors.ErrBadRequest
	}
	query.Query = text
	if entries, exists := values["status"]; exists {
		status, ok := parseAccessKeyCollectionStatus(entries[0])
		if !ok {
			return AccessKeyCollectionQuery{}, app_errors.ErrBadRequest
		}
		query.Status = &status
	}
	if entries, exists := values["page"]; exists {
		page, ok := parseCollectionPositiveInt64(entries[0])
		if !ok {
			return AccessKeyCollectionQuery{}, app_errors.ErrBadRequest
		}
		query.Page = page
	}
	if entries, exists := values["page_size"]; exists {
		pageSize, ok := parseCollectionPositiveInt64(entries[0])
		if !ok || pageSize > accessKeyCollectionMaxPageSize {
			return AccessKeyCollectionQuery{}, app_errors.ErrBadRequest
		}
		query.PageSize = pageSize
	}
	return query, nil
}

func parseAccessKeyCollectionStatus(value string) (state.AccessKeyStatus, bool) {
	status := state.AccessKeyStatus(value)
	switch status {
	case state.AccessKeyStatusActive, state.AccessKeyStatusDisabled:
		return status, true
	default:
		return "", false
	}
}
