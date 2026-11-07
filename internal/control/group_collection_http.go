package control

import (
	"github.com/gin-gonic/gin"

	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
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
	items := make([]groupCollectionHTTPItem, 0, len(result.Items))
	for _, item := range result.Items {
		projected := groupCollectionHTTPItem{GroupCollectionItem: item}
		if wantsSiteInfo(c) {
			projected.SiteInfo = &item.SiteInfo
		}
		items = append(items, projected)
	}
	response.SuccessI18n(c, "common.success", struct {
		ObservedAtMS int64                     `json:"observed_at_ms"`
		Summary      GroupCollectionSummary    `json:"summary"`
		Items        []groupCollectionHTTPItem `json:"items"`
		Pagination   GroupCollectionPagination `json:"pagination"`
	}{result.ObservedAtMS, result.Summary, items, result.Pagination})
}

// Shadow the service field so legacy HTTP clients see the original whitelist.
// A pointer to a pointer preserves explicit null for opted-in unconfigured groups.
type groupCollectionHTTPItem struct {
	GroupCollectionItem
	SiteInfo **GroupCollectionSiteInfo `json:"site_info,omitempty"`
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
		rawQuery, forceQuery, "q", "status", "sort", "page", "page_size", "include",
	)
	if apiErr != nil {
		return GroupCollectionQuery{}, apiErr
	}
	if entries, exists := values["include"]; exists && entries[0] != "site_info" {
		return GroupCollectionQuery{}, app_errors.ErrBadRequest
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
