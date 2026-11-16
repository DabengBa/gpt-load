package requestlog

import (
	"context"
	"fmt"
	"strings"

	"gpt-load/internal/storage/models"
)

type FacetGroup struct {
	ID   uint   `json:"id"`
	Name string `json:"name,omitempty"`
}

type FacetClientModel struct {
	Value string `json:"value"`
}

type FacetResult struct {
	Groups       []FacetGroup       `json:"groups"`
	ClientModels []FacetClientModel `json:"client_models"`
}

func (service *Service) Facets(ctx context.Context, input ListQuery) (FacetResult, error) {
	result := FacetResult{Groups: []FacetGroup{}, ClientModels: []FacetClientModel{}}
	input.Cursor = nil
	groupInput := input
	groupInput.GroupID = nil
	groupBaseInput := withoutAttemptFilters(groupInput)
	base := service.db.WithContext(ctx).Model(&models.RequestLog{}).Select("id")
	base = applyListIdentityFilters(base, groupBaseInput)
	base = applyListMetricFilters(base, groupBaseInput)
	groups := service.db.WithContext(ctx).Table("request_log_attempts AS attempt").
		Select("attempt.group_id AS id, MAX(attempt.group_name) AS name").
		Where("attempt.request_id IN (?) AND attempt.group_id > 0", base)
	conditions, arguments := attemptFilterConditions(groupInput)
	if len(conditions) > 0 {
		groups = groups.Where(strings.Join(conditions, " AND "), arguments...)
	}
	if err := groups.Group("attempt.group_id").Order("attempt.group_id").Scan(&result.Groups).Error; err != nil {
		return FacetResult{}, fmt.Errorf("query request log group facets: %w", err)
	}
	modelInput := input
	modelInput.ClientModel = ""
	modelBaseInput := withoutAttemptFilters(modelInput)
	modelsQuery := service.db.WithContext(ctx).Model(&models.RequestLog{}).
		Select("DISTINCT client_model AS value").Where("client_model <> ?", "")
	modelsQuery = applyListIdentityFilters(modelsQuery, modelBaseInput)
	modelsQuery = applyListMetricFilters(modelsQuery, modelBaseInput)
	conditions, arguments = attemptFilterConditions(modelInput)
	if len(conditions) > 0 {
		attempts := service.db.WithContext(ctx).Table("request_log_attempts AS attempt").
			Select("attempt.request_id").Where(strings.Join(conditions, " AND "), arguments...)
		modelsQuery = modelsQuery.Where("id IN (?)", attempts)
	}
	if err := modelsQuery.Order("client_model").Scan(&result.ClientModels).Error; err != nil {
		return FacetResult{}, fmt.Errorf("query request log model facets: %w", err)
	}
	return result, nil
}

func withoutAttemptFilters(input ListQuery) ListQuery {
	input.GroupID = nil
	input.ChannelID = ""
	input.CredentialID = nil
	input.UpstreamModel = ""
	input.AttemptStatusCode = nil
	input.FailureCategory = ""
	input.AttemptErrorCode = ""
	return input
}
