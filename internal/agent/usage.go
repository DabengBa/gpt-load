package agent

import (
	"errors"
	"net/url"
	"strconv"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/platform/epochms"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
	"gpt-load/internal/requestlog"
)

const agentUsageCustomMaxMS = 30 * epochms.MillisecondsPerDay

// errAgentUsageTotalMismatch marks a locally detected inconsistency in a usage
// report. It is never a client-visible message; the handler maps it to a
// generic internal error.
var errAgentUsageTotalMismatch = errors.New("agent usage report is internally inconsistent")

type agentUsagePreset struct {
	bucketWidthMS int64
	bucketCount   int
	granularity   requestlog.UsageGranularity
}

func agentUsageRangePreset(value string) (agentUsagePreset, bool) {
	switch value {
	case "1h":
		return agentUsagePreset{epochms.MillisecondsPerHour, 1, requestlog.UsageGranularityHour}, true
	case "24h":
		return agentUsagePreset{epochms.MillisecondsPerHour, 24, requestlog.UsageGranularityHour}, true
	case "3d":
		return agentUsagePreset{3 * epochms.MillisecondsPerHour, 24, requestlog.UsageGranularityHour}, true
	case "7d":
		return agentUsagePreset{6 * epochms.MillisecondsPerHour, 28, requestlog.UsageGranularityHour}, true
	case "15d":
		return agentUsagePreset{12 * epochms.MillisecondsPerHour, 30, requestlog.UsageGranularityHour}, true
	case "30d":
		return agentUsagePreset{epochms.MillisecondsPerDay, 30, requestlog.UsageGranularityDay}, true
	default:
		return agentUsagePreset{}, false
	}
}

func (server *Server) handleUsage(c *gin.Context) {
	if !requireScope(c, ScopeDiagnosticsRead) {
		return
	}
	observedAtMS := server.now().UTC().UnixMilli()
	query, apiErr := parseAgentUsageQuery(c.Request.URL.RawQuery, observedAtMS)
	if apiErr != nil {
		writeAgentServiceError(c, "agent_usage", apiErr)
		return
	}
	report, err := server.usage.QueryUsage(c.Request.Context(), query)
	if err != nil {
		writeAgentServiceError(c, "agent_usage", mapAgentReadError(err))
		return
	}
	view, err := mapAgentUsage(observedAtMS, query, report)
	if err != nil {
		writeAgentServiceError(c, "agent_usage", app_errors.ErrInternalServer)
		return
	}
	response.SuccessI18n(c, "common.success", view)
}

func parseAgentUsageQuery(
	rawQuery string,
	observedAtMS int64,
) (requestlog.UsageQuery, *app_errors.APIError) {
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return requestlog.UsageQuery{}, app_errors.ErrBadRequest
	}
	allowed := map[string]struct{}{
		"range": {}, "from_ms": {}, "to_ms": {}, "group_id": {}, "channel_id": {},
		"credential_id": {}, "upstream_model": {}, "breakdown_page": {},
		"breakdown_page_size": {}, "breakdown_sort": {}, "breakdown_sort_direction": {},
	}
	for key, value := range values {
		if _, ok := allowed[key]; !ok || len(value) != 1 {
			return requestlog.UsageQuery{}, app_errors.ErrBadRequest
		}
	}
	if err := validateSafeMilliseconds(observedAtMS); err != nil {
		return requestlog.UsageQuery{}, app_errors.ErrInternalServer
	}
	query := requestlog.UsageQuery{}
	rangeValue := "24h"
	if value, ok := agentQueryValue(values, "range"); ok {
		rangeValue = value
	}
	fromValue, hasFrom := agentQueryValue(values, "from_ms")
	toValue, hasTo := agentQueryValue(values, "to_ms")
	if hasFrom != hasTo {
		return requestlog.UsageQuery{}, app_errors.ErrValidation
	}
	if hasFrom {
		if _, hasRange := agentQueryValue(values, "range"); hasRange {
			return requestlog.UsageQuery{}, app_errors.ErrValidation
		}
		fromMS, err := parseCanonicalSafeMilliseconds(fromValue)
		if err != nil {
			return requestlog.UsageQuery{}, app_errors.ErrBadRequest
		}
		toMS, err := parseCanonicalSafeMilliseconds(toValue)
		if err != nil {
			return requestlog.UsageQuery{}, app_errors.ErrBadRequest
		}
		if fromMS >= toMS || toMS-fromMS > agentUsageCustomMaxMS {
			return requestlog.UsageQuery{}, app_errors.ErrValidation
		}
		bucketWidthMS := epochms.MillisecondsPerDay
		if toMS-fromMS <= epochms.MillisecondsPerDay {
			query.Granularity = requestlog.UsageGranularityHour
			bucketWidthMS = epochms.MillisecondsPerHour
		} else {
			query.Granularity = requestlog.UsageGranularityDay
		}
		if fromMS%bucketWidthMS != 0 || toMS%bucketWidthMS != 0 {
			return requestlog.UsageQuery{}, app_errors.ErrValidation
		}
		query.FromMS = fromMS
		query.ToMS = toMS
		query.BucketWidthMS = bucketWidthMS
	} else {
		preset, ok := agentUsageRangePreset(rangeValue)
		if !ok {
			return requestlog.UsageQuery{}, app_errors.ErrValidation
		}
		fromMS, toMS, err := epochms.WindowEndingAt(observedAtMS, preset.bucketWidthMS, preset.bucketCount)
		if err != nil {
			return requestlog.UsageQuery{}, app_errors.ErrInternalServer
		}
		query.FromMS = fromMS
		query.ToMS = toMS
		query.Granularity = preset.granularity
		query.BucketWidthMS = preset.bucketWidthMS
	}
	if value, ok := agentQueryValue(values, "group_id"); ok {
		parsed, apiErr := parseAgentScopedID(value)
		if apiErr != nil {
			return requestlog.UsageQuery{}, apiErr
		}
		query.GroupID = &parsed
	}
	if value, ok := agentQueryValue(values, "channel_id"); ok {
		channelID := channel.ID(value)
		if _, known := agentChannelRegistry.Get(channelID); !known {
			return requestlog.UsageQuery{}, app_errors.ErrValidation
		}
		query.ChannelID = channelID
	}
	if value, ok := agentQueryValue(values, "credential_id"); ok {
		parsed, apiErr := parseAgentScopedID(value)
		if apiErr != nil {
			return requestlog.UsageQuery{}, apiErr
		}
		query.CredentialID = &parsed
	}
	if value, ok := agentQueryValue(values, "upstream_model"); ok {
		if !validAgentModel(value) {
			return requestlog.UsageQuery{}, app_errors.ErrValidation
		}
		query.UpstreamModel = value
	}
	if value, ok := agentQueryValue(values, "breakdown_page"); ok {
		parsed, err := parseCanonicalSafeUint(value)
		if err != nil || parsed == 0 || parsed > uint64(maxSafeInteger) {
			return requestlog.UsageQuery{}, app_errors.ErrBadRequest
		}
		query.BreakdownPage = int(parsed)
	}
	if value, ok := agentQueryValue(values, "breakdown_page_size"); ok {
		switch value {
		case "20":
			query.BreakdownPageSize = 20
		case "50":
			query.BreakdownPageSize = 50
		case "100":
			query.BreakdownPageSize = 100
		default:
			return requestlog.UsageQuery{}, app_errors.ErrBadRequest
		}
	}
	if value, ok := agentQueryValue(values, "breakdown_sort"); ok {
		sortValue := requestlog.UsageBreakdownSort(value)
		if !agentUsageBreakdownSort(sortValue) {
			return requestlog.UsageQuery{}, app_errors.ErrBadRequest
		}
		query.BreakdownSort = sortValue
	}
	if value, ok := agentQueryValue(values, "breakdown_sort_direction"); ok {
		direction := requestlog.UsageBreakdownSortDirection(value)
		if direction != requestlog.UsageBreakdownSortAscending &&
			direction != requestlog.UsageBreakdownSortDescending {
			return requestlog.UsageQuery{}, app_errors.ErrBadRequest
		}
		query.BreakdownSortDirection = direction
	}
	return query, nil
}

func agentUsageBreakdownSort(sortValue requestlog.UsageBreakdownSort) bool {
	switch sortValue {
	case requestlog.UsageBreakdownSortModel,
		requestlog.UsageBreakdownSortGroup,
		requestlog.UsageBreakdownSortChannel,
		requestlog.UsageBreakdownSortRequestCount,
		requestlog.UsageBreakdownSortSuccessCount,
		requestlog.UsageBreakdownSortFailureCount,
		requestlog.UsageBreakdownSortSuccessRate,
		requestlog.UsageBreakdownSortAverageLatency,
		requestlog.UsageBreakdownSortUncachedInputTokens,
		requestlog.UsageBreakdownSortCacheReadTokens,
		requestlog.UsageBreakdownSortCacheWrite5MTokens,
		requestlog.UsageBreakdownSortCacheWrite1HTokens,
		requestlog.UsageBreakdownSortCacheWriteUnknown,
		requestlog.UsageBreakdownSortOutputTokens,
		requestlog.UsageBreakdownSortTotalTokens,
		requestlog.UsageBreakdownSortEstimatedCost:
		return true
	default:
		return false
	}
}

func mapAgentUsage(
	observedAtMS int64,
	query requestlog.UsageQuery,
	report requestlog.UsageReport,
) (UsageView, error) {
	summary, err := mapAgentUsageAggregate(report.Summary)
	if err != nil {
		return UsageView{}, err
	}
	view := UsageView{
		SchemaVersion: SchemaVersion,
		ObservedAtMS:  observedAtMS,
		FromMS:        query.FromMS,
		ToMS:          query.ToMS,
		Granularity:   string(query.Granularity),
		BucketWidthMS: query.BucketWidthMS,
		Summary:       summary,
		Series:        make([]UsageSeriesPointView, 0, len(report.Series)),
		Breakdown: UsageBreakdownView{
			Scope: report.Breakdown.Scope,
			Rows:  make([]UsageBreakdownRowView, 0, len(report.Breakdown.Rows)),
			Pagination: UsagePaginationView{
				Page:       report.Breakdown.Pagination.Page,
				PageSize:   report.Breakdown.Pagination.PageSize,
				TotalItems: report.Breakdown.Pagination.TotalItems,
				TotalPages: report.Breakdown.Pagination.TotalPages,
			},
		},
	}
	for _, point := range report.Series {
		if err := validateSafeMilliseconds(point.BucketStartMS); err != nil {
			return UsageView{}, err
		}
		if err := validateSafeMilliseconds(point.BucketEndMS); err != nil {
			return UsageView{}, err
		}
		aggregate, err := mapAgentUsageAggregate(point.UsageAggregate)
		if err != nil {
			return UsageView{}, err
		}
		view.Series = append(view.Series, UsageSeriesPointView{
			BucketStartMS: point.BucketStartMS,
			BucketEndMS:   point.BucketEndMS,
			Aggregate:     aggregate,
		})
	}
	total, err := mapAgentUsageAggregate(report.Breakdown.Total)
	if err != nil {
		return UsageView{}, err
	}
	if total != summary {
		return UsageView{}, errAgentUsageTotalMismatch
	}
	view.Breakdown.Total = total
	for _, row := range report.Breakdown.Rows {
		if !validAgentModel(row.Model) {
			return UsageView{}, errAgentUsageTotalMismatch
		}
		aggregate, err := mapAgentUsageAggregate(row.UsageAggregate)
		if err != nil {
			return UsageView{}, err
		}
		view.Breakdown.Rows = append(view.Breakdown.Rows, UsageBreakdownRowView{
			Model:     row.Model,
			GroupID:   row.GroupID,
			Aggregate: aggregate,
		})
	}
	return view, nil
}

func mapAgentUsageAggregate(source requestlog.UsageAggregate) (UsageAggregateView, error) {
	values := []int64{
		source.RequestCount, source.SuccessCount, source.FailureCount,
		source.UncachedInputTokens, source.CacheReadTokens, source.CacheWrite5MTokens,
		source.CacheWrite1HTokens, source.CacheWriteUnknownTokens, source.OutputTokens,
		source.UsageMissingCount, source.PartialCount, source.UnpricedRequestCount,
		source.PricingPartialCount,
	}
	for _, value := range values {
		if value < 0 || value > maxSafeInteger {
			return UsageAggregateView{}, errAgentUsageTotalMismatch
		}
	}
	totalTokens, ok := checkedAgentTokenTotal(
		source.UncachedInputTokens,
		source.CacheReadTokens,
		source.CacheWrite5MTokens,
		source.CacheWrite1HTokens,
		source.CacheWriteUnknownTokens,
		source.OutputTokens,
	)
	if !ok {
		return UsageAggregateView{}, errAgentUsageTotalMismatch
	}
	if source.EstimatedCostNanoUSD < 0 {
		return UsageAggregateView{}, errAgentUsageTotalMismatch
	}
	return UsageAggregateView{
		RequestCount:            source.RequestCount,
		SuccessCount:            source.SuccessCount,
		FailureCount:            source.FailureCount,
		UncachedInputTokens:     source.UncachedInputTokens,
		CacheReadTokens:         source.CacheReadTokens,
		CacheWrite5MTokens:      source.CacheWrite5MTokens,
		CacheWrite1HTokens:      source.CacheWrite1HTokens,
		CacheWriteUnknownTokens: source.CacheWriteUnknownTokens,
		OutputTokens:            source.OutputTokens,
		TotalTokens:             totalTokens,
		EstimatedNanoUSD:        strconv.FormatInt(source.EstimatedCostNanoUSD, 10),
		UsageMissingCount:       source.UsageMissingCount,
		PartialCount:            source.PartialCount,
		UnpricedRequestCount:    source.UnpricedRequestCount,
		PricingPartialCount:     source.PricingPartialCount,
	}, nil
}

func checkedAgentTokenTotal(tokens ...int64) (int64, bool) {
	total := int64(0)
	for _, value := range tokens {
		if value < 0 || value > maxSafeInteger-total {
			return 0, false
		}
		total += value
	}
	return total, true
}
