package control

import (
	"errors"
	"fmt"
	"net/url"
	"strconv"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/pricing"
	"gpt-load/internal/protocol"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

// parseRequestLogQuery 把查询串解析为存储层查询条件。
// 校验顺序即错误优先级：时间范围在前，分页参数在后；任一阶段失败立即返回。
func parseRequestLogQuery(rawQuery string) (requestlog.ListQuery, *app_errors.APIError) {
	values, apiErr := parseCollectionQueryValues(
		rawQuery,
		false,
		"from_ms", "to_ms", "group_id", "channel_id", "credential_id",
		"client_model", "upstream_model", "model_consistency", "access_key_id", "affinity_key",
		"status", "request_id", "protocol", "operation", "stream", "final_status_code",
		"usage_state", "cost_state", "pricing_completeness", "cache_present",
		"attempt_status_code", "failure_category", "error_code",
		"retry_state", "retry_count_min", "retry_count_max",
		"first_response_min_ms", "first_response_max_ms",
		"duration_min_ms", "duration_max_ms",
		"input_tokens_min", "input_tokens_max",
		"output_tokens_min", "output_tokens_max",
		"cost_min_nano_usd", "cost_max_nano_usd",
		"limit", "cursor",
	)
	if apiErr != nil {
		return requestlog.ListQuery{}, apiErr
	}

	query := requestlog.ListQuery{Limit: defaultRequestLogLimit}
	for _, parse := range []func(url.Values, *requestlog.ListQuery) *app_errors.APIError{
		parseRequestLogTimeRange,
		parseRequestLogIdentityFilters,
		parseRequestLogModelFilters,
		parseRequestLogRoutingFilters,
		parseRequestLogStateFilters,
		parseRequestLogAttemptFilters,
		parseRequestLogRangeFilters,
		parseRequestLogPageQuery,
	} {
		if apiErr := parse(values, &query); apiErr != nil {
			return requestlog.ListQuery{}, apiErr
		}
	}
	return query, nil
}

// parseRequestLogTimeRange 解析 from_ms/to_ms，并校验区间顺序。
func parseRequestLogTimeRange(values url.Values, query *requestlog.ListQuery) *app_errors.APIError {
	if value, ok := singleQueryValue(values, "from_ms"); ok {
		parsed, err := parseCanonicalSafeMilliseconds(value)
		if err != nil {
			return app_errors.ErrBadRequest
		}
		query.FromMS = &parsed
	}
	if value, ok := singleQueryValue(values, "to_ms"); ok {
		parsed, err := parseCanonicalSafeMilliseconds(value)
		if err != nil {
			return app_errors.ErrBadRequest
		}
		query.ToMS = &parsed
	}
	if query.FromMS != nil && query.ToMS != nil && *query.FromMS >= *query.ToMS {
		return app_errors.ErrValidation
	}
	return nil
}

// parseRequestLogIdentityFilters 解析分组、渠道与凭据身份过滤。
func parseRequestLogIdentityFilters(values url.Values, query *requestlog.ListQuery) *app_errors.APIError {
	if value, ok := singleQueryValue(values, "group_id"); ok {
		parsed, apiErr := parseRequestLogID(value)
		if apiErr != nil {
			return apiErr
		}
		query.GroupID = &parsed
	}
	if value, ok := singleQueryValue(values, "channel_id"); ok {
		parsed, apiErr := parseRequestLogChannelID(value)
		if apiErr != nil {
			return apiErr
		}
		query.ChannelID = parsed
	}
	if value, ok := singleQueryValue(values, "credential_id"); ok {
		parsed, apiErr := parseRequestLogID(value)
		if apiErr != nil {
			return apiErr
		}
		query.CredentialID = &parsed
	}
	return nil
}

// parseRequestLogModelFilters 解析客户端/上游模型与模型一致性过滤。
func parseRequestLogModelFilters(values url.Values, query *requestlog.ListQuery) *app_errors.APIError {
	if value, ok := singleQueryValue(values, "client_model"); ok {
		if !validUsageModel(value) {
			return app_errors.ErrValidation
		}
		query.ClientModel = value
	}
	if value, ok := singleQueryValue(values, "upstream_model"); ok {
		if !validUsageModel(value) {
			return app_errors.ErrValidation
		}
		query.UpstreamModel = value
	}
	if value, ok := singleQueryValue(values, "model_consistency"); ok {
		consistency := telemetry.ModelConsistency(value)
		switch consistency {
		case telemetry.ModelConsistencyNotApplicable,
			telemetry.ModelConsistencyMatch,
			telemetry.ModelConsistencyUnknown,
			telemetry.ModelConsistencyMismatch:
			query.ModelConsistency = consistency
		default:
			return app_errors.ErrValidation
		}
	}
	return nil
}

// parseRequestLogRoutingFilters 解析访问密钥、亲和键、协议、操作、流式与最终状态码。
func parseRequestLogRoutingFilters(values url.Values, query *requestlog.ListQuery) *app_errors.APIError {
	if value, ok := singleQueryValue(values, "access_key_id"); ok {
		parsed, apiErr := parseRequestLogID(value)
		if apiErr != nil {
			return apiErr
		}
		query.AccessKeyID = &parsed
	}
	if value, ok := singleQueryValue(values, "affinity_key"); ok {
		if !requestlog.ValidAffinityKey(value) {
			return app_errors.ErrBadRequest
		}
		query.AffinityKey = value
	}
	if value, ok := singleQueryValue(values, "protocol"); ok {
		parsed := protocol.Protocol(value)
		if !parsed.Valid() {
			return app_errors.ErrValidation
		}
		query.Protocol = parsed
	}
	if value, ok := singleQueryValue(values, "operation"); ok {
		operation := execution.Operation(value)
		if !operation.Valid() {
			return app_errors.ErrValidation
		}
		query.Operation = operation
	}
	if value, ok := singleQueryValue(values, "stream"); ok {
		parsed, valid := parseRequestLogBool(value)
		if !valid {
			return app_errors.ErrValidation
		}
		query.Stream = &parsed
	}
	if value, ok := singleQueryValue(values, "final_status_code"); ok {
		parsed, apiErr := parseRequestLogStatusCode(value)
		if apiErr != nil {
			return apiErr
		}
		query.FinalStatusCode = &parsed
	}
	return nil
}

// parseRequestLogStateFilters 解析请求状态、用量与计费状态、定价完整性及缓存过滤。
func parseRequestLogStateFilters(values url.Values, query *requestlog.ListQuery) *app_errors.APIError {
	if value, ok := singleQueryValue(values, "status"); ok {
		status := telemetry.RequestStatus(value)
		switch status {
		case telemetry.RequestStatusSuccess,
			telemetry.RequestStatusError,
			telemetry.RequestStatusIncomplete,
			telemetry.RequestStatusCanceled:
			query.Status = status
		default:
			return app_errors.ErrValidation
		}
	}
	if value, ok := singleQueryValue(values, "usage_state"); ok {
		state := usage.State(value)
		switch state {
		case usage.StateComplete, usage.StatePartial, usage.StateMissing, usage.StateNotApplicable:
			query.UsageState = state
		default:
			return app_errors.ErrValidation
		}
	}
	if value, ok := singleQueryValue(values, "cost_state"); ok {
		state := pricing.CostState(value)
		switch state {
		case pricing.CostStatePriced, pricing.CostStateUnpriced, pricing.CostStateNotApplicable:
			query.CostState = state
		default:
			return app_errors.ErrValidation
		}
	}
	if value, ok := singleQueryValue(values, "pricing_completeness"); ok {
		completeness := pricing.Completeness(value)
		switch completeness {
		case pricing.CompletenessComplete,
			pricing.CompletenessPartial,
			pricing.CompletenessUnavailable,
			pricing.CompletenessNotApplicable:
			query.PricingCompleteness = completeness
		default:
			return app_errors.ErrValidation
		}
	}
	if value, ok := singleQueryValue(values, "cache_present"); ok {
		parsed, valid := parseRequestLogBool(value)
		if !valid {
			return app_errors.ErrValidation
		}
		query.CachePresent = &parsed
	}
	return nil
}

// parseRequestLogAttemptFilters 解析尝试状态码、失败分类、错误码与重试状态。
func parseRequestLogAttemptFilters(values url.Values, query *requestlog.ListQuery) *app_errors.APIError {
	if value, ok := singleQueryValue(values, "attempt_status_code"); ok {
		parsed, apiErr := parseRequestLogStatusCode(value)
		if apiErr != nil {
			return apiErr
		}
		query.AttemptStatusCode = &parsed
	}
	if value, ok := singleQueryValue(values, "failure_category"); ok {
		category := telemetry.FailureCategory(value)
		switch category {
		case telemetry.FailureCategoryOK,
			telemetry.FailureCategoryRateLimited,
			telemetry.FailureCategoryModelUnavailable,
			telemetry.FailureCategoryInvalidKey,
			telemetry.FailureCategoryUpstreamHost,
			telemetry.FailureCategoryClientError,
			telemetry.FailureCategoryConversionUnsupported,
			telemetry.FailureCategoryDownstreamCancel,
			telemetry.FailureCategoryAuthenticationRequired,
			telemetry.FailureCategoryBilling,
			telemetry.FailureCategoryAmbiguous:
			query.FailureCategory = category
		default:
			return app_errors.ErrValidation
		}
	}
	if value, ok := singleQueryValue(values, "error_code"); ok {
		if !validUsageModel(value) {
			return app_errors.ErrValidation
		}
		query.AttemptErrorCode = value
	}
	if value, ok := singleQueryValue(values, "retry_state"); ok {
		state := requestlog.RetryState(value)
		switch state {
		case requestlog.RetryStateRetried, requestlog.RetryStateNotRetried:
			query.RetryState = state
		default:
			return app_errors.ErrValidation
		}
	}
	return nil
}

// parseRequestLogRangeFilters 解析重试次数与各项数值区间过滤。
func parseRequestLogRangeFilters(values url.Values, query *requestlog.ListQuery) *app_errors.APIError {
	if apiErr := parseRequestLogIntRange(
		values,
		"retry_count_min",
		"retry_count_max",
		&query.RetryCountMin,
		&query.RetryCountMax,
	); apiErr != nil {
		return apiErr
	}
	for _, target := range []struct {
		minimumKey string
		maximumKey string
		minimum    **int64
		maximum    **int64
	}{
		{"first_response_min_ms", "first_response_max_ms", &query.FirstResponseMinMS, &query.FirstResponseMaxMS},
		{"duration_min_ms", "duration_max_ms", &query.DurationMinMS, &query.DurationMaxMS},
		{"input_tokens_min", "input_tokens_max", &query.InputTokensMin, &query.InputTokensMax},
		{"output_tokens_min", "output_tokens_max", &query.OutputTokensMin, &query.OutputTokensMax},
		{"cost_min_nano_usd", "cost_max_nano_usd", &query.CostMinNanoUSD, &query.CostMaxNanoUSD},
	} {
		if apiErr := parseRequestLogInt64Range(
			values,
			target.minimumKey,
			target.maximumKey,
			target.minimum,
			target.maximum,
		); apiErr != nil {
			return apiErr
		}
	}
	return nil
}

// parseRequestLogPageQuery 解析 request_id、limit 与游标；分页参数最后校验。
func parseRequestLogPageQuery(values url.Values, query *requestlog.ListQuery) *app_errors.APIError {
	if value, ok := singleQueryValue(values, "request_id"); ok {
		if !canonicalLowercaseUUIDv4.MatchString(value) {
			return app_errors.ErrBadRequest
		}
		query.RequestID = value
	}
	if value, ok := singleQueryValue(values, "limit"); ok {
		parsed, err := parseCanonicalSafeUint(value)
		if err != nil {
			if errors.Is(err, errUnsafeCanonicalUint) {
				return app_errors.ErrValidation
			}
			return app_errors.ErrBadRequest
		}
		if parsed < 1 || parsed > maxRequestLogLimit {
			return app_errors.ErrValidation
		}
		query.Limit = int(parsed)
	}
	if value, ok := singleQueryValue(values, "cursor"); ok {
		cursor, err := decodeRequestLogCursor(value)
		if err != nil {
			return app_errors.ErrBadRequest
		}
		query.Cursor = cursor
	}
	return nil
}

func singleQueryValue(values url.Values, key string) (string, bool) {
	value, ok := values[key]
	if !ok {
		return "", false
	}
	return value[0], true
}

func parseRequestLogID(value string) (uint, *app_errors.APIError) {
	parsed, err := parseCanonicalSafePlatformUint(value)
	if err != nil {
		if errors.Is(err, errUnsafeCanonicalUint) {
			return 0, app_errors.ErrValidation
		}
		return 0, app_errors.ErrBadRequest
	}
	if parsed == 0 {
		return 0, app_errors.ErrValidation
	}
	return parsed, nil
}

func parseRequestLogChannelID(value string) (channel.ID, *app_errors.APIError) {
	channelID := channel.ID(value)
	if _, ok := requestLogChannelRegistry.Get(channelID); !ok {
		return "", app_errors.ErrValidation
	}
	return channelID, nil
}

func parseRequestLogBool(value string) (bool, bool) {
	switch value {
	case "true":
		return true, true
	case "false":
		return false, true
	default:
		return false, false
	}
}

func parseRequestLogStatusCode(value string) (int, *app_errors.APIError) {
	parsed, err := parseCanonicalNonNegativeInt64(value)
	if err != nil {
		return 0, app_errors.ErrBadRequest
	}
	if parsed > 999 {
		return 0, app_errors.ErrValidation
	}
	return int(parsed), nil
}

func parseRequestLogIntRange(
	values url.Values,
	minimumKey string,
	maximumKey string,
	minimum **int,
	maximum **int,
) *app_errors.APIError {
	var parsedMinimum *int64
	var parsedMaximum *int64
	if apiErr := parseRequestLogInt64Range(
		values,
		minimumKey,
		maximumKey,
		&parsedMinimum,
		&parsedMaximum,
	); apiErr != nil {
		return apiErr
	}
	if parsedMinimum != nil {
		value := int(*parsedMinimum)
		if int64(value) != *parsedMinimum {
			return app_errors.ErrValidation
		}
		*minimum = &value
	}
	if parsedMaximum != nil {
		value := int(*parsedMaximum)
		if int64(value) != *parsedMaximum {
			return app_errors.ErrValidation
		}
		*maximum = &value
	}
	return nil
}

func parseRequestLogInt64Range(
	values url.Values,
	minimumKey string,
	maximumKey string,
	minimum **int64,
	maximum **int64,
) *app_errors.APIError {
	if value, ok := singleQueryValue(values, minimumKey); ok {
		parsed, err := parseCanonicalNonNegativeInt64(value)
		if err != nil {
			return app_errors.ErrBadRequest
		}
		*minimum = &parsed
	}
	if value, ok := singleQueryValue(values, maximumKey); ok {
		parsed, err := parseCanonicalNonNegativeInt64(value)
		if err != nil {
			return app_errors.ErrBadRequest
		}
		*maximum = &parsed
	}
	if *minimum != nil && *maximum != nil && **minimum > **maximum {
		return app_errors.ErrValidation
	}
	return nil
}

func parseCanonicalNonNegativeInt64(value string) (int64, error) {
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil || parsed < 0 || strconv.FormatInt(parsed, 10) != value {
		return 0, fmt.Errorf("value must be a canonical non-negative integer")
	}
	return parsed, nil
}
