package requestlog

import (
	"context"
	"fmt"
	"strings"

	"gorm.io/gorm"

	"gpt-load/internal/storage/models"
)

const defaultListLimit = 50

func (service *Service) List(ctx context.Context, input ListQuery) (Page, error) {
	limit := input.Limit
	if limit <= 0 {
		limit = defaultListLimit
	}

	query := service.db.WithContext(ctx).
		Model(&models.RequestLog{}).
		Order("completed_at_ms DESC").
		Order("id DESC").
		Limit(limit + 1)
	query = applyListIdentityFilters(query, input)
	query = applyListMetricFilters(query, input)

	var rows []models.RequestLog
	if err := query.Find(&rows).Error; err != nil {
		return Page{}, fmt.Errorf("query request logs: %w", err)
	}

	hasNext := len(rows) > limit
	if hasNext {
		rows = rows[:limit]
	}
	records, err := decodeRequestLogRows(rows)
	if err != nil {
		return Page{}, err
	}
	if err := service.loadAccessKeyRefs(ctx, records); err != nil {
		return Page{}, err
	}
	if err := service.loadFinalExecutionObservations(ctx, records); err != nil {
		return Page{}, err
	}

	page := Page{Items: records}
	if hasNext {
		last := records[len(records)-1]
		page.NextCursor = &Cursor{
			CompletedAtMS: last.CompletedAtMS,
			RequestID:     last.RequestID,
		}
	}
	return page, nil
}

func applyListIdentityFilters(query *gorm.DB, input ListQuery) *gorm.DB {
	if input.FromMS != nil {
		query = query.Where("completed_at_ms >= ?", *input.FromMS)
	}
	if input.ToMS != nil {
		query = query.Where("completed_at_ms < ?", *input.ToMS)
	}
	if input.ClientModel != "" {
		query = query.Where("client_model = ?", input.ClientModel)
	}
	if input.ModelConsistency != "" {
		query = query.Where("model_consistency = ?", input.ModelConsistency)
	}
	if input.AccessKeyID != nil {
		query = query.Where("access_key_id = ?", *input.AccessKeyID)
	}
	if input.AffinityKey != "" {
		query = query.Where("affinity_key = ?", input.AffinityKey)
	}
	if input.Status != "" {
		query = query.Where("status = ?", input.Status)
	}
	if input.RequestID != "" {
		query = query.Where("id = ?", input.RequestID)
	}
	if input.Protocol != "" {
		query = query.Where("protocol = ?", input.Protocol)
	}
	if input.Operation != "" {
		query = query.Where("operation = ?", input.Operation)
	}
	if input.Stream != nil {
		query = query.Where("stream = ?", *input.Stream)
	}
	if input.FinalStatusCode != nil {
		query = query.Where("status_code = ?", *input.FinalStatusCode)
	}
	if input.UsageState != "" {
		query = query.Where("usage_state = ?", input.UsageState)
	}
	return query
}

func applyListMetricFilters(query *gorm.DB, input ListQuery) *gorm.DB {
	if input.CostState != "" {
		query = query.Where("cost_state = ?", input.CostState)
	}
	if input.PricingCompleteness != "" {
		query = query.Where("pricing_completeness = ?", input.PricingCompleteness)
	}
	if input.CachePresent != nil {
		expression := `(cache_read_tokens + cache_write_5m_tokens + cache_write_1h_tokens + cache_write_unknown_tokens) > 0`
		if !*input.CachePresent {
			expression = `(cache_read_tokens + cache_write_5m_tokens + cache_write_1h_tokens + cache_write_unknown_tokens) = 0`
		}
		query = query.Where(expression)
	}
	if input.RetryState == RetryStateRetried {
		query = query.Where("attempt_count > 1")
	} else if input.RetryState == RetryStateNotRetried {
		query = query.Where("attempt_count <= 1")
	}
	retryCountExpression := "CASE WHEN attempt_count > 1 THEN attempt_count - 1 ELSE 0 END"
	if input.RetryCountMin != nil {
		query = query.Where(retryCountExpression+" >= ?", *input.RetryCountMin)
	}
	if input.RetryCountMax != nil {
		query = query.Where(retryCountExpression+" <= ?", *input.RetryCountMax)
	}
	query = applyNullableRange(query, "first_response_ms", input.FirstResponseMinMS, input.FirstResponseMaxMS)
	query = applyNullableRange(query, "duration_ms", input.DurationMinMS, input.DurationMaxMS)
	query = applyNullableRange(
		query,
		"(uncached_input_tokens + cache_read_tokens + cache_write_5m_tokens + cache_write_1h_tokens + cache_write_unknown_tokens)",
		input.InputTokensMin,
		input.InputTokensMax,
	)
	query = applyNullableRange(query, "output_tokens", input.OutputTokensMin, input.OutputTokensMax)
	query = applyNullableRange(query, "estimated_cost_nano_usd", input.CostMinNanoUSD, input.CostMaxNanoUSD)
	query = applyAttemptFilters(query, input)
	if input.Cursor != nil {
		query = query.Where(
			"completed_at_ms < ? OR (completed_at_ms = ? AND id < ?)",
			input.Cursor.CompletedAtMS,
			input.Cursor.CompletedAtMS,
			input.Cursor.RequestID,
		)
	}
	return query
}

func applyNullableRange[T int | int64](
	query *gorm.DB,
	expression string,
	minimum *T,
	maximum *T,
) *gorm.DB {
	if minimum != nil {
		query = query.Where(expression+" >= ?", *minimum)
	}
	if maximum != nil {
		query = query.Where(expression+" <= ?", *maximum)
	}
	return query
}

func applyAttemptFilters(query *gorm.DB, input ListQuery) *gorm.DB {
	conditions := make([]string, 0, 8)
	arguments := make([]any, 0, 8)
	if input.GroupID != nil {
		conditions = append(conditions, "attempt.group_id = ?")
		arguments = append(arguments, *input.GroupID)
	}
	if input.ChannelID != "" {
		conditions = append(conditions, "attempt.channel_id = ?")
		arguments = append(arguments, input.ChannelID)
	}
	if input.CredentialID != nil {
		conditions = append(conditions, "attempt.credential_id = ?")
		arguments = append(arguments, *input.CredentialID)
	}
	if input.UpstreamModel != "" {
		conditions = append(conditions, "attempt.upstream_model = ?")
		arguments = append(arguments, input.UpstreamModel)
	}
	if input.AttemptStatusCode != nil {
		conditions = append(conditions, "attempt.status_code = ?")
		arguments = append(arguments, *input.AttemptStatusCode)
	}
	if input.FailureCategory != "" {
		conditions = append(conditions, "attempt.failure_category = ?")
		arguments = append(arguments, input.FailureCategory)
	}
	if input.AttemptErrorCode != "" {
		conditions = append(conditions, "attempt.error_code = ?")
		arguments = append(arguments, input.AttemptErrorCode)
	}
	if len(conditions) == 0 {
		return query
	}
	statement := `EXISTS (
		SELECT 1 FROM request_log_attempts AS attempt
		WHERE attempt.request_id = request_logs.id AND ` +
		strings.Join(conditions, " AND ") + `
	)`
	return query.Where(statement, arguments...)
}

// Get returns one request with all retry attempts and its frozen pricing receipt.
