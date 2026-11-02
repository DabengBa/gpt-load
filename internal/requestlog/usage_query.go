package requestlog

import (
	"context"
	"fmt"

	"time"

	"gorm.io/gorm"

	"gpt-load/internal/platform/epochms"
	"gpt-load/internal/storage/dbtx"
	"gpt-load/internal/storage/models"
)

const (
	usageDistributionLimit        = 5
	usageBreakdownDefaultPageSize = 20
	maxUsageSeriesHours           = 30 * 24
	usageRollbackTimeout          = time.Second
)

func (service *Service) QueryUsage(ctx context.Context, input UsageQuery) (UsageReport, error) {
	if service == nil || service.db == nil {
		return UsageReport{}, fmt.Errorf("query usage: database is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	input = normalizeUsageBreakdownQuery(input)
	bucketWidthMS, err := validateUsageQuery(input)
	if err != nil {
		return UsageReport{}, err
	}

	var report UsageReport
	err = dbtx.Run(ctx, service.db, dbtx.Options{
		Mode:           dbtx.ReadSnapshot,
		CleanupTimeout: usageRollbackTimeout,
		Operation:      "usage read transaction",
	}, func(connection *gorm.DB) error {
		if err := validateUsageStatIntegrity(usageStatScope(connection, input)); err != nil {
			return err
		}
		summary, err := queryUsageSummary(usageStatScope(connection, input))
		if err != nil {
			return err
		}
		series, err := queryUsageSeries(usageStatScope(connection, input), bucketWidthMS)
		if err != nil {
			return err
		}
		distributions, err := queryUsageDistributions(
			usageStatScope(connection, input), summary, input.AccessKeyID != nil,
		)
		if err != nil {
			return err
		}
		breakdown, err := queryUsageBreakdown(usageStatScope(connection, input), summary, input.AccessKeyID != nil, input)
		if err != nil {
			return err
		}
		report = UsageReport{
			Summary: summary, Series: series, Distributions: distributions, Breakdown: breakdown,
		}
		return nil
	})
	if err != nil {
		return UsageReport{}, fmt.Errorf("query usage: %w", err)
	}
	return report, nil
}

func validateUsageQuery(input UsageQuery) (int64, error) {
	input = normalizeUsageBreakdownQuery(input)
	if input.FromMS < 0 || input.ToMS <= input.FromMS {
		return 0, fmt.Errorf("query usage: invalid time range")
	}
	if input.ToMS-input.FromMS >
		int64(maxUsageSeriesHours)*epochms.MillisecondsPerHour {
		return 0, fmt.Errorf("query usage: time range exceeds %d hours", maxUsageSeriesHours)
	}
	if input.AccessKeyID != nil && *input.AccessKeyID == 0 {
		return 0, fmt.Errorf("query usage: invalid access key scope")
	}
	if input.ChannelID != "" && !validChannelID(string(input.ChannelID)) {
		return 0, fmt.Errorf("query usage: invalid channel scope")
	}
	if input.CredentialID != nil && *input.CredentialID == 0 {
		return 0, fmt.Errorf("query usage: invalid credential scope")
	}
	if input.BreakdownPage == 0 {
		input.BreakdownPage = 1
	}
	if input.BreakdownPageSize == 0 {
		input.BreakdownPageSize = usageBreakdownDefaultPageSize
	}
	if input.BreakdownPage < 1 ||
		(input.BreakdownPageSize != 20 && input.BreakdownPageSize != 50 && input.BreakdownPageSize != 100) {
		return 0, fmt.Errorf("query usage: invalid breakdown pagination")
	}
	if !validUsageBreakdownSort(input.BreakdownSort) {
		return 0, fmt.Errorf("query usage: invalid breakdown sort %q", input.BreakdownSort)
	}
	if input.BreakdownSortDirection != UsageBreakdownSortAscending &&
		input.BreakdownSortDirection != UsageBreakdownSortDescending {
		return 0, fmt.Errorf("query usage: invalid breakdown sort direction %q", input.BreakdownSortDirection)
	}
	if input.AccessKeyID != nil &&
		(input.BreakdownSort == UsageBreakdownSortGroup || input.BreakdownSort == UsageBreakdownSortChannel) {
		return 0, fmt.Errorf("query usage: access-key scope cannot sort by admin identity")
	}
	bucketWidthMS := input.BucketWidthMS
	switch input.Granularity {
	case UsageGranularityHour:
		if bucketWidthMS == 0 {
			bucketWidthMS = epochms.MillisecondsPerHour
		}
	case UsageGranularityDay:
		if bucketWidthMS == 0 {
			bucketWidthMS = epochms.MillisecondsPerDay
		}
	default:
		return 0, fmt.Errorf("query usage: unsupported granularity %q", input.Granularity)
	}
	if bucketWidthMS < epochms.MillisecondsPerHour ||
		bucketWidthMS > epochms.MillisecondsPerDay ||
		bucketWidthMS%epochms.MillisecondsPerHour != 0 {
		return 0, fmt.Errorf("query usage: invalid bucket width %d", bucketWidthMS)
	}
	if input.Granularity == UsageGranularityHour &&
		bucketWidthMS >= epochms.MillisecondsPerDay {
		return 0, fmt.Errorf("query usage: hourly granularity requires a sub-day bucket")
	}
	if input.Granularity == UsageGranularityDay &&
		bucketWidthMS != epochms.MillisecondsPerDay {
		return 0, fmt.Errorf("query usage: daily granularity requires a day bucket")
	}
	if input.FromMS%bucketWidthMS != 0 || input.ToMS%bucketWidthMS != 0 ||
		(input.ToMS-input.FromMS)%bucketWidthMS != 0 {
		return 0, fmt.Errorf("query usage: time range is not bucket aligned")
	}
	return bucketWidthMS, nil
}

func normalizeUsageBreakdownQuery(input UsageQuery) UsageQuery {
	if input.BreakdownPage == 0 {
		input.BreakdownPage = 1
	}
	if input.BreakdownPageSize == 0 {
		input.BreakdownPageSize = usageBreakdownDefaultPageSize
	}
	if input.BreakdownSort == "" {
		input.BreakdownSort = UsageBreakdownSortEstimatedCost
	}
	if input.BreakdownSortDirection == "" {
		input.BreakdownSortDirection = defaultUsageBreakdownSortDirection(input.BreakdownSort)
	}
	return input
}

func defaultUsageBreakdownSortDirection(sort UsageBreakdownSort) UsageBreakdownSortDirection {
	switch sort {
	case UsageBreakdownSortModel, UsageBreakdownSortGroup, UsageBreakdownSortChannel:
		return UsageBreakdownSortAscending
	default:
		return UsageBreakdownSortDescending
	}
}

func validUsageBreakdownSort(sort UsageBreakdownSort) bool {
	switch sort {
	case UsageBreakdownSortModel,
		UsageBreakdownSortGroup,
		UsageBreakdownSortChannel,
		UsageBreakdownSortRequestCount,
		UsageBreakdownSortSuccessCount,
		UsageBreakdownSortFailureCount,
		UsageBreakdownSortAttemptCount,
		UsageBreakdownSortAttemptFailureCount,
		UsageBreakdownSortNormalAttemptCount,
		UsageBreakdownSortSlowAttemptCount,
		UsageBreakdownSortFaultyAttemptCount,
		UsageBreakdownSortSuccessRate,
		UsageBreakdownSortAverageDuration,
		UsageBreakdownSortAverageFirstResponse,
		UsageBreakdownSortUncachedInputTokens,
		UsageBreakdownSortCacheReadTokens,
		UsageBreakdownSortCacheWrite5MTokens,
		UsageBreakdownSortCacheWrite1HTokens,
		UsageBreakdownSortCacheWriteUnknown,
		UsageBreakdownSortOutputTokens,
		UsageBreakdownSortTotalTokens,
		UsageBreakdownSortEstimatedCost,
		UsageBreakdownSortCacheRate,
		UsageBreakdownSortTokenEfficiency:
		return true
	default:
		return false
	}
}

func usageStatScope(db *gorm.DB, input UsageQuery) *gorm.DB {
	scope := db.Session(&gorm.Session{NewDB: true}).Model(&models.UsageStat{}).
		Where("bucket_start_ms >= ? AND bucket_start_ms < ?", input.FromMS, input.ToMS).
		// Older versions aggregated zero-attempt requests under the unbound
		// (group_id=0, model='') key. Keep those derived rows invisible so home
		// and monitor share the current contract.
		Where("NOT (group_id = ? AND model = ?)", 0, "")
	if input.GroupID != nil {
		scope = scope.Where("group_id = ?", *input.GroupID)
	}
	if input.ChannelID != "" {
		scope = scope.Where("channel_id = ?", input.ChannelID)
	}
	if input.CredentialID != nil {
		scope = scope.Where("credential_id = ?", *input.CredentialID)
	}
	if input.AccessKeyID != nil {
		scope = scope.Where("access_key_id = ?", *input.AccessKeyID)
	}
	if input.UpstreamModel != "" {
		scope = scope.Where("model = ?", input.UpstreamModel)
	}
	if input.ClientModel != "" {
		scope = scope.Where("client_model = ?", input.ClientModel)
	}
	return scope
}

func validateUsageStatIntegrity(scope *gorm.DB) error {
	var integrity usageStatIntegrity
	if err := scope.Select(`
		COALESCE(MAX(CASE
			WHEN bucket_start_ms < 0
				OR bucket_start_ms % 3600000 != 0
			THEN 1 ELSE 0 END), 0) AS invalid_bucket,
		COALESCE(MAX(CASE
			WHEN request_count < 0 OR success_count < 0 OR failure_count < 0
					OR usage_missing_count < 0 OR partial_count < 0 OR unpriced_request_count < 0
					OR pricing_partial_count < 0
				OR CASE
					WHEN request_count < success_count THEN 1
					WHEN request_count - success_count != failure_count THEN 1
					ELSE 0
				END = 1
			THEN 1 ELSE 0 END), 0) AS invalid_count,
		COALESCE(MAX(CASE
			WHEN uncached_input_tokens < 0 OR output_tokens < 0 OR cache_read_tokens < 0
					OR cache_write_5m_tokens < 0 OR cache_write_1h_tokens < 0
					OR cache_write_unknown_tokens < 0
			THEN 1 ELSE 0 END), 0) AS invalid_token,
		COALESCE(MAX(CASE
			WHEN duration_ms_total < 0 OR duration_sample_count < 0
				OR duration_sample_count > request_count
			THEN 1 ELSE 0 END), 0) AS invalid_duration,
		COALESCE(MAX(CASE
			WHEN first_response_ms_total < 0 OR first_response_sample_count < 0
				OR first_response_sample_count > request_count
			THEN 1 ELSE 0 END), 0) AS invalid_first_response,
		COALESCE(MAX(CASE WHEN success_output_tokens < 0 OR success_duration_ms_total < 0 OR success_efficiency_sample_count < 0 OR success_efficiency_sample_count > success_count THEN 1 ELSE 0 END), 0) AS invalid_success_efficiency,
		COALESCE(MAX(CASE
			WHEN estimated_cost_nano_usd < 0
			THEN 1 ELSE 0 END), 0) AS invalid_cost
	`).Find(&integrity).Error; err != nil {
		return fmt.Errorf("check usage stat integrity: %w", err)
	}
	if integrity.InvalidBucket != 0 || integrity.InvalidCount != 0 ||
		integrity.InvalidToken != 0 || integrity.InvalidCost != 0 || integrity.InvalidDuration != 0 ||
		integrity.InvalidFirstResponse != 0 || integrity.InvalidSuccessEfficiency != 0 {
		return fmt.Errorf("check usage stat integrity: corrupt row")
	}
	return nil
}

func queryUsageSummary(scope *gorm.DB) (UsageAggregate, error) {
	var summary UsageAggregate
	if err := scope.Select(usageAggregateSelect).Find(&summary).Error; err != nil {
		return UsageAggregate{}, fmt.Errorf("query usage summary: %w", err)
	}
	if err := validateUsageAggregate(summary); err != nil {
		return UsageAggregate{}, fmt.Errorf("validate usage summary: %w", err)
	}
	return summary, nil
}

func queryUsageSeries(scope *gorm.DB, bucketWidthMS int64) ([]UsageSeriesPoint, error) {
	if bucketWidthMS < epochms.MillisecondsPerHour ||
		bucketWidthMS > epochms.MillisecondsPerDay ||
		bucketWidthMS%epochms.MillisecondsPerHour != 0 {
		return nil, fmt.Errorf("query usage series: invalid bucket width %d", bucketWidthMS)
	}
	var source []usageHourPoint
	if err := scope.Select("bucket_start_ms, " + usageAggregateSelect).
		Group("bucket_start_ms").
		Order("bucket_start_ms ASC").
		Find(&source).Error; err != nil {
		return nil, fmt.Errorf("query usage source series: %w", err)
	}

	for _, point := range source {
		if err := validateUsageAggregate(point.UsageAggregate); err != nil {
			return nil, fmt.Errorf("validate usage source series: %w", err)
		}
	}
	if bucketWidthMS == epochms.MillisecondsPerHour {
		series := make([]UsageSeriesPoint, 0, len(source))
		for _, point := range source {
			series = append(series, UsageSeriesPoint{
				BucketStartMS:  point.BucketStartMS,
				BucketEndMS:    point.BucketStartMS + epochms.MillisecondsPerHour,
				UsageAggregate: point.UsageAggregate,
			})
		}
		return series, nil
	}
	return mergeUsageHours(source, bucketWidthMS)
}

// compareUsageBreakdownSamplePresence keeps rows without first-response
// samples after rows with samples in both sort directions, matching the
// frontend that renders a missing average first response as "—".

// Aliases follow GORM's snake-case mapping for embedded UsageAggregate fields (for example, 5M -> 5_m).
const usageAggregateSelect = "" +
	"COALESCE(SUM(request_count), 0) AS request_count, " +
	"COALESCE(SUM(success_count), 0) AS success_count, " +
	"COALESCE(SUM(failure_count), 0) AS failure_count, " +
	"COALESCE(SUM(uncached_input_tokens), 0) AS uncached_input_tokens, " +
	"COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens, " +
	"COALESCE(SUM(cache_write_5m_tokens), 0) AS cache_write5_m_tokens, " +
	"COALESCE(SUM(cache_write_1h_tokens), 0) AS cache_write1_h_tokens, " +
	"COALESCE(SUM(cache_write_unknown_tokens), 0) AS cache_write_unknown_tokens, " +
	"COALESCE(SUM(output_tokens), 0) AS output_tokens, " +
	"COALESCE(SUM(success_output_tokens), 0) AS success_output_tokens, " +
	"COALESCE(SUM(success_duration_ms_total), 0) AS success_duration_ms_total, " +
	"COALESCE(SUM(success_efficiency_sample_count), 0) AS success_efficiency_sample_count, " +
	"COALESCE(SUM(estimated_cost_nano_usd), 0) AS estimated_cost_nano_usd, " +
	"COALESCE(SUM(duration_ms_total), 0) AS duration_ms_total, " +
	"COALESCE(SUM(duration_sample_count), 0) AS duration_sample_count, " +
	"COALESCE(SUM(first_response_ms_total), 0) AS first_response_ms_total, " +
	"COALESCE(SUM(first_response_sample_count), 0) AS first_response_sample_count, " +
	"COALESCE(SUM(usage_missing_count), 0) AS usage_missing_count, " +
	"COALESCE(SUM(partial_count), 0) AS partial_count, " +
	"COALESCE(SUM(unpriced_request_count), 0) AS unpriced_request_count, " +
	"COALESCE(SUM(pricing_partial_count), 0) AS pricing_partial_count"

const usageDistributionTotalTokensExpression = "" +
	"COALESCE(SUM(uncached_input_tokens), 0) + " +
	"COALESCE(SUM(cache_read_tokens), 0) + " +
	"COALESCE(SUM(cache_write_5m_tokens), 0) + " +
	"COALESCE(SUM(cache_write_1h_tokens), 0) + " +
	"COALESCE(SUM(cache_write_unknown_tokens), 0) + " +
	"COALESCE(SUM(output_tokens), 0)"

const usageDistributionAggregateSelect = "" +
	"COALESCE(SUM(request_count), 0) AS request_count, " +
	usageDistributionTotalTokensExpression + " AS total_tokens, " +
	"COALESCE(SUM(estimated_cost_nano_usd), 0) AS estimated_cost_nano_usd"

type usageBreakdownRow struct {
	Model       string
	ClientModel string
	GroupID     *uint
	ChannelID   *string
	UsageAggregate
}

type usageHourPoint struct {
	BucketStartMS int64
	UsageAggregate
}

type usageStatIntegrity struct {
	InvalidBucket            int64
	InvalidCount             int64
	InvalidToken             int64
	InvalidCost              int64
	InvalidDuration          int64
	InvalidFirstResponse     int64
	InvalidSuccessEfficiency int64
}

type usageDistributionRow struct {
	GroupID     uint
	AccessKeyID uint
	Model       string
	UsageDistributionAggregate
}
