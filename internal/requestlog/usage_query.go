package requestlog

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	"gorm.io/gorm"

	"gpt-load/internal/channel"
	"gpt-load/internal/platform/epochms"
	"gpt-load/internal/pricing"
	"gpt-load/internal/storage/dbtx"
	"gpt-load/internal/storage/models"
	"gpt-load/internal/usage"
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
		UsageBreakdownSortSuccessRate,
		UsageBreakdownSortAverageLatency,
		UsageBreakdownSortUncachedInputTokens,
		UsageBreakdownSortCacheReadTokens,
		UsageBreakdownSortCacheWrite5MTokens,
		UsageBreakdownSortCacheWrite1HTokens,
		UsageBreakdownSortCacheWriteUnknown,
		UsageBreakdownSortOutputTokens,
		UsageBreakdownSortTotalTokens,
		UsageBreakdownSortEstimatedCost:
		return true
	default:
		return false
	}
}

func queryUsageDistributions(
	scope *gorm.DB,
	summary UsageAggregate,
	accessKeyScoped bool,
) (UsageDistributions, error) {
	result := UsageDistributions{
		Group:     make(map[UsageDistributionMetric]UsageDistribution, 3),
		Model:     make(map[UsageDistributionMetric]UsageDistribution, 3),
		AccessKey: make(map[UsageDistributionMetric]UsageDistribution, 3),
	}
	dimensions := []UsageDistributionDimension{
		UsageDistributionDimensionGroup,
		UsageDistributionDimensionModel,
		UsageDistributionDimensionAccessKey,
	}
	if accessKeyScoped {
		dimensions = []UsageDistributionDimension{UsageDistributionDimensionModel}
	}
	for _, dimension := range dimensions {
		for _, metric := range []UsageDistributionMetric{
			UsageDistributionMetricRequests,
			UsageDistributionMetricTokens,
			UsageDistributionMetricCost,
		} {
			distribution, err := queryUsageDistribution(scope, summary, dimension, metric)
			if err != nil {
				return UsageDistributions{}, err
			}
			switch dimension {
			case UsageDistributionDimensionGroup:
				result.Group[metric] = distribution
			case UsageDistributionDimensionModel:
				result.Model[metric] = distribution
			case UsageDistributionDimensionAccessKey:
				result.AccessKey[metric] = distribution
			}
		}
	}
	return result, nil
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
			WHEN estimated_cost_nano_usd < 0
			THEN 1 ELSE 0 END), 0) AS invalid_cost
	`).Find(&integrity).Error; err != nil {
		return fmt.Errorf("check usage stat integrity: %w", err)
	}
	if integrity.InvalidBucket != 0 || integrity.InvalidCount != 0 ||
		integrity.InvalidToken != 0 || integrity.InvalidCost != 0 || integrity.InvalidDuration != 0 {
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

func queryUsageBreakdown(
	scope *gorm.DB,
	summary UsageAggregate,
	accessKeyScoped bool,
	input UsageQuery,
) (UsageBreakdown, error) {
	input = normalizeUsageBreakdownQuery(input)
	breakdownPage := input.BreakdownPage
	breakdownPageSize := input.BreakdownPageSize
	if breakdownPage-1 > maxUsageBreakdownInt()/breakdownPageSize {
		return UsageBreakdown{}, fmt.Errorf("query usage breakdown: pagination offset overflows int")
	}

	usageRows, err := queryUsageBreakdownRows(scope, accessKeyScoped)
	if err != nil {
		return UsageBreakdown{}, err
	}
	attemptRows, attemptTotal, err := queryUsageAttemptBreakdown(scope, input, accessKeyScoped)
	if err != nil {
		return UsageBreakdown{}, err
	}
	rows, err := mergeUsageBreakdownRows(usageRows, attemptRows, accessKeyScoped)
	if err != nil {
		return UsageBreakdown{}, err
	}
	if err := sortUsageBreakdownRows(scope, rows, input); err != nil {
		return UsageBreakdown{}, err
	}
	totalItems := len(rows)
	start := (breakdownPage - 1) * breakdownPageSize
	if start > totalItems {
		start = totalItems
	}
	end := totalItems
	if totalItems-start > breakdownPageSize {
		end = start + breakdownPageSize
	}
	pageRows := make([]UsageBreakdownRow, 0, end-start)
	for _, candidate := range rows[start:end] {
		pageRows = append(pageRows, candidate.row)
	}
	breakdown := UsageBreakdown{
		Scope:        "admin",
		Rows:         pageRows,
		Total:        summary,
		AttemptTotal: attemptTotal,
		Pagination: UsagePagination{
			Page: breakdownPage, PageSize: breakdownPageSize, TotalItems: totalItems,
			TotalPages: totalPages(totalItems, breakdownPageSize),
		},
	}
	if accessKeyScoped {
		breakdown.Scope = "access_key"
	}
	return breakdown, nil
}

type usageAttemptBreakdownRow struct {
	Model               string
	GroupID             *uint
	ChannelID           *string
	AttemptCount        int64
	AttemptFailureCount int64 `gorm:"column:attempt_failure_count"`
}

type usageBreakdownKey struct {
	Model     string
	GroupID   uint
	ChannelID string
}

type usageBreakdownCandidate struct {
	row         UsageBreakdownRow
	groupName   string
	channelName string
}

func queryUsageBreakdownRows(scope *gorm.DB, accessKeyScoped bool) ([]usageBreakdownRow, error) {
	query := scope.Session(&gorm.Session{})
	if accessKeyScoped {
		query = query.Select("usage_stats.model, NULL AS group_id, NULL AS channel_id, " + usageAggregateSelect).
			Group("usage_stats.model")
	} else {
		query = query.Select("usage_stats.model, usage_stats.group_id, usage_stats.channel_id, " + usageAggregateSelect).
			Group("usage_stats.model, usage_stats.group_id, usage_stats.channel_id")
	}
	var rows []usageBreakdownRow
	if err := query.Find(&rows).Error; err != nil {
		return nil, fmt.Errorf("query usage breakdown rows: %w", err)
	}
	return rows, nil
}

func usageAttemptStatScope(db *gorm.DB, input UsageQuery) *gorm.DB {
	scope := db.Session(&gorm.Session{NewDB: true}).Model(&models.UsageAttemptStat{}).
		Where("bucket_start_ms >= ? AND bucket_start_ms < ?", input.FromMS, input.ToMS)
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
	return scope
}

func queryUsageAttemptBreakdown(
	scope *gorm.DB,
	input UsageQuery,
	accessKeyScoped bool,
) ([]usageAttemptBreakdownRow, UsageAttemptAggregate, error) {
	attemptScope := usageAttemptStatScope(scope, input)
	var total UsageAttemptAggregate
	if err := attemptScope.Select("COALESCE(SUM(attempt_count), 0) AS attempt_count, COALESCE(SUM(failure_count), 0) AS attempt_failure_count").Find(&total).Error; err != nil {
		return nil, UsageAttemptAggregate{}, fmt.Errorf("query usage attempt total: %w", err)
	}
	if err := validateUsageAttemptAggregate(total); err != nil {
		return nil, UsageAttemptAggregate{}, err
	}
	query := attemptScope.Select("model, NULL AS group_id, NULL AS channel_id, SUM(attempt_count) AS attempt_count, SUM(failure_count) AS attempt_failure_count")
	if !accessKeyScoped {
		query = attemptScope.Select("model, group_id, channel_id, SUM(attempt_count) AS attempt_count, SUM(failure_count) AS attempt_failure_count").
			Group("model, group_id, channel_id")
	} else {
		query = query.Group("model")
	}
	var rows []usageAttemptBreakdownRow
	if err := query.Find(&rows).Error; err != nil {
		return nil, UsageAttemptAggregate{}, fmt.Errorf("query usage attempt breakdown: %w", err)
	}
	for _, row := range rows {
		if err := validateUsageAttemptAggregate(UsageAttemptAggregate{AttemptCount: row.AttemptCount, AttemptFailureCount: row.AttemptFailureCount}); err != nil {
			return nil, UsageAttemptAggregate{}, err
		}
	}
	return rows, total, nil
}

func validateUsageAttemptAggregate(value UsageAttemptAggregate) error {
	if value.AttemptCount < 0 || value.AttemptFailureCount < 0 || value.AttemptFailureCount > value.AttemptCount {
		return fmt.Errorf("invalid usage attempt aggregate")
	}
	return nil
}

func mergeUsageBreakdownRows(
	usageRows []usageBreakdownRow,
	attemptRows []usageAttemptBreakdownRow,
	accessKeyScoped bool,
) ([]usageBreakdownCandidate, error) {
	merged := make(map[usageBreakdownKey]usageBreakdownCandidate, len(usageRows)+len(attemptRows))
	for _, row := range usageRows {
		if err := validateUsageAggregate(row.UsageAggregate); err != nil {
			return nil, fmt.Errorf("validate usage breakdown: %w", err)
		}
		key := usageBreakdownKey{Model: row.Model}
		if row.GroupID != nil {
			key.GroupID = *row.GroupID
		}
		if row.ChannelID != nil {
			key.ChannelID = *row.ChannelID
		}
		merged[key] = usageBreakdownCandidate{row: UsageBreakdownRow{
			Model: row.Model, GroupID: cloneUintPointer(row.GroupID), ChannelID: cloneStringPointer(row.ChannelID),
			UsageAggregate: row.UsageAggregate,
		}}
	}
	for _, row := range attemptRows {
		attempts := UsageAttemptAggregate{AttemptCount: row.AttemptCount, AttemptFailureCount: row.AttemptFailureCount}
		key := usageBreakdownKey{Model: row.Model}
		if row.GroupID != nil {
			key.GroupID = *row.GroupID
		}
		if row.ChannelID != nil {
			key.ChannelID = *row.ChannelID
		}
		candidate, ok := merged[key]
		if !ok {
			candidate.row.Model = row.Model
			if !accessKeyScoped {
				candidate.row.GroupID = cloneUintPointer(row.GroupID)
				candidate.row.ChannelID = cloneStringPointer(row.ChannelID)
			}
		}
		if ok {
			candidate.row.AttemptCount += attempts.AttemptCount
			candidate.row.AttemptFailureCount += attempts.AttemptFailureCount
		} else {
			candidate.row.AttemptCount = attempts.AttemptCount
			candidate.row.AttemptFailureCount = attempts.AttemptFailureCount
		}
		if err := validateUsageAttemptAggregate(candidate.row.UsageAttemptAggregate); err != nil {
			return nil, err
		}
		merged[key] = candidate
	}
	rows := make([]usageBreakdownCandidate, 0, len(merged))
	for _, candidate := range merged {
		rows = append(rows, candidate)
	}
	return rows, nil
}

func cloneUintPointer(value *uint) *uint {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func cloneStringPointer(value *string) *string {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func sortUsageBreakdownRows(scope *gorm.DB, rows []usageBreakdownCandidate, input UsageQuery) error {
	groupNames := make(map[uint]string)
	groupIDs := make([]uint, 0)
	seenGroups := make(map[uint]struct{})
	for index := range rows {
		if rows[index].row.GroupID == nil {
			continue
		}
		id := *rows[index].row.GroupID
		if _, seen := seenGroups[id]; !seen {
			seenGroups[id] = struct{}{}
			groupIDs = append(groupIDs, id)
		}
	}
	if len(groupIDs) > 0 {
		var groups []models.Group
		if err := scope.Session(&gorm.Session{NewDB: true}).Model(&models.Group{}).Where("id IN ?", groupIDs).Find(&groups).Error; err != nil {
			return fmt.Errorf("query usage breakdown group names: %w", err)
		}
		for _, group := range groups {
			groupNames[group.ID] = group.Name
		}
	}
	for index := range rows {
		if rows[index].row.GroupID != nil {
			id := *rows[index].row.GroupID
			rows[index].groupName = groupNames[id]
			if rows[index].groupName == "" {
				rows[index].groupName = "#" + fmt.Sprint(id)
			}
		}
		if rows[index].row.ChannelID != nil {
			id := channel.ID(*rows[index].row.ChannelID)
			if descriptor, ok := channel.NewRegistry().Get(id); ok {
				rows[index].channelName = descriptor.Name
			} else {
				rows[index].channelName = "#" + *rows[index].row.ChannelID
			}
		}
	}
	sort.Slice(rows, func(left, right int) bool {
		comparison := compareUsageBreakdownPrimary(rows[left], rows[right], input.BreakdownSort)
		if input.BreakdownSortDirection == UsageBreakdownSortDescending {
			comparison = -comparison
		}
		if comparison != 0 {
			return comparison < 0
		}
		if rows[left].row.Model != rows[right].row.Model {
			return rows[left].row.Model < rows[right].row.Model
		}
		leftGroup, rightGroup := uint(0), uint(0)
		if rows[left].row.GroupID != nil {
			leftGroup = *rows[left].row.GroupID
		}
		if rows[right].row.GroupID != nil {
			rightGroup = *rows[right].row.GroupID
		}
		if leftGroup != rightGroup {
			return leftGroup < rightGroup
		}
		leftChannel, rightChannel := "", ""
		if rows[left].row.ChannelID != nil {
			leftChannel = *rows[left].row.ChannelID
		}
		if rows[right].row.ChannelID != nil {
			rightChannel = *rows[right].row.ChannelID
		}
		return leftChannel < rightChannel
	})
	return nil
}

func compareUsageBreakdownPrimary(left, right usageBreakdownCandidate, sortBy UsageBreakdownSort) int {
	compareInt64 := func(a, b int64) int {
		if a < b {
			return -1
		}
		if a > b {
			return 1
		}
		return 0
	}
	leftAggregate, rightAggregate := left.row.UsageAggregate, right.row.UsageAggregate
	switch sortBy {
	case UsageBreakdownSortModel:
		return strings.Compare(left.row.Model, right.row.Model)
	case UsageBreakdownSortGroup:
		return strings.Compare(left.groupName, right.groupName)
	case UsageBreakdownSortChannel:
		return strings.Compare(left.channelName, right.channelName)
	case UsageBreakdownSortRequestCount:
		return compareInt64(leftAggregate.RequestCount, rightAggregate.RequestCount)
	case UsageBreakdownSortSuccessCount:
		return compareInt64(leftAggregate.SuccessCount, rightAggregate.SuccessCount)
	case UsageBreakdownSortFailureCount:
		return compareInt64(leftAggregate.FailureCount, rightAggregate.FailureCount)
	case UsageBreakdownSortSuccessRate:
		leftRate, rightRate := float64(leftAggregate.SuccessCount), float64(rightAggregate.SuccessCount)
		if leftAggregate.RequestCount > 0 {
			leftRate /= float64(leftAggregate.RequestCount)
		}
		if rightAggregate.RequestCount > 0 {
			rightRate /= float64(rightAggregate.RequestCount)
		}
		if leftRate < rightRate {
			return -1
		}
		if leftRate > rightRate {
			return 1
		}
		return 0
	case UsageBreakdownSortAverageLatency:
		leftRate, rightRate := float64(leftAggregate.DurationMsTotal), float64(rightAggregate.DurationMsTotal)
		if leftAggregate.DurationSampleCount > 0 {
			leftRate /= float64(leftAggregate.DurationSampleCount)
		}
		if rightAggregate.DurationSampleCount > 0 {
			rightRate /= float64(rightAggregate.DurationSampleCount)
		}
		if leftRate < rightRate {
			return -1
		}
		if leftRate > rightRate {
			return 1
		}
		return 0
	case UsageBreakdownSortUncachedInputTokens:
		return compareInt64(leftAggregate.UncachedInputTokens, rightAggregate.UncachedInputTokens)
	case UsageBreakdownSortCacheReadTokens:
		return compareInt64(leftAggregate.CacheReadTokens, rightAggregate.CacheReadTokens)
	case UsageBreakdownSortCacheWrite5MTokens:
		return compareInt64(leftAggregate.CacheWrite5MTokens, rightAggregate.CacheWrite5MTokens)
	case UsageBreakdownSortCacheWrite1HTokens:
		return compareInt64(leftAggregate.CacheWrite1HTokens, rightAggregate.CacheWrite1HTokens)
	case UsageBreakdownSortCacheWriteUnknown:
		return compareInt64(leftAggregate.CacheWriteUnknownTokens, rightAggregate.CacheWriteUnknownTokens)
	case UsageBreakdownSortOutputTokens:
		return compareInt64(leftAggregate.OutputTokens, rightAggregate.OutputTokens)
	case UsageBreakdownSortTotalTokens:
		leftTokens, _ := usageAggregateTotalTokens(leftAggregate)
		rightTokens, _ := usageAggregateTotalTokens(rightAggregate)
		return compareInt64(leftTokens, rightTokens)
	case UsageBreakdownSortEstimatedCost:
		return compareInt64(leftAggregate.EstimatedCostNanoUSD, rightAggregate.EstimatedCostNanoUSD)
	default:
		return 0
	}
}

func maxUsageBreakdownInt() int {
	return int(^uint(0) >> 1)
}

func totalPages(totalItems, pageSize int) int {
	if totalItems == 0 {
		return 0
	}
	return (totalItems-1)/pageSize + 1
}
func queryUsageDistribution(
	scope *gorm.DB,
	summary UsageAggregate,
	dimension UsageDistributionDimension,
	metric UsageDistributionMetric,
) (UsageDistribution, error) {
	var rows []usageDistributionRow
	query := scope.Session(&gorm.Session{})
	switch dimension {
	case UsageDistributionDimensionGroup:
		query = query.Select("group_id, 0 AS access_key_id, '' AS model, "+usageDistributionAggregateSelect).
			Where("group_id IN (?)", scope.Session(&gorm.Session{NewDB: true}).
				Model(&models.Group{}).Select("id")).
			Group("group_id")
	case UsageDistributionDimensionModel:
		query = query.Select("0 AS group_id, 0 AS access_key_id, model, " + usageDistributionAggregateSelect).
			Group("model")
	case UsageDistributionDimensionAccessKey:
		query = query.Select("0 AS group_id, access_key_id, '' AS model, "+usageDistributionAggregateSelect).
			Where("access_key_id IN (?)", scope.Session(&gorm.Session{NewDB: true}).
				Model(&models.AccessKey{}).Select("id")).
			Group("access_key_id")
	default:
		return UsageDistribution{}, fmt.Errorf(
			"query usage distribution: unsupported dimension %q",
			dimension,
		)
	}
	switch metric {
	case UsageDistributionMetricRequests:
		query = query.
			Order("SUM(request_count) DESC").
			Order("SUM(estimated_cost_nano_usd) DESC")
	case UsageDistributionMetricTokens:
		query = query.
			Order(usageDistributionTotalTokensExpression + " DESC").
			Order("SUM(request_count) DESC").
			Order("SUM(estimated_cost_nano_usd) DESC")
	case UsageDistributionMetricCost:
		query = query.
			Order("SUM(estimated_cost_nano_usd) DESC").
			Order("SUM(request_count) DESC")
	default:
		return UsageDistribution{}, fmt.Errorf(
			"query usage distribution: unsupported metric %q",
			metric,
		)
	}
	switch dimension {
	case UsageDistributionDimensionGroup:
		query = query.Order("group_id ASC")
	case UsageDistributionDimensionModel:
		query = query.Order("model ASC")
	case UsageDistributionDimensionAccessKey:
		query = query.Order("access_key_id ASC")
	}
	if err := query.
		Limit(usageDistributionLimit).
		Find(&rows).Error; err != nil {
		return UsageDistribution{}, fmt.Errorf("query usage distribution: %w", err)
	}

	result := UsageDistribution{
		Dimension: dimension,
		Metric:    metric,
		Items:     make([]UsageDistributionItem, 0, len(rows)),
	}
	visible := UsageDistributionAggregate{}
	for _, row := range rows {
		var err error
		visible, err = addUsageDistributionAggregates(visible, row.UsageDistributionAggregate)
		if err != nil {
			return UsageDistribution{}, fmt.Errorf("sum usage distribution: %w", err)
		}
		result.Items = append(result.Items, UsageDistributionItem{
			GroupID: row.GroupID, AccessKeyID: row.AccessKeyID, Model: row.Model,
			UsageDistributionAggregate: row.UsageDistributionAggregate,
		})
	}
	other, err := subtractUsageDistributionAggregate(summary, visible)
	if err != nil {
		return UsageDistribution{}, fmt.Errorf("calculate other usage distribution: %w", err)
	}
	if other.RequestCount > 0 || other.TotalTokens > 0 || other.EstimatedCostNanoUSD > 0 {
		result.Other = &other
	}
	return result, nil
}

func mergeUsageHours(source []usageHourPoint, bucketWidthMS int64) ([]UsageSeriesPoint, error) {
	// queryUsageSeries orders source by bucket_start_ms ASC; adjacent-only
	// bucket folding depends on it.
	series := make([]UsageSeriesPoint, 0, len(source))
	for _, point := range source {
		bucketStartMS, err := epochms.AlignDown(
			point.BucketStartMS,
			bucketWidthMS,
		)
		if err != nil {
			return nil, fmt.Errorf("align usage bucket: %w", err)
		}
		if len(series) == 0 || series[len(series)-1].BucketStartMS != bucketStartMS {
			series = append(series, UsageSeriesPoint{
				BucketStartMS:  bucketStartMS,
				BucketEndMS:    bucketStartMS + bucketWidthMS,
				UsageAggregate: point.UsageAggregate,
			})
			continue
		}
		merged, err := addUsageAggregates(series[len(series)-1].UsageAggregate, point.UsageAggregate)
		if err != nil {
			return nil, fmt.Errorf("merge usage bucket %d: %w", bucketStartMS, err)
		}
		series[len(series)-1].UsageAggregate = merged
	}
	return series, nil
}

func addUsageAggregates(left, right UsageAggregate) (UsageAggregate, error) {
	result := UsageAggregate{}
	fields := []struct {
		name        string
		left, right int64
		target      *int64
	}{
		{"request count", left.RequestCount, right.RequestCount, &result.RequestCount},
		{"success count", left.SuccessCount, right.SuccessCount, &result.SuccessCount},
		{"failure count", left.FailureCount, right.FailureCount, &result.FailureCount},
		{"uncached input tokens", left.UncachedInputTokens, right.UncachedInputTokens, &result.UncachedInputTokens},
		{"cache read tokens", left.CacheReadTokens, right.CacheReadTokens, &result.CacheReadTokens},
		{"cache write 5m tokens", left.CacheWrite5MTokens, right.CacheWrite5MTokens, &result.CacheWrite5MTokens},
		{"cache write 1h tokens", left.CacheWrite1HTokens, right.CacheWrite1HTokens, &result.CacheWrite1HTokens},
		{"cache write unknown tokens", left.CacheWriteUnknownTokens, right.CacheWriteUnknownTokens, &result.CacheWriteUnknownTokens},
		{"output tokens", left.OutputTokens, right.OutputTokens, &result.OutputTokens},
		{"usage missing count", left.UsageMissingCount, right.UsageMissingCount, &result.UsageMissingCount},
		{"partial count", left.PartialCount, right.PartialCount, &result.PartialCount},
		{"unpriced request count", left.UnpricedRequestCount, right.UnpricedRequestCount, &result.UnpricedRequestCount},
		{"pricing partial count", left.PricingPartialCount, right.PricingPartialCount, &result.PricingPartialCount},
		{"duration ms total", left.DurationMsTotal, right.DurationMsTotal, &result.DurationMsTotal},
		{"duration sample count", left.DurationSampleCount, right.DurationSampleCount, &result.DurationSampleCount},
	}
	for _, field := range fields {
		value, ok := usage.CheckedAdd(field.left, field.right)
		if !ok {
			return UsageAggregate{}, fmt.Errorf("%s overflow or negative", field.name)
		}
		*field.target = value
	}
	cost, ok := pricing.CheckedAddNanoUSD(
		pricing.NanoUSD(left.EstimatedCostNanoUSD),
		pricing.NanoUSD(right.EstimatedCostNanoUSD),
	)
	if !ok {
		return UsageAggregate{}, fmt.Errorf("estimated cost nano USD overflow or negative")
	}
	result.EstimatedCostNanoUSD = int64(cost)
	if err := validateUsageAggregate(result); err != nil {
		return UsageAggregate{}, err
	}
	return result, nil
}

func addUsageDistributionAggregates(
	left, right UsageDistributionAggregate,
) (UsageDistributionAggregate, error) {
	requests, ok := usage.CheckedAdd(left.RequestCount, right.RequestCount)
	if !ok {
		return UsageDistributionAggregate{}, fmt.Errorf("request count overflow or negative")
	}
	tokens, ok := usage.CheckedAdd(left.TotalTokens, right.TotalTokens)
	if !ok {
		return UsageDistributionAggregate{}, fmt.Errorf("total tokens overflow or negative")
	}
	cost, ok := pricing.CheckedAddNanoUSD(
		pricing.NanoUSD(left.EstimatedCostNanoUSD),
		pricing.NanoUSD(right.EstimatedCostNanoUSD),
	)
	if !ok {
		return UsageDistributionAggregate{}, fmt.Errorf("estimated cost overflow or negative")
	}
	return UsageDistributionAggregate{
		RequestCount: requests, TotalTokens: tokens, EstimatedCostNanoUSD: int64(cost),
	}, nil
}

func subtractUsageDistributionAggregate(
	total UsageAggregate,
	part UsageDistributionAggregate,
) (UsageDistributionAggregate, error) {
	totalTokens, err := usageAggregateTotalTokens(total)
	if err != nil {
		return UsageDistributionAggregate{}, err
	}
	if part.RequestCount < 0 || part.RequestCount > total.RequestCount ||
		part.TotalTokens < 0 || part.TotalTokens > totalTokens ||
		part.EstimatedCostNanoUSD < 0 || part.EstimatedCostNanoUSD > total.EstimatedCostNanoUSD {
		return UsageDistributionAggregate{}, fmt.Errorf("invalid distribution remainder")
	}
	return UsageDistributionAggregate{
		RequestCount:         total.RequestCount - part.RequestCount,
		TotalTokens:          totalTokens - part.TotalTokens,
		EstimatedCostNanoUSD: total.EstimatedCostNanoUSD - part.EstimatedCostNanoUSD,
	}, nil
}

func validateUsageAggregate(aggregate UsageAggregate) error {
	fields := []struct {
		name  string
		value int64
	}{
		{"request count", aggregate.RequestCount},
		{"success count", aggregate.SuccessCount},
		{"failure count", aggregate.FailureCount},
		{"uncached input tokens", aggregate.UncachedInputTokens},
		{"cache read tokens", aggregate.CacheReadTokens},
		{"cache write 5m tokens", aggregate.CacheWrite5MTokens},
		{"cache write 1h tokens", aggregate.CacheWrite1HTokens},
		{"cache write unknown tokens", aggregate.CacheWriteUnknownTokens},
		{"output tokens", aggregate.OutputTokens},
		{"usage missing count", aggregate.UsageMissingCount},
		{"partial count", aggregate.PartialCount},
		{"unpriced request count", aggregate.UnpricedRequestCount},
		{"pricing partial count", aggregate.PricingPartialCount},
		{"duration ms total", aggregate.DurationMsTotal},
		{"duration sample count", aggregate.DurationSampleCount},
	}
	for _, field := range fields {
		if field.value < 0 {
			return fmt.Errorf("negative %s", field.name)
		}
	}
	requestCount, ok := usage.CheckedAdd(aggregate.SuccessCount, aggregate.FailureCount)
	if !ok || requestCount != aggregate.RequestCount {
		return fmt.Errorf("request count does not equal success plus failure")
	}
	if aggregate.DurationSampleCount > aggregate.RequestCount {
		return fmt.Errorf("duration sample count exceeds request count")
	}
	if _, err := usageAggregateTotalTokens(aggregate); err != nil {
		return err
	}
	if aggregate.EstimatedCostNanoUSD < 0 {
		return fmt.Errorf("invalid cost")
	}
	return nil
}

func usageAggregateTotalTokens(aggregate UsageAggregate) (int64, error) {
	totalTokens := int64(0)
	for _, value := range []int64{
		aggregate.UncachedInputTokens,
		aggregate.CacheReadTokens,
		aggregate.CacheWrite5MTokens,
		aggregate.CacheWrite1HTokens,
		aggregate.CacheWriteUnknownTokens,
		aggregate.OutputTokens,
	} {
		var added bool
		totalTokens, added = usage.CheckedAdd(totalTokens, value)
		if !added {
			return 0, fmt.Errorf("total tokens overflow")
		}
	}
	return totalTokens, nil
}

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
	"COALESCE(SUM(estimated_cost_nano_usd), 0) AS estimated_cost_nano_usd, " +
	"COALESCE(SUM(duration_ms_total), 0) AS duration_ms_total, " +
	"COALESCE(SUM(duration_sample_count), 0) AS duration_sample_count, " +
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
	Model     string
	GroupID   *uint
	ChannelID *string
	UsageAggregate
}

type usageHourPoint struct {
	BucketStartMS int64
	UsageAggregate
}

type usageStatIntegrity struct {
	InvalidBucket   int64
	InvalidCount    int64
	InvalidToken    int64
	InvalidCost     int64
	InvalidDuration int64
}

type usageDistributionRow struct {
	GroupID     uint
	AccessKeyID uint
	Model       string
	UsageDistributionAggregate
}
