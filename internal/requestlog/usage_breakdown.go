package requestlog

import (
	"fmt"
	"sort"
	"strings"

	"gorm.io/gorm"

	"gpt-load/internal/channel"
	"gpt-load/internal/storage/models"
	"gpt-load/internal/usage"
)

type usageAttemptBreakdownRow struct {
	Model               string
	ClientModel         string
	GroupID             *uint
	ChannelID           *string
	AttemptCount        int64
	AttemptFailureCount int64 `gorm:"column:attempt_failure_count"`
	NormalAttemptCount  int64 `gorm:"column:normal_attempt_count"`
	SlowAttemptCount    int64 `gorm:"column:slow_attempt_count"`
	FaultyAttemptCount  int64 `gorm:"column:faulty_attempt_count"`
}

type usageBreakdownKey struct {
	Model       string
	ClientModel string
	GroupID     uint
	ChannelID   string
}

type usageBreakdownCandidate struct {
	row         UsageBreakdownRow
	groupName   string
	channelName string
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

	usageRows, err := queryUsageBreakdownRows(scope, accessKeyScoped, input.BreakdownByClientModel)
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

func queryUsageBreakdownRows(scope *gorm.DB, accessKeyScoped bool, byClientModel bool) ([]usageBreakdownRow, error) {
	query := scope.Session(&gorm.Session{})
	clientSelect := "'' AS client_model, "
	if byClientModel {
		clientSelect = "usage_stats.client_model, "
	}
	if accessKeyScoped {
		query = query.Select(clientSelect + "usage_stats.model, NULL AS group_id, NULL AS channel_id, " + usageAggregateSelect).
			Group("usage_stats.model")
	} else {
		query = query.Select(clientSelect + "usage_stats.model, usage_stats.group_id, usage_stats.channel_id, " + usageAggregateSelect).
			Group("usage_stats.model, usage_stats.group_id, usage_stats.channel_id")
	}
	if byClientModel {
		query = query.Group("usage_stats.client_model")
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
	if input.ClientModel != "" {
		scope = scope.Where("client_model = ?", input.ClientModel)
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
	if err := attemptScope.Select(
		"COALESCE(SUM(attempt_count), 0) AS attempt_count, " +
			"COALESCE(SUM(failure_count), 0) AS attempt_failure_count, " +
			"COALESCE(SUM(normal_attempt_count), 0) AS normal_attempt_count, " +
			"COALESCE(SUM(slow_attempt_count), 0) AS slow_attempt_count, " +
			"COALESCE(SUM(faulty_attempt_count), 0) AS faulty_attempt_count",
	).Find(&total).Error; err != nil {
		return nil, UsageAttemptAggregate{}, fmt.Errorf("query usage attempt total: %w", err)
	}
	if err := validateUsageAttemptAggregate(total); err != nil {
		return nil, UsageAttemptAggregate{}, err
	}
	clientSelect := "'' AS client_model, "
	if input.BreakdownByClientModel {
		clientSelect = "client_model, "
	}
	query := attemptScope.Select(
		clientSelect + "model, NULL AS group_id, NULL AS channel_id, SUM(attempt_count) AS attempt_count, " +
			"SUM(failure_count) AS attempt_failure_count, " +
			"SUM(normal_attempt_count) AS normal_attempt_count, " +
			"SUM(slow_attempt_count) AS slow_attempt_count, " +
			"SUM(faulty_attempt_count) AS faulty_attempt_count",
	)
	if !accessKeyScoped {
		query = attemptScope.Select(
			clientSelect + "model, group_id, channel_id, SUM(attempt_count) AS attempt_count, " +
				"SUM(failure_count) AS attempt_failure_count, " +
				"SUM(normal_attempt_count) AS normal_attempt_count, " +
				"SUM(slow_attempt_count) AS slow_attempt_count, " +
				"SUM(faulty_attempt_count) AS faulty_attempt_count",
		).
			Group("model, group_id, channel_id")
	} else {
		query = query.Group("model")
	}
	if input.BreakdownByClientModel {
		query = query.Group("client_model")
	}
	var rows []usageAttemptBreakdownRow
	if err := query.Find(&rows).Error; err != nil {
		return nil, UsageAttemptAggregate{}, fmt.Errorf("query usage attempt breakdown: %w", err)
	}
	for _, row := range rows {
		if err := validateUsageAttemptAggregate(attemptAggregate(row)); err != nil {
			return nil, UsageAttemptAggregate{}, err
		}
	}
	return rows, total, nil
}

func attemptAggregate(row usageAttemptBreakdownRow) UsageAttemptAggregate {
	return UsageAttemptAggregate{
		AttemptCount: row.AttemptCount, AttemptFailureCount: row.AttemptFailureCount,
		NormalAttemptCount: row.NormalAttemptCount, SlowAttemptCount: row.SlowAttemptCount,
		FaultyAttemptCount: row.FaultyAttemptCount,
	}
}

func validateUsageAttemptAggregate(value UsageAttemptAggregate) error {
	if value.AttemptCount < 0 || value.AttemptFailureCount < 0 ||
		value.AttemptFailureCount > value.AttemptCount || value.NormalAttemptCount < 0 ||
		value.SlowAttemptCount < 0 || value.FaultyAttemptCount < 0 {
		return fmt.Errorf("invalid usage attempt aggregate")
	}
	feedbackCount, ok := usage.CheckedAdd(value.NormalAttemptCount, value.SlowAttemptCount)
	if !ok {
		return fmt.Errorf("invalid usage attempt feedback aggregate")
	}
	feedbackCount, ok = usage.CheckedAdd(feedbackCount, value.FaultyAttemptCount)
	if !ok || feedbackCount > value.AttemptCount {
		return fmt.Errorf("invalid usage attempt feedback aggregate")
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
		key := usageBreakdownKey{Model: row.Model, ClientModel: row.ClientModel}
		if row.GroupID != nil {
			key.GroupID = *row.GroupID
		}
		if row.ChannelID != nil {
			key.ChannelID = *row.ChannelID
		}
		merged[key] = usageBreakdownCandidate{row: UsageBreakdownRow{
			Model: row.Model, ClientModel: row.ClientModel, GroupID: cloneUintPointer(row.GroupID), ChannelID: cloneStringPointer(row.ChannelID),
			UsageAggregate: row.UsageAggregate,
		}}
	}
	for _, row := range attemptRows {
		attempts := attemptAggregate(row)
		key := usageBreakdownKey{Model: row.Model, ClientModel: row.ClientModel}
		if row.GroupID != nil {
			key.GroupID = *row.GroupID
		}
		if row.ChannelID != nil {
			key.ChannelID = *row.ChannelID
		}
		candidate, ok := merged[key]
		if !ok {
			candidate.row.Model = row.Model
			candidate.row.ClientModel = row.ClientModel
			if !accessKeyScoped {
				candidate.row.GroupID = cloneUintPointer(row.GroupID)
				candidate.row.ChannelID = cloneStringPointer(row.ChannelID)
			}
		}
		if ok {
			if err := checkedInt64Add(&candidate.row.AttemptCount, attempts.AttemptCount, "attempt_count"); err != nil {
				return nil, err
			}
			if err := checkedInt64Add(&candidate.row.AttemptFailureCount, attempts.AttemptFailureCount, "attempt_failure_count"); err != nil {
				return nil, err
			}
			if err := checkedInt64Add(&candidate.row.NormalAttemptCount, attempts.NormalAttemptCount, "normal_attempt_count"); err != nil {
				return nil, err
			}
			if err := checkedInt64Add(&candidate.row.SlowAttemptCount, attempts.SlowAttemptCount, "slow_attempt_count"); err != nil {
				return nil, err
			}
			if err := checkedInt64Add(&candidate.row.FaultyAttemptCount, attempts.FaultyAttemptCount, "faulty_attempt_count"); err != nil {
				return nil, err
			}
		} else {
			candidate.row.UsageAttemptAggregate = attempts
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
	var sortError error
	sort.Slice(rows, func(left, right int) bool {
		if sortError != nil {
			return false
		}
		comparison := compareUsageBreakdownSamplePresence(rows[left], rows[right], input.BreakdownSort)
		if comparison == 0 {
			comparison, sortError = compareUsageBreakdownPrimary(rows[left], rows[right], input.BreakdownSort)
			if sortError != nil {
				return false
			}
			if input.BreakdownSortDirection == UsageBreakdownSortDescending {
				comparison = -comparison
			}
		}
		if comparison != 0 {
			return comparison < 0
		}
		if rows[left].row.Model != rows[right].row.Model {
			return rows[left].row.Model < rows[right].row.Model
		}
		if rows[left].row.ClientModel != rows[right].row.ClientModel {
			return rows[left].row.ClientModel < rows[right].row.ClientModel
		}
		return compareUsageBreakdownIdentity(rows[left].row, rows[right].row) < 0
	})
	if sortError != nil {
		return fmt.Errorf("sort usage breakdown: %w", sortError)
	}
	return nil
}

func compareUsageBreakdownIdentity(left, right UsageBreakdownRow) int {
	if left.Model != right.Model {
		return strings.Compare(left.Model, right.Model)
	}
	if left.ClientModel != right.ClientModel {
		return strings.Compare(left.ClientModel, right.ClientModel)
	}
	leftGroup, rightGroup := optionalUintValue(left.GroupID), optionalUintValue(right.GroupID)
	if leftGroup != rightGroup {
		if leftGroup < rightGroup {
			return -1
		}
		return 1
	}
	leftChannel, rightChannel := optionalStringValue(left.ChannelID), optionalStringValue(right.ChannelID)
	return strings.Compare(leftChannel, rightChannel)
}

func optionalUintValue(value *uint) uint {
	if value == nil {
		return 0
	}
	return *value
}

func optionalStringValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func compareUsageBreakdownSamplePresence(left, right usageBreakdownCandidate, sortBy UsageBreakdownSort) int {
	if sortBy != UsageBreakdownSortAverageFirstResponse && sortBy != UsageBreakdownSortCacheRate && sortBy != UsageBreakdownSortTokenEfficiency {
		return 0
	}
	leftHasSamples := left.row.UsageAggregate.FirstResponseSampleCount > 0
	rightHasSamples := right.row.UsageAggregate.FirstResponseSampleCount > 0
	if sortBy == UsageBreakdownSortCacheRate {
		leftHasSamples = left.row.UncachedInputTokens+left.row.CacheReadTokens > 0
		rightHasSamples = right.row.UncachedInputTokens+right.row.CacheReadTokens > 0
	}
	if sortBy == UsageBreakdownSortTokenEfficiency {
		leftHasSamples = left.row.SuccessCount > 0 && left.row.SuccessEfficiencySampleCount == left.row.SuccessCount && left.row.SuccessDurationMsTotal > 0
		rightHasSamples = right.row.SuccessCount > 0 && right.row.SuccessEfficiencySampleCount == right.row.SuccessCount && right.row.SuccessDurationMsTotal > 0
	}
	switch {
	case leftHasSamples == rightHasSamples:
		return 0
	case leftHasSamples:
		return -1
	default:
		return 1
	}
}

func compareRatios(leftN, leftD, rightN, rightD float64) int {
	if leftD == 0 || rightD == 0 {
		return 0
	}
	left, right := leftN/leftD, rightN/rightD
	if left < right {
		return -1
	}
	if left > right {
		return 1
	}
	return 0
}

func compareUsageBreakdownPrimary(left, right usageBreakdownCandidate, sortBy UsageBreakdownSort) (int, error) {
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
		return strings.Compare(left.row.Model, right.row.Model), nil
	case UsageBreakdownSortGroup:
		return strings.Compare(left.groupName, right.groupName), nil
	case UsageBreakdownSortChannel:
		return strings.Compare(left.channelName, right.channelName), nil
	case UsageBreakdownSortRequestCount:
		return compareInt64(leftAggregate.RequestCount, rightAggregate.RequestCount), nil
	case UsageBreakdownSortSuccessCount:
		return compareInt64(leftAggregate.SuccessCount, rightAggregate.SuccessCount), nil
	case UsageBreakdownSortFailureCount:
		return compareInt64(leftAggregate.FailureCount, rightAggregate.FailureCount), nil
	case UsageBreakdownSortAttemptCount:
		return compareInt64(left.row.AttemptCount, right.row.AttemptCount), nil
	case UsageBreakdownSortAttemptFailureCount:
		return compareInt64(left.row.AttemptFailureCount, right.row.AttemptFailureCount), nil
	case UsageBreakdownSortNormalAttemptCount:
		return compareInt64(left.row.NormalAttemptCount, right.row.NormalAttemptCount), nil
	case UsageBreakdownSortSlowAttemptCount:
		return compareInt64(left.row.SlowAttemptCount, right.row.SlowAttemptCount), nil
	case UsageBreakdownSortFaultyAttemptCount:
		return compareInt64(left.row.FaultyAttemptCount, right.row.FaultyAttemptCount), nil
	case UsageBreakdownSortSuccessRate:
		leftRate, rightRate := float64(leftAggregate.SuccessCount), float64(rightAggregate.SuccessCount)
		if leftAggregate.RequestCount > 0 {
			leftRate /= float64(leftAggregate.RequestCount)
		}
		if rightAggregate.RequestCount > 0 {
			rightRate /= float64(rightAggregate.RequestCount)
		}
		if leftRate < rightRate {
			return -1, nil
		}
		if leftRate > rightRate {
			return 1, nil
		}
		return 0, nil
	case UsageBreakdownSortAverageDuration:
		leftRate, rightRate := float64(leftAggregate.DurationMsTotal), float64(rightAggregate.DurationMsTotal)
		if leftAggregate.DurationSampleCount > 0 {
			leftRate /= float64(leftAggregate.DurationSampleCount)
		}
		if rightAggregate.DurationSampleCount > 0 {
			rightRate /= float64(rightAggregate.DurationSampleCount)
		}
		if leftRate < rightRate {
			return -1, nil
		}
		if leftRate > rightRate {
			return 1, nil
		}
		return 0, nil
	case UsageBreakdownSortAverageFirstResponse:
		leftRate, rightRate := float64(leftAggregate.FirstResponseMsTotal), float64(rightAggregate.FirstResponseMsTotal)
		if leftAggregate.FirstResponseSampleCount > 0 {
			leftRate /= float64(leftAggregate.FirstResponseSampleCount)
		}
		if rightAggregate.FirstResponseSampleCount > 0 {
			rightRate /= float64(rightAggregate.FirstResponseSampleCount)
		}
		if leftRate < rightRate {
			return -1, nil
		}
		if leftRate > rightRate {
			return 1, nil
		}
		return 0, nil
	case UsageBreakdownSortCacheRate:
		return compareRatios(float64(leftAggregate.CacheReadTokens), float64(leftAggregate.UncachedInputTokens)+float64(leftAggregate.CacheReadTokens),
			float64(rightAggregate.CacheReadTokens), float64(rightAggregate.UncachedInputTokens)+float64(rightAggregate.CacheReadTokens)), nil
	case UsageBreakdownSortTokenEfficiency:
		if leftAggregate.SuccessCount == 0 || leftAggregate.SuccessEfficiencySampleCount != leftAggregate.SuccessCount || leftAggregate.SuccessDurationMsTotal <= 0 ||
			rightAggregate.SuccessCount == 0 || rightAggregate.SuccessEfficiencySampleCount != rightAggregate.SuccessCount || rightAggregate.SuccessDurationMsTotal <= 0 {
			return 0, nil
		}
		return compareRatios(float64(leftAggregate.SuccessOutputTokens), float64(leftAggregate.SuccessDurationMsTotal),
			float64(rightAggregate.SuccessOutputTokens), float64(rightAggregate.SuccessDurationMsTotal)), nil
	case UsageBreakdownSortUncachedInputTokens:
		return compareInt64(leftAggregate.UncachedInputTokens, rightAggregate.UncachedInputTokens), nil
	case UsageBreakdownSortCacheReadTokens:
		return compareInt64(leftAggregate.CacheReadTokens, rightAggregate.CacheReadTokens), nil
	case UsageBreakdownSortCacheWrite5MTokens:
		return compareInt64(leftAggregate.CacheWrite5MTokens, rightAggregate.CacheWrite5MTokens), nil
	case UsageBreakdownSortCacheWrite1HTokens:
		return compareInt64(leftAggregate.CacheWrite1HTokens, rightAggregate.CacheWrite1HTokens), nil
	case UsageBreakdownSortCacheWriteUnknown:
		return compareInt64(leftAggregate.CacheWriteUnknownTokens, rightAggregate.CacheWriteUnknownTokens), nil
	case UsageBreakdownSortOutputTokens:
		return compareInt64(leftAggregate.OutputTokens, rightAggregate.OutputTokens), nil
	case UsageBreakdownSortTotalTokens:
		leftTokens, err := usageAggregateTotalTokens(leftAggregate)
		if err != nil {
			return 0, fmt.Errorf("left total tokens: %w", err)
		}
		rightTokens, err := usageAggregateTotalTokens(rightAggregate)
		if err != nil {
			return 0, fmt.Errorf("right total tokens: %w", err)
		}
		return compareInt64(leftTokens, rightTokens), nil
	case UsageBreakdownSortEstimatedCost:
		return compareInt64(leftAggregate.EstimatedCostNanoUSD, rightAggregate.EstimatedCostNanoUSD), nil
	default:
		return 0, nil
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
