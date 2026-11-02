package requestlog

import (
	"fmt"

	"gorm.io/gorm"

	"gpt-load/internal/platform/epochms"
	"gpt-load/internal/pricing"
	"gpt-load/internal/storage/models"
	"gpt-load/internal/usage"
)

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
		{"success output tokens", left.SuccessOutputTokens, right.SuccessOutputTokens, &result.SuccessOutputTokens},
		{"success duration total", left.SuccessDurationMsTotal, right.SuccessDurationMsTotal, &result.SuccessDurationMsTotal},
		{"success efficiency samples", left.SuccessEfficiencySampleCount, right.SuccessEfficiencySampleCount, &result.SuccessEfficiencySampleCount},
		{"usage missing count", left.UsageMissingCount, right.UsageMissingCount, &result.UsageMissingCount},
		{"partial count", left.PartialCount, right.PartialCount, &result.PartialCount},
		{"unpriced request count", left.UnpricedRequestCount, right.UnpricedRequestCount, &result.UnpricedRequestCount},
		{"pricing partial count", left.PricingPartialCount, right.PricingPartialCount, &result.PricingPartialCount},
		{"duration ms total", left.DurationMsTotal, right.DurationMsTotal, &result.DurationMsTotal},
		{"duration sample count", left.DurationSampleCount, right.DurationSampleCount, &result.DurationSampleCount},
		{"first response ms total", left.FirstResponseMsTotal, right.FirstResponseMsTotal, &result.FirstResponseMsTotal},
		{"first response sample count", left.FirstResponseSampleCount, right.FirstResponseSampleCount, &result.FirstResponseSampleCount},
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
		{"success output tokens", aggregate.SuccessOutputTokens},
		{"success duration total", aggregate.SuccessDurationMsTotal},
		{"success efficiency samples", aggregate.SuccessEfficiencySampleCount},
		{"usage missing count", aggregate.UsageMissingCount},
		{"partial count", aggregate.PartialCount},
		{"unpriced request count", aggregate.UnpricedRequestCount},
		{"pricing partial count", aggregate.PricingPartialCount},
		{"duration ms total", aggregate.DurationMsTotal},
		{"duration sample count", aggregate.DurationSampleCount},
		{"first response ms total", aggregate.FirstResponseMsTotal},
		{"first response sample count", aggregate.FirstResponseSampleCount},
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
	if aggregate.FirstResponseSampleCount > aggregate.RequestCount {
		return fmt.Errorf("first response sample count exceeds request count")
	}
	if aggregate.SuccessEfficiencySampleCount > aggregate.SuccessCount {
		return fmt.Errorf("success efficiency sample count exceeds success count")
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
