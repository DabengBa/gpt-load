package requestlog

import (
	"context"
	"fmt"

	"gorm.io/gorm"

	"gpt-load/internal/storage/dbtx"
	"gpt-load/internal/storage/models"
)

// QueryGroupUsage returns request-level traffic outcomes by group for a half-open
// time window. It intentionally reads request_logs rather than hourly usage_stats
// so a rolling window does not include a partially overlapping bucket, which is
// also why the control-plane exclusion has to be applied here and not only on the
// aggregation-journal side.
func (service *Service) QueryGroupUsage(
	ctx context.Context,
	input GroupUsageQuery,
) (map[uint]GroupUsage, error) {
	if service == nil || service.db == nil {
		return nil, fmt.Errorf("query group usage: database is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if input.FromMS < 0 || input.ToMS <= input.FromMS {
		return nil, fmt.Errorf("query group usage: invalid time range")
	}

	result := make(map[uint]GroupUsage)
	err := dbtx.Run(ctx, service.db, dbtx.Options{
		Mode:           dbtx.ReadSnapshot,
		CleanupTimeout: usageRollbackTimeout,
		Operation:      "group usage read transaction",
	}, func(connection *gorm.DB) error {
		var rows []struct {
			GroupID      uint  `gorm:"column:group_id"`
			RequestCount int64 `gorm:"column:request_count"`
			SuccessCount int64 `gorm:"column:success_count"`
		}
		if err := withoutControlPlaneObservations(connection.Model(&models.RequestLog{})).
			Select("group_id, COUNT(*) AS request_count, "+
				"COALESCE(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END), 0) AS success_count").
			Where("completed_at_ms >= ? AND completed_at_ms < ?", input.FromMS, input.ToMS).
			Where("group_id > 0").
			Group("group_id").
			Order("group_id ASC").
			Find(&rows).Error; err != nil {
			return fmt.Errorf("query group usage rows: %w", err)
		}
		for _, row := range rows {
			if row.RequestCount < 0 || row.RequestCount > maxJSONSafeInteger ||
				row.SuccessCount < 0 || row.SuccessCount > maxJSONSafeInteger ||
				row.SuccessCount > row.RequestCount {
				return fmt.Errorf("query group usage: invalid aggregate for group %d", row.GroupID)
			}
			result[row.GroupID] = GroupUsage{
				RequestCount: row.RequestCount,
				SuccessCount: row.SuccessCount,
			}
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("query group usage: %w", err)
	}
	return result, nil
}
