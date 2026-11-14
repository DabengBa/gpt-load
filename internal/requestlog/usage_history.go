package requestlog

import (
	"fmt"
	"strings"

	"gorm.io/gorm"
)

type usageHistoryHour struct {
	Model         string
	ClientModel   string
	GroupID       uint
	ChannelID     string
	BucketStartMS int64
	RequestCount  int64
	SuccessCount  int64
}

// queryUsageBreakdownHistory only reads final page identities. The tuple OR is
// intentional: independent IN predicates would include cross-product routes.
func queryUsageBreakdownHistory(scope *gorm.DB, rows []UsageBreakdownRow, input UsageQuery, width int64) error {
	if len(rows) == 0 {
		return nil
	}
	keys := make(map[usageBreakdownKey]int, len(rows))
	conditions := make([]string, 0, len(rows))
	args := make([]any, 0, len(rows)*4)
	for i := range rows {
		rows[i].History = make([]UsageHistoryPoint, 0)
		row := &rows[i]
		key := usageBreakdownKey{Model: row.Model, ClientModel: row.ClientModel}
		condition := "model = ?"
		args = append(args, row.Model)
		if input.BreakdownByClientModel {
			condition += " AND client_model = ?"
			args = append(args, row.ClientModel)
		}
		if input.AccessKeyID == nil {
			key.GroupID, key.ChannelID = *row.GroupID, *row.ChannelID
			condition += " AND group_id = ? AND channel_id = ?"
			args = append(args, key.GroupID, key.ChannelID)
		}
		keys[key] = i
		conditions = append(conditions, "("+condition+")")
	}
	clientSelect, groupSelect, groupBy := "'' AS client_model", "0 AS group_id, '' AS channel_id", "model, bucket_start_ms"
	if input.BreakdownByClientModel {
		clientSelect = "client_model"
		groupBy += ", client_model"
	}
	if input.AccessKeyID == nil {
		groupSelect = "group_id, channel_id"
		groupBy += ", group_id, channel_id"
	}
	var source []usageHistoryHour
	if err := scope.Where("("+strings.Join(conditions, " OR ")+")", args...).
		Select("model, " + clientSelect + ", " + groupSelect + ", bucket_start_ms, SUM(request_count) AS request_count, SUM(success_count) AS success_count").
		Group(groupBy).Order("bucket_start_ms ASC").Find(&source).Error; err != nil {
		return fmt.Errorf("query usage breakdown history: %w", err)
	}
	for _, hour := range source {
		key := usageBreakdownKey{Model: hour.Model, ClientModel: hour.ClientModel, GroupID: hour.GroupID, ChannelID: hour.ChannelID}
		index, ok := keys[key]
		if !ok {
			return fmt.Errorf("query usage breakdown history: unexpected identity")
		}
		if hour.RequestCount < 0 || hour.SuccessCount < 0 || hour.SuccessCount > hour.RequestCount {
			return fmt.Errorf("query usage breakdown history: invalid counts")
		}
		bucketStart := hour.BucketStartMS - hour.BucketStartMS%width
		history := &rows[index].History
		if len(*history) == 0 || (*history)[len(*history)-1].BucketStartMS != bucketStart {
			*history = append(*history, UsageHistoryPoint{BucketStartMS: bucketStart, BucketEndMS: bucketStart + width})
		}
		point := &(*history)[len(*history)-1]
		if err := checkedInt64Add(&point.RequestCount, hour.RequestCount, "request_count"); err != nil {
			return err
		}
		if err := checkedInt64Add(&point.SuccessCount, hour.SuccessCount, "success_count"); err != nil {
			return err
		}
	}
	return nil
}
