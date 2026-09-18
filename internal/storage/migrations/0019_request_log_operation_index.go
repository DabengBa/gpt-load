package migrations

import (
	"fmt"
	"strings"

	"gorm.io/gorm"
)

const ID0019 = "0019_request_log_operation_index"

const (
	requestLogOperationTable0019 = "request_logs"
	requestLogOperationIndex0019 = "idx_request_logs_operation_completed_id"
)

// Up0019 为操作筛选增加与游标顺序一致的索引，不改历史日志。
func Up0019(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("request log operation index migration: database is nil")
	}
	if !db.Migrator().HasTable(requestLogOperationTable0019) {
		return fmt.Errorf(
			"request log operation index migration: table %q is missing",
			requestLogOperationTable0019,
		)
	}
	if !db.Migrator().HasIndex(requestLogOperationTable0019, requestLogOperationIndex0019) {
		if err := db.Exec(
			"CREATE INDEX idx_request_logs_operation_completed_id " +
				"ON request_logs (operation, completed_at_ms DESC, id DESC)",
		).Error; err != nil {
			return fmt.Errorf("create request log operation index: %w", err)
		}
	}
	return Validate0019(db)
}

// Validate0019 confirms the operation index covers the cursor ordering exactly.
func Validate0019(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("validate request log operation index: database is nil")
	}
	if !db.Migrator().HasTable(requestLogOperationTable0019) {
		return fmt.Errorf("request log operation index: table %q is missing", requestLogOperationTable0019)
	}
	if !db.Migrator().HasIndex(requestLogOperationTable0019, requestLogOperationIndex0019) {
		return fmt.Errorf("request log operation index %q is missing", requestLogOperationIndex0019)
	}
	switch strings.ToLower(db.Dialector.Name()) {
	case "sqlite":
		var columns []struct {
			Name string
			Desc int
			Key  int
		}
		if err := db.Raw("PRAGMA index_xinfo('idx_request_logs_operation_completed_id')").
			Scan(&columns).Error; err != nil {
			return fmt.Errorf("inspect request log operation index columns: %w", err)
		}
		var names []string
		var descending []bool
		for _, column := range columns {
			if column.Key == 1 {
				names = append(names, column.Name)
				descending = append(descending, column.Desc != 0)
			}
		}
		want := []struct {
			name string
			desc bool
		}{
			{name: "operation"},
			{name: "completed_at_ms", desc: true},
			{name: "id", desc: true},
		}
		if len(names) != len(want) {
			return fmt.Errorf("request log operation index columns = %v, want %v", names, want)
		}
		for index := range want {
			if !strings.EqualFold(names[index], want[index].name) || descending[index] != want[index].desc {
				return fmt.Errorf(
					"request log operation index columns = %v descending = %v, want %v",
					names, descending, want,
				)
			}
		}
		return nil
	default:
		return fmt.Errorf("validate request log operation index: unsupported database driver %q", db.Dialector.Name())
	}
}
