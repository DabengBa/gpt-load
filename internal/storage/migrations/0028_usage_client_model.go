package migrations

import (
	"fmt"
	"strings"

	"gorm.io/gorm"
)

const ID0028 = "0028_usage_client_model"

func Up0028(db *gorm.DB) error {
	if db == nil || db.Dialector == nil || !strings.EqualFold(db.Dialector.Name(), "sqlite") {
		return fmt.Errorf("usage client model migration: unsupported or nil database")
	}
	for _, table := range []string{"usage_aggregation_journal", "usage_attempt_aggregation_journals", "usage_stats", "usage_attempt_stats"} {
		if !db.Migrator().HasColumn(table, "client_model") {
			if err := db.Exec(fmt.Sprintf("ALTER TABLE %s ADD COLUMN client_model varchar(255) NOT NULL DEFAULT ''", table)).Error; err != nil {
				return fmt.Errorf("add %s.client_model: %w", table, err)
			}
		}
	}
	for _, item := range []struct{ table, index string }{
		{"usage_stats", "idx_usage_stats_identity"},
		{"usage_attempt_stats", "idx_usage_attempt_stats_identity"},
	} {
		if err := db.Exec("DROP INDEX IF EXISTS " + item.index).Error; err != nil {
			return err
		}
		if err := db.Exec(fmt.Sprintf("CREATE UNIQUE INDEX %s ON %s (bucket_start_ms, access_key_id, channel_id, group_id, credential_id, model, client_model)", item.index, item.table)).Error; err != nil {
			return err
		}
	}
	return Validate0028(db)
}

func Validate0028(db *gorm.DB) error {
	if db == nil || db.Dialector == nil || !strings.EqualFold(db.Dialector.Name(), "sqlite") {
		return fmt.Errorf("validate usage client model: unsupported or nil database")
	}
	for _, table := range []string{"usage_aggregation_journal", "usage_attempt_aggregation_journals", "usage_stats", "usage_attempt_stats"} {
		if !db.Migrator().HasColumn(table, "client_model") {
			return fmt.Errorf("usage client model column %s.client_model is missing", table)
		}
	}
	for _, item := range []struct{ table, index string }{{"usage_stats", "idx_usage_stats_identity"}, {"usage_attempt_stats", "idx_usage_attempt_stats_identity"}} {
		var indexes []struct {
			Name   string
			Unique int
		}
		if err := db.Raw("PRAGMA index_list(" + item.table + ")").Scan(&indexes).Error; err != nil {
			return err
		}
		unique := false
		for _, index := range indexes {
			if index.Name == item.index && index.Unique == 1 {
				unique = true
			}
		}
		if !unique {
			return fmt.Errorf("usage client model index %s is missing or not unique", item.index)
		}
		var columns []struct{ Name string }
		if err := db.Raw("PRAGMA index_info(" + item.index + ")").Scan(&columns).Error; err != nil {
			return err
		}
		want := []string{"bucket_start_ms", "access_key_id", "channel_id", "group_id", "credential_id", "model", "client_model"}
		if len(columns) != len(want) {
			return fmt.Errorf("usage client model index %s has invalid columns", item.index)
		}
		for i, name := range want {
			if columns[i].Name != name {
				return fmt.Errorf("usage client model index %s has invalid columns", item.index)
			}
		}
	}
	return nil
}
