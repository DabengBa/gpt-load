package migrations

import (
	"fmt"

	"gorm.io/gorm"
)

const ID0029 = "0029_group_site_info"

func Up0029(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("site info migration: nil database")
	}
	statements := []string{
		`CREATE TABLE IF NOT EXISTS group_site_configs (group_id integer PRIMARY KEY, config_version integer NOT NULL DEFAULT 1, user_id varchar(128) NOT NULL DEFAULT '', access_token text NOT NULL DEFAULT '', site_root_url varchar(512) NOT NULL DEFAULT '', site_name varchar(255) NOT NULL DEFAULT '', balance_quota integer, quota_per_unit real, remaining_amount real, currency_symbol varchar(16) NOT NULL DEFAULT '$', user_group varchar(128) NOT NULL DEFAULT '', token_group varchar(128) NOT NULL DEFAULT '', effective_group varchar(128) NOT NULL DEFAULT '', group_ratio real, status varchar(32) NOT NULL DEFAULT 'unconfigured', last_error text NOT NULL DEFAULT '', fetched_at_ms integer, last_attempt_at_ms integer, updated_at_ms integer NOT NULL, FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE)`,
		`CREATE TABLE IF NOT EXISTS site_announcements (id integer PRIMARY KEY AUTOINCREMENT, group_id integer NOT NULL, site_root_url varchar(512) NOT NULL, site_name varchar(255) NOT NULL DEFAULT '', dedup_key varchar(191) NOT NULL, content text NOT NULL, extra text NOT NULL DEFAULT '', level varchar(32) NOT NULL DEFAULT 'default', source varchar(32) NOT NULL DEFAULT 'announcement', event_at_ms integer NOT NULL, fetched_at_ms integer NOT NULL, FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE)`,
		`CREATE INDEX IF NOT EXISTS idx_group_site_configs_fetched_at_ms ON group_site_configs(fetched_at_ms)`,
		`CREATE INDEX IF NOT EXISTS idx_group_site_configs_last_attempt_at_ms ON group_site_configs(last_attempt_at_ms)`,
		`CREATE INDEX IF NOT EXISTS idx_site_announcements_group_id ON site_announcements(group_id)`,
		`CREATE INDEX IF NOT EXISTS idx_site_announcements_site_root_url ON site_announcements(site_root_url)`,
		`CREATE INDEX IF NOT EXISTS idx_site_announcements_event_at_ms ON site_announcements(event_at_ms)`,
		`CREATE UNIQUE INDEX IF NOT EXISTS idx_site_announcements_group_dedup ON site_announcements(group_id, dedup_key)`,
	}
	for _, statement := range statements {
		if err := db.Exec(statement).Error; err != nil {
			return fmt.Errorf("create site info schema: %w", err)
		}
	}
	return Validate0029(db)
}

func Validate0029(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("validate site info: nil database")
	}
	tables := map[string][]string{
		"group_site_configs": {"group_id", "config_version", "user_id", "access_token", "site_root_url", "site_name", "balance_quota", "quota_per_unit", "remaining_amount", "currency_symbol", "user_group", "token_group", "effective_group", "group_ratio", "status", "last_error", "fetched_at_ms", "last_attempt_at_ms", "updated_at_ms"},
		"site_announcements": {"id", "group_id", "site_root_url", "site_name", "dedup_key", "content", "extra", "level", "source", "event_at_ms", "fetched_at_ms"},
	}
	for table, columns := range tables {
		if !db.Migrator().HasTable(table) {
			return fmt.Errorf("site info table %s missing", table)
		}
		for _, column := range columns {
			if !db.Migrator().HasColumn(table, column) {
				return fmt.Errorf("site info column %s.%s missing", table, column)
			}
		}
		var keys []struct {
			Table    string
			From     string
			To       string
			OnDelete string `gorm:"column:on_delete"`
		}
		if err := db.Raw("PRAGMA foreign_key_list('" + table + "')").Scan(&keys).Error; err != nil {
			return err
		}
		found := false
		for _, key := range keys {
			if key.Table == "groups" && key.From == "group_id" && key.To == "id" && key.OnDelete == "CASCADE" {
				found = true
			}
		}
		if !found {
			return fmt.Errorf("site info foreign key %s.group_id missing", table)
		}
	}
	for table, indexes := range map[string][]string{"group_site_configs": {"idx_group_site_configs_fetched_at_ms", "idx_group_site_configs_last_attempt_at_ms"}, "site_announcements": {"idx_site_announcements_group_id", "idx_site_announcements_site_root_url", "idx_site_announcements_event_at_ms", "idx_site_announcements_group_dedup"}} {
		for _, index := range indexes {
			if !db.Migrator().HasIndex(table, index) {
				return fmt.Errorf("site info index %s missing", index)
			}
		}
	}
	var indexes []struct {
		Name    string
		Unique  int
		Partial int
	}
	if err := db.Raw("PRAGMA index_list('site_announcements')").Scan(&indexes).Error; err != nil {
		return err
	}
	validUnique := false
	for _, index := range indexes {
		if index.Name == "idx_site_announcements_group_dedup" && index.Unique == 1 && index.Partial == 0 {
			validUnique = true
		}
	}
	if !validUnique {
		return fmt.Errorf("site announcement dedup index must be unique and non-partial")
	}
	var columns []struct {
		Seqno int
		Name  string
	}
	if err := db.Raw("PRAGMA index_info('idx_site_announcements_group_dedup')").Scan(&columns).Error; err != nil {
		return err
	}
	if len(columns) != 2 || columns[0].Name != "group_id" || columns[1].Name != "dedup_key" {
		return fmt.Errorf("site announcement dedup index has incorrect columns")
	}
	return nil
}
