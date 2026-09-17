package migrations

import (
	"fmt"
	"strings"

	"gorm.io/gorm"
)

const ID0018 = "0018_usage_journal_bucket_index"

const (
	usageJournalTable0018 = "usage_aggregation_journal"
	usageJournalIndex0018 = "idx_usage_aggregation_journal_bucket_start"
)

// Up0018 adds the leading bucket_start_ms index used by retention page-selects.
// The existing pending_bucket index leads with `applied` and cannot serve the
// bucket_start_ms range scan + ordering performed by usage journal cleanup.
func Up0018(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("usage journal bucket index migration: database is nil")
	}
	if !db.Migrator().HasTable(usageJournalTable0018) {
		return fmt.Errorf("usage journal bucket index migration: table %q is missing", usageJournalTable0018)
	}
	if !db.Migrator().HasIndex(usageJournalTable0018, usageJournalIndex0018) {
		if err := db.Exec(
			"CREATE INDEX idx_usage_aggregation_journal_bucket_start " +
				"ON usage_aggregation_journal (bucket_start_ms, request_id)",
		).Error; err != nil {
			return fmt.Errorf("create usage journal bucket index: %w", err)
		}
	}
	return Validate0018(db)
}

// Validate0018 confirms the bucket_start_ms index contract.
func Validate0018(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("validate usage journal bucket index: database is nil")
	}
	if !db.Migrator().HasTable(usageJournalTable0018) {
		return fmt.Errorf("usage journal table %q is missing", usageJournalTable0018)
	}
	if !db.Migrator().HasIndex(usageJournalTable0018, usageJournalIndex0018) {
		return fmt.Errorf("usage journal index %q is missing", usageJournalIndex0018)
	}
	switch strings.ToLower(db.Dialector.Name()) {
	case "sqlite":
		var columns []struct {
			Name string
			Key  int
		}
		if err := db.Raw("PRAGMA index_xinfo('idx_usage_aggregation_journal_bucket_start')").Scan(&columns).Error; err != nil {
			return fmt.Errorf("inspect usage journal bucket index columns: %w", err)
		}
		var names []string
		for _, column := range columns {
			if column.Key == 1 {
				names = append(names, column.Name)
			}
		}
		want := []string{"bucket_start_ms", "request_id"}
		if len(names) < len(want) {
			return fmt.Errorf("usage journal bucket index columns = %v, want prefix %v", names, want)
		}
		for index := range want {
			if !strings.EqualFold(names[index], want[index]) {
				return fmt.Errorf("usage journal bucket index columns = %v, want prefix %v", names, want)
			}
		}
		return nil
	case "postgres", "postgresql":
		var index struct {
			Unique     bool
			Definition string
		}
		if err := db.Raw(`
			SELECT i.indisunique AS unique, pg_get_indexdef(i.indexrelid) AS definition
			FROM pg_class AS table_class
			JOIN pg_index AS i ON i.indrelid = table_class.oid
			JOIN pg_class AS index_class ON index_class.oid = i.indexrelid
			WHERE table_class.relname = ? AND index_class.relname = ?
		`, usageJournalTable0018, usageJournalIndex0018).Scan(&index).Error; err != nil {
			return fmt.Errorf("inspect usage journal bucket index: %w", err)
		}
		if index.Definition == "" {
			return fmt.Errorf("usage journal bucket index %q is missing", usageJournalIndex0018)
		}
		if index.Unique {
			return fmt.Errorf("usage journal bucket index is unique")
		}
		normalized := strings.Join(strings.Fields(strings.ToLower(index.Definition)), " ")
		if !strings.Contains(normalized, "(bucket_start_ms, request_id)") {
			return fmt.Errorf("usage journal bucket index definition = %q", index.Definition)
		}
		return nil
	default:
		return fmt.Errorf("validate usage journal bucket index: unsupported database driver %q", db.Dialector.Name())
	}
}
