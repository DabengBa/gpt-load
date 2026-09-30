package migrations

import (
	"fmt"

	"gorm.io/gorm"
)

const ID0013 = "0013_usage_attempt_stats"

type usageAttemptAggregationJournal0013 struct {
	RequestID     string `gorm:"column:request_id;type:varchar(36);primaryKey;not null"`
	Sequence      int    `gorm:"primaryKey;not null;check:chk_usage_attempt_journal_sequence,sequence > 0"`
	BucketStartMS int64  `gorm:"column:bucket_start_ms;not null;check:chk_usage_attempt_journal_bucket,bucket_start_ms >= 0;index:idx_usage_attempt_journal_pending_bucket,priority:2"`
	AccessKeyID   uint   `gorm:"not null"`
	GroupID       uint   `gorm:"not null;check:chk_usage_attempt_journal_group,group_id > 0"`
	ChannelID     string `gorm:"type:varchar(64);not null;default:''"`
	CredentialID  uint   `gorm:"not null;check:chk_usage_attempt_journal_credential,credential_id > 0"`
	Model         string `gorm:"type:varchar(255);not null"`
	AttemptCount  int64  `gorm:"not null;check:chk_usage_attempt_journal_attempt_count,attempt_count = 1"`
	FailureCount  int64  `gorm:"not null;check:chk_usage_attempt_journal_failure_count,failure_count >= 0 AND failure_count <= attempt_count"`
	Applied       bool   `gorm:"not null;default:false;check:chk_usage_attempt_journal_applied,applied IN (TRUE, FALSE);index:idx_usage_attempt_journal_pending_bucket,priority:1"`
}

func (usageAttemptAggregationJournal0013) TableName() string {
	return "usage_attempt_aggregation_journals"
}

type usageAttemptStat0013 struct {
	ID            uint   `gorm:"primaryKey;autoIncrement"`
	BucketStartMS int64  `gorm:"column:bucket_start_ms;not null;check:chk_usage_attempt_stat_bucket,bucket_start_ms >= 0;uniqueIndex:idx_usage_attempt_stats_identity,priority:1"`
	AccessKeyID   uint   `gorm:"not null;uniqueIndex:idx_usage_attempt_stats_identity,priority:2"`
	ChannelID     string `gorm:"type:varchar(64);not null;uniqueIndex:idx_usage_attempt_stats_identity,priority:3"`
	GroupID       uint   `gorm:"not null;uniqueIndex:idx_usage_attempt_stats_identity,priority:4"`
	CredentialID  uint   `gorm:"not null;uniqueIndex:idx_usage_attempt_stats_identity,priority:5"`
	Model         string `gorm:"type:varchar(255);not null;uniqueIndex:idx_usage_attempt_stats_identity,priority:6"`
	AttemptCount  int64  `gorm:"not null;default:0;check:chk_usage_attempt_stat_attempt_count,attempt_count >= 0"`
	FailureCount  int64  `gorm:"not null;default:0;check:chk_usage_attempt_stat_failure_count,failure_count >= 0;check:chk_usage_attempt_stat_failure_le_attempt, failure_count <= attempt_count"`
}

func (usageAttemptStat0013) TableName() string {
	return "usage_attempt_stats"
}

var usageAttemptStatsTables0013 = []struct {
	name  string
	model any
}{
	{"usage_attempt_aggregation_journals", &usageAttemptAggregationJournal0013{}},
	{"usage_attempt_stats", &usageAttemptStat0013{}},
}

// Up0013 adds durable route-attempt aggregates without changing request-level
// usage statistics.
func Up0013(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("usage attempt stats migration: database is nil")
	}
	for _, table := range usageAttemptStatsTables0013 {
		if err := db.AutoMigrate(table.model); err != nil {
			return fmt.Errorf("create %s: %w", table.name, err)
		}
	}
	return Validate0013(db)
}

var requiredColumns0013 = map[string][]string{
	"usage_attempt_aggregation_journals": {
		"request_id", "sequence", "bucket_start_ms", "access_key_id", "group_id", "channel_id",
		"credential_id", "model", "attempt_count", "failure_count", "applied",
	},
	"usage_attempt_stats": {
		"id", "bucket_start_ms", "access_key_id", "channel_id", "group_id", "credential_id",
		"model", "attempt_count", "failure_count",
	},
}

// Validate0013 verifies that both durable route-attempt tables and their
// required columns are present.
func Validate0013(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("validate usage attempt stats: database is nil")
	}
	for _, table := range usageAttemptStatsTables0013 {
		if !db.Migrator().HasTable(table.name) {
			return fmt.Errorf("usage attempt stats table %q is missing", table.name)
		}
		for _, column := range requiredColumns0013[table.name] {
			if !db.Migrator().HasColumn(table.name, column) {
				return fmt.Errorf("usage attempt stats column %s.%s is missing", table.name, column)
			}
		}
	}
	return nil
}
