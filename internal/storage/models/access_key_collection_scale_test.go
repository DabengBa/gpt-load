package models_test

import (
	"fmt"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	gormlogger "gorm.io/gorm/logger"

	"gpt-load/internal/storage/models"
)

// accessKeyCollectionProbeRow mirrors control.accessKeyCollectionRow; the
// benchmark reproduces the exact Select shape of captureAccessKeyCollectionRecords
// so the numbers reflect the production collection path.
type accessKeyCollectionProbeRow struct {
	PriceMultiplierMicros *int64
	ID                    uint
	Name                  string
	KeySuffix             string
	Status                string
	Filters               models.JSON
	RPMLimit              int64
	ExpiresAtMS           *int64
	CreatedAtMS           int64
	UpdatedAtMS           int64
	LastRequestAtMS       *int64
}

func openProbeDB(t testing.TB, name string) *gorm.DB {
	t.Helper()
	path := filepath.Join(t.TempDir(), name+".db")
	dsn := fmt.Sprintf("file:%s?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)", filepath.ToSlash(path))
	db, err := gorm.Open(sqlite.Open(dsn), &gorm.Config{
		Logger: gormlogger.Default.LogMode(gormlogger.Silent),
	})
	if err != nil {
		t.Fatalf("open probe db: %v", err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("probe db pool: %v", err)
	}
	sqlDB.SetMaxOpenConns(4)
	sqlDB.SetMaxIdleConns(4)
	t.Cleanup(func() { _ = sqlDB.Close() })
	if err := db.AutoMigrate(
		&models.AccessKey{},
		&models.AccessKeyCostLimitRule{},
		&models.RequestLog{},
	); err != nil {
		t.Fatalf("migrate probe db: %v", err)
	}
	return db
}

func seedAccessKeyScale(t testing.TB, db *gorm.DB, keys int, logs int) {
	t.Helper()
	now := time.Now().UnixMilli()
	const batch = 500
	for start := 0; start < keys; start += batch {
		end := start + batch
		if end > keys {
			end = keys
		}
		rows := make([]models.AccessKey, 0, end-start)
		for i := start; i < end; i++ {
			rows = append(rows, models.AccessKey{
				Name:        fmt.Sprintf("key-%06d", i),
				KeyValue:    fmt.Sprintf("sk-probe-%06d", i),
				KeyHash:     fmt.Sprintf("hash-%06d", i),
				KeySuffix:   "0a1b",
				Status:      "active",
				RPMLimit:    600,
				CreatedAtMS: now,
				UpdatedAtMS: now,
			})
		}
		if err := db.CreateInBatches(rows, batch).Error; err != nil {
			t.Fatalf("seed access keys: %v", err)
		}
	}
	if logs == 0 {
		return
	}
	// RequestLog's column count keeps 400 rows under SQLite's variable limit.
	const logBatch = 400
	for start := 0; start < logs; start += logBatch {
		end := start + logBatch
		if end > logs {
			end = logs
		}
		rows := make([]models.RequestLog, 0, end-start)
		for i := start; i < end; i++ {
			rows = append(rows, models.RequestLog{
				ID:            fmt.Sprintf("req-%08d", i),
				CompletedAtMS: now - int64(logs-i),
				AccessKeyID:   uint(i%keys) + 1,
				GroupID:       1,
				Protocol:      "openai",
				ClientModel:   "gpt-4",
				Status:        "success",
				StatusCode:    200,
				DurationMs:    120,
				ErrorSummary:  "",
			})
		}
		if err := db.CreateInBatches(rows, logBatch).Error; err != nil {
			t.Fatalf("seed request logs: %v", err)
		}
	}
}

// runCollectionCapture reproduces captureAccessKeyCollectionRecords: one
// Select with a per-row correlated MAX probe into request_logs, a second
// cost-rule Find, then the in-memory map/sort the query layer performs.
func runCollectionCapture(b *testing.B, db *gorm.DB) {
	var rows []accessKeyCollectionProbeRow
	var costLimitRows []models.AccessKeyCostLimitRule
	err := db.Transaction(func(tx *gorm.DB) error {
		if err := tx.Model(&models.AccessKey{}).
			Select(
				"access_keys.id", "access_keys.name", "access_keys.key_suffix",
				"access_keys.status", "access_keys.filters", "access_keys.rpm_limit",
				"access_keys.expires_at_ms", "access_keys.price_multiplier_micros",
				"access_keys.created_at_ms", "access_keys.updated_at_ms",
				"(SELECT MAX(request_logs.completed_at_ms) FROM request_logs WHERE request_logs.access_key_id = access_keys.id) AS last_request_at_ms",
			).
			Order("access_keys.id ASC").
			Scan(&rows).Error; err != nil {
			return err
		}
		return tx.Order("access_key_id ASC, CASE WHEN kind = 'total' THEN 0 ELSE 1 END ASC, period_seconds ASC, id ASC").
			Find(&costLimitRows).Error
	})
	if err != nil {
		b.Fatalf("collection capture: %v", err)
	}
	// In-memory phase: group rules, materialize records, sort — the work the
	// query layer performs after the snapshot read.
	rulesByAccessKey := make(map[uint][]models.AccessKeyCostLimitRule, len(costLimitRows))
	for _, row := range costLimitRows {
		rulesByAccessKey[row.AccessKeyID] = append(rulesByAccessKey[row.AccessKeyID], row)
	}
	type record struct {
		id uint
	}
	records := make([]record, 0, len(rows))
	for _, row := range rows {
		_ = rulesByAccessKey[row.ID]
		records = append(records, record{id: row.ID})
	}
	sort.Slice(records, func(i, j int) bool { return records[i].id < records[j].id })
	b.ReportMetric(float64(len(rows)), "rows")
}

// BenchmarkAccessKeyCollectionScale measures the collection capture path at
// the documented typical scale (≤1k keys) and at 10× pressure, against empty
// and large request_logs tables.
func BenchmarkAccessKeyCollectionScale(b *testing.B) {
	scales := []struct {
		name string
		keys int
		logs int
	}{
		{"keys1k_logs0", 1000, 0},
		{"keys1k_logs500k", 1000, 500000},
		{"keys10k_logs500k", 10000, 500000},
	}
	for _, scale := range scales {
		b.Run(scale.name, func(b *testing.B) {
			db := openProbeDB(b, scale.name)
			seedAccessKeyScale(b, db, scale.keys, scale.logs)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				runCollectionCapture(b, db)
			}
		})
	}
}
