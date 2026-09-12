package migrations_test

import (
	"strings"
	"testing"

	"gpt-load/internal/storage/migrations"
	"gpt-load/internal/storage/models"
)

func legacyAffinityRequestLog(id string) models.RequestLog {
	return models.RequestLog{
		ID: id, CompletedAtMS: 1000,
		AccessKeyID: 1, Protocol: "openai-completions", ClientModel: "gpt-test",
		UpstreamModel: "gpt-test", ModelConsistency: "not_applicable",
		Status: "success", StatusCode: 200, DurationMs: 10,
		UsageState: "not_applicable", CostState: "not_applicable",
		PricingCompleteness: "not_applicable",
	}
}

func Test0014AffinityObservabilityAddsColumnsAndBackfillsLegacyRows(t *testing.T) {
	db := openInitialTestDatabase(t)
	if err := migrations.Up0001(db); err != nil {
		t.Fatalf("Up0001() error = %v", err)
	}

	legacy := legacyAffinityRequestLog("00000000-0000-4000-8000-000000000013")
	// 本插入特意针对 0014 之前的 schema，因此亲和观测列像其他迁移前夹具一样被省略。
	if err := db.Omit("ContinuityHit", "AffinitySource", "AffinityState").Create(&legacy).Error; err != nil {
		t.Fatalf("create pre-0014 request log: %v", err)
	}

	if err := migrations.Up0014(db); err != nil {
		t.Fatalf("Up0014() error = %v", err)
	}
	if err := migrations.Validate0014(db); err != nil {
		t.Fatalf("Validate0014() error = %v", err)
	}
	for _, column := range []string{"continuity_hit", "affinity_source", "affinity_state"} {
		if !db.Migrator().HasColumn("request_logs", column) {
			t.Fatalf("request_logs.%s is missing", column)
		}
	}

	var upgraded models.RequestLog
	if err := db.Take(&upgraded, "id = ?", legacy.ID).Error; err != nil {
		t.Fatalf("read upgraded request log: %v", err)
	}
	if upgraded.ContinuityHit || upgraded.AffinitySource != "none" || upgraded.AffinityState != "no_signal" {
		t.Fatalf("upgraded legacy row affinity = %#v, want bounded zero values", upgraded)
	}

	fresh := legacyAffinityRequestLog("00000000-0000-4000-8000-000000000014")
	if err := db.Create(&fresh).Error; err != nil {
		t.Fatalf("create post-0014 request log: %v", err)
	}
	var stored models.RequestLog
	if err := db.Take(&stored, "id = ?", fresh.ID).Error; err != nil {
		t.Fatalf("read post-0014 request log: %v", err)
	}
	if stored.ContinuityHit || stored.AffinitySource != "none" || stored.AffinityState != "no_signal" {
		t.Fatalf("post-0014 row affinity = %#v, want bounded zero values", stored)
	}
}

func Test0014AffinityObservabilityIsRecoverable(t *testing.T) {
	db := openMigratedInitialTestDatabase(t)

	if err := migrations.Up0014(db); err != nil {
		t.Fatalf("second Up0014() error = %v", err)
	}
	if err := migrations.Validate0014(db); err != nil {
		t.Fatalf("Validate0014() error = %v", err)
	}
}

func Test0014ValidateRecoverableRejectsMissingTable(t *testing.T) {
	db := openInitialTestDatabase(t)

	if err := migrations.ValidateRecoverable0014(db); err == nil ||
		!strings.Contains(err.Error(), "is missing") {
		t.Fatalf("ValidateRecoverable0014() error = %v, want missing-table rejection", err)
	}
}
