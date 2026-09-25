package migrations_test

import (
	"testing"

	"gpt-load/internal/storage/migrations"
	"gpt-load/internal/storage/models"
)

func TestBillingFailureCategoryMigrationPreservesAttemptsAndAcceptsBilling(t *testing.T) {
	t.Parallel()
	db := openInitialTestDatabase(t)
	if err := migrations.Up0001(db); err != nil {
		t.Fatalf("Up0001() error = %v", err)
	}
	if err := migrations.Up0006(db); err != nil {
		t.Fatalf("Up0006() error = %v", err)
	}
	request := models.RequestLog{
		ID: "00000000-0000-4000-8000-000000000023", CompletedAtMS: 1000,
		AccessKeyID: 1, Protocol: "openai-completions", ClientModel: "gpt-test",
		UpstreamModel: "gpt-test", ModelConsistency: "not_applicable",
		Status: "error", StatusCode: 402, DurationMs: 10, ErrorSummary: "billing",
		UsageState: "not_applicable", CostState: "not_applicable",
		PricingCompleteness: "not_applicable",
	}
	if err := db.Omit("AffinityKey", "ContinuityHit", "AffinitySource", "AffinityState").Create(&request).Error; err != nil {
		t.Fatalf("create request log: %v", err)
	}
	attempt := models.RequestLogAttempt{
		RequestID: request.ID, Sequence: 1, CompletedAtMS: 1000,
		GroupID: 1, GroupName: "group", ChannelID: "openai", CredentialID: 1,
		StatusCode: 429, DurationMs: 5, FailureCategory: "rate_limited",
		FailureOrigin: "upstream", FailureScope: "credential",
		RetryDirective: "next_candidate", Effect: "cooldown_credential",
		RuleID: "upstream.rate_limited",
		Action: "cooldown_credential", ErrorSummary: "limited",
	}
	if err := db.Create(&attempt).Error; err != nil {
		t.Fatalf("create pre-migration attempt: %v", err)
	}

	if err := migrations.Up0023(db); err != nil {
		t.Fatalf("Up0023() error = %v", err)
	}
	if err := migrations.Validate0023(db); err != nil {
		t.Fatalf("Validate0023() error = %v", err)
	}

	var preserved models.RequestLogAttempt
	if err := db.Take(&preserved, "request_id = ? AND sequence = 1", request.ID).Error; err != nil {
		t.Fatalf("read preserved attempt: %v", err)
	}
	if preserved.FailureCategory != "rate_limited" || preserved.RuleID != "upstream.rate_limited" ||
		preserved.FailureScope != "credential" || preserved.Effect != "cooldown_credential" {
		t.Fatalf("preserved attempt = %#v", preserved)
	}

	billing := models.RequestLogAttempt{
		RequestID: request.ID, Sequence: 2, CompletedAtMS: 1001,
		GroupID: 1, GroupName: "group", ChannelID: "openai", CredentialID: 1,
		StatusCode: 402, DurationMs: 6, FailureCategory: "billing",
		FailureOrigin: "upstream", FailureScope: "credential",
		RetryDirective: "next_candidate", Effect: "cooldown_credential",
		RuleID: "billing.insufficient_balance",
		Action: "cooldown_credential", ErrorSummary: "insufficient balance",
	}
	if err := db.Create(&billing).Error; err != nil {
		t.Fatalf("create billing attempt: %v", err)
	}
	invalid := billing
	invalid.Sequence = 3
	invalid.FailureCategory = "made_up_category"
	if err := db.Create(&invalid).Error; err == nil {
		t.Fatal("migration accepted an invalid failure category")
	}

	if err := migrations.Up0023(db); err != nil {
		t.Fatalf("repeated Up0023() error = %v", err)
	}
}

func TestBillingFailureCategoryMigrationKeepsIndexesAndForeignKey(t *testing.T) {
	t.Parallel()
	db := openInitialTestDatabase(t)
	if err := migrations.Up0001(db); err != nil {
		t.Fatal(err)
	}
	if err := migrations.Up0006(db); err != nil {
		t.Fatal(err)
	}
	if err := migrations.Up0023(db); err != nil {
		t.Fatal(err)
	}
	for _, index := range []string{
		"idx_request_log_attempts_group_completed_request",
		"idx_request_log_attempts_channel_completed_request",
		"idx_request_log_attempts_credential_completed_request",
		"idx_request_log_attempts_model_completed_request",
		"idx_request_log_attempts_status_completed_request",
		"idx_request_log_attempts_failure_completed_request",
		"idx_request_log_attempts_error_completed_request",
	} {
		if !db.Migrator().HasIndex("request_log_attempts", index) {
			t.Fatalf("request_log_attempts index %q is missing", index)
		}
	}
	var foreignKeys []struct {
		Table    string
		From     string
		To       string
		OnDelete string `gorm:"column:on_delete"`
	}
	if err := db.Raw("PRAGMA foreign_key_list('request_log_attempts')").Scan(&foreignKeys).Error; err != nil {
		t.Fatal(err)
	}
	if len(foreignKeys) != 1 || foreignKeys[0].Table != "request_logs" ||
		foreignKeys[0].From != "request_id" || foreignKeys[0].To != "id" ||
		foreignKeys[0].OnDelete != "CASCADE" {
		t.Fatalf("request_log_attempts foreign keys = %#v", foreignKeys)
	}
}
