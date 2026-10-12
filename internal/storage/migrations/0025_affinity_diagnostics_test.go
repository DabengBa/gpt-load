package migrations_test

import (
	"gpt-load/internal/storage/migrations"
	"testing"
)

func TestAffinityDiagnosticsMigration(t *testing.T) {
	db := openInitialTestDatabase(t)
	if err := db.Exec("CREATE TABLE request_logs (id TEXT PRIMARY KEY)").Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Exec("INSERT INTO request_logs (id) VALUES ('historical')").Error; err != nil {
		t.Fatal(err)
	}
	if err := migrations.Up0025(db); err != nil {
		t.Fatal(err)
	}
	if err := migrations.Up0025(db); err != nil {
		t.Fatal(err)
	}
	if err := migrations.Validate0025(db); err != nil {
		t.Fatal(err)
	}
	var row struct {
		ResolvedAffinityCredentialID uint
		FirstAttemptCredentialID     uint
		AffinityHitAssignmentReached bool
	}
	if err := db.Table("request_logs").Take(&row).Error; err != nil {
		t.Fatal(err)
	}
	if row.ResolvedAffinityCredentialID != 0 || row.FirstAttemptCredentialID != 0 || row.AffinityHitAssignmentReached {
		t.Fatalf("historical defaults = %+v", row)
	}
	if err := db.Exec("UPDATE request_logs SET resolved_affinity_credential_id = 41, first_attempt_credential_id = 42, affinity_hit_assignment_reached = true").Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Table("request_logs").Take(&row).Error; err != nil {
		t.Fatal(err)
	}
	if row.ResolvedAffinityCredentialID != 41 || row.FirstAttemptCredentialID != 42 || !row.AffinityHitAssignmentReached {
		t.Fatalf("persisted diagnostics = %+v", row)
	}
}
