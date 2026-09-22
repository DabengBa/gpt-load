package storage

import (
	"bytes"
	"reflect"
	"strings"
	"testing"

	"gorm.io/gorm"

	migrationfiles "gpt-load/internal/storage/migrations"
	"gpt-load/internal/storage/models"
)

func TestMigrationRegistryContainsOrderedMigrations(t *testing.T) {
	wantIDs := []string{
		migrationfiles.ID0001,
		migrationfiles.ID0002,
		migrationfiles.ID0003,
		migrationfiles.ID0004,
		migrationfiles.ID0005,
		migrationfiles.ID0006,
		migrationfiles.ID0007,
		migrationfiles.ID0008,
		migrationfiles.ID0009,
		migrationfiles.ID0010,
		migrationfiles.ID0011,
		migrationfiles.ID0012,
		migrationfiles.ID0013,
		migrationfiles.ID0014,
		migrationfiles.ID0015,
		migrationfiles.ID0016,
		migrationfiles.ID0017,
		migrationfiles.ID0018,
		migrationfiles.ID0019,
		migrationfiles.ID0020,
		migrationfiles.ID0021,
		migrationfiles.ID0022,
	}
	if len(migrations) != len(wantIDs) {
		t.Fatalf("migration registry length = %d, want %d", len(migrations), len(wantIDs))
	}
	for index, entry := range migrations {
		if entry.ID != wantIDs[index] || entry.Up == nil || entry.Validate == nil {
			t.Fatalf("migration registry entry %d = %#v", index, entry)
		}
	}
}

func TestAgentChangeProposalMigrationSurvivesRealFreshUpgradeAndRepeatStart(t *testing.T) {
	fresh := openInternalMigrationTestDatabase(t)
	if err := AutoMigrate(fresh); err != nil {
		t.Fatalf("fresh AutoMigrate() error = %v", err)
	}
	assertAgentChangeProposalBindingSchema(t, fresh)
	insertAgentChangeProposalBinding(t, fresh, "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222")
	if err := AutoMigrate(fresh); err != nil {
		t.Fatalf("repeat AutoMigrate() error = %v", err)
	}
	assertAgentChangeProposalBindingSchema(t, fresh)
	assertAgentChangeProposalBinding(t, fresh, "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222")

	upgrade := openInternalMigrationTestDatabase(t)
	if err := applyMigrationRegistry(upgrade, migrations[:len(migrations)-1]); err != nil {
		t.Fatalf("pre-0021 migration registry error = %v", err)
	}
	if err := applyMigrations(upgrade); err != nil {
		t.Fatalf("upgrade through 0021 error = %v", err)
	}
	assertAgentChangeProposalBindingSchema(t, upgrade)
	insertAgentChangeProposalBinding(t, upgrade, "33333333-3333-4333-8333-333333333333", "44444444-4444-4444-8444-444444444444")
	if err := applyMigrations(upgrade); err != nil {
		t.Fatalf("repeat upgraded migration registry error = %v", err)
	}
	assertAgentChangeProposalBinding(t, upgrade, "33333333-3333-4333-8333-333333333333", "44444444-4444-4444-8444-444444444444")
}

func assertAgentChangeProposalBindingSchema(t *testing.T, db *gorm.DB) {
	t.Helper()
	if !db.Migrator().HasTable("agent_change_proposals") {
		t.Fatal("agent_change_proposals table is missing")
	}
	if !db.Migrator().HasColumn("control_operations", "proposal_id") {
		t.Fatal("control_operations.proposal_id is missing")
	}
	if !db.Migrator().HasIndex("control_operations", "idx_control_operations_proposal_id") {
		t.Fatal("proposal_id unique binding index is missing")
	}
}

func insertAgentChangeProposalBinding(t *testing.T, db *gorm.DB, proposalID, operationID string) {
	t.Helper()
	row := models.ControlOperation{
		OperationID:        operationID,
		IdempotencyKey:     "55555555-5555-4555-8555-555555555555",
		DigestVersion:      1,
		RequestDigest:      bytes.Repeat([]byte{1}, 32),
		OperationKind:      "model_route_schedule_apply",
		ResourceIdentity:   "proposal:" + proposalID,
		ProposalID:         &proposalID,
		CanonicalResult:    []byte(`{"proposal_id":"` + proposalID + `"}`),
		RequiredStages:     models.JSON([]byte(`["db_committed","snapshot_published","completed"]`)),
		LastCompletedStage: "db_committed",
		CreatedAtMS:        1,
		UpdatedAtMS:        1,
	}
	if err := db.Create(&row).Error; err != nil {
		t.Fatalf("insert proposal binding: %v", err)
	}
}

func assertAgentChangeProposalBinding(t *testing.T, db *gorm.DB, proposalID, operationID string) {
	t.Helper()
	var row models.ControlOperation
	if err := db.Where("proposal_id = ?", proposalID).First(&row).Error; err != nil {
		t.Fatalf("load proposal binding: %v", err)
	}
	if row.OperationID != operationID {
		t.Fatalf("proposal binding operation = %q, want %q", row.OperationID, operationID)
	}
}

func TestMigrationRegistryUsesOneOrderedChainForFreshAndExistingDatabases(t *testing.T) {
	entries, calls := testMigrationRegistry()

	fresh := openInternalMigrationTestDatabase(t)
	if err := applyMigrationRegistry(fresh, entries); err != nil {
		t.Fatalf("migrate fresh database: %v", err)
	}
	if !reflect.DeepEqual(*calls, []string{"0001_test", "0002_test"}) {
		t.Fatalf("fresh migration calls = %v, want [0001_test 0002_test]", *calls)
	}

	existing := openInternalMigrationTestDatabase(t)
	if err := existing.AutoMigrate(&schemaMigration{}); err != nil {
		t.Fatalf("create existing migration ledger: %v", err)
	}
	if err := existing.Create(&schemaMigration{ID: entries[0].ID}).Error; err != nil {
		t.Fatalf("record existing migration: %v", err)
	}
	*calls = nil
	if err := applyMigrationRegistry(existing, entries); err != nil {
		t.Fatalf("migrate existing database: %v", err)
	}
	if !reflect.DeepEqual(*calls, []string{"0002_test"}) {
		t.Fatalf("existing migration calls = %v, want [0002_test]", *calls)
	}
}

func TestApplyMigrationRegistryRejectsOutOfOrderEntries(t *testing.T) {
	entries, _ := testMigrationRegistry()
	entries[0], entries[1] = entries[1], entries[0]

	err := applyMigrationRegistry(openInternalMigrationTestDatabase(t), entries)
	if err == nil || !strings.Contains(err.Error(), "migration registry entry 1") {
		t.Fatalf("applyMigrationRegistry() error = %v, want out-of-order registry rejection", err)
	}
}

func testMigrationRegistry() ([]migration, *[]string) {
	calls := make([]string, 0, 2)
	entry := func(id string) migration {
		return migration{
			ID: id,
			Up: func(*gorm.DB) error {
				calls = append(calls, id)
				return nil
			},
			Validate: func(*gorm.DB) error { return nil },
		}
	}
	return []migration{entry("0001_test"), entry("0002_test")}, &calls
}
