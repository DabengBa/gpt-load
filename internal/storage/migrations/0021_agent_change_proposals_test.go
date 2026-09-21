package migrations_test

import (
	"bytes"
	"testing"

	"gpt-load/internal/storage/migrations"
	"gpt-load/internal/storage/models"
)

func TestAgentChangeProposalMigrationFreshUpgradeAndRepeatPreserveBinding(t *testing.T) {
	db := openInitialTestDatabase(t)
	if err := migrations.Up0001(db); err != nil {
		t.Fatalf("Up0001() error = %v", err)
	}
	if err := migrations.Up0021(db); err != nil {
		t.Fatalf("first Up0021() error = %v", err)
	}
	if err := migrations.Validate0021(db); err != nil {
		t.Fatalf("first Validate0021() error = %v", err)
	}
	if !db.Migrator().HasTable("agent_change_proposals") ||
		!db.Migrator().HasColumn("control_operations", "proposal_id") ||
		!db.Migrator().HasIndex("control_operations", "idx_control_operations_proposal_id") {
		t.Fatal("proposal migration did not install the durable binding schema")
	}

	proposalID := "11111111-1111-4111-8111-111111111111"
	first := models.ControlOperation{
		OperationID:        "22222222-2222-4222-8222-222222222222",
		IdempotencyKey:     "33333333-3333-4333-8333-333333333333",
		DigestVersion:      1,
		RequestDigest:      bytes.Repeat([]byte{1}, 32),
		OperationKind:      "model_route_schedule_apply",
		ResourceIdentity:   "proposal:" + proposalID,
		ProposalID:         &proposalID,
		CanonicalResult:    []byte(`{"proposal_id":"11111111-1111-4111-8111-111111111111"}`),
		RequiredStages:     models.JSON([]byte(`["db_committed","snapshot_published","completed"]`)),
		LastCompletedStage: "db_committed",
		CreatedAtMS:        1,
		UpdatedAtMS:        1,
	}
	if err := db.Create(&first).Error; err != nil {
		t.Fatalf("insert first proposal binding: %v", err)
	}
	duplicate := first
	duplicate.CommitSequence = 0
	duplicate.OperationID = "44444444-4444-4444-8444-444444444444"
	duplicate.IdempotencyKey = "55555555-5555-4555-8555-555555555555"
	if err := db.Create(&duplicate).Error; err == nil {
		t.Fatal("proposal binding unique index accepted duplicate proposal_id")
	}

	if err := migrations.Up0021(db); err != nil {
		t.Fatalf("repeat Up0021() error = %v", err)
	}
	if err := migrations.Validate0021(db); err != nil {
		t.Fatalf("repeat Validate0021() error = %v", err)
	}
	var persisted models.ControlOperation
	if err := db.Where("proposal_id = ?", proposalID).First(&persisted).Error; err != nil {
		t.Fatalf("load binding after repeat migration: %v", err)
	}
	if persisted.OperationID != first.OperationID {
		t.Fatalf("binding changed after repeat migration: got %q want %q", persisted.OperationID, first.OperationID)
	}
}
