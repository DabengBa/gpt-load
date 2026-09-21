package migrations

import (
	"fmt"

	"gorm.io/gorm"

	"gpt-load/internal/storage/models"
)

const ID0021 = "0021_agent_change_proposals"

const (
	agentChangeProposalTable0021      = "agent_change_proposals"
	controlOperationTable0021         = "control_operations"
	controlOperationProposalIDCol0021 = "proposal_id"
	controlOperationProposalIdx0021   = "idx_control_operations_proposal_id"
	agentChangeProposalCreatorIdx0021 = "idx_agent_change_proposal_creator"
	agentChangeProposalStateIdx0021   = "idx_agent_change_proposal_state"
)

var requiredColumns0021 = []string{
	"id",
	"created_at_ms",
	"updated_at_ms",
	"base_snapshot_revision",
	"creator_credential_id",
	"state",
	"updates",
	"approved_runtime_epoch",
	"approved_by",
	"approved_at_ms",
}

// Up0021 installs the immutable Agent change-proposal ledger and the durable
// proposal -> operation binding used to replay an already applied proposal.
func Up0021(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("agent change proposal migration: database is nil")
	}
	if db.Migrator().HasTable(agentChangeProposalTable0021) {
		if err := validateColumns0021(db); err != nil {
			return err
		}
	} else {
		if err := db.AutoMigrate(&models.AgentChangeProposal{}); err != nil {
			return fmt.Errorf("create agent change proposal schema: %w", err)
		}
	}
	if !db.Migrator().HasTable(controlOperationTable0021) {
		return fmt.Errorf(
			"agent change proposal migration: table %q is missing",
			controlOperationTable0021,
		)
	}
	if !db.Migrator().HasColumn(controlOperationTable0021, controlOperationProposalIDCol0021) {
		if err := db.Migrator().AddColumn(&models.ControlOperation{}, "ProposalID"); err != nil {
			return fmt.Errorf("add control operation proposal binding column: %w", err)
		}
	}
	if !db.Migrator().HasIndex(controlOperationTable0021, controlOperationProposalIdx0021) {
		if err := db.Exec(
			"CREATE UNIQUE INDEX " + controlOperationProposalIdx0021 +
				" ON control_operations (proposal_id) WHERE proposal_id IS NOT NULL",
		).Error; err != nil {
			return fmt.Errorf("create control operation proposal binding index: %w", err)
		}
	}
	return Validate0021(db)
}

func validateColumns0021(db *gorm.DB) error {
	for _, column := range requiredColumns0021 {
		if !db.Migrator().HasColumn(agentChangeProposalTable0021, column) {
			return fmt.Errorf("agent change proposal column %s.%s is missing", agentChangeProposalTable0021, column)
		}
	}
	return nil
}

// Validate0021 confirms the proposal ledger and binding contract.
func Validate0021(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("validate agent change proposal schema: database is nil")
	}
	if !db.Migrator().HasTable(agentChangeProposalTable0021) {
		return fmt.Errorf("agent change proposal table %q is missing", agentChangeProposalTable0021)
	}
	if err := validateColumns0021(db); err != nil {
		return err
	}
	for _, index := range []string{agentChangeProposalCreatorIdx0021, agentChangeProposalStateIdx0021} {
		if !db.Migrator().HasIndex(agentChangeProposalTable0021, index) {
			return fmt.Errorf("agent change proposal index %q is missing", index)
		}
	}
	if !db.Migrator().HasColumn(controlOperationTable0021, controlOperationProposalIDCol0021) {
		return fmt.Errorf("control operation column %s is missing", controlOperationProposalIDCol0021)
	}
	if !db.Migrator().HasIndex(controlOperationTable0021, controlOperationProposalIdx0021) {
		return fmt.Errorf("control operation proposal binding index %q is missing", controlOperationProposalIdx0021)
	}
	return nil
}
