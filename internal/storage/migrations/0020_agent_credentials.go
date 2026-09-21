package migrations

import (
	"fmt"
	"strings"

	"gorm.io/gorm"

	"gpt-load/internal/storage/models"
)

const ID0020 = "0020_agent_credentials"

const (
	agentCredentialTable0020  = "agent_credentials"
	agentCredentialUnique0020 = "idx_agent_credentials_secret_hash"
)

var requiredColumns0020 = []string{
	"id",
	"name",
	"secret_hash",
	"scopes",
	"status",
	"expires_at_ms",
	"disabled_at_ms",
	"created_at_ms",
	"updated_at_ms",
}

// Up0020 installs the independently issued Agent machine credential ledger.
// Only the HMAC digest of the Bearer secret is stored here.
func Up0020(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("agent credential migration: database is nil")
	}
	if err := validateExistingTable0020(db); err != nil {
		return err
	}
	if !db.Migrator().HasTable(agentCredentialTable0020) {
		if err := db.AutoMigrate(&models.AgentCredential{}); err != nil {
			return fmt.Errorf("create agent credential schema: %w", err)
		}
	}
	return Validate0020(db)
}

// validateExistingTable0020 reports a partially created schema before
// AutoMigrate runs, so an interrupted migration is not silently accepted.
func validateExistingTable0020(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("validate agent credential schema: database is nil")
	}
	if !db.Migrator().HasTable(agentCredentialTable0020) {
		return nil
	}
	return validateColumns0020(db)
}

// Validate0020 confirms the complete agent credential schema contract.
func Validate0020(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("validate agent credential schema: database is nil")
	}
	if !db.Migrator().HasTable(agentCredentialTable0020) {
		return fmt.Errorf("agent credential table %q is missing", agentCredentialTable0020)
	}
	if err := validateColumns0020(db); err != nil {
		return err
	}
	if !db.Migrator().HasIndex(agentCredentialTable0020, agentCredentialUnique0020) {
		return fmt.Errorf("agent credential unique index %q is missing", agentCredentialUnique0020)
	}
	return validateAgentCredentialSecretHash0020(db)
}

func validateColumns0020(db *gorm.DB) error {
	for _, column := range requiredColumns0020 {
		if !db.Migrator().HasColumn(agentCredentialTable0020, column) {
			return fmt.Errorf("agent credential column %s.%s is missing", agentCredentialTable0020, column)
		}
	}
	return nil
}

func validateAgentCredentialSecretHash0020(db *gorm.DB) error {
	columns, err := db.Migrator().ColumnTypes(agentCredentialTable0020)
	if err != nil {
		return fmt.Errorf("inspect agent credential columns: %w", err)
	}
	for _, column := range columns {
		if !strings.EqualFold(column.Name(), "secret_hash") {
			continue
		}
		if nullable, known := column.Nullable(); known && nullable {
			return fmt.Errorf("agent credential column %s is nullable", "secret_hash")
		}
		typeName := strings.ToLower(column.DatabaseTypeName())
		if !strings.Contains(typeName, "char") && !strings.Contains(typeName, "text") {
			return fmt.Errorf("agent credential column %s type %q is not textual", "secret_hash", typeName)
		}
		return nil
	}
	return fmt.Errorf("agent credential column %s is missing", "secret_hash")
}
