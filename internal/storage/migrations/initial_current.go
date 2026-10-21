package migrations

import (
	"fmt"

	"gorm.io/gorm"
)

// ValidateCurrent0001 rejects retired subscription schemas before any pending
// migration is applied, then preserves the ordinary initial-schema checks.
func ValidateCurrent0001(db *gorm.DB) error {
	for _, table := range []string{"credential_stages", "credential_observations", "credential_reset_operations"} {
		if db.Migrator().HasTable(table) {
			return fmt.Errorf("unsupported subscription schema: table %q; a fresh database is required", table)
		}
	}
	for _, column := range []string{"auth_state", "auth_error_code"} {
		if db.Migrator().HasColumn("credentials", column) {
			return fmt.Errorf("unsupported subscription schema: credentials.%s; a fresh database is required", column)
		}
	}
	return Validate0001(db)
}
