package migrations

import (
	"fmt"

	"gorm.io/gorm"
)

const ID0010 = "0010_single_credential_per_group"

const credentialGroupUniqueIndex0010 = "idx_credentials_group_id_unique"

type credentialGroupUniqueConstraint0010 struct {
	GroupID uint `gorm:"column:group_id;uniqueIndex:idx_credentials_group_id_unique"`
}

func (credentialGroupUniqueConstraint0010) TableName() string { return "credentials" }

// Up0010 adds the database-level invariant that the control API enforces:
// one upstream credential can exist in a group at most once.
func Up0010(db *gorm.DB) error {
	if db == nil || !db.Migrator().HasTable("credentials") {
		return fmt.Errorf("single credential constraint: credentials table is missing")
	}
	var duplicate struct {
		GroupID uint
		Count   int64
	}
	if err := db.Table("credentials").
		Select("group_id, COUNT(*) AS count").
		Group("group_id").Having("COUNT(*) > 1").Take(&duplicate).Error; err == nil {
		return fmt.Errorf("single credential constraint: group %d has %d credentials", duplicate.GroupID, duplicate.Count)
	} else if err != nil && err != gorm.ErrRecordNotFound {
		return fmt.Errorf("check duplicate credentials: %w", err)
	}
	if !db.Migrator().HasIndex("credentials", credentialGroupUniqueIndex0010) {
		if err := db.Migrator().CreateIndex(&credentialGroupUniqueConstraint0010{}, "GroupID"); err != nil {
			return fmt.Errorf("create %s: %w", credentialGroupUniqueIndex0010, err)
		}
	}
	return Validate0010(db)
}

func validateSingleCredentialConstraint0010(db *gorm.DB) error {
	if db == nil || !db.Migrator().HasTable("credentials") {
		return fmt.Errorf("validate single credential constraint: credentials table is missing")
	}
	return nil
}

func Validate0010(db *gorm.DB) error {
	if err := validateSingleCredentialConstraint0010(db); err != nil {
		return err
	}
	if !db.Migrator().HasIndex("credentials", credentialGroupUniqueIndex0010) {
		return fmt.Errorf("credentials.%s is missing", credentialGroupUniqueIndex0010)
	}
	return nil
}
