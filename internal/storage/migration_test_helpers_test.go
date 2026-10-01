package storage

import (
	"testing"

	"gorm.io/gorm"
)

func openInternalMigrationTestDatabase(t *testing.T) *gorm.DB {
	t.Helper()
	db, err := Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })
	return db
}
