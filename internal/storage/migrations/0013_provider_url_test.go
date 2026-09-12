package migrations_test

import (
	"strings"
	"testing"

	"gpt-load/internal/storage/migrations"
)

func TestProviderURLMigrationAddsNullableTextColumnAndPreservesRows(t *testing.T) {
	db := openInitialTestDatabase(t)
	for _, migrate := range []func() error{
		func() error { return migrations.Up0001(db) },
		func() error { return migrations.Up0002(db) },
		func() error { return migrations.Up0003(db) },
		func() error { return migrations.Up0004(db) },
		func() error { return migrations.Up0005(db) },
		func() error { return migrations.Up0006(db) },
		func() error { return migrations.Up0007(db) },
		func() error { return migrations.Up0008(db) },
		func() error { return migrations.Up0009(db) },
		func() error { return migrations.Up0010(db) },
		func() error { return migrations.Up0011(db) },
		func() error { return migrations.Up0012(db) },
	} {
		if err := migrate(); err != nil {
			t.Fatalf("prepare previous schema: %v", err)
		}
	}

	if err := db.Exec(`INSERT INTO groups (
		id, name, channel_id, connection_type, params, models, enabled, created_at_ms, updated_at_ms
	) VALUES (1, 'provider url migration', 'openai', 'api_key', '{}', '[]', true, 1, 1)`).Error; err != nil {
		t.Fatalf("create group: %v", err)
	}

	if err := migrations.Up0013(db); err != nil {
		t.Fatalf("Up0013() error = %v", err)
	}
	if err := migrations.Validate0013(db); err != nil {
		t.Fatalf("Validate0013() error = %v", err)
	}
	if !db.Migrator().HasColumn("groups", "provider_url") {
		t.Fatal("groups.provider_url is missing")
	}

	type providerURLRow struct {
		ID          uint
		ProviderURL *string
	}
	var row providerURLRow
	if err := db.Table("groups").Select("id", "provider_url").Where("id = ?", 1).Take(&row).Error; err != nil {
		t.Fatalf("read group provider_url: %v", err)
	}
	if row.ID != 1 {
		t.Fatalf("existing row ID after migration = %d, want 1", row.ID)
	}
	if row.ProviderURL != nil {
		t.Fatalf("existing row provider_url = %v, want nil", *row.ProviderURL)
	}

	// Verify column type is text-compatible and nullable
	columns, err := db.Migrator().ColumnTypes("groups")
	if err != nil {
		t.Fatalf("inspect groups columns: %v", err)
	}
	var found bool
	for _, column := range columns {
		if column.Name() == "provider_url" {
			found = true
			typeName := strings.ToLower(column.DatabaseTypeName())
			if !strings.Contains(typeName, "text") && !strings.Contains(typeName, "char") && typeName != "clob" {
				t.Fatalf("groups.provider_url type = %q, want text-compatible", typeName)
			}
			if nullable, known := column.Nullable(); known && !nullable {
				t.Fatalf("groups.provider_url nullable = false, want true")
			}
		}
	}
	if !found {
		t.Fatal("groups.provider_url column not found in ColumnTypes")
	}

	if err := migrations.Up0013(db); err != nil {
		t.Fatalf("repeated Up0013() error = %v", err)
	}
}

func TestProviderURLMigrationHandlesExistingColumn(t *testing.T) {
	db := openInitialTestDatabase(t)
	if err := migrations.Up0001(db); err != nil {
		t.Fatalf("Up0001() error = %v", err)
	}

	// Add column manually first, then run migration
	if err := db.Exec("ALTER TABLE groups ADD COLUMN provider_url text NULL").Error; err != nil {
		t.Fatalf("pre-add provider_url column: %v", err)
	}
	if err := migrations.Up0013(db); err != nil {
		t.Fatalf("Up0013() after pre-existing column error = %v", err)
	}
	if err := migrations.Validate0013(db); err != nil {
		t.Fatalf("Validate0013() after pre-existing column error = %v", err)
	}
}

func TestProviderURLMigrationRejectsMissingColumnInValidate(t *testing.T) {
	db := openInitialTestDatabase(t)
	if err := migrations.Up0001(db); err != nil {
		t.Fatalf("Up0001() error = %v", err)
	}

	if err := migrations.Validate0013(db); err == nil {
		t.Fatal("Validate0013() without column error = nil, want missing column error")
	} else if !strings.Contains(err.Error(), "provider_url") {
		t.Fatalf("Validate0013() error = %v, want column missing error", err)
	}
}
