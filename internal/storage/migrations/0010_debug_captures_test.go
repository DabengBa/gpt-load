package migrations_test

import (
	"strings"
	"sync"
	"testing"

	"gorm.io/gorm/schema"

	"gpt-load/internal/storage/migrations"
	"gpt-load/internal/storage/models"
)

func TestDebugCaptureMigrationCreatesQueryableBlobSchema(t *testing.T) {
	db := openInitialTestDatabase(t)
	if err := migrations.Up0001(db); err != nil {
		t.Fatal(err)
	}
	if err := migrations.Up0010(db); err != nil {
		t.Fatalf("Up0010() error = %v", err)
	}
	if err := migrations.Validate0010(db); err != nil {
		t.Fatalf("Validate0010() error = %v", err)
	}
	for _, table := range []string{"debug_captures", "debug_capture_attempts", "debug_capture_chunks"} {
		if !db.Migrator().HasTable(table) {
			t.Errorf("missing %s", table)
		}
	}
	for table, column := range map[string]string{
		"debug_captures":         "error",
		"debug_capture_attempts": "error",
	} {
		definition := initialColumns(t, db, table)[column]
		if definition.NotNull != 1 {
			t.Errorf("%s.%s notnull = %d, want 1", table, column, definition.NotNull)
		}
		if definition.DefaultValue != nil {
			t.Errorf("%s.%s default = %q, want no database default", table, column, *definition.DefaultValue)
		}
	}
	if got := strings.ToLower(initialColumns(t, db, "debug_capture_chunks")["data"].Type); !strings.Contains(got, "blob") {
		t.Fatalf("debug_capture_chunks.data type = %q, want SQLite blob-compatible type", got)
	}
	for _, index := range []struct{ table, name string }{
		{"debug_captures", "idx_debug_captures_request_id"},
		{"debug_captures", "idx_debug_captures_created_at_ms"},
		{"debug_captures", "idx_debug_captures_state"},
		{"debug_captures", "idx_debug_captures_protocol"},
		{"debug_captures", "idx_debug_captures_operation"},
		{"debug_captures", "idx_debug_captures_access_key_id"},
		{"debug_capture_chunks", "idx_debug_capture_chunks_attempt_part_direction_id"},
	} {
		if !db.Migrator().HasIndex(index.table, index.name) {
			t.Errorf("missing index %s on %s", index.name, index.table)
		}
	}
	if !db.Migrator().HasConstraint("debug_capture_attempts", "fk_debug_captures_attempts") {
		t.Fatal("debug capture attempt foreign key is missing")
	}
	if !db.Migrator().HasConstraint("debug_capture_chunks", "fk_debug_capture_attempts_chunks") {
		t.Fatal("debug capture chunk attempt foreign key is missing")
	}
	if !db.Migrator().HasConstraint("debug_capture_chunks", "fk_debug_capture_chunks_capture") {
		t.Fatal("debug capture chunk capture foreign key is missing")
	}

	captureA := models.DebugCapture{ID: "capture-a", State: "active"}
	captureB := models.DebugCapture{ID: "capture-b", State: "active"}
	if err := db.Create(&captureA).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Create(&captureB).Error; err != nil {
		t.Fatal(err)
	}
	attempt := models.DebugCaptureAttempt{ID: "attempt-a", CaptureID: captureA.ID, State: "active", Sequence: 1}
	if err := db.Create(&attempt).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Create(&models.DebugCaptureChunk{CaptureID: "missing", AttemptID: attempt.ID, Part: "body", Direction: "request", Data: []byte("x")}).Error; err == nil {
		t.Fatal("chunk with missing capture was accepted")
	}
	if err := db.Create(&models.DebugCaptureChunk{CaptureID: captureB.ID, AttemptID: attempt.ID, Part: "body", Direction: "request", Data: []byte("x")}).Error; err == nil {
		t.Fatal("chunk with mismatched capture and attempt was accepted")
	}

	if err := migrations.Up0010(db); err != nil {
		t.Fatalf("repeated Up0010() error = %v", err)
	}
}

func TestDebugCaptureMigrationValidationRejectsMissingCompositeForeignKey(t *testing.T) {
	db := openInitialTestDatabase(t)
	if err := migrations.Up0001(db); err != nil {
		t.Fatal(err)
	}
	if err := migrations.Up0010(db); err != nil {
		t.Fatalf("Up0010() error = %v", err)
	}
	if err := db.Migrator().DropConstraint("debug_capture_chunks", "fk_debug_capture_attempts_chunks"); err != nil {
		t.Fatalf("DropConstraint() error = %v", err)
	}
	if err := db.Exec("CREATE INDEX idx_debug_capture_chunks_attempt_part_direction_id ON debug_capture_chunks (attempt_id, part, direction, id)").Error; err != nil {
		t.Fatalf("restore dropped index: %v", err)
	}
	if err := migrations.Validate0010(db); err == nil {
		t.Fatal("Validate0010() error = nil, want missing composite foreign key error")
	} else if !strings.Contains(err.Error(), "composite capture/attempt foreign key") {
		t.Fatalf("Validate0010() error = %v, want composite foreign key error", err)
	}
}

func TestDebugCaptureTextErrorsRemainNonNullWithoutDefaults(t *testing.T) {
	parsed, err := schema.Parse(&models.DebugCapture{}, &sync.Map{}, schema.NamingStrategy{})
	if err != nil {
		t.Fatalf("parse DebugCapture schema: %v", err)
	}
	attemptParsed, err := schema.Parse(&models.DebugCaptureAttempt{}, &sync.Map{}, schema.NamingStrategy{})
	if err != nil {
		t.Fatalf("parse DebugCaptureAttempt schema: %v", err)
	}
	for name, parsedSchema := range map[string]*schema.Schema{
		"DebugCapture":        parsed,
		"DebugCaptureAttempt": attemptParsed,
	} {
		field := parsedSchema.FieldsByDBName["error"]
		if field == nil {
			t.Fatalf("%s error field is missing", name)
		}
		if !field.NotNull {
			t.Errorf("%s error NotNull = false, want true", name)
		}
		if field.HasDefaultValue {
			t.Errorf("%s error HasDefaultValue = true, want false for MySQL TEXT portability", name)
		}
	}
}
