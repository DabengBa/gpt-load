package migrations_test

import (
	"testing"

	"gpt-load/internal/storage/migrations"
)

func TestDebugCaptureChunkCompressionMigrationPreservesLegacyRows(t *testing.T) {
	db := openInitialTestDatabase(t)
	if err := db.Exec(`CREATE TABLE debug_capture_chunks (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		capture_id VARCHAR(64) NOT NULL,
		attempt_id VARCHAR(64) NOT NULL,
		part VARCHAR(16) NOT NULL,
		direction VARCHAR(16) NOT NULL,
		data BLOB NOT NULL,
		created_at_ms INTEGER NOT NULL
	)`).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Exec(`INSERT INTO debug_capture_chunks (capture_id, attempt_id, part, direction, data, created_at_ms) VALUES ('capture', 'attempt', 'body', 'request', x'00ff', 1)`).Error; err != nil {
		t.Fatal(err)
	}
	if err := migrations.Up0026(db); err != nil {
		t.Fatal(err)
	}
	if err := migrations.Up0026(db); err != nil {
		t.Fatal(err)
	}
	var row struct {
		Data     []byte
		Encoding string
	}
	if err := db.Table("debug_capture_chunks").Select("data", "encoding").Take(&row).Error; err != nil {
		t.Fatal(err)
	}
	if string(row.Data) != string([]byte{0, 0xff}) || row.Encoding != "identity" {
		t.Fatalf("legacy row data/encoding = %x/%q", row.Data, row.Encoding)
	}
}
