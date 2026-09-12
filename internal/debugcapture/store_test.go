//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package debugcapture

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"gpt-load/internal/storage"
	"gpt-load/internal/storage/models"
)

func newTestStore(t *testing.T, now *time.Time) *Store {
	t.Helper()
	db, err := storage.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	if err := storage.AutoMigrate(db); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		sqlDB, err := db.DB()
		if err == nil {
			_ = sqlDB.Close()
		}
	})
	store, err := NewWithClock(db, func() time.Time { return *now })
	if err != nil {
		t.Fatal(err)
	}
	return store
}

func TestDBStoreCapturesSensitiveBytesChunksQueriesAndStreamingZIP(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	session, err := store.StartSession(SessionMetadata{
		RequestID: "req-1", AccessKeyID: 7, Protocol: "openai", Operation: "chat.completions",
		Fields: map[string]any{"tenant": "red"},
	})
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := session.StartAttempt(AttemptMetadata{Fields: map[string]any{"try": 1}})
	if err != nil {
		t.Fatal(err)
	}
	rawHeaders := []byte("Authorization: Bearer secret\r\nX-Bytes: \x00\xff\r\n")
	rawBody := bytes.Repeat([]byte{0, 1, 2, 255}, 10000)
	if err := attempt.AppendHeaders(DirectionRequest, rawHeaders); err != nil {
		t.Fatal(err)
	}
	if err := attempt.AppendBodyPart(DirectionRequest, bytes.NewReader(rawBody[:1000])); err != nil {
		t.Fatal(err)
	}
	if err := attempt.AppendBodyPart(DirectionRequest, bytes.NewReader(rawBody[1000:9000])); err != nil {
		t.Fatal(err)
	}
	if err := attempt.AppendBodyPart(DirectionRequest, bytes.NewReader(rawBody[9000:])); err != nil {
		t.Fatal(err)
	}
	var chunkCount int64
	if err := store.db.Model(&models.DebugCaptureChunk{}).Where("attempt_id = ?", attempt.ID()).Count(&chunkCount).Error; err != nil {
		t.Fatal(err)
	}
	if chunkCount != 4 {
		t.Fatalf("stored chunk count = %d, want headers plus three body chunks", chunkCount)
	}
	if err := attempt.Complete(); err != nil {
		t.Fatal(err)
	}
	if err := attempt.AppendBodyPart(DirectionRequest, strings.NewReader("late")); !errors.Is(err, ErrClosed) {
		t.Fatalf("append after attempt completion = %v, want ErrClosed", err)
	}
	if err := session.Complete(); err != nil {
		t.Fatal(err)
	}

	record, err := store.ReadSession(session.ID())
	if err != nil {
		t.Fatal(err)
	}
	if record.State != StateCompleted || record.Attempts[0].State != StateCompleted {
		t.Fatalf("lifecycle record = %#v", record)
	}
	gotHeaders, err := store.ReadPart(session.ID(), attempt.ID(), PartHeaders, DirectionRequest)
	if err != nil || !bytes.Equal(gotHeaders, rawHeaders) {
		t.Fatalf("headers = %x, err = %v", gotHeaders, err)
	}
	gotBody, err := store.ReadPart(session.ID(), attempt.ID(), PartBody, DirectionRequest)
	if err != nil || !bytes.Equal(gotBody, rawBody) {
		t.Fatalf("body length = %d, err = %v", len(gotBody), err)
	}

	matches, err := store.QuerySessions(SessionQuery{RequestID: "req-1", AccessKeyID: 7, Protocol: "openai", Operation: "chat.completions", Limit: 10})
	if err != nil || len(matches) != 1 || matches[0].ID != session.ID() {
		t.Fatalf("query matches = %#v, err = %v", matches, err)
	}
	var exported bytes.Buffer
	if err := store.ExportZIP(session.ID(), &exported); err != nil {
		t.Fatal(err)
	}
	archive, err := zip.NewReader(bytes.NewReader(exported.Bytes()), int64(exported.Len()))
	if err != nil {
		t.Fatal(err)
	}
	files := make(map[string][]byte)
	for _, file := range archive.File {
		reader, err := file.Open()
		if err != nil {
			t.Fatal(err)
		}
		content, readErr := io.ReadAll(reader)
		closeErr := reader.Close()
		if readErr != nil || closeErr != nil {
			t.Fatalf("read %s: read=%v close=%v", file.Name, readErr, closeErr)
		}
		files[file.Name] = content
	}
	if !bytes.Equal(files[archivePartName(attempt.ID(), PartHeaders, DirectionRequest)], rawHeaders) ||
		!bytes.Equal(files[archivePartName(attempt.ID(), PartBody, DirectionRequest)], rawBody) {
		t.Fatal("ZIP did not preserve raw bytes")
	}
}

func TestDBStoreExportsEmptyRawParts(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	session, err := store.StartSession(SessionMetadata{RequestID: "empty-parts"})
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := session.StartAttempt(AttemptMetadata{})
	if err != nil {
		t.Fatal(err)
	}
	if err := attempt.Complete(); err != nil {
		t.Fatal(err)
	}
	if err := session.Complete(); err != nil {
		t.Fatal(err)
	}

	var exported bytes.Buffer
	if err := store.ExportZIP(session.ID(), &exported); err != nil {
		t.Fatal(err)
	}
	archive, err := zip.NewReader(bytes.NewReader(exported.Bytes()), int64(exported.Len()))
	if err != nil {
		t.Fatal(err)
	}
	entries := make(map[string]uint64, len(archive.File))
	for _, file := range archive.File {
		entries[file.Name] = file.UncompressedSize64
	}
	for _, direction := range []Direction{DirectionRequest, DirectionResponse} {
		for _, part := range []Part{PartHeaders, PartBody} {
			name := archivePartName(attempt.ID(), part, direction)
			size, ok := entries[name]
			if !ok || size != 0 {
				t.Fatalf("ZIP entry %q = %d, exists=%t; want present zero-byte raw part", name, size, ok)
			}
		}
	}
}

func TestDBStorePersistsAttemptOutcomeEvents(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	session, err := store.StartSession(SessionMetadata{RequestID: "events"})
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := session.StartAttempt(AttemptMetadata{Fields: map[string]any{"kind": "forward"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := attempt.RecordResponseFlush(); err != nil {
		t.Fatal(err)
	}
	if err := attempt.RecordResponseShortWrite(2, 4); err != nil {
		t.Fatal(err)
	}
	if err := attempt.RecordResponseError(errors.New("upstream response failed")); err != nil {
		t.Fatal(err)
	}
	if err := attempt.RecordResponseTermination("timeout", "deadline exceeded"); err != nil {
		t.Fatal(err)
	}
	if err := attempt.RecordResponseHijack(nil); err != nil {
		t.Fatal(err)
	}
	if err := attempt.RecordRequestOutcome("context_canceled", context.Canceled); err != nil {
		t.Fatal(err)
	}
	if err := attempt.Complete(); err != nil {
		t.Fatal(err)
	}
	if err := session.Complete(); err != nil {
		t.Fatal(err)
	}

	record, err := store.ReadSession(session.ID())
	if err != nil {
		t.Fatal(err)
	}
	if len(record.Attempts) != 1 || len(record.Attempts[0].Metadata.Events) != 6 {
		t.Fatalf("attempt events = %#v, want six events", record.Attempts)
	}
	if record.Attempts[0].Metadata.Events[1].Written != 2 ||
		record.Attempts[0].Metadata.Events[1].Requested != 4 {
		t.Fatalf("short write event = %#v", record.Attempts[0].Metadata.Events[1])
	}
	if record.Attempts[0].Metadata.Events[2].Error != "upstream response failed" ||
		record.Attempts[0].Metadata.Events[3].Kind != "response_termination" ||
		record.Attempts[0].Metadata.Events[3].Outcome != "timeout" ||
		record.Attempts[0].Metadata.Events[4].Error != "" ||
		record.Attempts[0].Metadata.Events[5].Error != context.Canceled.Error() {
		t.Fatalf("event errors = %#v", record.Attempts[0].Metadata.Events)
	}
}

func TestDBStoreLongAppendCrossingExpiryRollsBackAllChunks(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	session, err := store.StartSession(SessionMetadata{RequestID: "slow-append"})
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := session.StartAttempt(AttemptMetadata{})
	if err != nil {
		t.Fatal(err)
	}

	if err := attempt.AppendBodyPart(DirectionRequest, &expiryAdvancingReader{now: &now}); !errors.Is(err, ErrExpired) {
		t.Fatalf("long append error = %v, want ErrExpired", err)
	}
	var chunks int64
	if err := store.db.Model(&models.DebugCaptureChunk{}).Where("attempt_id = ?", attempt.ID()).Count(&chunks).Error; err != nil {
		t.Fatal(err)
	}
	if chunks != 0 {
		t.Fatalf("chunks after expired append = %d, want zero", chunks)
	}
}

type expiryAdvancingReader struct {
	now      *time.Time
	advanced bool
}

func (reader *expiryAdvancingReader) Read(buffer []byte) (int, error) {
	if reader.advanced {
		return 0, io.EOF
	}
	*reader.now = reader.now.Add(retention)
	reader.advanced = true
	return copy(buffer, []byte("must-not-persist")), nil
}
func TestDBStoreExpiryTerminalLockAndBatchedCleanup(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	session, err := store.StartSession(SessionMetadata{RequestID: "expiry"})
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := session.StartAttempt(AttemptMetadata{})
	if err != nil {
		t.Fatal(err)
	}
	now = now.Add(retention)
	if _, err := store.ReadSession(session.ID()); !errors.Is(err, ErrExpired) {
		t.Fatalf("ReadSession() error = %v, want ErrExpired", err)
	}
	if err := attempt.AppendBodyPart(DirectionResponse, strings.NewReader("late")); !errors.Is(err, ErrExpired) {
		t.Fatalf("AppendBodyPart() error = %v, want ErrExpired", err)
	}
	if err := session.Complete(); !errors.Is(err, ErrExpired) {
		t.Fatalf("Complete() error = %v, want ErrExpired", err)
	}
	var expiredRow models.DebugCapture
	if err := store.db.First(&expiredRow, "id = ?", session.ID()).Error; err != nil {
		t.Fatal(err)
	}
	if expiredRow.State != string(StateExpired) {
		t.Fatalf("expired session state = %q, want durable expired state", expiredRow.State)
	}
	removed, err := store.Cleanup()
	if err != nil || removed != 1 {
		t.Fatalf("Cleanup() = %d, %v; want one removal", removed, err)
	}
	var attempts, chunks int64
	if err := store.db.Model(&models.DebugCaptureAttempt{}).Where("capture_id = ?", session.ID()).Count(&attempts).Error; err != nil {
		t.Fatal(err)
	}
	if err := store.db.Model(&models.DebugCaptureChunk{}).Where("capture_id = ?", session.ID()).Count(&chunks).Error; err != nil {
		t.Fatal(err)
	}
	if attempts != 0 || chunks != 0 {
		t.Fatalf("cleanup children = attempts %d, chunks %d; want zero", attempts, chunks)
	}
	if _, err := store.ReadSession(session.ID()); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ReadSession after cleanup = %v, want ErrNotFound", err)
	}
	removed, err = store.Cleanup()
	if err != nil || removed != 0 {
		t.Fatalf("repeat Cleanup() = %d, %v; want zero", removed, err)
	}
}

func TestDBStoreUpdateMetadataPatchesTopLevelAndFieldsTransactionally(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	session, err := store.StartSession(SessionMetadata{
		RequestID: "request-before", AccessKeyID: 7, Protocol: "openai", Operation: "chat",
		Fields: map[string]any{"keep": "yes", "replace": "old"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := session.UpdateMetadata(SessionMetadata{
		RequestID: "request-after", AccessKeyID: 9, Protocol: "responses", Operation: "generate",
		Fields: map[string]any{"replace": "new", "added": map[string]any{"raw": "value"}},
	}); err != nil {
		t.Fatal(err)
	}
	record, err := store.ReadSession(session.ID())
	if err != nil {
		t.Fatal(err)
	}
	if record.RequestID != "request-after" || record.AccessKeyID != 9 || record.Protocol != "responses" || record.Operation != "generate" {
		t.Fatalf("patched top-level metadata = %#v", record)
	}
	if record.Metadata.Fields["keep"] != "yes" || record.Metadata.Fields["replace"] != "new" {
		t.Fatalf("patched fields = %#v", record.Metadata.Fields)
	}
	added, ok := record.Metadata.Fields["added"].(map[string]any)
	if !ok || added["raw"] != "value" {
		t.Fatalf("nested field = %#v", record.Metadata.Fields["added"])
	}
	var row models.DebugCapture
	if err := store.db.First(&row, "id = ?", session.ID()).Error; err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(row.Metadata, []byte(`"request_id":"request-after"`)) || !bytes.Contains(row.Metadata, []byte(`"added"`)) {
		t.Fatalf("stored metadata bytes = %s", row.Metadata)
	}
}

func TestDBStoreUpdateMetadataHonorsExpiryAndTerminalState(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	expired, err := store.StartSession(SessionMetadata{RequestID: "expired"})
	if err != nil {
		t.Fatal(err)
	}
	now = now.Add(retention)
	if err := expired.UpdateMetadata(SessionMetadata{RequestID: "must-not-write"}); !errors.Is(err, ErrExpired) {
		t.Fatalf("expired UpdateMetadata() = %v, want ErrExpired", err)
	}
	var expiredRow models.DebugCapture
	if err := store.db.First(&expiredRow, "id = ?", expired.ID()).Error; err != nil {
		t.Fatal(err)
	}
	if expiredRow.State != string(StateExpired) {
		t.Fatalf("expired metadata state = %q, want durable expired state", expiredRow.State)
	}
	now = now.Add(-retention)
	terminal, err := store.StartSession(SessionMetadata{RequestID: "active"})
	if err != nil {
		t.Fatal(err)
	}
	if err := terminal.Complete(); err != nil {
		t.Fatal(err)
	}
	if err := terminal.UpdateMetadata(SessionMetadata{Protocol: "must-not-write"}); !errors.Is(err, ErrClosed) {
		t.Fatalf("terminal UpdateMetadata() = %v, want ErrClosed", err)
	}
}
func TestDBStoreFailureOutcomeIsQueryable(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	session, err := store.StartSession(SessionMetadata{RequestID: "failed"})
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := session.StartAttempt(AttemptMetadata{})
	if err != nil {
		t.Fatal(err)
	}
	wantErr := errors.New("upstream failed")
	if err := attempt.Fail(wantErr); err != nil {
		t.Fatal(err)
	}
	if err := session.Fail(wantErr); err != nil {
		t.Fatal(err)
	}
	record, err := store.ReadSession(session.ID())
	if err != nil {
		t.Fatal(err)
	}
	if record.State != StateFailed || record.Error != wantErr.Error() || record.Attempts[0].State != StateFailed {
		t.Fatalf("failure record = %#v", record)
	}
}

func TestDBStoreUpdateMetadataRollsBackOnDatabaseError(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	session, err := store.StartSession(SessionMetadata{RequestID: "before", Fields: map[string]any{"keep": true}})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.db.Exec(`CREATE TRIGGER deny_capture_metadata BEFORE UPDATE OF request_id ON debug_captures BEGIN SELECT RAISE(ABORT, 'denied'); END`).Error; err != nil {
		t.Fatal(err)
	}
	if err := session.UpdateMetadata(SessionMetadata{RequestID: "after", Protocol: "responses", Fields: map[string]any{"new": true}}); err == nil {
		t.Fatal("UpdateMetadata() error = nil, want database error")
	}
	if err := store.db.Exec(`DROP TRIGGER deny_capture_metadata`).Error; err != nil {
		t.Fatal(err)
	}
	record, err := store.ReadSession(session.ID())
	if err != nil {
		t.Fatal(err)
	}
	if record.RequestID != "before" || record.Protocol != "" || len(record.Metadata.Fields) != 1 || record.Metadata.Fields["keep"] != true {
		t.Fatalf("metadata after rollback = %#v", record)
	}
}
func TestDBStoreDatabaseErrorDoesNotMutateTerminalState(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	session, err := store.StartSession(SessionMetadata{})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.db.Exec(`CREATE TRIGGER deny_capture_terminal BEFORE UPDATE ON debug_captures BEGIN SELECT RAISE(ABORT, 'denied'); END`).Error; err != nil {
		t.Fatal(err)
	}
	if err := session.Complete(); err == nil {
		t.Fatal("Complete() error = nil, want database error")
	}
	if err := store.db.Exec(`DROP TRIGGER deny_capture_terminal`).Error; err != nil {
		t.Fatal(err)
	}
	record, err := store.ReadSession(session.ID())
	if err != nil {
		t.Fatal(err)
	}
	if record.State != StateActive {
		t.Fatalf("state after failed terminal update = %q, want active", record.State)
	}
}
