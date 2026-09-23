//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package container

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"gpt-load/internal/control"
	"gpt-load/internal/debugcapture"
	"gpt-load/internal/gateway"
	"gpt-load/internal/storage"
)

func TestDebugCaptureFactoryPanicReleasesRuntimeAdmission(t *testing.T) {
	db, err := storage.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	if err := storage.AutoMigrate(db); err != nil {
		t.Fatal(err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })
	var panicNow atomic.Bool
	store, err := debugcapture.NewWithClock(db, func() time.Time {
		if panicNow.Load() {
			panic("debug capture clock panic")
		}
		return time.Now()
	})
	if err != nil {
		t.Fatal(err)
	}
	runtime := debugcapture.NewRuntimeWithInterval(true, store, time.Hour)
	if err := runtime.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()
		_ = runtime.Stop(ctx)
	}()
	factory := newDebugCaptureFactory(store, runtime)
	panicNow.Store(true)
	func() {
		defer func() {
			if recovered := recover(); recovered == nil {
				t.Fatal("StartSession() did not propagate the storage panic")
			}
		}()
		_, _ = factory.StartSession(gateway.CaptureSessionMetadata{RequestID: "panic-start"})
	}()
	stopCtx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := runtime.Stop(stopCtx); err != nil {
		t.Fatalf("Stop() error = %v, want admission released after panic", err)
	}
}

func TestDebugCaptureFailurePanicReleasesRuntimeAdmission(t *testing.T) {
	db, err := storage.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	if err := storage.AutoMigrate(db); err != nil {
		t.Fatal(err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })
	var panicNow atomic.Bool
	store, err := debugcapture.NewWithClock(db, func() time.Time {
		if panicNow.Load() {
			panic("debug capture clock panic")
		}
		return time.Now()
	})
	if err != nil {
		t.Fatal(err)
	}
	runtime := debugcapture.NewRuntimeWithInterval(true, store, time.Hour)
	if err := runtime.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()
		_ = runtime.Stop(ctx)
	}()
	factory := newDebugCaptureFactory(store, runtime)
	session, err := factory.StartSession(gateway.CaptureSessionMetadata{RequestID: "panic-fail"})
	if err != nil {
		t.Fatal(err)
	}
	panicNow.Store(true)
	func() {
		defer func() {
			if recovered := recover(); recovered == nil {
				t.Fatal("Fail() did not propagate the storage panic")
			}
		}()
		_ = session.Fail(errors.New("terminal failure"))
	}()
	stopCtx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := runtime.Stop(stopCtx); err != nil {
		t.Fatalf("Stop() error = %v, want admission released after panic", err)
	}
}

func TestDebugCaptureSessionRetainsAdmissionAfterTerminalFailure(t *testing.T) {
	db, err := storage.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	if err := storage.AutoMigrate(db); err != nil {
		t.Fatal(err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })
	store, err := debugcapture.New(db)
	if err != nil {
		t.Fatal(err)
	}
	runtime := debugcapture.NewRuntimeWithInterval(true, store, time.Hour)
	if err := runtime.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = runtime.Stop(context.Background()) })
	factory := newDebugCaptureFactory(store, runtime)
	session, err := factory.StartSession(gateway.CaptureSessionMetadata{RequestID: "terminal-failure"})
	if err != nil {
		t.Fatal(err)
	}
	if err := db.Exec("CREATE TRIGGER deny_capture_terminal BEFORE UPDATE ON debug_captures WHEN NEW.state <> OLD.state BEGIN SELECT RAISE(ABORT, 'denied'); END").Error; err != nil {
		t.Fatal(err)
	}
	if err := session.Complete(); err == nil {
		t.Fatal("Complete() error = nil, want injected terminal failure")
	}
	stopCtx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	if err := runtime.Stop(stopCtx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Stop() error = %v, want deadline while terminal fallback remains possible", err)
	}
	if err := session.Fail(errors.New("terminal fallback")); err == nil {
		t.Fatal("Fail() error = nil, want injected terminal failure")
	}
	if err := runtime.Stop(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := db.Exec("DROP TRIGGER deny_capture_terminal").Error; err != nil {
		t.Fatal(err)
	}
}

func TestDebugCaptureFactoryPreservesRawGatewayAndProbeCommunication(t *testing.T) {
	db, err := storage.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}

	t.Run("probe raw communication", func(t *testing.T) {
		db, err := storage.Open(":memory:")
		if err != nil {
			t.Fatal(err)
		}
		if err := storage.AutoMigrate(db); err != nil {
			t.Fatal(err)
		}
		sqlDB, err := db.DB()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = sqlDB.Close() })
		store, err := debugcapture.New(db)
		if err != nil {
			t.Fatal(err)
		}
		factory := newDebugCaptureFactory(store, nil)
		session, err := factory.StartProbeCapture(control.ProbeCaptureSessionMetadata{
			RequestID: "probe-request-1", Protocol: "openai", Operation: "probe",
			Fields: map[string]string{"provider": "openai", "model": "gpt-4o"},
		})
		if err != nil {
			t.Fatal(err)
		}
		attempt, err := session.StartProbeAttempt(control.ProbeCaptureAttemptMetadata{
			AttemptID: "probe-attempt-1", Sequence: 1,
			Fields: map[string]string{"provider": "openai", "model": "gpt-4o", "route_mode": "native"},
		})
		if err != nil {
			t.Fatal(err)
		}
		requestHeaders := []byte("Content-Type: application/json\r\n")
		requestBody := []byte(`{"prompt":"probe"}`)
		responseHeaders := []byte("HTTP 502\r\nContent-Type: application/json\r\n")
		responseBody := []byte(`{"error":"raw provider failure"}`)
		for _, write := range []func() error{
			func() error { return attempt.AppendRequestHeaders(requestHeaders) },
			func() error { return attempt.AppendRequestBody(requestBody) },
			func() error { return attempt.AppendResponseHeaders(responseHeaders) },
			func() error { return attempt.AppendResponseBody(responseBody) },
			func() error { return attempt.RecordResponseTermination("read_error", "unexpected EOF") },
			attempt.Complete,
			session.Complete,
		} {
			if err := write(); err != nil {
				t.Fatal(err)
			}
		}
		records, err := store.QuerySessions(debugcapture.SessionQuery{RequestID: "probe-request-1", Operation: "probe", Limit: 1})
		if err != nil || len(records) != 1 {
			t.Fatalf("QuerySessions() = %d records, err = %v", len(records), err)
		}
		record := records[0]
		if len(record.Attempts) != 1 || record.Attempts[0].Sequence != 1 ||
			record.Attempts[0].Metadata.Fields["logical_attempt_id"] != "probe-attempt-1" ||
			record.Attempts[0].Metadata.Fields["provider"] != "openai" ||
			len(record.Attempts[0].Metadata.Events) == 0 {
			t.Fatalf("probe capture record = %#v", record)
		}
		for _, test := range []struct {
			part      debugcapture.Part
			direction debugcapture.Direction
			want      []byte
		}{
			{part: debugcapture.PartHeaders, direction: debugcapture.DirectionRequest, want: requestHeaders},
			{part: debugcapture.PartBody, direction: debugcapture.DirectionRequest, want: requestBody},
			{part: debugcapture.PartHeaders, direction: debugcapture.DirectionResponse, want: responseHeaders},
			{part: debugcapture.PartBody, direction: debugcapture.DirectionResponse, want: responseBody},
		} {
			got, err := store.ReadPart(record.ID, record.Attempts[0].ID, test.part, test.direction)
			if err != nil || !bytes.Equal(got, test.want) {
				t.Fatalf("ReadPart(%s/%s) = %q, err = %v; want %q", test.direction, test.part, got, err, test.want)
			}
		}
		var archive bytes.Buffer
		if err := store.ExportZIP(record.ID, &archive); err != nil {
			t.Fatal(err)
		}
		zipReader, err := zip.NewReader(bytes.NewReader(archive.Bytes()), int64(archive.Len()))
		if err != nil {
			t.Fatal(err)
		}
		var exported []byte
		for _, file := range zipReader.File {
			reader, err := file.Open()
			if err != nil {
				t.Fatal(err)
			}
			data, readErr := io.ReadAll(reader)
			closeErr := reader.Close()
			if readErr != nil || closeErr != nil {
				t.Fatalf("read ZIP entry %q: read=%v close=%v", file.Name, readErr, closeErr)
			}
			exported = append(exported, data...)
		}
		for _, expected := range [][]byte{requestHeaders, requestBody, responseHeaders, responseBody} {
			if !bytes.Contains(exported, expected) {
				t.Fatalf("debug capture ZIP did not contain %q", expected)
			}
		}
	})
	if err := storage.AutoMigrate(db); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		sqlDB, err := db.DB()
		if err == nil {
			_ = sqlDB.Close()
		}
	})
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store, err := debugcapture.NewWithClock(db, func() time.Time { return now })
	if err != nil {
		t.Fatal(err)
	}
	factory := newDebugCaptureFactory(store, nil)
	session, err := factory.StartSession(gateway.CaptureSessionMetadata{
		RequestID: "request-1", AccessKeyID: 7, Protocol: "openai",
		Method: http.MethodPost, Path: "/v1/chat/completions",
		Headers: http.Header{"Authorization": {"Bearer plaintext-secret"}, "Cookie": {"session=plaintext"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := session.StartAttempt(gateway.CaptureAttemptMetadata{
		AttemptID: "request-1:1", Sequence: 1, Fields: map[string]string{"kind": "forward"},
	})
	if err != nil {
		t.Fatal(err)
	}
	requestBody := []byte{0, 1, 2, 255}
	responseBody := []byte(`{"ok":true}`)
	if err := attempt.AppendRequestHeaders([]byte("Authorization: Bearer plaintext-secret\r\n")); err != nil {
		t.Fatal(err)
	}
	if err := attempt.AppendRequestBody(requestBody); err != nil {
		t.Fatal(err)
	}
	if err := attempt.AppendResponseHeaders([]byte("HTTP 200\r\nSet-Cookie: response-secret\r\n")); err != nil {
		t.Fatal(err)
	}
	if err := attempt.AppendResponseBody(responseBody); err != nil {
		t.Fatal(err)
	}
	events, ok := attempt.(interface{ RecordResponseFlush() error })
	if !ok {
		t.Fatal("capture attempt does not expose response event persistence")
	}
	if err := events.RecordResponseFlush(); err != nil {
		t.Fatal(err)
	}
	terminationEvents, ok := attempt.(interface {
		RecordResponseTermination(string, string) error
	})
	if !ok {
		t.Fatal("capture attempt does not expose response termination persistence")
	}
	if err := terminationEvents.RecordResponseTermination("eof", ""); err != nil {
		t.Fatal(err)
	}
	if err := attempt.Complete(); err != nil {
		t.Fatal(err)
	}
	if err := session.Complete(); err != nil {
		t.Fatal(err)
	}

	record, err := store.ReadSession(recordID(store, "request-1"))
	if err != nil {
		t.Fatal(err)
	}
	if record.Metadata.Fields["client_method"] != http.MethodPost ||
		record.Metadata.Fields["client_path"] != "/v1/chat/completions" {
		t.Fatalf("session metadata = %#v", record.Metadata)
	}
	if len(record.Attempts) != 1 || record.Attempts[0].Metadata.Fields["logical_attempt_id"] != "request-1:1" ||
		len(record.Attempts[0].Metadata.Events) != 2 {
		t.Fatalf("attempt metadata = %#v", record.Attempts)
	}
	gotRequestBody, err := store.ReadPart(record.ID, record.Attempts[0].ID, debugcapture.PartBody, debugcapture.DirectionRequest)
	if err != nil || !bytes.Equal(gotRequestBody, requestBody) {
		t.Fatalf("request body = %x, err = %v", gotRequestBody, err)
	}
	gotResponseBody, err := store.ReadPart(record.ID, record.Attempts[0].ID, debugcapture.PartBody, debugcapture.DirectionResponse)
	if err != nil || !bytes.Equal(gotResponseBody, responseBody) {
		t.Fatalf("response body = %q, err = %v", gotResponseBody, err)
	}
}

func recordID(store *debugcapture.Store, requestID string) string {
	records, err := store.QuerySessions(debugcapture.SessionQuery{RequestID: requestID, Limit: 1})
	if err != nil || len(records) != 1 {
		return ""
	}
	return records[0].ID
}
