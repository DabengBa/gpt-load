//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package debugcapture

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"gpt-load/internal/storage"
	"gpt-load/internal/storage/models"
)

func TestRetentionPeriodIs24Hours(t *testing.T) {
	if got := RetentionPeriod(); got != 24*time.Hour {
		t.Fatalf("RetentionPeriod() = %s, want 24h", got)
	}
}

func TestRuntimeSweepsOnStartupAndDrainsRegisteredSessions(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	expired, err := store.StartSession(SessionMetadata{RequestID: "expired"})
	if err != nil {
		t.Fatal(err)
	}
	now = now.Add(retention)
	runtime := NewRuntimeWithInterval(true, store, time.Millisecond)
	if err := runtime.Start(); err != nil {
		t.Fatal(err)
	}
	if _, err := store.ReadSession(expired.ID()); !errors.Is(err, ErrNotFound) {
		t.Fatalf("startup sweep ReadSession() = %v, want ErrNotFound", err)
	}

	release, ok := runtime.AcquireSession()
	if !ok {
		t.Fatal("AcquireSession() = false, want admitted session")
	}
	stopDone := make(chan error, 1)
	go func() { stopDone <- runtime.Stop(context.Background()) }()
	select {
	case err := <-stopDone:
		t.Fatalf("Stop() returned before session release: %v", err)
	case <-time.After(10 * time.Millisecond):
	}
	release()
	if err := <-stopDone; err != nil {
		t.Fatalf("Stop() error = %v", err)
	}
	health, err := runtime.Health()
	if err != nil {
		t.Fatal(err)
	}
	if health.RemovedTotal < 1 || health.SweepTotal < 1 || health.Enabled != true {
		t.Fatalf("runtime health = %#v", health)
	}
}

func TestRuntimeEnabledCleansExpiredSessionsAndAdmits(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	expired, err := store.StartSession(SessionMetadata{RequestID: "expired-always-enabled"})
	if err != nil {
		t.Fatal(err)
	}
	now = now.Add(retention)
	runtime := NewRuntime(true, store)
	if err := runtime.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = runtime.Stop(context.Background()) }()
	if _, err := store.ReadSession(expired.ID()); !errors.Is(err, ErrNotFound) {
		t.Fatalf("startup sweep ReadSession() = %v, want ErrNotFound", err)
	}
	release, ok := runtime.AcquireSession()
	if !ok {
		t.Fatal("AcquireSession() = false, want always-enabled admission")
	}
	release()
	health, err := runtime.Health()
	if err != nil {
		t.Fatal(err)
	}
	if !health.Enabled || !health.Running {
		t.Fatalf("health = %#v, want enabled running runtime", health)
	}
}

func TestRuntimeLimitsActiveCaptureSessions(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	runtime := NewRuntimeWithInterval(true, newTestStore(t, &now), time.Hour)
	if err := runtime.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = runtime.Stop(context.Background()) }()

	releases := make([]func(), 0, maxActiveCaptureSessions)
	for range maxActiveCaptureSessions {
		release, ok := runtime.AcquireSession()
		if !ok {
			t.Fatal("AcquireSession() rejected a session below the configured limit")
		}
		defer release()
		releases = append(releases, release)
	}
	if release, ok := runtime.AcquireSession(); ok || release != nil {
		t.Fatal("AcquireSession() accepted a session above the configured limit")
	}
	health, err := runtime.Health()
	if err != nil {
		t.Fatal(err)
	}
	if health.Active != 0 || health.Error != "session_limit_reached" {
		t.Fatalf("health = %#v, want exhausted admission reported even with no persisted active rows", health)
	}
	for _, release := range releases {
		release()
	}
	health, err = runtime.Health()
	if err != nil || health.Error != "" {
		t.Fatalf("health after release = %#v, err = %v", health, err)
	}
}

func TestRuntimeDisabledStillCleansExpiredSessions(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	expired, err := store.StartSession(SessionMetadata{RequestID: "expired-disabled"})
	if err != nil {
		t.Fatal(err)
	}
	now = now.Add(retention)
	runtime := NewRuntimeWithInterval(false, store, time.Hour)
	if err := runtime.Start(); err != nil {
		t.Fatal(err)
	}
	if err := runtime.Stop(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := store.ReadSession(expired.ID()); !errors.Is(err, ErrNotFound) {
		t.Fatalf("disabled runtime ReadSession() = %v, want ErrNotFound", err)
	}
}

func TestRuntimeDisabledPeriodicallyDeletesExpiredRawChunks(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	session, err := store.StartSession(SessionMetadata{RequestID: "periodic-disabled"})
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := session.StartAttempt(AttemptMetadata{})
	if err != nil {
		t.Fatal(err)
	}
	if err := attempt.AppendBodyPart(DirectionRequest, strings.NewReader("synthetic payload")); err != nil {
		t.Fatal(err)
	}
	remaining, err := store.StartSession(SessionMetadata{RequestID: "remaining"})
	if err != nil {
		t.Fatal(err)
	}
	count := func(model any, where string, id string) int64 {
		t.Helper()
		var n int64
		if err := store.db.Model(model).Where(where, id).Count(&n).Error; err != nil {
			t.Fatal(err)
		}
		return n
	}
	if count(&models.DebugCaptureChunk{}, "capture_id = ?", session.ID()) != 1 {
		t.Fatal("expected stored raw chunk")
	}
	runtime := NewRuntimeWithInterval(false, store, 10*time.Millisecond)
	if err := runtime.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = runtime.Stop(context.Background()) })
	if release, ok := runtime.AcquireSession(); ok || release != nil {
		t.Fatal("disabled capture admitted session")
	}
	result := store.db.Exec("UPDATE debug_captures SET expires_at_ms = ? WHERE id = ?", now.Add(-time.Millisecond).UnixMilli(), session.ID())
	if result.Error != nil || result.RowsAffected != 1 {
		t.Fatalf("expire row: %v, affected %d", result.Error, result.RowsAffected)
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		health, err := runtime.Health()
		if err != nil {
			t.Fatal(err)
		}
		if count(&models.DebugCapture{}, "id = ?", session.ID()) == 0 && health.SweepTotal >= 2 && health.RemovedTotal >= 1 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if count(&models.DebugCapture{}, "id = ?", session.ID()) != 0 || count(&models.DebugCaptureAttempt{}, "capture_id = ?", session.ID()) != 0 || count(&models.DebugCaptureChunk{}, "capture_id = ?", session.ID()) != 0 {
		t.Fatal("expired session and raw children must be deleted")
	}
	if count(&models.DebugCapture{}, "id = ?", remaining.ID()) != 1 {
		t.Fatal("unexpired session removed")
	}
	health, err := runtime.Health()
	if err != nil || health.Enabled || health.Running || health.SweepTotal < 2 || health.RemovedTotal < 1 {
		t.Fatalf("disabled health = %#v, %v", health, err)
	}
}

func TestRuntimeDisabledStopWaitsForCleanup(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	runtime := NewRuntimeWithInterval(false, newTestStore(t, &now), time.Millisecond)
	var calls atomic.Int32
	entered := make(chan struct{})
	release := make(chan struct{})
	runtime.now = func() time.Time {
		if calls.Add(1) == 2 {
			close(entered)
			<-release
		}
		return now
	}
	if err := runtime.Start(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-entered:
	case <-time.After(time.Second):
		close(release)
		_ = runtime.Stop(context.Background())
		t.Fatal("periodic cleanup did not start")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if err := runtime.Stop(ctx); !errors.Is(err, context.DeadlineExceeded) {
		close(release)
		t.Fatalf("Stop() = %v, want deadline exceeded while cleanup is blocked", err)
	}
	close(release)
	if err := runtime.Stop(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := runtime.Stop(context.Background()); err != nil {
		t.Fatalf("repeated Stop() = %v", err)
	}
}

func TestRuntimePeriodicallyCleansExpiredSessions(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	session, err := store.StartSession(SessionMetadata{RequestID: "periodic-expiry"})
	if err != nil {
		t.Fatal(err)
	}
	runtime := NewRuntimeWithInterval(true, store, time.Millisecond)
	if err := runtime.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = runtime.Stop(context.Background()) }()
	if err := store.db.Exec("UPDATE debug_captures SET expires_at_ms = ? WHERE id = ?", now.Add(-time.Millisecond).UnixMilli(), session.ID()).Error; err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if _, err := store.ReadSession(session.ID()); errors.Is(err, ErrNotFound) {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("periodic cleanup did not remove expired session %s", session.ID())
}

func TestRuntimeHealthReportsCountsUnavailableWithoutFailing(t *testing.T) {
	db, err := storage.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatal(err)
	}
	if err := sqlDB.Close(); err != nil {
		t.Fatal(err)
	}
	store, err := New(db)
	if err != nil {
		t.Fatal(err)
	}
	runtime := NewRuntimeWithInterval(true, store, time.Hour)
	if err := runtime.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = runtime.Stop(context.Background()) }()

	health, err := runtime.Health()
	if err != nil {
		t.Fatalf("Health() error = %v, want degraded health response", err)
	}
	if health.Error != "counts_unavailable" {
		t.Fatalf("Health().Error = %q, want counts_unavailable", health.Error)
	}
}

func TestRuntimeStartFailureRejectsSessionAdmission(t *testing.T) {
	runtime := NewRuntimeWithInterval(true, nil, time.Hour)
	if err := runtime.Start(); err == nil {
		t.Fatal("Start() error = nil, want missing-store error")
	}
	if release, ok := runtime.AcquireSession(); ok || release != nil {
		t.Fatalf("AcquireSession() = (release-nil:%t, ok:%t), want rejected admission", release == nil, ok)
	}
}
