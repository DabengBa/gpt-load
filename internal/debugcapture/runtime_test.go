//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package debugcapture

import (
	"context"
	"errors"
	"testing"
	"time"

	"gpt-load/internal/storage"
)

func TestRuntimeSweepsOnStartupAndDrainsRegisteredSessions(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	expired, err := store.StartSession(SessionMetadata{RequestID: "expired"})
	if err != nil {
		t.Fatal(err)
	}
	now = now.Add(retention)
	runtime := NewRuntimeWithInterval(store, time.Millisecond)
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

func TestRuntimeAlwaysEnabledCleansExpiredSessionsAndAdmits(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	expired, err := store.StartSession(SessionMetadata{RequestID: "expired-always-enabled"})
	if err != nil {
		t.Fatal(err)
	}
	now = now.Add(retention)
	runtime := NewRuntime(store)
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
		t.Fatalf("health = %#v, want always-enabled running runtime", health)
	}
}

func TestRuntimePeriodicallyCleansExpiredSessions(t *testing.T) {
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store := newTestStore(t, &now)
	session, err := store.StartSession(SessionMetadata{RequestID: "periodic-expiry"})
	if err != nil {
		t.Fatal(err)
	}
	runtime := NewRuntimeWithInterval(store, time.Millisecond)
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
	runtime := NewRuntimeWithInterval(store, time.Hour)
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
	runtime := NewRuntimeWithInterval(nil, time.Hour)
	if err := runtime.Start(); err == nil {
		t.Fatal("Start() error = nil, want missing-store error")
	}
	if release, ok := runtime.AcquireSession(); ok || release != nil {
		t.Fatalf("AcquireSession() = (release-nil:%t, ok:%t), want rejected admission", release == nil, ok)
	}
}
