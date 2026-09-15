package control

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"gpt-load/internal/health"
	"gpt-load/internal/state"
)

type fakeRuntimeTicker struct {
	ticks    chan time.Time
	stopped  chan struct{}
	stopOnce sync.Once
}

func newFakeRuntimeTicker() *fakeRuntimeTicker {
	return &fakeRuntimeTicker{
		ticks:   make(chan time.Time, 8),
		stopped: make(chan struct{}),
	}
}

func (ticker *fakeRuntimeTicker) C() <-chan time.Time {
	return ticker.ticks
}

func (ticker *fakeRuntimeTicker) Stop() {
	ticker.stopOnce.Do(func() { close(ticker.stopped) })
}

type controlledOperationRecovery struct {
	started  chan struct{}
	returned chan struct{}
}

type controlledStageCleaner struct {
	calls chan time.Time
}

func (cleaner *controlledStageCleaner) CleanupCredentialStages(_ context.Context, now time.Time) error {
	cleaner.calls <- now
	return nil
}

func (recovery *controlledOperationRecovery) RunOperationRecovery(ctx context.Context) {
	close(recovery.started)
	<-ctx.Done()
	close(recovery.returned)
}

type fakeRuntimeClock struct {
	mu  sync.Mutex
	now time.Time
}

func (clock *fakeRuntimeClock) set(now time.Time) {
	clock.mu.Lock()
	clock.now = now
	clock.mu.Unlock()
}

func (clock *fakeRuntimeClock) current() time.Time {
	clock.mu.Lock()
	defer clock.mu.Unlock()
	return clock.now
}

type controlledRequestLogCleaner struct {
	calls        chan time.Time
	release      chan struct{}
	returned     chan struct{}
	active       atomic.Int64
	maxActive    atomic.Int64
	ignoreCancel bool
}

func newControlledRequestLogCleaner(ignoreCancel bool) *controlledRequestLogCleaner {
	return &controlledRequestLogCleaner{
		calls:        make(chan time.Time, 8),
		release:      make(chan struct{}, 8),
		returned:     make(chan struct{}, 8),
		ignoreCancel: ignoreCancel,
	}
}

func (cleaner *controlledRequestLogCleaner) Sweep(ctx context.Context, now time.Time) {
	active := cleaner.active.Add(1)
	for {
		maxActive := cleaner.maxActive.Load()
		if active <= maxActive || cleaner.maxActive.CompareAndSwap(maxActive, active) {
			break
		}
	}
	cleaner.calls <- now
	if cleaner.ignoreCancel {
		<-cleaner.release
	} else {
		select {
		case <-cleaner.release:
		case <-ctx.Done():
		}
	}
	cleaner.active.Add(-1)
	cleaner.returned <- struct{}{}
}

func TestRuntimeRunsOnlyExplicitRuntimes(t *testing.T) {
	t.Parallel()
	recovery := &controlledOperationRecovery{
		started:  make(chan struct{}),
		returned: make(chan struct{}),
	}
	unexpectedTicker := make(chan struct{})
	runtime := &Runtime{
		operationRecovery: recovery,
		newTicker: func(time.Duration) runtimeTicker {
			close(unexpectedTicker)
			return newFakeRuntimeTicker()
		},
	}

	cancel, done := startRuntime(t, runtime)
	awaitSignal(t, recovery.started)
	select {
	case <-unexpectedTicker:
		t.Fatal("Runtime.Run started an unexpected scheduler")
	default:
	}
	cancel()
	awaitSignal(t, recovery.returned)
	awaitSignal(t, done)
}

func TestRuntimeRunsOperationRecoveryUntilCancellation(t *testing.T) {
	t.Parallel()
	recovery := &controlledOperationRecovery{
		started:  make(chan struct{}),
		returned: make(chan struct{}),
	}
	runtime := &Runtime{operationRecovery: recovery}

	cancel, done := startRuntime(t, runtime)
	awaitSignal(t, recovery.started)
	cancel()
	awaitSignal(t, recovery.returned)
	awaitSignal(t, done)
}

func TestRuntimeSweepsRequestLogsImmediatelyAndHourlyWithoutOverlap(t *testing.T) {
	t.Parallel()
	base := time.Date(2026, time.July, 24, 12, 0, 0, 0, time.UTC)
	clock := &fakeRuntimeClock{now: base}
	cleaner := newControlledRequestLogCleaner(false)
	retentionTicker := newFakeRuntimeTicker()
	created := make(chan time.Duration, 1)
	runtime := newTestRuntime(retentionTicker, created, clock.current)
	runtime.requestLogCleaner = cleaner

	cancel, done := startRuntime(t, runtime)
	if interval := awaitValue(t, created); interval != time.Hour {
		t.Fatalf("retention ticker interval = %v, want 1h", interval)
	}
	if got := awaitValue(t, cleaner.calls); !got.Equal(base) {
		t.Fatalf("immediate Sweep time = %v, want %v", got, base)
	}

	clock.set(base.Add(time.Hour))
	retentionTicker.ticks <- base.Add(99 * time.Hour)
	select {
	case got := <-cleaner.calls:
		t.Fatalf("overlapping Sweep started at %v before first returned", got)
	case <-time.After(25 * time.Millisecond):
	}
	cleaner.release <- struct{}{}
	awaitSignal(t, cleaner.returned)
	if got := awaitValue(t, cleaner.calls); !got.Equal(base.Add(time.Hour)) {
		t.Fatalf("hourly Sweep time = %v, want injected clock %v", got, base.Add(time.Hour))
	}
	cleaner.release <- struct{}{}
	awaitSignal(t, cleaner.returned)

	stopRuntime(t, cancel, done)
	awaitSignal(t, retentionTicker.stopped)
	if got := cleaner.maxActive.Load(); got != 1 {
		t.Fatalf("maximum concurrent Sweeps = %d, want 1", got)
	}
}

func TestRuntimeSweepsCredentialStagesWithoutRequestLogCleaner(t *testing.T) {
	t.Parallel()
	base := time.Date(2026, time.August, 13, 8, 0, 0, 0, time.UTC)
	retentionTicker := newFakeRuntimeTicker()
	created := make(chan time.Duration, 1)
	runtime := newTestRuntime(retentionTicker, created, func() time.Time { return base })
	cleaner := &controlledStageCleaner{calls: make(chan time.Time, 2)}
	runtime.stageCleaner = cleaner

	cancel, done := startRuntime(t, runtime)
	if interval := awaitValue(t, created); interval != time.Hour {
		t.Fatalf("retention interval = %v", interval)
	}
	if got := awaitValue(t, cleaner.calls); !got.Equal(base) {
		t.Fatalf("cleanup time = %v", got)
	}
	stopRuntime(t, cancel, done)
	awaitSignal(t, retentionTicker.stopped)
}

func TestRuntimeCancellationWaitsForRetentionSweep(t *testing.T) {
	t.Parallel()
	cleaner := newControlledRequestLogCleaner(true)
	retentionTicker := newFakeRuntimeTicker()
	created := make(chan time.Duration, 1)
	runtime := newTestRuntime(retentionTicker, created, time.Now)
	runtime.requestLogCleaner = cleaner

	cancel, done := startRuntime(t, runtime)
	if interval := awaitValue(t, created); interval != time.Hour {
		t.Fatalf("retention interval = %v, want 1h", interval)
	}
	_ = awaitValue(t, cleaner.calls)
	cancel()
	select {
	case <-done:
		t.Fatal("Runtime.Run returned before active retention Sweep completed")
	case <-time.After(25 * time.Millisecond):
	}
	cleaner.release <- struct{}{}
	awaitSignal(t, cleaner.returned)
	awaitSignal(t, done)
	awaitSignal(t, retentionTicker.stopped)
}

func TestCooldownProblemDoesNotAffectCandidateCollection(t *testing.T) {
	t.Parallel()
	base := time.Date(2026, time.July, 22, 12, 0, 0, 0, time.UTC)
	registry := state.NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]state.CredentialEntry{{
		ID: 1, GroupID: 10, Version: 1, IdentityGeneration: 1, Fingerprint: "test-1", AuthState: state.CredentialAuthStateReady, EncryptedValue: "cipher-one",
	}}); err != nil {
		t.Fatalf("Replace() error = %v", err)
	}
	stats := health.NewStatsStore()
	for sample := 0; sample < 10; sample++ {
		stats.RecordSuccess(1, base)
	}
	stats.RecordProblem(1, health.FailureCategoryRateLimited, 429, base)

	candidates := registry.CollectCredentialCandidates([]uint{10}, nil, base)
	if len(candidates) != 1 {
		t.Fatalf("CollectCandidates() = %#v, want one candidate", candidates)
	}
}

func newTestRuntime(
	retentionTicker *fakeRuntimeTicker,
	created chan<- time.Duration,
	now func() time.Time,
) *Runtime {
	return &Runtime{
		now: now,
		newTicker: func(interval time.Duration) runtimeTicker {
			created <- interval
			if interval != retentionInterval {
				panic("unexpected runtime ticker interval: " + interval.String())
			}
			return retentionTicker
		},
	}
}

func TestRuntimeBlacklistReleaseMaintenanceStartsRunsImmediatelyAndTicks(t *testing.T) {
	t.Parallel()
	base := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := state.NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]state.CredentialEntry{{
		ID: 1, GroupID: 1, Version: 1, IdentityGeneration: 1,
		Fingerprint: "runtime-release", AuthState: state.CredentialAuthStateReady,
		EncryptedValue: "cipher",
	}}); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}
	if _, changed := registry.SetBlacklistedWithChange(1); !changed ||
		!registry.SetBlacklistReleaseAt(1, base.Add(-time.Second)) {
		t.Fatal("failed to seed expired credential blacklist")
	}
	releaseTicker := newFakeRuntimeTicker()
	created := make(chan time.Duration, 1)
	runtime := &Runtime{
		registry: registry,
		now:      func() time.Time { return base },
		newTicker: func(interval time.Duration) runtimeTicker {
			created <- interval
			if interval != blacklistReleaseInterval {
				testingPanic("unexpected ticker interval", interval)
			}
			return releaseTicker
		},
	}
	cancel, done := startRuntime(t, runtime)
	if interval := awaitValue(t, created); interval != blacklistReleaseInterval {
		t.Fatalf("release ticker interval = %v, want %v", interval, blacklistReleaseInterval)
	}
	awaitCondition(t, func() bool { return !registry.Snapshot()[0].Blacklisted })

	if _, changed := registry.SetBlacklistedWithChange(1); !changed ||
		!registry.SetBlacklistReleaseAt(1, base.Add(-time.Second)) {
		t.Fatal("failed to seed tick release")
	}
	releaseTicker.ticks <- base
	awaitCondition(t, func() bool { return !registry.Snapshot()[0].Blacklisted })

	cancel()
	awaitSignal(t, done)
	awaitSignal(t, releaseTicker.stopped)
}

func TestRuntimeBlacklistReleaseClearsCredentialHealthProblemState(t *testing.T) {
	t.Parallel()
	base := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := state.NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]state.CredentialEntry{{
		ID: 1, GroupID: 1, Version: 1, IdentityGeneration: 1,
		Fingerprint: "runtime-stats", AuthState: state.CredentialAuthStateReady,
		EncryptedValue: "cipher",
	}}); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}
	if _, changed := registry.SetBlacklistedWithChange(1); !changed ||
		!registry.SetBlacklistReleaseAt(1, base.Add(-time.Second)) {
		t.Fatal("failed to seed expired credential blacklist")
	}
	stats := health.NewStatsStore()
	stats.RecordFailure(1, health.FailureCategoryInvalidKey, 401, base)

	runtime := &Runtime{
		registry:    registry,
		healthStats: stats,
		now:         func() time.Time { return base },
	}
	runtime.releaseExpiredBlacklists(base)

	if registry.Snapshot()[0].Blacklisted {
		t.Fatal("expired credential remains blacklisted")
	}
	got := stats.Snapshot(1, base)
	if got.Problem != 1 || got.Failure != 1 || got.ConsecutiveProblem != 0 ||
		got.ConsecutiveFailure != 0 || got.LastFailureCategory != health.FailureCategoryAmbiguous ||
		got.LastStatusCode != 0 {
		t.Fatalf("health stats after release = %#v, want retained buckets with cleared problem state", got)
	}
}

func TestRuntimeBlacklistReleaseMaintenanceCancellationSkipsInitialRun(t *testing.T) {
	t.Parallel()
	base := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := state.NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]state.CredentialEntry{{
		ID: 1, GroupID: 1, Version: 1, IdentityGeneration: 1,
		Fingerprint: "runtime-cancel", AuthState: state.CredentialAuthStateReady,
		EncryptedValue: "cipher",
	}}); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}
	if _, changed := registry.SetBlacklistedWithChange(1); !changed ||
		!registry.SetBlacklistReleaseAt(1, base.Add(-time.Second)) {
		t.Fatal("failed to seed expired credential blacklist")
	}
	ticker := newFakeRuntimeTicker()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	done := make(chan struct{})
	go func() {
		defer close(done)
		runtime := &Runtime{registry: registry, now: func() time.Time { return base }}
		runtime.runBlacklistRelease(ctx, ticker)
	}()
	awaitSignal(t, done)
	awaitSignal(t, ticker.stopped)
	if !registry.Snapshot()[0].Blacklisted {
		t.Fatal("canceled release maintenance performed an initial release")
	}
}

func awaitCondition(t *testing.T, condition func() bool) {
	t.Helper()
	deadline := time.NewTimer(time.Second)
	defer deadline.Stop()
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for {
		if condition() {
			return
		}
		select {
		case <-deadline.C:
			t.Fatal("timed out waiting for condition")
		case <-ticker.C:
		}
	}
}

func testingPanic(message string, value time.Duration) { panic(message + ": " + value.String()) }

func startRuntime(t *testing.T, runtime *Runtime) (context.CancelFunc, <-chan struct{}) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		runtime.Run(ctx)
	}()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Errorf("Runtime.Run did not return during cleanup")
		}
	})
	return cancel, done
}

func stopRuntime(t *testing.T, cancel context.CancelFunc, done <-chan struct{}) {
	t.Helper()
	cancel()
	awaitSignal(t, done)
}

func awaitValue[T any](t *testing.T, channel <-chan T) T {
	t.Helper()
	select {
	case value := <-channel:
		return value
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for channel value")
		var zero T
		return zero
	}
}

func awaitSignal(t *testing.T, channel <-chan struct{}) {
	t.Helper()
	select {
	case <-channel:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for signal")
	}
}
