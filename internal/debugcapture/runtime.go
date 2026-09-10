//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package debugcapture

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

const defaultCleanupInterval = time.Hour

// Runtime owns cleanup and shutdown admission for the optional capture store.
// Capture requests remain independent of the cleanup worker: sweep failures are
// recorded for health reporting and never returned to the data plane.
type Runtime struct {
	store    *Store
	enabled  bool
	interval time.Duration
	now      func() time.Time

	mu             sync.Mutex
	started        bool
	stopping       bool
	startFailed    bool
	stop           chan struct{}
	workerDone     chan struct{}
	startupDone    chan struct{}
	sessionsDone   chan struct{}
	activeSessions int

	sweepTotal        atomic.Uint64
	removedTotal      atomic.Uint64
	sweepFailureTotal atomic.Uint64
	statsMu           sync.RWMutex
	lastSweepAt       time.Time
	lastSweepFailure  time.Time
}

func NewRuntime(enabled bool, store *Store) *Runtime {
	return NewRuntimeWithInterval(enabled, store, defaultCleanupInterval)
}

func NewRuntimeWithInterval(enabled bool, store *Store, interval time.Duration) *Runtime {
	if interval <= 0 {
		interval = defaultCleanupInterval
	}
	return &Runtime{
		store:    store,
		enabled:  enabled,
		interval: interval,
		now:      time.Now,
	}
}

func RetentionPeriod() time.Duration { return retention }

func (r *Runtime) Start() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	if r.started {
		r.mu.Unlock()
		return errors.New("debug capture runtime is already started")
	}
	if r.stopping {
		r.mu.Unlock()
		return errors.New("debug capture runtime is stopping")
	}
	if r.startFailed {
		r.mu.Unlock()
		return errors.New("debug capture runtime failed to start")
	}
	if r.enabled && r.store == nil {
		r.startFailed = true
		r.mu.Unlock()
		return errors.New("debug capture runtime store is required")
	}
	r.started = true
	if !r.enabled {
		r.mu.Unlock()
		r.sweep()
		return nil
	}
	r.stop = make(chan struct{})
	r.workerDone = make(chan struct{})
	r.startupDone = make(chan struct{})
	r.sessionsDone = make(chan struct{})
	close(r.sessionsDone)
	stop := r.stop
	workerDone := r.workerDone
	startupDone := r.startupDone
	r.mu.Unlock()

	go r.run(stop, workerDone, startupDone)
	<-startupDone
	return nil
}

func (r *Runtime) run(stop <-chan struct{}, workerDone chan<- struct{}, startupDone chan<- struct{}) {
	defer close(workerDone)
	r.sweep()
	close(startupDone)
	ticker := time.NewTicker(r.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			r.sweep()
		case <-stop:
			return
		}
	}
}

func (r *Runtime) sweep() {
	if r == nil || r.store == nil {
		return
	}
	removed, err := r.store.Cleanup()
	r.sweepTotal.Add(1)
	r.removedTotal.Add(uint64(maxInt(removed, 0)))
	now := r.now().UTC()
	r.statsMu.Lock()
	r.lastSweepAt = now
	if err != nil {
		r.sweepFailureTotal.Add(1)
		r.lastSweepFailure = now
	}
	r.statsMu.Unlock()
}

// AcquireSession prevents shutdown from closing the database while a capture
// session is still being finalized. The returned release function is idempotent.
func (r *Runtime) AcquireSession() (func(), bool) {
	if r == nil || !r.enabled {
		return nil, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.stopping || r.startFailed || !r.started {
		return nil, false
	}
	if r.activeSessions == 0 {
		r.sessionsDone = make(chan struct{})
	}
	r.activeSessions++
	var once sync.Once
	return func() {
		once.Do(func() {
			r.mu.Lock()
			defer r.mu.Unlock()
			if r.activeSessions == 0 {
				return
			}
			r.activeSessions--
			if r.activeSessions == 0 && r.sessionsDone != nil {
				close(r.sessionsDone)
			}
		})
	}, true
}

func (r *Runtime) Stop(ctx context.Context) error {
	if r == nil {
		return nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	r.mu.Lock()
	if !r.enabled || !r.started {
		r.mu.Unlock()
		return nil
	}
	if !r.stopping {
		r.stopping = true
		close(r.stop)
	}
	workerDone := r.workerDone
	sessionsDone := r.sessionsDone
	r.mu.Unlock()

	if err := waitChannel(ctx, workerDone); err != nil {
		return err
	}
	if err := waitChannel(ctx, sessionsDone); err != nil {
		return err
	}
	return nil
}

func waitChannel(ctx context.Context, done <-chan struct{}) error {
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (r *Runtime) Health() (Health, error) {
	if r == nil {
		return Health{}, nil
	}
	r.mu.Lock()
	running := r.enabled && r.started && !r.stopping && !r.startFailed
	r.mu.Unlock()
	r.statsMu.RLock()
	lastSweepAt := r.lastSweepAt
	lastFailureAt := r.lastSweepFailure
	r.statsMu.RUnlock()
	result := Health{
		Enabled:           r.enabled,
		Running:           running,
		RetentionSeconds:  int64(retention / time.Second),
		SweepTotal:        r.sweepTotal.Load(),
		RemovedTotal:      r.removedTotal.Load(),
		SweepFailureTotal: r.sweepFailureTotal.Load(),
	}
	if !lastSweepAt.IsZero() {
		result.LastSweepAt = &lastSweepAt
	}
	if !lastFailureAt.IsZero() {
		result.LastFailureAt = &lastFailureAt
	}
	if !r.enabled || r.store == nil {
		return result, nil
	}
	counts, err := r.store.Counts()
	if err != nil {
		result.Error = "counts_unavailable"
		return result, nil
	}
	result.Active = counts.Active
	result.Completed = counts.Completed
	result.Failed = counts.Failed
	return result, nil
}

func maxInt(value, floor int) int {
	if value < floor {
		return floor
	}
	return value
}
