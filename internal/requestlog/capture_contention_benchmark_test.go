//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package requestlog

import (
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"gorm.io/gorm"
	"gpt-load/internal/debugcapture"
	"gpt-load/internal/storage"
	"gpt-load/internal/storage/models"
)

type measuredWriter struct {
	inner     batchWriter
	mu        sync.Mutex
	times     []time.Duration
	starts    map[string]time.Time
	latencies []time.Duration
	events    []measuredEvent
	windows   [][2]time.Time
}

type measuredEvent struct {
	started time.Time
	latency time.Duration
}

func (w *measuredWriter) WriteBatch(ctx context.Context, rows []models.RequestLog) error {
	start := time.Now()
	err := w.inner.WriteBatch(ctx, rows)
	end := time.Now()
	w.mu.Lock()
	w.times = append(w.times, end.Sub(start))
	w.windows = append(w.windows, [2]time.Time{start, end})
	if err == nil {
		for _, row := range rows {
			if at, ok := w.starts[row.ID]; ok {
				w.latencies = append(w.latencies, end.Sub(at))
				w.events = append(w.events, measuredEvent{at, end.Sub(at)})
			}
		}
	}
	w.mu.Unlock()
	return err
}

func durationPercentile(values []time.Duration, p int) time.Duration {
	if len(values) == 0 {
		return 0
	}
	xs := append([]time.Duration(nil), values...)
	sort.Slice(xs, func(i, j int) bool { return xs[i] < xs[j] })
	return xs[int(math.Ceil(float64(len(xs))*float64(p)/100))-1]
}

// Exercise the real cleanup while the request-log worker is running. Cleanup
// is explicitly triggered, not the runtime timer; all input is synthetic.
func TestSyntheticRunningCleanupContention(t *testing.T) {
	if os.Getenv("GPTLOAD_CAPTURE_BENCH") != "1" {
		t.Skip("opt-in synthetic benchmark")
	}
	for trial := 1; trial <= 5; trial++ {
		for _, clean := range []bool{trial%2 == 0, trial%2 != 0} {
			t.Run(fmt.Sprintf("trial-%d-clean-%v", trial, clean), func(t *testing.T) {
				db, err := storage.Open(filepath.Join(t.TempDir(), "bench.db"))
				if err != nil {
					t.Fatal(err)
				}
				sqlDB, err := db.DB()
				if err != nil {
					t.Fatal(err)
				}
				defer sqlDB.Close()
				if err = storage.AutoMigrate(db); err != nil {
					t.Fatal(err)
				}
				err = db.Transaction(func(tx *gorm.DB) error {
					for i := 0; i < 500; i++ {
						id := fmt.Sprintf("capture-%d", i)
						aid := fmt.Sprintf("attempt-%d", i)
						if e := tx.Exec("INSERT INTO debug_captures (id,created_at_ms,expires_at_ms,state,error) VALUES (?,0,1,'completed','')", id).Error; e != nil {
							return e
						}
						if e := tx.Exec("INSERT INTO debug_capture_attempts (id,capture_id,sequence,started_at_ms,state,error) VALUES (?,?,1,0,'completed','')", aid, id).Error; e != nil {
							return e
						}
						for j := 0; j < 100; j++ {
							if e := tx.Exec("INSERT INTO debug_capture_chunks (capture_id,attempt_id,part,direction,data,created_at_ms) VALUES (?,?,'body','request',zeroblob(4096),0)", id, aid).Error; e != nil {
								return e
							}
						}
					}
					return nil
				})
				if err != nil {
					t.Fatal(err)
				}
				store, err := debugcapture.New(db)
				if err != nil {
					t.Fatal(err)
				}
				service := NewService(db, nil, staticRetentionPolicy{days: 7})
				logger := logrus.New()
				logger.SetOutput(io.Discard)
				service.logger = logger
				writer := &measuredWriter{inner: service.writer, starts: make(map[string]time.Time)}
				service.writer = writer
				if err = service.Start(); err != nil {
					t.Fatal(err)
				}
				type cleanupResult struct {
					start, end time.Time
					removed    int
					err        error
				}
				done := make(chan cleanupResult, 1)
				var trigger time.Time
				ticker := time.NewTicker(time.Millisecond)
				defer ticker.Stop()
				begin := time.Now()
				for i := 0; i < 10000; i++ {
					<-ticker.C
					if i == 1000 {
						trigger = time.Now()
						if clean {
							go func() { start := time.Now(); n, e := store.Cleanup(); done <- cleanupResult{start, time.Now(), n, e} }()
						}
					}
					id := aggregationRequestID(i)
					writer.mu.Lock()
					writer.starts[id] = time.Now()
					writer.mu.Unlock()
					service.Emit(testEvent(id))
				}
				var result cleanupResult
				if clean {
					result = <-done
					if result.err != nil || result.removed != 500 {
						t.Fatalf("cleanup removed=%d err=%v", result.removed, result.err)
					}
				}
				duration := time.Since(begin)
				ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
				defer cancel()
				if err = service.Stop(ctx); err != nil {
					t.Fatal(err)
				}
				stats := service.Stats()
				var count int64
				if err = db.Model(&models.RequestLog{}).Count(&count).Error; err != nil {
					t.Fatal(err)
				}
				var remaining int64
				if err = db.Model(&models.DebugCapture{}).Count(&remaining).Error; err != nil {
					t.Fatal(err)
				}
				if (clean && remaining != 0) || (!clean && remaining != 500) {
					t.Fatalf("remaining=%d", remaining)
				}
				overlaps := 0
				for _, w := range writer.windows {
					if clean && w[0].Before(result.end) && w[1].After(result.start) {
						overlaps++
					}
				}
				var near []time.Duration
				var window []time.Duration
				writer.mu.Lock()
				for _, v := range writer.latencies {
					near = append(near, v)
				}
				for _, e := range writer.events {
					if !e.started.Before(trigger) && e.started.Before(trigger.Add(2*time.Second)) {
						window = append(window, e.latency)
					}
				}
				writer.mu.Unlock()
				t.Logf("trial=%d clean=%v events=10000 elapsed=%s trigger=%s cleanup=%s removed=%d remaining=%d batch_count=%d overlap_batches=%d success_samples=%d event_p95=%s event_p99=%s window_samples=%d window_p95=%s window_p99=%s persisted=%d dropped_queue=%d dropped_failed=%d dropped_shutdown=%d", trial, clean, duration, trigger.Sub(begin), result.end.Sub(result.start), result.removed, remaining, len(writer.windows), overlaps, len(near), durationPercentile(near, 95), durationPercentile(near, 99), len(window), durationPercentile(window, 95), durationPercentile(window, 99), count, stats.DroppedQueueFullTotal, stats.DroppedPersistFailedTotal, stats.DroppedShutdownTotal)
			})
		}
	}
}
func (w *measuredWriter) percentile(p int) time.Duration {
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.times) == 0 {
		return 0
	}
	xs := append([]time.Duration(nil), w.times...)
	sort.Slice(xs, func(i, j int) bool { return xs[i] < xs[j] })
	return xs[int(math.Ceil(float64(len(xs))*float64(p)/100))-1]
}

// Opt-in isolated integration workload, not a normal test or production database.
func TestSyntheticCaptureContention(t *testing.T) {
	if os.Getenv("GPTLOAD_CAPTURE_BENCH") != "1" {
		t.Skip("opt-in synthetic benchmark")
	}
	for _, backlog := range []int{0, 500} {
		for trial := 1; trial <= 3; trial++ {
			t.Run(fmt.Sprintf("backlog-%d-trial-%d", backlog, trial), func(t *testing.T) {
				db, err := storage.Open(filepath.Join(t.TempDir(), "bench.db"))
				if err != nil {
					t.Fatal(err)
				}
				sqlDB, _ := db.DB()
				defer sqlDB.Close()
				if err = storage.AutoMigrate(db); err != nil {
					t.Fatal(err)
				}
				if backlog > 0 {
					err = db.Transaction(func(tx *gorm.DB) error {
						for i := 0; i < backlog; i++ {
							id := fmt.Sprintf("capture-%d", i)
							attempt := fmt.Sprintf("attempt-%d", i)
							if e := tx.Exec("INSERT INTO debug_captures (id,created_at_ms,expires_at_ms,state,error) VALUES (?,0,1,'completed','')", id).Error; e != nil {
								return e
							}
							if e := tx.Exec("INSERT INTO debug_capture_attempts (id,capture_id,sequence,started_at_ms,state,error) VALUES (?,?,1,0,'completed','')", attempt, id).Error; e != nil {
								return e
							}
							for j := 0; j < 100; j++ {
								if e := tx.Exec("INSERT INTO debug_capture_chunks (capture_id,attempt_id,part,direction,data,created_at_ms) VALUES (?,?,'body','request',zeroblob(4096),0)", id, attempt).Error; e != nil {
									return e
								}
							}
						}
						return nil
					})
					if err != nil {
						t.Fatal(err)
					}
				}
				store, err := debugcapture.New(db)
				if err != nil {
					t.Fatal(err)
				}
				runtime := debugcapture.NewRuntime(true, store)
				service := NewService(db, nil, staticRetentionPolicy{days: 7})
				logger := logrus.New()
				logger.SetOutput(io.Discard)
				service.logger = logger
				writer := &measuredWriter{inner: service.writer}
				service.writer = writer
				if err = service.Start(); err != nil {
					t.Fatal(err)
				}
				type startResult struct {
					elapsed time.Duration
					err     error
				}
				done := make(chan startResult, 1)
				go func() { start := time.Now(); err := runtime.Start(); done <- startResult{time.Since(start), err} }()
				for i := 0; i < 500; i++ {
					service.Emit(testEvent(aggregationRequestID(i)))
				}
				ticker := time.NewTicker(2 * time.Millisecond)
				for i := 500; i < 5000; i++ {
					<-ticker.C
					service.Emit(testEvent(aggregationRequestID(i)))
				}
				ticker.Stop()
				result := <-done
				if result.err != nil {
					t.Fatal(result.err)
				}
				startup := result.elapsed
				health, err := runtime.Health()
				if err != nil || health.SweepFailureTotal != 0 || health.RemovedTotal != uint64(backlog) {
					t.Fatalf("startup sweep: health=%+v err=%v", health, err)
				}
				ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
				defer cancel()
				if err = service.Stop(ctx); err != nil {
					t.Fatal(err)
				}
				if err = runtime.Stop(ctx); err != nil {
					t.Fatal(err)
				}
				stats := service.Stats()
				t.Logf("backlog=%d chunks=%d initial_burst=500 rate=500/s events=5000 runtime_start=%s write_batch_n=%d p95=%s p99=%s stats=%+v", backlog, backlog*100, startup, len(writer.times), writer.percentile(95), writer.percentile(99), stats)
			})
		}
	}
}
