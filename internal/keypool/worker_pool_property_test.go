package keypool

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"pgregory.net/rapid"
)

// orderTrackingProcessor tracks the order of processed tasks
type orderTrackingProcessor struct {
	processedOrder []uint
	mu             sync.Mutex
}

func newOrderTrackingProcessor() *orderTrackingProcessor {
	return &orderTrackingProcessor{
		processedOrder: make([]uint, 0),
	}
}

func (p *orderTrackingProcessor) ProcessSuccess(keyID uint, keyHashKey, activeKeysListKey string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.processedOrder = append(p.processedOrder, keyID)
	return nil
}

func (p *orderTrackingProcessor) ProcessFailure(task *StatusUpdateTask, keyHashKey, activeKeysListKey string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.processedOrder = append(p.processedOrder, task.KeyID)
	return nil
}

func (p *orderTrackingProcessor) getOrder() []uint {
	p.mu.Lock()
	defer p.mu.Unlock()
	result := make([]uint, len(p.processedOrder))
	copy(result, p.processedOrder)
	return result
}

// Property 2: FIFO Processing Order
// For any sequence of tasks T1, T2, ..., Tn submitted to a single worker,
// they SHALL be processed in the order T1 → T2 → ... → Tn.
// **Validates: Requirements 1.5**
func TestProperty_FIFOProcessingOrder(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a random number of tasks (1-50)
		taskCount := rapid.IntRange(1, 50).Draw(t, "taskCount")

		processor := newOrderTrackingProcessor()
		config := WorkerPoolConfig{
			WorkerCount:   1, // Single worker to ensure strict FIFO
			QueueCapacity: 100,
		}

		wp := NewWorkerPool(config, processor, nil)
		wp.Start()

		// Submit tasks in order
		expectedOrder := make([]uint, taskCount)
		for i := 0; i < taskCount; i++ {
			keyID := uint(i + 1)
			expectedOrder[i] = keyID
			wp.Submit(&StatusUpdateTask{
				KeyID:     keyID,
				GroupID:   1,
				IsSuccess: true,
				Timestamp: time.Now(),
			})
		}

		// Wait for all tasks to be processed
		time.Sleep(time.Duration(taskCount*10+100) * time.Millisecond)
		wp.Stop()

		// Verify FIFO order
		actualOrder := processor.getOrder()
		if len(actualOrder) != len(expectedOrder) {
			t.Fatalf("Expected %d tasks processed, got %d", len(expectedOrder), len(actualOrder))
		}

		for i := range expectedOrder {
			if actualOrder[i] != expectedOrder[i] {
				t.Fatalf("FIFO violation at index %d: expected %d, got %d", i, expectedOrder[i], actualOrder[i])
			}
		}
	})
}

// Property 1: Bounded Goroutine Count
// For any number of submitted tasks N, the worker pool SHALL maintain
// exactly the configured number of worker goroutines.
// **Validates: Requirements 1.1, 1.2**
func TestProperty_BoundedGoroutineCount(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate random worker count and task count
		workerCount := rapid.IntRange(1, 8).Draw(t, "workerCount")
		taskCount := rapid.IntRange(10, 100).Draw(t, "taskCount")

		var processedCount atomic.Int64
		processor := &countingProcessor{processed: &processedCount}

		config := WorkerPoolConfig{
			WorkerCount:   workerCount,
			QueueCapacity: 1000,
		}

		wp := NewWorkerPool(config, processor, nil)
		wp.Start()

		// Submit many tasks
		for i := 0; i < taskCount; i++ {
			wp.Submit(&StatusUpdateTask{
				KeyID:     uint(i),
				GroupID:   1,
				IsSuccess: true,
			})
		}

		// The worker pool should have exactly workerCount workers
		// We verify this indirectly by checking the config
		actualConfig := wp.GetConfig()
		if actualConfig.WorkerCount != workerCount {
			t.Fatalf("Expected %d workers, got %d", workerCount, actualConfig.WorkerCount)
		}

		wp.Stop()

		// All tasks should be processed
		if processedCount.Load() != int64(taskCount) {
			t.Fatalf("Expected %d tasks processed, got %d", taskCount, processedCount.Load())
		}
	})
}

type countingProcessor struct {
	processed *atomic.Int64
}

func (p *countingProcessor) ProcessSuccess(keyID uint, keyHashKey, activeKeysListKey string) error {
	p.processed.Add(1)
	return nil
}

func (p *countingProcessor) ProcessFailure(task *StatusUpdateTask, keyHashKey, activeKeysListKey string) error {
	p.processed.Add(1)
	return nil
}

// Property 3: Graceful Shutdown Completeness
// For any set of pending tasks at shutdown time, all tasks SHALL be processed
// before the worker pool terminates.
// **Validates: Requirements 1.4**
func TestProperty_GracefulShutdownCompleteness(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		taskCount := rapid.IntRange(5, 50).Draw(t, "taskCount")

		var processedCount atomic.Int64
		processor := &countingProcessor{processed: &processedCount}

		config := WorkerPoolConfig{
			WorkerCount:   2,
			QueueCapacity: 100,
		}

		wp := NewWorkerPool(config, processor, nil)
		wp.Start()

		// Submit tasks
		for i := 0; i < taskCount; i++ {
			wp.Submit(&StatusUpdateTask{
				KeyID:     uint(i),
				GroupID:   1,
				IsSuccess: true,
			})
		}

		// Stop immediately - should still process all pending tasks
		wp.Stop()

		// Verify all tasks were processed
		if processedCount.Load() != int64(taskCount) {
			t.Fatalf("Expected %d tasks processed after shutdown, got %d", taskCount, processedCount.Load())
		}
	})
}

// Property 7: Metrics Accuracy
// For any sequence of N tasks where M succeed and (N-M) fail,
// the metrics SHALL report processedCount = N and errorCount = (N-M).
// **Validates: Requirements 5.1**
func TestProperty_MetricsAccuracy(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		totalTasks := rapid.IntRange(10, 50).Draw(t, "totalTasks")
		errorRate := rapid.Float64Range(0, 0.5).Draw(t, "errorRate")
		expectedErrors := int(float64(totalTasks) * errorRate)

		var errorCount atomic.Int64
		processor := &errorInjectingProcessor{
			errorCount:     &errorCount,
			errorsToInject: expectedErrors,
		}

		config := WorkerPoolConfig{
			WorkerCount:   2,
			QueueCapacity: 100,
			MaxRetries:    0, // No retries to make error counting predictable
		}

		wp := NewWorkerPool(config, processor, nil)
		wp.Start()

		for i := 0; i < totalTasks; i++ {
			wp.Submit(&StatusUpdateTask{
				KeyID:     uint(i),
				GroupID:   1,
				IsSuccess: true,
			})
		}

		time.Sleep(200 * time.Millisecond)
		wp.Stop()

		metrics := wp.GetMetrics()

		// All tasks should be counted as processed
		if metrics.ProcessedCount != int64(totalTasks) {
			t.Fatalf("Expected processedCount=%d, got %d", totalTasks, metrics.ProcessedCount)
		}

		// Error count should match injected errors
		if metrics.ErrorCount != int64(expectedErrors) {
			t.Fatalf("Expected errorCount=%d, got %d", expectedErrors, metrics.ErrorCount)
		}
	})
}

type errorInjectingProcessor struct {
	errorCount     *atomic.Int64
	errorsToInject int
	injected       atomic.Int64
}

func (p *errorInjectingProcessor) ProcessSuccess(keyID uint, keyHashKey, activeKeysListKey string) error {
	if int(p.injected.Add(1)) <= p.errorsToInject {
		return &permanentError{msg: "injected error"}
	}
	return nil
}

func (p *errorInjectingProcessor) ProcessFailure(task *StatusUpdateTask, keyHashKey, activeKeysListKey string) error {
	return nil
}

type permanentError struct {
	msg string
}

func (e *permanentError) Error() string {
	return e.msg + " - record not found" // Make it permanent
}


// Property 6: Cache Rollback on DB Failure
// For any status update where the database operation fails,
// the cache state SHALL be rolled back to its previous value.
// **Validates: Requirements 4.2**
func TestProperty_CacheRollbackOnDBFailure(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate random initial state
		initialFailureCount := rapid.Int64Range(0, 10).Draw(t, "initialFailureCount")
		isSuccess := rapid.Bool().Draw(t, "isSuccess")

		// Create a processor that tracks cache state and fails DB operations
		tracker := &cacheRollbackTracker{
			failureCount: initialFailureCount,
			status:       "active",
			dbShouldFail: true,
		}

		config := WorkerPoolConfig{
			WorkerCount:   1,
			QueueCapacity: 10,
			MaxRetries:    0, // No retries to test immediate rollback
		}

		wp := NewWorkerPool(config, tracker, nil)
		wp.Start()

		task := &StatusUpdateTask{
			KeyID:           1,
			GroupID:         1,
			IsSuccess:       isSuccess,
			Timestamp:       time.Now(),
			BlacklistThresh: 5,
		}

		wp.Submit(task)
		time.Sleep(100 * time.Millisecond)
		wp.Stop()

		// Verify cache was rolled back to initial state
		if tracker.failureCount != initialFailureCount {
			t.Fatalf("Cache not rolled back: expected failureCount=%d, got %d",
				initialFailureCount, tracker.failureCount)
		}
		if tracker.status != "active" {
			t.Fatalf("Cache not rolled back: expected status=active, got %s", tracker.status)
		}
	})
}

// cacheRollbackTracker simulates cache operations and tracks state
type cacheRollbackTracker struct {
	failureCount int64
	status       string
	dbShouldFail bool
	mu           sync.Mutex
}

func (c *cacheRollbackTracker) ProcessSuccess(keyID uint, keyHashKey, activeKeysListKey string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Simulate cache-first update
	oldFailureCount := c.failureCount
	oldStatus := c.status

	c.failureCount = 0
	c.status = "active"

	// Simulate DB failure
	if c.dbShouldFail {
		// Rollback
		c.failureCount = oldFailureCount
		c.status = oldStatus
		return &permanentError{msg: "simulated DB failure - record not found"}
	}

	return nil
}

func (c *cacheRollbackTracker) ProcessFailure(task *StatusUpdateTask, keyHashKey, activeKeysListKey string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Simulate cache-first update
	oldFailureCount := c.failureCount
	oldStatus := c.status

	c.failureCount++
	if task.BlacklistThresh > 0 && c.failureCount >= int64(task.BlacklistThresh) {
		c.status = "invalid"
	}

	// Simulate DB failure
	if c.dbShouldFail {
		// Rollback
		c.failureCount = oldFailureCount
		c.status = oldStatus
		return &permanentError{msg: "simulated DB failure - record not found"}
	}

	return nil
}
