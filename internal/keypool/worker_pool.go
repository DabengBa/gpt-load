package keypool

import (
	"sync"
	"sync/atomic"
	"time"

	"github.com/sirupsen/logrus"
)

// StatusUpdateTask represents a key status update task
type StatusUpdateTask struct {
	KeyID           uint
	GroupID         uint
	IsSuccess       bool
	ErrorMessage    string
	Timestamp       time.Time
	RetryCount      int
	BlacklistThresh int // Blacklist threshold from group config
}

// WorkerPoolConfig holds configuration for the worker pool
type WorkerPoolConfig struct {
	WorkerCount    int           // Number of worker goroutines (default: 4)
	QueueCapacity  int           // Task queue capacity (default: 10000)
	MaxRetries     int           // Max retry attempts for transient errors (default: 3)
	RetryBaseDelay time.Duration // Base delay for exponential backoff (default: 100ms)
}

// DefaultWorkerPoolConfig returns default configuration
func DefaultWorkerPoolConfig() WorkerPoolConfig {
	return WorkerPoolConfig{
		WorkerCount:    4,
		QueueCapacity:  10000,
		MaxRetries:     3,
		RetryBaseDelay: 100 * time.Millisecond,
	}
}

// WorkerPoolMetrics holds metrics for monitoring
type WorkerPoolMetrics struct {
	QueueLength    int64
	ProcessedCount int64
	ErrorCount     int64
	DroppedCount   int64
}

// TaskProcessor defines the interface for processing status updates
type TaskProcessor interface {
	ProcessSuccess(keyID uint, keyHashKey, activeKeysListKey string) error
	ProcessFailure(task *StatusUpdateTask, keyHashKey, activeKeysListKey string) error
}


// WorkerPool manages a fixed number of workers for processing status update tasks
type WorkerPool struct {
	config    WorkerPoolConfig
	taskChan  chan *StatusUpdateTask
	stopChan  chan struct{}
	wg        sync.WaitGroup
	processor TaskProcessor
	logger    *logrus.Entry

	// Metrics (atomic for thread-safety)
	queueLength    atomic.Int64
	processedCount atomic.Int64
	errorCount     atomic.Int64
	droppedCount   atomic.Int64

	// State
	running atomic.Bool
}

// NewWorkerPool creates a new worker pool with the given configuration
func NewWorkerPool(config WorkerPoolConfig, processor TaskProcessor, logger *logrus.Entry) *WorkerPool {
	if config.WorkerCount <= 0 {
		config.WorkerCount = DefaultWorkerPoolConfig().WorkerCount
	}
	if config.QueueCapacity <= 0 {
		config.QueueCapacity = DefaultWorkerPoolConfig().QueueCapacity
	}
	if config.MaxRetries < 0 {
		config.MaxRetries = DefaultWorkerPoolConfig().MaxRetries
	}
	if config.RetryBaseDelay <= 0 {
		config.RetryBaseDelay = DefaultWorkerPoolConfig().RetryBaseDelay
	}

	if logger == nil {
		logger = logrus.NewEntry(logrus.StandardLogger())
	}

	return &WorkerPool{
		config:    config,
		taskChan:  make(chan *StatusUpdateTask, config.QueueCapacity),
		stopChan:  make(chan struct{}),
		processor: processor,
		logger:    logger.WithField("component", "worker_pool"),
	}
}

// Start launches the worker goroutines
func (wp *WorkerPool) Start() {
	if wp.running.Swap(true) {
		wp.logger.Warn("Worker pool already running")
		return
	}

	wp.logger.WithFields(logrus.Fields{
		"worker_count":   wp.config.WorkerCount,
		"queue_capacity": wp.config.QueueCapacity,
	}).Info("Starting worker pool")

	for i := 0; i < wp.config.WorkerCount; i++ {
		wp.wg.Add(1)
		go wp.worker(i)
	}
}

// Submit adds a task to the queue
// If the queue is full, it will block and process the task synchronously to ensure
// status updates are never dropped
func (wp *WorkerPool) Submit(task *StatusUpdateTask) bool {
	if !wp.running.Load() {
		wp.logger.Warn("Cannot submit task: worker pool not running")
		return false
	}

	select {
	case wp.taskChan <- task:
		wp.queueLength.Add(1)
		// Check queue capacity warning threshold (80%)
		currentLen := wp.queueLength.Load()
		threshold := int64(float64(wp.config.QueueCapacity) * 0.8)
		if currentLen >= threshold {
			wp.logger.WithFields(logrus.Fields{
				"queue_length": currentLen,
				"capacity":     wp.config.QueueCapacity,
			}).Warn("Task queue approaching capacity")
		}
		return true
	default:
		// Queue is full - process synchronously to ensure status update is not lost
		wp.logger.WithFields(logrus.Fields{
			"key_id":   task.KeyID,
			"group_id": task.GroupID,
		}).Warn("Task queue full, processing synchronously to avoid dropping status update")
		wp.processTaskSync(task)
		return true
	}
}

// processTaskSync processes a task synchronously when the queue is full
// This ensures status updates are never dropped
func (wp *WorkerPool) processTaskSync(task *StatusUpdateTask) {
	keyHashKey := formatKeyHashKey(task.KeyID)
	activeKeysListKey := formatActiveKeysListKey(task.GroupID)

	var err error
	for attempt := 0; attempt <= wp.config.MaxRetries; attempt++ {
		if attempt > 0 {
			delay := wp.config.RetryBaseDelay * time.Duration(1<<(attempt-1))
			wp.logger.WithFields(logrus.Fields{
				"key_id":  task.KeyID,
				"attempt": attempt,
				"delay":   delay,
			}).Debug("Retrying synchronous task")
			time.Sleep(delay)
		}

		if task.IsSuccess {
			err = wp.processor.ProcessSuccess(task.KeyID, keyHashKey, activeKeysListKey)
		} else {
			err = wp.processor.ProcessFailure(task, keyHashKey, activeKeysListKey)
		}

		if err == nil {
			wp.processedCount.Add(1)
			return
		}

		// Check if error is permanent (non-retryable)
		if isPermanentError(err) {
			wp.logger.WithFields(logrus.Fields{
				"key_id": task.KeyID,
				"error":  err,
			}).Error("Permanent error processing synchronous task, not retrying")
			wp.errorCount.Add(1)
			wp.processedCount.Add(1)
			return
		}

		wp.logger.WithFields(logrus.Fields{
			"key_id":  task.KeyID,
			"attempt": attempt + 1,
			"error":   err,
		}).Warn("Transient error processing synchronous task")
	}

	// All retries exhausted
	wp.logger.WithFields(logrus.Fields{
		"key_id":      task.KeyID,
		"max_retries": wp.config.MaxRetries,
		"error":       err,
	}).Error("All retries exhausted for synchronous task")
	wp.errorCount.Add(1)
	wp.processedCount.Add(1)
}

// Stop gracefully shuts down the worker pool
func (wp *WorkerPool) Stop() {
	if !wp.running.Swap(false) {
		wp.logger.Warn("Worker pool already stopped")
		return
	}

	wp.logger.Info("Stopping worker pool...")

	// Signal workers to stop
	close(wp.stopChan)

	// Wait for all workers to finish
	wp.wg.Wait()

	// Drain remaining tasks
	wp.drainRemainingTasks()

	wp.logger.WithFields(logrus.Fields{
		"processed": wp.processedCount.Load(),
		"errors":    wp.errorCount.Load(),
		"dropped":   wp.droppedCount.Load(),
	}).Info("Worker pool stopped")
}

// GetMetrics returns current metrics snapshot
func (wp *WorkerPool) GetMetrics() WorkerPoolMetrics {
	return WorkerPoolMetrics{
		QueueLength:    wp.queueLength.Load(),
		ProcessedCount: wp.processedCount.Load(),
		ErrorCount:     wp.errorCount.Load(),
		DroppedCount:   wp.droppedCount.Load(),
	}
}

// GetConfig returns the worker pool configuration
func (wp *WorkerPool) GetConfig() WorkerPoolConfig {
	return wp.config
}

// IsRunning returns whether the worker pool is running
func (wp *WorkerPool) IsRunning() bool {
	return wp.running.Load()
}


// worker is the main loop for each worker goroutine
func (wp *WorkerPool) worker(id int) {
	defer wp.wg.Done()

	logger := wp.logger.WithField("worker_id", id)
	logger.Debug("Worker started")

	for {
		select {
		case <-wp.stopChan:
			logger.Debug("Worker received stop signal")
			return
		case task, ok := <-wp.taskChan:
			if !ok {
				logger.Debug("Task channel closed")
				return
			}
			wp.queueLength.Add(-1)
			wp.processTask(task, logger)
		}
	}
}

// processTask handles a single task with retry logic
func (wp *WorkerPool) processTask(task *StatusUpdateTask, logger *logrus.Entry) {
	keyHashKey := formatKeyHashKey(task.KeyID)
	activeKeysListKey := formatActiveKeysListKey(task.GroupID)

	var err error
	for attempt := 0; attempt <= wp.config.MaxRetries; attempt++ {
		if attempt > 0 {
			delay := wp.config.RetryBaseDelay * time.Duration(1<<(attempt-1))
			logger.WithFields(logrus.Fields{
				"key_id":  task.KeyID,
				"attempt": attempt,
				"delay":   delay,
			}).Debug("Retrying task")
			time.Sleep(delay)
		}

		if task.IsSuccess {
			err = wp.processor.ProcessSuccess(task.KeyID, keyHashKey, activeKeysListKey)
		} else {
			err = wp.processor.ProcessFailure(task, keyHashKey, activeKeysListKey)
		}

		if err == nil {
			wp.processedCount.Add(1)
			return
		}

		// Check if error is permanent (non-retryable)
		if isPermanentError(err) {
			logger.WithFields(logrus.Fields{
				"key_id": task.KeyID,
				"error":  err,
			}).Error("Permanent error processing task, not retrying")
			wp.errorCount.Add(1)
			wp.processedCount.Add(1)
			return
		}

		logger.WithFields(logrus.Fields{
			"key_id":  task.KeyID,
			"attempt": attempt + 1,
			"error":   err,
		}).Warn("Transient error processing task")
	}

	// All retries exhausted
	logger.WithFields(logrus.Fields{
		"key_id":      task.KeyID,
		"max_retries": wp.config.MaxRetries,
		"error":       err,
	}).Error("All retries exhausted for task")
	wp.errorCount.Add(1)
	wp.processedCount.Add(1)
}

// drainRemainingTasks processes any tasks left in the queue after stop signal
func (wp *WorkerPool) drainRemainingTasks() {
	remaining := 0
	for {
		select {
		case task, ok := <-wp.taskChan:
			if !ok {
				return
			}
			remaining++
			wp.queueLength.Add(-1)
			wp.processTask(task, wp.logger)
		default:
			if remaining > 0 {
				wp.logger.WithField("count", remaining).Info("Drained remaining tasks")
			}
			return
		}
	}
}

// Helper functions for key formatting
func formatKeyHashKey(keyID uint) string {
	return "key:" + uintToString(keyID)
}

func formatActiveKeysListKey(groupID uint) string {
	return "group:" + uintToString(groupID) + ":active_keys"
}

func uintToString(n uint) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// isPermanentError checks if an error should not be retried
func isPermanentError(err error) bool {
	if err == nil {
		return false
	}
	errStr := err.Error()
	// Record not found is permanent
	if contains(errStr, "record not found") {
		return true
	}
	// Invalid key ID
	if contains(errStr, "invalid key") {
		return true
	}
	return false
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsAt(s, substr))
}

func containsAt(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
