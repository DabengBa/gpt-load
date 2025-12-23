package keypool

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockProcessor implements TaskProcessor for testing
type mockProcessor struct {
	successCalls atomic.Int64
	failureCalls atomic.Int64
	processedIDs []uint
	mu           sync.Mutex
	shouldFail   bool
	failCount    int
	currentFails int
}

func newMockProcessor() *mockProcessor {
	return &mockProcessor{
		processedIDs: make([]uint, 0),
	}
}

func (m *mockProcessor) ProcessSuccess(keyID uint, keyHashKey, activeKeysListKey string) error {
	m.successCalls.Add(1)
	m.mu.Lock()
	m.processedIDs = append(m.processedIDs, keyID)
	m.mu.Unlock()

	if m.shouldFail && m.currentFails < m.failCount {
		m.currentFails++
		return errors.New("transient error")
	}
	return nil
}

func (m *mockProcessor) ProcessFailure(task *StatusUpdateTask, keyHashKey, activeKeysListKey string) error {
	m.failureCalls.Add(1)
	m.mu.Lock()
	m.processedIDs = append(m.processedIDs, task.KeyID)
	m.mu.Unlock()

	if m.shouldFail && m.currentFails < m.failCount {
		m.currentFails++
		return errors.New("transient error")
	}
	return nil
}

func (m *mockProcessor) getProcessedIDs() []uint {
	m.mu.Lock()
	defer m.mu.Unlock()
	result := make([]uint, len(m.processedIDs))
	copy(result, m.processedIDs)
	return result
}

func TestNewWorkerPool_DefaultConfig(t *testing.T) {
	processor := newMockProcessor()
	config := DefaultWorkerPoolConfig()

	wp := NewWorkerPool(config, processor, nil)

	assert.NotNil(t, wp)
	assert.Equal(t, 4, wp.config.WorkerCount)
	assert.Equal(t, 10000, wp.config.QueueCapacity)
	assert.Equal(t, 3, wp.config.MaxRetries)
	assert.Equal(t, 100*time.Millisecond, wp.config.RetryBaseDelay)
}

func TestNewWorkerPool_CustomConfig(t *testing.T) {
	processor := newMockProcessor()
	config := WorkerPoolConfig{
		WorkerCount:    8,
		QueueCapacity:  5000,
		MaxRetries:     5,
		RetryBaseDelay: 200 * time.Millisecond,
	}

	wp := NewWorkerPool(config, processor, nil)

	assert.Equal(t, 8, wp.config.WorkerCount)
	assert.Equal(t, 5000, wp.config.QueueCapacity)
	assert.Equal(t, 5, wp.config.MaxRetries)
	assert.Equal(t, 200*time.Millisecond, wp.config.RetryBaseDelay)
}

func TestNewWorkerPool_InvalidConfig_UsesDefaults(t *testing.T) {
	processor := newMockProcessor()
	config := WorkerPoolConfig{
		WorkerCount:    -1,
		QueueCapacity:  0,
		MaxRetries:     -1,
		RetryBaseDelay: 0,
	}

	wp := NewWorkerPool(config, processor, nil)

	// Should use defaults for invalid values
	assert.Equal(t, 4, wp.config.WorkerCount)
	assert.Equal(t, 10000, wp.config.QueueCapacity)
	assert.Equal(t, 3, wp.config.MaxRetries)
	assert.Equal(t, 100*time.Millisecond, wp.config.RetryBaseDelay)
}

func TestWorkerPool_StartAndStop(t *testing.T) {
	processor := newMockProcessor()
	config := WorkerPoolConfig{
		WorkerCount:   2,
		QueueCapacity: 100,
	}

	wp := NewWorkerPool(config, processor, nil)

	assert.False(t, wp.IsRunning())

	wp.Start()
	assert.True(t, wp.IsRunning())

	// Give workers time to start
	time.Sleep(10 * time.Millisecond)

	wp.Stop()
	assert.False(t, wp.IsRunning())
}

func TestWorkerPool_SubmitTask(t *testing.T) {
	processor := newMockProcessor()
	config := WorkerPoolConfig{
		WorkerCount:   2,
		QueueCapacity: 100,
	}
	logger := logrus.NewEntry(logrus.StandardLogger())

	wp := NewWorkerPool(config, processor, logger)
	wp.Start()
	defer wp.Stop()

	task := &StatusUpdateTask{
		KeyID:     1,
		GroupID:   1,
		IsSuccess: true,
		Timestamp: time.Now(),
	}

	ok := wp.Submit(task)
	assert.True(t, ok)

	// Wait for processing
	time.Sleep(50 * time.Millisecond)

	assert.Equal(t, int64(1), processor.successCalls.Load())
}

func TestWorkerPool_SubmitWhenNotRunning(t *testing.T) {
	processor := newMockProcessor()
	config := DefaultWorkerPoolConfig()

	wp := NewWorkerPool(config, processor, nil)
	// Don't start the pool

	task := &StatusUpdateTask{
		KeyID:     1,
		GroupID:   1,
		IsSuccess: true,
	}

	ok := wp.Submit(task)
	assert.False(t, ok)
}

func TestWorkerPool_QueueFull_ProcessesSynchronously(t *testing.T) {
	processor := newMockProcessor()
	config := WorkerPoolConfig{
		WorkerCount:   1,
		QueueCapacity: 2,
	}

	wp := NewWorkerPool(config, processor, nil)
	wp.Start()
	defer wp.Stop()

	// Submit more tasks than queue capacity
	// When queue is full, tasks should be processed synchronously instead of dropped
	taskCount := 5
	for i := 0; i < taskCount; i++ {
		ok := wp.Submit(&StatusUpdateTask{
			KeyID:     uint(i),
			GroupID:   1,
			IsSuccess: true,
		})
		assert.True(t, ok, "Submit should always succeed (either queued or processed synchronously)")
	}

	// Wait for any remaining queued tasks to be processed
	time.Sleep(100 * time.Millisecond)

	// Verify all tasks were processed and none were dropped
	metrics := wp.GetMetrics()
	assert.Equal(t, int64(0), metrics.DroppedCount, "No tasks should be dropped")
	assert.Equal(t, int64(taskCount), metrics.ProcessedCount, "All tasks should be processed")
}

func TestWorkerPool_GetMetrics(t *testing.T) {
	processor := newMockProcessor()
	config := WorkerPoolConfig{
		WorkerCount:   2,
		QueueCapacity: 100,
	}

	wp := NewWorkerPool(config, processor, nil)
	wp.Start()

	// Submit some tasks
	for i := 0; i < 5; i++ {
		wp.Submit(&StatusUpdateTask{
			KeyID:     uint(i),
			GroupID:   1,
			IsSuccess: true,
		})
	}

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	wp.Stop()

	metrics := wp.GetMetrics()
	assert.Equal(t, int64(5), metrics.ProcessedCount)
	assert.Equal(t, int64(0), metrics.ErrorCount)
}

func TestWorkerPool_GracefulShutdown_ProcessesPendingTasks(t *testing.T) {
	processor := newMockProcessor()
	config := WorkerPoolConfig{
		WorkerCount:   1,
		QueueCapacity: 100,
	}

	wp := NewWorkerPool(config, processor, nil)
	wp.Start()

	// Submit tasks
	taskCount := 10
	for i := 0; i < taskCount; i++ {
		wp.Submit(&StatusUpdateTask{
			KeyID:     uint(i),
			GroupID:   1,
			IsSuccess: true,
		})
	}

	// Stop immediately - should still process all tasks
	wp.Stop()

	metrics := wp.GetMetrics()
	assert.Equal(t, int64(taskCount), metrics.ProcessedCount)
}

func TestFormatKeyHashKey(t *testing.T) {
	assert.Equal(t, "key:0", formatKeyHashKey(0))
	assert.Equal(t, "key:1", formatKeyHashKey(1))
	assert.Equal(t, "key:123", formatKeyHashKey(123))
	assert.Equal(t, "key:4294967295", formatKeyHashKey(4294967295))
}

func TestFormatActiveKeysListKey(t *testing.T) {
	assert.Equal(t, "group:0:active_keys", formatActiveKeysListKey(0))
	assert.Equal(t, "group:1:active_keys", formatActiveKeysListKey(1))
	assert.Equal(t, "group:123:active_keys", formatActiveKeysListKey(123))
}

func TestIsPermanentError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{"nil error", nil, false},
		{"record not found", errors.New("record not found"), true},
		{"invalid key", errors.New("invalid key ID"), true},
		{"transient error", errors.New("connection timeout"), false},
		{"database locked", errors.New("database is locked"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isPermanentError(tt.err)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestWorkerPool_ProcessesSuccessAndFailureTasks(t *testing.T) {
	processor := newMockProcessor()
	config := WorkerPoolConfig{
		WorkerCount:   2,
		QueueCapacity: 100,
	}

	wp := NewWorkerPool(config, processor, nil)
	wp.Start()

	// Submit success task
	wp.Submit(&StatusUpdateTask{
		KeyID:     1,
		GroupID:   1,
		IsSuccess: true,
	})

	// Submit failure task
	wp.Submit(&StatusUpdateTask{
		KeyID:        2,
		GroupID:      1,
		IsSuccess:    false,
		ErrorMessage: "test error",
	})

	time.Sleep(100 * time.Millisecond)
	wp.Stop()

	assert.Equal(t, int64(1), processor.successCalls.Load())
	assert.Equal(t, int64(1), processor.failureCalls.Load())
}

func TestWorkerPool_DoubleStart_NoOp(t *testing.T) {
	processor := newMockProcessor()
	config := DefaultWorkerPoolConfig()

	wp := NewWorkerPool(config, processor, nil)

	wp.Start()
	wp.Start() // Should be no-op

	assert.True(t, wp.IsRunning())

	wp.Stop()
}

func TestWorkerPool_DoubleStop_NoOp(t *testing.T) {
	processor := newMockProcessor()
	config := DefaultWorkerPoolConfig()

	wp := NewWorkerPool(config, processor, nil)
	wp.Start()

	wp.Stop()
	wp.Stop() // Should be no-op

	assert.False(t, wp.IsRunning())
}

func TestWorkerPool_ConcurrentSubmit(t *testing.T) {
	processor := newMockProcessor()
	config := WorkerPoolConfig{
		WorkerCount:   4,
		QueueCapacity: 1000,
	}

	wp := NewWorkerPool(config, processor, nil)
	wp.Start()

	var wg sync.WaitGroup
	taskCount := 100

	for i := 0; i < taskCount; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			wp.Submit(&StatusUpdateTask{
				KeyID:     uint(id),
				GroupID:   1,
				IsSuccess: true,
			})
		}(i)
	}

	wg.Wait()
	time.Sleep(200 * time.Millisecond)
	wp.Stop()

	metrics := wp.GetMetrics()
	require.Equal(t, int64(taskCount), metrics.ProcessedCount)
}
