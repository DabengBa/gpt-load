package keypool

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// integrationProcessor simulates a real processor for integration testing
type integrationProcessor struct {
	successCount atomic.Int64
	failureCount atomic.Int64
	keyStates    sync.Map // map[uint]*keyState
}

type keyState struct {
	failureCount int64
	status       string
	mu           sync.Mutex
}

func newIntegrationProcessor() *integrationProcessor {
	return &integrationProcessor{}
}

func (p *integrationProcessor) getOrCreateKeyState(keyID uint) *keyState {
	state, _ := p.keyStates.LoadOrStore(keyID, &keyState{
		failureCount: 0,
		status:       "active",
	})
	return state.(*keyState)
}

func (p *integrationProcessor) ProcessSuccess(keyID uint, keyHashKey, activeKeysListKey string) error {
	p.successCount.Add(1)
	state := p.getOrCreateKeyState(keyID)
	state.mu.Lock()
	defer state.mu.Unlock()
	state.failureCount = 0
	state.status = "active"
	return nil
}

func (p *integrationProcessor) ProcessFailure(task *StatusUpdateTask, keyHashKey, activeKeysListKey string) error {
	p.failureCount.Add(1)
	state := p.getOrCreateKeyState(task.KeyID)
	state.mu.Lock()
	defer state.mu.Unlock()
	state.failureCount++
	if task.BlacklistThresh > 0 && state.failureCount >= int64(task.BlacklistThresh) {
		state.status = "invalid"
	}
	return nil
}

func (p *integrationProcessor) getKeyState(keyID uint) (int64, string) {
	state := p.getOrCreateKeyState(keyID)
	state.mu.Lock()
	defer state.mu.Unlock()
	return state.failureCount, state.status
}

// TestIntegration_CompleteStatusUpdateFlow tests the full status update flow
func TestIntegration_CompleteStatusUpdateFlow(t *testing.T) {
	processor := newIntegrationProcessor()
	config := WorkerPoolConfig{
		WorkerCount:   4,
		QueueCapacity: 1000,
		MaxRetries:    3,
	}

	wp := NewWorkerPool(config, processor, nil)
	wp.Start()

	// Test 1: Submit success updates
	for i := 0; i < 10; i++ {
		wp.Submit(&StatusUpdateTask{
			KeyID:     uint(i),
			GroupID:   1,
			IsSuccess: true,
			Timestamp: time.Now(),
		})
	}

	// Test 2: Submit failure updates
	for i := 10; i < 20; i++ {
		wp.Submit(&StatusUpdateTask{
			KeyID:           uint(i),
			GroupID:         1,
			IsSuccess:       false,
			ErrorMessage:    "test error",
			Timestamp:       time.Now(),
			BlacklistThresh: 5,
		})
	}

	// Wait for processing
	time.Sleep(200 * time.Millisecond)
	wp.Stop()

	// Verify counts
	assert.Equal(t, int64(10), processor.successCount.Load())
	assert.Equal(t, int64(10), processor.failureCount.Load())

	// Verify metrics
	metrics := wp.GetMetrics()
	assert.Equal(t, int64(20), metrics.ProcessedCount)
	assert.Equal(t, int64(0), metrics.ErrorCount)
}

// TestIntegration_HighConcurrency tests high concurrency scenario
func TestIntegration_HighConcurrency(t *testing.T) {
	processor := newIntegrationProcessor()
	config := WorkerPoolConfig{
		WorkerCount:   8,
		QueueCapacity: 10000,
	}

	wp := NewWorkerPool(config, processor, nil)
	wp.Start()

	// Submit tasks from multiple goroutines
	var wg sync.WaitGroup
	taskCount := 1000
	goroutines := 10

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(gid int) {
			defer wg.Done()
			for i := 0; i < taskCount/goroutines; i++ {
				keyID := uint(gid*1000 + i)
				wp.Submit(&StatusUpdateTask{
					KeyID:     keyID,
					GroupID:   1,
					IsSuccess: i%2 == 0, // Alternate success/failure
					Timestamp: time.Now(),
				})
			}
		}(g)
	}

	wg.Wait()
	time.Sleep(500 * time.Millisecond)
	wp.Stop()

	// Verify all tasks processed
	metrics := wp.GetMetrics()
	assert.Equal(t, int64(taskCount), metrics.ProcessedCount)
}

// TestIntegration_BlacklistThreshold tests key blacklisting
func TestIntegration_BlacklistThreshold(t *testing.T) {
	processor := newIntegrationProcessor()
	config := WorkerPoolConfig{
		WorkerCount:   1, // Single worker for predictable order
		QueueCapacity: 100,
	}

	wp := NewWorkerPool(config, processor, nil)
	wp.Start()

	keyID := uint(1)
	threshold := 3

	// Submit failures until threshold
	for i := 0; i < threshold; i++ {
		wp.Submit(&StatusUpdateTask{
			KeyID:           keyID,
			GroupID:         1,
			IsSuccess:       false,
			ErrorMessage:    "test error",
			Timestamp:       time.Now(),
			BlacklistThresh: threshold,
		})
	}

	time.Sleep(200 * time.Millisecond)
	wp.Stop()

	// Verify key is blacklisted
	failureCount, status := processor.getKeyState(keyID)
	assert.Equal(t, int64(threshold), failureCount)
	assert.Equal(t, "invalid", status)
}

// TestIntegration_SuccessResetsFailures tests that success resets failure count
func TestIntegration_SuccessResetsFailures(t *testing.T) {
	processor := newIntegrationProcessor()
	config := WorkerPoolConfig{
		WorkerCount:   1,
		QueueCapacity: 100,
	}

	wp := NewWorkerPool(config, processor, nil)
	wp.Start()

	keyID := uint(1)

	// Submit some failures
	for i := 0; i < 3; i++ {
		wp.Submit(&StatusUpdateTask{
			KeyID:           keyID,
			GroupID:         1,
			IsSuccess:       false,
			Timestamp:       time.Now(),
			BlacklistThresh: 10, // High threshold so it doesn't blacklist
		})
	}

	time.Sleep(100 * time.Millisecond)

	// Verify failures accumulated
	failureCount, _ := processor.getKeyState(keyID)
	assert.Equal(t, int64(3), failureCount)

	// Submit success
	wp.Submit(&StatusUpdateTask{
		KeyID:     keyID,
		GroupID:   1,
		IsSuccess: true,
		Timestamp: time.Now(),
	})

	time.Sleep(100 * time.Millisecond)
	wp.Stop()

	// Verify failure count reset
	failureCount, status := processor.getKeyState(keyID)
	assert.Equal(t, int64(0), failureCount)
	assert.Equal(t, "active", status)
}

// TestIntegration_QueueCapacityHandling tests queue full behavior
// When queue is full, tasks are processed synchronously instead of being dropped
func TestIntegration_QueueCapacityHandling(t *testing.T) {
	// Use a slow processor to fill the queue
	slowProcessor := &slowProcessor{delay: 50 * time.Millisecond}
	config := WorkerPoolConfig{
		WorkerCount:   1,
		QueueCapacity: 5,
	}

	wp := NewWorkerPool(config, slowProcessor, nil)
	wp.Start()

	// Try to submit more tasks than queue capacity
	taskCount := 20
	submitted := 0
	for i := 0; i < taskCount; i++ {
		if wp.Submit(&StatusUpdateTask{
			KeyID:     uint(i),
			GroupID:   1,
			IsSuccess: true,
		}) {
			submitted++
		}
	}

	wp.Stop()

	// All tasks should be submitted (either queued or processed synchronously)
	assert.Equal(t, taskCount, submitted, "All tasks should be submitted successfully")

	// All tasks should be processed, none dropped
	metrics := wp.GetMetrics()
	assert.Equal(t, int64(0), metrics.DroppedCount, "No tasks should be dropped")
	assert.Equal(t, int64(taskCount), metrics.ProcessedCount, "All tasks should be processed")
}

type slowProcessor struct {
	delay time.Duration
}

func (p *slowProcessor) ProcessSuccess(keyID uint, keyHashKey, activeKeysListKey string) error {
	time.Sleep(p.delay)
	return nil
}

func (p *slowProcessor) ProcessFailure(task *StatusUpdateTask, keyHashKey, activeKeysListKey string) error {
	time.Sleep(p.delay)
	return nil
}
