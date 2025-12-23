package keypool

import (
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

// mockRetryProcessor tracks retry attempts and can simulate different error types
type mockRetryProcessor struct {
	callCount      atomic.Int32
	failUntil      int32 // fail this many times before succeeding
	permanentError bool  // if true, return permanent error
	errorMessage   string
}

func (m *mockRetryProcessor) ProcessSuccess(keyID uint, keyHashKey, activeKeysListKey string) error {
	count := m.callCount.Add(1)
	if count <= m.failUntil {
		if m.permanentError {
			return errors.New(m.errorMessage)
		}
		return errors.New("transient error: connection timeout")
	}
	return nil
}

func (m *mockRetryProcessor) ProcessFailure(task *StatusUpdateTask, keyHashKey, activeKeysListKey string) error {
	count := m.callCount.Add(1)
	if count <= m.failUntil {
		if m.permanentError {
			return errors.New(m.errorMessage)
		}
		return errors.New("transient error: connection timeout")
	}
	return nil
}

func TestRetry_TransientErrorRetries(t *testing.T) {
	// Processor fails twice then succeeds
	processor := &mockRetryProcessor{
		failUntil:      2,
		permanentError: false,
	}

	config := WorkerPoolConfig{
		WorkerCount:    1,
		QueueCapacity:  10,
		MaxRetries:     3,
		RetryBaseDelay: 10 * time.Millisecond, // Short delay for testing
	}

	wp := NewWorkerPool(config, processor, nil)
	wp.Start()
	defer wp.Stop()

	task := &StatusUpdateTask{
		KeyID:     1,
		GroupID:   1,
		IsSuccess: true,
		Timestamp: time.Now(),
	}

	wp.Submit(task)

	// Wait for processing
	time.Sleep(200 * time.Millisecond)

	// Should have been called 3 times (2 failures + 1 success)
	if processor.callCount.Load() != 3 {
		t.Errorf("Expected 3 calls, got %d", processor.callCount.Load())
	}

	metrics := wp.GetMetrics()
	if metrics.ProcessedCount != 1 {
		t.Errorf("Expected 1 processed, got %d", metrics.ProcessedCount)
	}
	if metrics.ErrorCount != 0 {
		t.Errorf("Expected 0 errors, got %d", metrics.ErrorCount)
	}
}

func TestRetry_PermanentErrorNoRetry(t *testing.T) {
	// Processor returns permanent error
	processor := &mockRetryProcessor{
		failUntil:      100, // Always fail
		permanentError: true,
		errorMessage:   "record not found",
	}

	config := WorkerPoolConfig{
		WorkerCount:    1,
		QueueCapacity:  10,
		MaxRetries:     3,
		RetryBaseDelay: 10 * time.Millisecond,
	}

	wp := NewWorkerPool(config, processor, nil)
	wp.Start()
	defer wp.Stop()

	task := &StatusUpdateTask{
		KeyID:     1,
		GroupID:   1,
		IsSuccess: true,
		Timestamp: time.Now(),
	}

	wp.Submit(task)

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	// Should have been called only once (no retry for permanent error)
	if processor.callCount.Load() != 1 {
		t.Errorf("Expected 1 call (no retry for permanent error), got %d", processor.callCount.Load())
	}

	metrics := wp.GetMetrics()
	if metrics.ProcessedCount != 1 {
		t.Errorf("Expected 1 processed, got %d", metrics.ProcessedCount)
	}
	if metrics.ErrorCount != 1 {
		t.Errorf("Expected 1 error, got %d", metrics.ErrorCount)
	}
}

func TestRetry_InvalidKeyPermanentError(t *testing.T) {
	// Processor returns "invalid key" error
	processor := &mockRetryProcessor{
		failUntil:      100,
		permanentError: true,
		errorMessage:   "invalid key ID",
	}

	config := WorkerPoolConfig{
		WorkerCount:    1,
		QueueCapacity:  10,
		MaxRetries:     3,
		RetryBaseDelay: 10 * time.Millisecond,
	}

	wp := NewWorkerPool(config, processor, nil)
	wp.Start()
	defer wp.Stop()

	task := &StatusUpdateTask{
		KeyID:     999,
		GroupID:   1,
		IsSuccess: false,
		Timestamp: time.Now(),
	}

	wp.Submit(task)

	time.Sleep(100 * time.Millisecond)

	// Should not retry for invalid key
	if processor.callCount.Load() != 1 {
		t.Errorf("Expected 1 call (no retry for invalid key), got %d", processor.callCount.Load())
	}
}

func TestRetry_AllRetriesExhausted(t *testing.T) {
	// Processor always fails with transient error
	processor := &mockRetryProcessor{
		failUntil:      100, // Always fail
		permanentError: false,
	}

	config := WorkerPoolConfig{
		WorkerCount:    1,
		QueueCapacity:  10,
		MaxRetries:     2,
		RetryBaseDelay: 10 * time.Millisecond,
	}

	wp := NewWorkerPool(config, processor, nil)
	wp.Start()
	defer wp.Stop()

	task := &StatusUpdateTask{
		KeyID:     1,
		GroupID:   1,
		IsSuccess: true,
		Timestamp: time.Now(),
	}

	wp.Submit(task)

	// Wait for all retries
	time.Sleep(300 * time.Millisecond)

	// Should have been called MaxRetries + 1 times (initial + retries)
	expectedCalls := int32(config.MaxRetries + 1)
	if processor.callCount.Load() != expectedCalls {
		t.Errorf("Expected %d calls, got %d", expectedCalls, processor.callCount.Load())
	}

	metrics := wp.GetMetrics()
	if metrics.ErrorCount != 1 {
		t.Errorf("Expected 1 error after exhausting retries, got %d", metrics.ErrorCount)
	}
}

func TestRetry_ExponentialBackoff(t *testing.T) {
	// Track timing between calls
	var callTimes []time.Time
	processor := &mockRetryProcessor{
		failUntil:      3,
		permanentError: false,
	}

	// Wrap to track timing
	originalProcess := processor.ProcessSuccess
	callTimesPtr := &callTimes
	wrappedProcessor := &timingProcessor{
		inner:     processor,
		callTimes: callTimesPtr,
	}

	config := WorkerPoolConfig{
		WorkerCount:    1,
		QueueCapacity:  10,
		MaxRetries:     3,
		RetryBaseDelay: 50 * time.Millisecond,
	}

	wp := NewWorkerPool(config, wrappedProcessor, nil)
	wp.Start()
	defer wp.Stop()

	task := &StatusUpdateTask{
		KeyID:     1,
		GroupID:   1,
		IsSuccess: true,
		Timestamp: time.Now(),
	}

	wp.Submit(task)

	// Wait for processing
	time.Sleep(500 * time.Millisecond)

	// Verify exponential backoff timing
	times := *callTimesPtr
	if len(times) < 3 {
		t.Fatalf("Expected at least 3 calls, got %d", len(times))
	}

	// First retry should be ~50ms after first call
	delay1 := times[1].Sub(times[0])
	if delay1 < 40*time.Millisecond || delay1 > 100*time.Millisecond {
		t.Errorf("First retry delay expected ~50ms, got %v", delay1)
	}

	// Second retry should be ~100ms after second call (2x base)
	delay2 := times[2].Sub(times[1])
	if delay2 < 80*time.Millisecond || delay2 > 200*time.Millisecond {
		t.Errorf("Second retry delay expected ~100ms, got %v", delay2)
	}

	_ = originalProcess // Silence unused warning
}

// timingProcessor wraps a processor to track call times
type timingProcessor struct {
	inner     TaskProcessor
	callTimes *[]time.Time
}

func (t *timingProcessor) ProcessSuccess(keyID uint, keyHashKey, activeKeysListKey string) error {
	*t.callTimes = append(*t.callTimes, time.Now())
	return t.inner.ProcessSuccess(keyID, keyHashKey, activeKeysListKey)
}

func (t *timingProcessor) ProcessFailure(task *StatusUpdateTask, keyHashKey, activeKeysListKey string) error {
	*t.callTimes = append(*t.callTimes, time.Now())
	return t.inner.ProcessFailure(task, keyHashKey, activeKeysListKey)
}

// TestIsPermanentError is in worker_pool_test.go - additional edge cases here
func TestIsPermanentError_EdgeCases(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{"record not found in message", errors.New("failed: record not found in database"), true},
		{"invalid key without ID", errors.New("invalid key"), true},
		{"generic error", errors.New("something went wrong"), false},
		{"empty error", errors.New(""), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isPermanentError(tt.err)
			if result != tt.expected {
				t.Errorf("isPermanentError(%v) = %v, expected %v", tt.err, result, tt.expected)
			}
		})
	}
}

func TestRetry_FailureTaskRetries(t *testing.T) {
	// Test retry for failure tasks (not just success)
	processor := &mockRetryProcessor{
		failUntil:      1,
		permanentError: false,
	}

	config := WorkerPoolConfig{
		WorkerCount:    1,
		QueueCapacity:  10,
		MaxRetries:     3,
		RetryBaseDelay: 10 * time.Millisecond,
	}

	wp := NewWorkerPool(config, processor, nil)
	wp.Start()
	defer wp.Stop()

	task := &StatusUpdateTask{
		KeyID:           1,
		GroupID:         1,
		IsSuccess:       false, // Failure task
		ErrorMessage:    "API error",
		Timestamp:       time.Now(),
		BlacklistThresh: 3,
	}

	wp.Submit(task)

	time.Sleep(150 * time.Millisecond)

	// Should have been called 2 times (1 failure + 1 success)
	if processor.callCount.Load() != 2 {
		t.Errorf("Expected 2 calls, got %d", processor.callCount.Load())
	}

	metrics := wp.GetMetrics()
	if metrics.ErrorCount != 0 {
		t.Errorf("Expected 0 errors (retry succeeded), got %d", metrics.ErrorCount)
	}
}
