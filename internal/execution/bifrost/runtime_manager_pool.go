package bifrost

import (
	"bytes"
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"gpt-load/internal/execution"
)

type managedProviderRuntime interface {
	execution.Executor
	BeginShutdown() <-chan struct{}
}

type managedRuntimeFactory func(context.Context, effectiveProviderConfig) (managedProviderRuntime, error)

type runtimeLease struct {
	runtime managedProviderRuntime
	release func()
	once    sync.Once
}

func (lease *runtimeLease) Release() {
	if lease == nil {
		return
	}
	lease.once.Do(func() {
		if lease.release != nil {
			lease.release()
		}
	})
}

type runtimeManager struct {
	factory managedRuntimeFactory

	mu             sync.Mutex
	entries        map[string]*runtimeEntry
	active         map[string][]byte
	activeKnown    bool
	closed         bool
	shuttingDown   int
	shutdownDone   chan struct{}
	shutdownClosed atomic.Bool
}

type runtimeEntry struct {
	config       effectiveProviderConfig
	runtime      managedProviderRuntime
	ready        chan struct{}
	initializing bool
	initErr      error
	refs         int
	retiring     bool
}

func newRuntimeManagerPool(factory managedRuntimeFactory) *runtimeManager {
	return &runtimeManager{
		factory:      factory,
		entries:      make(map[string]*runtimeEntry),
		active:       make(map[string][]byte),
		shutdownDone: make(chan struct{}),
	}
}

func (manager *runtimeManager) acquire(
	ctx context.Context,
	config effectiveProviderConfig,
) (*runtimeLease, error) {
	if manager == nil || manager.factory == nil {
		return nil, fmt.Errorf("provider runtime manager is unavailable")
	}
	for {
		manager.mu.Lock()
		if manager.closed {
			manager.mu.Unlock()
			return nil, fmt.Errorf("provider runtime manager is shut down")
		}
		if entry, exists := manager.entries[config.fingerprint]; exists {
			if !bytes.Equal(entry.config.canonical, config.canonical) {
				manager.mu.Unlock()
				return nil, fmt.Errorf("provider runtime fingerprint collision")
			}
			if entry.initializing {
				ready := entry.ready
				manager.mu.Unlock()
				select {
				case <-ready:
					manager.mu.Lock()
					initErr := entry.initErr
					manager.mu.Unlock()
					if initErr != nil {
						return nil, initErr
					}
					continue
				case <-ctx.Done():
					return nil, ctx.Err()
				}
			}
			entry.refs++
			manager.mu.Unlock()
			return manager.lease(entry), nil
		}
		entry := &runtimeEntry{
			config:       cloneEffectiveProviderConfig(config),
			ready:        make(chan struct{}),
			initializing: true,
			refs:         1,
			retiring:     manager.activeKnown && !manager.configIsActive(config),
		}
		manager.entries[config.fingerprint] = entry
		manager.mu.Unlock()

		runtime, err := manager.factory(ctx, cloneEffectiveProviderConfig(config))
		manager.mu.Lock()
		entry.initializing = false
		if err != nil {
			entry.initErr = err
			delete(manager.entries, config.fingerprint)
			close(entry.ready)
			manager.maybeCloseShutdownLocked()
			manager.mu.Unlock()
			return nil, err
		}
		entry.runtime = runtime
		close(entry.ready)
		manager.mu.Unlock()
		return manager.lease(entry), nil
	}
}

func (manager *runtimeManager) lease(entry *runtimeEntry) *runtimeLease {
	return &runtimeLease{
		runtime: entry.runtime,
		release: func() { manager.release(entry) },
	}
}

func (manager *runtimeManager) release(entry *runtimeEntry) {
	manager.mu.Lock()
	if entry.refs > 0 {
		entry.refs--
	}
	retired := manager.retireEntryLocked(entry)
	manager.mu.Unlock()
	manager.startShutdown(retired)
}

func (manager *runtimeManager) reconcile(configs []effectiveProviderConfig) {
	if manager == nil {
		return
	}
	active := make(map[string][]byte, len(configs))
	for _, config := range configs {
		active[config.fingerprint] = append([]byte(nil), config.canonical...)
	}
	manager.mu.Lock()
	if manager.closed {
		manager.mu.Unlock()
		return
	}
	manager.activeKnown = true
	manager.active = active
	retired := make([]managedProviderRuntime, 0)
	for _, entry := range manager.entries {
		entry.retiring = !manager.configIsActive(entry.config)
		if runtime := manager.retireEntryLocked(entry); runtime != nil {
			retired = append(retired, runtime)
		}
	}
	manager.mu.Unlock()
	for _, runtime := range retired {
		manager.startShutdown(runtime)
	}
}

func (manager *runtimeManager) beginShutdown() <-chan struct{} {
	if manager == nil {
		done := make(chan struct{})
		close(done)
		return done
	}
	manager.mu.Lock()
	manager.closed = true
	manager.activeKnown = true
	manager.active = map[string][]byte{}
	retired := make([]managedProviderRuntime, 0, len(manager.entries))
	for _, entry := range manager.entries {
		entry.retiring = true
		if runtime := manager.retireEntryLocked(entry); runtime != nil {
			retired = append(retired, runtime)
		}
	}
	manager.maybeCloseShutdownLocked()
	done := manager.shutdownDone
	manager.mu.Unlock()
	for _, runtime := range retired {
		manager.startShutdown(runtime)
	}
	return done
}

func (manager *runtimeManager) configIsActive(config effectiveProviderConfig) bool {
	fingerprint := config.fingerprint
	canonical := config.canonical
	if config.credentialPartitionID != 0 {
		switch {
		case config.targetFingerprint != "":
			fingerprint = config.targetFingerprint
			canonical = config.targetCanonical
		case config.baseFingerprint != "":
			fingerprint = config.baseFingerprint
			canonical = config.baseCanonical
		}
	}
	activeCanonical, exists := manager.active[fingerprint]
	return exists && bytes.Equal(activeCanonical, canonical)
}

func (manager *runtimeManager) retireCredential(credentialID uint) {
	if manager == nil || credentialID == 0 {
		return
	}
	manager.mu.Lock()
	retired := make([]managedProviderRuntime, 0)
	for _, entry := range manager.entries {
		if entry.config.credentialPartitionID != credentialID {
			continue
		}
		entry.retiring = true
		if runtime := manager.retireEntryLocked(entry); runtime != nil {
			retired = append(retired, runtime)
		}
	}
	manager.mu.Unlock()
	for _, runtime := range retired {
		manager.startShutdown(runtime)
	}
}

func (manager *runtimeManager) retireEntryLocked(entry *runtimeEntry) managedProviderRuntime {
	if entry == nil || entry.initializing || !entry.retiring || entry.refs != 0 || entry.runtime == nil {
		return nil
	}
	current, exists := manager.entries[entry.config.fingerprint]
	if !exists || current != entry {
		return nil
	}
	delete(manager.entries, entry.config.fingerprint)
	manager.shuttingDown++
	return entry.runtime
}

func (manager *runtimeManager) startShutdown(runtime managedProviderRuntime) {
	if runtime == nil {
		return
	}
	done := runtime.BeginShutdown()
	go func() {
		<-done
		manager.mu.Lock()
		manager.shuttingDown--
		manager.maybeCloseShutdownLocked()
		manager.mu.Unlock()
	}()
}

func (manager *runtimeManager) maybeCloseShutdownLocked() {
	if !manager.closed || len(manager.entries) != 0 || manager.shuttingDown != 0 || manager.shutdownClosed.Load() {
		return
	}
	manager.shutdownClosed.Store(true)
	close(manager.shutdownDone)
}
