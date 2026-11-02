package bifrost

import (
	"context"
	"fmt"
	"sync"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/outboundproxy"
)

// RuntimeManager owns lazily-created Bifrost cores keyed by canonical provider
// construction config, with credential partitions only where the SDK caches
// credential-derived objects inside a Core.
type RuntimeManager struct {
	registry *channel.Registry
	options  runtimeOptions
	pool     *runtimeManager

	mu      sync.Mutex
	started bool
	closed  bool
	rootCtx context.Context
	cancel  context.CancelFunc
}

func newRuntimeManager(options runtimeOptions, registry *channel.Registry) (*RuntimeManager, error) {
	options = normalizeRuntimeOptions(options)
	if _, err := newRuntimeShell(options, registry); err != nil {
		return nil, err
	}
	manager := &RuntimeManager{
		registry: registry,
		options:  options,
	}
	manager.pool = newRuntimeManagerPool(func(_ context.Context, config effectiveProviderConfig) (managedProviderRuntime, error) {
		manager.mu.Lock()
		rootCtx := manager.rootCtx
		manager.mu.Unlock()
		if rootCtx == nil {
			return nil, fmt.Errorf("provider runtime manager is not started")
		}
		return newConfiguredRuntime(rootCtx, manager.options, manager.registry, config)
	})
	return manager, nil
}

// Start enables lazy provider initialization. It does not create a Bifrost core.
func (manager *RuntimeManager) Start(ctx context.Context) error {
	if manager == nil {
		return fmt.Errorf("start provider runtime manager: manager is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()
	if manager.closed {
		return fmt.Errorf("start provider runtime manager: manager is shut down")
	}
	if manager.started {
		return nil
	}
	manager.rootCtx, manager.cancel = context.WithCancel(ctx)
	manager.started = true
	return nil
}

// Execute acquires exactly one runtime lease for the frozen attempt target.
func (manager *RuntimeManager) Execute(parent context.Context, spec execution.AttemptSpec) execution.AttemptResult {
	config, failure := manager.configForAttempt(spec)
	if failure != nil {
		return *failure
	}
	lease, err := manager.pool.acquire(parent, config)
	if err != nil {
		return notSentUnaryFailure(execution.ErrorKindInternal, "initialize provider runtime")
	}
	defer lease.Release()
	return lease.runtime.Execute(parent, spec)
}

// ExecuteStream holds the runtime lease through the complete synchronous stream lifecycle.
func (manager *RuntimeManager) ExecuteStream(
	parent context.Context,
	spec execution.AttemptSpec,
	sink execution.StreamSink,
) execution.StreamResult {
	config, failure := manager.configForAttempt(spec)
	if failure != nil {
		return streamFromAttemptFailure(*failure)
	}
	lease, err := manager.pool.acquire(parent, config)
	if err != nil {
		return notSentStreamFailure(execution.ErrorKindInternal, "initialize provider runtime")
	}
	defer lease.Release()
	return lease.runtime.ExecuteStream(parent, spec, sink)
}

func (manager *RuntimeManager) configForAttempt(spec execution.AttemptSpec) (effectiveProviderConfig, *execution.AttemptResult) {
	if manager == nil || manager.registry == nil || manager.pool == nil {
		failure := notSentUnaryFailure(execution.ErrorKindInternal, "provider runtime manager is unavailable")
		return effectiveProviderConfig{}, &failure
	}
	manager.mu.Lock()
	started := manager.started
	closed := manager.closed
	manager.mu.Unlock()
	if !started || closed {
		failure := notSentUnaryFailure(execution.ErrorKindInternal, "provider runtime manager is not running")
		return effectiveProviderConfig{}, &failure
	}
	resolved, err := manager.registry.ResolveExecutionTarget(channel.ID(spec.ChannelID), spec.TargetConfig)
	if err != nil {
		failure := notSentUnaryFailure(execution.ErrorKindInvalidRequest, "invalid channel target")
		return effectiveProviderConfig{}, &failure
	}
	config, err := buildEffectiveProviderConfigForAttempt(resolved, spec, manager.options.allowPrivateNetwork)
	if err != nil {
		failure := notSentUnaryFailure(execution.ErrorKindInvalidRequest, "invalid provider runtime config")
		return effectiveProviderConfig{}, &failure
	}
	if resolved.ProviderKind == channel.ProviderAzureOpenAI {
		credential, credentialErr := manager.registry.ValidateCredential(
			channel.ID(spec.ChannelID),
			spec.Credential.Data(),
		)
		if credentialErr != nil {
			failure := notSentUnaryFailure(execution.ErrorKindInvalidRequest, "invalid credential")
			return effectiveProviderConfig{}, &failure
		}
		apiKey, _ := credential.Value("api_key")
		if apiKey == "" {
			config, err = partitionProviderRuntime(config, spec.Credential)
			if err != nil {
				failure := notSentUnaryFailure(execution.ErrorKindInternal, "partition provider runtime")
				return effectiveProviderConfig{}, &failure
			}
			return config, nil
		}
	}
	if spec.Proxy.Source != outboundproxy.SourceCredential || spec.Proxy.Config.Mode == outboundproxy.ModeDirect {
		return config, nil
	}
	config, err = partitionProviderRuntime(config, spec.Credential)
	if err != nil {
		failure := notSentUnaryFailure(execution.ErrorKindInternal, "partition provider runtime")
		return effectiveProviderConfig{}, &failure
	}
	return config, nil
}

// BeginShutdown rejects new acquisitions and begins bounded-independent Core shutdowns.
func (manager *RuntimeManager) BeginShutdown() <-chan struct{} {
	if manager == nil {
		done := make(chan struct{})
		close(done)
		return done
	}
	manager.mu.Lock()
	manager.closed = true
	if manager.cancel != nil {
		manager.cancel()
	}
	manager.mu.Unlock()
	return manager.pool.beginShutdown()
}

// Shutdown waits until all leased runtimes finish and their Core shutdowns return.
func (manager *RuntimeManager) Shutdown() {
	<-manager.BeginShutdown()
}

// RetireCredential retires provider runtimes that cache objects derived from
// the specified logical credential. In-flight leases finish before shutdown.
func (manager *RuntimeManager) RetireCredential(credentialID uint) {
	if manager == nil || manager.pool == nil {
		return
	}
	manager.pool.retireCredential(credentialID)
}

var _ execution.Executor = (*RuntimeManager)(nil)
