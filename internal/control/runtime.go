package control

import (
	"context"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"gpt-load/internal/affinity"
	"gpt-load/internal/state"
)

const (
	retentionInterval        = time.Hour
	blacklistReleaseInterval = time.Minute
)

type credentialMutationCoordinator interface {
	Do(uint, func())
}

type operationRecoveryRuntime interface {
	RunOperationRecovery(context.Context)
}

type catalogSyncRuntime interface {
	Run(context.Context)
}

// RequestLogCleaner is the control-owned scheduling view of request log
// retention. The requestlog package owns all cleanup semantics.
type RequestLogCleaner interface {
	Sweep(context.Context, time.Time)
}

type credentialStageCleaner interface {
	CleanupCredentialStages(context.Context, time.Time) error
}

type runtimeTicker interface {
	C() <-chan time.Time
	Stop()
}

type standardRuntimeTicker struct {
	ticker *time.Ticker
}

func (ticker standardRuntimeTicker) C() <-chan time.Time {
	return ticker.ticker.C
}

func (ticker standardRuntimeTicker) Stop() {
	ticker.ticker.Stop()
}

type Runtime struct {
	requestLogCleaner RequestLogCleaner
	stageCleaner      credentialStageCleaner
	operationRecovery operationRecoveryRuntime
	catalogSync       catalogSyncRuntime
	oauthCallback     *OAuthCallbackManager
	registry          *state.CredentialRegistry
	affinityCleaner   affinity.DurableBindingCleaner
	affinityPolicy    func() affinity.DurablePolicy
	healthStats       interface{ ClearProblemState(uint) }
	now               func() time.Time
	newTicker         func(time.Duration) runtimeTicker
}

func NewRuntime(
	registry *state.CredentialRegistry,
	requestLogCleaner RequestLogCleaner,
	operationRecovery *Service,
	catalogSync *CatalogSyncCoordinator,
	affinityCleaner affinity.DurableBindingCleaner,
	affinityPolicy func() affinity.DurablePolicy,
) *Runtime {
	runtime := &Runtime{
		requestLogCleaner: requestLogCleaner,
		stageCleaner:      operationRecovery,
		operationRecovery: operationRecovery,
		catalogSync:       catalogSync,
		registry:          registry,
		affinityCleaner:   affinityCleaner,
		affinityPolicy:    affinityPolicy,
		now:               time.Now,
		newTicker: func(interval time.Duration) runtimeTicker {
			return standardRuntimeTicker{ticker: time.NewTicker(interval)}
		},
	}
	if operationRecovery != nil {
		runtime.oauthCallback = operationRecovery.oauthCallback
	}
	return runtime
}

// SetHealthStats connects release maintenance to the shared health window.
func (runtime *Runtime) SetHealthStats(stats interface{ ClearProblemState(uint) }) {
	if runtime != nil {
		runtime.healthStats = stats
	}
}

func (runtime *Runtime) Run(ctx context.Context) {
	var wait sync.WaitGroup
	if runtime.registry != nil {
		releaseTicker := runtime.newTicker(blacklistReleaseInterval)
		wait.Go(func() {
			runtime.runBlacklistRelease(ctx, releaseTicker)
		})
	}
	if runtime.requestLogCleaner != nil || runtime.stageCleaner != nil || runtime.affinityCleaner != nil {
		retentionTicker := runtime.newTicker(retentionInterval)
		wait.Go(func() {
			runtime.runRetention(ctx, retentionTicker)
		})
	}
	if runtime.operationRecovery != nil {
		wait.Go(func() {
			runtime.operationRecovery.RunOperationRecovery(ctx)
		})
	}
	if runtime.catalogSync != nil {
		wait.Go(func() {
			runtime.catalogSync.Run(ctx)
		})
	}
	if runtime.oauthCallback != nil {
		wait.Go(func() {
			runtime.oauthCallback.Run(ctx)
		})
	}
	wait.Wait()
}

func (runtime *Runtime) runRetention(ctx context.Context, ticker runtimeTicker) {
	defer ticker.Stop()
	if ctx.Err() != nil {
		return
	}
	runtime.sweepRetention(ctx, runtime.now())
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C():
			if ctx.Err() != nil {
				return
			}
			runtime.sweepRetention(ctx, runtime.now())
		}
	}
}

func (runtime *Runtime) sweepRetention(ctx context.Context, now time.Time) {
	if runtime.requestLogCleaner != nil {
		runtime.requestLogCleaner.Sweep(ctx, now)
	}
	if runtime.stageCleaner != nil {
		if err := runtime.stageCleaner.CleanupCredentialStages(ctx, now); err != nil {
			logrus.WithError(err).WithField("event", "control.credential_stage_cleanup_failed").Warn("credential stage cleanup failed")
		}
	}
	if runtime.affinityCleaner != nil && runtime.affinityPolicy != nil {
		policy := runtime.affinityPolicy()
		if !policy.Valid() {
			logrus.WithField("event", "control.affinity_cleanup_invalid_policy").Warn("durable affinity cleanup skipped because runtime policy is unavailable")
			return
		}
		if err := runtime.affinityCleaner.SweepAffinityBindings(ctx, now, policy); err != nil {
			logrus.WithError(err).WithField("event", "control.affinity_cleanup_failed").Warn("durable affinity cleanup failed")
		}
	}
}

// runBlacklistRelease applies scheduled credential and route-entry blacklist
// releases using only in-memory registry state. It never calls an upstream.
func (runtime *Runtime) runBlacklistRelease(ctx context.Context, ticker runtimeTicker) {
	defer ticker.Stop()
	if ctx.Err() != nil {
		return
	}
	runtime.releaseExpiredBlacklists(runtime.now())
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C():
			if ctx.Err() != nil {
				return
			}
			runtime.releaseExpiredBlacklists(runtime.now())
		}
	}
}

func (runtime *Runtime) releaseExpiredBlacklists(now time.Time) {
	if runtime.registry == nil {
		return
	}
	runtime.registry.ReleaseExpiredBlacklists(now, func(credentialID uint) {
		if runtime.healthStats != nil {
			runtime.healthStats.ClearProblemState(credentialID)
		}
	})
}
