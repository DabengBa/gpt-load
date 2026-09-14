package control

import (
	"context"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

const retentionInterval = time.Hour

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
	now               func() time.Time
	newTicker         func(time.Duration) runtimeTicker
}

func NewRuntime(
	requestLogCleaner RequestLogCleaner,
	operationRecovery *Service,
	catalogSync *CatalogSyncCoordinator,
) *Runtime {
	runtime := &Runtime{
		requestLogCleaner: requestLogCleaner,
		stageCleaner:      operationRecovery,
		operationRecovery: operationRecovery,
		catalogSync:       catalogSync,
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

func (runtime *Runtime) Run(ctx context.Context) {
	var wait sync.WaitGroup
	if runtime.requestLogCleaner != nil || runtime.stageCleaner != nil {
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
}
