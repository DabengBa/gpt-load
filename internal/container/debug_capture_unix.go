//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package container

import (
	"github.com/gin-gonic/gin"
	"go.uber.org/dig"
	"gorm.io/gorm"

	"gpt-load/internal/app"
	"gpt-load/internal/control"
	"gpt-load/internal/debugcapture"
	"gpt-load/internal/gateway"
	"gpt-load/internal/platform/config"
)

func provideDebugCapture(container *dig.Container) error {
	providers := []any{
		func(db *gorm.DB) (*debugcapture.Store, error) {
			return debugcapture.New(db)
		},
		func(cfg *config.Config, store *debugcapture.Store) *debugcapture.Runtime {
			return debugcapture.NewRuntime(cfg.DebugCaptureEnabled, store)
		},
		func(runtime *debugcapture.Runtime) app.DebugCaptureRuntime {
			return runtime
		},
	}
	for _, provider := range providers {
		if err := container.Provide(provider); err != nil {
			return err
		}
	}
	return nil
}

func configureDebugCapture(container *dig.Container) error {
	return container.Invoke(func(
		cfg *config.Config,
		store *debugcapture.Store,
		runtime *debugcapture.Runtime,
		service *control.Service,
		gatewayHandler *gateway.Handler,
		engine *gin.Engine,
	) error {
		service.SetDebugCaptureReader(store)
		service.SetDebugCaptureHealthReader(runtime)
		if cfg.DebugCaptureEnabled {
			gatewayHandler.SetCaptureFactory(newDebugCaptureFactory(store, runtime))
		}
		engine.Use(gatewayHandler.CaptureMiddleware())
		return nil
	})
}
