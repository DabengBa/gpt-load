package gateway

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/httplifecycle"
)

func TestCaptureLifecycleRetainsRealCancellation(t *testing.T) {
	for _, shutdown := range []bool{false, true} {
		name := "client_cancellation"
		if shutdown {
			name = "server_shutdown"
		}
		t.Run(name, func(t *testing.T) {
			factory := &captureTestFactory{}
			coordinator := httplifecycle.NewCoordinator()
			handler := &Handler{captureFactory: factory, lifecycle: coordinator}
			engine := gin.New()
			engine.Use(coordinator.TrackAll(), handler.CaptureMiddleware())
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			engine.POST("/v1/responses", handler.HTTPModule().BeforeAuth[0], func(c *gin.Context) {
				if shutdown {
					coordinator.BeginShutdown()
				} else {
					cancel()
				}
				c.Status(http.StatusNoContent)
			})
			request := httptest.NewRequest(http.MethodPost, "/v1/responses", nil).WithContext(ctx)
			engine.ServeHTTP(httptest.NewRecorder(), request)
			if !waitForCapture(t, func() bool {
				session, _ := factory.snapshot()
				return session != nil && errors.Is(session.failureValue(), context.Canceled)
			}) {
				t.Fatal("real cancellation did not fail capture after lifecycle release")
			}
		})
	}
}
