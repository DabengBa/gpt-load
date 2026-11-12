package control

import (
	"net/http"
	"testing"

	"github.com/gin-gonic/gin"
	"gpt-load/internal/platform/config"
)

func TestModelProbeHTTPRejectsUnauthenticatedRequest(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	executor := &credentialProbeTestExecutor{result: successfulCredentialProbeResult()}
	fixture.service.executor = executor
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "model-probe-auth"}, fixture.service).RegisterRoutes(engine)
	response := serveCredentialRequest(t, engine, http.MethodPost, "/api/model-probe", `{"targets":[{"group_id":1,"model":"gpt-4o"}]}`, "", "")
	if response.Code != http.StatusUnauthorized || len(executor.recordedCalls()) != 0 {
		t.Fatalf("unauthenticated probe status = %d, calls = %d", response.Code, len(executor.recordedCalls()))
	}
}
