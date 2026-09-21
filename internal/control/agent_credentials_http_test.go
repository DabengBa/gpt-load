package control

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/agent"
	"gpt-load/internal/platform/config"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/storage/models"
)

func TestAgentCredentialCreateReplayAndConflictHideSecret(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	fixture.service.SetAgentCredentialStore(agent.NewCredentialStore(fixture.db, fixture.encryption))
	server := NewServer(&config.Config{AuthKey: "admin-auth-key"}, fixture.service)
	engine := gin.New()
	server.RegisterRoutes(engine)

	body := []byte(`{"name":"diagnostics","scopes":["diagnostics:read"]}`)
	key := "00000000-0000-4000-8000-000000000001"
	request := func(payload []byte) *httptest.ResponseRecorder {
		t.Helper()
		req := httptest.NewRequest(http.MethodPost, "/api/agent-credentials", bytes.NewReader(payload))
		req.Header.Set("Authorization", "Bearer admin-auth-key")
		req.Header.Set("Idempotency-Key", key)
		recorder := httptest.NewRecorder()
		engine.ServeHTTP(recorder, req)
		return recorder
	}

	first := request(body)
	if first.Code != http.StatusOK {
		t.Fatalf("first create status = %d: %s", first.Code, first.Body.String())
	}
	var firstEnvelope struct {
		Data AgentCredentialCreateResponse `json:"data"`
	}
	if err := json.Unmarshal(first.Body.Bytes(), &firstEnvelope); err != nil {
		t.Fatalf("decode first create response: %v", err)
	}
	secret := firstEnvelope.Data.Secret
	if secret == "" || !strings.Contains(first.Body.String(), secret) {
		t.Fatalf("first create response did not return secret once: %s", first.Body.String())
	}

	replay := request(body)
	if replay.Code != http.StatusOK {
		t.Fatalf("replay status = %d: %s", replay.Code, replay.Body.String())
	}
	var replayEnvelope struct {
		Data AgentCredentialCreateResponse `json:"data"`
	}
	if err := json.Unmarshal(replay.Body.Bytes(), &replayEnvelope); err != nil {
		t.Fatalf("decode replay response: %v", err)
	}
	if !replayEnvelope.Data.Replayed || replayEnvelope.Data.Secret != "" || strings.Contains(replay.Body.String(), secret) {
		t.Fatalf("replay exposed secret: %s", replay.Body.String())
	}

	conflict := request([]byte(`{"name":"different","scopes":["diagnostics:read"]}`))
	if conflict.Code != http.StatusConflict ||
		!strings.Contains(conflict.Body.String(), app_errors.ErrIdempotencyKeyReused.Code) ||
		strings.Contains(conflict.Body.String(), secret) {
		t.Fatalf("conflicting idempotency request = %d %s", conflict.Code, conflict.Body.String())
	}

	var operation models.ControlOperation
	if err := fixture.db.Where("idempotency_key = ?", key).First(&operation).Error; err != nil {
		t.Fatalf("load control operation: %v", err)
	}
	if strings.Contains(string(operation.CanonicalResult), secret) {
		t.Fatalf("durable canonical result contains plaintext secret: %s", operation.CanonicalResult)
	}
}
