package control

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/storage/models"
)

func TestMultiProtocolHTTPIntegration(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	request := httptest.NewRequest(http.MethodGet, "/api/channels", nil)
	request.Header.Set("Authorization", "Bearer test-auth-key")
	recorder := httptest.NewRecorder()
	engine.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), `"channel_id":"multi_protocol"`) || !strings.Contains(recorder.Body.String(), `"channel_id":"alibaba"`) {
		t.Fatalf("channels status %d: %s", recorder.Code, recorder.Body.String())
	}
	if path := os.Getenv("CHANNEL_INTEGRATION_RESPONSE"); path != "" {
		if err := os.WriteFile(path, recorder.Body.Bytes(), 0600); err != nil {
			t.Fatal(err)
		}
	}
	request = httptest.NewRequest(http.MethodPost, "/api/groups", strings.NewReader(`{"name":"multi integration","channel_id":"multi_protocol","params":{"base_url":"https://gateway.example.test/team/"},"models":[{"id":"test-model","alias":"test-model","alias_enabled":true}],"credential":"fixture-key"}`))
	request.Header.Set("Authorization", "Bearer test-auth-key")
	request.Header.Set("Content-Type", "application/json")
	setRequiredTestIdempotencyHeader(request)
	recorder = httptest.NewRecorder()
	engine.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK || strings.Contains(recorder.Body.String(), "fixture-key") {
		t.Fatalf("create status %d: %s", recorder.Code, recorder.Body.String())
	}
}

func TestMultiProtocolHTTPDefaultNameAndVersionDeploymentPrefix(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	for _, tc := range []struct{ base, name, normalized string }{
		{"https://Gateway.Example.test:8443/v1/", "gateway.example.test", "https://gateway.example.test:8443/v1"},
		{"https://gateway.example.test:8443/team/v1", "gateway.example.test-2", "https://gateway.example.test:8443/team/v1"},
	} {
		payload, err := json.Marshal(map[string]any{
			"channel_id": "multi_protocol", "params": map[string]string{"base_url": tc.base},
			"models": []any{}, "credential": "fixture-key",
		})
		if err != nil {
			t.Fatal(err)
		}
		request := httptest.NewRequest(http.MethodPost, "/api/groups", strings.NewReader(string(payload)))
		request.Header.Set("Authorization", "Bearer test-auth-key")
		request.Header.Set("Content-Type", "application/json")
		setRequiredTestIdempotencyHeader(request)
		recorder := httptest.NewRecorder()
		engine.ServeHTTP(recorder, request)
		var response struct {
			Data struct {
				GroupID   uint   `json:"group_id"`
				GroupName string `json:"group_name"`
			} `json:"data"`
		}
		if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil || recorder.Code != http.StatusOK || response.Data.GroupName != tc.name {
			t.Fatalf("create status %d, body %s, decode error %v", recorder.Code, recorder.Body.String(), err)
		}
		var group models.Group
		if err := fixture.db.First(&group, response.Data.GroupID).Error; err != nil {
			t.Fatal(err)
		}
		var params map[string]string
		if err := json.Unmarshal(group.Params, &params); err != nil {
			t.Fatal(err)
		}
		if params["base_url"] != tc.normalized {
			t.Fatalf("stored base URL = %q, want %q", params["base_url"], tc.normalized)
		}
	}
}
