package control

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/platform/config"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/storage/models"
)

type facetTestReader struct {
	recordingRequestLogReader
	queries []requestlog.ListQuery
}

func TestRequestLogFacetHTTPDatabase(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	row := models.RequestLog{ID: "00000000-0000-4000-8000-000000000123", CompletedAtMS: 1000, AccessKeyID: 1, Protocol: "openai", ClientModel: "historical-model", Status: "success", StatusCode: 200, AttemptCount: 2}
	if err := fixture.db.Create(&row).Error; err != nil {
		t.Fatal(err)
	}
	attempts := []models.RequestLogAttempt{
		{RequestID: row.ID, Sequence: 1, CompletedAtMS: 1000, GroupID: 12, GroupName: "historical", CredentialID: 1, StatusCode: 429, FailureCategory: "rate_limited", Action: "retry"},
		{RequestID: row.ID, Sequence: 2, CompletedAtMS: 1000, GroupID: 13, GroupName: "final", CredentialID: 1, StatusCode: 200, FailureCategory: "ok", Action: "terminate"},
	}
	if err := fixture.db.Create(&attempts).Error; err != nil {
		t.Fatal(err)
	}
	fixture.service.requestLogs = requestlog.NewService(fixture.db, nil, nil)
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	req := httptest.NewRequest(http.MethodGet, "/api/logs/facets?group_id=12&attempt_status_code=200", nil)
	req.Header.Set("Authorization", "Bearer test-auth-key")
	w := httptest.NewRecorder()
	engine.ServeHTTP(w, req)
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"groups":[{"id":13,"name":"final"}]`) || !strings.Contains(w.Body.String(), `"client_models":[]`) {
		t.Fatalf("HTTP facet result: %d %s", w.Code, w.Body.String())
	}
}

func (reader *facetTestReader) Facets(_ context.Context, query requestlog.ListQuery) (requestlog.FacetResult, error) {
	reader.queries = append(reader.queries, query)
	return requestlog.FacetResult{Groups: []requestlog.FacetGroup{}, ClientModels: []requestlog.FacetClientModel{}}, nil
}

func TestRequestLogFacetAuthorization(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	key, err := fixture.service.CreateAccessKey(t.Context(), AccessKeyCreateRequest{Name: "facet viewer"})
	if err != nil {
		t.Fatal(err)
	}
	reader := &facetTestReader{}
	fixture.service.requestLogs = reader
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	for _, tc := range []struct {
		key  string
		want int
	}{{"", 401}, {key.Key, 403}, {"test-auth-key", 200}} {
		req := httptest.NewRequest(http.MethodGet, "/api/logs/facets", nil)
		if tc.key != "" {
			req.Header.Set("Authorization", "Bearer "+tc.key)
		}
		w := httptest.NewRecorder()
		engine.ServeHTTP(w, req)
		if w.Code != tc.want {
			t.Errorf("key %q: status %d, want %d: %s", tc.key, w.Code, tc.want, w.Body.String())
		}
	}
	if len(reader.queries) != 1 {
		t.Fatalf("facet calls = %d, want 1", len(reader.queries))
	}
	req := httptest.NewRequest(http.MethodGet, "/api/logs/facets?group_id=12&client_model=alpha&attempt_status_code=429&limit=1", nil)
	req.Header.Set("Authorization", "Bearer test-auth-key")
	w := httptest.NewRecorder()
	engine.ServeHTTP(w, req)
	if w.Code != 200 || len(reader.queries) != 2 {
		t.Fatalf("facet query: %d %s", w.Code, w.Body.String())
	}
	q := reader.queries[1]
	if q.GroupID == nil || *q.GroupID != 12 || q.ClientModel != "alpha" || q.AttemptStatusCode == nil || *q.AttemptStatusCode != 429 {
		t.Fatalf("parsed query = %#v", q)
	}
	if !strings.Contains(w.Body.String(), `"groups":[]`) || !strings.Contains(w.Body.String(), `"client_models":[]`) {
		t.Fatalf("empty arrays: %s", w.Body.String())
	}
	req = httptest.NewRequest(http.MethodGet, "/api/logs/facets?group_id=invalid", nil)
	req.Header.Set("Authorization", "Bearer test-auth-key")
	w = httptest.NewRecorder()
	engine.ServeHTTP(w, req)
	if w.Code != 400 || len(reader.queries) != 2 {
		t.Fatalf("invalid query: %d %s", w.Code, w.Body.String())
	}
}
