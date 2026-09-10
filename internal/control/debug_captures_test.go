//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package control

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/debugcapture"
	"gpt-load/internal/platform/config"
)

func TestDebugCaptureAdminAPIListsDetailsAndDownloadsPlaintext(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	now := time.Date(2026, time.September, 9, 10, 0, 0, 0, time.UTC)
	store, err := debugcapture.NewWithClock(fixture.db, func() time.Time { return now })
	if err != nil {
		t.Fatal(err)
	}
	fixture.service.SetDebugCaptureReader(store)
	session, err := store.StartSession(debugcapture.SessionMetadata{
		RequestID: "request-1", Protocol: "openai", Fields: map[string]any{"secret": "plaintext-secret"},
	})
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := session.StartAttempt(debugcapture.AttemptMetadata{Fields: map[string]any{"logical_attempt_id": "request-1:1"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := attempt.AppendHeaders(debugcapture.DirectionRequest, []byte("Authorization: Bearer plaintext-secret\r\n")); err != nil {
		t.Fatal(err)
	}
	if err := attempt.AppendBodyPart(debugcapture.DirectionRequest, bytes.NewReader([]byte("raw-request-body"))); err != nil {
		t.Fatal(err)
	}
	if err := attempt.Complete(); err != nil {
		t.Fatal(err)
	}
	if err := session.Complete(); err != nil {
		t.Fatal(err)
	}

	server := NewServer(&config.Config{AuthKey: "admin-secret"}, fixture.service)
	engine := gin.New()
	server.RegisterRoutes(engine)

	list := serveDebugCaptureRequest(engine, http.MethodGet, "/api/debug-captures?limit=1", "admin-secret")
	if list.Code != http.StatusOK || !bytes.Contains(list.Body.Bytes(), []byte("plaintext-secret")) {
		t.Fatalf("list response = %d %s", list.Code, list.Body.String())
	}
	if list.Header().Get("Cache-Control") != "no-store" || list.Header().Get("Pragma") != "no-cache" {
		t.Fatalf("list response cache headers = %#v", list.Header())
	}
	var listEnvelope struct {
		Data debugCaptureListResponse `json:"data"`
	}
	if err := json.Unmarshal(list.Body.Bytes(), &listEnvelope); err != nil {
		t.Fatal(err)
	}
	if len(listEnvelope.Data.Items) != 1 || listEnvelope.Data.Items[0].ID != session.ID() {
		t.Fatalf("list data = %#v", listEnvelope.Data)
	}

	detail := serveDebugCaptureRequest(engine, http.MethodGet, "/api/debug-captures/"+session.ID(), "admin-secret")
	if detail.Code != http.StatusOK || !bytes.Contains(detail.Body.Bytes(), []byte("plaintext-secret")) {
		t.Fatalf("detail response = %d %s", detail.Code, detail.Body.String())
	}
	if detail.Header().Get("Cache-Control") != "no-store" || detail.Header().Get("Pragma") != "no-cache" {
		t.Fatalf("detail response cache headers = %#v", detail.Header())
	}

	download := serveDebugCaptureRequest(engine, http.MethodGet, "/api/debug-captures/"+session.ID()+"/download", "admin-secret")
	if download.Code != http.StatusOK || download.Header().Get("Content-Type") != "application/zip" ||
		download.Header().Get("Content-Disposition") == "" {
		t.Fatalf("download response = %d headers=%#v", download.Code, download.Header())
	}
	archive, err := zip.NewReader(bytes.NewReader(download.Body.Bytes()), int64(download.Body.Len()))
	if err != nil {
		t.Fatal(err)
	}
	var exported []byte
	for _, file := range archive.File {
		reader, err := file.Open()
		if err != nil {
			t.Fatal(err)
		}
		content, readErr := io.ReadAll(reader)
		_ = reader.Close()
		if readErr != nil {
			t.Fatal(readErr)
		}
		exported = append(exported, content...)
	}
	if !bytes.Contains(exported, []byte("plaintext-secret")) || !bytes.Contains(exported, []byte("raw-request-body")) {
		t.Fatalf("ZIP omitted plaintext capture: %q", exported)
	}
}

func TestDebugCaptureAdminAPIErrorResponsesAreNotCacheable(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	server := NewServer(&config.Config{AuthKey: "admin-secret"}, fixture.service)
	engine := gin.New()
	server.RegisterRoutes(engine)

	response := serveDebugCaptureRequest(engine, http.MethodGet, "/api/debug-captures?unknown=value", "admin-secret")
	if response.Code != http.StatusBadRequest {
		t.Fatalf("invalid list status = %d %s", response.Code, response.Body.String())
	}
	if response.Header().Get("Cache-Control") != "no-store" || response.Header().Get("Pragma") != "no-cache" {
		t.Fatalf("invalid list cache headers = %#v", response.Header())
	}
}

func TestDebugCaptureAdminAPIRejectsAccessKeyAndTraversal(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	created, err := fixture.service.CreateAccessKey(t.Context(), AccessKeyCreateRequest{Name: "capture-readonly"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(&config.Config{AuthKey: "admin-secret"}, fixture.service)
	engine := gin.New()
	server.RegisterRoutes(engine)

	accessKey := serveDebugCaptureRequest(engine, http.MethodGet, "/api/debug-captures", created.Key)
	if accessKey.Code != http.StatusForbidden {
		t.Fatalf("access-key status = %d %s, want 403", accessKey.Code, accessKey.Body.String())
	}
	if accessKey.Header().Get("Cache-Control") != "no-store" || accessKey.Header().Get("Pragma") != "no-cache" {
		t.Fatalf("access-key cache headers = %#v", accessKey.Header())
	}
	invalid := serveDebugCaptureRequest(engine, http.MethodGet, "/api/debug-captures/../download", "admin-secret")
	if invalid.Code != http.StatusNotFound && invalid.Code != http.StatusBadRequest {
		t.Fatalf("traversal status = %d %s", invalid.Code, invalid.Body.String())
	}
	malformed := serveDebugCaptureRequest(engine, http.MethodGet, "/api/debug-captures/not-a-capture-id", "admin-secret")
	if malformed.Code != http.StatusBadRequest {
		t.Fatalf("malformed ID status = %d %s", malformed.Code, malformed.Body.String())
	}
}

func serveDebugCaptureRequest(engine *gin.Engine, method, path, token string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(method, path, nil)
	request.Header.Set("Authorization", "Bearer "+token)
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)
	return response
}
