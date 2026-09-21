package agent

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/debugcapture"
	"gpt-load/internal/platform/httproute"
	"gpt-load/internal/platform/i18n"
	"gpt-load/internal/pricing"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/state"
	"gpt-load/internal/usage"
)

type agentHTTPRequestLogs struct {
	record    requestlog.Record
	listPage  requestlog.Page
	err       error
	listCalls int
	getCalls  int
}

func (reader *agentHTTPRequestLogs) List(context.Context, requestlog.ListQuery) (requestlog.Page, error) {
	reader.listCalls++
	if reader.err != nil {
		return requestlog.Page{}, reader.err
	}
	return reader.listPage, nil
}

func (reader *agentHTTPRequestLogs) Get(context.Context, string) (requestlog.Record, error) {
	reader.getCalls++
	if reader.err != nil {
		return requestlog.Record{}, reader.err
	}
	return reader.record, nil
}

type agentHTTPEvidence struct {
	sessions []debugcapture.SessionRecord
	err      error
	calls    int
}

func (reader *agentHTTPEvidence) QuerySessions(debugcapture.SessionQuery) ([]debugcapture.SessionRecord, error) {
	reader.calls++
	if reader.err != nil {
		return nil, reader.err
	}
	return reader.sessions, nil
}

type agentHTTPEvidenceHealth struct {
	health debugcapture.Health
	calls  int
}

type agentHTTPUsage struct {
	calls  int
	report requestlog.UsageReport
}

func (reader *agentHTTPUsage) QueryUsage(context.Context, requestlog.UsageQuery) (requestlog.UsageReport, error) {
	reader.calls++
	return reader.report, nil
}

type agentHTTPStats struct {
	calls int
}

func (reader *agentHTTPStats) Stats() requestlog.Stats {
	reader.calls++
	return requestlog.Stats{}
}

type agentHTTPSnapshot struct {
	snapshot *state.ConfigSnapshot
	calls    int
}

func (reader *agentHTTPSnapshot) Current() *state.ConfigSnapshot {
	reader.calls++
	return reader.snapshot
}

func (reader *agentHTTPEvidenceHealth) Health() (debugcapture.Health, error) {
	reader.calls++
	return reader.health, nil
}

func agentHTTPTestEngine(t *testing.T, server *Server) *gin.Engine {
	t.Helper()
	registry, err := httproute.NewRegistry(server.Module())
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}
	engine := gin.New()
	if err := registry.Bind(engine); err != nil {
		t.Fatalf("Bind() error = %v", err)
	}
	return engine
}

func TestAgentEvidenceEndpointRedactsCaptureContentAndRejectsAdminToken(t *testing.T) {
	if err := i18n.Init(); err != nil {
		t.Fatalf("i18n.Init() error = %v", err)
	}
	store, _ := newAgentTestStore(t)
	_, secret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "diagnostics",
		Scopes: []Scope{ScopeDiagnosticsRead},
	})
	requestID := "11111111-1111-4111-8111-111111111111"
	logs := &agentHTTPRequestLogs{record: requestlog.Record{RequestID: requestID}}
	now := time.UnixMilli(1000)
	session := debugcapture.SessionRecord{}
	session.ID = "capture-1"
	session.RequestID = requestID
	session.CreatedAt = now
	session.ExpiresAt = now.Add(time.Hour)
	session.State = debugcapture.StateCompleted
	session.Metadata.Fields = make(map[string]any)
	session.Metadata.Fields["authorization"] = "Bearer capture-secret"
	session.Metadata.Fields["cookie"] = "session-cookie"
	session.Metadata.Fields["request_body"] = "request-body-secret"
	session.Metadata.Fields["response_body"] = "response-body-secret"
	evidence := &agentHTTPEvidence{sessions: []debugcapture.SessionRecord{session}}
	health := &agentHTTPEvidenceHealth{health: debugcapture.Health{
		Enabled:          true,
		RetentionSeconds: 3600,
	}}
	server := NewServer(Dependencies{
		Credentials:    store,
		RequestLogs:    logs,
		Evidence:       evidence,
		EvidenceHealth: health,
	})
	engine := agentHTTPTestEngine(t, server)

	request := httptest.NewRequest(http.MethodGet, APIPrefix+"/requests/"+requestID+"/evidence", nil)
	request.Header.Set("Authorization", "Bearer "+secret)
	recorder := httptest.NewRecorder()
	engine.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("evidence status = %d, want 200: %s", recorder.Code, recorder.Body.String())
	}
	body := recorder.Body.String()
	for _, sensitive := range []string{
		"capture-secret", "session-cookie", "request-body-secret", "response-body-secret",
	} {
		if strings.Contains(body, sensitive) {
			t.Fatalf("evidence response contains %q: %s", sensitive, body)
		}
	}
	if !strings.Contains(body, `"raw_content_exposed":false`) ||
		!strings.Contains(body, `"state":"captured"`) {
		t.Fatalf("evidence response = %s, want redacted captured state", body)
	}
	if evidence.calls != 1 || health.calls != 1 {
		t.Fatalf("evidence calls = %d, health calls = %d, want one query each", evidence.calls, health.calls)
	}

	evidence.sessions = nil
	notRetainedRequest := httptest.NewRequest(http.MethodGet, APIPrefix+"/requests/"+requestID+"/evidence", nil)
	notRetainedRequest.Header.Set("Authorization", "Bearer "+secret)
	notRetainedRecorder := httptest.NewRecorder()
	engine.ServeHTTP(notRetainedRecorder, notRetainedRequest)
	if notRetainedRecorder.Code != http.StatusOK ||
		!strings.Contains(notRetainedRecorder.Body.String(), `"state":"not_retained"`) {
		t.Fatalf("not-retained response = %d %s", notRetainedRecorder.Code, notRetainedRecorder.Body.String())
	}

	evidence.err = errors.New("query failed")
	unavailableRequest := httptest.NewRequest(http.MethodGet, APIPrefix+"/requests/"+requestID+"/evidence", nil)
	unavailableRequest.Header.Set("Authorization", "Bearer "+secret)
	unavailableRecorder := httptest.NewRecorder()
	engine.ServeHTTP(unavailableRecorder, unavailableRequest)
	if unavailableRecorder.Code != http.StatusOK ||
		!strings.Contains(unavailableRecorder.Body.String(), `"state":"unavailable"`) {
		t.Fatalf("unavailable response = %d %s", unavailableRecorder.Code, unavailableRecorder.Body.String())
	}

	adminRequest := httptest.NewRequest(http.MethodGet, APIPrefix+"/capabilities", nil)
	adminRequest.Header.Set("Authorization", "Bearer sk-admin-control-key")
	adminRecorder := httptest.NewRecorder()
	engine.ServeHTTP(adminRecorder, adminRequest)
	if adminRecorder.Code != http.StatusUnauthorized {
		t.Fatalf("admin token status = %d, want 401", adminRecorder.Code)
	}
}

func TestBuildEvidenceDistinguishesNotRetainedAndUnavailableWithoutWrites(t *testing.T) {
	t.Parallel()
	health := &agentHTTPEvidenceHealth{health: debugcapture.Health{
		Enabled:          true,
		RetentionSeconds: 3600,
	}}
	evidence := &agentHTTPEvidence{}
	server := NewServer(Dependencies{Evidence: evidence, EvidenceHealth: health})
	requestID := "11111111-1111-4111-8111-111111111111"
	attempts := []requestlog.Attempt{{Sequence: 1}}

	view, err := server.buildEvidence(context.Background(), requestID, attempts)
	if err != nil || view.State != EvidenceStateNotRetained || evidence.calls != 1 || health.calls != 1 {
		t.Fatalf("not-retained evidence = %#v, err=%v, calls=%d/%d", view, err, evidence.calls, health.calls)
	}

	evidence.err = errors.New("read failed")
	view, err = server.buildEvidence(context.Background(), requestID, attempts)
	if err != nil || view.State != EvidenceStateUnavailable {
		t.Fatalf("unavailable evidence = %#v, err=%v", view, err)
	}
}

func TestAgentReadRoutesUseOnlyDeclaredReadDependencies(t *testing.T) {
	if err := i18n.Init(); err != nil {
		t.Fatalf("i18n.Init() error = %v", err)
	}
	store, _ := newAgentTestStore(t)
	_, secret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "diagnostics",
		Scopes: []Scope{ScopeDiagnosticsRead},
	})
	requestID := "11111111-1111-4111-8111-111111111111"
	logs := &agentHTTPRequestLogs{}
	logs.record.RequestID = requestID
	logs.record.UsageState = usage.StateNotApplicable
	logs.record.CostState = pricing.CostStateNotApplicable
	logs.record.PricingCompleteness = pricing.CompletenessNotApplicable
	usage := &agentHTTPUsage{}
	stats := &agentHTTPStats{}
	evidence := &agentHTTPEvidence{}
	health := &agentHTTPEvidenceHealth{health: debugcapture.Health{Enabled: true}}
	snapshot := &agentHTTPSnapshot{snapshot: &state.ConfigSnapshot{}}
	server := NewServer(Dependencies{
		Credentials:     store,
		RequestLogs:     logs,
		RequestLogStats: stats,
		Usage:           usage,
		Evidence:        evidence,
		EvidenceHealth:  health,
		Snapshots:       snapshot,
	})
	engine := agentHTTPTestEngine(t, server)
	paths := []string{
		APIPrefix + "/capabilities",
		APIPrefix + "/requests",
		APIPrefix + "/requests/" + requestID,
		APIPrefix + "/routes",
		APIPrefix + "/health",
		APIPrefix + "/usage",
		APIPrefix + "/requests/" + requestID + "/evidence",
	}
	for _, path := range paths {
		request := httptest.NewRequest(http.MethodGet, path, nil)
		request.Header.Set("Authorization", "Bearer "+secret)
		recorder := httptest.NewRecorder()
		engine.ServeHTTP(recorder, request)
		if recorder.Code != http.StatusOK {
			t.Fatalf("GET %s status = %d, want 200: %s", path, recorder.Code, recorder.Body.String())
		}
	}
	if logs.listCalls != 1 || logs.getCalls != 2 || usage.calls != 1 || stats.calls != 1 {
		t.Fatalf("read dependency calls = list:%d get:%d usage:%d stats:%d, want list/usage/stats once and detail/evidence get twice", logs.listCalls, logs.getCalls, usage.calls, stats.calls)
	}
	if evidence.calls != 2 || health.calls != 3 || snapshot.calls == 0 {
		t.Fatalf("read-only dependency calls = evidence:%d health:%d snapshot:%d", evidence.calls, health.calls, snapshot.calls)
	}
}
