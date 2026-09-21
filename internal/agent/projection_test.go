package agent

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/debugcapture"
	"gpt-load/internal/platform/httproute"
	"gpt-load/internal/platform/i18n"
)

func TestProjectCaptureUsesExplicitLogicalAttemptLink(t *testing.T) {
	t.Parallel()
	requestID := "11111111-1111-4111-8111-111111111111"
	now := time.UnixMilli(1000)
	attempt := debugcapture.AttemptRecord{}
	attempt.ID = "capture-attempt-1"
	attempt.Sequence = 99
	attempt.State = debugcapture.StateCompleted
	attempt.StartedAt = now
	attempt.Metadata.Fields = make(map[string]any)
	attempt.Metadata.Fields["logical_attempt_id"] = requestID + ":2"
	attempt.Metadata.Fields["authorization"] = "Bearer secret"
	attempt.Metadata.Fields["request_body"] = "sensitive body"
	session := debugcapture.SessionRecord{}
	session.ID = "capture-1"
	session.RequestID = requestID
	session.CreatedAt = now
	session.ExpiresAt = now.Add(time.Hour)
	session.State = debugcapture.StateCompleted
	session.Attempts = []debugcapture.AttemptRecord{attempt}
	view := projectCapture(requestID, session, map[int]struct{}{2: {}})

	if got := view.Attempts[0].StorageSequence; got != 99 {
		t.Fatalf("storage sequence = %d, want 99", got)
	}
	if got := *view.Attempts[0].LogicalAttemptRef; got != requestID+":2" {
		t.Fatalf("logical attempt ref = %q, want %q", got, requestID+":2")
	}
	if got := *view.Attempts[0].LogicalAttemptSequence; got != 2 {
		t.Fatalf("logical attempt sequence = %d, want 2", got)
	}
	if view.Attempts[0].LinkState != AttemptLinkStateLinked {
		t.Fatalf("link state = %q, want %q", view.Attempts[0].LinkState, AttemptLinkStateLinked)
	}
	unknownAttempt := debugcapture.AttemptRecord{}
	unknownAttempt.Metadata.Fields = make(map[string]any)
	unknownAttempt.Metadata.Fields["logical_attempt_id"] = requestID + ":99"
	unknownSession := debugcapture.SessionRecord{}
	unknownSession.ID = "capture-2"
	unknownSession.RequestID = requestID
	unknownSession.CreatedAt = now
	unknownSession.ExpiresAt = now.Add(time.Hour)
	unknownSession.State = debugcapture.StateCompleted
	unknownSession.Attempts = []debugcapture.AttemptRecord{unknownAttempt}
	unknown := projectCapture(requestID, unknownSession, map[int]struct{}{2: {}})
	if unknown.Attempts[0].LinkState != AttemptLinkStateUnlinked ||
		unknown.Attempts[0].LogicalAttemptRef != nil ||
		unknown.Attempts[0].LogicalAttemptSequence != nil {
		t.Fatalf("unknown logical attempt was linked: %#v", unknown.Attempts[0])
	}
	raw, err := json.Marshal(view)
	if err != nil {
		t.Fatalf("marshal projection: %v", err)
	}
	if strings.Contains(string(raw), "Bearer secret") || strings.Contains(string(raw), "sensitive body") {
		t.Fatal("projection contains sensitive capture content")
	}
}

func TestAgentCapabilitiesRequiresDiagnosticsScope(t *testing.T) {
	if err := i18n.Init(); err != nil {
		t.Fatalf("i18n.Init() error = %v", err)
	}
	store, _ := newAgentTestStore(t)
	_, secret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "apply-only",
		Scopes: []Scope{ScopeChangesApply},
	})
	server := NewServer(Dependencies{Credentials: store})
	registry, err := httproute.NewRegistry(server.Module())
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}
	engine := gin.New()
	if err := registry.Bind(engine); err != nil {
		t.Fatalf("Bind() error = %v", err)
	}

	request := httptest.NewRequest("GET", APIPrefix+"/capabilities", nil)
	request.Header.Set("Authorization", "Bearer "+secret)
	recorder := httptest.NewRecorder()
	engine.ServeHTTP(recorder, request)
	if recorder.Code != 403 {
		t.Fatalf("status = %d, want 403", recorder.Code)
	}
}

func TestAgentCapabilitiesReturnsVersionedReadContract(t *testing.T) {
	if err := i18n.Init(); err != nil {
		t.Fatalf("i18n.Init() error = %v", err)
	}
	store, _ := newAgentTestStore(t)
	_, secret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "diagnostics",
		Scopes: []Scope{ScopeDiagnosticsRead},
	})
	server := NewServer(Dependencies{Credentials: store})
	registry, err := httproute.NewRegistry(server.Module())
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}
	engine := gin.New()
	if err := registry.Bind(engine); err != nil {
		t.Fatalf("Bind() error = %v", err)
	}

	request := httptest.NewRequest("GET", APIPrefix+"/capabilities", nil)
	request.Header.Set("Authorization", "Bearer "+secret)
	recorder := httptest.NewRecorder()
	engine.ServeHTTP(recorder, request)
	if recorder.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", recorder.Code, recorder.Body.String())
	}
	var envelope struct {
		Code int          `json:"code"`
		Data Capabilities `json:"data"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode capabilities response: %v", err)
	}
	if envelope.Code != 0 || envelope.Data.SchemaVersion != SchemaVersion {
		t.Fatalf("capabilities envelope = %#v, want code 0/schema version %d", envelope, SchemaVersion)
	}
	if len(envelope.Data.Endpoints) == 0 || envelope.Data.Endpoints[0].SideEffects {
		t.Fatalf("capabilities endpoints = %#v, want read-only endpoint metadata", envelope.Data.Endpoints)
	}
	if !strings.Contains(recorder.Body.String(), `"schema_version":1`) {
		t.Fatalf("response = %s, want schema_version 1", recorder.Body.String())
	}
}
