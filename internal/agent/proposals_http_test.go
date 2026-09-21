package agent

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"gpt-load/internal/platform/i18n"
	"gpt-load/internal/storage/models"
)

type proposalHTTPServiceStub struct {
	createPrincipal uint
	createCalled    bool
	applyPrincipal  uint
	applyKey        string
	getProposalID   string
}

func (stub *proposalHTTPServiceStub) CreateChangeProposal(
	_ context.Context,
	principal Principal,
	_ CreateChangeProposalInput,
) (ChangeProposalView, error) {
	stub.createPrincipal = principal.CredentialID
	stub.createCalled = true
	return ChangeProposalView{SchemaVersion: SchemaVersion}, nil
}

func (stub *proposalHTTPServiceStub) ApplyChangeProposal(
	_ context.Context,
	principal Principal,
	proposalID string,
	idempotencyKey string,
) (ChangeProposalView, error) {
	stub.applyPrincipal = principal.CredentialID
	stub.applyKey = idempotencyKey
	return ChangeProposalView{
		SchemaVersion: SchemaVersion,
		ProposalID:    proposalID,
	}, nil
}

func (stub *proposalHTTPServiceStub) GetChangeProposal(
	_ context.Context,
	proposalID string,
) (ChangeProposalView, error) {
	stub.getProposalID = proposalID
	return ChangeProposalView{SchemaVersion: SchemaVersion, ProposalID: proposalID}, nil
}

func (stub *proposalHTTPServiceStub) GetControlOperation(
	_ context.Context,
	operationID string,
) (ControlOperationView, error) {
	return ControlOperationView{SchemaVersion: SchemaVersion, OperationID: operationID}, nil
}

func TestAgentProposalRoutesEnforceScopesAndForwardIdentity(t *testing.T) {
	if err := i18n.Init(); err != nil {
		t.Fatalf("i18n.Init() error = %v", err)
	}
	store, _ := newAgentTestStore(t)
	_, proposeSecret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "proposer",
		Scopes: []Scope{ScopeChangesPropose},
	})
	_, applySecret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "applier",
		Scopes: []Scope{ScopeChangesApply},
	})
	stub := &proposalHTTPServiceStub{}
	server := NewServer(Dependencies{Credentials: store})
	server.SetChangeProposalService(stub)
	engine := agentHTTPTestEngine(t, server)

	createRequest := httptest.NewRequest(
		http.MethodPost,
		APIPrefix+"/change-proposals",
		jsonBody(`{"updates":[{"group_id":1,"entry_id":"entry-1","weight":2,"priority":1,"expected_weight":1,"expected_priority":1}]}`),
	)
	createRequest.Header.Set("Authorization", "Bearer "+proposeSecret)
	createRecorder := httptest.NewRecorder()
	engine.ServeHTTP(createRecorder, createRequest)
	if createRecorder.Code != http.StatusOK || !stub.createCalled || stub.createPrincipal == 0 {
		t.Fatalf("create proposal = %d, created=%v principal=%d body=%s", createRecorder.Code, stub.createCalled, stub.createPrincipal, createRecorder.Body.String())
	}

	const proposalID = "11111111-1111-4111-8111-111111111111"
	const idempotencyKey = "22222222-2222-4222-8222-222222222222"
	applyRequest := httptest.NewRequest(
		http.MethodPost,
		APIPrefix+"/change-proposals/"+proposalID+"/apply",
		nil,
	)
	applyRequest.Header.Set("Authorization", "Bearer "+applySecret)
	applyRequest.Header.Set("Idempotency-Key", idempotencyKey)
	applyRecorder := httptest.NewRecorder()
	engine.ServeHTTP(applyRecorder, applyRequest)
	if applyRecorder.Code != http.StatusOK || stub.applyPrincipal == 0 || stub.applyKey != idempotencyKey {
		t.Fatalf("apply proposal = %d, principal=%d key=%q body=%s", applyRecorder.Code, stub.applyPrincipal, stub.applyKey, applyRecorder.Body.String())
	}
}

func TestAgentProposalApplyEnforcesCredentialState(t *testing.T) {
	if err := i18n.Init(); err != nil {
		t.Fatalf("i18n.Init() error = %v", err)
	}
	store, _ := newAgentTestStore(t)
	cred, secret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "applier",
		Scopes: []Scope{ScopeChangesApply},
	})
	stub := &proposalHTTPServiceStub{}
	server := NewServer(Dependencies{Credentials: store})
	server.SetChangeProposalService(stub)
	engine := agentHTTPTestEngine(t, server)

	const proposalID = "11111111-1111-4111-8111-111111111111"
	const idempotencyKey = "22222222-2222-4222-8222-222222222222"

	// First apply should succeed
	applyRequest := httptest.NewRequest(
		http.MethodPost,
		APIPrefix+"/change-proposals/"+proposalID+"/apply",
		nil,
	)
	applyRequest.Header.Set("Authorization", "Bearer "+secret)
	applyRequest.Header.Set("Idempotency-Key", idempotencyKey)
	applyRecorder := httptest.NewRecorder()
	engine.ServeHTTP(applyRecorder, applyRequest)
	if applyRecorder.Code != http.StatusOK {
		t.Fatalf("first apply = %d, want 200, body=%s", applyRecorder.Code, applyRecorder.Body.String())
	}
	if stub.applyPrincipal != cred.ID {
		t.Fatalf("apply principal = %d, want %d", stub.applyPrincipal, cred.ID)
	}

	if err := store.db.Model(&models.AgentCredential{}).Where("id = ?", cred.ID).
		Update("scopes", models.JSON([]byte(`[]`))).Error; err != nil {
		t.Fatalf("revoke credential scope error = %v", err)
	}
	stub.applyPrincipal = 0
	revokedScopeRequest := httptest.NewRequest(
		http.MethodPost,
		APIPrefix+"/change-proposals/"+proposalID+"/apply",
		nil,
	)
	revokedScopeRequest.Header.Set("Authorization", "Bearer "+secret)
	revokedScopeRequest.Header.Set("Idempotency-Key", "33333333-3333-4333-8333-333333333333")
	revokedScopeRecorder := httptest.NewRecorder()
	engine.ServeHTTP(revokedScopeRecorder, revokedScopeRequest)
	if revokedScopeRecorder.Code != http.StatusUnauthorized && revokedScopeRecorder.Code != http.StatusForbidden {
		t.Fatalf("apply after scope revoke = %d, want 401/403, body=%s", revokedScopeRecorder.Code, revokedScopeRecorder.Body.String())
	}
	if stub.applyPrincipal != 0 {
		t.Fatalf("apply after scope revoke forwarded principal = %d, want 0", stub.applyPrincipal)
	}
	if err := store.db.Model(&models.AgentCredential{}).Where("id = ?", cred.ID).
		Update("scopes", models.JSON([]byte(`["changes:apply"]`))).Error; err != nil {
		t.Fatalf("restore credential scope error = %v", err)
	}

	// Disable credential and verify next apply fails
	stub.applyPrincipal = 0
	if _, err := store.Disable(context.Background(), cred.ID); err != nil {
		t.Fatalf("disable credential error = %v", err)
	}

	applyRequest2 := httptest.NewRequest(
		http.MethodPost,
		APIPrefix+"/change-proposals/"+proposalID+"/apply",
		nil,
	)
	applyRequest2.Header.Set("Authorization", "Bearer "+secret)
	applyRequest2.Header.Set("Idempotency-Key", idempotencyKey)
	applyRecorder2 := httptest.NewRecorder()
	engine.ServeHTTP(applyRecorder2, applyRequest2)
	if applyRecorder2.Code != http.StatusUnauthorized && applyRecorder2.Code != http.StatusForbidden {
		t.Fatalf("apply after disable = %d, want 401/403, body=%s", applyRecorder2.Code, applyRecorder2.Body.String())
	}
	if stub.applyPrincipal != 0 {
		t.Fatalf("apply after disable forwarded principal = %d, want 0", stub.applyPrincipal)
	}
}

func TestAgentProposalApplyRejectsExpiredCredential(t *testing.T) {
	if err := i18n.Init(); err != nil {
		t.Fatalf("i18n.Init() error = %v", err)
	}
	store, _ := newAgentTestStore(t)
	now := time.Now()
	expiredAtMS := now.Add(-time.Hour).UnixMilli()
	_, secret := createTestCredential(t, store, CredentialCreateInput{
		Name:        "expired-applier",
		Scopes:      []Scope{ScopeChangesApply},
		ExpiresAtMS: &expiredAtMS,
	})
	stub := &proposalHTTPServiceStub{}
	server := NewServer(Dependencies{Credentials: store})
	server.SetChangeProposalService(stub)
	engine := agentHTTPTestEngine(t, server)

	const proposalID = "11111111-1111-4111-8111-111111111111"
	const idempotencyKey = "22222222-2222-4222-8222-222222222222"

	applyRequest := httptest.NewRequest(
		http.MethodPost,
		APIPrefix+"/change-proposals/"+proposalID+"/apply",
		nil,
	)
	applyRequest.Header.Set("Authorization", "Bearer "+secret)
	applyRequest.Header.Set("Idempotency-Key", idempotencyKey)
	applyRecorder := httptest.NewRecorder()
	engine.ServeHTTP(applyRecorder, applyRequest)
	if applyRecorder.Code != http.StatusUnauthorized && applyRecorder.Code != http.StatusForbidden {
		t.Fatalf("apply with expired credential = %d, want 401/403, body=%s", applyRecorder.Code, applyRecorder.Body.String())
	}
	if stub.applyPrincipal != 0 {
		t.Fatalf("apply with expired credential forwarded principal = %d, want 0", stub.applyPrincipal)
	}
}

func TestAgentProposalApplyRejectsMissingScope(t *testing.T) {
	if err := i18n.Init(); err != nil {
		t.Fatalf("i18n.Init() error = %v", err)
	}
	store, _ := newAgentTestStore(t)
	_, secret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "no-scope",
		Scopes: []Scope{ScopeDiagnosticsRead}, // No changes:apply scope
	})
	stub := &proposalHTTPServiceStub{}
	server := NewServer(Dependencies{Credentials: store})
	server.SetChangeProposalService(stub)
	engine := agentHTTPTestEngine(t, server)

	const proposalID = "11111111-1111-4111-8111-111111111111"
	const idempotencyKey = "22222222-2222-4222-8222-222222222222"

	applyRequest := httptest.NewRequest(
		http.MethodPost,
		APIPrefix+"/change-proposals/"+proposalID+"/apply",
		nil,
	)
	applyRequest.Header.Set("Authorization", "Bearer "+secret)
	applyRequest.Header.Set("Idempotency-Key", idempotencyKey)
	applyRecorder := httptest.NewRecorder()
	engine.ServeHTTP(applyRecorder, applyRequest)
	if applyRecorder.Code != http.StatusUnauthorized && applyRecorder.Code != http.StatusForbidden {
		t.Fatalf("apply with missing scope = %d, want 401/403, body=%s", applyRecorder.Code, applyRecorder.Body.String())
	}
	if stub.applyPrincipal != 0 {
		t.Fatalf("apply with missing scope forwarded principal = %d, want 0", stub.applyPrincipal)
	}
}

func TestAgentProposalCreateRejectsNonIntegerRouteValues(t *testing.T) {
	if err := i18n.Init(); err != nil {
		t.Fatalf("i18n.Init() error = %v", err)
	}
	store, _ := newAgentTestStore(t)
	_, secret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "proposer",
		Scopes: []Scope{ScopeChangesPropose},
	})
	stub := &proposalHTTPServiceStub{}
	server := NewServer(Dependencies{Credentials: store})
	server.SetChangeProposalService(stub)
	engine := agentHTTPTestEngine(t, server)

	for _, body := range []string{
		`{"updates":[{"group_id":1,"entry_id":"entry","weight":null,"priority":1,"expected_weight":1,"expected_priority":1}]}`,
		`{"updates":[{"group_id":1,"entry_id":"entry","weight":1.5,"priority":1,"expected_weight":1,"expected_priority":1}]}`,
		`{"updates":[{"group_id":1,"entry_id":"entry","weight":"1","priority":1,"expected_weight":1,"expected_priority":1}]}`,
	} {
		request := httptest.NewRequest(http.MethodPost, APIPrefix+"/change-proposals", jsonBody(body))
		request.Header.Set("Authorization", "Bearer "+secret)
		recorder := httptest.NewRecorder()
		engine.ServeHTTP(recorder, request)
		if recorder.Code == http.StatusOK {
			t.Fatalf("body %s accepted with 200: %s", body, recorder.Body.String())
		}
	}
	if stub.createCalled {
		t.Fatal("strict-invalid proposal body reached proposal service")
	}
}

func jsonBody(value string) *strings.Reader {
	return strings.NewReader(value)
}
