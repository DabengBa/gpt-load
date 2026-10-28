package control

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/agent"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/platform/httproute"
	"gpt-load/internal/storage/models"
)

func TestChangeProposalApprovalIsAdminOnly(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "proposal-http-secret")
	groupModels := loadCreatedGroupModels(t, fixture, groupID)
	proposal, err := fixture.service.CreateChangeProposal(t.Context(), agent.Principal{
		CredentialID: 9,
		Scopes:       []agent.Scope{agent.ScopeChangesPropose},
	}, agent.CreateChangeProposalInput{Updates: []agent.ChangeProposalUpdateInput{{
		GroupID:          groupID,
		EntryID:          groupModels[0].EntryID,
		Weight:           agent.ProposalInt{Set: true, Value: 2},
		Priority:         agent.ProposalInt{Set: true, Value: 1},
		ExpectedWeight:   agent.ProposalInt{Set: true, Value: 1},
		ExpectedPriority: agent.ProposalInt{Set: true, Value: 1},
	}}})
	if err != nil {
		t.Fatalf("CreateChangeProposal() error = %v", err)
	}

	server := NewServer(&config.Config{AuthKey: "proposal-admin-auth"}, fixture.service)
	engine := gin.New()
	server.RegisterRoutes(engine)
	path := "/api/agent-change-proposals/" + proposal.ProposalID + "/approve"
	request := httptest.NewRequest(http.MethodPost, path, bytes.NewBufferString(`{"approved_by":"admin"}`))
	request.Header.Set("Authorization", "Bearer proposal-admin-auth")
	recorder := httptest.NewRecorder()
	engine.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("admin approval status = %d: %s", recorder.Code, recorder.Body.String())
	}
	var envelope struct {
		Data agent.ChangeProposalView `json:"data"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode approval response: %v", err)
	}
	if envelope.Data.State != agent.ProposalStateApproved || envelope.Data.ApprovedBy == nil || *envelope.Data.ApprovedBy != "admin" {
		t.Fatalf("approval response = %#v, want approved admin", envelope.Data)
	}

	revokePath := "/api/agent-change-proposals/" + proposal.ProposalID + "/revoke"
	revokeRequest := httptest.NewRequest(http.MethodPost, revokePath, nil)
	revokeRequest.Header.Set("Authorization", "Bearer proposal-admin-auth")
	revokeRecorder := httptest.NewRecorder()
	engine.ServeHTTP(revokeRecorder, revokeRequest)
	if revokeRecorder.Code != http.StatusOK {
		t.Fatalf("admin revoke status = %d: %s", revokeRecorder.Code, revokeRecorder.Body.String())
	}
	var revokeEnvelope struct {
		Data agent.ChangeProposalView `json:"data"`
	}
	if err := json.Unmarshal(revokeRecorder.Body.Bytes(), &revokeEnvelope); err != nil {
		t.Fatalf("decode revoke response: %v", err)
	}
	if revokeEnvelope.Data.State != agent.ProposalStateRevoked {
		t.Fatalf("revoke response = %#v, want revoked", revokeEnvelope.Data)
	}

	agentRequest := httptest.NewRequest(http.MethodPost, path, bytes.NewBufferString(`{"approved_by":"agent"}`))
	agentRequest.Header.Set("Authorization", "Bearer gla_not-an-admin-token")
	agentRecorder := httptest.NewRecorder()
	engine.ServeHTTP(agentRecorder, agentRequest)
	if agentRecorder.Code != http.StatusUnauthorized {
		t.Fatalf("Agent token approval status = %d, want 401: %s", agentRecorder.Code, agentRecorder.Body.String())
	}

	agentRevokeRequest := httptest.NewRequest(http.MethodPost, revokePath, nil)
	agentRevokeRequest.Header.Set("Authorization", "Bearer gla_not-an-admin-token")
	agentRevokeRecorder := httptest.NewRecorder()
	engine.ServeHTTP(agentRevokeRecorder, agentRevokeRequest)
	if agentRevokeRecorder.Code != http.StatusUnauthorized {
		t.Fatalf("Agent token revoke status = %d, want 401: %s", agentRevokeRecorder.Code, agentRevokeRecorder.Body.String())
	}
}

// TestChangeProposalConcurrentHTTPApplyBindsSingleOperation drives concurrent Agent HTTP apply requests for one approved proposal.
func TestChangeProposalConcurrentHTTPApplyBindsSingleOperation(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	credentialStore := agent.NewCredentialStore(fixture.db, fixture.encryption)
	fixture.service.SetAgentCredentialStore(credentialStore)
	created, err := fixture.service.CreateAgentCredentialIdempotent(
		t.Context(),
		"00000000-0000-4000-8000-0000000000a1",
		AgentCredentialCreateRequest{
			Name: "concurrent-applier",
			Scopes: []string{
				string(agent.ScopeChangesPropose),
				string(agent.ScopeChangesApply),
			},
		},
	)
	if err != nil {
		t.Fatalf("CreateAgentCredentialIdempotent() error = %v", err)
	}
	if created.Secret == "" {
		t.Fatal("created agent credential did not return a secret")
	}

	groupID := createGroupWithCredentials(t, fixture, "proposal-concurrent-secret")
	groupModels := loadCreatedGroupModels(t, fixture, groupID)
	if len(groupModels) != 1 || groupModels[0].EntryID == "" {
		t.Fatalf("created group models = %#v, want one persisted entry", groupModels)
	}
	principal := agent.Principal{
		CredentialID: created.ID,
		Scopes:       []agent.Scope{agent.ScopeChangesPropose, agent.ScopeChangesApply},
	}
	proposal, err := fixture.service.CreateChangeProposal(t.Context(), principal, agent.CreateChangeProposalInput{
		Updates: []agent.ChangeProposalUpdateInput{{
			GroupID:          groupID,
			EntryID:          groupModels[0].EntryID,
			Weight:           agent.ProposalInt{Set: true, Value: 2},
			Priority:         agent.ProposalInt{Set: true, Value: 1},
			ExpectedWeight:   agent.ProposalInt{Set: true, Value: 1},
			ExpectedPriority: agent.ProposalInt{Set: true, Value: 1},
		}},
	})
	if err != nil {
		t.Fatalf("CreateChangeProposal() error = %v", err)
	}
	if _, err := fixture.service.ApproveChangeProposal(
		t.Context(),
		proposal.ProposalID,
		ApproveChangeProposalInput{ApprovedBy: "admin"},
	); err != nil {
		t.Fatalf("ApproveChangeProposal() error = %v", err)
	}

	agentServer := agent.NewServer(agent.Dependencies{Credentials: credentialStore})
	agentServer.SetChangeProposalService(fixture.service)
	engine := gin.New()
	registry, err := httproute.NewRegistry(agentServer.Module())
	if err != nil {
		t.Fatalf("httproute.NewRegistry(agent) error = %v", err)
	}
	if err := registry.Bind(engine); err != nil {
		t.Fatalf("registry.Bind(engine) error = %v", err)
	}

	path := agent.APIPrefix + "/change-proposals/" + proposal.ProposalID + "/apply"
	requests := []string{
		"11111111-1111-4111-8111-111111111111",
		"22222222-2222-4222-8222-222222222222",
	}
	type applyOutcome struct {
		status      int
		operationID string
		body        string
	}
	outcomes := make([]applyOutcome, len(requests))
	start := make(chan struct{})
	var wait sync.WaitGroup
	for index, key := range requests {
		wait.Add(1)
		go func(index int, key string) {
			defer wait.Done()
			request := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(nil))
			request.Header.Set("Authorization", "Bearer "+created.Secret)
			request.Header.Set("Idempotency-Key", key)
			recorder := httptest.NewRecorder()
			<-start
			engine.ServeHTTP(recorder, request)
			outcome := applyOutcome{status: recorder.Code, body: recorder.Body.String()}
			var envelope struct {
				Data agent.ChangeProposalView `json:"data"`
			}
			if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err == nil && envelope.Data.Execution.OperationID != nil {
				outcome.operationID = *envelope.Data.Execution.OperationID
			}
			outcomes[index] = outcome
		}(index, key)
	}
	close(start)
	wait.Wait()

	for index, outcome := range outcomes {
		if outcome.status != http.StatusOK {
			t.Fatalf("apply request %d status = %d: %s", index, outcome.status, outcome.body)
		}
		if outcome.operationID == "" {
			t.Fatalf("apply request %d returned no operation id: %s", index, outcome.body)
		}
	}
	if outcomes[0].operationID != outcomes[1].operationID {
		t.Fatalf(
			"concurrent operations = %q and %q, want the same operation",
			outcomes[0].operationID,
			outcomes[1].operationID,
		)
	}

	var bindings int64
	if err := fixture.db.Model(&models.ControlOperation{}).
		Where("proposal_id = ?", proposal.ProposalID).
		Count(&bindings).Error; err != nil {
		t.Fatalf("count proposal operation bindings: %v", err)
	}
	if bindings != 1 {
		t.Fatalf("proposal operation bindings = %d, want 1", bindings)
	}

	updated := loadCreatedGroupModels(t, fixture, groupID)
	if len(updated) != 1 || updated[0].Weight == nil || *updated[0].Weight != 2 {
		t.Fatalf("updated group models = %#v, want one entry with weight 2", updated)
	}
}
