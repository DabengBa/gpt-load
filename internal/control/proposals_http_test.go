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

// TestChangeProposalConcurrentApplyMixedHTTPMCPBindSingleOperation drives the
// real Agent HTTP and MCP apply handlers with simultaneous distinct
// idempotency-key requests for one approved proposal. It asserts that both
// adapters preserve the same durable operation binding and route mutation.
func TestChangeProposalConcurrentApplyMixedHTTPMCPBindSingleOperation(t *testing.T) {
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
	requests := []struct {
		key string
		mcp bool
	}{
		{key: "11111111-1111-4111-8111-111111111111"},
		{key: "22222222-2222-4222-8222-222222222222"},
		{key: "33333333-3333-4333-8333-333333333333", mcp: true},
		{key: "44444444-4444-4444-8444-444444444444", mcp: true},
	}
	type applyOutcome struct {
		status      int
		operationID string
		body        string
	}
	outcomes := make([]applyOutcome, len(requests))
	start := make(chan struct{})
	var wait sync.WaitGroup
	for index, requestSpec := range requests {
		wait.Add(1)
		go func(index int, requestSpec struct {
			key string
			mcp bool
		}) {
			defer wait.Done()
			requestPath := path
			var requestBody *bytes.Reader
			if requestSpec.mcp {
				requestPath = agent.APIPrefix + "/mcp"
				requestBody = bytes.NewReader([]byte(
					`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"agent_apply_proposal","arguments":{"proposal_id":"` + proposal.ProposalID + `","idempotency_key":"` + requestSpec.key + `"}}}`,
				))
			} else {
				requestBody = bytes.NewReader(nil)
			}
			request := httptest.NewRequest(http.MethodPost, requestPath, requestBody)
			request.Header.Set("Authorization", "Bearer "+created.Secret)
			if requestSpec.mcp {
				request.Header.Set("Content-Type", "application/json")
				request.Header.Set("Accept", "application/json, text/event-stream")
				request.Host = "example.com"
			} else {
				request.Header.Set("Idempotency-Key", requestSpec.key)
			}
			recorder := httptest.NewRecorder()
			<-start
			engine.ServeHTTP(recorder, request)
			outcome := applyOutcome{status: recorder.Code, body: recorder.Body.String()}
			var envelope struct {
				Data   agent.ChangeProposalView `json:"data"`
				Result struct {
					StructuredContent agent.ChangeProposalView `json:"structuredContent"`
					IsError           bool                     `json:"isError"`
				} `json:"result"`
			}
			if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err == nil {
				view := envelope.Data
				if requestSpec.mcp {
					if envelope.Result.IsError {
						outcome.body += " (MCP isError=true)"
					} else {
						view = envelope.Result.StructuredContent
					}
				}
				if view.Execution.OperationID != nil {
					outcome.operationID = *view.Execution.OperationID
				}
			}
			outcomes[index] = outcome
		}(index, requestSpec)
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
