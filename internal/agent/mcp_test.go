package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/platform/i18n"
	"gpt-load/internal/pricing"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/usage"
)

func TestMCPStatelessToolsAndCredentialRevocation(t *testing.T) {
	if err := i18n.Init(); err != nil {
		t.Fatal(err)
	}
	store, _ := newAgentTestStore(t)
	credential, secret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "mcp-diagnostics",
		Scopes: []Scope{ScopeDiagnosticsRead},
	})
	server := NewServer(Dependencies{Credentials: store})
	engine := agentHTTPTestEngine(t, server)

	initialize := mcpRequest(t, engine, secret, `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}`)
	if initialize.Code != http.StatusOK {
		t.Fatalf("initialize = %d %s", initialize.Code, initialize.Body.String())
	}
	tools := mcpRequest(t, engine, secret, `{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}`)
	if tools.Code != http.StatusOK {
		t.Fatalf("tools/list = %d %s", tools.Code, tools.Body.String())
	}
	for _, name := range []string{mcpToolCapabilities, mcpToolRequest, mcpToolEvidence, mcpToolRoutes, mcpToolHealth, mcpToolUsage} {
		if !strings.Contains(tools.Body.String(), name) {
			t.Fatalf("tools/list missing %q: %s", name, tools.Body.String())
		}
	}
	for _, name := range []string{mcpToolCreate, mcpToolApply, mcpToolProposal, mcpToolOperation} {
		if strings.Contains(tools.Body.String(), name) {
			t.Fatalf("diagnostics-only tools/list exposed %q: %s", name, tools.Body.String())
		}
	}

	if _, err := store.Disable(context.Background(), credential.ID); err != nil {
		t.Fatal(err)
	}
	revoked := mcpRequest(t, engine, secret, `{"jsonrpc":"2.0","id":3,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}`)
	if revoked.Code != http.StatusUnauthorized {
		t.Fatalf("revoked initialize = %d %s", revoked.Code, revoked.Body.String())
	}
}

func TestMCPProposalToolScopeAndStructuredResult(t *testing.T) {
	if err := i18n.Init(); err != nil {
		t.Fatal(err)
	}
	store, _ := newAgentTestStore(t)
	_, applySecret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "mcp-applier",
		Scopes: []Scope{ScopeChangesApply},
	})
	stub := &proposalHTTPServiceStub{}
	server := NewServer(Dependencies{Credentials: store})
	server.SetChangeProposalService(stub)
	engine := agentHTTPTestEngine(t, server)

	tools := mcpRequest(t, engine, applySecret, `{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}`)
	if tools.Code != http.StatusOK || !strings.Contains(tools.Body.String(), mcpToolApply) {
		t.Fatalf("apply tools/list = %d %s", tools.Code, tools.Body.String())
	}
	if strings.Contains(tools.Body.String(), mcpToolCreate) || strings.Contains(tools.Body.String(), mcpToolCapabilities) {
		t.Fatalf("apply-only tools/list exposed another scope: %s", tools.Body.String())
	}
	forbiddenRead := mcpRequest(t, engine, applySecret, `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"agent_get_capabilities","arguments":{}}}`)
	if forbiddenRead.Code != http.StatusOK || !strings.Contains(forbiddenRead.Body.String(), `"isError":true`) || !strings.Contains(forbiddenRead.Body.String(), `FORBIDDEN`) {
		t.Fatalf("apply-only direct diagnostics call = %d %s", forbiddenRead.Code, forbiddenRead.Body.String())
	}

	call := mcpRequest(t, engine, applySecret, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"agent_apply_proposal","arguments":{"proposal_id":"11111111-1111-4111-8111-111111111111","idempotency_key":"22222222-2222-4222-8222-222222222222"}}}`)
	if call.Code != http.StatusOK || !strings.Contains(call.Body.String(), `"structuredContent"`) {
		t.Fatalf("apply call = %d %s", call.Code, call.Body.String())
	}
	if stub.applyPrincipal == 0 || stub.applyKey != "22222222-2222-4222-8222-222222222222" {
		t.Fatalf("apply forwarding = principal %d key %q", stub.applyPrincipal, stub.applyKey)
	}
}

func TestMCPResourcesAndStatelessMethodBoundary(t *testing.T) {
	if err := i18n.Init(); err != nil {
		t.Fatal(err)
	}
	store, _ := newAgentTestStore(t)
	_, secret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "mcp-resource-reader",
		Scopes: []Scope{ScopeDiagnosticsRead},
	})
	requestID := "11111111-1111-4111-8111-111111111111"
	server := NewServer(Dependencies{
		Credentials: store,
		RequestLogs: &agentHTTPRequestLogs{record: requestlog.Record{
			RequestID: requestID, CompletedAtMS: 1,
			UsageState: usage.StateNotApplicable, CostState: pricing.CostStateNotApplicable,
			PricingCompleteness: pricing.CompletenessNotApplicable,
		}},
	})
	engine := agentHTTPTestEngine(t, server)

	resource := mcpRequest(t, engine, secret, `{"jsonrpc":"2.0","id":1,"method":"resources/read","params":{"uri":"gpt-load://agent/requests/11111111-1111-4111-8111-111111111111"}}`)
	if resource.Code != http.StatusOK || !strings.Contains(resource.Body.String(), requestID) {
		t.Fatalf("resource read = %d %s", resource.Code, resource.Body.String())
	}

	get := httptest.NewRequest(http.MethodGet, APIPrefix+"/mcp", nil)
	get.Header.Set("Authorization", "Bearer "+secret)
	getRecorder := httptest.NewRecorder()
	engine.ServeHTTP(getRecorder, get)
	if getRecorder.Code != http.StatusMethodNotAllowed {
		t.Fatalf("authenticated GET = %d %s, want 405", getRecorder.Code, getRecorder.Body.String())
	}

	delete := httptest.NewRequest(http.MethodDelete, APIPrefix+"/mcp", nil)
	delete.Header.Set("Authorization", "Bearer "+secret)
	deleteRecorder := httptest.NewRecorder()
	engine.ServeHTTP(deleteRecorder, delete)
	if deleteRecorder.Code != http.StatusMethodNotAllowed {
		t.Fatalf("authenticated DELETE = %d %s, want 405", deleteRecorder.Code, deleteRecorder.Body.String())
	}
}

func TestMCPReadPayloadAndScopeMatchHTTP(t *testing.T) {
	if err := i18n.Init(); err != nil {
		t.Fatal(err)
	}
	store, _ := newAgentTestStore(t)
	_, diagnosticsSecret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "mcp-http-consistency-reader",
		Scopes: []Scope{ScopeDiagnosticsRead},
	})
	_, applySecret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "mcp-http-consistency-applier",
		Scopes: []Scope{ScopeChangesApply},
	})
	requestID := "11111111-1111-4111-8111-111111111111"
	server := NewServer(Dependencies{
		Credentials: store,
		RequestLogs: &agentHTTPRequestLogs{record: requestlog.Record{
			RequestID: requestID, CompletedAtMS: 1,
			UsageState: usage.StateNotApplicable, CostState: pricing.CostStateNotApplicable,
			PricingCompleteness: pricing.CompletenessNotApplicable,
		}},
	})
	engine := agentHTTPTestEngine(t, server)

	httpCapabilitiesRequest := httptest.NewRequest(http.MethodGet, APIPrefix+"/capabilities", nil)
	httpCapabilitiesRequest.Header.Set("Authorization", "Bearer "+diagnosticsSecret)
	httpCapabilitiesRecorder := httptest.NewRecorder()
	engine.ServeHTTP(httpCapabilitiesRecorder, httpCapabilitiesRequest)
	if httpCapabilitiesRecorder.Code != http.StatusOK {
		t.Fatalf("HTTP capabilities = %d %s", httpCapabilitiesRecorder.Code, httpCapabilitiesRecorder.Body.String())
	}
	var httpCapabilitiesEnvelope struct {
		Data Capabilities `json:"data"`
	}
	if err := json.Unmarshal(httpCapabilitiesRecorder.Body.Bytes(), &httpCapabilitiesEnvelope); err != nil {
		t.Fatalf("decode HTTP capabilities: %v", err)
	}

	mcpCapabilitiesRecorder := mcpRequest(t, engine, diagnosticsSecret, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"agent_get_capabilities","arguments":{}}}`)
	if mcpCapabilitiesRecorder.Code != http.StatusOK {
		t.Fatalf("MCP capabilities = %d %s", mcpCapabilitiesRecorder.Code, mcpCapabilitiesRecorder.Body.String())
	}
	var mcpCapabilitiesEnvelope struct {
		Result struct {
			StructuredContent Capabilities `json:"structuredContent"`
		} `json:"result"`
	}
	if err := json.Unmarshal(mcpCapabilitiesRecorder.Body.Bytes(), &mcpCapabilitiesEnvelope); err != nil {
		t.Fatalf("decode MCP capabilities: %v", err)
	}
	if !reflect.DeepEqual(httpCapabilitiesEnvelope.Data, mcpCapabilitiesEnvelope.Result.StructuredContent) {
		t.Fatalf("capabilities diverged: HTTP=%#v MCP=%#v", httpCapabilitiesEnvelope.Data, mcpCapabilitiesEnvelope.Result.StructuredContent)
	}

	httpRequest := httptest.NewRequest(http.MethodGet, APIPrefix+"/requests/"+requestID, nil)
	httpRequest.Header.Set("Authorization", "Bearer "+diagnosticsSecret)
	httpRecorder := httptest.NewRecorder()
	engine.ServeHTTP(httpRecorder, httpRequest)
	if httpRecorder.Code != http.StatusOK {
		t.Fatalf("HTTP request = %d %s", httpRecorder.Code, httpRecorder.Body.String())
	}
	var httpRequestEnvelope struct {
		Data RequestDetail `json:"data"`
	}
	if err := json.Unmarshal(httpRecorder.Body.Bytes(), &httpRequestEnvelope); err != nil {
		t.Fatalf("decode HTTP request: %v", err)
	}

	mcpRequestRecorder := mcpRequest(t, engine, diagnosticsSecret, `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"agent_get_request","arguments":{"request_id":"11111111-1111-4111-8111-111111111111"}}}`)
	if mcpRequestRecorder.Code != http.StatusOK {
		t.Fatalf("MCP request = %d %s", mcpRequestRecorder.Code, mcpRequestRecorder.Body.String())
	}
	var mcpRequestEnvelope struct {
		Result struct {
			StructuredContent RequestDetail `json:"structuredContent"`
		} `json:"result"`
	}
	if err := json.Unmarshal(mcpRequestRecorder.Body.Bytes(), &mcpRequestEnvelope); err != nil {
		t.Fatalf("decode MCP request: %v", err)
	}
	if !reflect.DeepEqual(httpRequestEnvelope.Data, mcpRequestEnvelope.Result.StructuredContent) {
		t.Fatalf("request projection diverged: HTTP=%#v MCP=%#v", httpRequestEnvelope.Data, mcpRequestEnvelope.Result.StructuredContent)
	}

	httpForbiddenRequest := httptest.NewRequest(http.MethodGet, APIPrefix+"/capabilities", nil)
	httpForbiddenRequest.Header.Set("Authorization", "Bearer "+applySecret)
	httpForbiddenRecorder := httptest.NewRecorder()
	engine.ServeHTTP(httpForbiddenRecorder, httpForbiddenRequest)
	if httpForbiddenRecorder.Code != http.StatusForbidden {
		t.Fatalf("HTTP missing-scope status = %d %s", httpForbiddenRecorder.Code, httpForbiddenRecorder.Body.String())
	}
	var httpForbiddenEnvelope struct {
		Code string `json:"code"`
	}
	if err := json.Unmarshal(httpForbiddenRecorder.Body.Bytes(), &httpForbiddenEnvelope); err != nil {
		t.Fatalf("decode HTTP missing-scope response: %v", err)
	}

	mcpForbiddenRecorder := mcpRequest(t, engine, applySecret, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"agent_get_capabilities","arguments":{}}}`)
	if mcpForbiddenRecorder.Code != http.StatusOK {
		t.Fatalf("MCP missing-scope status = %d %s", mcpForbiddenRecorder.Code, mcpForbiddenRecorder.Body.String())
	}
	var mcpForbiddenEnvelope struct {
		Result struct {
			StructuredContent mcpErrorResult `json:"structuredContent"`
		} `json:"result"`
	}
	if err := json.Unmarshal(mcpForbiddenRecorder.Body.Bytes(), &mcpForbiddenEnvelope); err != nil {
		t.Fatalf("decode MCP missing-scope response: %v", err)
	}
	if httpForbiddenEnvelope.Code != mcpForbiddenEnvelope.Result.StructuredContent.Code {
		t.Fatalf("scope error codes diverged: HTTP=%q MCP=%q", httpForbiddenEnvelope.Code, mcpForbiddenEnvelope.Result.StructuredContent.Code)
	}
}

func TestMCPOriginTrustBoundary(t *testing.T) {
	if err := i18n.Init(); err != nil {
		t.Fatal(err)
	}
	store, _ := newAgentTestStore(t)
	_, secret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "mcp-origin",
		Scopes: []Scope{ScopeDiagnosticsRead},
	})

	newEngine := func(cidrs []string) *gin.Engine {
		server := NewServer(Dependencies{Credentials: store, MCPTrustedProxyCIDRs: cidrs})
		return agentHTTPTestEngine(t, server)
	}

	newOriginRequest := func() *http.Request {
		request := httptest.NewRequest(http.MethodPost, APIPrefix+"/mcp", strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}`))
		request.Header.Set("Authorization", "Bearer "+secret)
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("Origin", "https://example.com")
		request.Header.Set("X-Forwarded-Proto", "https")
		request.Host = "example.com"
		request.RemoteAddr = "192.0.2.1:1234"
		return request
	}
	spoofed := newOriginRequest()
	spoofedRecorder := httptest.NewRecorder()
	newEngine(nil).ServeHTTP(spoofedRecorder, spoofed)
	if spoofedRecorder.Code != http.StatusForbidden {
		t.Fatalf("untrusted forwarded scheme = %d %s, want 403", spoofedRecorder.Code, spoofedRecorder.Body.String())
	}

	trusted := newOriginRequest()
	trustedRecorder := httptest.NewRecorder()
	newEngine([]string{"192.0.2.0/24"}).ServeHTTP(trustedRecorder, trusted)
	if trustedRecorder.Code != http.StatusOK {
		t.Fatalf("trusted forwarded scheme = %d %s, want 200", trustedRecorder.Code, trustedRecorder.Body.String())
	}

	multiple := newOriginRequest()
	multiple.Header.Add("X-Forwarded-Proto", "http")
	multipleRecorder := httptest.NewRecorder()
	newEngine([]string{"192.0.2.0/24"}).ServeHTTP(multipleRecorder, multiple)
	if multipleRecorder.Code != http.StatusForbidden {
		t.Fatalf("multiple forwarded schemes = %d %s, want 403", multipleRecorder.Code, multipleRecorder.Body.String())
	}

	invalid := newOriginRequest()
	invalid.Header.Set("X-Forwarded-Proto", "ftp")
	invalidRecorder := httptest.NewRecorder()
	newEngine([]string{"192.0.2.0/24"}).ServeHTTP(invalidRecorder, invalid)
	if invalidRecorder.Code != http.StatusForbidden {
		t.Fatalf("invalid forwarded scheme = %d %s, want 403", invalidRecorder.Code, invalidRecorder.Body.String())
	}

	forwardedHost := newOriginRequest()
	forwardedHost.Header.Del("X-Forwarded-Proto")
	forwardedHost.Header.Set("Origin", "http://evil.example")
	forwardedHost.Header.Set("X-Forwarded-Host", "evil.example")
	forwardedHostRecorder := httptest.NewRecorder()
	newEngine(nil).ServeHTTP(forwardedHostRecorder, forwardedHost)
	if forwardedHostRecorder.Code != http.StatusForbidden {
		t.Fatalf("untrusted forwarded host = %d %s, want 403", forwardedHostRecorder.Code, forwardedHostRecorder.Body.String())
	}
}

func mcpRequest(t *testing.T, engine http.Handler, secret, body string) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(http.MethodPost, APIPrefix+"/mcp", strings.NewReader(body))
	request.Header.Set("Authorization", "Bearer "+secret)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json, text/event-stream")
	request.Host = "example.com"
	recorder := httptest.NewRecorder()
	engine.ServeHTTP(recorder, request)
	return recorder
}
