package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/url"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/mark3labs/mcp-go/mcp"
	mcpserver "github.com/mark3labs/mcp-go/server"

	"gpt-load/internal/debugcapture"
	app_errors "gpt-load/internal/platform/errors"
)

const (
	mcpToolCapabilities = "agent_get_capabilities"
	mcpToolRequest      = "agent_get_request"
	mcpToolEvidence     = "agent_get_evidence"
	mcpToolRoutes       = "agent_list_routes"
	mcpToolHealth       = "agent_get_health"
	mcpToolUsage        = "agent_get_usage"
	mcpToolProposal     = "agent_get_proposal"
	mcpToolOperation    = "agent_get_operation"
	mcpToolCreate       = "agent_create_proposal"
	mcpToolApply        = "agent_apply_proposal"

	mcpResourceRequests   = "gpt-load://agent/requests/{request_id}"
	mcpResourceEvidence   = "gpt-load://agent/evidence/{request_id}"
	mcpResourceProposals  = "gpt-load://agent/proposals/{proposal_id}"
	mcpResourceOperations = "gpt-load://agent/operations/{operation_id}"
)

type mcpPrincipalContextKey struct{}

type mcpHTTPHandler interface {
	ServeHTTP(http.ResponseWriter, *http.Request)
}

type mcpApplyInput struct {
	ProposalID     string `json:"proposal_id"`
	IdempotencyKey string `json:"idempotency_key"`
}

type mcpRequestInput struct {
	RequestID string `json:"request_id"`
}

type mcpResourceResult struct {
	URI   string `json:"uri"`
	Value any    `json:"value"`
}

type mcpErrorResult struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (server *Server) buildMCPServer() *mcpserver.StreamableHTTPServer {
	mcpService := mcpserver.NewMCPServer(
		"gpt-load-agent-control-plane",
		server.version,
		mcpserver.WithToolCapabilities(false),
		mcpserver.WithResourceCapabilities(false, false),
		mcpserver.WithToolFilter(server.filterMCPTools),
	)
	for _, tool := range server.mcpTools() {
		mcpService.AddTool(tool.tool, tool.handler)
	}
	mcpService.AddResourceTemplates(
		mcpserver.ServerResourceTemplate{
			Template: mcp.NewResourceTemplate(mcpResourceRequests, "Request detail", mcp.WithTemplateDescription("Redacted request log and attempt projection.")),
			Handler:  server.mcpRequestResource,
		},
		mcpserver.ServerResourceTemplate{
			Template: mcp.NewResourceTemplate(mcpResourceEvidence, "Request evidence", mcp.WithTemplateDescription("Redacted debug-capture metadata; raw content is never returned.")),
			Handler:  server.mcpEvidenceResource,
		},
		mcpserver.ServerResourceTemplate{
			Template: mcp.NewResourceTemplate(mcpResourceProposals, "Change proposal", mcp.WithTemplateDescription("Immutable model-route change proposal and execution projection.")),
			Handler:  server.mcpProposalResource,
		},
		mcpserver.ServerResourceTemplate{
			Template: mcp.NewResourceTemplate(mcpResourceOperations, "Control operation", mcp.WithTemplateDescription("Read-only operation recovery projection.")),
			Handler:  server.mcpOperationResource,
		},
	)
	return mcpserver.NewStreamableHTTPServer(
		mcpService,
		mcpserver.WithStateLess(true),
		mcpserver.WithDisableStreaming(true),
		mcpserver.WithHTTPContextFunc(func(ctx context.Context, _ *http.Request) context.Context { return ctx }),
	)
}

type mcpTool struct {
	tool    mcp.Tool
	handler mcpserver.ToolHandlerFunc
}

func (server *Server) mcpTools() []mcpTool {
	objectSchema := func(properties map[string]any, required ...string) json.RawMessage {
		value, _ := json.Marshal(map[string]any{
			"type":                 "object",
			"properties":           properties,
			"required":             required,
			"additionalProperties": false,
		})
		return value
	}
	outputSchema := mcp.ToolOutputSchema{Type: "object", Properties: map[string]any{}}
	tool := func(name, description string, schema json.RawMessage, handler mcpserver.ToolHandlerFunc) mcpTool {
		definition := mcp.NewToolWithRawSchema(name, description, schema)
		definition.OutputSchema = outputSchema
		return mcpTool{tool: definition, handler: handler}
	}
	requestSchema := objectSchema(map[string]any{
		"request_id": map[string]any{"type": "string", "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"},
	}, "request_id")
	proposalSchema := objectSchema(map[string]any{
		"proposal_id": map[string]any{"type": "string"},
	}, "proposal_id")
	createSchema := objectSchema(map[string]any{
		"updates": map[string]any{"type": "array", "items": map[string]any{"type": "object"}},
	}, "updates")
	applySchema := objectSchema(map[string]any{
		"proposal_id":     map[string]any{"type": "string"},
		"idempotency_key": map[string]any{"type": "string"},
	}, "proposal_id", "idempotency_key")
	return []mcpTool{
		tool(mcpToolCapabilities, "Return the authenticated Agent capabilities and scope-aware endpoint contract.", objectSchema(nil), server.mcpCapabilitiesTool),
		tool(mcpToolRequest, "Return one redacted request log with ordered attempts and evidence link.", requestSchema, server.mcpRequestTool),
		tool(mcpToolEvidence, "Return redacted evidence metadata for one request; raw capture content is unavailable.", requestSchema, server.mcpEvidenceTool),
		tool(mcpToolRoutes, "Return the current model-route read projection.", objectSchema(nil), server.mcpRoutesTool),
		tool(mcpToolHealth, "Return current runtime and evidence health.", objectSchema(nil), server.mcpHealthTool),
		tool(mcpToolUsage, "Return the default bounded usage projection.", objectSchema(nil), server.mcpUsageTool),
		tool(mcpToolProposal, "Return one immutable change proposal and execution projection.", proposalSchema, server.mcpProposalTool),
		tool(mcpToolOperation, "Return one read-only control-operation recovery projection.", objectSchema(map[string]any{"operation_id": map[string]any{"type": "string"}}, "operation_id"), server.mcpOperationTool),
		tool(mcpToolCreate, "Create a model-route weight/priority proposal. Approval is separate and administrator-only.", createSchema, server.mcpCreateProposalTool),
		tool(mcpToolApply, "Apply only an already-approved model-route proposal using a caller-supplied idempotency key.", applySchema, server.mcpApplyProposalTool),
	}
}

func (server *Server) filterMCPTools(ctx context.Context, tools []mcp.Tool) []mcp.Tool {
	principal, ok := mcpPrincipal(ctx)
	if !ok {
		return nil
	}
	result := make([]mcp.Tool, 0, len(tools))
	for _, tool := range tools {
		if !server.mcpToolAllowed(tool.Name, principal) {
			continue
		}
		result = append(result, tool)
	}
	return result
}

func (server *Server) mcpToolAllowed(name string, principal Principal) bool {
	switch name {
	case mcpToolCapabilities, mcpToolRequest, mcpToolEvidence, mcpToolRoutes, mcpToolHealth, mcpToolUsage:
		return principal.HasScope(ScopeDiagnosticsRead)
	case mcpToolProposal, mcpToolOperation:
		return server.changeProposals != nil && principal.HasScope(ScopeDiagnosticsRead)
	case mcpToolCreate:
		return server.changeProposals != nil && principal.HasScope(ScopeChangesPropose)
	case mcpToolApply:
		return server.changeProposals != nil && principal.HasScope(ScopeChangesApply)
	default:
		return false
	}
}

func (server *Server) handleMCP(c *gin.Context) {
	if c.Request.Method != http.MethodPost {
		http.Error(c.Writer, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
		c.Abort()
		return
	}
	if err := server.validateMCPOrigin(c.Request); err != nil {
		http.Error(c.Writer, err.Error(), http.StatusForbidden)
		c.Abort()
		return
	}
	principal, ok := CurrentPrincipal(c)
	if !ok || server.mcpHTTP == nil {
		http.Error(c.Writer, http.StatusText(http.StatusServiceUnavailable), http.StatusServiceUnavailable)
		c.Abort()
		return
	}
	request := c.Request.WithContext(context.WithValue(c.Request.Context(), mcpPrincipalContextKey{}, principal))
	server.mcpHTTP.ServeHTTP(c.Writer, request)
	c.Abort()
}

func mcpPrincipal(ctx context.Context) (Principal, bool) {
	if ctx == nil {
		return Principal{}, false
	}
	principal, ok := ctx.Value(mcpPrincipalContextKey{}).(Principal)
	return principal, ok
}

func (server *Server) validateMCPOrigin(request *http.Request) error {
	if request == nil {
		return nil
	}
	originValues := request.Header.Values("Origin")
	if len(originValues) == 0 {
		return nil
	}
	if len(originValues) != 1 || strings.Contains(originValues[0], ",") {
		return errors.New("invalid MCP Origin")
	}
	origin, err := url.Parse(originValues[0])
	if err != nil || origin.Scheme == "" || origin.Host == "" || origin.User != nil || origin.Path != "" || origin.RawQuery != "" || origin.Fragment != "" {
		return errors.New("invalid MCP Origin")
	}
	scheme := "http"
	if request.TLS != nil {
		scheme = "https"
	}
	peer := request.RemoteAddr
	values := request.Header.Values("X-Forwarded-Proto")
	if len(values) > 1 || (len(values) == 1 && strings.Contains(values[0], ",")) {
		return errors.New("invalid forwarded MCP scheme")
	}
	trusted, valid := server.mcpPeerTrusted(peer)
	if len(values) == 1 {
		forwarded := strings.TrimSpace(values[0])
		if forwarded != "http" && forwarded != "https" {
			return errors.New("invalid forwarded MCP scheme")
		}
		if valid && trusted {
			scheme = forwarded
		}
	}
	return nilIfMCPOriginMatches(origin, scheme, request.Host)
}

func nilIfMCPOriginMatches(origin *url.URL, scheme, host string) error {
	if !strings.EqualFold(origin.Scheme, scheme) || !mcpHostsEqual(origin.Host, host, scheme) {
		return errors.New("MCP Origin does not match request origin")
	}
	return nil
}

func mcpHostsEqual(left, right, scheme string) bool {
	leftHost, leftPort := splitMCPHost(left, scheme)
	rightHost, rightPort := splitMCPHost(right, scheme)
	return strings.EqualFold(leftHost, rightHost) && leftPort == rightPort
}

func splitMCPHost(value, scheme string) (string, string) {
	host := value
	if parsed, err := url.Parse("//" + value); err == nil && parsed.Hostname() != "" {
		host = parsed.Hostname()
		port := parsed.Port()
		if port == "" {
			if scheme == "https" {
				port = "443"
			} else {
				port = "80"
			}
		}
		return host, port
	}
	return host, ""
}

func (server *Server) mcpPeerTrusted(remoteAddr string) (bool, bool) {
	host, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		host = remoteAddr
	}
	peer := net.ParseIP(host)
	if peer == nil {
		return false, false
	}
	for _, raw := range server.mcpTrustedProxyCIDRs {
		_, network, err := net.ParseCIDR(raw)
		if err == nil && network.Contains(peer) {
			return true, true
		}
	}
	return false, true
}

func decodeMCPArguments(request mcp.CallToolRequest, target any) error {
	data, err := json.Marshal(request.Params.Arguments)
	if err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	return nil
}

func mcpScopeError(principal Principal, scope Scope) error {
	if !principal.HasScope(scope) {
		return app_errors.ErrForbidden
	}
	return nil
}

func mcpAPIError(err error) *app_errors.APIError {
	if err == nil {
		return nil
	}
	var apiErr *app_errors.APIError
	if errors.As(err, &apiErr) {
		return apiErr
	}
	return app_errors.ErrInternalServer
}

func mcpToolError(err error) (*mcp.CallToolResult, error) {
	apiErr := mcpAPIError(err)
	payload := mcpErrorResult{Code: apiErr.Code, Message: apiErr.Message}
	result := mcp.NewToolResultStructured(payload, apiErr.Code+": "+apiErr.Message)
	result.IsError = true
	return result, nil
}

func mcpJSONResult(value any) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultJSON(value)
}

func (server *Server) mcpCapabilitiesTool(ctx context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	principal, ok := mcpPrincipal(ctx)
	if !ok {
		return mcpToolError(app_errors.ErrUnauthorized)
	}
	if err := mcpScopeError(principal, ScopeDiagnosticsRead); err != nil {
		return mcpToolError(err)
	}
	return mcpJSONResult(Capabilities{
		SchemaVersion:            SchemaVersion,
		Server:                   ServerIdentity{UptimeSeconds: server.uptimeSeconds(), SnapshotRevision: server.snapshotRevision()},
		Caller:                   callerIdentity(principal),
		Endpoints:                agentEndpointCapabilities(server.features),
		Limits:                   Limits{DefaultPageSize: DefaultPageLimit, MaxPageSize: MaxPageLimit},
		Features:                 server.features,
		EvidenceRetentionSeconds: int64(debugcapture.RetentionPeriod().Seconds()),
		DecimalStringFields: []string{
			"request.usage.*",
			"request.cost.estimated_nano_usd",
			"usage.summary.estimated_nano_usd",
			"usage.series[].aggregate.estimated_nano_usd",
			"usage.breakdown.total.estimated_nano_usd",
			"usage.breakdown.rows[].aggregate.estimated_nano_usd",
		},
	})
}

func (server *Server) mcpRequestTool(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return server.mcpReadTool(ctx, request, ScopeDiagnosticsRead, func(id string) (any, error) { return server.mcpRequestView(ctx, id) })
}

func (server *Server) mcpEvidenceTool(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return server.mcpReadTool(ctx, request, ScopeDiagnosticsRead, func(id string) (any, error) { return server.mcpEvidenceView(ctx, id) })
}

func (server *Server) mcpProposalTool(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return server.mcpReadTool(ctx, request, ScopeDiagnosticsRead, func(id string) (any, error) {
		service, err := server.changeProposalService()
		if err != nil {
			return nil, err
		}
		return service.GetChangeProposal(ctx, id)
	})
}

func (server *Server) mcpOperationTool(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	var input struct {
		OperationID string `json:"operation_id"`
	}
	if err := decodeMCPArguments(request, &input); err != nil || input.OperationID == "" {
		return mcpToolError(app_errors.ErrBadRequest)
	}
	principal, ok := mcpPrincipal(ctx)
	if !ok {
		return mcpToolError(app_errors.ErrUnauthorized)
	}
	if err := mcpScopeError(principal, ScopeDiagnosticsRead); err != nil {
		return mcpToolError(err)
	}
	service, err := server.changeProposalService()
	if err != nil {
		return mcpToolError(err)
	}
	value, err := service.GetControlOperation(ctx, input.OperationID)
	if err != nil {
		return mcpToolError(err)
	}
	return mcpJSONResult(value)
}

func (server *Server) mcpRoutesTool(ctx context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	principal, ok := mcpPrincipal(ctx)
	if !ok {
		return mcpToolError(app_errors.ErrUnauthorized)
	}
	if err := mcpScopeError(principal, ScopeDiagnosticsRead); err != nil {
		return mcpToolError(err)
	}
	snapshot := server.currentSnapshot()
	if snapshot == nil {
		return mcpToolError(app_errors.ErrInternalServer)
	}
	return mcpJSONResult(buildRouteIndex(snapshot))
}

func (server *Server) mcpHealthTool(ctx context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	principal, ok := mcpPrincipal(ctx)
	if !ok {
		return mcpToolError(app_errors.ErrUnauthorized)
	}
	if err := mcpScopeError(principal, ScopeDiagnosticsRead); err != nil {
		return mcpToolError(err)
	}
	if server.currentSnapshot() == nil {
		return mcpToolError(app_errors.ErrInternalServer)
	}
	return mcpJSONResult(server.healthView())
}

func (server *Server) mcpUsageTool(ctx context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	principal, ok := mcpPrincipal(ctx)
	if !ok {
		return mcpToolError(app_errors.ErrUnauthorized)
	}
	if err := mcpScopeError(principal, ScopeDiagnosticsRead); err != nil {
		return mcpToolError(err)
	}
	if server.usage == nil {
		return mcpToolError(app_errors.ErrInternalServer)
	}
	now := server.now().UTC().UnixMilli()
	query, apiErr := parseAgentUsageQuery("", now)
	if apiErr != nil {
		return mcpToolError(apiErr)
	}
	report, err := server.usage.QueryUsage(ctx, query)
	if err != nil {
		return mcpToolError(mapAgentReadError(err))
	}
	view, err := mapAgentUsage(now, query, report)
	if err != nil {
		return mcpToolError(err)
	}
	return mcpJSONResult(view)
}

func (server *Server) mcpCreateProposalTool(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	principal, ok := mcpPrincipal(ctx)
	if !ok {
		return mcpToolError(app_errors.ErrUnauthorized)
	}
	if err := mcpScopeError(principal, ScopeChangesPropose); err != nil {
		return mcpToolError(err)
	}
	var input CreateChangeProposalInput
	if err := decodeMCPArguments(request, &input); err != nil {
		return mcpToolError(app_errors.ErrBadRequest)
	}
	service, err := server.changeProposalService()
	if err != nil {
		return mcpToolError(err)
	}
	value, err := service.CreateChangeProposal(ctx, principal, input)
	if err != nil {
		return mcpToolError(err)
	}
	return mcpJSONResult(value)
}

func (server *Server) mcpApplyProposalTool(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	principal, ok := mcpPrincipal(ctx)
	if !ok {
		return mcpToolError(app_errors.ErrUnauthorized)
	}
	if err := mcpScopeError(principal, ScopeChangesApply); err != nil {
		return mcpToolError(err)
	}
	var input mcpApplyInput
	if err := decodeMCPArguments(request, &input); err != nil || !canonicalLowercaseUUIDv4.MatchString(input.ProposalID) || !canonicalLowercaseUUIDv4.MatchString(input.IdempotencyKey) {
		return mcpToolError(app_errors.ErrBadRequest)
	}
	service, err := server.changeProposalService()
	if err != nil {
		return mcpToolError(err)
	}
	value, err := service.ApplyChangeProposal(ctx, principal, input.ProposalID, input.IdempotencyKey)
	if err != nil {
		return mcpToolError(err)
	}
	return mcpJSONResult(value)
}

func (server *Server) mcpReadTool(ctx context.Context, request mcp.CallToolRequest, scope Scope, read func(string) (any, error)) (*mcp.CallToolResult, error) {
	principal, ok := mcpPrincipal(ctx)
	if !ok {
		return mcpToolError(app_errors.ErrUnauthorized)
	}
	if err := mcpScopeError(principal, scope); err != nil {
		return mcpToolError(err)
	}
	var input mcpRequestInput
	if err := decodeMCPArguments(request, &input); err != nil || !canonicalLowercaseUUIDv4.MatchString(input.RequestID) {
		if request.Params.Name == mcpToolProposal {
			var proposal struct {
				ProposalID string `json:"proposal_id"`
			}
			if decodeMCPArguments(request, &proposal) == nil && canonicalLowercaseUUIDv4.MatchString(proposal.ProposalID) {
				value, err := read(proposal.ProposalID)
				if err != nil {
					return mcpToolError(err)
				}
				return mcpJSONResult(value)
			}
		}
		return mcpToolError(app_errors.ErrBadRequest)
	}
	value, err := read(input.RequestID)
	if err != nil {
		return mcpToolError(err)
	}
	return mcpJSONResult(value)
}

func (server *Server) mcpRequestView(ctx context.Context, requestID string) (RequestDetail, error) {
	if server.requestLogs == nil {
		return RequestDetail{}, app_errors.ErrInternalServer
	}
	record, err := server.requestLogs.Get(ctx, requestID)
	if err != nil {
		return RequestDetail{}, mapAgentReadError(err)
	}
	if record.RequestID == "" {
		return RequestDetail{}, app_errors.ErrResourceNotFound
	}
	summary, err := projectRequestSummary(record)
	if err != nil {
		return RequestDetail{}, app_errors.ErrInternalServer
	}
	summary.ErrorSummary = server.scrub(record.ErrorSummary)
	attempts := make([]AttemptView, 0, len(record.Attempts))
	for _, attempt := range record.Attempts {
		attempts = append(attempts, projectAttempt(record.RequestID, attempt, server.scrub))
	}
	evidence, err := server.buildEvidence(ctx, record.RequestID, record.Attempts)
	if err != nil {
		return RequestDetail{}, err
	}
	return RequestDetail{SchemaVersion: SchemaVersion, Request: summary, Attempts: attempts, Evidence: EvidenceLink{State: evidence.State, CaptureRefs: captureIDs(evidence), Path: APIPrefix + "/requests/" + requestID + "/evidence"}}, nil
}

func (server *Server) mcpEvidenceView(ctx context.Context, requestID string) (EvidenceView, error) {
	if server.requestLogs == nil {
		return EvidenceView{}, app_errors.ErrInternalServer
	}
	record, err := server.requestLogs.Get(ctx, requestID)
	if err != nil {
		return EvidenceView{}, mapAgentReadError(err)
	}
	if record.RequestID == "" {
		return EvidenceView{}, app_errors.ErrResourceNotFound
	}
	return server.buildEvidence(ctx, requestID, record.Attempts)
}

func (server *Server) mcpResourcePrincipal(ctx context.Context) (Principal, error) {
	principal, ok := mcpPrincipal(ctx)
	if !ok {
		return Principal{}, app_errors.ErrUnauthorized
	}
	if err := mcpScopeError(principal, ScopeDiagnosticsRead); err != nil {
		return Principal{}, err
	}
	return principal, nil
}

func mcpTextResource(uri string, value any) ([]mcp.ResourceContents, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	return []mcp.ResourceContents{mcp.TextResourceContents{URI: uri, MIMEType: "application/json", Text: string(data)}}, nil
}

func (server *Server) mcpRequestResource(ctx context.Context, request mcp.ReadResourceRequest) ([]mcp.ResourceContents, error) {
	if _, err := server.mcpResourcePrincipal(ctx); err != nil {
		return nil, err
	}
	id, ok := mcpURIValue(request.Params.URI, "gpt-load://agent/requests/")
	if !ok || !canonicalLowercaseUUIDv4.MatchString(id) {
		return nil, app_errors.ErrBadRequest
	}
	value, err := server.mcpRequestView(ctx, id)
	if err != nil {
		return nil, err
	}
	return mcpTextResource(request.Params.URI, value)
}

func (server *Server) mcpEvidenceResource(ctx context.Context, request mcp.ReadResourceRequest) ([]mcp.ResourceContents, error) {
	if _, err := server.mcpResourcePrincipal(ctx); err != nil {
		return nil, err
	}
	id, ok := mcpURIValue(request.Params.URI, "gpt-load://agent/evidence/")
	if !ok || !canonicalLowercaseUUIDv4.MatchString(id) {
		return nil, app_errors.ErrBadRequest
	}
	value, err := server.mcpEvidenceView(ctx, id)
	if err != nil {
		return nil, err
	}
	return mcpTextResource(request.Params.URI, value)
}

func (server *Server) mcpProposalResource(ctx context.Context, request mcp.ReadResourceRequest) ([]mcp.ResourceContents, error) {
	if _, err := server.mcpResourcePrincipal(ctx); err != nil {
		return nil, err
	}
	id, ok := mcpURIValue(request.Params.URI, "gpt-load://agent/proposals/")
	if !ok || !canonicalLowercaseUUIDv4.MatchString(id) {
		return nil, app_errors.ErrBadRequest
	}
	service, err := server.changeProposalService()
	if err != nil {
		return nil, err
	}
	value, err := service.GetChangeProposal(ctx, id)
	if err != nil {
		return nil, err
	}
	return mcpTextResource(request.Params.URI, value)
}

func (server *Server) mcpOperationResource(ctx context.Context, request mcp.ReadResourceRequest) ([]mcp.ResourceContents, error) {
	if _, err := server.mcpResourcePrincipal(ctx); err != nil {
		return nil, err
	}
	id, ok := mcpURIValue(request.Params.URI, "gpt-load://agent/operations/")
	if !ok || id == "" {
		return nil, app_errors.ErrBadRequest
	}
	service, err := server.changeProposalService()
	if err != nil {
		return nil, err
	}
	value, err := service.GetControlOperation(ctx, id)
	if err != nil {
		return nil, err
	}
	return mcpTextResource(request.Params.URI, value)
}

func mcpURIValue(uri, prefix string) (string, bool) {
	if !strings.HasPrefix(uri, prefix) {
		return "", false
	}
	value, err := url.PathUnescape(strings.TrimPrefix(uri, prefix))
	return value, err == nil && value != "" && !strings.Contains(value, "/")
}
