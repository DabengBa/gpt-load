package agent

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/debugcapture"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/httproute"
	"gpt-load/internal/platform/response"
)

// ModuleName is the route-registry module name for the Agent surface.
const ModuleName = "agent"

// APIPrefix is the versioned Agent HTTP namespace.
const APIPrefix = "/api/agent/v1"

// NamespacePrefix is the broader namespace owned by the Agent module.
const NamespacePrefix = "/api/agent"

// requestIDRouteParam is the canonical request identifier path parameter.
const requestIDRouteParam = "request_id"

const (
	proposalIDRouteParam        = "proposal_id"
	operationIDRouteParam       = "operation_id"
	maxAgentJSONBodyBytes int64 = 32 << 20
)

// Module assembles the Agent HTTP module. The module is control-owned but uses
// its own auth policy, so the admin AUTH_KEY and data-plane access keys can
// never authenticate here.
func (server *Server) Module() httproute.Module {
	return httproute.Module{
		Name:              ModuleName,
		Owner:             httproute.OwnerControl,
		Auth:              httproute.AuthAgent,
		Prefix:            APIPrefix,
		NamespacePrefixes: []string{NamespacePrefix},
		Authenticate:      server.authenticate(),
		Routes: []httproute.Route{
			{
				Name:     "capabilities",
				Methods:  []string{http.MethodGet},
				Path:     "/capabilities",
				Handlers: gin.HandlersChain{server.handleCapabilities},
			},
			{
				Name:     "requests.list",
				Methods:  []string{http.MethodGet},
				Path:     "/requests",
				Handlers: gin.HandlersChain{server.handleListRequests},
			},
			{
				Name:     "requests.get",
				Methods:  []string{http.MethodGet},
				Path:     "/requests/:" + requestIDRouteParam,
				Handlers: gin.HandlersChain{server.handleGetRequest},
			},
			{
				Name:     "requests.evidence",
				Methods:  []string{http.MethodGet},
				Path:     "/requests/:" + requestIDRouteParam + "/evidence",
				Handlers: gin.HandlersChain{server.handleGetEvidence},
			},
			{
				Name:     "routes.list",
				Methods:  []string{http.MethodGet},
				Path:     "/routes",
				Handlers: gin.HandlersChain{server.handleListRoutes},
			},
			{
				Name:     "health.get",
				Methods:  []string{http.MethodGet},
				Path:     "/health",
				Handlers: gin.HandlersChain{server.handleHealth},
			},
			{
				Name:     "usage.get",
				Methods:  []string{http.MethodGet},
				Path:     "/usage",
				Handlers: gin.HandlersChain{server.handleUsage},
			},
			{
				Name:     "change-proposals.create",
				Methods:  []string{http.MethodPost},
				Path:     "/change-proposals",
				Handlers: gin.HandlersChain{server.handleCreateChangeProposal},
			},
			{
				Name:     "change-proposals.get",
				Methods:  []string{http.MethodGet},
				Path:     "/change-proposals/:" + proposalIDRouteParam,
				Handlers: gin.HandlersChain{server.handleGetChangeProposal},
			},
			{
				Name:     "change-proposals.apply",
				Methods:  []string{http.MethodPost},
				Path:     "/change-proposals/:" + proposalIDRouteParam + "/apply",
				Handlers: gin.HandlersChain{server.handleApplyChangeProposal},
			},
			{
				Name:     "control-operations.get",
				Methods:  []string{http.MethodGet},
				Path:     "/control-operations/:" + operationIDRouteParam,
				Handlers: gin.HandlersChain{server.handleGetControlOperation},
			},
			{
				Name:     "mcp",
				Methods:  []string{http.MethodPost, http.MethodGet, http.MethodDelete},
				Path:     "/mcp",
				Handlers: gin.HandlersChain{server.handleMCP},
			},
		},
		NotFound:         agentRouteNotFound,
		MethodNotAllowed: agentMethodNotAllowed,
	}
}

func (server *Server) handleCapabilities(c *gin.Context) {
	if !requireScope(c, ScopeDiagnosticsRead) {
		return
	}
	principal, ok := CurrentPrincipal(c)
	if !ok {
		writeAgentServiceError(c, "agent_capabilities", app_errors.ErrUnauthorized)
		return
	}
	features := server.features
	response.SuccessI18n(c, "common.success", Capabilities{
		SchemaVersion: SchemaVersion,
		Server: ServerIdentity{
			UptimeSeconds:    server.uptimeSeconds(),
			SnapshotRevision: server.snapshotRevision(),
		},
		Caller:    callerIdentity(principal),
		Endpoints: agentEndpointCapabilities(features),
		Limits: Limits{
			DefaultPageSize: DefaultPageLimit,
			MaxPageSize:     MaxPageLimit,
		},
		Features:                 features,
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

func agentEndpointCapabilities(features Features) []EndpointCapability {
	endpoints := []EndpointCapability{
		{
			Name:          "capabilities",
			Method:        http.MethodGet,
			Path:          APIPrefix + "/capabilities",
			RequiredScope: ScopeDiagnosticsRead,
			SideEffects:   false,
		},
		{
			Name:          "requests.list",
			Method:        http.MethodGet,
			Path:          APIPrefix + "/requests",
			RequiredScope: ScopeDiagnosticsRead,
			SideEffects:   false,
		},
		{
			Name:          "requests.get",
			Method:        http.MethodGet,
			Path:          APIPrefix + "/requests/{" + requestIDRouteParam + "}",
			RequiredScope: ScopeDiagnosticsRead,
			SideEffects:   false,
		},
		{
			Name:          "requests.evidence",
			Method:        http.MethodGet,
			Path:          APIPrefix + "/requests/{" + requestIDRouteParam + "}/evidence",
			RequiredScope: ScopeDiagnosticsRead,
			SideEffects:   false,
		},
		{
			Name:          "routes.list",
			Method:        http.MethodGet,
			Path:          APIPrefix + "/routes",
			RequiredScope: ScopeDiagnosticsRead,
			SideEffects:   false,
		},
		{
			Name:          "health.get",
			Method:        http.MethodGet,
			Path:          APIPrefix + "/health",
			RequiredScope: ScopeDiagnosticsRead,
			SideEffects:   false,
		},
		{
			Name:          "usage.get",
			Method:        http.MethodGet,
			Path:          APIPrefix + "/usage",
			RequiredScope: ScopeDiagnosticsRead,
			SideEffects:   false,
		},
	}
	if !features.ChangeProposals {
		return endpoints
	}
	return append(endpoints,
		EndpointCapability{
			Name:          "change-proposals.create",
			Method:        http.MethodPost,
			Path:          APIPrefix + "/change-proposals",
			RequiredScope: ScopeChangesPropose,
			SideEffects:   true,
		},
		EndpointCapability{
			Name:          "change-proposals.get",
			Method:        http.MethodGet,
			Path:          APIPrefix + "/change-proposals/{" + proposalIDRouteParam + "}",
			RequiredScope: ScopeDiagnosticsRead,
			SideEffects:   false,
		},
		EndpointCapability{
			Name:          "change-proposals.apply",
			Method:        http.MethodPost,
			Path:          APIPrefix + "/change-proposals/{" + proposalIDRouteParam + "}/apply",
			RequiredScope: ScopeChangesApply,
			SideEffects:   true,
		},
		EndpointCapability{
			Name:          "control-operations.get",
			Method:        http.MethodGet,
			Path:          APIPrefix + "/control-operations/{" + operationIDRouteParam + "}",
			RequiredScope: ScopeDiagnosticsRead,
			SideEffects:   false,
		},
	)
}

func bindStrictAgentJSON(c *gin.Context, target any) error {
	if c.Request.ContentLength > maxAgentJSONBodyBytes {
		return &http.MaxBytesError{Limit: maxAgentJSONBodyBytes}
	}
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxAgentJSONBodyBytes)
	raw, err := io.ReadAll(c.Request.Body)
	if err != nil {
		return err
	}
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return fmt.Errorf("request body must be a JSON object")
	}
	if err := rejectDuplicateAgentJSONFields(raw); err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("decode JSON request: multiple values")
		}
		return err
	}
	return nil
}

func rejectDuplicateAgentJSONFields(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var walk func() error
	walk = func() error {
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		delimiter, ok := token.(json.Delim)
		if !ok {
			return nil
		}
		switch delimiter {
		case '{':
			seen := make(map[string]struct{})
			for decoder.More() {
				keyToken, err := decoder.Token()
				if err != nil {
					return err
				}
				key, ok := keyToken.(string)
				if !ok {
					return fmt.Errorf("object key must be a string")
				}
				if _, duplicate := seen[key]; duplicate {
					return fmt.Errorf("duplicate field %q", key)
				}
				seen[key] = struct{}{}
				if err := walk(); err != nil {
					return err
				}
			}
			_, err = decoder.Token()
			return err
		case '[':
			for decoder.More() {
				if err := walk(); err != nil {
					return err
				}
			}
			_, err = decoder.Token()
			return err
		default:
			return fmt.Errorf("unexpected JSON delimiter %q", delimiter)
		}
	}
	if err := walk(); err != nil {
		return err
	}
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return fmt.Errorf("multiple JSON values")
	}
	return nil
}
