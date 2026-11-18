package agent

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/httproute"
	"gpt-load/internal/platform/response"
)

// ModelRouteService exposes only the configured model routes, never upstream
// connection configuration. The control plane owns all persistence.
type ModelRouteService interface {
	ListAgentModelRoutes(context.Context) ([]ModelRouteGroup, error)
	SaveAgentModelRoute(ctx context.Context, groupID uint, entryID string, input ModelRouteInput) (ModelRoute, error)
	DeleteAgentModelRoute(ctx context.Context, groupID uint, entryID string) error
}

type ModelRouteGroup struct {
	GroupID uint         `json:"group_id"`
	Enabled bool         `json:"enabled"`
	Entries []ModelRoute `json:"entries"`
}

type ModelRoute struct {
	EntryID         string `json:"entry_id"`
	ID              string `json:"id"`
	Alias           string `json:"alias"`
	ReasoningEffort string `json:"reasoning_effort"`
	Priority        *int   `json:"priority"`
	Weight          *int   `json:"weight"`
	Enabled         bool   `json:"enabled"`
}

type ModelRouteInput struct {
	ID              RouteField[string] `json:"id"`
	Alias           RouteField[string] `json:"alias"`
	ReasoningEffort RouteField[string] `json:"reasoning_effort"`
	Priority        RouteField[int]    `json:"priority"`
	Weight          RouteField[int]    `json:"weight"`
}

// RouteField distinguishes absence (keep) from null (clear) in a patch.
type RouteField[T string | int] struct {
	Set   bool
	Value *T
}

func (field *RouteField[T]) UnmarshalJSON(data []byte) error {
	field.Set = true
	return json.Unmarshal(data, &field.Value)
}

func (server *Server) SetModelRouteService(service ModelRouteService) {
	server.modelRoutes = service
	server.features.ModelRoutes = service != nil
}

const agentModelRoutesPath = "/model-routes"
const agentGroupModelRoutesPath = "/groups/:group_id/model-routes"
const agentModelRoutePath = agentGroupModelRoutesPath + "/:entry_id"

func modelRouteCapabilities() []EndpointCapability {
	const collection = APIPrefix + "/groups/{group_id}/model-routes"
	const entry = collection + "/{entry_id}"
	return []EndpointCapability{
		{Name: "model-routes.list", Method: http.MethodGet, Path: APIPrefix + agentModelRoutesPath, RequiredScope: ScopeDiagnosticsRead},
		{Name: "model-routes.create", Method: http.MethodPost, Path: collection, RequiredScope: ScopeRoutesWrite, SideEffects: true},
		{Name: "model-routes.update", Method: http.MethodPatch, Path: entry, RequiredScope: ScopeRoutesWrite, SideEffects: true},
		{Name: "model-routes.delete", Method: http.MethodDelete, Path: entry, RequiredScope: ScopeRoutesWrite, SideEffects: true},
	}
}

func (server *Server) modelRouteHTTPRoutes() []httproute.Route {
	return []httproute.Route{
		{Name: "model-routes.list", Methods: []string{http.MethodGet}, Path: agentModelRoutesPath, Handlers: gin.HandlersChain{server.handleListModelRoutes}},
		{Name: "model-routes.create", Methods: []string{http.MethodPost}, Path: agentGroupModelRoutesPath, Handlers: gin.HandlersChain{server.handleWriteModelRoute}},
		{Name: "model-routes.update", Methods: []string{http.MethodPatch}, Path: agentModelRoutePath, Handlers: gin.HandlersChain{server.handleWriteModelRoute}},
		{Name: "model-routes.delete", Methods: []string{http.MethodDelete}, Path: agentModelRoutePath, Handlers: gin.HandlersChain{server.handleWriteModelRoute}},
	}
}

func (server *Server) handleListModelRoutes(c *gin.Context) {
	if !requireScope(c, ScopeDiagnosticsRead) {
		return
	}
	if server.modelRoutes == nil {
		writeModelRouteError(c, app_errors.ErrResourceNotFound)
		return
	}
	groups, err := server.modelRoutes.ListAgentModelRoutes(c.Request.Context())
	if err != nil {
		writeModelRouteError(c, err)
		return
	}
	response.SuccessI18n(c, "common.success", struct {
		SchemaVersion int               `json:"schema_version"`
		Groups        []ModelRouteGroup `json:"groups"`
	}{SchemaVersion: SchemaVersion, Groups: groups})
}

func (server *Server) handleWriteModelRoute(c *gin.Context) {
	if !requireScope(c, ScopeRoutesWrite) {
		return
	}
	if server.modelRoutes == nil {
		writeModelRouteError(c, app_errors.ErrResourceNotFound)
		return
	}
	groupID, apiErr := parseAgentScopedID(c.Param("group_id"))
	if apiErr != nil {
		writeModelRouteError(c, apiErr)
		return
	}
	ctx := c.Request.Context()
	entryID := c.Param("entry_id")
	if c.Request.Method == http.MethodDelete {
		if err := server.modelRoutes.DeleteAgentModelRoute(ctx, groupID, entryID); err != nil {
			writeModelRouteError(c, err)
			return
		}
		response.SuccessI18n(c, "common.success", nil)
		return
	}
	var input ModelRouteInput
	if err := bindStrictAgentJSON(c, &input); err != nil {
		writeModelRouteError(c, mapAgentJSONError(err))
		return
	}
	if (entryID == "" || input.ID.Set) &&
		(input.ID.Value == nil || !validAgentModel(strings.TrimSpace(*input.ID.Value))) ||
		input.Alias.Value != nil && !validAgentModel(strings.TrimSpace(*input.Alias.Value)) {
		writeModelRouteError(c, app_errors.ErrValidation)
		return
	}
	entry, err := server.modelRoutes.SaveAgentModelRoute(ctx, groupID, entryID, input)
	if err != nil {
		writeModelRouteError(c, err)
		return
	}
	response.SuccessI18n(c, "common.success", entry)
}

func writeModelRouteError(c *gin.Context, err error) {
	// Never forward control-layer diagnostics (including structured Data).
	var apiErr *app_errors.APIError
	if errors.As(err, &apiErr) {
		for _, safe := range []*app_errors.APIError{
			app_errors.ErrBadRequest, app_errors.ErrInvalidJSON, app_errors.ErrRequestTooLarge,
			app_errors.ErrValidation, app_errors.ErrResourceNotFound,
			app_errors.ErrControlOperationIncomplete, app_errors.ErrControlRecoveryPending,
		} {
			if apiErr.Code == safe.Code {
				writeAgentServiceError(c, "agent_model_routes", safe)
				return
			}
		}
	}
	writeAgentServiceError(c, "agent_model_routes", app_errors.ErrInternalServer)
}
