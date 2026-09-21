package agent

import (
	"errors"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
)

const principalContextKey = "gpt-load.agent.principal"

// authenticate resolves the Bearer Agent credential on every request so a
// disabled or expired credential stops working immediately. It never accepts
// the administrator AUTH_KEY or an AccessKey.
func (server *Server) authenticate() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Cache-Control", "no-store")
		token, ok := bearerToken(c.GetHeader("Authorization"))
		if !ok {
			response.ErrorI18nFromAPIError(c, app_errors.ErrUnauthorized, "auth.invalid_key")
			c.Abort()
			return
		}
		principal, err := server.credentials.Authenticate(c.Request.Context(), token)
		if err != nil {
			if errors.Is(err, errAgentCredentialInvalid) {
				response.ErrorI18nFromAPIError(c, app_errors.ErrUnauthorized, "auth.invalid_key")
				c.Abort()
				return
			}
			writeAgentServiceError(c, "agent_authenticate", err)
			c.Abort()
			return
		}
		c.Set(principalContextKey, principal)
		c.Next()
	}
}

func bearerToken(header string) (string, bool) {
	fields := strings.Fields(header)
	if len(fields) != 2 || !strings.EqualFold(fields[0], "Bearer") || fields[1] == "" {
		return "", false
	}
	return fields[1], true
}

// CurrentPrincipal returns the authenticated Agent identity for this request.
func CurrentPrincipal(c *gin.Context) (Principal, bool) {
	if c == nil {
		return Principal{}, false
	}
	value, exists := c.Get(principalContextKey)
	if !exists {
		return Principal{}, false
	}
	principal, ok := value.(Principal)
	return principal, ok
}

// requireScope enforces one explicit scope at the handler boundary. Missing or
// unknown scopes fail closed with a forbidden response.
func requireScope(c *gin.Context, scope Scope) bool {
	principal, ok := CurrentPrincipal(c)
	if !ok || !principal.HasScope(scope) {
		response.ErrorI18nFromAPIError(c, app_errors.ErrForbidden, "auth.forbidden")
		return false
	}
	return true
}

func callerIdentity(principal Principal) CallerIdentity {
	return CallerIdentity{
		CredentialID: principal.CredentialID,
		Name:         principal.Name,
		Scopes:       append([]Scope(nil), principal.Scopes...),
		ExpiresAtMS:  cloneInt64(principal.ExpiresAtMS),
	}
}

func agentRouteNotFound(c *gin.Context) {
	c.Header("Cache-Control", "no-store")
	response.ErrorI18nFromAPIError(c, agentRouteNotFoundError, "route.not_found")
}

func agentMethodNotAllowed(c *gin.Context) {
	c.Header("Cache-Control", "no-store")
	response.ErrorI18nFromAPIError(c, agentMethodNotAllowedError, "route.method_not_allowed")
}

var (
	agentRouteNotFoundError = &app_errors.APIError{
		HTTPStatus: http.StatusNotFound,
		Code:       "ROUTE_NOT_FOUND",
		Message:    "route.not_found",
	}
	agentMethodNotAllowedError = &app_errors.APIError{
		HTTPStatus: http.StatusMethodNotAllowed,
		Code:       "METHOD_NOT_ALLOWED",
		Message:    "route.method_not_allowed",
	}
)
