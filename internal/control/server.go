package control

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"

	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"

	"gpt-load/internal/platform/config"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/i18n"
	"gpt-load/internal/platform/response"
	"gpt-load/internal/platform/utils"
	"gpt-load/internal/releasecheck"
)

// ReleaseUpdateChecker is the control-plane on-demand view of the public release checker.
type ReleaseUpdateChecker interface {
	Check(context.Context, bool) (*releasecheck.Update, error)
}

type Server struct {
	authDigest        [sha256.Size]byte
	service           *Service
	systemInfo        systemInfoResponse
	authFailures      *authFailureLimiter
	compareDigest     func([]byte, []byte) int
	logger            *logrus.Logger
	authFailureEvents *utils.RateLimitedEventCounter
	startedAt         time.Time
	now               func() time.Time
	releaseChecker    ReleaseUpdateChecker
}

const maxControlJSONBodyBytes int64 = 32 << 20

func NewServer(cfg *config.Config, service *Service) *Server {
	now := time.Now
	if err := loadModelTraceConfigFromEnvironment(os.Getenv); err != nil {
		logrus.WithError(err).Error("invalid ModelTrace configuration; fingerprint detection is disabled")
	}

	return &Server{
		authDigest:    sha256.Sum256([]byte(cfg.AuthKey)),
		service:       service,
		systemInfo:    newSystemInfoResponse(cfg),
		authFailures:  newAuthFailureLimiter(),
		compareDigest: subtle.ConstantTimeCompare,
		logger:        logrus.StandardLogger(),
		startedAt:     now().UTC(),
		now:           now,
		authFailureEvents: utils.NewRateLimitedEventCounter(
			time.Minute,
			time.Now,
		),
	}
}

// NewServerWithReleaseUpdateChecker wires the on-demand public update checker.
func NewServerWithReleaseUpdateChecker(
	cfg *config.Config,
	service *Service,
	releaseChecker ReleaseUpdateChecker,
) *Server {
	server := NewServer(cfg, service)
	server.releaseChecker = releaseChecker
	return server
}

func (s *Server) handleGetSettings(c *gin.Context) {
	result, err := s.service.GetSettings(c.Request.Context())
	if err != nil {
		writeServiceError(c, "get_settings", err)
		return
	}
	writeSettingsResponse(c, result)
}

func (s *Server) handleUpdateSettings(c *gin.Context) {
	var request SettingsUpdateRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "update_settings", mapControlJSONError(err))
		return
	}
	result, err := s.service.UpdateSettings(c.Request.Context(), request)
	if err != nil {
		writeServiceError(c, "update_settings", err)
		return
	}
	writeSettingsResponse(c, result)
}

func (s *Server) handleGetGroupSummary(c *gin.Context) {
	id, ok := groupID(c, "get_group_summary")
	if !ok {
		return
	}
	result, err := s.service.GetGroupSummary(c.Request.Context(), id)
	if err != nil {
		writeServiceError(c, "get_group_summary", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleGetGroupSettings(c *gin.Context) {
	id, ok := groupID(c, "get_group_settings")
	if !ok {
		return
	}
	result, err := s.service.GetGroupSettings(c.Request.Context(), id)
	if err != nil {
		writeServiceError(c, "get_group_settings", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleCreateGroup(c *gin.Context) {
	idempotencyKey, ok := requiredIdempotencyKey(c, "create_group")
	if !ok {
		return
	}
	var request GroupCreateRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "create_group", mapControlJSONError(err))
		return
	}
	result, err := s.service.CreateGroupIdempotent(
		c.Request.Context(),
		idempotencyKey,
		request,
	)
	if err != nil {
		writeServiceError(c, "create_group", err)
		return
	}
	setMutationResourceLocator(
		c,
		fmt.Sprintf("group:%d", result.GroupID),
	)
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleCopyGroup(c *gin.Context) {
	idempotencyKey, ok := requiredIdempotencyKey(c, "copy_group")
	if !ok {
		return
	}
	id, ok := groupID(c, "copy_group")
	if !ok {
		return
	}
	if err := bindOptionalEmptyJSONObject(c); err != nil {
		writeServiceError(c, "copy_group", mapControlJSONError(err))
		return
	}
	result, err := s.service.CopyGroupIdempotent(
		c.Request.Context(),
		idempotencyKey,
		id,
	)
	if err != nil {
		writeServiceError(c, "copy_group", err)
		return
	}
	setMutationResourceLocator(
		c,
		fmt.Sprintf("group:%d", result.GroupID),
	)
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleUpdateGroupSettings(c *gin.Context) {
	id, ok := groupID(c, "update_group_settings")
	if !ok {
		return
	}
	var request GroupSettingsUpdateRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "update_group_settings", mapControlJSONError(err))
		return
	}
	result, err := s.service.UpdateGroupSettings(c.Request.Context(), id, request)
	if err != nil {
		writeServiceError(c, "update_group_settings", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleUpdateGroupModels(c *gin.Context) {
	id, ok := groupID(c, "update_group_models")
	if !ok {
		return
	}
	var request GroupModelsUpdateRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "update_group_models", mapControlJSONError(err))
		return
	}
	result, err := s.service.UpdateGroupModels(c.Request.Context(), id, request)
	if err != nil {
		writeServiceError(c, "update_group_models", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleGetGroupModels(c *gin.Context) {
	id, ok := groupID(c, "get_group_models")
	if !ok {
		return
	}
	result, err := s.service.GetGroupModels(c.Request.Context(), id)
	if err != nil {
		writeServiceError(c, "get_group_models", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleDeleteGroup(c *gin.Context) {
	id, ok := groupID(c, "delete_group")
	if !ok {
		return
	}
	if err := s.service.DeleteGroup(c.Request.Context(), id); err != nil {
		writeServiceError(c, "delete_group", err)
		return
	}
	response.SuccessI18n(c, "common.success", nil)
}

func (s *Server) handleGetGroupCredential(c *gin.Context) {
	groupID, ok := groupID(c, "get_group_credential")
	if !ok {
		return
	}
	result, err := s.service.GetGroupCredential(c.Request.Context(), groupID)
	if err != nil {
		writeServiceError(c, "get_group_credential", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) resolveGroupCredential(c *gin.Context) {
	expectedID, err := strconv.ParseUint(c.GetHeader("X-Credential-ID"), 10, strconv.IntSize)
	if err != nil || expectedID == 0 {
		writeServiceError(c, "resolve_group_credential", app_errors.ErrValidation)
		c.Abort()
		return
	}
	id, ok := groupID(c, "resolve_group_credential")
	if !ok {
		c.Abort()
		return
	}
	credentialID, err := s.service.currentGroupCredentialID(c.Request.Context(), id)
	if err == nil && uint(expectedID) != credentialID {
		err = app_errors.ErrCredentialVersionConflict
	}
	if err == nil && credentialID == 0 {
		err = credentialNotFoundError()
	}
	if err != nil {
		writeServiceError(c, "resolve_group_credential", err)
		c.Abort()
		return
	}
	// Pin the ID before auditing and mutation so a concurrent replacement is
	// rejected by the existing group-and-ID checks, never silently retargeted.
	c.Params = append(c.Params, gin.Param{Key: "credential_id", Value: strconv.FormatUint(uint64(credentialID), 10)})
}

func (s *Server) handleRevealGroupCredential(c *gin.Context) {
	groupID, ok := groupID(c, "reveal_group_credential")
	if !ok {
		return
	}
	credentialID, ok := credentialID(c, "reveal_group_credential")
	if !ok {
		return
	}
	if err := bindOptionalEmptyJSONObject(c); err != nil {
		writeServiceError(c, "reveal_group_credential", mapControlJSONError(err))
		return
	}
	result, err := s.service.RevealGroupCredential(c.Request.Context(), groupID, credentialID)
	if err != nil {
		writeServiceError(c, "reveal_group_credential", err)
		return
	}
	setSecretResponseHeaders(c)
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleUpdateGroupCredential(c *gin.Context) {
	groupID, ok := groupID(c, "update_group_credential")
	if !ok {
		return
	}
	credentialID, ok := credentialID(c, "update_group_credential")
	if !ok {
		return
	}
	var request CredentialUpdateRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "update_group_credential", mapControlJSONError(err))
		return
	}
	result, err := s.service.UpdateGroupCredential(c.Request.Context(), groupID, credentialID, request)
	if err != nil {
		writeServiceError(c, "update_group_credential", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleDeleteGroupCredential(c *gin.Context) {
	groupID, ok := groupID(c, "delete_group_credential")
	if !ok {
		return
	}
	credentialID, ok := credentialID(c, "delete_group_credential")
	if !ok {
		return
	}
	if err := bindOptionalEmptyJSONObject(c); err != nil {
		writeServiceError(c, "delete_group_credential", mapControlJSONError(err))
		return
	}
	if err := s.service.DeleteGroupCredential(c.Request.Context(), groupID, credentialID); err != nil {
		writeServiceError(c, "delete_group_credential", err)
		return
	}
	response.SuccessI18n(c, "common.success", nil)
}

func (s *Server) handleRestoreGroupCredential(c *gin.Context) {
	groupID, ok := groupID(c, "restore_group_credential")
	if !ok {
		return
	}
	credentialID, ok := credentialID(c, "restore_group_credential")
	if !ok {
		return
	}
	if err := bindOptionalEmptyJSONObject(c); err != nil {
		writeServiceError(c, "restore_group_credential", mapControlJSONError(err))
		return
	}
	result, err := s.service.RestoreGroupCredential(c.Request.Context(), groupID, credentialID)
	if err != nil {
		writeServiceError(c, "restore_group_credential", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleTestGroupCredential(c *gin.Context) {
	groupID, ok := groupID(c, "test_group_credential")
	if !ok {
		return
	}
	credentialID, ok := credentialID(c, "test_group_credential")
	if !ok {
		return
	}
	if err := bindOptionalEmptyJSONObject(c); err != nil {
		writeServiceError(c, "test_group_credential", mapControlJSONError(err))
		return
	}
	result, err := s.service.TestGroupCredential(c.Request.Context(), groupID, credentialID)
	if err != nil {
		writeServiceError(c, "test_group_credential", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleImportGroupCredentials(c *gin.Context) {
	id, ok := groupID(c, "import_group_credentials")
	if !ok {
		return
	}
	idempotencyKey, ok := requiredIdempotencyKey(c, "import_group_credentials")
	if !ok {
		return
	}
	var request CredentialImportRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "import_group_credentials", mapControlJSONError(err))
		return
	}
	result, err := s.service.ImportGroupCredentialsIdempotent(
		c.Request.Context(), idempotencyKey, id, request,
	)
	if err != nil {
		writeServiceError(c, "import_group_credentials", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleDiscoverGroupModels(c *gin.Context) {
	id, ok := groupID(c, "discover_group_models")
	if !ok {
		return
	}
	if err := bindOptionalEmptyJSONObject(c); err != nil {
		writeServiceError(c, "discover_group_models", mapControlJSONError(err))
		return
	}
	result, err := s.service.DiscoverGroupModels(c.Request.Context(), id)
	if err != nil {
		writeServiceError(c, "discover_group_models", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleDiscoverModels(c *gin.Context) {
	var request ModelDiscoveryRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeDiscoveryError(c, mapControlJSONError(err))
		return
	}
	result, err := s.service.DiscoverModels(c.Request.Context(), request)
	if err != nil {
		writeDiscoveryError(c, err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleCreateAccessKey(c *gin.Context) {
	idempotencyKey, ok := requiredIdempotencyKey(c, "create_access_key")
	if !ok {
		return
	}
	var request AccessKeyCreateRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "create_access_key", mapControlJSONError(err))
		return
	}
	result, err := s.service.CreateAccessKeyIdempotent(
		c.Request.Context(),
		idempotencyKey,
		request,
	)
	if err != nil {
		writeServiceError(c, "create_access_key", err)
		return
	}
	setMutationResourceLocator(
		c,
		fmt.Sprintf("access-key:%d", result.ID),
	)
	setSecretResponseHeaders(c)
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleListAccessKeyOptions(c *gin.Context) {
	result, err := s.service.ListAccessKeyOptions(c.Request.Context())
	if err != nil {
		writeServiceError(c, "list_access_key_options", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleRevealAccessKey(c *gin.Context) {
	id, ok := accessKeyID(c)
	if !ok {
		return
	}
	result, err := s.service.RevealAccessKey(c.Request.Context(), id)
	if err != nil {
		writeServiceError(c, "reveal_access_key", err)
		return
	}
	setSecretResponseHeaders(c)
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleRotateAccessKey(c *gin.Context) {
	id, ok := accessKeyID(c)
	if !ok {
		return
	}
	idempotencyKey, ok := requiredIdempotencyKey(c, "rotate_access_key")
	if !ok {
		return
	}
	if err := bindOptionalEmptyJSONObject(c); err != nil {
		writeServiceError(c, "rotate_access_key", mapControlJSONError(err))
		return
	}
	result, err := s.service.RotateAccessKeyIdempotent(
		c.Request.Context(),
		idempotencyKey,
		id,
	)
	if err != nil {
		writeServiceError(c, "rotate_access_key", err)
		return
	}
	setMutationResourceLocator(c, fmt.Sprintf("access-key:%d", result.ID))
	setSecretResponseHeaders(c)
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleUpdateAccessKey(c *gin.Context) {
	id, ok := accessKeyID(c)
	if !ok {
		return
	}
	var request AccessKeyUpdateRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "update_access_key", mapControlJSONError(err))
		return
	}
	result, err := s.service.UpdateAccessKey(c.Request.Context(), id, request)
	if err != nil {
		writeServiceError(c, "update_access_key", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleResetAccessKeyCostLimits(c *gin.Context) {
	id, ok := accessKeyID(c)
	if !ok {
		return
	}
	var request AccessKeyCostLimitResetRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "reset_access_key_cost_limits", mapControlJSONError(err))
		return
	}
	if err := s.service.ResetAccessKeyCostLimitRules(
		c.Request.Context(),
		id,
		request.RuleIDs,
	); err != nil {
		writeServiceError(c, "reset_access_key_cost_limits", err)
		return
	}
	response.SuccessI18n(c, "common.success", nil)
}

func (s *Server) handleDeleteAccessKey(c *gin.Context) {
	id, ok := accessKeyID(c)
	if !ok {
		return
	}
	if err := s.service.DeleteAccessKey(c.Request.Context(), id); err != nil {
		writeServiceError(c, "delete_access_key", err)
		return
	}
	response.SuccessI18n(c, "common.success", nil)
}

func (s *Server) handleModelRouteScheduleIndex(c *gin.Context) {
	result, err := s.service.GetModelRouteScheduleIndex()
	if err != nil {
		writeServiceError(c, "list_model_route_schedule", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleModelRouteScheduleDetail(c *gin.Context) {
	request, apiErr := parseModelRouteScheduleDetailQuery(c)
	if apiErr != nil {
		writeServiceError(c, "get_model_route_schedule_detail", apiErr)
		return
	}
	result, err := s.service.GetModelRouteScheduleDetail(request)
	if err != nil {
		writeServiceError(c, "get_model_route_schedule_detail", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleUpdateModelRouteSchedule(c *gin.Context) {
	var request modelRouteSchedulePatchRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "update_model_route_schedule", mapControlJSONError(err))
		return
	}
	result, err := s.service.UpdateModelRouteSchedule(c.Request.Context(), request)
	if err != nil {
		writeServiceError(c, "update_model_route_schedule", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleRecoverModelRouteScheduleEntry(c *gin.Context) {
	var request modelRouteScheduleRecoverRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "recover_model_route_schedule", mapControlJSONError(err))
		return
	}
	result, err := s.service.RecoverModelRouteScheduleEntry(request)
	if err != nil {
		writeServiceError(c, "recover_model_route_schedule", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func bindStrictJSON(c *gin.Context, target any) error {
	if c.Request.ContentLength > maxControlJSONBodyBytes {
		return &http.MaxBytesError{Limit: maxControlJSONBodyBytes}
	}
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxControlJSONBodyBytes)
	raw, err := io.ReadAll(c.Request.Body)
	if err != nil {
		return err
	}
	return decodeStrictControlJSONObject(raw, target)
}

func bindOptionalEmptyJSONObject(c *gin.Context) error {
	decoder, err := newControlJSONDecoder(c)
	if err != nil {
		return err
	}
	var object map[string]json.RawMessage
	if err := decoder.Decode(&object); err != nil {
		if errors.Is(err, io.EOF) {
			return nil
		}
		return err
	}
	if object == nil || len(object) != 0 {
		return fmt.Errorf("request body must be an empty JSON object")
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("decode JSON request: multiple values")
		}
		return err
	}
	return nil
}

func newControlJSONDecoder(c *gin.Context) (*json.Decoder, error) {
	if c.Request.ContentLength > maxControlJSONBodyBytes {
		return nil, &http.MaxBytesError{Limit: maxControlJSONBodyBytes}
	}
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxControlJSONBodyBytes)
	return json.NewDecoder(c.Request.Body), nil
}

func mapControlJSONError(err error) *app_errors.APIError {
	if errors.Is(err, app_errors.ErrValidation) {
		return app_errors.ErrValidation
	}
	var maxBytesError *http.MaxBytesError
	if errors.As(err, &maxBytesError) {
		return app_errors.ErrRequestTooLarge
	}
	return app_errors.ErrInvalidJSON
}

func requiredIdempotencyKey(
	c *gin.Context,
	operation string,
) (string, bool) {
	values := c.Request.Header.Values("Idempotency-Key")
	if len(values) == 0 || len(values) == 1 && values[0] == "" {
		writeServiceError(c, operation, app_errors.ErrIdempotencyKeyRequired)
		return "", false
	}
	if len(values) != 1 || validateIdempotencyKey(values[0]) != nil {
		writeServiceError(c, operation, app_errors.ErrInvalidIdempotencyKey)
		return "", false
	}
	return values[0], true
}

func writeSettingsResponse(c *gin.Context, settings SettingsResponse) {
	c.Header("Cache-Control", "no-store")
	c.Header("Pragma", "no-cache")
	c.Header("Content-Language", i18n.GetLanguageFromContext(c))
	c.Header("Vary", "Accept-Language")
	response.SuccessI18n(c, "common.success", settings)
}

func setSecretResponseHeaders(c *gin.Context) {
	c.Header("Cache-Control", "no-store")
	c.Header("Pragma", "no-cache")
}

func positiveIDParam(c *gin.Context, param, operation string) (uint, bool) {
	parsed, err := strconv.ParseUint(c.Param(param), 10, strconv.IntSize)
	if err != nil || parsed == 0 {
		writeServiceError(c, operation, app_errors.ErrBadRequest)
		return 0, false
	}
	return uint(parsed), true
}

func accessKeyID(c *gin.Context) (uint, bool) {
	return positiveIDParam(c, "id", "access_key_id")
}

func groupID(c *gin.Context, operation string) (uint, bool) {
	return positiveIDParam(c, "group_id", operation)
}

func credentialID(c *gin.Context, operation string) (uint, bool) {
	return positiveIDParam(c, "credential_id", operation)
}

func writeServiceError(c *gin.Context, operation string, err error) {
	writeServiceErrorResponse(c, operation, err)
}

func writeDiscoveryError(c *gin.Context, err error) {
	writeServiceErrorResponse(c, "discover_models", err)
}

func writeServiceErrorResponse(
	c *gin.Context,
	operation string,
	err error,
) {
	if requestWasCanceled(c.Request.Context(), err) {
		return
	}

	var apiErr *app_errors.APIError
	if errors.As(err, &apiErr) {
		setMutationErrorCode(c, apiErr.Code)
		if apiErr.HTTPStatus >= http.StatusInternalServerError {
			logServiceError(operation, err, apiErr.Code)
		}
		response.ErrorI18nFromAPIError(
			c,
			apiErr,
			serviceErrorMessageID(operation, err, apiErr),
		)
		return
	}

	setMutationErrorCode(c, app_errors.ErrInternalServer.Code)
	logServiceError(operation, err, app_errors.ErrInternalServer.Code)
	response.ErrorI18nFromAPIError(c, app_errors.ErrInternalServer, "internal_error")
}

func requestWasCanceled(ctx context.Context, err error) bool {
	return ctx != nil &&
		errors.Is(ctx.Err(), context.Canceled) &&
		errors.Is(err, context.Canceled)
}

func serviceErrorMessageID(
	operation string,
	err error,
	apiErr *app_errors.APIError,
) string {
	switch apiErr.Code {
	case app_errors.ErrIdempotencyKeyRequired.Code:
		return "idempotency.required"
	case app_errors.ErrInvalidIdempotencyKey.Code:
		return "idempotency.invalid"
	case app_errors.ErrIdempotencyKeyReused.Code:
		return "idempotency.reused"
	case app_errors.ErrIdempotencyResultExpired.Code:
		return "idempotency.expired"
	case app_errors.ErrControlOperationIncomplete.Code:
		return "control.operation_incomplete"
	case app_errors.ErrControlRecoveryPending.Code:
		return "control.recovery_pending"
	case app_errors.ErrModelPriceUnpricedConfirmationRequired.Code:
		return "model_price.unpriced_confirmation_required"
	case app_errors.ErrModelPriceReferenced.Code:
		return "model_price.referenced"
	case app_errors.ErrModelPriceAutomaticDeleteForbidden.Code:
		return "model_price.automatic_delete_forbidden"
	case modelRouteScheduleRevisionConflict.Code:
		return "bad_request"
	case app_errors.ErrRequestTooLarge.Code:
		return "request_too_large"
	case app_errors.ErrBadRequest.Code, app_errors.ErrInvalidJSON.Code, app_errors.ErrValidation.Code:
		return "bad_request"
	case app_errors.ErrResourceNotFound.Code:
		switch operation {
		case "update_model_price", "reset_model_price", "delete_model_price":
			return "model_price.not_found"
		}
		var resourceErr *controlResourceNotFoundError
		if errors.As(err, &resourceErr) {
			if resourceErr.resource == "group" {
				return "group.not_found"
			}
			return "credential.not_found"
		}
		switch operation {
		case "list_groups", "get_group_summary", "get_group_settings", "get_group_models",
			"update_group_settings", "delete_group",
			"update_group_models", "import_group_credentials",
			"discover_group_models", "list_group_credentials", "download_all_group_credentials":
			return "group.not_found"
		default:
			return "credential.not_found"
		}
	case app_errors.ErrNoActiveCredential.Code:
		return "group.no_active_credential"
	case app_errors.ErrDuplicateResource.Code:
		if operation == "create_group" || operation == "update_group_settings" {
			return "group.name_exists"
		}
		return "bad_request"
	case app_errors.ErrGroupInUse.Code:
		return "group.in_use"
	case app_errors.ErrChannelTargetConflict.Code:
		return "group.channel_target_conflict"
	case app_errors.ErrModelNameConflict.Code:
		return "group.model_name_conflict"
	case app_errors.ErrBadGateway.Code:
		return "bad_gateway"
	default:
		return "internal_error"
	}
}

func logServiceError(operation string, err error, code string) {
	fields := logrus.Fields{
		"operation":  operation,
		"error_code": code,
		"error_type": fmt.Sprintf("%T", err),
	}
	var operationErr *controlOperationError
	if errors.As(err, &operationErr) {
		if operationErr.stage != "" {
			fields["stage"] = operationErr.stage
		}
		if operationErr.mismatchKind != "" {
			fields["mismatch_kind"] = operationErr.mismatchKind
		}
		if operationErr.groupID != 0 {
			fields["group_id"] = operationErr.groupID
		}
		if operationErr.credentialID != 0 {
			fields["credential_id"] = operationErr.credentialID
		}
	}
	utils.LogPlaneBestEffort(
		logrus.StandardLogger(),
		logrus.ErrorLevel,
		utils.LogPlaneControl,
		fields,
		"Operation failed",
	)
}
