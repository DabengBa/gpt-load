package agent

import (
	"context"
	"errors"
	"fmt"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"

	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
	"gpt-load/internal/platform/utils"
)

func writeAgentServiceError(c *gin.Context, operation string, err error) {
	if requestWasCanceled(c.Request.Context(), err) {
		return
	}
	var apiErr *app_errors.APIError
	if errors.As(err, &apiErr) {
		if apiErr.HTTPStatus >= 500 {
			logAgentServiceError(operation, err, apiErr.Code)
		}
		switch apiErr.Code {
		case app_errors.ErrUnauthorized.Code:
			response.ErrorI18nFromAPIError(c, apiErr, "auth.invalid_key")
		case app_errors.ErrForbidden.Code:
			response.ErrorI18nFromAPIError(c, apiErr, "auth.forbidden")
		case app_errors.ErrRequestTooLarge.Code:
			response.ErrorI18nFromAPIError(c, apiErr, "request_too_large")
		case app_errors.ErrResourceNotFound.Code:
			// The Agent surface is machine-facing; the stable code is
			// authoritative and there is no generic localized 404 message.
			response.Error(c, apiErr)
		case app_errors.ErrBadRequest.Code,
			app_errors.ErrInvalidJSON.Code,
			app_errors.ErrValidation.Code:
			response.ErrorI18nFromAPIError(c, apiErr, "bad_request")
		case app_errors.ErrIdempotencyKeyRequired.Code,
			app_errors.ErrInvalidIdempotencyKey.Code,
			app_errors.ErrIdempotencyKeyReused.Code,
			app_errors.ErrIdempotencyResultExpired.Code,
			app_errors.ErrControlOperationIncomplete.Code,
			app_errors.ErrControlRecoveryPending.Code,
			"CHANGE_PROPOSAL_NOT_APPROVED",
			"CHANGE_PROPOSAL_REVOKED",
			"CHANGE_PROPOSAL_RUNTIME_CONFLICT",
			"CHANGE_PROPOSAL_REVISION_CONFLICT",
			"CHANGE_PROPOSAL_VALUE_CONFLICT",
			"CHANGE_PROPOSAL_STATE_CONFLICT":
			response.Error(c, apiErr)
		default:
			response.ErrorI18nFromAPIError(c, apiErr, "internal_error")
		}
		return
	}
	logAgentServiceError(operation, err, app_errors.ErrInternalServer.Code)
	response.ErrorI18nFromAPIError(c, app_errors.ErrInternalServer, "internal_error")
}

func requestWasCanceled(ctx context.Context, err error) bool {
	return ctx != nil &&
		errors.Is(ctx.Err(), context.Canceled) &&
		errors.Is(err, context.Canceled)
}

func logAgentServiceError(operation string, err error, code string) {
	utils.LogPlaneBestEffort(
		logrus.StandardLogger(),
		logrus.ErrorLevel,
		utils.LogPlaneControl,
		logrus.Fields{
			"operation":  operation,
			"error_code": code,
			"error_type": fmt.Sprintf("%T", err),
		},
		"Agent operation failed",
	)
}
