package control

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"

	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
	"gpt-load/internal/platform/utils"
)

func writeServiceError(c *gin.Context, operation string, err error) {
	writeServiceErrorResponse(c, operation, err)
}
func writeDiscoveryError(c *gin.Context, err error) {
	writeServiceErrorResponse(c, "discover_models", err)
}

func writeServiceErrorResponse(c *gin.Context, operation string, err error) {
	if requestWasCanceled(c.Request.Context(), err) {
		return
	}
	var apiErr *app_errors.APIError
	if errors.As(err, &apiErr) {
		setMutationErrorCode(c, apiErr.Code)
		if apiErr.HTTPStatus >= http.StatusInternalServerError {
			logServiceError(operation, err, apiErr.Code)
		}
		response.ErrorI18nFromAPIError(c, apiErr, serviceErrorMessageID(operation, err, apiErr))
		return
	}
	setMutationErrorCode(c, app_errors.ErrInternalServer.Code)
	logServiceError(operation, err, app_errors.ErrInternalServer.Code)
	response.ErrorI18nFromAPIError(c, app_errors.ErrInternalServer, "internal_error")
}

func requestWasCanceled(ctx context.Context, err error) bool {
	return ctx != nil && errors.Is(ctx.Err(), context.Canceled) && errors.Is(err, context.Canceled)
}

func serviceErrorMessageID(operation string, err error, apiErr *app_errors.APIError) string {
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
		case "list_groups", "get_group_summary", "get_group_settings", "get_group_models", "update_group_settings", "delete_group", "update_group_models", "import_group_credentials", "discover_group_models", "list_group_credentials", "download_all_group_credentials":
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
	fields := logrus.Fields{"operation": operation, "error_code": code, "error_type": fmt.Sprintf("%T", err)}
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
	utils.LogPlaneBestEffort(logrus.StandardLogger(), logrus.ErrorLevel, utils.LogPlaneControl, fields, "Operation failed")
}
