package control

import (
	"context"
	"net/url"
	"regexp"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/telemetry"
)

const (
	defaultRequestLogLimit = 50
	maxRequestLogLimit     = 200
	requestLogCursorV2     = 2
)

var canonicalLowercaseUUIDv4 = regexp.MustCompile(
	`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`,
)

var requestLogChannelRegistry = channel.NewRegistry()

type RequestLogReader interface {
	List(context.Context, requestlog.ListQuery) (requestlog.Page, error)
	Get(context.Context, string) (requestlog.Record, error)
}

func (service *Service) ListRequestLogs(
	ctx context.Context,
	query requestlog.ListQuery,
) (requestlog.Page, error) {
	if service.requestLogs == nil {
		return requestlog.Page{}, app_errors.ErrInternalServer
	}
	page, err := service.requestLogs.List(ctx, query)
	if err != nil {
		if requestWasCanceled(ctx, err) {
			return requestlog.Page{}, err
		}
		return requestlog.Page{}, app_errors.ParseDBError(err)
	}
	return page, nil
}

func (service *Service) GetRequestLog(ctx context.Context, requestID string) (requestlog.Record, error) {
	if service.requestLogs == nil {
		return requestlog.Record{}, app_errors.ErrInternalServer
	}
	record, err := service.requestLogs.Get(ctx, requestID)
	if err != nil {
		if requestWasCanceled(ctx, err) {
			return requestlog.Record{}, err
		}
		return requestlog.Record{}, app_errors.ParseDBError(err)
	}
	return record, nil
}

func (service *Service) GetAccessKeyRequestLog(
	ctx context.Context,
	requestID string,
	accessKeyID uint,
) (requestlog.Record, error) {
	if accessKeyID == 0 {
		return requestlog.Record{}, app_errors.ErrUnauthorized
	}
	page, err := service.ListRequestLogs(ctx, requestlog.ListQuery{
		AccessKeyID: &accessKeyID,
		RequestID:   requestID,
		Limit:       1,
	})
	if err != nil {
		return requestlog.Record{}, err
	}
	if len(page.Items) != 1 {
		return requestlog.Record{}, app_errors.ErrResourceNotFound
	}
	return sanitizeAccessKeyRequestLog(page.Items[0]), nil
}

func (s *Server) handleListRequestLogs(c *gin.Context) {
	accessKeyID, accessKeyScoped := currentAccessKeyID(c)
	if accessKeyScoped && requestLogQueryUsesInternalFields(c.Request.URL.RawQuery) {
		writeServiceError(c, "list_request_logs", app_errors.ErrBadRequest)
		return
	}
	query, apiErr := parseRequestLogQuery(c.Request.URL.RawQuery)
	if apiErr != nil {
		writeServiceError(c, "list_request_logs", apiErr)
		return
	}
	if accessKeyScoped {
		query.AccessKeyID = &accessKeyID
	}
	page, err := s.service.ListRequestLogs(c.Request.Context(), query)
	if err != nil {
		writeServiceError(c, "list_request_logs", err)
		return
	}
	if accessKeyScoped {
		for index := range page.Items {
			page.Items[index] = sanitizeAccessKeyRequestLog(page.Items[index])
		}
	}
	result, err := mapRequestLogListResponse(
		page,
		s.service.CredentialLabels(requestLogCredentialIDs(page.Items)),
	)
	if err != nil {
		writeServiceError(c, "list_request_logs", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleGetRequestLog(c *gin.Context) {
	requestID := c.Param("request_id")
	if !canonicalLowercaseUUIDv4.MatchString(requestID) {
		writeServiceError(c, "get_request_log", app_errors.ErrBadRequest)
		return
	}
	var record requestlog.Record
	var err error
	if accessKeyID, scoped := currentAccessKeyID(c); scoped {
		record, err = s.service.GetAccessKeyRequestLog(
			c.Request.Context(),
			requestID,
			accessKeyID,
		)
	} else {
		record, err = s.service.GetRequestLog(c.Request.Context(), requestID)
	}
	if err != nil {
		writeServiceError(c, "get_request_log", err)
		return
	}
	result, err := mapRequestLogDetailResponse(
		record,
		s.service.CredentialLabels(requestLogCredentialIDs([]requestlog.Record{record})),
	)
	if err != nil {
		writeServiceError(c, "get_request_log", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func requestLogQueryUsesInternalFields(rawQuery string) bool {
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return false
	}
	for _, key := range []string{
		"group_id",
		"channel_id",
		"credential_id",
		"upstream_model",
		"model_consistency",
		"access_key_id",
		"affinity_key",
		"attempt_status_code",
		"failure_category",
		"error_code",
		"retry_state",
		"retry_count_min",
		"retry_count_max",
	} {
		if _, exists := values[key]; exists {
			return true
		}
	}
	return false
}

func sanitizeAccessKeyRequestLog(record requestlog.Record) requestlog.Record {
	record.UpstreamModel = ""
	record.UpstreamReportedModel = ""
	record.ModelConsistency = telemetry.ModelConsistencyNotApplicable
	record.AttemptCount = 0
	record.AffinityHit = false
	record.ResolvedAffinityCredentialID = 0
	record.FirstAttemptCredentialID = 0
	record.AffinityHitAssignmentReached = false
	record.ContinuityHit = false
	record.AffinityKey = ""
	record.AffinitySource = requestlog.AffinitySourceNone
	record.AffinityState = requestlog.AffinityStateNoSignal
	record.GroupID = 0
	record.ChannelID = ""
	record.CredentialID = 0
	record.RouteMode = ""
	record.UpstreamProtocol = ""
	record.Attempts = []requestlog.Attempt{}
	return record
}
