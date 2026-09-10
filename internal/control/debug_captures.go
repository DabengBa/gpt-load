package control

import (
	"context"
	"encoding/hex"
	"errors"
	"io"
	"net/url"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/debugcapture"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
)

const (
	defaultDebugCaptureLimit = 50
	maxDebugCaptureLimit     = 200
)

type DebugCaptureReader interface {
	QuerySessions(debugcapture.SessionQuery) ([]debugcapture.SessionRecord, error)
	ReadSession(string) (debugcapture.SessionRecord, error)
	ExportZIP(string, io.Writer) error
}

type DebugCaptureHealthReader interface {
	Health() (debugcapture.Health, error)
}

type debugCaptureListResponse struct {
	Items  []debugcapture.SessionRecord `json:"items"`
	Offset int                          `json:"offset"`
	Limit  int                          `json:"limit"`
	More   bool                         `json:"more"`
}

func (service *Service) SetDebugCaptureReader(reader DebugCaptureReader) {
	if service != nil {
		service.debugCaptures = reader
	}
}

func (service *Service) SetDebugCaptureHealthReader(reader DebugCaptureHealthReader) {
	if service != nil {
		service.debugCaptureHealth = reader
	}
}

func (service *Service) ListDebugCaptures(
	ctx context.Context,
	query debugcapture.SessionQuery,
) (debugCaptureListResponse, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return debugCaptureListResponse{}, err
		}
	}
	if service == nil || service.debugCaptures == nil {
		return debugCaptureListResponse{}, app_errors.ErrInternalServer
	}
	limit := query.Limit
	if limit <= 0 {
		limit = defaultDebugCaptureLimit
	}
	query.Limit = limit + 1
	items, err := service.debugCaptures.QuerySessions(query)
	if err != nil {
		return debugCaptureListResponse{}, mapDebugCaptureStoreError(err)
	}
	more := len(items) > limit
	if more {
		items = items[:limit]
	}
	if items == nil {
		items = []debugcapture.SessionRecord{}
	}
	return debugCaptureListResponse{
		Items: items, Offset: query.Offset, Limit: limit, More: more,
	}, nil
}

func (service *Service) ReadDebugCapture(ctx context.Context, id string) (debugcapture.SessionRecord, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return debugcapture.SessionRecord{}, err
		}
	}
	if service == nil || service.debugCaptures == nil {
		return debugcapture.SessionRecord{}, app_errors.ErrInternalServer
	}
	record, err := service.debugCaptures.ReadSession(id)
	if err != nil {
		return debugcapture.SessionRecord{}, mapDebugCaptureStoreError(err)
	}
	return record, nil
}

func (service *Service) ExportDebugCapture(ctx context.Context, id string, output io.Writer) error {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	if service == nil || service.debugCaptures == nil {
		return app_errors.ErrInternalServer
	}
	if err := service.debugCaptures.ExportZIP(id, output); err != nil {
		return mapDebugCaptureStoreError(err)
	}
	return nil
}

func mapDebugCaptureStoreError(err error) error {
	switch {
	case errors.Is(err, debugcapture.ErrNotFound), errors.Is(err, debugcapture.ErrExpired):
		return app_errors.ErrResourceNotFound
	case err == nil:
		return nil
	default:
		return app_errors.ErrInternalServer
	}
}

func parseDebugCaptureID(value string) (string, *app_errors.APIError) {
	if len(value) != 32 {
		return "", app_errors.ErrBadRequest
	}
	decoded, err := hex.DecodeString(value)
	if err != nil || len(decoded) != 16 || strings.ToLower(value) != value {
		return "", app_errors.ErrBadRequest
	}
	return value, nil
}

func parseDebugCaptureQuery(rawQuery string, forceQuery bool) (debugcapture.SessionQuery, *app_errors.APIError) {
	if forceQuery && rawQuery == "" {
		return debugcapture.SessionQuery{}, app_errors.ErrBadRequest
	}
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return debugcapture.SessionQuery{}, app_errors.ErrBadRequest
	}
	allowed := map[string]struct{}{
		"request_id": {}, "access_key_id": {}, "protocol": {}, "operation": {}, "state": {},
		"created_after_ms": {}, "created_before_ms": {}, "offset": {}, "limit": {},
	}
	for key, value := range values {
		if _, ok := allowed[key]; !ok || len(value) != 1 {
			return debugcapture.SessionQuery{}, app_errors.ErrBadRequest
		}
	}
	query := debugcapture.SessionQuery{Limit: defaultDebugCaptureLimit}
	if value, ok := singleQueryValue(values, "request_id"); ok {
		query.RequestID = value
	}
	if value, ok := singleQueryValue(values, "protocol"); ok {
		query.Protocol = value
	}
	if value, ok := singleQueryValue(values, "operation"); ok {
		query.Operation = value
	}
	if value, ok := singleQueryValue(values, "state"); ok {
		query.State = debugcapture.State(value)
		switch query.State {
		case debugcapture.StateActive, debugcapture.StateCompleted, debugcapture.StateFailed, debugcapture.StateExpired:
		default:
			return debugcapture.SessionQuery{}, app_errors.ErrBadRequest
		}
	}
	if value, ok := singleQueryValue(values, "access_key_id"); ok {
		parsed, err := parseCanonicalSafePlatformUint(value)
		if err != nil || parsed == 0 {
			return debugcapture.SessionQuery{}, app_errors.ErrBadRequest
		}
		query.AccessKeyID = parsed
	}
	if value, ok := singleQueryValue(values, "created_after_ms"); ok {
		parsed, err := parseCanonicalSafeMilliseconds(value)
		if err != nil {
			return debugcapture.SessionQuery{}, app_errors.ErrBadRequest
		}
		query.CreatedAfter = time.UnixMilli(parsed).UTC()
	}
	if value, ok := singleQueryValue(values, "created_before_ms"); ok {
		parsed, err := parseCanonicalSafeMilliseconds(value)
		if err != nil {
			return debugcapture.SessionQuery{}, app_errors.ErrBadRequest
		}
		query.CreatedBefore = time.UnixMilli(parsed).UTC()
	}
	if !query.CreatedAfter.IsZero() && !query.CreatedBefore.IsZero() &&
		!query.CreatedAfter.Before(query.CreatedBefore) {
		return debugcapture.SessionQuery{}, app_errors.ErrValidation
	}
	if value, ok := singleQueryValue(values, "offset"); ok {
		parsed, err := parseCanonicalSafePlatformUint(value)
		if err != nil || parsed > (^uint(0)>>1) {
			return debugcapture.SessionQuery{}, app_errors.ErrBadRequest
		}
		query.Offset = int(parsed)
	}
	if value, ok := singleQueryValue(values, "limit"); ok {
		parsed, err := parseCanonicalSafePlatformUint(value)
		if err != nil || parsed == 0 || parsed > maxDebugCaptureLimit {
			return debugcapture.SessionQuery{}, app_errors.ErrBadRequest
		}
		query.Limit = int(parsed)
	}
	return query, nil
}

func (server *Server) handleListDebugCaptures(c *gin.Context) {
	setSecretResponseHeaders(c)
	query, apiErr := parseDebugCaptureQuery(c.Request.URL.RawQuery, c.Request.URL.ForceQuery)
	if apiErr != nil {
		writeServiceError(c, "list_debug_captures", apiErr)
		return
	}
	result, err := server.service.ListDebugCaptures(c.Request.Context(), query)
	if err != nil {
		writeServiceError(c, "list_debug_captures", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}

func (server *Server) handleGetDebugCapture(c *gin.Context) {
	setSecretResponseHeaders(c)
	id, apiErr := parseDebugCaptureID(c.Param("capture_id"))
	if apiErr != nil {
		writeServiceError(c, "get_debug_capture", apiErr)
		return
	}
	result, err := server.service.ReadDebugCapture(c.Request.Context(), id)
	if err != nil {
		writeServiceError(c, "get_debug_capture", err)
		return
	}
	setSecretResponseHeaders(c)
	response.SuccessI18n(c, "common.success", result)
}

func (server *Server) handleDownloadDebugCapture(c *gin.Context) {
	setSecretResponseHeaders(c)
	id, apiErr := parseDebugCaptureID(c.Param("capture_id"))
	if apiErr != nil {
		writeServiceError(c, "download_debug_capture", apiErr)
		return
	}
	if _, err := server.service.ReadDebugCapture(c.Request.Context(), id); err != nil {
		writeServiceError(c, "download_debug_capture", err)
		return
	}
	c.Header("Content-Type", "application/zip")
	c.Header("Content-Disposition", `attachment; filename="debug-capture-`+id+`.zip"`)
	c.Header("Cache-Control", "no-store")
	if err := server.service.ExportDebugCapture(c.Request.Context(), id, c.Writer); err != nil {
		if !c.Writer.Written() {
			writeServiceError(c, "download_debug_capture", err)
			return
		}
		logServiceError("download_debug_capture", err, app_errors.ErrInternalServer.Code)
	}
}
