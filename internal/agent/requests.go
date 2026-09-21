package agent

import (
	"context"
	"errors"
	"net/url"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/debugcapture"
	"gpt-load/internal/execution"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
	"gpt-load/internal/protocol"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/telemetry"
)

var agentChannelRegistry = channel.NewRegistry()

func (server *Server) handleListRequests(c *gin.Context) {
	if !requireScope(c, ScopeDiagnosticsRead) {
		return
	}
	query, apiErr := parseRequestListQuery(c.Request.URL.RawQuery)
	if apiErr != nil {
		writeAgentServiceError(c, "agent_list_requests", apiErr)
		return
	}
	page, err := server.requestLogs.List(c.Request.Context(), query)
	if err != nil {
		writeAgentServiceError(c, "agent_list_requests", mapAgentReadError(err))
		return
	}
	items := make([]RequestSummary, 0, len(page.Items))
	for _, record := range page.Items {
		summary, err := projectRequestSummary(record)
		if err != nil {
			writeAgentServiceError(c, "agent_list_requests", app_errors.ErrInternalServer)
			return
		}
		summary.ErrorSummary = server.scrub(record.ErrorSummary)
		items = append(items, summary)
	}
	var nextCursor *string
	if page.NextCursor != nil {
		encoded, err := encodeRequestCursor(*page.NextCursor)
		if err != nil {
			writeAgentServiceError(c, "agent_list_requests", app_errors.ErrInternalServer)
			return
		}
		nextCursor = &encoded
	}
	response.SuccessI18n(c, "common.success", RequestList{
		SchemaVersion: SchemaVersion,
		Items:         items,
		NextCursor:    nextCursor,
		Page: PageInfo{
			Limit:    query.Limit,
			Returned: len(items),
			HasMore:  page.NextCursor != nil,
		},
	})
}

func (server *Server) handleGetRequest(c *gin.Context) {
	if !requireScope(c, ScopeDiagnosticsRead) {
		return
	}
	requestID := c.Param("request_id")
	if !canonicalLowercaseUUIDv4.MatchString(requestID) {
		writeAgentServiceError(c, "agent_get_request", app_errors.ErrBadRequest)
		return
	}
	record, err := server.requestLogs.Get(c.Request.Context(), requestID)
	if err != nil {
		writeAgentServiceError(c, "agent_get_request", mapAgentReadError(err))
		return
	}
	if record.RequestID == "" {
		writeAgentServiceError(c, "agent_get_request", app_errors.ErrResourceNotFound)
		return
	}
	summary, err := projectRequestSummary(record)
	if err != nil {
		writeAgentServiceError(c, "agent_get_request", app_errors.ErrInternalServer)
		return
	}
	summary.ErrorSummary = server.scrub(record.ErrorSummary)
	attempts := make([]AttemptView, 0, len(record.Attempts))
	for _, attempt := range record.Attempts {
		attempts = append(attempts, projectAttempt(record.RequestID, attempt, server.scrub))
	}
	evidence, err := server.buildEvidence(c.Request.Context(), record.RequestID, record.Attempts)
	if err != nil {
		writeAgentServiceError(c, "agent_get_request", err)
		return
	}
	response.SuccessI18n(c, "common.success", RequestDetail{
		SchemaVersion: SchemaVersion,
		Request:       summary,
		Attempts:      attempts,
		Evidence: EvidenceLink{
			State:       evidence.State,
			CaptureRefs: captureIDs(evidence),
			Path:        "/api/agent/v1/requests/" + record.RequestID + "/evidence",
		},
	})
}

func (server *Server) handleGetEvidence(c *gin.Context) {
	if !requireScope(c, ScopeDiagnosticsRead) {
		return
	}
	requestID := c.Param("request_id")
	if !canonicalLowercaseUUIDv4.MatchString(requestID) {
		writeAgentServiceError(c, "agent_get_evidence", app_errors.ErrBadRequest)
		return
	}
	if server.requestLogs == nil {
		writeAgentServiceError(c, "agent_get_evidence", app_errors.ErrInternalServer)
		return
	}
	record, err := server.requestLogs.Get(c.Request.Context(), requestID)
	if err != nil {
		writeAgentServiceError(c, "agent_get_evidence", mapAgentReadError(err))
		return
	}
	if record.RequestID == "" {
		writeAgentServiceError(c, "agent_get_evidence", app_errors.ErrResourceNotFound)
		return
	}
	evidence, err := server.buildEvidence(c.Request.Context(), requestID, record.Attempts)
	if err != nil {
		writeAgentServiceError(c, "agent_get_evidence", err)
		return
	}
	response.SuccessI18n(c, "common.success", evidence)
}

func captureIDs(evidence EvidenceView) []string {
	result := make([]string, 0, len(evidence.Captures))
	for _, capture := range evidence.Captures {
		result = append(result, capture.CaptureID)
	}
	return result
}

// buildEvidence returns redacted evidence metadata for one request. It never
// returns raw headers or bodies, and it distinguishes "no retained capture"
// from "capture unavailable". It never calls a store method that can write.
func (server *Server) buildEvidence(
	ctx context.Context,
	requestID string,
	attempts []requestlog.Attempt,
) (EvidenceView, error) {
	view := unavailableEvidence(EvidenceStateUnavailable)
	view.RequestID = requestID
	var health debugcapture.Health
	healthKnown := false
	if server.evidenceHealth != nil {
		observed, err := server.evidenceHealth.Health()
		if err == nil {
			health = observed
			healthKnown = true
		}
	}
	view.CapturesEnabled = healthKnown && health.Enabled
	view.RetentionSeconds = health.RetentionSeconds
	if !healthKnown || health.RetentionSeconds == 0 {
		view.RetentionSeconds = int64(debugcapture.RetentionPeriod().Seconds())
	}
	if server.evidence == nil || !view.CapturesEnabled {
		return view, nil
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return view, err
		}
	}
	sessions, err := server.evidence.QuerySessions(debugcapture.SessionQuery{
		RequestID: requestID,
		Limit:     -1,
	})
	if err != nil {
		// A store failure is reported as unavailable, never as "no capture".
		view.State = EvidenceStateUnavailable
		return view, nil
	}
	if len(sessions) == 0 {
		view.State = EvidenceStateNotRetained
		return view, nil
	}
	view.State = EvidenceStateCaptured
	captures := make([]CaptureView, 0, len(sessions))
	validAttemptSequences := make(map[int]struct{}, len(attempts))
	for _, attempt := range attempts {
		if attempt.Sequence > 0 {
			validAttemptSequences[attempt.Sequence] = struct{}{}
		}
	}
	for _, session := range sessions {
		captures = append(captures, projectCapture(requestID, session, validAttemptSequences))
	}
	view.Captures = captures
	return view, nil
}

func mapAgentReadError(err error) error {
	if err == nil {
		return nil
	}
	var apiErr *app_errors.APIError
	if errors.As(err, &apiErr) {
		return err
	}
	return app_errors.ParseDBError(err)
}

func parseRequestListQuery(rawQuery string) (requestlog.ListQuery, *app_errors.APIError) {
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return requestlog.ListQuery{}, app_errors.ErrBadRequest
	}
	allowed := map[string]struct{}{
		"from_ms": {}, "to_ms": {}, "group_id": {}, "channel_id": {},
		"credential_id": {}, "access_key_id": {}, "client_model": {},
		"upstream_model": {}, "status": {}, "protocol": {}, "operation": {},
		"stream": {}, "request_id": {}, "limit": {}, "cursor": {},
	}
	for key, value := range values {
		if _, ok := allowed[key]; !ok || len(value) != 1 {
			return requestlog.ListQuery{}, app_errors.ErrBadRequest
		}
	}
	query := requestlog.ListQuery{Limit: DefaultPageLimit}
	if value, ok := agentQueryValue(values, "from_ms"); ok {
		parsed, err := parseCanonicalSafeMilliseconds(value)
		if err != nil {
			return requestlog.ListQuery{}, app_errors.ErrBadRequest
		}
		query.FromMS = &parsed
	}
	if value, ok := agentQueryValue(values, "to_ms"); ok {
		parsed, err := parseCanonicalSafeMilliseconds(value)
		if err != nil {
			return requestlog.ListQuery{}, app_errors.ErrBadRequest
		}
		query.ToMS = &parsed
	}
	if query.FromMS != nil && query.ToMS != nil && *query.FromMS >= *query.ToMS {
		return requestlog.ListQuery{}, app_errors.ErrValidation
	}
	if value, ok := agentQueryValue(values, "group_id"); ok {
		parsed, apiErr := parseAgentScopedID(value)
		if apiErr != nil {
			return requestlog.ListQuery{}, apiErr
		}
		query.GroupID = &parsed
	}
	if value, ok := agentQueryValue(values, "channel_id"); ok {
		channelID := channel.ID(value)
		if _, known := agentChannelRegistry.Get(channelID); !known {
			return requestlog.ListQuery{}, app_errors.ErrValidation
		}
		query.ChannelID = channelID
	}
	if value, ok := agentQueryValue(values, "credential_id"); ok {
		parsed, apiErr := parseAgentScopedID(value)
		if apiErr != nil {
			return requestlog.ListQuery{}, apiErr
		}
		query.CredentialID = &parsed
	}
	if value, ok := agentQueryValue(values, "access_key_id"); ok {
		parsed, apiErr := parseAgentScopedID(value)
		if apiErr != nil {
			return requestlog.ListQuery{}, apiErr
		}
		query.AccessKeyID = &parsed
	}
	if value, ok := agentQueryValue(values, "client_model"); ok {
		if !validAgentModel(value) {
			return requestlog.ListQuery{}, app_errors.ErrValidation
		}
		query.ClientModel = value
	}
	if value, ok := agentQueryValue(values, "upstream_model"); ok {
		if !validAgentModel(value) {
			return requestlog.ListQuery{}, app_errors.ErrValidation
		}
		query.UpstreamModel = value
	}
	if value, ok := agentQueryValue(values, "status"); ok {
		status := telemetry.RequestStatus(value)
		switch status {
		case telemetry.RequestStatusSuccess, telemetry.RequestStatusError,
			telemetry.RequestStatusIncomplete, telemetry.RequestStatusCanceled:
			query.Status = status
		default:
			return requestlog.ListQuery{}, app_errors.ErrValidation
		}
	}
	if value, ok := agentQueryValue(values, "protocol"); ok {
		parsed := protocol.Protocol(value)
		if !parsed.Valid() {
			return requestlog.ListQuery{}, app_errors.ErrValidation
		}
		query.Protocol = parsed
	}
	if value, ok := agentQueryValue(values, "operation"); ok {
		parsed := execution.Operation(value)
		if !parsed.Valid() {
			return requestlog.ListQuery{}, app_errors.ErrValidation
		}
		query.Operation = parsed
	}
	if value, ok := agentQueryValue(values, "stream"); ok {
		switch value {
		case "true":
			parsed := true
			query.Stream = &parsed
		case "false":
			parsed := false
			query.Stream = &parsed
		default:
			return requestlog.ListQuery{}, app_errors.ErrValidation
		}
	}
	if value, ok := agentQueryValue(values, "request_id"); ok {
		if !canonicalLowercaseUUIDv4.MatchString(value) {
			return requestlog.ListQuery{}, app_errors.ErrBadRequest
		}
		query.RequestID = value
	}
	if value, ok := agentQueryValue(values, "limit"); ok {
		parsed, err := parseCanonicalSafeUint(value)
		if err != nil {
			return requestlog.ListQuery{}, app_errors.ErrBadRequest
		}
		if parsed < 1 || parsed > MaxPageLimit {
			return requestlog.ListQuery{}, app_errors.ErrValidation
		}
		query.Limit = int(parsed)
	}
	if value, ok := agentQueryValue(values, "cursor"); ok {
		cursor, err := decodeRequestCursor(value)
		if err != nil {
			return requestlog.ListQuery{}, app_errors.ErrBadRequest
		}
		query.Cursor = cursor
	}
	return query, nil
}

func agentQueryValue(values url.Values, key string) (string, bool) {
	value, ok := values[key]
	if !ok {
		return "", false
	}
	return value[0], true
}

func parseAgentScopedID(value string) (uint, *app_errors.APIError) {
	parsed, err := parseCanonicalSafePlatformUint(value)
	if err != nil {
		if errors.Is(err, errUnsafeCanonicalUint) {
			return 0, app_errors.ErrValidation
		}
		return 0, app_errors.ErrBadRequest
	}
	if parsed == 0 {
		return 0, app_errors.ErrValidation
	}
	return parsed, nil
}

func validAgentModel(value string) bool {
	if !utf8.ValidString(value) || len(value) < 1 || len(value) > 255 ||
		strings.TrimSpace(value) != value {
		return false
	}
	for _, character := range value {
		if unicode.IsControl(character) {
			return false
		}
	}
	return true
}
