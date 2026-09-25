package gateway

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"gpt-load/internal/ratelimit"
	"gpt-load/internal/state"
)

type denyingAccessKeyRPMLimiter struct {
	retryAfter time.Duration
}

func (limiter denyingAccessKeyRPMLimiter) Allow(uint, int64) ratelimit.LimitDecision {
	return ratelimit.LimitDecision{Allowed: false, RetryAfter: limiter.retryAfter}
}

func newAdmissionRequest(
	method string,
	path string,
	body string,
) *http.Request {
	var reader *strings.Reader
	if body == "" {
		reader = strings.NewReader("")
	} else {
		reader = strings.NewReader(body)
	}
	request := httptest.NewRequest(method, path, reader)
	if body != "" {
		request.Header.Set("Content-Type", "application/json")
	}
	return request
}

func forwardAdmissionInput(request *http.Request) admissionInput {
	return admissionInput{
		request:   request,
		route:     route{Protocol: protocol.OpenAICompletions, Kind: endpointForward},
		snapshot:  &state.ConfigSnapshot{},
		accessKey: state.AccessKeyView{ID: 1},
	}
}

func TestAdmitRequestRateLimitedCarriesRetryAfter(t *testing.T) {
	t.Parallel()
	handler := &Handler{
		limiter:  denyingAccessKeyRPMLimiter{retryAfter: 2500 * time.Millisecond},
		dialects: dialect.NewSet(dialect.NewOpenAI()),
	}
	outcome := handler.admitRequest(context.Background(), forwardAdmissionInput(
		newAdmissionRequest(http.MethodPost, "/v1/chat/completions", `{"model":"gpt-4"}`),
	))
	if outcome.rejection == nil {
		t.Fatalf("expected rejection, got %+v", outcome)
	}
	if outcome.rejection.reason != reasonAccessKeyRateLimited {
		t.Fatalf("expected rate-limited reason, got %+v", outcome.rejection.reason)
	}
	if outcome.rejection.retryAfter == nil || *outcome.rejection.retryAfter != 3 {
		t.Fatalf("expected Retry-After of 3 seconds, got %v", outcome.rejection.retryAfter)
	}
}

func TestAdmitRequestModelsEndpointSkipsBodyDecode(t *testing.T) {
	t.Parallel()
	handler := &Handler{
		limiter:  unlimitedAccessKeyRPMLimiter{},
		dialects: dialect.NewSet(dialect.NewOpenAI()),
	}
	input := forwardAdmissionInput(
		newAdmissionRequest(http.MethodGet, "/v1/models", ""),
	)
	input.route = route{Protocol: protocol.OpenAICompletions, Kind: endpointModels}
	outcome := handler.admitRequest(context.Background(), input)
	if outcome.dispatch == nil || !outcome.dispatch.modelsOnly {
		t.Fatalf("expected models-only dispatch, got %+v", outcome)
	}
}

func TestAdmitRequestModelsEndpointRejectsNonIdentityEncoding(t *testing.T) {
	t.Parallel()
	handler := &Handler{
		limiter:  unlimitedAccessKeyRPMLimiter{},
		dialects: dialect.NewSet(dialect.NewOpenAI()),
	}
	request := newAdmissionRequest(http.MethodGet, "/v1/models", "")
	request.Header.Set("Accept-Encoding", "gzip")
	input := forwardAdmissionInput(request)
	input.route = route{Protocol: protocol.OpenAICompletions, Kind: endpointModels}
	outcome := handler.admitRequest(context.Background(), input)
	if outcome.rejection == nil || outcome.rejection.reason != reasonNotAcceptable {
		t.Fatalf("expected not-acceptable rejection, got %+v", outcome)
	}
}

func TestAdmitRequestRejectsUnknownDialect(t *testing.T) {
	t.Parallel()
	handler := &Handler{
		limiter:  unlimitedAccessKeyRPMLimiter{},
		dialects: dialect.Set{},
	}
	outcome := handler.admitRequest(context.Background(), forwardAdmissionInput(
		newAdmissionRequest(http.MethodPost, "/v1/chat/completions", `{"model":"gpt-4"}`),
	))
	if outcome.rejection == nil || outcome.rejection.reason != reasonEndpointNotFound {
		t.Fatalf("expected endpoint-not-found rejection, got %+v", outcome)
	}
}

func TestAdmitRequestRejectsUnsupportedContentEncoding(t *testing.T) {
	t.Parallel()
	handler := &Handler{
		limiter:  unlimitedAccessKeyRPMLimiter{},
		dialects: dialect.NewSet(dialect.NewOpenAI()),
	}
	request := newAdmissionRequest(http.MethodPost, "/v1/chat/completions", `{"model":"gpt-4"}`)
	request.Header.Set("Content-Encoding", "compress")
	outcome := handler.admitRequest(context.Background(), forwardAdmissionInput(request))
	if outcome.rejection == nil ||
		outcome.rejection.reason != reasonUnsupportedContentEncoding {
		t.Fatalf("expected unsupported-encoding rejection, got %+v", outcome)
	}
	if outcome.rejection.headers.Get("Accept-Encoding") == "" {
		t.Fatal("expected Accept-Encoding response header on rejection")
	}
}

func TestAdmitRequestDispatchCarriesForwardContext(t *testing.T) {
	t.Parallel()
	handler := &Handler{
		limiter:  unlimitedAccessKeyRPMLimiter{},
		dialects: dialect.NewSet(dialect.NewOpenAI()),
		registry: state.NewCredentialRegistry(),
	}
	snapshot := &state.ConfigSnapshot{
		ExecutionCandidates: state.ExecutionCandidateIndex{
			protocol.OpenAICompletions: {
				execution.OperationChatCompletion: {
					"gpt-4": {{GroupID: 1}},
				},
			},
		},
	}
	input := forwardAdmissionInput(newAdmissionRequest(
		http.MethodPost,
		"/v1/chat/completions",
		`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`,
	))
	input.snapshot = snapshot
	outcome := handler.admitRequest(context.Background(), input)
	if outcome.rejection != nil {
		t.Fatalf("unexpected rejection: %+v", outcome.rejection.reason)
	}
	dispatch := outcome.dispatch
	if dispatch == nil || dispatch.modelsOnly {
		t.Fatalf("expected forward dispatch, got %+v", outcome)
	}
	if dispatch.model != "gpt-4" {
		t.Fatalf("expected model gpt-4, got %q", dispatch.model)
	}
	if dispatch.query.ClientProtocol != protocol.OpenAICompletions ||
		dispatch.query.Operation != execution.OperationChatCompletion {
		t.Fatalf("unexpected scheduler query: %+v", dispatch.query)
	}
	if dispatch.dialect == nil || dispatch.parsed == nil {
		t.Fatal("dispatch must carry dialect and parsed request")
	}
	if dispatch.quotaAdmission == nil || dispatch.quotaAdmission.accessKeyID != 1 {
		t.Fatal("dispatch must carry the quota admission ticket")
	}
	if outcome.inspected == nil {
		t.Fatal("successful admission must expose inspected metadata")
	}
}

func TestAdmitRequestRejectsInvalidProtocolBody(t *testing.T) {
	t.Parallel()
	handler := &Handler{
		limiter:  unlimitedAccessKeyRPMLimiter{},
		dialects: dialect.NewSet(dialect.NewOpenAI()),
	}
	outcome := handler.admitRequest(context.Background(), forwardAdmissionInput(
		newAdmissionRequest(http.MethodPost, "/v1/chat/completions", `{"model":`),
	))
	if outcome.rejection == nil ||
		outcome.rejection.reason != reasonInvalidProtocolRequest {
		t.Fatalf("expected invalid-protocol rejection, got %+v", outcome)
	}
}

func TestAdmitRequestRejectsUnknownResponseBinding(t *testing.T) {
	t.Parallel()
	handler := &Handler{
		limiter:          unlimitedAccessKeyRPMLimiter{},
		dialects:         dialect.NewSet(dialect.NewOpenAIResponses()),
		registry:         state.NewCredentialRegistry(),
		responseBindings: state.NewResponseBindings(),
	}
	input := forwardAdmissionInput(newAdmissionRequest(
		http.MethodPost,
		"/v1/responses",
		`{"model":"gpt-4.1","previous_response_id":"resp_missing","input":"hi"}`,
	))
	input.route = route{Protocol: protocol.OpenAIResponses, Kind: endpointForward}
	outcome := handler.admitRequest(context.Background(), input)
	if outcome.rejection == nil ||
		outcome.rejection.reason != reasonResponseBindingNotFound {
		t.Fatalf("expected binding-not-found rejection, got %+v", outcome)
	}
}

func TestAdmitRequestCancelAfterInspectKeepsInspectedMetadata(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	handler := &Handler{
		limiter: unlimitedAccessKeyRPMLimiter{},
		dialects: dialect.NewSet(
			&cancelingSuccessfulInspectDialect{
				Dialect: dialect.NewOpenAI(),
				cancel:  cancel,
			},
		),
		registry: state.NewCredentialRegistry(),
	}
	outcome := handler.admitRequest(ctx, forwardAdmissionInput(newAdmissionRequest(
		http.MethodPost,
		"/v1/chat/completions",
		`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`,
	)))
	if !outcome.cancelled {
		t.Fatalf("expected cancellation, got %+v", outcome)
	}
	if outcome.inspected == nil {
		t.Fatal("post-inspect cancellation must still expose inspected metadata")
	}
}
