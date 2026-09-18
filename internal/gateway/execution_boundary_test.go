package gateway

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	subscriptionproviders "gpt-load/internal/subscription/providers"
	subscriptionruntime "gpt-load/internal/subscription/runtime"
)

func TestNormalizeChannelCredentialRequiresCanonicalStoredObject(t *testing.T) {
	t.Parallel()
	channels, subscriptions := testCredentialRuntimes(t)

	credential, err := normalizeChannelCredential(
		channels,
		subscriptions,
		channel.OpenAI,
		"api_key",
		` {"api_key":"sk-typed"} `,
	)
	if err != nil {
		t.Fatalf("normalizeChannelCredential() error = %v", err)
	}
	if credential.apiKey != "sk-typed" || string(credential.payload) != `{"api_key":"sk-typed"}` {
		t.Fatalf("normalizeChannelCredential() = %q %s", credential.apiKey, credential.payload)
	}

	for _, invalid := range []string{"", " ", "sk-legacy", `"sk-legacy"`, `[]`, `{}`, `{"api_key":""}`} {
		if credential, err := normalizeChannelCredential(channels, subscriptions, channel.OpenAI, "api_key", invalid); err == nil {
			t.Fatalf("normalizeChannelCredential(%q) = %#v, nil", invalid, credential)
		}
	}
}

func TestCodexSearchErrorsDoNotAffectModelHealth(t *testing.T) {
	for _, status := range []int{http.StatusNotFound, http.StatusTooManyRequests, http.StatusBadGateway} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			evidence := &execution.ErrorEvidence{Kind: execution.ErrorKindHTTP, StatusCode: status, Summary: "search unavailable"}
			if status == http.StatusTooManyRequests {
				evidence.Hint = execution.FailureHintRateLimited
				evidence.ScopeHint = execution.ErrorScopeModel
			}
			decision := judgeUpstreamResult(UpstreamResult{
				DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: status, ExecutionError: evidence,
			}, time.Now(), health.DecisionContext{Method: http.MethodPost, Operation: execution.OperationWebSearch})
			wantEffect := health.EffectNone
			if status == http.StatusBadGateway {
				wantEffect = health.EffectSkipGroup
			}
			if decision.Effect != wantEffect || !decision.CooldownUntil.IsZero() {
				t.Fatalf("search must not penalize model health: %#v", decision)
			}
		})
	}
}

func TestNormalizeChannelCredentialPreservesStructuredCloudSecrets(t *testing.T) {
	t.Parallel()
	channels, subscriptions := testCredentialRuntimes(t)

	got, err := normalizeChannelCredential(
		channels,
		subscriptions,
		channel.AWSBedrock,
		"api_key",
		` {"access_key":"AKIA_TEST","secret_key":"bedrock-secret","session_token":"bedrock-session"} `,
	)
	if err != nil {
		t.Fatalf("normalizeChannelCredential() error = %v", err)
	}
	if got.apiKey != "" || string(got.payload) !=
		`{"access_key":"AKIA_TEST","secret_key":"bedrock-secret","session_token":"bedrock-session"}` {
		t.Fatalf("normalized credential = %#v / %s", got, got.payload)
	}
	if want := []string{"AKIA_TEST", "bedrock-secret", "bedrock-session"}; !reflect.DeepEqual(got.secrets, want) {
		t.Fatalf("secrets = %#v, want %#v", got.secrets, want)
	}
}

func TestNormalizeChannelCredentialUsesBoundSubscriptionDriver(t *testing.T) {
	channels, subscriptions := testCredentialRuntimes(t)
	got, err := normalizeChannelCredential(
		channels,
		subscriptions,
		channel.Codex,
		"subscription",
		`{"type":"codex","access_token":"access-secret","refresh_token":"refresh-secret","account_id":"account-one"}`,
	)
	if err != nil {
		t.Fatal(err)
	}
	if got.apiKey != "" || len(got.payload) == 0 || !reflect.DeepEqual(got.secrets[:2], []string{"access-secret", "refresh-secret"}) {
		t.Fatalf("normalized subscription credential = %#v", got)
	}
}

func testCredentialRuntimes(t *testing.T) (*channel.Registry, *subscriptionruntime.Runtime) {
	t.Helper()
	channels := channel.NewRegistry()
	subscriptions, err := subscriptionruntime.NewRuntime(channels, subscriptionproviders.Implementations()...)
	if err != nil {
		t.Fatal(err)
	}
	return channels, subscriptions
}

func TestDecisionEvidenceCanonicalizesOnlyBufferedUpstreamStreamFailures(t *testing.T) {
	newResult := func(reason StreamEndReason, evidence *execution.ErrorEvidence) UpstreamResult {
		return UpstreamResult{
			DispatchState:      execution.DispatchMaybeSent,
			StatusCode:         http.StatusOK,
			BufferedStream:     true,
			HTTPCommitted:      true,
			ClientVisibleBytes: int64(len(bufferedStreamHeartbeat)),
			ExecutionError:     evidence,
			Stream:             StreamObservation{EndReason: reason},
		}
	}
	newEvidence := func(kind execution.ErrorKind, origin execution.ErrorOrigin, hint execution.FailureHint, code string) *execution.ErrorEvidence {
		return &execution.ErrorEvidence{
			Kind: kind, OriginHint: origin, ScopeHint: execution.ErrorScopeRequest,
			Hint: hint, StatusCode: http.StatusOK, Code: code,
			Summary: "provider stream failed", ReplaySafety: execution.ReplaySafetyUnknown,
			Header: http.Header{"X-Evidence": {"kept"}},
		}
	}

	tests := []struct {
		name            string
		result          func(*execution.ErrorEvidence) UpstreamResult
		wantCode        string
		wantSamePointer bool
	}{
		{
			name:     "SSE error unknown code is canonicalized on a clone",
			result:   func(evidence *execution.ErrorEvidence) UpstreamResult { return newResult(StreamEndSSEError, evidence) },
			wantCode: "upstream_sse_error",
		},
		{
			name: "upstream failure unknown code is canonicalized on a clone",
			result: func(evidence *execution.ErrorEvidence) UpstreamResult {
				return newResult(StreamEndUpstreamFailure, evidence)
			},
			wantCode: "upstream_sse_error",
		},
		{
			name: "empty evidence is created only for the upstream stream reason",
			result: func(_ *execution.ErrorEvidence) UpstreamResult {
				return newResult(StreamEndUpstreamFailure, nil)
			},
			wantCode: "upstream_sse_error",
		},
		{
			name: "empty origin provider evidence is canonicalized",
			result: func(_ *execution.ErrorEvidence) UpstreamResult {
				return newResult(StreamEndSSEError, newEvidence(
					execution.ErrorKindProvider, "", "", "internal_error",
				))
			},
			wantCode: "upstream_sse_error",
		},
		{
			name: "client origin provider evidence is unchanged",
			result: func(_ *execution.ErrorEvidence) UpstreamResult {
				return newResult(StreamEndSSEError, newEvidence(
					execution.ErrorKindProvider, execution.ErrorOriginClient, "", "internal_error",
				))
			},
			wantCode:        "internal_error",
			wantSamePointer: true,
		},
		{
			name: "server overload preserves the health capacity code",
			result: func(_ *execution.ErrorEvidence) UpstreamResult {
				evidence := newEvidence(
					execution.ErrorKindProvider, execution.ErrorOriginUpstream, "", "server_is_overloaded",
				)
				evidence.ScopeHint = execution.ErrorScopeGroup
				evidence.ReplaySafety = execution.ReplaySafetyRejectedBeforeProcessing
				return newResult(StreamEndSSEError, evidence)
			},
			wantCode: "server_is_overloaded",
		},
		{
			name: "server overload keeps the replay unsafe fallback",
			result: func(_ *execution.ErrorEvidence) UpstreamResult {
				evidence := newEvidence(
					execution.ErrorKindProvider, execution.ErrorOriginUpstream, "", "server_is_overloaded",
				)
				evidence.StatusCode = http.StatusServiceUnavailable
				evidence.ScopeHint = execution.ErrorScopeGroup
				evidence.ReplaySafety = execution.ReplaySafetyUnknown
				result := newResult(StreamEndSSEError, evidence)
				result.StatusCode = http.StatusServiceUnavailable
				return result
			},
			wantCode: "server_is_overloaded",
		},
		{
			name: "non buffered stream is unchanged",
			result: func(evidence *execution.ErrorEvidence) UpstreamResult {
				result := newResult(StreamEndSSEError, evidence)
				result.BufferedStream = false
				return result
			},
			wantCode:        "internal_error",
			wantSamePointer: true,
		},
		{
			name: "released payload is unchanged",
			result: func(evidence *execution.ErrorEvidence) UpstreamResult {
				result := newResult(StreamEndSSEError, evidence)
				result.PayloadReleased = true
				return result
			},
			wantCode:        "internal_error",
			wantSamePointer: true,
		},
		{
			name: "uncommitted heartbeat is unchanged",
			result: func(evidence *execution.ErrorEvidence) UpstreamResult {
				result := newResult(StreamEndSSEError, evidence)
				result.HTTPCommitted = false
				return result
			},
			wantCode:        "internal_error",
			wantSamePointer: true,
		},
		{
			name: "no visible bytes is unchanged",
			result: func(evidence *execution.ErrorEvidence) UpstreamResult {
				result := newResult(StreamEndSSEError, evidence)
				result.ClientVisibleBytes = 0
				return result
			},
			wantCode:        "internal_error",
			wantSamePointer: true,
		},
		{
			name: "internal evidence is unchanged",
			result: func(_ *execution.ErrorEvidence) UpstreamResult {
				return newResult(StreamEndSSEError, newEvidence(
					execution.ErrorKindInternal, execution.ErrorOriginInternal, "", "internal_error",
				))
			},
			wantCode: "internal_error",
		},
		{
			name: "downstream evidence is unchanged",
			result: func(_ *execution.ErrorEvidence) UpstreamResult {
				return newResult(StreamEndSSEError, newEvidence(
					execution.ErrorKindProvider, execution.ErrorOriginDownstream, "", "internal_error",
				))
			},
			wantCode: "internal_error",
		},
		{
			name: "canceled evidence is unchanged",
			result: func(_ *execution.ErrorEvidence) UpstreamResult {
				return newResult(StreamEndSSEError, newEvidence(
					execution.ErrorKindCanceled, execution.ErrorOriginDownstream, "", "internal_error",
				))
			},
			wantCode: "internal_error",
		},
		{
			name: "canceled result error is unchanged",
			result: func(evidence *execution.ErrorEvidence) UpstreamResult {
				result := newResult(StreamEndSSEError, evidence)
				result.Err = context.Canceled
				return result
			},
			wantCode:        "internal_error",
			wantSamePointer: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			evidence := newEvidence(execution.ErrorKindProvider, execution.ErrorOriginUpstream, "", "internal_error")
			result := test.result(evidence)
			var original *execution.ErrorEvidence
			if result.ExecutionError != nil {
				copy := result.ExecutionError.Clone()
				original = &copy
			}
			got, err := decisionEvidence(result)
			if err != nil {
				t.Fatalf("decisionEvidence() error = %v", err)
			}
			if got == nil || got.Code != test.wantCode {
				t.Fatalf("decisionEvidence() = %#v, want code %q", got, test.wantCode)
			}
			if test.wantSamePointer && got != result.ExecutionError {
				t.Fatalf("decisionEvidence() returned a normalized copy for an ineligible result")
			}
			if result.ExecutionError != nil && original != nil && !reflect.DeepEqual(*result.ExecutionError, *original) {
				t.Fatalf("decisionEvidence() mutated original evidence: got %#v, want %#v", *result.ExecutionError, *original)
			}
			if test.wantCode == "upstream_sse_error" {
				decision := judgeUpstreamResult(result, timeNowForBufferedTest(), health.DecisionContext{
					Method: http.MethodPost, Operation: execution.OperationResponsesCreate,
					BufferedReplayEligible: true,
				})
				if decision.Retry != health.RetryNextCandidate {
					t.Fatalf("buffered failure decision = %#v, want candidate retry", decision)
				}
			}
		})
	}
}

func TestDecisionEvidencePreservesServerOverloadedHealthDecisions(t *testing.T) {
	newResult := func(status int, replaySafety execution.ReplaySafety) UpstreamResult {
		evidence := &execution.ErrorEvidence{
			Kind: execution.ErrorKindProvider, OriginHint: execution.ErrorOriginUpstream,
			ScopeHint: execution.ErrorScopeGroup, StatusCode: status,
			Code: "server_is_overloaded", Summary: "provider is overloaded",
			ReplaySafety: replaySafety,
		}
		return UpstreamResult{
			DispatchState: execution.DispatchMaybeSent, StatusCode: status,
			BufferedStream: true, HTTPCommitted: true,
			ClientVisibleBytes: int64(len(bufferedStreamHeartbeat)),
			ExecutionError:     evidence,
			Stream:             StreamObservation{EndReason: StreamEndSSEError},
		}
	}

	tests := []struct {
		name           string
		status         int
		replaySafety   execution.ReplaySafety
		buffered       bool
		replayEligible bool
		want           health.Decision
	}{
		{
			name:           "replay eligible uses transient capacity decision",
			status:         http.StatusOK,
			replaySafety:   execution.ReplaySafetyRejectedBeforeProcessing,
			buffered:       true,
			replayEligible: true,
			want: health.Decision{
				Category: health.FailureCategoryUpstreamHostError,
				Origin:   execution.ErrorOriginUpstream, Scope: execution.ErrorScopeGroup,
				Retry: health.RetryNextCandidate, Effect: health.EffectNone,
				RuleID: "candidate.transient_capacity",
			},
		},
		{
			name:           "replay ineligible keeps safe fallback",
			status:         http.StatusServiceUnavailable,
			replaySafety:   execution.ReplaySafetyUnknown,
			buffered:       true,
			replayEligible: false,
			want: health.Decision{
				Category: health.FailureCategoryUpstreamHostError,
				Origin:   execution.ErrorOriginUpstream, Scope: execution.ErrorScopeGroup,
				Retry: health.RetryNone, Effect: health.EffectSkipGroup,
				RuleID: "upstream.host_error.replay_unsafe",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := newResult(test.status, test.replaySafety)
			result.BufferedStream = test.buffered
			evidence, err := decisionEvidence(result)
			if err != nil {
				t.Fatalf("decisionEvidence() error = %v", err)
			}
			if evidence == nil || evidence.Code != "server_is_overloaded" {
				t.Fatalf("decisionEvidence() = %#v, want preserved capacity code", evidence)
			}
			result.ExecutionError = evidence
			decision := judgeUpstreamResult(result, timeNowForBufferedTest(), health.DecisionContext{
				Method: http.MethodPost, Operation: execution.OperationResponsesCreate,
				BufferedReplayEligible: test.replayEligible,
			})
			if decision.Category != test.want.Category || decision.Origin != test.want.Origin ||
				decision.Scope != test.want.Scope || decision.Retry != test.want.Retry ||
				decision.Effect != test.want.Effect || decision.RuleID != test.want.RuleID {
				t.Fatalf("judgeUpstreamResult() = %#v, want %#v", decision, test.want)
			}
		})
	}
}

func TestJudgeUpstreamResultUsesNeutralExecutionEvidence(t *testing.T) {
	t.Parallel()

	now := time.Unix(1_800_000_000, 0)
	result := judgeUpstreamResult(UpstreamResult{
		DispatchState:   execution.DispatchMaybeSent,
		ResponseStarted: true,
		StatusCode:      http.StatusTooManyRequests,
		ExecutionError: &execution.ErrorEvidence{
			Kind: execution.ErrorKindHTTP, StatusCode: http.StatusTooManyRequests,
			Summary: "upstream rejected request", RetryAfter: 7 * time.Second,
		},
	}, now, health.DecisionContext{DefaultRateLimitCooldown: time.Minute})
	if result.Category != health.FailureCategoryRateLimited ||
		result.Effect != health.EffectCooldownCredential ||
		!result.CooldownUntil.Equal(now.Add(7*time.Second)) {
		t.Fatalf("JudgeExecution() = %#v", result)
	}

	result = judgeUpstreamResult(UpstreamResult{
		DispatchState: execution.DispatchNotSent,
		ExecutionError: &execution.ErrorEvidence{
			Kind: execution.ErrorKindTransport, Summary: "connection failed",
		},
	}, now, health.DecisionContext{DefaultRateLimitCooldown: time.Minute})
	if result.Category != health.FailureCategoryUpstreamHostError || result.Effect != health.EffectSkipGroup {
		t.Fatalf("JudgeExecution(not sent) = %#v", result)
	}
}

func TestJudgeUpstreamResultKeepsUpstreamTimeoutOutOfDownstreamCancellation(t *testing.T) {
	t.Parallel()

	evidence := &execution.ErrorEvidence{
		Kind: execution.ErrorKindTimeout, OriginHint: execution.ErrorOriginUpstream,
		ScopeHint: execution.ErrorScopeGroup, Summary: "upstream connection timed out",
	}
	upstream := upstreamFromExecutionResult(
		context.Background(),
		ForwardInput{},
		execution.AttemptResult{
			DispatchState: execution.DispatchNotSent,
			Error:         evidence,
		},
	)
	decision := judgeUpstreamResult(upstream, time.Now(), health.DecisionContext{})
	if decision.Category != health.FailureCategoryUpstreamHostError ||
		decision.Origin != execution.ErrorOriginUpstream ||
		decision.Scope != execution.ErrorScopeGroup ||
		decision.Retry != health.RetryNextCandidate ||
		decision.Effect != health.EffectSkipGroup ||
		decision.RuleID != "transport.not_sent" {
		t.Fatalf("timeout decision = %#v", decision)
	}
}

func TestJudgeUpstreamResultClassifiesUncommittedProtocolFailureAsUpstream(t *testing.T) {
	t.Parallel()

	decision := judgeUpstreamResult(UpstreamResult{
		Err:             fmt.Errorf("%w: invalid execution stream event", ErrUpstreamProtocol),
		DispatchState:   execution.DispatchMaybeSent,
		ResponseStarted: true,
		StatusCode:      http.StatusOK,
	}, time.Now(), health.DecisionContext{})
	if decision.Category != health.FailureCategoryAmbiguous ||
		decision.Origin != execution.ErrorOriginUpstream ||
		decision.Scope != execution.ErrorScopeRequest ||
		decision.Retry != health.RetryNone ||
		decision.Effect != health.EffectNone ||
		decision.RuleID != "stream.protocol_error" {
		t.Fatalf("protocol decision = %#v", decision)
	}
}

func TestJudgeUpstreamResultUsesStableStreamTerminalRules(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		endReason  StreamEndReason
		wantRule   health.RuleID
		wantOrigin execution.ErrorOrigin
		wantScope  execution.ErrorScope
	}{
		{name: "provider error", endReason: StreamEndSSEError, wantRule: "stream.provider_error", wantOrigin: execution.ErrorOriginUpstream, wantScope: execution.ErrorScopeRequest},
		{name: "upstream terminated", endReason: StreamEndUpstreamTerminated, wantRule: "stream.upstream_terminated", wantOrigin: execution.ErrorOriginUpstream, wantScope: execution.ErrorScopeRequest},
		{name: "protocol error", endReason: StreamEndUpstreamProtocolError, wantRule: "stream.protocol_error", wantOrigin: execution.ErrorOriginUpstream, wantScope: execution.ErrorScopeRequest},
		{name: "idle timeout", endReason: StreamEndIdleTimeout, wantRule: "stream.idle_timeout", wantOrigin: execution.ErrorOriginUpstream, wantScope: execution.ErrorScopeRequest},
		{name: "provider incomplete", endReason: StreamEndProviderIncomplete, wantRule: "stream.provider_incomplete", wantOrigin: execution.ErrorOriginUpstream, wantScope: execution.ErrorScopeRequest},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision := judgeUpstreamResult(UpstreamResult{
				DispatchState:   execution.DispatchMaybeSent,
				ResponseStarted: true,
				StatusCode:      http.StatusOK,
				Committed:       true,
				Stream:          streamTerminalObservation(test.endReason),
			}, time.Now(), health.DecisionContext{})
			if decision.Category != health.FailureCategoryAmbiguous ||
				decision.Origin != test.wantOrigin || decision.Scope != test.wantScope ||
				decision.Retry != health.RetryNone || decision.Effect != health.EffectNone ||
				decision.RuleID != test.wantRule {
				t.Fatalf("judgeUpstreamResult() = %#v", decision)
			}
		})
	}
}

func TestNormalizeUpstreamResultContractFailsClosedWithoutBodyInference(t *testing.T) {
	t.Parallel()

	result := normalizeUpstreamResultContract(UpstreamResult{
		StatusCode:         http.StatusTooManyRequests,
		Body:               []byte("private raw body"),
		ClassificationBody: []byte("private classification body"),
		Err:                errors.New("private raw error"),
	})
	if result.DispatchState != execution.DispatchMaybeSent || result.ExecutionError == nil ||
		result.ExecutionError.Kind != execution.ErrorKindInternal ||
		result.ExecutionError.Code != "attempt_result_contract_invalid" ||
		result.ExecutionError.Summary != "Attempt forwarder returned an invalid result." ||
		result.Body != nil || result.ClassificationBody != nil {
		t.Fatalf("normalized result = %#v", result)
	}
	for _, value := range []string{result.ErrorSummary, result.ExecutionError.Summary, fmt.Sprint(result.Err)} {
		if strings.Contains(value, "private") {
			t.Fatalf("invalid forwarder contract leaked private detail: %q", value)
		}
	}
}
