//go:build !windows

package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/state"
)

func TestHandlerBufferedCredentialRefreshRespectsFirstProviderAndGlobalBudgets(t *testing.T) {
	type attempt struct {
		credentialID uint
		forceRefresh bool
	}
	const successPayload = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"recovered\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	for _, tc := range []struct {
		name                  string
		firstAttempts, budget int
		secondStatus          int
		want                  []attempt
		wantSuccess           bool
	}{
		{"default refresh succeeds", 1, 2, 200, []attempt{{1, false}, {1, true}}, true},
		{"configured refresh succeeds", 3, 3, 200, []attempt{{1, false}, {1, true}}, true},
		{"refresh rejected switches provider", 3, 3, 401, []attempt{{1, false}, {1, true}, {2, false}}, true},
		{"other failure after refresh switches provider", 3, 3, 400, []attempt{{1, false}, {1, true}, {2, false}}, true},
		{"one attempt prevents refresh", 3, 1, 401, []attempt{{1, false}}, false},
		{"refresh consumes last attempt", 3, 2, 401, []attempt{{1, false}, {1, true}}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var got []attempt
			executor := fakeExecutionExecutor{stream: func(_ context.Context, spec execution.AttemptSpec, sink execution.StreamSink) execution.StreamResult {
				got = append(got, attempt{spec.Credential.ID, spec.ForceCredentialRefresh})
				status := http.StatusOK
				if spec.Credential.ID == 1 {
					status = http.StatusUnauthorized
					if len(got) == 2 {
						status = tc.secondStatus
					}
				}
				result := execution.StreamResult{
					DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: status,
					Header: http.Header{"Content-Type": {"text/event-stream"}},
				}
				if status != http.StatusOK {
					result.Header.Set("Content-Type", "application/json")
					result.Error = &execution.ErrorEvidence{
						Kind: execution.ErrorKindHTTP, StatusCode: status,
						OriginHint:   execution.ErrorOriginUpstream,
						ReplaySafety: execution.ReplaySafetyRejectedBeforeProcessing,
					}
					if status == http.StatusUnauthorized {
						result.Error.Hint = execution.FailureHintRefreshRequired
						result.Error.ScopeHint = execution.ErrorScopeCredential
						result.Error.Summary = "access token expired"
					} else {
						result.Error.Hint = execution.FailureHintRequestRejected
						result.Error.Summary = "request rejected"
					}
					return result
				}
				for _, event := range []execution.StreamEvent{
					{Sequence: 1, Kind: execution.StreamEventReady, StatusCode: status, Header: result.Header},
					{Sequence: 2, Kind: execution.StreamEventData, Data: []byte(successPayload)},
				} {
					if err := sink(event); err != nil {
						t.Errorf("stream sink: %v", err)
						return result
					}
				}
				return result
			}}
			forwarder := &capturingExecutionForwarder{delegate: NewExecutionForwarder(executor)}
			handler, manager, registry := newHandlerForTest(t, forwarder)
			handler.newRandom = func() *rand.Rand { return rand.New(zeroSource{}) }
			input := state.CompileInput{
				SystemSettings: config.Settings{
					state.SettingRetryCount: tc.budget, state.SettingFirstProviderAttempts: tc.firstAttempts,
				},
				ChannelRegistry: channel.NewRegistry(),
				AccessKeys: []state.AccessKeyConfig{{
					ID: 1, Name: "client", KeyHash: handler.encryption.Hash("gl-client"), Status: state.AccessKeyStatusActive,
				}},
			}
			var entries []state.CredentialEntry
			for _, id := range []uint{1, 2} {
				input.Groups = append(input.Groups, state.GroupConfig{
					ID: id, Name: fmt.Sprintf("subscription-%d", id), ChannelID: channel.Codex,
					ConnectionType: "subscription", Params: json.RawMessage(`{}`),
					Models: []state.ModelConfig{{ID: "gpt-4o", ConcurrencyLimit: 1}}, Enabled: true,
				})
				fingerprint := fmt.Sprintf("account-%d", id)
				input.Credentials = append(input.Credentials, state.CredentialConfig{
					ID: id, GroupID: id, Version: 1, IdentityGeneration: uint64(id), Fingerprint: fingerprint,
				})
				canonical := fmt.Sprintf(`{"type":"codex","access_token":"access-%d","refresh_token":"refresh-%d","account_id":"account-%d"}`, id, id, id)
				encrypted, err := handler.encryption.Encrypt(canonical)
				if err != nil {
					t.Fatal(err)
				}
				entries = append(entries, state.CredentialEntry{
					ID: id, GroupID: id, Version: 1, IdentityGeneration: uint64(id),
					Fingerprint: fingerprint, EncryptedValue: encrypted,
				})
			}
			if _, err := manager.Publish(input); err != nil {
				t.Fatal(err)
			}
			if err := registry.ReplaceCredentials(entries); err != nil {
				t.Fatal(err)
			}
			engine := gin.New()
			bindGatewayRoutesForTest(t, engine, handler)
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o","stream":true}`))
			request.Header.Set("Authorization", "Bearer gl-client")
			response := httptest.NewRecorder()
			engine.ServeHTTP(response, request)

			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("attempts (credential, force refresh) = %v, want %v", got, tc.want)
			}
			if len(forwarder.results) == 0 {
				t.Fatal("no upstream attempt")
			}
			first := forwarder.results[0]
			if !first.BufferedStream || !first.HTTPCommitted || first.PayloadReleased || first.ClientVisibleBytes == 0 {
				t.Fatalf("first result = %#v, want heartbeat-only buffered failure", first)
			}
			if tc.wantSuccess && response.Body.String() != bufferedStreamHeartbeat+successPayload {
				t.Errorf("body = %q, want one heartbeat followed by recovered payload", response.Body.String())
			}
			if !tc.wantSuccess && strings.Contains(response.Body.String(), "recovered") {
				t.Errorf("exhausted request released successful payload: %q", response.Body.String())
			}
		})
	}
}
