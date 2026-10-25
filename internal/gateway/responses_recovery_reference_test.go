package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/state"
)

// Rotate on the exact lookup after affinitySelectionAvailable has checked the
// current identity and readiness. This deterministically exercises that narrow
// race without sleeps, while the provider/executor boundary remains real HTTP.
type recoveryReferenceRotatingRegistry struct {
	*state.CredentialRegistry
	armed  atomic.Bool
	checks atomic.Int32
	rotate func()
}

func (registry *recoveryReferenceRotatingRegistry) ActiveEncryptedCredentialDataIfMatch(ref state.CredentialRef) (string, bool) {
	if registry.armed.Load() && registry.checks.Add(1) == 2 {
		registry.rotate()
	}
	return registry.CredentialRegistry.ActiveEncryptedCredentialDataIfMatch(ref)
}

func TestResponsesRecoveryCurrentReferenceVersionRefresh(t *testing.T) {
	for _, buffered := range []bool{false, true} {
		for _, afterHealth := range []bool{false, true} {
			name := "unary"
			if buffered {
				name = "buffered"
			}
			if afterHealth {
				name += "/after_health_check"
			} else {
				name += "/before_replay_selection"
			}
			t.Run(name, func(t *testing.T) {
				captured := &capturingExecutionForwarder{delegate: newConcreteTestExecutionForwarder(t)}
				handler, engine, _ := newContinuationFixture(t, captured)
				registry := &recoveryReferenceRotatingRegistry{CredentialRegistry: handler.registry.(*state.CredentialRegistry)}
				handler.registry = registry
				encrypted, err := handler.encryption.Encrypt(`{"api_key":"sk-refreshed"}`)
				if err != nil {
					t.Fatal(err)
				}
				registry.rotate = func() {
					ref, ok := registry.CredentialRef(1)
					if !ok || !registry.ReplaceCredentialSecretIfMatch(1, ref.Version, ref.Version+1, "refreshed-fingerprint", encrypted) {
						t.Error("same-identity secret/version refresh failed")
					}
				}
				type observed struct {
					auth string
					body json.RawMessage
				}
				requests := make(chan observed, 4)
				var calls atomic.Int32
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var body json.RawMessage
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
					}
					requests <- observed{auth: r.Header.Get("Authorization"), body: body}
					first := calls.Add(1) == 1
					if first {
						if afterHealth {
							registry.armed.Store(true)
						} else {
							registry.rotate()
						}
					}
					if buffered {
						w.Header().Set("Content-Type", "text/event-stream")
						if first {
							_, _ = w.Write([]byte("event: error\ndata: {\"type\":\"error\",\"error\":{\"code\":\"invalid_encrypted_content\",\"message\":\"encrypted content could not be verified\"}}\n\n"))
						} else {
							_, _ = w.Write([]byte(bufferedResponsesFramingSuccessFixture))
						}
						w.(http.Flusher).Flush()
						return
					}
					w.Header().Set("Content-Type", "application/json")
					if first {
						w.WriteHeader(400)
						_, _ = w.Write([]byte(`{"error":{"code":"invalid_encrypted_content","message":"encrypted content could not be verified"}}`))
					} else {
						_, _ = w.Write(storedResponse("refreshed-response").Body)
					}
				}))
				t.Cleanup(server.Close)
				setContinuationChannel(t, handler, channel.OpenAI, server.URL)
				handler.manager.Current().Settings.RetryCount = 3
				useAffinityRandomValues(handler, 0, 0, 0)
				initial, _ := registry.CredentialRef(1)
				body := recoveryHistory
				if buffered {
					body = strings.TrimSuffix(body, "}") + `,"stream":true,"store":false}`
				}
				response := performBufferedResponsesFramingRequest(t, engine, body)
				wantCalls := 2
				if afterHealth {
					wantCalls = 1
				}
				if got := int(calls.Load()); got != wantCalls {
					t.Errorf("provider calls = %d, want %d; pinned recovery must not fall through", got, wantCalls)
				}
				for i := 0; i < int(calls.Load()); i++ {
					request := <-requests
					wantAuth := "Bearer sk-one"
					if i > 0 {
						wantAuth = "Bearer sk-refreshed"
					}
					if request.auth != wantAuth {
						t.Errorf("attempt %d authorization = %q, want %q", i, request.auth, wantAuth)
					}
					if i > 0 && bytes.Contains(request.body, []byte("opaque")) {
						t.Errorf("replay kept rejected history: %s", request.body)
					}
				}
				current, _ := registry.CredentialRef(1)
				if current.IdentityGeneration != initial.IdentityGeneration || current.Version <= initial.Version {
					t.Fatalf("fixture did not refresh version while preserving identity: before=%+v after=%+v", initial, current)
				}
				if buffered {
					first := captured.results[0]
					if !first.Committed || !first.HTTPCommitted || first.PayloadReleased {
						t.Fatalf("first result not heartbeat-only committed: %+v", first)
					}
					want := bufferedStreamHeartbeat + bufferedResponsesFramingSuccessFixture
					if afterHealth {
						want = bufferedStreamHeartbeat + bufferedResponsesFramingAmbiguousErrorEvent
					}
					if got := response.Body.String(); got != want {
						t.Errorf("buffered downstream = %q, want %q", got, want)
					}
				} else if afterHealth {
					if response.Code != 400 || !bytes.Contains(response.Body.Bytes(), []byte("invalid_encrypted_content")) {
						t.Errorf("original rejection lost: status=%d body=%s", response.Code, response.Body.String())
					}
				} else if response.Code != 200 {
					t.Errorf("refreshed replay status=%d body=%s", response.Code, response.Body.String())
				}
			})
		}
	}
}
