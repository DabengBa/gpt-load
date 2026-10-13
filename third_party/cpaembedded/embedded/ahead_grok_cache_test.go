package embedded

import (
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestAheadGrokCacheFinalWire(t *testing.T) {
	validEncryptedContent := aheadValidGrokEncryptedContent()
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			type capture struct{ body, conversation string }
			captured := make(chan capture, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Errorf("read outbound body: %v", err)
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				captured <- capture{string(body), r.Header.Get("x-grok-conv-id")}
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, "event: response.completed\ndata: "+fmt.Sprintf(`{"type":"response.completed","response":{"id":"resp_fixture","object":"response","status":"completed","model":"grok-4.3","output":[{"type":"reasoning","id":"rs_fixture","summary":[],"encrypted_content":%q},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"fixture answer"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`, validEncryptedContent)+"\n\n")
			}))
			t.Cleanup(server.Close)
			executor := testGrokHTTPExecutor(server.URL)
			send := func(scope, attempt, input string) capture {
				t.Helper()
				payload := []byte(fmt.Sprintf(`{"model":"grok-4.3","input":[{"role":"user","content":%q}],"prompt_cache_key":"caller-key","session_id":"caller-session"}`, input))
				request := ExecuteRequest{Model: "grok-4.3", Format: "openai-response", AttemptID: attempt, ContinuityKey: scope, Payload: payload, OriginalRequest: payload, ProxyURL: "direct"}
				if stream {
					response, err := executor.ExecuteStreamCanonical(t.Context(), "credential-1", testGrokExecutionCredential(), request)
					if err != nil {
						t.Fatal(err)
					}
					for chunk := range response.Chunks {
						if chunk.Err != nil {
							t.Fatal(chunk.Err)
						}
					}
				} else {
					if _, err := executor.ExecuteCanonical(t.Context(), "credential-1", testGrokExecutionCredential(), request); err != nil {
						t.Fatal(err)
					}
				}
				got := <-captured
				t.Logf("scope=%q attempt=%q final body=%s conversation=%s", scope, attempt, got.body, got.conversation)
				key := gjson.Get(got.body, "prompt_cache_key").String()
				if key == "" || key == "caller-key" || key != got.conversation {
					t.Errorf("gateway cache identity: body key=%q conversation=%q", key, got.conversation)
				}
				if gjson.Get(got.body, "session_id").Exists() {
					t.Errorf("caller session survived: %s", got.body)
				}
				return got
			}
			// A fresh server URL makes the test continuity scope unique even if
			// the SDK's process-wide replay cache persists between tests.
			scope := server.URL + "\x00tenant\x00credential-1\x00grok-4.3"
			first := send(scope, "turn-1", "first question")
			second := send(scope, "turn-2", "second question")
			key := func(c capture) string { return gjson.Get(c.body, "prompt_cache_key").String() }
			if key(first) != key(second) {
				t.Errorf("continuity changed: %q / %q", key(first), key(second))
			}
			otherCredential := send(server.URL+"\x00tenant\x00credential-2\x00grok-4.3", "turn-3", "other credential")
			otherModel := send(server.URL+"\x00tenant\x00credential-1\x00other-model", "turn-4", "other model scope")
			if key(first) == key(otherCredential) || key(first) == key(otherModel) || key(otherCredential) == key(otherModel) {
				t.Error("distinct continuity scopes reused cache identity")
			}
			attemptA := send("", server.URL+"attempt-a", "no continuity")
			attemptAgain := send("", server.URL+"attempt-a", "same attempt")
			attemptB := send("", server.URL+"attempt-b", "different attempt")
			if key(attemptA) != key(attemptAgain) || key(attemptA) == key(attemptB) || key(attemptA) == key(first) {
				t.Error("attempt cache isolation failed")
			}
			if strings.Contains(second.body, validEncryptedContent) || gjson.Get(second.body, `input.#(type=="reasoning")`).Exists() {
				t.Errorf("unexpected reasoning replay injected into second final HTTP body: %s", second.body)
			}
		})
	}
}

func aheadValidGrokEncryptedContent() string {
	buf := make([]byte, 0, 256)
	for i := 0; len(buf) < 256; i++ {
		sum := sha256.Sum256([]byte{byte(i), byte(i >> 8), byte(i >> 16)})
		buf = append(buf, sum[:]...)
	}
	return base64.RawStdEncoding.EncodeToString(buf[:256])
}
