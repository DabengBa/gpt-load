package gateway

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/state"
)

// retry_count 是一次请求的上游尝试总次数上限：0 与 1 都只尝试一次，2 表示失败后可换一次候选。
func TestRetryAttemptLimitTreatsRetryCountAsTotalAttemptBudget(t *testing.T) {
	for _, test := range []struct {
		retryCount int
		want       int
	}{
		{retryCount: 0, want: 1},
		{retryCount: 1, want: 1},
		{retryCount: 2, want: 2},
		{retryCount: 3, want: 3},
		{retryCount: 100, want: 100},
	} {
		if got := retryAttemptLimit(test.retryCount); got != test.want {
			t.Fatalf("retryAttemptLimit(retry_count=%d) = %d, want %d", test.retryCount, got, test.want)
		}
	}
}

// 预算只来自系统 retry_count，并在请求开始时冻结：跨分组链条不会叠加，也不会被后到的分组放大或缩小；
// 分组里历史遗留的 retry_count 一律不生效。
func TestHandlerFreezesSystemRetryBudgetAcrossGroups(t *testing.T) {
	for _, test := range []struct {
		name            string
		systemCount     int
		groupCounts     []int
		decryptFailures int
		updatedCount    *int
		wantAttempts    int
	}{
		{name: "system zero overrides legacy group retries", systemCount: 0, groupCounts: []int{4, 4, 4, 4}, wantAttempts: 1},
		{name: "system budget of two crosses groups", systemCount: 2, groupCounts: []int{0, 0, 0, 0}, wantAttempts: 2},
		{name: "later group cannot increase request budget", systemCount: 1, groupCounts: []int{4, 4, 4, 4}, wantAttempts: 1},
		{name: "later group cannot reduce request budget", systemCount: 3, groupCounts: []int{1, 0, 0, 0}, wantAttempts: 3},
		{name: "preparation failure does not consume forward budget", systemCount: 2, groupCounts: []int{0, 0, 0, 0}, decryptFailures: 1, wantAttempts: 2},
		{name: "single attempt budget still skips unusable candidates", systemCount: 0, groupCounts: []int{4, 4, 4, 4}, decryptFailures: 1, wantAttempts: 1},
		{name: "runtime update only affects the next request", systemCount: 2, groupCounts: []int{4, 4, 4, 4}, updatedCount: new(0), wantAttempts: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			invalid := UpstreamResult{
				StatusCode: http.StatusUnauthorized, Header: make(http.Header), RequestWritten: true,
				Body:               []byte(`{"error":"invalid_api_key"}`),
				ClassificationBody: []byte(`{"error":"invalid_api_key"}`),
			}
			forwarder := &scriptedForwarder{results: []UpstreamResult{invalid, invalid, invalid, invalid, invalid}}
			handler, manager, _ := newHandlerForTest(t, forwarder, "sk-first", "sk-second", "sk-third", "sk-fourth")
			handler.newRandom = func() *rand.Rand { return rand.New(zeroSource{}) }
			input := state.CompileInput{
				SystemSettings:  config.Settings{state.SettingRetryCount: test.systemCount},
				ChannelRegistry: channel.NewRegistry(),
				AccessKeys: []state.AccessKeyConfig{{
					ID: 1, Name: "client", KeyHash: handler.encryption.Hash("gl-client"), Status: state.AccessKeyStatusActive,
				}},
			}
			for index, groupCount := range test.groupCounts {
				id := uint(index + 1)
				input.Groups = append(input.Groups, state.GroupConfig{
					ID: id, Name: fmt.Sprintf("group-%d", id), ChannelID: channel.OpenAI, ConnectionType: "api_key",
					Params: json.RawMessage(`{}`), Models: []state.ModelConfig{{ID: "gpt-4o"}}, Enabled: true,
					Settings: config.Settings{state.SettingRetryCount: groupCount},
				})
				input.Credentials = append(input.Credentials, state.CredentialConfig{
					ID: id, GroupID: id, Version: 1, IdentityGeneration: uint64(id),
					Fingerprint: fmt.Sprintf("credential-%d", id),
				})
			}
			if _, err := manager.Publish(input); err != nil {
				t.Fatal(err)
			}
			if test.decryptFailures > 0 {
				handler.encryption = &failCredentialDecrypt{Service: handler.encryption, remaining: test.decryptFailures}
			}
			if test.updatedCount != nil {
				forwarder.onCall = func(index int) {
					if index == 0 {
						input.SystemSettings[state.SettingRetryCount] = *test.updatedCount
						if _, err := manager.Publish(input); err != nil {
							t.Fatal(err)
						}
					}
				}
			}
			engine := gin.New()
			bindGatewayRoutesForTest(t, engine, handler)
			send := func() {
				request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o"}`))
				request.Header.Set("Authorization", "Bearer gl-client")
				response := httptest.NewRecorder()
				engine.ServeHTTP(response, request)
				if response.Code != http.StatusUnauthorized {
					t.Fatalf("status = %d, want 401; body=%s", response.Code, response.Body.String())
				}
			}
			send()
			if len(forwarder.inputs) != test.wantAttempts {
				t.Fatalf("attempts = %d, want system budget %d", len(forwarder.inputs), test.wantAttempts)
			}
			for index, attempt := range forwarder.inputs {
				if want := uint(index + 1 + test.decryptFailures); attempt.Group.ID != want {
					t.Fatalf("attempt %d group = %d, want %d", index, attempt.Group.ID, want)
				}
			}
			if test.updatedCount != nil {
				send()
				want := *test.updatedCount
				if want < 1 {
					want = 1
				}
				if got := len(forwarder.inputs) - test.wantAttempts; got != want {
					t.Fatalf("next request attempts = %d, want updated system budget %d", got, want)
				}
			}
		})
	}
}
