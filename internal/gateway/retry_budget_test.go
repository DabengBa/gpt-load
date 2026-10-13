package gateway

import (
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

// first_provider_attempts 允许在同一供应商（同分组、凭据、上游模型 Selection）
// 上重复实际执行：配置 2、全局预算 4 时失败序列产生 A-A-B-C；预算耗尽截断。
// 重试必须共享同一全局 retry_count 预算，成功立即结束。
func TestHandlerReplaysFirstProviderWithinBudget(t *testing.T) {
	for _, tc := range []struct {
		name             string
		attempts, budget int
		want             []uint
		success          bool
	}{
		{"two", 2, 4, []uint{1, 1, 2, 3}, false},
		{"three", 3, 4, []uint{1, 1, 1, 2}, false},
		{"truncated", 9, 2, []uint{1, 1}, false},
		{"success", 3, 4, []uint{1, 1}, true},
	} {
		t.Run(tc.name, func(t *testing.T) { testFirstProviderSequence(t, tc.attempts, tc.budget, tc.want, tc.success) })
	}
}

func testFirstProviderSequence(t *testing.T, attempts, budget int, want []uint, success bool) {
	unauthorized := UpstreamResult{
		StatusCode: http.StatusUnauthorized, Header: make(http.Header), RequestWritten: true,
		Body:               []byte(`{"error":"invalid_api_key"}`),
		ClassificationBody: []byte(`{"error":"invalid_api_key"}`),
	}
	forwarder := &scriptedForwarder{results: []UpstreamResult{unauthorized, unauthorized, unauthorized, unauthorized}}
	if success {
		forwarder.results[1] = UpstreamResult{StatusCode: http.StatusOK, Header: make(http.Header), Body: []byte(`{"choices":[{"message":{"content":"ok"}}]}`)}
	}
	handler, manager, _ := newHandlerForTest(t, forwarder, "sk-first", "sk-second", "sk-third", "sk-fourth")
	handler.newRandom = func() *rand.Rand { return rand.New(zeroSource{}) }
	input := state.CompileInput{
		SystemSettings: config.Settings{
			state.SettingRetryCount:            budget,
			state.SettingFirstProviderAttempts: attempts,
		},
		ChannelRegistry: channel.NewRegistry(),
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: handler.encryption.Hash("gl-client"), Status: state.AccessKeyStatusActive,
		}},
	}
	for _, groupID := range []uint{1, 2, 3, 4} {
		input.Groups = append(input.Groups, state.GroupConfig{
			ID: groupID, Name: fmt.Sprintf("group-%d", groupID), ChannelID: channel.OpenAI, ConnectionType: "api_key",
			Params: json.RawMessage(`{}`), Models: []state.ModelConfig{{ID: "gpt-4o"}}, Enabled: true,
		})
		input.Credentials = append(input.Credentials, state.CredentialConfig{
			ID: groupID, GroupID: groupID, Version: 1, IdentityGeneration: uint64(groupID),
			Fingerprint: fmt.Sprintf("credential-%d", groupID),
		})
	}
	if _, err := manager.Publish(input); err != nil {
		t.Fatal(err)
	}
	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o"}`))
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)

	expectedStatus := http.StatusUnauthorized
	if success {
		expectedStatus = http.StatusOK
	}
	if response.Code != expectedStatus {
		t.Fatalf("status = %d, want 401; body=%s", response.Code, response.Body.String())
	}
	gotGroups := make([]uint, 0, len(forwarder.inputs))
	for _, attempt := range forwarder.inputs {
		gotGroups = append(gotGroups, attempt.Group.ID)
	}
	if !reflect.DeepEqual(gotGroups, want) {
		t.Fatalf("attempt sequence = %v, want A-A-B-C %v", gotGroups, want)
	}
}

// first_provider_attempts=1（默认）保持现状：首次失败后必须换候选，首供应商不重试。
func TestHandlerFirstProviderAttemptsDefaultOneNeverReplaysFirstCandidate(t *testing.T) {
	unauthorized := UpstreamResult{
		StatusCode: http.StatusUnauthorized, Header: make(http.Header), RequestWritten: true,
		Body:               []byte(`{"error":"invalid_api_key"}`),
		ClassificationBody: []byte(`{"error":"invalid_api_key"}`),
	}
	forwarder := &scriptedForwarder{results: []UpstreamResult{unauthorized, unauthorized, unauthorized}}
	handler, manager, _ := newHandlerForTest(t, forwarder, "sk-first", "sk-second", "sk-third")
	handler.newRandom = func() *rand.Rand { return rand.New(zeroSource{}) }
	input := state.CompileInput{
		SystemSettings:  config.Settings{state.SettingRetryCount: 3},
		ChannelRegistry: channel.NewRegistry(),
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: handler.encryption.Hash("gl-client"), Status: state.AccessKeyStatusActive,
		}},
	}
	for _, groupID := range []uint{1, 2, 3} {
		input.Groups = append(input.Groups, state.GroupConfig{
			ID: groupID, Name: fmt.Sprintf("group-%d", groupID), ChannelID: channel.OpenAI, ConnectionType: "api_key",
			Params: json.RawMessage(`{}`), Models: []state.ModelConfig{{ID: "gpt-4o"}}, Enabled: true,
		})
		input.Credentials = append(input.Credentials, state.CredentialConfig{
			ID: groupID, GroupID: groupID, Version: 1, IdentityGeneration: uint64(groupID),
			Fingerprint: fmt.Sprintf("credential-%d", groupID),
		})
	}
	if _, err := manager.Publish(input); err != nil {
		t.Fatal(err)
	}
	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o"}`))
	request.Header.Set("Authorization", "Bearer gl-client")
	engine.ServeHTTP(httptest.NewRecorder(), request)

	gotGroups := make([]uint, 0, len(forwarder.inputs))
	for _, attempt := range forwarder.inputs {
		gotGroups = append(gotGroups, attempt.Group.ID)
	}
	if !reflect.DeepEqual(gotGroups, []uint{1, 2, 3}) {
		t.Fatalf("attempt sequence = %v, want A-B-C without first-provider replay", gotGroups)
	}
}

type credentialCheckCountingRegistry struct {
	*state.CredentialRegistry
	checks int
}

func (registry *credentialCheckCountingRegistry) ActiveEncryptedCredentialDataIfMatch(ref state.CredentialRef) (string, bool) {
	registry.checks++
	return registry.CredentialRegistry.ActiveEncryptedCredentialDataIfMatch(ref)
}

func TestHandlerFirstProviderRotationEndsReplayWithoutSpinning(t *testing.T) {
	rejected := UpstreamResult{
		StatusCode: http.StatusUnauthorized, Header: make(http.Header), RequestWritten: true,
		Body:               []byte(`{"error":"invalid_api_key"}`),
		ClassificationBody: []byte(`{"error":"invalid_api_key"}`),
	}
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		rejected,
		{StatusCode: http.StatusOK, Header: make(http.Header), Body: []byte(`{"ok":true}`)},
	}}
	handler, manager, registry := newHandlerForTest(t, forwarder, "sk-first", "sk-backup")
	handler.newRandom = func() *rand.Rand { return rand.New(zeroSource{}) }
	publishHandlerPolicySettings(t, handler, manager, 2, config.Settings{
		state.SettingRetryCount:            2,
		state.SettingFirstProviderAttempts: 1000,
	}, nil)
	counted := &credentialCheckCountingRegistry{CredentialRegistry: registry}
	handler.registry = counted
	forwarder.onCall = func(index int) {
		if index != 0 {
			return
		}
		encrypted, err := handler.encryption.Encrypt(`{"api_key":"sk-rotated"}`)
		if err != nil {
			t.Fatal(err)
		}
		ref, ok := registry.CredentialRef(forwarder.inputs[0].Credential.ID)
		if !ok || !registry.ReplaceCredentialSecretIfMatch(ref.ID, ref.Version, ref.Version+1, "rotated", encrypted) {
			t.Fatal("credential rotation failed")
		}
	}
	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o"}`))
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)

	if response.Code != http.StatusOK || len(forwarder.inputs) != 2 {
		t.Fatalf("status=%d forwards=%d, want success within the two-attempt budget", response.Code, len(forwarder.inputs))
	}
	if first, second := forwarder.inputs[0].Group.ID, forwarder.inputs[1].Group.ID; first != 1 || second != 2 {
		t.Fatalf("attempt sequence = [%d %d], want A-B after rotation", first, second)
	}
	// One check for each actual forward plus at most one rejected replay;
	// the configured replay cap must not amplify stale-reference work.
	if counted.checks > 3 {
		t.Fatalf("credential identity checks = %d, want at most 3 for two forwards", counted.checks)
	}
}

func TestHandlerRotationBeforeFirstForwardKeepsFirstProviderReplay(t *testing.T) {
	rejected := UpstreamResult{
		StatusCode: http.StatusUnauthorized, Header: make(http.Header), RequestWritten: true,
		Body:               []byte(`{"error":"invalid_api_key"}`),
		ClassificationBody: []byte(`{"error":"invalid_api_key"}`),
	}
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		rejected,
		{StatusCode: http.StatusOK, Header: make(http.Header), Body: []byte(`{"ok":true}`)},
	}}
	handler, manager, registry := newHandlerForTest(t, forwarder, "sk-first", "sk-backup")
	handler.newRandom = func() *rand.Rand { return rand.New(zeroSource{}) }
	publishHandlerPolicySettings(t, handler, manager, 2, config.Settings{
		state.SettingRetryCount:            2,
		state.SettingFirstProviderAttempts: 2,
	}, nil)
	handler.registry = &mutatingRuntimeRegistry{CredentialRegistry: registry, mutate: func() {
		encrypted, err := handler.encryption.Encrypt(`{"api_key":"sk-rotated"}`)
		if err != nil {
			t.Fatal(err)
		}
		if !registry.ReplaceCredentialSecretIfMatch(1, 1, 2, "rotated", encrypted) {
			t.Fatal("credential rotation failed")
		}
	}}
	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o"}`))
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)

	if response.Code != http.StatusOK || len(forwarder.inputs) != 2 {
		t.Fatalf("status=%d forwards=%d, want success after replaying the first actual provider", response.Code, len(forwarder.inputs))
	}
	if first, second := forwarder.inputs[0].Group.ID, forwarder.inputs[1].Group.ID; first != 2 || second != 2 {
		t.Fatalf("attempt sequence = [%d %d], want B-B after skipping stale A", first, second)
	}
}
