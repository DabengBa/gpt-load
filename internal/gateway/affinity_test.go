package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"

	"gpt-load/internal/affinity"
	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/testutil/encryptiontest"
)

type affinityFixedRandSource struct {
	value int64
}

const affinitySecondCredentialRand int64 = 3000

func (source affinityFixedRandSource) Int63() int64 { return source.value }
func (affinityFixedRandSource) Seed(int64)          {}

func TestHandlerLearnsAndReusesAutomaticAffinity(t *testing.T) {
	forwarder := &scriptedForwarder{results: successfulAffinityResults(2)}
	handler, _, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	useAffinityRandomValues(handler, 0, affinitySecondCredentialRand)
	engine := newAffinityTestEngine(t, handler)

	serveAffinityRequest(t, engine, `{
		"model":"gpt-4o","temperature":0.1,
		"messages":[
			{"role":"system","content":"Be helpful"},
			{"role":"user","content":"Hello"}
		]
	}`)
	serveAffinityRequest(t, engine, `{
		"model":"gpt-4o","temperature":1,
		"messages":[
			{"role":"system","content":[{"type":"text","text":"Be helpful"}]},
			{"role":"user","content":[{"type":"input_text","text":"Hello"}]},
			{"role":"assistant","content":"prior answer"},
			{"role":"user","content":"later turn"}
		]
	}`)

	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one", "sk-one"})
	events := sink.snapshot()
	assertAffinityHits(t, events, []bool{false, true})
	if events[0].AffinityKey == "" || events[0].AffinityKey != events[1].AffinityKey ||
		!affinity.ValidDisplayKey(events[0].AffinityKey) {
		t.Fatalf("affinity display keys = %q / %q, want equal canonical masked projections", events[0].AffinityKey, events[1].AffinityKey)
	}
}

func TestHandlerIsolatesAffinityAcrossClientModels(t *testing.T) {
	forwarder := &scriptedForwarder{results: successfulAffinityResults(2)}
	handler, manager, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	addAffinityModelRoute(t, manager.Current(), "gpt-4o-mini", 1)
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	useAffinityRandomValues(handler, 0)
	engine := newAffinityTestEngine(t, handler)

	serveAffinityModelRequest(t, engine, "gpt-4o")
	serveAffinityModelRequest(t, engine, "gpt-4o-mini")

	// 客户端模型是亲和键的一部分：第二个模型只能看到第一个凭据，且不得命中
	// 第一个模型学到的亲和绑定。
	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one", "sk-one"})
	events := sink.snapshot()
	assertAffinityHits(t, events, []bool{false, false})
	assertAffinitySources(t, events, []telemetry.AffinitySource{
		telemetry.AffinitySourcePromptPrefix, telemetry.AffinitySourcePromptPrefix,
	})
	assertAffinityStates(t, events, []telemetry.AffinityState{
		telemetry.AffinityStateCacheMiss, telemetry.AffinityStateCacheMiss,
	})
}

func TestHandlerPreservesAffinityWhenModelCandidateRangeChangesWithoutBoundAttempt(t *testing.T) {
	forwarder := &scriptedForwarder{results: successfulAffinityResults(3)}
	handler, manager, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	useAffinityRandomValues(handler, 0)
	engine := newAffinityTestEngine(t, handler)

	serveAffinityModelRequest(t, engine, "gpt-4o")
	// 将 gpt-4o 候选范围缩小到第二个凭据，使存储的首选项变得不合格。
	// 目标从未进入 Forward，因此后续成功不能静默迁移亲和绑定。
	addAffinityModelRoute(t, manager.Current(), "gpt-4o", 2)
	serveAffinityModelRequest(t, engine, "gpt-4o")
	serveAffinityModelRequest(t, engine, "gpt-4o")

	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one", "sk-two", "sk-two"})
	events := sink.snapshot()
	assertAffinityHits(t, events, []bool{false, false, false})
	assertAffinityStates(t, events, []telemetry.AffinityState{
		telemetry.AffinityStateCacheMiss,
		telemetry.AffinityStateTargetUnavailable,
		telemetry.AffinityStateTargetUnavailable,
	})
}

func TestHandlerDoesNotLearnAffinityForNonParticipatingGroup(t *testing.T) {
	forwarder := &scriptedForwarder{results: successfulAffinityResults(2)}
	handler, manager, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	snapshot := manager.Current()
	group := snapshot.Groups[1]
	group.AffinityEnabled = false
	snapshot.Groups[1] = group
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	useAffinityRandomValues(handler, 0, affinitySecondCredentialRand)
	engine := newAffinityTestEngine(t, handler)
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"stable conversation"}]}`

	serveAffinityRequest(t, engine, body)
	serveAffinityRequest(t, engine, body)

	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one", "sk-one"})
	assertAffinityHits(t, sink.snapshot(), []bool{false, false})
}

func TestHandlerGroupAffinityOverrideWinsOverGlobalSetting(t *testing.T) {
	tests := []struct {
		name           string
		globalEnabled  bool
		groupOverrides config.Settings
		wantKeys       []string
		wantHits       []bool
	}{
		{
			name: "group enables when global is disabled", globalEnabled: false,
			groupOverrides: config.Settings{state.SettingAffinityEnabled: true},
			wantKeys:       []string{"sk-one", "sk-one"}, wantHits: []bool{false, true},
		},
		{
			name: "group disables when global is enabled", globalEnabled: true,
			groupOverrides: config.Settings{state.SettingAffinityEnabled: false},
			wantKeys:       []string{"sk-one", "sk-one"}, wantHits: []bool{false, false},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			forwarder := &scriptedForwarder{results: successfulAffinityResults(2)}
			handler, manager, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
			current := manager.Current()
			group := current.Groups[1]
			resolved, err := state.ResolveGroupRuntimeSettings(state.RuntimeSettings{
				AffinityEnabled:  test.globalEnabled,
				AffinityTTL:      time.Hour,
				AffinityCapacity: 10_000,
			}, test.groupOverrides)
			if err != nil {
				t.Fatal(err)
			}
			group.AffinityEnabled = resolved.AffinityEnabled
			current.Settings.AffinityEnabled = test.globalEnabled
			current.Groups[1] = group
			sink := &recordingRequestLogSink{}
			handler.requestLogSink = sink
			useAffinityRandomValues(handler, 0, affinitySecondCredentialRand)
			engine := newAffinityTestEngine(t, handler)
			body := `{"model":"gpt-4o","messages":[{"role":"user","content":"stable conversation"}]}`

			serveAffinityRequest(t, engine, body)
			serveAffinityRequest(t, engine, body)

			assertAffinityAttemptKeys(t, forwarder.inputs, test.wantKeys)
			assertAffinityHits(t, sink.snapshot(), test.wantHits)
		})
	}
}

func TestHandlerDoesNotApplyAffinityWithoutInitialUserText(t *testing.T) {
	forwarder := &scriptedForwarder{results: successfulAffinityResults(2)}
	handler, _, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	useAffinityRandomValues(handler, 0, affinitySecondCredentialRand)
	engine := newAffinityTestEngine(t, handler)
	body := `{"model":"gpt-4o","messages":[{"role":"system","content":"shared instruction"}]}`

	serveAffinityRequest(t, engine, body)
	serveAffinityRequest(t, engine, body)

	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one", "sk-one"})
	events := sink.snapshot()
	assertAffinityHits(t, events, []bool{false, false})
	assertAffinitySources(t, events, []telemetry.AffinitySource{
		telemetry.AffinitySourceNone, telemetry.AffinitySourceNone,
	})
	assertAffinityStates(t, events, []telemetry.AffinityState{
		telemetry.AffinityStateNoSignal, telemetry.AffinityStateNoSignal,
	})
}

func TestHandlerReportsGroupDisabledAffinityState(t *testing.T) {
	forwarder := &scriptedForwarder{results: successfulAffinityResults(2)}
	handler, manager, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	useAffinityRandomValues(handler, 0, 1)

	engine := newAffinityTestEngine(t, handler)
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"stable conversation"}]}`

	serveAffinityRequest(t, engine, body)
	snapshot := manager.Current()
	group := snapshot.Groups[1]
	group.AffinityEnabled = false
	snapshot.Groups[1] = group
	serveAffinityRequest(t, engine, body)

	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one", "sk-two"})
	events := sink.snapshot()
	assertAffinityHits(t, events, []bool{false, false})
	assertAffinityStates(t, events, []telemetry.AffinityState{
		telemetry.AffinityStateCacheMiss, telemetry.AffinityStateGroupDisabled,
	})
}

func TestHandlerLearnsAffinityOnlyFromCleanCompletedStream(t *testing.T) {
	forwarder := &scriptedForwarder{streamResults: []UpstreamResult{
		{StatusCode: http.StatusOK, Committed: true, Stream: StreamObservation{EndReason: StreamEndProviderIncomplete}},
		{StatusCode: http.StatusOK, Committed: true, Stream: StreamObservation{EndReason: StreamEndCleanEOF}},
		{StatusCode: http.StatusOK, Committed: true, Stream: StreamObservation{EndReason: StreamEndCleanEOF}},
	}}
	handler, _, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	useAffinityRandomValues(handler, 0, affinitySecondCredentialRand, 0)
	engine := newAffinityTestEngine(t, handler)
	body := `{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"stable stream"}]}`

	serveAffinityRequest(t, engine, body)
	serveAffinityRequest(t, engine, body)
	serveAffinityRequest(t, engine, body)

	assertAffinityAttemptKeys(t, forwarder.streamInputs, []string{"sk-one", "sk-one", "sk-one"})
	assertAffinityHits(t, sink.snapshot(), []bool{false, false, true})
}

func TestHandlerUsesPromptCacheKeyForAffinityButPromptPrefixForContinuity(t *testing.T) {
	t.Parallel()

	handler, manager, _ := newHandlerForTest(t, &scriptedForwarder{}, "sk-one")
	snapshot := manager.Current()
	metadata := dialect.RequestMetadata{
		PromptCacheKey: "cache-a",
		AffinityPrefix: []byte(`{"v":1,"user":["stable"]}`),
	}
	resolved := handler.resolveRequestAffinity(
		t.Context(), snapshot, 1, protocol.OpenAIResponses, "gpt-4o", execution.OperationResponsesCreate,
		metadata, map[uint]state.CredentialRef{1: {ID: 1, GroupID: 1, IdentityGeneration: 1}},
	)
	wantAffinity := affinity.DeriveKey(
		handler.encryption, 1, protocol.OpenAIResponses, "gpt-4o", execution.OperationResponsesCreate,
		affinity.SignalPromptCacheKey, []byte("cache-a"),
	)
	wantContinuity := affinity.DeriveKey(
		handler.encryption, 1, protocol.OpenAIResponses, "gpt-4o", execution.OperationResponsesCreate,
		affinity.SignalPromptPrefix, metadata.AffinityPrefix,
	)
	if resolved.key != wantAffinity {
		t.Fatalf("affinity key = %q, want explicit cache key %q", resolved.key, wantAffinity)
	}
	if resolved.continuityKey != string(wantContinuity) {
		t.Fatalf("continuity key = %q, want prompt prefix %q", resolved.continuityKey, wantContinuity)
	}
}

func TestHandlerIgnoresAffinityAfterCredentialIdentityChanges(t *testing.T) {
	handler, manager, _ := newHandlerForTest(t, &scriptedForwarder{}, "sk-one")
	snapshot := manager.Current()
	prefix := []byte(`{"v":1,"user":["hello"]}`)
	oldRef := state.CredentialRef{ID: 1, GroupID: 1, IdentityGeneration: 1}
	initial := handler.resolveRequestAffinity(
		t.Context(), snapshot,
		1,
		protocol.OpenAICompletions,
		"gpt-4o",
		execution.OperationChatCompletion,
		dialect.RequestMetadata{AffinityPrefix: prefix},
		map[uint]state.CredentialRef{1: oldRef},
	)
	if initial.preferredCredentialID != 0 || !initial.key.Valid() {
		t.Fatalf("initial affinity = %#v, want valid miss", initial)
	}
	if !handler.affinityCache.RecordSuccess(
		initial.key,
		initial.observation,
		affinity.Target{GroupID: 1, CredentialID: 1, IdentityGeneration: 1},
	) {
		t.Fatal("RecordSuccess() = false, want cached identity")
	}

	hit := handler.resolveRequestAffinity(
		t.Context(), snapshot,
		1,
		protocol.OpenAICompletions,
		"gpt-4o",
		execution.OperationChatCompletion,
		dialect.RequestMetadata{AffinityPrefix: prefix},
		map[uint]state.CredentialRef{1: oldRef},
	)
	if hit.preferredCredentialID != 1 {
		t.Fatalf("preferred credential = %d, want 1", hit.preferredCredentialID)
	}
	changedRef := oldRef
	changedRef.IdentityGeneration = 2
	stale := handler.resolveRequestAffinity(
		t.Context(), snapshot,
		1,
		protocol.OpenAICompletions,
		"gpt-4o",
		execution.OperationChatCompletion,
		dialect.RequestMetadata{AffinityPrefix: prefix},
		map[uint]state.CredentialRef{1: changedRef},
	)
	if stale.preferredCredentialID != 0 {
		t.Fatalf("preferred credential after identity change = %d, want 0", stale.preferredCredentialID)
	}
	if stale.state != telemetry.AffinityStateTargetUnavailable {
		t.Fatalf("stale state = %q, want %q", stale.state, telemetry.AffinityStateTargetUnavailable)
	}
}

func TestHandlerDerivesPrivateContinuityWithoutReenablingDisabledAffinity(t *testing.T) {
	handler, manager, _ := newHandlerForTest(t, &scriptedForwarder{}, "sk-one")
	snapshot := manager.Current()
	snapshot.Settings.AffinityCapacity = 0
	resolved := handler.resolveRequestAffinity(
		t.Context(), snapshot,
		1,
		protocol.OpenAICompletions,
		"gpt-4o",
		execution.OperationChatCompletion,
		dialect.RequestMetadata{AffinityPrefix: []byte(`{"v":1,"user":["hello"]}`)},
		map[uint]state.CredentialRef{1: {ID: 1, GroupID: 1, IdentityGeneration: 1}},
	)
	if resolved.key.Valid() || resolved.preferredCredentialID != 0 || resolved.continuityKey == "" ||
		resolved.displayKey == "" || resolved.displayKey == resolved.continuityKey {
		t.Fatalf("disabled affinity resolution = %#v", resolved)
	}
	if resolved.state != telemetry.AffinityStateCacheUnavailable {
		t.Fatalf("disabled affinity state = %q, want %q", resolved.state, telemetry.AffinityStateCacheUnavailable)
	}
}

func TestHandlerReportsPromptPrefixAffinitySourceAndState(t *testing.T) {
	forwarder := &scriptedForwarder{results: successfulAffinityResults(2)}
	handler, _, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	useAffinityRandomValues(handler, 0, affinitySecondCredentialRand)

	engine := newAffinityTestEngine(t, handler)
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"stable conversation"}]}`

	serveAffinityRequest(t, engine, body)
	serveAffinityRequest(t, engine, body)

	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one", "sk-one"})
	events := sink.snapshot()
	assertAffinityHits(t, events, []bool{false, true})
	assertAffinityContinuityHits(t, events, []bool{false, false})
	assertAffinitySources(t, events, []telemetry.AffinitySource{
		telemetry.AffinitySourcePromptPrefix, telemetry.AffinitySourcePromptPrefix,
	})
	assertAffinityStates(t, events, []telemetry.AffinityState{
		telemetry.AffinityStateCacheMiss, telemetry.AffinityStateHit,
	})
}

func TestWebsocketBindingOnlyResolvesAffinityButContinuationIsKeyless(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		conn, err := (&websocket.Upgrader{}).Upgrade(writer, request, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		var turns atomic.Int32
		for {
			if _, _, err = conn.ReadMessage(); err != nil {
				return
			}
			if err := conn.WriteMessage(websocket.TextMessage, websocketCompleted(
				fmt.Sprintf("resp_%d", turns.Add(1)), "")); err != nil {
				return
			}
		}
	}))
	defer upstream.Close()
	_, engine, sink := newAffinityWebsocketFixture(t, upstream.URL+"/v1")
	server := httptest.NewServer(engine)
	defer server.Close()

	conn := dialGatewayWebsocket(t, server.URL)
	body := `{"type":"response.create","model":"public","prompt_cache_key":"binding-key"}`
	if err := conn.WriteMessage(websocket.TextMessage, []byte(body)); err != nil {
		t.Fatal(err)
	}
	if _, _, err := conn.ReadMessage(); err != nil {
		t.Fatal(err)
	}
	waitWebsocketLogs(t, sink, 1)

	if err := conn.WriteMessage(websocket.TextMessage, []byte(body)); err != nil {
		t.Fatal(err)
	}
	if _, _, err := conn.ReadMessage(); err != nil {
		t.Fatal(err)
	}
	events := waitWebsocketLogs(t, sink, 2)
	if events[0].AffinityKey == "" || events[0].AffinityKey != events[1].AffinityKey ||
		!affinity.ValidDisplayKey(events[0].AffinityKey) {
		t.Fatalf("binding-only affinity keys = %q / %q, want equal canonical projections", events[0].AffinityKey, events[1].AffinityKey)
	}

	continuation := `{"type":"response.create","model":"public","input":"continue","previous_response_id":"resp_1","store":false}`
	if err := conn.WriteMessage(websocket.TextMessage, []byte(continuation)); err != nil {
		t.Fatal(err)
	}
	if _, _, err := conn.ReadMessage(); err != nil {
		t.Fatal(err)
	}
	events = waitWebsocketLogs(t, sink, 3)
	if events[2].AffinityKey != "" {
		t.Fatalf("continuation affinity key = %q, want empty", events[2].AffinityKey)
	}
}

func TestWebsocketPromptCacheKeyAffinitySeparatesContinuity(t *testing.T) {
	var turns atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		conn, err := (&websocket.Upgrader{}).Upgrade(writer, request, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		for {
			if _, _, err = conn.ReadMessage(); err != nil {
				return
			}
			if conn.WriteMessage(websocket.TextMessage, websocketCompleted(fmt.Sprintf("resp_%d", turns.Add(1)), "")) != nil {
				return
			}
		}
	}))
	defer upstream.Close()
	_, engine, sink := newAffinityWebsocketFixture(t, upstream.URL+"/v1")
	server := httptest.NewServer(engine)
	defer server.Close()

	first := dialGatewayWebsocket(t, server.URL)
	_ = first.WriteMessage(websocket.TextMessage, []byte(`{"type":"response.create","model":"public","prompt_cache_key":"ws-cache-key"}`))
	if _, _, err := first.ReadMessage(); err != nil {
		t.Fatal(err)
	}
	events := waitWebsocketLogs(t, sink, 1)
	assertAffinityHits(t, events, []bool{false})
	assertAffinityStates(t, events, []telemetry.AffinityState{telemetry.AffinityStateCacheMiss})
	assertAffinitySources(t, events, []telemetry.AffinitySource{telemetry.AffinitySourcePromptCacheKey})

	second := dialGatewayWebsocket(t, server.URL)
	_ = second.WriteMessage(websocket.TextMessage, []byte(`{"type":"response.create","model":"public","prompt_cache_key":"ws-cache-key"}`))
	if _, _, err := second.ReadMessage(); err != nil {
		t.Fatal(err)
	}
	events = waitWebsocketLogs(t, sink, 2)
	assertAffinityHits(t, events, []bool{false, true})
	assertAffinityContinuityHits(t, events, []bool{false, false})
	assertAffinityStates(t, events, []telemetry.AffinityState{
		telemetry.AffinityStateCacheMiss, telemetry.AffinityStateHit,
	})
	assertAffinitySources(t, events, []telemetry.AffinitySource{
		telemetry.AffinitySourcePromptCacheKey, telemetry.AffinitySourcePromptCacheKey,
	})
	if events[0].AffinityKey == "" || events[0].AffinityKey != events[1].AffinityKey ||
		!affinity.ValidDisplayKey(events[0].AffinityKey) {
		t.Fatalf("WebSocket reconnect affinity keys = %q / %q, want equal canonical display keys", events[0].AffinityKey, events[1].AffinityKey)
	}

	third := dialGatewayWebsocket(t, server.URL)
	_ = third.WriteMessage(websocket.TextMessage, []byte(`{"type":"response.create","model":"public","input":"continue","previous_response_id":"resp_1","store":false}`))
	if _, _, err := third.ReadMessage(); err != nil {
		t.Fatal(err)
	}
	events = waitWebsocketLogs(t, sink, 3)
	assertAffinityHits(t, events, []bool{false, true, false})
	assertAffinityContinuityHits(t, events, []bool{false, false, true})
	assertAffinitySources(t, events, []telemetry.AffinitySource{
		telemetry.AffinitySourcePromptCacheKey, telemetry.AffinitySourcePromptCacheKey, telemetry.AffinitySourceNone,
	})
}

func newAffinityWebsocketFixture(t *testing.T, endpoint string) (*Handler, *gin.Engine, *recordingRequestLogSink) {
	t.Helper()
	service := encryptiontest.Service(t, "affinity-websocket-key")
	input := state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []state.GroupConfig{
			{
				ID: 1, Name: "ws", ConnectionType: "api_key", ChannelID: channel.OpenAI,
				Params: json.RawMessage(fmt.Sprintf(`{"base_url":%q}`, endpoint)),
				Models: []state.ModelConfig{{ID: "upstream", Alias: "public"}}, Enabled: true,
			},
			{
				ID: 2, Name: "ws-two", ConnectionType: "api_key", ChannelID: channel.OpenAI,
				Params: json.RawMessage(fmt.Sprintf(`{"base_url":%q}`, endpoint)),
				Models: []state.ModelConfig{{ID: "upstream", Alias: "public"}}, Enabled: true,
			},
		},
		Credentials: []state.CredentialConfig{testCredentialConfig(1, 1), testCredentialConfig(2, 2)},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: service.Hash("gl-client"), Status: state.AccessKeyStatusActive,
		}},
	}
	manager := state.NewManager()
	if _, err := manager.Publish(input); err != nil {
		t.Fatal(err)
	}
	registry := state.NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]state.CredentialEntry{
		testCredentialEntry(t, service, 1, 1, "upstream-key-1"),
		testCredentialEntry(t, service, 2, 2, "upstream-key-2"),
	}); err != nil {
		t.Fatal(err)
	}
	handler := NewHandler(
		manager, registry, service, newTestExecutionForwarder(t),
		dialect.NewSet(dialect.NewOpenAIResponses()), health.NewStatsStore(),
		health.NewMutationCoordinator(), nil, nil, nil,
	)
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	return handler, engine, sink
}

func TestResponsesContinuationDoesNotLearnAffinity(t *testing.T) {
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		storedResponse("first"), storedResponse("second"), storedResponse("third"),
	}}
	handler, engine, sink := newContinuationFixture(t, forwarder)
	// 续接请求会创建自己的调度迭代器但不参与亲和，因此随机值按请求顺序
	// 消耗，最后一个普通请求需要落到第二个凭据上。
	useAffinityRandomValues(handler, 0, 0, 1)

	serveContinuation(t, engine, "gl-client", `{"model":"gpt-4o","input":"root-turn","store":true}`, http.StatusOK)
	serveContinuation(t, engine, "gl-client", `{"model":"gpt-4o","input":"continuation-turn","previous_response_id":"first","store":false}`, http.StatusOK)
	serveContinuation(t, engine, "gl-client", `{"model":"gpt-4o","input":"continuation-turn"}`, http.StatusOK)

	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one", "sk-one", "sk-two"})
	events := sink.snapshot()
	assertAffinityHits(t, events, []bool{false, false, false})
	assertAffinityContinuityHits(t, events, []bool{false, true, false})
	assertAffinityStates(t, events, []telemetry.AffinityState{
		telemetry.AffinityStateCacheMiss, telemetry.AffinityStateNone, telemetry.AffinityStateCacheMiss,
	})
}

func successfulAffinityResults(count int) []UpstreamResult {
	results := make([]UpstreamResult, count)
	for index := range results {
		results[index] = successfulAffinityResult()
	}
	return results
}

func successfulAffinityResult() UpstreamResult {
	return UpstreamResult{
		StatusCode:     http.StatusOK,
		Header:         make(http.Header),
		Body:           []byte(`{"ok":true}`),
		RequestWritten: true,
	}
}

func useAffinityRandomValues(handler *Handler, values ...int64) {
	index := 0
	handler.newRandom = func() *rand.Rand {
		value := int64(0)
		if index < len(values) {
			value = values[index]
			index++
		}
		return rand.New(affinityFixedRandSource{value: value})
	}
}

func addAffinityModelRoute(
	t *testing.T,
	snapshot *state.ConfigSnapshot,
	externalModel string,
	groupID uint,
) {
	t.Helper()
	byOperation := snapshot.ExecutionCandidates[protocol.OpenAICompletions]
	if byOperation == nil {
		t.Fatal("OpenAI Completions execution candidates are missing")
	}
	byModel := byOperation[execution.OperationChatCompletion]
	base := byModel["gpt-4o"]
	if len(base) == 0 {
		t.Fatalf("gpt-4o routes = %d, want at least one", len(base))
	}
	var route state.RouteTarget
	for _, candidate := range base {
		if candidate.GroupID == groupID {
			route = candidate
			break
		}
	}
	if route.GroupID == 0 {
		t.Fatalf("gpt-4o has no route for group %d", groupID)
	}
	route.GroupID = groupID
	route.UpstreamModelID = externalModel
	byModel[externalModel] = []state.RouteTarget{route}
}

func moveSecondAffinityCredentialToGroup(
	t *testing.T,
	snapshot *state.ConfigSnapshot,
	registry *state.CredentialRegistry,
) {
	t.Helper()
	group := snapshot.Groups[1]
	group.ID = 2
	group.Name = "openai-two"
	group.Models = []state.ModelConfig{{ID: "gpt-4o-mini"}}
	snapshot.Groups[2] = group
	catalog := snapshot.GroupCatalog[1]
	catalog.ID = 2
	catalog.Name = group.Name
	snapshot.GroupCatalog[2] = catalog
	addAffinityModelRoute(t, snapshot, "gpt-4o-mini", 2)

	refs := registry.CaptureActiveCredentialRefs([]uint{1, 2})
	if len(refs) != 2 {
		t.Fatalf("captured credential refs = %d, want 2", len(refs))
	}
	entries := make([]state.CredentialEntry, 0, len(refs))
	for _, ref := range refs {
		groupID := uint(1)
		if ref.ID == 2 {
			groupID = 2
		}
		entries = append(entries, state.CredentialEntry{
			ID: ref.ID, GroupID: groupID, Version: ref.Version,
			IdentityGeneration: ref.IdentityGeneration, Fingerprint: ref.Fingerprint,
			EncryptedValue: ref.EncryptedValue,
		})
	}
	if err := registry.ReplaceCredentials(entries); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}
}

func serveAffinityModelRequest(t *testing.T, engine http.Handler, model string) {
	t.Helper()
	serveAffinityRequest(
		t,
		engine,
		`{"model":"`+model+`","messages":[{"role":"user","content":"stable conversation"}]}`,
	)
}

func newAffinityTestEngine(t *testing.T, handler *Handler) http.Handler {
	t.Helper()
	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)
	return engine
}

func serveAffinityRequest(t *testing.T, engine http.Handler, body string) {
	t.Helper()
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		bytes.NewBufferString(body),
	)
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("response = %d %s, want 200", response.Code, response.Body.String())
	}
}

func assertAffinityAttemptKeys(t *testing.T, inputs []ForwardInput, want []string) {
	t.Helper()
	got := make([]string, 0, len(inputs))
	for _, input := range inputs {
		got = append(got, input.APIKey)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("attempt keys = %#v, want %#v", got, want)
	}
}

func assertAffinityHits(t *testing.T, events []telemetry.RequestEvent, want []bool) {
	t.Helper()
	got := make([]bool, 0, len(events))
	for _, event := range events {
		got = append(got, event.AffinityHit)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("affinity hits = %#v, want %#v", got, want)
	}
}

func assertAffinityContinuityHits(t *testing.T, events []telemetry.RequestEvent, want []bool) {
	t.Helper()
	got := make([]bool, 0, len(events))
	for _, event := range events {
		got = append(got, event.ContinuityHit)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("continuity hits = %#v, want %#v", got, want)
	}
}

func assertAffinitySources(t *testing.T, events []telemetry.RequestEvent, want []telemetry.AffinitySource) {
	t.Helper()
	got := make([]telemetry.AffinitySource, 0, len(events))
	for _, event := range events {
		got = append(got, event.AffinitySource)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("affinity sources = %#v, want %#v", got, want)
	}
}

func assertAffinityStates(t *testing.T, events []telemetry.RequestEvent, want []telemetry.AffinityState) {
	t.Helper()
	got := make([]telemetry.AffinityState, 0, len(events))
	for _, event := range events {
		got = append(got, event.AffinityState)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("affinity states = %#v, want %#v", got, want)
	}
}

func assertAffinityUpstreamModels(t *testing.T, inputs []ForwardInput, want []string) {
	t.Helper()
	got := make([]string, 0, len(inputs))
	for _, input := range inputs {
		got = append(got, input.UpstreamModelID)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("upstream models = %#v, want %#v", got, want)
	}
}

func affinityUpstreamFailure() UpstreamResult {
	return UpstreamResult{
		StatusCode:         http.StatusUnauthorized,
		Header:             make(http.Header),
		Body:               []byte(`{"error":"invalid key"}`),
		ClassificationBody: []byte(`{"error":"invalid key"}`),
		RequestWritten:     true,
	}
}

// affinityRuntimeInput rebuilds the runtime configuration that
// newHandlerForTest publishes for the two-credential affinity fixture. Tests
// republish it with an unrelated change so the resulting revision bump is the
// only variable under test.
func affinityRuntimeInput(
	handler *Handler,
	settings config.Settings,
	extraGroups ...state.GroupConfig,
) state.CompileInput {
	groups := []state.GroupConfig{
		affinityFixtureGroup(1, "gpt-4o"),
		affinityFixtureGroup(2, "gpt-4o"),
	}
	groups = append(groups, extraGroups...)
	return state.CompileInput{
		SystemSettings:  settings,
		ChannelRegistry: channel.NewRegistry(),
		Groups:          groups,
		Credentials: []state.CredentialConfig{
			{ID: 1, GroupID: 1, Version: 1, IdentityGeneration: 1, Fingerprint: "credential-1"},
			{ID: 2, GroupID: 2, Version: 1, IdentityGeneration: 2, Fingerprint: "credential-2"},
		},
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: handler.encryption.Hash("gl-client"),
			Status: state.AccessKeyStatusActive,
		}},
	}
}

func affinityFixtureGroup(id uint, model string) state.GroupConfig {
	return state.GroupConfig{
		ConnectionType: "api_key", ID: id, Name: "openai", ChannelID: channel.OpenAI,
		Params: json.RawMessage(`{}`), Enabled: true,
		Models: []state.ModelConfig{{ID: model, EntryID: "e000000000001"}},
	}
}

// TestAffinityBindingSurvivesUnrelatedSnapshotChanges proves R3/R7 and R4: an
// established binding keeps preferring its supplier across a revision bump that
// only touches an unrelated group (R3/R7) or publishes an unchanged
// configuration (R4). Each request line asserts both the actual upstream
// attempts and the persisted affinity_state sequence, which regression-tests
// the request_logs observability contract. The memory-only cases stay sensitive
// because a bare miss with the second random value selects the second
// credential. The store-backed case covers R5b: an affinity capacity change
// evicts only the hot copy, so the binding is recovered through durable
// read-through and is never reported as a final cache_miss.
func TestAffinityBindingSurvivesUnrelatedSnapshotChanges(t *testing.T) {
	baseSettings := config.Settings{state.SettingRetryCount: testDefaultRetryBudget}
	tests := []struct {
		name        string
		settings    config.Settings
		extraGroups []state.GroupConfig
		withStore   bool
		wantKeys    []string
		wantHits    []bool
		wantStates  []telemetry.AffinityState
	}{
		{
			name:     "unrelated group added (R3/R7)",
			settings: baseSettings,
			extraGroups: []state.GroupConfig{
				affinityFixtureGroup(3, "unrelated-model"),
			},
			wantKeys: []string{"sk-one", "sk-one", "sk-one"},
			wantHits: []bool{false, true, true},
			wantStates: []telemetry.AffinityState{
				telemetry.AffinityStateCacheMiss,
				telemetry.AffinityStateHit,
				telemetry.AffinityStateHit,
			},
		},
		{
			name:     "revision only publish (R4)",
			settings: baseSettings,
			wantKeys: []string{"sk-one", "sk-one", "sk-one"},
			wantHits: []bool{false, true, true},
			wantStates: []telemetry.AffinityState{
				telemetry.AffinityStateCacheMiss,
				telemetry.AffinityStateHit,
				telemetry.AffinityStateHit,
			},
		},
		{
			name: "affinity capacity change keeps durable binding (R5b)",
			settings: config.Settings{
				state.SettingRetryCount:       testDefaultRetryBudget,
				state.SettingAffinityCapacity: 1,
			},
			withStore: true,
			wantKeys:  []string{"sk-one", "sk-one", "sk-one"},
			wantHits:  []bool{false, true, true},
			wantStates: []telemetry.AffinityState{
				telemetry.AffinityStateCacheMiss,
				telemetry.AffinityStateHit,
				telemetry.AffinityStateHit,
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			forwarder := &scriptedForwarder{results: successfulAffinityResults(3)}
			handler, manager, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
			if test.withStore {
				handler.affinityStore = &affinityStoreStub{}
			}
			sink := &recordingRequestLogSink{}
			handler.requestLogSink = sink
			// A bare miss with these values selects the second credential, so the
			// expected hits only hold while the learned binding survives.
			useAffinityRandomValues(handler, 0, 1, 1)
			engine := newAffinityTestEngine(t, handler)
			body := `{"model":"gpt-4o","messages":[{"role":"user","content":"stable conversation"}]}`

			serveAffinityRequest(t, engine, body)
			serveAffinityRequest(t, engine, body)
			revisionBefore := manager.Current().Revision

			next, err := manager.Publish(
				affinityRuntimeInput(handler, test.settings, test.extraGroups...),
			)
			if err != nil {
				t.Fatalf("Publish() error = %v", err)
			}
			if next.Revision <= revisionBefore {
				t.Fatalf("revision after publish = %d, want > %d", next.Revision, revisionBefore)
			}

			serveAffinityRequest(t, engine, body)

			assertAffinityAttemptKeys(t, forwarder.inputs, test.wantKeys)
			events := sink.snapshot()
			assertAffinityHits(t, events, test.wantHits)
			assertAffinityStates(t, events, test.wantStates)
		})
	}
}

// TestAffinityHitIsAttemptedBeforeFallbackAfterUpstreamFailure proves R1/R2:
// the request actually attempts the bound supplier first, and only after that
// supplier returns a real failure does the same request fall back to another
// supplier.
func TestAffinityHitIsAttemptedBeforeFallbackAfterUpstreamFailure(t *testing.T) {
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		successfulAffinityResult(),
		affinityUpstreamFailure(),
		successfulAffinityResult(),
	}}
	handler, _, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	// Request 1 learns credential 1. Request 2 must attempt credential 1 first
	// through affinity, and only then fall back to credential 2.
	useAffinityRandomValues(handler, 0, 1)
	engine := newAffinityTestEngine(t, handler)
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"stable conversation"}]}`

	serveAffinityRequest(t, engine, body)
	serveAffinityRequest(t, engine, body)

	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one", "sk-one", "sk-two"})
	events := sink.snapshot()
	assertAffinityHits(t, events, []bool{false, true})
	assertAffinityStates(t, events, []telemetry.AffinityState{
		telemetry.AffinityStateCacheMiss,
		telemetry.AffinityStateHit,
	})
	attempts := events[1].Attempts
	if len(attempts) != 2 || attempts[0].CredentialID != 1 || attempts[1].CredentialID != 2 {
		t.Fatalf("second request attempts = %#v, want credential 1 then credential 2", attempts)
	}
	if attempts[0].WillRetry != true {
		t.Fatalf("failed bound attempt WillRetry = %v, want true", attempts[0].WillRetry)
	}
}
