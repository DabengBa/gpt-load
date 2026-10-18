package gateway

import (
	"bytes"
	"context"
	"fmt"
	"github.com/gorilla/websocket"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/telemetry"
)

func TestScheduleCapacityAffinityOverflowAndStreamRelease(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "http", true: "stream"}[stream], func(t *testing.T) {
			forwarder := &scriptedForwarder{results: successfulAffinityResults(2), streamResults: []UpstreamResult{performanceTestStreamResult(health.Feedback{})}}
			h, _, _ := newHandlerForTest(t, forwarder, "sk-one")
			store := &affinityStoreStub{}
			h.affinityStore = store
			group := affinityBucketGroup(1, channel.OpenAI, "gpt-4o", 1, 1)
			group.Models[0].ConcurrencyLimit = 1
			input := affinityBucketRuntimeInput(h, config.Settings{}, group)
			input.Credentials = input.Credentials[:1]
			if _, err := h.manager.Publish(input); err != nil {
				t.Fatal(err)
			}
			engine := newAffinityTestEngine(t, h)
			body := `{"model":"gpt-4o","messages":[{"role":"user","content":"stable capacity affinity"}]}`
			serveAffinityBucketRequest(t, engine, "chat", body)
			key := state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}
			occupied, ok := h.manager.AcquireCapacity(key, false)
			if !ok {
				t.Fatal("occupy")
			}
			defer occupied.Release()
			assertOccupancy := func() {
				if h.manager.CapacityInUse(key) != 2 {
					t.Errorf("affinity occupancy=%d, want 2", h.manager.CapacityInUse(key))
				}
			}
			forwarder.onCall = func(int) { assertOccupancy() }
			forwarder.onStreamCall = func(_ int, _ http.ResponseWriter) { assertOccupancy() }
			if stream {
				body = strings.TrimSuffix(body, "}") + `,"stream":true}`
			}
			serveAffinityBucketRequest(t, engine, "chat", body)
			if h.manager.CapacityInUse(key) != 1 {
				t.Fatal("execution leaked or released another reservation")
			}
			if _, ok := h.manager.AcquireCapacity(key, false); ok {
				t.Fatal("ordinary request admitted during overflow")
			}
			assertAffinityStoreTargetCredential(t, store, 1, 1)
		})
	}
}

func TestScheduleCapacityWebsocketIdleAndFixedIdentity(t *testing.T) {
	t.Run("fixed_without_affinity", func(t *testing.T) { testScheduleCapacityWebsocket(t, false) })
	t.Run("valid_affinity_overflow", func(t *testing.T) { testScheduleCapacityWebsocket(t, true) })
}

func testScheduleCapacityWebsocket(t *testing.T, affinityEnabled bool) {
	t.Helper()
	h, engine, input := websocketTestHandler(t, "http://127.0.0.1:1", channel.CLIProxyAPI)
	input.Groups[0].Models[0].ConcurrencyLimit = 1
	input.Groups[0].Models[0].EntryID = "e000000000001"
	input.SystemSettings = config.Settings{state.SettingAffinityEnabled: affinityEnabled}
	if _, err := h.manager.Publish(input); err != nil {
		t.Fatal(err)
	}
	key := state.RouteEntryKey{GroupID: 1, EntryID: h.manager.Current().Groups[1].Models[0].EntryID}
	sink := &recordingRequestLogSink{}
	h.requestLogSink = sink
	var turns atomic.Int32
	h.forwarder = websocketScriptForwarder{AttemptForwarder: h.forwarder, open: func(context.Context, ForwardInput) (execution.WebsocketSession, execution.WebsocketResult) {
		session := &websocketScriptSession{done: make(chan struct{})}
		session.turn = func(ctx context.Context, body []byte, emit func(context.Context, []byte) error) execution.WebsocketResult {
			n := turns.Add(1)
			want := 1
			if affinityEnabled && n == 2 {
				want = 2
			}
			if h.manager.CapacityInUse(key) != want {
				t.Error("active turn must occupy slot")
			}
			_ = emit(ctx, websocketCompleted(fmt.Sprintf("resp_capacity_%d", n), ""))
			return execution.WebsocketResult{DispatchState: execution.DispatchMaybeSent}
		}
		return session, execution.WebsocketResult{DispatchState: execution.DispatchNotSent}
	}}
	server := httptest.NewServer(engine)
	defer server.Close()
	conn := dialGatewayWebsocket(t, server.URL)
	if h.manager.CapacityInUse(key) != 0 {
		t.Fatal("idle WS occupied slot")
	}
	body := []byte(`{"type":"response.create","model":"public","input":"hello","store":false}`)
	_ = conn.WriteMessage(websocket.TextMessage, body)
	_, _, err := conn.ReadMessage()
	if err != nil {
		t.Fatal(err)
	}
	waitWebsocketLogs(t, sink, 1)
	if h.manager.CapacityInUse(key) != 0 {
		t.Fatal("completed turn leaked slot")
	}
	occupied, ok := h.manager.AcquireCapacity(key, false)
	if !ok {
		t.Fatal("occupy fixed entry")
	}
	defer occupied.Release()
	_ = conn.WriteMessage(websocket.TextMessage, body)
	_, event, err := conn.ReadMessage()
	if err != nil {
		t.Fatal(err)
	}
	if !affinityEnabled && !strings.Contains(string(event), "schedule_capacity_exhausted") {
		t.Fatalf("event=%s", event)
	}
	events := waitWebsocketLogs(t, sink, 2)
	if !affinityEnabled && (turns.Load() != 1 || len(events[1].Attempts) != 0) {
		t.Fatal("fixed identity capacity denial dispatched")
	}
	if affinityEnabled && (turns.Load() != 2 || len(events[1].Attempts) != 1 || !strings.Contains(string(event), "response.completed")) {
		t.Fatalf("affinity overflow failed: turns=%d event=%s", turns.Load(), event)
	}
	if h.manager.CapacityInUse(key) != 1 {
		t.Fatal("WS completion leaked reservation")
	}
}

func TestScheduleCapacityAfterProviderFailurePreservesEvidence(t *testing.T) {
	forwarder := &scriptedForwarder{results: []UpstreamResult{{StatusCode: 401, Header: make(http.Header), Body: []byte(`{"error":"invalid_api_key"}

}`), ClassificationBody: []byte(`{"error":"invalid_api_key"}`), RequestWritten: true}}}
	h, _, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
	sink := &recordingRequestLogSink{}
	h.requestLogSink = sink
	groups := []state.GroupConfig{affinityBucketGroup(1, channel.OpenAI, "gpt-4o", 1, 1), affinityBucketGroup(2, channel.OpenAI, "gpt-4o", 2, 1)}
	for i := range groups {
		groups[i].Models[0].ConcurrencyLimit = 1
	}
	publishAffinityBucketConfig(t, h, config.Settings{state.SettingRetryCount: 2, state.SettingAffinityEnabled: false}, groups...)
	occupied, ok := h.manager.AcquireCapacity(state.RouteEntryKey{GroupID: 2, EntryID: "e000000000001"}, false)
	if !ok {
		t.Fatal("occupy")
	}
	defer occupied.Release()
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o"}`))
	request.Header.Set("Authorization", "Bearer gl-client")
	response := httptest.NewRecorder()
	newAffinityTestEngine(t, h).ServeHTTP(response, request)
	events := sink.snapshot()
	if response.Code != 401 || len(events) != 1 || len(events[0].Attempts) != 1 || len(forwarder.inputs) != 1 {
		t.Fatalf("status=%d events=%+v", response.Code, events)
	}
	if !strings.Contains(events[0].ErrorSummary, "schedule_capacity_exhausted") {
		t.Fatalf("missing separate capacity evidence: %+v", events[0])
	}
}

func TestScheduleCapacityFailureLogNamesCapacityNotNoCandidate(t *testing.T) {
	for _, maxAttempts := range []int{1, 3} {
		t.Run(fmt.Sprintf("max_attempts=%d", maxAttempts), func(t *testing.T) {
			forwarder := &scriptedForwarder{results: successfulAffinityResults(maxAttempts)}
			handler, _, _ := newHandlerForTest(t, forwarder, "sk-one")
			sink := &recordingRequestLogSink{}
			handler.requestLogSink = sink
			group := affinityBucketGroup(1, channel.OpenAI, "gpt-4o", 1, 1)
			group.Models[0].ConcurrencyLimit = 1
			input := affinityBucketRuntimeInput(handler, config.Settings{state.SettingRetryCount: maxAttempts}, group)
			input.Credentials = input.Credentials[:1]
			if _, err := handler.manager.Publish(input); err != nil {
				t.Fatal(err)
			}
			key := state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}
			occupied, ok := handler.manager.AcquireCapacity(key, false)
			if !ok {
				t.Fatal("occupy")
			}
			t.Cleanup(occupied.Release)
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o","messages":[{"role":"user","content":"capacity"}]}`))
			request.Header.Set("Authorization", "Bearer gl-client")
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			newAffinityTestEngine(t, handler).ServeHTTP(response, request)
			events := sink.snapshot()
			if response.Code != 503 || len(events) != 1 || len(forwarder.inputs) != 0 {
				t.Fatalf("status=%d events=%+v forwards=%d", response.Code, events, len(forwarder.inputs))
			}
			event := events[0]
			if event.ErrorCode != "schedule_capacity_exhausted" ||
				!strings.Contains(event.ErrorSummary, "All eligible route entries are at capacity.") ||
				strings.Contains(event.ErrorSummary, "remaining eligible") {
				t.Fatalf("capacity denial logged as %q %q", event.ErrorCode, event.ErrorSummary)
			}
			if len(event.Attempts) != 0 {
				t.Fatalf("capacity denial created attempts: %+v", event.Attempts)
			}
			if handler.manager.CapacityInUse(key) != 1 {
				t.Fatal("denied request changed occupancy")
			}
			if !strings.Contains(response.Body.String(), "schedule_capacity_exhausted") {
				t.Fatalf("body=%s", response.Body.String())
			}
		})
	}
}

func TestScheduleCapacityHTTPAdmission(t *testing.T) {
	for _, allFull := range []bool{false, true} {
		t.Run(map[bool]string{false: "fallback", true: "exhausted"}[allFull], func(t *testing.T) {
			forwarder := &scriptedForwarder{results: successfulAffinityResults(1)}
			handler, _, _ := newHandlerForTest(t, forwarder, "sk-one", "sk-two")
			sink := &recordingRequestLogSink{}
			handler.requestLogSink = sink
			useAffinityRandomValues(handler, 0)
			groups := []state.GroupConfig{affinityBucketGroup(1, channel.OpenAI, "gpt-4o", 1, 1), affinityBucketGroup(2, channel.OpenAI, "gpt-4o", 2, 1)}
			for i := range groups {
				groups[i].Models[0].ConcurrencyLimit = 1
			}
			publishAffinityBucketConfig(t, handler, config.Settings{state.SettingRetryCount: 0}, groups...)
			first, ok := handler.manager.AcquireCapacity(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000001"}, false)
			if !ok {
				t.Fatal("occupy first")
			}
			t.Cleanup(first.Release)
			if allFull {
				second, ok := handler.manager.AcquireCapacity(state.RouteEntryKey{GroupID: 2, EntryID: "e000000000001"}, false)
				if !ok {
					t.Fatal("occupy second")
				}
				t.Cleanup(second.Release)
			}
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"gpt-4o","messages":[{"role":"user","content":"capacity"}]}`))
			request.Header.Set("Authorization", "Bearer gl-client")
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			newAffinityTestEngine(t, handler).ServeHTTP(response, request)
			events := sink.snapshot()
			if len(events) != 1 {
				t.Fatalf("events=%d", len(events))
			}
			if allFull {
				if response.Code != 503 || !strings.Contains(response.Body.String(), "schedule_capacity_exhausted") {
					t.Fatalf("response=%d %s", response.Code, response.Body.String())
				}
				if len(forwarder.inputs) != 0 || len(events[0].Attempts) != 0 {
					t.Fatal("capacity denial created provider attempt")
				}
			} else {
				if response.Code != 200 {
					t.Fatalf("response=%d %s", response.Code, response.Body.String())
				}
				assertAffinityInputCredentials(t, forwarder.inputs, []uint{2})
				if len(events[0].Attempts) != 1 {
					t.Fatalf("attempts=%d, want 1", len(events[0].Attempts))
				}
				if handler.manager.CapacityInUse(state.RouteEntryKey{GroupID: 2, EntryID: "e000000000001"}) != 0 {
					t.Fatal("HTTP slot leaked")
				}
			}
		})
	}
}

func TestScheduleCapacitySensitiveSummaryUsesLocalMarker(t *testing.T) {
	withAttempt := func(summary string, blocked bool) string {
		recorder := &requestRecorder{
			protocol: protocol.OpenAIEmbeddings,
			outcome:  requestOutcome{errorCode: "upstream_error", errorSummary: summary},
			attempts: []telemetry.Attempt{{ErrorCode: "upstream_error", ErrorSummary: summary}},
		}
		if blocked {
			recordCapacityBlocked(recorder)
		}
		recorder.freezeSensitiveInputErrorSummaries()
		return recorder.outcome.errorSummary
	}
	if got := withAttempt("invalid input: schedule_capacity_exhausted", false); strings.Contains(got, "remaining eligible") {
		t.Fatalf("upstream text created local capacity evidence: %q", got)
	}
	if got := withAttempt("invalid input", true); !strings.Contains(got, "remaining eligible") {
		t.Fatalf("local capacity evidence was lost: %q", got)
	}
}
