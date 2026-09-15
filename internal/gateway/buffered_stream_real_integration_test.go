package gateway

import (
	"bufio"
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/protocol"
)

func TestBufferedStreamNativeResponsesUnknownFailureRetriesWithoutLeak(t *testing.T) {
	const failedSSE = "event: response.created\n" +
		"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_failed\",\"object\":\"response\",\"status\":\"in_progress\",\"model\":\"public-model\",\"output\":[]}}\n\n" +
		"event: response.failed\n" +
		"data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_failed\",\"object\":\"response\",\"status\":\"failed\",\"error\":{\"code\":\"provider_unknown_code\",\"message\":\"first-candidate-secret-failure\"}}}\n\n"
	const successSSE = "event: response.created\n" +
		"data: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_second\",\"object\":\"response\",\"created_at\":123,\"status\":\"in_progress\",\"model\":\"public-model\",\"output\":[]}}\n\n" +
		"event: response.output_text.delta\n" +
		"data: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"output_index\":0,\"content_index\":0,\"item_id\":\"msg_second\",\"delta\":\"second-candidate-payload\"}\n\n" +
		"event: response.completed\n" +
		"data: {\"type\":\"response.completed\",\"sequence_number\":2,\"response\":{\"id\":\"resp_second\",\"object\":\"response\",\"created_at\":123,\"status\":\"completed\",\"model\":\"public-model\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n"

	var firstCalls atomic.Int32
	first := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		firstCalls.Add(1)
		writer.Header().Set("Content-Type", "text/event-stream")
		writer.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(writer, failedSSE)
		writer.(http.Flusher).Flush()
	}))
	t.Cleanup(first.Close)

	var secondCalls atomic.Int32
	second := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		secondCalls.Add(1)
		writer.Header().Set("Content-Type", "text/event-stream")
		writer.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(writer, successSSE)
		writer.(http.Flusher).Flush()
	}))
	t.Cleanup(second.Close)

	_, firstParams := testChannelConfig(
		t,
		protocol.OpenAIResponses,
		testUpstreamBaseURL(first.URL, protocol.OpenAIResponses),
	)
	_, secondParams := testChannelConfig(
		t,
		protocol.OpenAIResponses,
		testUpstreamBaseURL(second.URL, protocol.OpenAIResponses),
	)
	engine, _ := newDialectGatewayEngine(
		t,
		protocol.OpenAIResponses,
		"public-model",
		dialect.NewSet(dialect.NewOpenAIResponses()),
		dialectGatewayGroup{
			id: 1, name: "responses-first", upstreamURL: first.URL,
			channelID: channel.OpenAI, params: firstParams, apiKeys: []string{"sk-first"},
		},
		dialectGatewayGroup{
			id: 2, name: "responses-second", upstreamURL: second.URL,
			channelID: channel.OpenAI, params: secondParams, apiKeys: []string{"sk-second"},
		},
	)
	gateway := httptest.NewServer(engine)
	t.Cleanup(gateway.Close)

	request, err := http.NewRequest(
		http.MethodPost,
		gateway.URL+"/v1/responses",
		strings.NewReader(`{"model":"public-model","input":"hello","stream":true,"store":false}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer gl-client")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(response.Body)
		t.Fatalf("gateway status = %d body=%q", response.StatusCode, body)
	}
	reader := bufio.NewReader(response.Body)
	if line, err := reader.ReadString('\n'); err != nil || line != ": keep-alive\n" {
		t.Fatalf("first response line = %q, %v; want buffered heartbeat", line, err)
	}
	if line, err := reader.ReadString('\n'); err != nil || line != "\n" {
		t.Fatalf("heartbeat boundary = %q, %v", line, err)
	}
	body, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}

	if firstCalls.Load() != 1 || secondCalls.Load() != 1 {
		t.Fatalf("upstream calls = %d/%d, want first failure plus second candidate", firstCalls.Load(), secondCalls.Load())
	}
	if !bytes.Contains(body, []byte("second-candidate-payload")) {
		t.Fatalf("upstream calls = %d/%d body = %q, want second candidate payload", firstCalls.Load(), secondCalls.Load(), body)
	}
	for _, leaked := range []string{"provider_unknown_code", "first-candidate-secret-failure", "resp_failed"} {
		if bytes.Contains(body, []byte(leaked)) {
			t.Fatalf("failed candidate payload leaked %q: body=%q", leaked, body)
		}
	}
}

func TestBufferedStreamRetriesFirstResponseTimeoutBeforePayloadRelease(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		call := calls.Add(1)
		if call == 1 {
			select {
			case <-request.Context().Done():
			case <-time.After(150 * time.Millisecond):
			}
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := writer.(http.Flusher)
		if !ok {
			t.Fatal("upstream does not support flushing")
		}
		_, _ = io.WriteString(writer, "data: {\"id\":\"complete\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n")
		_, _ = io.WriteString(writer, "data: {\"id\":\"complete\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = io.WriteString(writer, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer upstream.Close()

	engine, _ := newStreamingGatewayEngine(t,
		streamGatewayGroup{id: 1, name: "timeout-a", upstreamURL: upstream.URL, apiKey: "sk-a", firstByte: 30 * time.Millisecond},
		streamGatewayGroup{id: 2, name: "timeout-b", upstreamURL: upstream.URL, apiKey: "sk-b", firstByte: 200 * time.Millisecond},
	)
	gateway := httptest.NewServer(engine)
	defer gateway.Close()

	request, err := http.NewRequest(http.MethodPost, gateway.URL+"/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer gl-client")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK {
		t.Fatalf("gateway status = %d body=%q", response.StatusCode, body)
	}
	if calls.Load() != 2 {
		t.Fatalf("upstream calls = %d, want timeout candidate plus backup", calls.Load())
	}
	if !bytes.Contains(body, []byte("complete")) || bytes.Contains(body, []byte("timeout")) {
		t.Fatalf("body = %q, want heartbeat and validated backup attempt", body)
	}
}

func TestBufferedStreamRealHTTPRetriesBeforePayloadRelease(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		call := calls.Add(1)
		writer.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := writer.(http.Flusher)
		if !ok {
			t.Fatal("upstream does not support flushing")
		}
		if call == 1 {
			_, _ = io.WriteString(writer, "data: {\"id\":\"partial\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"leak\"},\"finish_reason\":null}]}\n\n")
			flusher.Flush()
			return
		}
		_, _ = io.WriteString(writer, "data: {\"id\":\"complete\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n")
		_, _ = io.WriteString(writer, "data: {\"id\":\"complete\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = io.WriteString(writer, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer upstream.Close()

	engine, _ := newStreamingGatewayEngine(t,
		streamGatewayGroup{id: 1, name: "buffered-a", upstreamURL: upstream.URL, apiKey: "sk-a"},
		streamGatewayGroup{id: 2, name: "buffered-b", upstreamURL: upstream.URL, apiKey: "sk-b"},
	)
	gateway := httptest.NewServer(engine)
	defer gateway.Close()

	request, err := http.NewRequest(http.MethodPost, gateway.URL+"/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer gl-client")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK {
		t.Fatalf("gateway status = %d body=%q", response.StatusCode, body)
	}
	if calls.Load() != 2 {
		t.Fatalf("upstream calls = %d, want two candidates", calls.Load())
	}
	if bytes.Contains(body, []byte("partial")) || bytes.Contains(body, []byte("leak")) {
		t.Fatalf("failed attempt payload leaked: %q", body)
	}
	if !bytes.Contains(body, []byte("complete")) || !bytes.Contains(body, []byte(": keep-alive")) {
		t.Fatalf("body = %q, want heartbeat and validated second attempt", body)
	}
	if got := response.Header.Get("Cache-Control"); got != "no-cache, no-transform" {
		t.Fatalf("Cache-Control = %q", got)
	}
	if got := response.Header.Get("X-Accel-Buffering"); got != "no" {
		t.Fatalf("X-Accel-Buffering = %q", got)
	}
}
