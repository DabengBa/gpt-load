package gateway

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

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
		streamGatewayGroup{id: 1, name: "timeout-a", upstreamURL: upstream.URL, apiKey: "sk-a", firstByte: 30 * time.Millisecond, bufferedStream: true},
		streamGatewayGroup{id: 2, name: "timeout-b", upstreamURL: upstream.URL, apiKey: "sk-b", firstByte: 200 * time.Millisecond, bufferedStream: true},
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
		streamGatewayGroup{id: 1, name: "buffered-a", upstreamURL: upstream.URL, apiKey: "sk-a", bufferedStream: true},
		streamGatewayGroup{id: 2, name: "buffered-b", upstreamURL: upstream.URL, apiKey: "sk-b", bufferedStream: true},
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
