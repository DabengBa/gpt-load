package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/testutil/encryptiontest"
	"gpt-load/internal/testutil/fakeupstream"
)

func TestHandlerStreamsFakeUpstreamAndRetriesBeforeCommit(t *testing.T) {
	t.Run("valid fixture", func(t *testing.T) {
		upstream := newRecordedStreamUpstream(t, forcedBufferedOpenAIStream(), nil)

		engine, _ := newStreamingGatewayEngine(t, streamGatewayGroup{
			id: 1, name: "openai", upstreamURL: upstream.URL, apiKey: "sk-stream-one",
		})
		recorder := performStreamingRequest(engine)

		want := append([]byte(bufferedStreamHeartbeat), forcedBufferedOpenAIStream()...)
		if recorder.Code != http.StatusOK || !bytes.Equal(recorder.Body.Bytes(), want) || !recorder.Flushed {
			t.Fatalf("response = %d flushed=%t body=%q, want heartbeat plus released fixture", recorder.Code, recorder.Flushed, recorder.Body.Bytes())
		}
		requests := upstream.Requests()
		if len(requests) != 1 {
			t.Fatalf("upstream requests = %d, want 1", len(requests))
		}
		if got := requests[0].Get("Accept-Encoding"); got != "identity" {
			t.Fatalf("Accept-Encoding = %q, want identity", got)
		}
		if got := requests[0].Get("Authorization"); got != "Bearer sk-stream-one" {
			t.Fatalf("Authorization = %q", got)
		}
	})

	t.Run("retryable response then valid fixture", func(t *testing.T) {
		rejecting := fakeupstream.New(fakeupstream.Step{Status: http.StatusTooManyRequests, Fixture: "openai/429.json"})
		defer rejecting.Close()
		backup := newRecordedStreamUpstream(t, forcedBufferedOpenAIStream(), nil)

		engine, _ := newStreamingGatewayEngine(t,
			streamGatewayGroup{id: 1, name: "openai", upstreamURL: rejecting.URL, apiKey: "sk-stream-one"},
			streamGatewayGroup{id: 2, name: "openai-backup", upstreamURL: backup.URL, apiKey: "sk-stream-two"},
		)
		recorder := performStreamingRequest(engine)

		want := append([]byte(bufferedStreamHeartbeat), forcedBufferedOpenAIStream()...)
		if recorder.Code != http.StatusOK || !bytes.Equal(recorder.Body.Bytes(), want) {
			t.Fatalf("response = %d %q", recorder.Code, recorder.Body.Bytes())
		}
		if len(rejecting.Requests()) != 1 || len(backup.Requests()) != 1 {
			t.Fatalf("upstream requests = %d/%d, want 1/1", len(rejecting.Requests()), len(backup.Requests()))
		}
		if first, second := rejecting.Requests()[0].Headers.Get("Authorization"), backup.Requests()[0].Get("Authorization"); first == second {
			t.Fatalf("retry reused credential %q", first)
		}
	})
}

func TestHandlerTerminatesAliasedNonObjectProviderErrorWithoutReplay(t *testing.T) {
	var requests atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		writer.Header().Set("Content-Type", "text/event-stream")
		writer.Header().Set("Retry-After", "1")
		_, _ = io.WriteString(
			writer,
			"event: error\ndata: rate_limit_error provider-model\n\n",
		)
	}))
	defer upstream.Close()

	engine, _ := newStreamingGatewayEngine(
		t,
		streamGatewayGroup{
			id: 1, name: "first", upstreamURL: upstream.URL,
			apiKey: "sk-obviously-fake-first", modelID: "provider-model", alias: "gpt-4o",
		},
		streamGatewayGroup{
			id: 2, name: "second", upstreamURL: upstream.URL,
			apiKey: "sk-obviously-fake-second", modelID: "provider-model", alias: "gpt-4o",
		},
	)
	recorder := performStreamingRequestWithBody(engine, ineligibleStreamRequestBody)

	if recorder.Code != http.StatusOK ||
		recorder.Body.String() != bufferedStreamHeartbeat+bufferedOpenAIStreamFailure ||
		requests.Load() != 1 {
		t.Fatalf("response/requests = %d/%d body=%s",
			recorder.Code, requests.Load(), recorder.Body.String())
	}
	if strings.Contains(recorder.Body.String(), "rate_limit_error") ||
		strings.Contains(recorder.Body.String(), "provider-model") {
		t.Fatalf("provider payload reached downstream: %s", recorder.Body.String())
	}
}

func TestHandlerStreamAlwaysRequestsUpstreamUsage(t *testing.T) {
	upstream := fakeupstream.New(
		fakeupstream.Step{Status: http.StatusTooManyRequests, Fixture: "openai/429.json"},
		fakeupstream.Step{Status: http.StatusOK, Fixture: "openai/stream.sse", Stream: true},
	)
	defer upstream.Close()

	engine, _ := newStreamingGatewayEngine(t,
		streamGatewayGroup{id: 1, name: "first", upstreamURL: upstream.URL, apiKey: "sk-first"},
		streamGatewayGroup{id: 2, name: "second", upstreamURL: upstream.URL, apiKey: "sk-second"},
	)
	recorder := performStreamingRequest(engine)
	if recorder.Code != http.StatusOK {
		t.Fatalf("response = %d body=%q", recorder.Code, recorder.Body.Bytes())
	}
	requests := upstream.Requests()
	if len(requests) != 2 {
		t.Fatalf("upstream requests = %d, want 2", len(requests))
	}
	for index, request := range requests {
		var body map[string]any
		if err := json.Unmarshal(request.Body, &body); err != nil {
			t.Fatalf("decode upstream request %d: %v", index+1, err)
		}
		options, injected := body["stream_options"]
		if !injected {
			t.Fatalf("attempt %d missing stream_options; body=%#v", index+1, body)
		}
		object, ok := options.(map[string]any)
		if !ok || object["include_usage"] != true {
			t.Fatalf("attempt %d stream_options = %#v, want include_usage=true", index+1, options)
		}
	}
}

func TestHandlerStreamingDebugHeadersRejectUpstreamSpoofing(t *testing.T) {
	upstream := fakeupstream.New(fakeupstream.Step{
		Status: http.StatusOK, Fixture: "openai/stream.sse", Stream: true,
		Headers: http.Header{
			"X-GPTLoad-Group":    {"spoofed-group"},
			"X-GPTLoad-Key":      {"sk-spoofed-plaintext"},
			"X-GPTLoad-Attempts": {"999"},
		},
	})
	defer upstream.Close()

	engine, _ := newStreamingGatewayEngine(t, streamGatewayGroup{
		id: 1, name: "stream-group", upstreamURL: upstream.URL, apiKey: "sk-real-stream-key",
	})
	recorder := performStreamingRequest(engine)

	assertDebugHeaders(t, recorder.Header(), "stream-group", "1")
	if strings.Contains(recorder.Body.String(), "sk-real-stream-key") || strings.Contains(recorder.Body.String(), "sk-spoofed-plaintext") {
		t.Fatalf("stream response leaked a plaintext key: %s", recorder.Body.String())
	}
}

func TestHandlerTreatsSDKNormalizedCompressedStreamAsSingleAttempt(t *testing.T) {
	t.Run("normalized response does not blame credential or retry", func(t *testing.T) {
		compressed := newRecordedStreamUpstream(t, forcedBufferedOpenAIStream(), http.Header{"Content-Encoding": {"gzip"}})
		backup := newRecordedStreamUpstream(t, forcedBufferedOpenAIStream(), nil)

		engine, registry := newStreamingGatewayEngine(t,
			streamGatewayGroup{id: 1, name: "compressed", upstreamURL: compressed.URL, apiKey: "sk-compressed"},
			streamGatewayGroup{id: 2, name: "backup", upstreamURL: backup.URL, apiKey: "sk-backup"},
		)
		want := append([]byte(bufferedStreamHeartbeat), forcedBufferedOpenAIStream()...)
		first := performStreamingRequest(engine)
		if first.Code != http.StatusOK || !bytes.Equal(first.Body.Bytes(), want) {
			t.Fatalf("first response = %d %q", first.Code, first.Body.Bytes())
		}
		if len(compressed.Requests()) != 1 || len(backup.Requests()) != 0 {
			t.Fatalf("first request counts = compressed:%d backup:%d", len(compressed.Requests()), len(backup.Requests()))
		}
		if candidates := registry.CollectCredentialCandidates([]uint{1, 2}, nil, time.Time{}); len(candidates) != 2 {
			t.Fatalf("SDK-normalized response changed credential registry: %#v", candidates)
		}

		second := performStreamingRequest(engine)
		if second.Code != http.StatusOK || !bytes.Equal(second.Body.Bytes(), want) || len(compressed.Requests()) != 2 || len(backup.Requests()) != 0 {
			t.Fatalf("second response/counts = %d/%q compressed:%d backup:%d", second.Code, second.Body.Bytes(), len(compressed.Requests()), len(backup.Requests()))
		}
	})

	t.Run("multiple candidates still produce one logical attempt", func(t *testing.T) {
		first := newRecordedStreamUpstream(t, forcedBufferedOpenAIStream(), http.Header{"Content-Encoding": {"gzip"}})
		second := newRecordedStreamUpstream(t, forcedBufferedOpenAIStream(), http.Header{"Content-Encoding": {"br"}})

		engine, _ := newStreamingGatewayEngine(t,
			streamGatewayGroup{id: 1, name: "compressed-a", upstreamURL: first.URL, apiKey: "sk-plain-a"},
			streamGatewayGroup{id: 2, name: "compressed-b", upstreamURL: second.URL, apiKey: "sk-plain-b"},
		)
		recorder := performStreamingRequest(engine)
		want := append([]byte(bufferedStreamHeartbeat), forcedBufferedOpenAIStream()...)
		if recorder.Code != http.StatusOK || !bytes.Equal(recorder.Body.Bytes(), want) ||
			len(first.Requests())+len(second.Requests()) != 1 {
			t.Fatalf("response = %d %s", recorder.Code, recorder.Body.String())
		}
		for _, forbidden := range []string{"sk-plain-a", "sk-plain-b"} {
			if strings.Contains(recorder.Body.String(), forbidden) {
				t.Fatalf("protocol response exposed %q: %s", forbidden, recorder.Body.String())
			}
		}
	})
}

func TestHandlerStreamFirstEventTimeout(t *testing.T) {
	t.Run("request-written partial event times out without backup", func(t *testing.T) {
		var partialCalls atomic.Int64
		release := make(chan struct{})
		partial := newPartialStreamServer(&partialCalls, release)
		defer partial.Close()
		defer close(release)
		backup := newRecordedStreamUpstream(t, forcedBufferedOpenAIStream(), nil)
		defer backup.Close()

		engine, _ := newStreamingGatewayEngine(t,
			streamGatewayGroup{id: 1, name: "partial", upstreamURL: partial.URL, apiKey: "sk-partial", firstByte: 250 * time.Millisecond},
			streamGatewayGroup{id: 2, name: "backup", upstreamURL: backup.URL, apiKey: "sk-backup", firstByte: 200 * time.Millisecond},
		)
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
		defer cancel()
		request := httptest.NewRequestWithContext(ctx, http.MethodPost, "/v1/chat/completions", strings.NewReader(ineligibleStreamRequestBody))
		request.Header.Set("Authorization", "Bearer gl-client")
		recorder := httptest.NewRecorder()
		engine.ServeHTTP(recorder, request)

		if ctx.Err() != nil {
			t.Fatal("first-event timeout did not finish before the test deadline")
		}
		want := bufferedStreamHeartbeat + bufferedOpenAIStreamFailure
		if recorder.Code != http.StatusOK || recorder.Body.String() != want {
			t.Fatalf("response = %d %q, want buffered failure", recorder.Code, recorder.Body.String())
		}
		if partialCalls.Load() != 1 || bytes.Contains(recorder.Body.Bytes(), []byte("partial")) || len(backup.Requests()) != 0 {
			t.Fatalf("partial event/retry contract: body=%q primary=%d backup=%d", recorder.Body.Bytes(), partialCalls.Load(), len(backup.Requests()))
		}
	})

	t.Run("first partial candidate returns timeout without retry", func(t *testing.T) {
		var firstCalls, secondCalls atomic.Int64
		release := make(chan struct{})
		first := newPartialStreamServer(&firstCalls, release)
		defer first.Close()
		second := newPartialStreamServer(&secondCalls, release)
		defer second.Close()
		defer close(release)

		engine, _ := newStreamingGatewayEngine(t,
			streamGatewayGroup{id: 1, name: "partial-a", upstreamURL: first.URL, apiKey: "sk-a", firstByte: 250 * time.Millisecond},
			streamGatewayGroup{id: 2, name: "partial-b", upstreamURL: second.URL, apiKey: "sk-b", firstByte: 250 * time.Millisecond},
		)
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
		defer cancel()
		request := httptest.NewRequestWithContext(ctx, http.MethodPost, "/v1/chat/completions", strings.NewReader(ineligibleStreamRequestBody))
		request.Header.Set("Authorization", "Bearer gl-client")
		recorder := httptest.NewRecorder()
		engine.ServeHTTP(recorder, request)
		if ctx.Err() != nil {
			t.Fatal("first-event timeout did not finish before the test deadline")
		}
		if firstCalls.Load() != 1 || secondCalls.Load() != 0 {
			t.Fatalf("partial upstream attempts = %d/%d, want 1/0", firstCalls.Load(), secondCalls.Load())
		}

		want := bufferedStreamHeartbeat + bufferedOpenAIStreamFailure
		if recorder.Code != http.StatusOK || recorder.Body.String() != want || strings.Contains(recorder.Body.String(), "partial") {
			t.Fatalf("response = %d %s", recorder.Code, recorder.Body.String())
		}
	})
}

func TestHandlerStreamIdleAndDisconnectRetriesBeforeRelease(t *testing.T) {
	const partialEvent = "data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"first\"},\"finish_reason\":null}]}\n\n"
	tests := []struct {
		name       string
		handler    http.HandlerFunc
		idle       time.Duration
		want       string
		wantBackup int
	}{
		{
			name: "idle after commit retries before release",
			idle: 35 * time.Millisecond,
			handler: func(writer http.ResponseWriter, request *http.Request) {
				writer.Header().Set("Content-Type", "text/event-stream")
				_, _ = writer.Write([]byte(partialEvent))
				writer.(http.Flusher).Flush()
				// Bifrost Core v1.7.7 can finish the logical stream without
				// synchronously closing the underlying fasthttp connection.
				// Keep the fixture finite while still giving GPT-Load's idle
				// timeout enough time to stop the downstream stream.
				select {
				case <-request.Context().Done():
				case <-time.After(200 * time.Millisecond):
				}
			},
			want:       bufferedStreamHeartbeat + string(forcedBufferedOpenAIStream()),
			wantBackup: 1,
		},
		{
			name: "abrupt EOF after commit retries before release",
			idle: time.Second,
			handler: func(writer http.ResponseWriter, _ *http.Request) {
				// Close a chunked response without its terminating zero chunk so
				// the complete first SSE event is observable before the transport
				// reports an unexpected EOF. With a mismatched Content-Length,
				// Bifrost Core v1.7.7 rejects the response before exposing data.
				connection, buffered, err := writer.(http.Hijacker).Hijack()
				if err != nil {
					return
				}
				defer connection.Close()
				_, _ = fmt.Fprintf(buffered, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n%x\r\n%s\r\n", len(partialEvent), partialEvent)
				_ = buffered.Flush()
			},
			want:       bufferedStreamHeartbeat + string(forcedBufferedOpenAIStream()),
			wantBackup: 1,
		},
		{
			name: "activity resets idle deadline",
			idle: 120 * time.Millisecond,
			handler: func(writer http.ResponseWriter, request *http.Request) {
				writer.Header().Set("Content-Type", "text/event-stream")
				for _, event := range []string{
					"data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"one\"},\"finish_reason\":null}]}\n\n",
					"data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"two\"},\"finish_reason\":null}]}\n\n",
					"data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
				} {
					_, _ = writer.Write([]byte(event))
					writer.(http.Flusher).Flush()
					select {
					case <-time.After(40 * time.Millisecond):
					case <-request.Context().Done():
						return
					}
				}
			},
			want: bufferedStreamHeartbeat + "data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"one\"},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"two\"},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
			wantBackup: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			primary := httptest.NewServer(tt.handler)
			defer primary.Close()
			backup := newRecordedStreamUpstream(t, forcedBufferedOpenAIStream(), nil)

			engine, _ := newStreamingGatewayEngine(t,
				streamGatewayGroup{id: 1, name: "primary", upstreamURL: primary.URL, apiKey: "sk-primary", streamIdle: tt.idle},
				streamGatewayGroup{id: 2, name: "backup", upstreamURL: backup.URL, apiKey: "sk-backup", streamIdle: time.Second},
			)
			recorder := performStreamingRequest(engine)

			if recorder.Code != http.StatusOK || recorder.Body.String() != tt.want {
				t.Fatalf("response = %d %q, want %q", recorder.Code, recorder.Body.String(), tt.want)
			}
			if len(backup.Requests()) != tt.wantBackup {
				t.Fatalf("backup requests = %d, want %d", len(backup.Requests()), tt.wantBackup)
			}
		})
	}
}

func TestHandlerStopsAfterDownstreamCancellationWithoutRetry(t *testing.T) {
	upstreamStarted := make(chan struct{})
	upstreamFinished := make(chan struct{})
	primary := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, request *http.Request) {
		_, _ = io.Copy(io.Discard, request.Body)
		close(upstreamStarted)
		select {
		case <-request.Context().Done():
		case <-time.After(200 * time.Millisecond):
		}
		close(upstreamFinished)
	}))
	defer primary.Close()
	backup := fakeupstream.New(fakeupstream.Step{
		Status: http.StatusOK, Fixture: "openai/stream.sse", Stream: true,
	})
	defer backup.Close()

	engine, _ := newStreamingGatewayEngine(t,
		streamGatewayGroup{id: 1, name: "primary", upstreamURL: primary.URL, apiKey: "sk-primary"},
		streamGatewayGroup{id: 2, name: "backup", upstreamURL: backup.URL, apiKey: "sk-backup"},
	)
	gatewayServer := httptest.NewServer(engine)
	defer gatewayServer.Close()

	ctx, cancel := context.WithCancel(context.Background())
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, gatewayServer.URL+"/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o","stream":true}`))
	if err != nil {
		t.Fatalf("NewRequestWithContext() error = %v", err)
	}
	request.Header.Set("Authorization", "Bearer gl-client")
	done := make(chan error, 1)
	go func() {
		response, doErr := http.DefaultClient.Do(request)
		if response != nil {
			_ = response.Body.Close()
		}
		done <- doErr
	}()

	waitForStreamSignal(t, upstreamStarted, "primary request start")
	cancel()
	select {
	case err := <-done:
		if err != nil && !errors.Is(err, context.Canceled) {
			t.Fatalf("downstream cancellation request failed: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("downstream request did not stop after cancellation")
	}
	waitForStreamSignal(t, upstreamFinished, "finite upstream completion")
	if len(backup.Requests()) != 0 {
		t.Fatalf("downstream cancellation retried backup %d times", len(backup.Requests()))
	}
}

func TestStreamWriteDeadlineStopsRealTCPSlowReader(t *testing.T) {
	type writeResult struct {
		operation string
		err       error
	}
	done := make(chan writeResult, 1)
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	engine.GET("/", func(ginContext *gin.Context) {
		controlled := newStreamWriteController(ginContext.Writer, 25*time.Millisecond)
		defer func() { _ = controlled.clear() }()
		if err := controlled.writeHeader(http.StatusOK); err != nil {
			done <- writeResult{operation: "write header", err: err}
			return
		}
		chunk := bytes.Repeat([]byte("x"), 1024)
		for {
			if _, err := controlled.write(chunk); err != nil {
				done <- writeResult{operation: "write", err: err}
				return
			}
			if err := controlled.flush(); err != nil {
				done <- writeResult{operation: "flush", err: err}
				return
			}
		}
	})

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := &http.Server{Handler: engine}
	serveDone := make(chan error, 1)
	go func() {
		serveDone <- server.Serve(&smallWriteBufferListener{Listener: listener})
	}()

	var client net.Conn
	var responseBody io.ReadCloser
	t.Cleanup(func() {
		if client != nil {
			_ = client.Close()
		}
		if responseBody != nil {
			_ = responseBody.Close()
		}
		_ = server.Close()
		_ = listener.Close()
		select {
		case serveErr := <-serveDone:
			if serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
				t.Errorf("server.Serve() error = %v", serveErr)
			}
		case <-time.After(time.Second):
			t.Error("server.Serve() did not stop during cleanup")
		}
	})

	client, err = net.Dial("tcp", listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	if tcp, ok := client.(*net.TCPConn); ok {
		_ = tcp.SetReadBuffer(1024)
	}
	if _, err := fmt.Fprintf(client,
		"GET / HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n",
		listener.Addr().String(),
	); err != nil {
		t.Fatal(err)
	}
	response, err := http.ReadResponse(bufio.NewReader(client), &http.Request{Method: http.MethodGet})
	if err != nil {
		t.Fatal(err)
	}
	responseBody = response.Body

	select {
	case result := <-done:
		if result.err == nil {
			t.Fatal("handler error = nil")
		}
		if result.operation != "flush" {
			t.Fatalf("deadline error operation = %q, want flush: %v", result.operation, result.err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("slow reader did not trigger the sliding write deadline")
	}
}

func TestBufferedStreamRealTCPSlowClientStopsDuringRelease(t *testing.T) {
	payload := []byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + strings.Repeat("x", 1<<20) + "\"},\"finish_reason\":null}]}\n\n")
	executor := fakeExecutionExecutor{stream: func(_ context.Context, _ execution.AttemptSpec, sink execution.StreamSink) execution.StreamResult {
		if err := sink(execution.StreamEvent{Sequence: 1, Kind: execution.StreamEventReady, StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}}); err != nil {
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
		}
		for sequence, data := range [][]byte{payload, []byte("data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"}]}\n\n"), []byte("data: [DONE]\n\n")} {
			if err := sink(execution.StreamEvent{Sequence: uint64(sequence + 2), Kind: execution.StreamEventData, Data: data}); err != nil {
				return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindInternal, Summary: err.Error()}}
			}
		}
		return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
	}}
	forwarder := NewExecutionForwarder(executor)
	forwarder.writeTimeout = 25 * time.Millisecond
	input := executionForwardInput()
	input.BufferedStream = true
	done := make(chan UpstreamResult, 1)
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	engine.GET("/", func(ctx *gin.Context) { done <- forwarder.ForwardStream(ctx.Request.Context(), input, ctx.Writer) })
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := &http.Server{Handler: engine}
	serveDone := make(chan error, 1)
	go func() { serveDone <- server.Serve(&smallWriteBufferListener{Listener: listener}) }()
	var client net.Conn
	t.Cleanup(func() {
		if client != nil {
			_ = client.Close()
		}
		_ = server.Close()
		_ = listener.Close()
		select {
		case serveErr := <-serveDone:
			if serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
				t.Errorf("server.Serve() error = %v", serveErr)
			}
		case <-time.After(time.Second):
			t.Error("server.Serve() did not stop")
		}
	})
	client, err = net.Dial("tcp", listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	if tcp, ok := client.(*net.TCPConn); ok {
		_ = tcp.SetReadBuffer(1024)
	}
	if _, err := fmt.Fprintf(client, "GET / HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n", listener.Addr().String()); err != nil {
		t.Fatal(err)
	}
	response, err := http.ReadResponse(bufio.NewReader(client), &http.Request{Method: http.MethodGet})
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	select {
	case result := <-done:
		if result.Err == nil || result.Stream.EndReason != StreamEndDownstreamWriteFailure {
			t.Fatalf("buffered slow-client result = %#v", result)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("buffered slow client did not stop during release")
	}
}

func TestBufferedStreamRealTCPRSTCancelsUpstream(t *testing.T) {
	started := make(chan struct{})
	canceled := make(chan struct{})
	executor := fakeExecutionExecutor{stream: func(ctx context.Context, _ execution.AttemptSpec, sink execution.StreamSink) execution.StreamResult {
		if err := sink(execution.StreamEvent{Sequence: 1, Kind: execution.StreamEventReady, StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}}); err != nil {
			return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK}
		}
		close(started)
		<-ctx.Done()
		close(canceled)
		return execution.StreamResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true, StatusCode: http.StatusOK, Error: &execution.ErrorEvidence{Kind: execution.ErrorKindCanceled, Summary: "canceled"}}
	}}
	forwarder := NewExecutionForwarder(executor)
	input := executionForwardInput()
	input.BufferedStream = true
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	engine.GET("/", func(ctx *gin.Context) { _ = forwarder.ForwardStream(ctx.Request.Context(), input, ctx.Writer) })
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := &http.Server{Handler: engine}
	serveDone := make(chan error, 1)
	go func() { serveDone <- server.Serve(listener) }()
	var client net.Conn
	t.Cleanup(func() {
		if client != nil {
			_ = client.Close()
		}
		_ = server.Close()
		_ = listener.Close()
		select {
		case <-serveDone:
		case <-time.After(time.Second):
			t.Error("RST server did not stop")
		}
	})
	client, err = net.Dial("tcp", listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fmt.Fprintf(client, "GET / HTTP/1.1\r\nHost: %s\r\nConnection: keep-alive\r\n\r\n", listener.Addr().String()); err != nil {
		t.Fatal(err)
	}
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("buffered upstream did not start")
	}
	if tcp, ok := client.(*net.TCPConn); ok {
		_ = tcp.SetLinger(0)
	}
	_ = client.Close()
	select {
	case <-canceled:
	case <-time.After(2 * time.Second):
		t.Fatal("TCP RST did not cancel buffered upstream")
	}
}

func TestBufferedWriteDeadlineStopsRealTCPSlowReader(t *testing.T) {
	done := make(chan error, 1)
	handler := &Handler{writeTimeout: 25 * time.Millisecond}
	body := bytes.Repeat([]byte("x"), int(maxNonStreamingResponseBodyBytes))
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	engine.GET("/", func(ginContext *gin.Context) {
		done <- handler.writeUpstreamResponse(ginContext, UpstreamResult{
			StatusCode: http.StatusOK,
			Header: http.Header{
				"Content-Length": {strconv.Itoa(len(body))},
				"Content-Type":   {"application/octet-stream"},
			},
			Body: body,
		})
	})

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := &http.Server{Handler: engine}
	serveDone := make(chan error, 1)
	go func() {
		serveDone <- server.Serve(&smallWriteBufferListener{Listener: listener})
	}()

	var client net.Conn
	var responseBody io.ReadCloser
	t.Cleanup(func() {
		if client != nil {
			_ = client.Close()
		}
		if responseBody != nil {
			_ = responseBody.Close()
		}
		_ = server.Close()
		_ = listener.Close()
		select {
		case serveErr := <-serveDone:
			if serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
				t.Errorf("server.Serve() error = %v", serveErr)
			}
		case <-time.After(time.Second):
			t.Error("server.Serve() did not stop during cleanup")
		}
	})

	client, err = net.Dial("tcp", listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	if tcp, ok := client.(*net.TCPConn); ok {
		_ = tcp.SetReadBuffer(1024)
	}
	if _, err := fmt.Fprintf(client,
		"GET / HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n",
		listener.Addr().String(),
	); err != nil {
		t.Fatal(err)
	}
	response, err := http.ReadResponse(bufio.NewReader(client), &http.Request{Method: http.MethodGet})
	if err != nil {
		t.Fatal(err)
	}
	responseBody = response.Body

	select {
	case writeErr := <-done:
		if writeErr == nil {
			t.Fatal("writeUpstreamResponse() error = nil")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("slow reader did not trigger the buffered write deadline")
	}
}

func TestBufferedWriteDeadlineStopsEmptyResponseWithLargeHeaderSlowReader(t *testing.T) {
	done := make(chan error, 1)
	handler := &Handler{writeTimeout: 25 * time.Millisecond}
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	engine.GET("/", func(ginContext *gin.Context) {
		done <- handler.writeUpstreamResponse(ginContext, UpstreamResult{
			StatusCode: http.StatusOK,
			Header: http.Header{
				"X-Large": {strings.Repeat("x", 2<<20)},
			},
		})
	})

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := &http.Server{Handler: engine}
	serveDone := make(chan error, 1)
	go func() {
		serveDone <- server.Serve(&smallWriteBufferListener{Listener: listener})
	}()

	var client net.Conn
	t.Cleanup(func() {
		if client != nil {
			_ = client.Close()
		}
		_ = server.Close()
		_ = listener.Close()
		select {
		case serveErr := <-serveDone:
			if serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
				t.Errorf("server.Serve() error = %v", serveErr)
			}
		case <-time.After(time.Second):
			t.Error("server.Serve() did not stop during cleanup")
		}
	})

	client, err = net.Dial("tcp", listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	if tcp, ok := client.(*net.TCPConn); ok {
		_ = tcp.SetReadBuffer(1024)
	}
	if _, err := fmt.Fprintf(client,
		"GET / HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n",
		listener.Addr().String(),
	); err != nil {
		t.Fatal(err)
	}

	select {
	case writeErr := <-done:
		if writeErr == nil {
			t.Fatal("writeUpstreamResponse() error = nil")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("slow reader did not trigger the empty buffered response write deadline")
	}
}

type smallWriteBufferListener struct{ net.Listener }

func (listener *smallWriteBufferListener) Accept() (net.Conn, error) {
	connection, err := listener.Listener.Accept()
	if err == nil {
		if tcp, ok := connection.(*net.TCPConn); ok {
			_ = tcp.SetWriteBuffer(1024)
		}
	}
	return connection, err
}

type streamGatewayGroup struct {
	id          uint
	name        string
	upstreamURL string
	apiKey      string
	modelID     string
	alias       string
	firstByte   time.Duration
	streamIdle  time.Duration
}

func newStreamingGatewayEngine(t *testing.T, groups ...streamGatewayGroup) (*gin.Engine, *state.CredentialRegistry) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	keyService := encryptiontest.Service(t, "stream-handler-test-master-key")

	groupConfigs := make([]state.GroupConfig, 0, len(groups))
	entries := make([]state.CredentialEntry, 0, len(groups))
	credentialConfigs := make([]state.CredentialConfig, 0, len(groups))
	for index, group := range groups {
		modelID := group.modelID
		if modelID == "" {
			modelID = "gpt-4o"
		}
		baseURL := testUpstreamBaseURL(group.upstreamURL, protocol.OpenAICompletions)
		channelID, params := testChannelConfig(t, protocol.OpenAICompletions, baseURL)
		groupConfigs = append(groupConfigs, state.GroupConfig{ConnectionType: "api_key", ID: group.id, Name: group.name, ChannelID: channelID, Params: params,
			Models: []state.ModelConfig{{ID: modelID, Alias: group.alias}}, Enabled: true,
			Settings: config.Settings{},
		})
		credentialID := uint(index + 1)
		entries = append(entries, testCredentialEntry(t, keyService, credentialID, group.id, group.apiKey))
		credentialConfigs = append(credentialConfigs, testCredentialConfig(credentialID, group.id))
	}

	manager := state.NewManager()
	snapshot, err := manager.Publish(state.CompileInput{
		SystemSettings:  config.Settings{state.SettingRetryCount: testDefaultRetryBudget},
		ChannelRegistry: channel.NewRegistry(), Groups: groupConfigs,
		Credentials: credentialConfigs,
		AccessKeys: []state.AccessKeyConfig{{
			ID: 1, Name: "client", KeyHash: keyService.Hash("gl-client"),
			Status: state.AccessKeyStatusActive,
		}},
	})
	if err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	for _, group := range groups {
		view := snapshot.Groups[group.id]
		if group.firstByte > 0 {
			view.Timeouts.FirstByte = group.firstByte
		}
		if group.streamIdle > 0 {
			view.Timeouts.StreamIdle = group.streamIdle
		}
		snapshot.Groups[group.id] = view
	}

	registry := state.NewCredentialRegistry()
	if err := registry.ReplaceCredentials(entries); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}
	handler := NewHandler(
		manager,
		registry,
		keyService,
		newTestExecutionForwarder(t),
		dialect.NewSet(dialect.NewOpenAI()),
		health.NewStatsStore(),
		health.NewMutationCoordinator(),
		nil,
		nil,
		nil,
	)
	handler.newRandom = func() *rand.Rand { return rand.New(zeroSource{}) }
	engine := gin.New()
	bindGatewayRoutesForTest(t, engine, handler)
	return engine, registry
}

func performStreamingRequest(engine *gin.Engine) *httptest.ResponseRecorder {
	return performStreamingRequestWithBody(engine, `{"model":"gpt-4o","stream":true}`)
}

func performStreamingRequestWithBody(engine *gin.Engine, body string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	request.Header.Set("Authorization", "Bearer gl-client")
	recorder := httptest.NewRecorder()
	engine.ServeHTTP(recorder, request)
	return recorder
}

// TestHandlerStreamProgressivelyReleasesAfterValidEOF proves that under the
// forced buffered contract the client sees only heartbeat before the upstream
// reaches a valid terminal state. Once the complete payload with [DONE] is
// delivered, the buffered release makes the full payload visible.
func TestHandlerStreamProgressivelyReleasesAfterValidEOF(t *testing.T) {
	firstEventSent := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })

	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		// Send a partial event (no finish_reason) and block.
		_, _ = writer.Write([]byte("data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"progressive\"},\"finish_reason\":null}]}\n\n"))
		writer.(http.Flusher).Flush()
		close(firstEventSent)
		<-release
		// Send the complete payload with terminal state.
		_, _ = writer.Write(forcedBufferedOpenAIStream())
		writer.(http.Flusher).Flush()
	}))
	defer upstream.Close()

	engine, _ := newStreamingGatewayEngine(t, streamGatewayGroup{
		id: 1, name: "progressive", upstreamURL: upstream.URL, apiKey: "sk-progressive",
	})
	gatewayServer := httptest.NewServer(engine)
	defer gatewayServer.Close()

	request, _ := http.NewRequest(http.MethodPost, gatewayServer.URL+"/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4o","stream":true}`))
	request.Header.Set("Authorization", "Bearer gl-client")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer response.Body.Close()

	// Wait for the upstream to have sent the first event.
	<-firstEventSent

	// The client must only see the heartbeat while the upstream hasn't
	// reached a valid terminal state.
	reader := bufio.NewReader(response.Body)
	heartbeat, err := reader.ReadString('\n')
	if err != nil || heartbeat != ": keep-alive\n" {
		t.Fatalf("before release first line = %q, %v", heartbeat, err)
	}
	boundary, err := reader.ReadString('\n')
	if err != nil || boundary != "\n" {
		t.Fatalf("before release boundary = %q, %v", boundary, err)
	}

	// Release the completion payload.
	releaseOnce.Do(func() { close(release) })

	// After release, the client sees the full payload including the partial event
	// that was buffered before the terminal state.
	rest, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("ReadAll after release: %v", err)
	}
	want := "data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"progressive\"},\"finish_reason\":null}]}\n\n" +
		string(forcedBufferedOpenAIStream())
	if string(rest) != want {
		t.Fatalf("after release payload = %q, want %q", rest, want)
	}
}

// TestHandlerAliasedStreamProgressivelyReleasesAfterValidEOF proves the same
// contract through an alias group: the alias is rewritten and the payload is
// held until terminal state, then released intact.
func TestHandlerAliasedStreamProgressivelyReleasesAfterValidEOF(t *testing.T) {
	release := make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })

	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		// Send a partial event without terminal state, then block.
		model := request.URL.Query().Get("model")
		_, _ = writer.Write([]byte("data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"model\":\"" + model + "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"alias-progressive\"},\"finish_reason\":null}]}\n\n"))
		writer.(http.Flusher).Flush()
		<-release
		_, _ = writer.Write(forcedBufferedOpenAIStream())
		writer.(http.Flusher).Flush()
	}))
	defer upstream.Close()

	engine, _ := newStreamingGatewayEngine(t,
		streamGatewayGroup{id: 1, name: "alias-a", upstreamURL: upstream.URL,
			apiKey: "sk-alias-a", modelID: "provider-model", alias: "gpt-4o"},
	)
	gatewayServer := httptest.NewServer(engine)
	defer gatewayServer.Close()

	request, _ := http.NewRequest(http.MethodPost, gatewayServer.URL+"/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4o","stream":true}`))
	request.Header.Set("Authorization", "Bearer gl-client")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer response.Body.Close()

	reader := bufio.NewReader(response.Body)
	heartbeat, err := reader.ReadString('\n')
	if err != nil || heartbeat != ": keep-alive\n" {
		t.Fatalf("before release first line = %q, %v", heartbeat, err)
	}
	if boundary, err := reader.ReadString('\n'); err != nil || boundary != "\n" {
		t.Fatalf("before release boundary = %q, %v", boundary, err)
	}

	releaseOnce.Do(func() { close(release) })

	rest, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("ReadAll after release: %v", err)
	}
	// Alias rewrite can reorder JSON keys, so check content rather than
	// exact byte equality.
	if !strings.Contains(string(rest), "\"content\":\"alias-progressive\"") ||
		!strings.Contains(string(rest), "\"content\":\"first\"") ||
		!strings.Contains(string(rest), "finish_reason\":\"stop\"") ||
		!strings.Contains(string(rest), "data: [DONE]") {
		t.Fatalf("after release payload = %q, want progressive content + complete terminal state", rest)
	}
}

const (
	// ineligibleStreamRequestBody carries a provider tool, which makes the
	// request replay-ineligible so a pre-release buffered failure stays final.
	ineligibleStreamRequestBody = `{"model":"gpt-4o","stream":true,"tools":[{"type":"web_search"}]}`
	// bufferedOpenAIStreamFailure is the in-stream error the buffered path emits
	// after HTTP 200 was already committed by the keep-alive heartbeat.
	bufferedOpenAIStreamFailure = "data: {\"error\":{\"type\":\"server_error\",\"message\":\"The buffered upstream stream could not be completed.\",\"code\":\"buffered_stream_failed\"}}\n\n"
)

// forcedBufferedOpenAIStream is a protocol-complete Chat Completions SSE
// payload. Under the forced buffered policy every choice must close before
// [DONE] or the gateway never releases the payload.
func forcedBufferedOpenAIStream() []byte {
	return []byte(
		"data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"first\"},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
	)
}

// recordedStreamUpstream serves one SSE payload and records every request, so
// focused tests can observe dispatch counts and headers without relying on the
// shared non-buffered fixture set.
type recordedStreamUpstream struct {
	*httptest.Server
	mu      sync.Mutex
	headers []http.Header
}

func newRecordedStreamUpstream(t *testing.T, payload []byte, headers http.Header) *recordedStreamUpstream {
	t.Helper()
	upstream := &recordedStreamUpstream{}
	upstream.Server = httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		_, _ = io.Copy(io.Discard, request.Body)
		upstream.mu.Lock()
		upstream.headers = append(upstream.headers, request.Header.Clone())
		upstream.mu.Unlock()
		for name, values := range headers {
			for _, value := range values {
				writer.Header().Add(name, value)
			}
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = writer.Write(payload)
		writer.(http.Flusher).Flush()
	}))
	t.Cleanup(upstream.Server.Close)
	return upstream
}

func (upstream *recordedStreamUpstream) Requests() []http.Header {
	upstream.mu.Lock()
	defer upstream.mu.Unlock()
	result := make([]http.Header, len(upstream.headers))
	copy(result, upstream.headers)
	return result
}

func newPartialStreamServer(calls *atomic.Int64, release <-chan struct{}) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = writer.Write([]byte("data: partial\n"))
		writer.(http.Flusher).Flush()
		calls.Add(1)
		// 保持部分事件未结束，直到测试确认逻辑超时，再释放服务器完成清理。
		<-release
	}))
}

func waitForStreamSignal(t *testing.T, signal <-chan struct{}, description string) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(time.Second):
		t.Fatalf("timed out waiting for %s", description)
	}
}

type zeroSource struct{}

func (zeroSource) Int63() int64 { return 0 }
func (zeroSource) Seed(int64)   {}

var _ rand.Source = zeroSource{}
