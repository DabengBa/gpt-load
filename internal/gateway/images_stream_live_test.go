package gateway

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
)

// TestOpenAIImagesStreamDeliversFirstEventBeforeEOF proves the Images live
// exception: the first provider event (partial_image) reaches the client before
// the upstream reaches EOF, with no buffered heartbeat, spool or release gate.
func TestOpenAIImagesStreamDeliversFirstEventBeforeEOF(t *testing.T) {
	const (
		partial = "data: {\"type\":\"image_generation.partial_image\",\"b64_json\":\"AA==\"}\n\n"
		done    = "event: image_generation.completed\ndata: {\"type\":\"image_generation.completed\"}\n\n"
	)
	firstEventSent := make(chan struct{})
	releaseSecond := make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(releaseSecond) })

	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = writer.Write([]byte(partial))
		writer.(http.Flusher).Flush()
		close(firstEventSent)
		<-releaseSecond
		_, _ = writer.Write([]byte(done))
		writer.(http.Flusher).Flush()
	}))
	defer upstream.Close()

	params, err := json.Marshal(map[string]string{"base_url": upstream.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	engine, _ := newDialectGatewayEngine(t, protocol.OpenAIImages, "public-image",
		dialect.NewSet(dialect.NewOpenAIImages()),
		dialectGatewayGroup{id: 1, name: "images", upstreamURL: upstream.URL,
			apiKeys: []string{"sk-images"}, channelID: channel.OpenAI, params: params,
		},
	)
	gatewayServer := httptest.NewServer(engine)
	defer gatewayServer.Close()

	request, err := http.NewRequest(http.MethodPost, gatewayServer.URL+"/v1/images/generations",
		strings.NewReader(`{"model":"public-image","prompt":"draw","n":1,"stream":true}`))
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	request.Header.Set("Authorization", "Bearer gl-client")
	client := &http.Client{Timeout: 2 * time.Second}
	response, err := client.Do(request)
	if err != nil {
		t.Fatalf("stream request error = %v", err)
	}
	defer response.Body.Close()

	select {
	case <-firstEventSent:
	case <-time.After(time.Second):
		t.Fatal("upstream did not send first event")
	}

	if response.StatusCode != http.StatusOK {
		t.Fatalf("response status = %d", response.StatusCode)
	}

	reader := bufio.NewReader(response.Body)
	line, err := reader.ReadString('\n')
	if err != nil {
		t.Fatalf("read first line: %v", err)
	}
	blank, err := reader.ReadString('\n')
	if err != nil {
		t.Fatalf("read event boundary: %v", err)
	}
	if line == ": keep-alive\n" {
		t.Fatalf("Images live exception received a buffered heartbeat: %q", line+blank)
	}
	if !strings.Contains(line+blank, "partial_image") {
		t.Fatalf("first partial_image event was not delivered before EOF: %q", line+blank)
	}

	releaseOnce.Do(func() { close(releaseSecond) })
	rest, err := io.ReadAll(reader)
	if err != nil || !strings.Contains(string(rest), "image_generation.completed") {
		t.Fatalf("remaining stream = %q, %v", rest, err)
	}
}

// TestOpenAIImagesAliasedStreamDeliversFirstEventBeforeEOF proves the live
// exception also applies to aliased Images streams.
func TestOpenAIImagesAliasedStreamDeliversFirstEventBeforeEOF(t *testing.T) {
	const (
		partial = "data: {\"type\":\"image_generation.partial_image\",\"b64_json\":\"AA==\"}\n\n"
		done    = "event: image_generation.completed\ndata: {\"type\":\"image_generation.completed\"}\n\n"
	)
	firstEventSent := make(chan struct{})
	releaseSecond := make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(releaseSecond) })

	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = writer.Write([]byte(partial))
		writer.(http.Flusher).Flush()
		close(firstEventSent)
		<-releaseSecond
		_, _ = writer.Write([]byte(done))
		writer.(http.Flusher).Flush()
	}))
	defer upstream.Close()

	params, err := json.Marshal(map[string]string{"base_url": upstream.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	engine, _ := newDialectGatewayEngine(t, protocol.OpenAIImages, "public-image",
		dialect.NewSet(dialect.NewOpenAIImages()),
		dialectGatewayGroup{id: 1, name: "images-alias", upstreamURL: upstream.URL,
			apiKeys: []string{"sk-image-alias"}, channelID: channel.OpenAI, params: params,
			models: []state.ModelConfig{{ID: "provider-image-model", Alias: "public-image"}},
		},
	)
	gatewayServer := httptest.NewServer(engine)
	defer gatewayServer.Close()

	request, err := http.NewRequest(http.MethodPost, gatewayServer.URL+"/v1/images/generations",
		strings.NewReader(`{"model":"public-image","prompt":"draw","n":1,"stream":true}`))
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	request.Header.Set("Authorization", "Bearer gl-client")
	client := &http.Client{Timeout: 2 * time.Second}
	response, err := client.Do(request)
	if err != nil {
		t.Fatalf("stream request error = %v", err)
	}
	defer response.Body.Close()

	select {
	case <-firstEventSent:
	case <-time.After(time.Second):
		t.Fatal("upstream did not send first event")
	}

	if response.StatusCode != http.StatusOK {
		t.Fatalf("response status = %d", response.StatusCode)
	}

	reader := bufio.NewReader(response.Body)
	line, err := reader.ReadString('\n')
	if err != nil {
		t.Fatalf("read first line: %v", err)
	}
	blank, err := reader.ReadString('\n')
	if err != nil {
		t.Fatalf("read event boundary: %v", err)
	}
	if line == ": keep-alive\n" {
		t.Fatalf("aliased Images live exception received a buffered heartbeat: %q", line+blank)
	}
	if !strings.Contains(line+blank, "partial_image") {
		t.Fatalf("first partial_image event was not delivered before EOF: %q", line+blank)
	}

	releaseOnce.Do(func() { close(releaseSecond) })
	rest, err := io.ReadAll(reader)
	if err != nil || !strings.Contains(string(rest), "image_generation.completed") {
		t.Fatalf("remaining stream = %q, %v", rest, err)
	}
}
