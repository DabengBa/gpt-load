package gateway

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"gpt-load/internal/channel"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/state"
)

func TestWebsocketFiltersReasoningStatusWhenGroupEnabled(t *testing.T) {
	received := make(chan []byte, 1)
	upstream := httptestWebsocketServer(t, func(conn *websocket.Conn) {
		_, body, err := conn.ReadMessage()
		if err != nil {
			return
		}
		received <- body
		_ = conn.WriteMessage(websocket.TextMessage, websocketCompleted("resp_filter", ""))
	})
	defer upstream.Close()

	h, engine, input := websocketTestHandler(t, upstream.URL+"/v1", channel.OpenAI)
	input.Groups[0].Settings = config.Settings{
		state.SettingResponsesReasoningStatusFilterEnabled: true,
	}
	if _, err := h.manager.Publish(input); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(engine)
	defer server.Close()

	conn := dialGatewayWebsocket(t, server.URL)
	_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"response.create","model":"public","input":[{"type":"reasoning","status":"completed","summary":[]}]}`))
	if _, _, err := conn.ReadMessage(); err != nil {
		t.Fatal(err)
	}

	select {
	case body := <-received:
		if bytes.Contains(body, []byte(`"type":"reasoning","status"`)) {
			t.Fatalf("reasoning status reached upstream: %s", body)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("upstream did not receive a WebSocket turn")
	}
}

func httptestWebsocketServer(t *testing.T, handler func(*websocket.Conn)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		handler(conn)
	}))
}
