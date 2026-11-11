package gateway

import (
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
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
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("enabled=%t", enabled), func(t *testing.T) {
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
				state.SettingResponsesReasoningStatusFilterEnabled: enabled,
			}
			if _, err := h.manager.Publish(input); err != nil {
				t.Fatal(err)
			}
			server := httptest.NewServer(engine)
			defer server.Close()

			conn := dialGatewayWebsocket(t, server.URL)
			defer conn.Close()
			if err := conn.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
				t.Fatal(err)
			}
			if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"response.create","model":"public","input":`+reasoningStatusInput+`}`)); err != nil {
				t.Fatal(err)
			}
			if _, _, err := conn.ReadMessage(); err != nil {
				t.Fatal(err)
			}

			select {
			case body := <-received:
				var request struct {
					Input jsontext.Value `json:"input"`
				}
				if err := json.Unmarshal(body, &request); err != nil {
					t.Fatal(err)
				}
				want := reasoningStatusInput
				if enabled {
					want = reasoningStatusFilteredInput
				}
				assertJSONEqual(t, string(request.Input), want)
			case <-time.After(2 * time.Second):
				t.Fatal("upstream did not receive a WebSocket turn")
			}
		})
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
