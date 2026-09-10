package codex

import (
	"testing"

	"github.com/sirupsen/logrus"
)

// 日志形状取自 pinned CPA 的 codex_websockets_session.go：
// "codex websockets: upstream disconnected session=%s auth=%s url=%s reason=%s err=%v"
func TestWebsocketLogHookRemovesRawCloseErrorBody(t *testing.T) {
	for _, test := range []struct {
		message string
		want    string
		class   string
		code    int
	}{
		{
			message: "codex websockets: upstream disconnected session=gptload-codex-ws-9f2 auth=1 url=wss://chatgpt.com/backend-api/codex/responses reason=closed err=websocket: close 1006 (abnormal closure): unexpected EOF",
			want:    "codex websockets: upstream disconnected session=gptload-codex-ws-9f2 auth=1 url=wss://chatgpt.com/backend-api/codex/responses reason=closed",
			class:   "websocket_closed",
			code:    1006,
		},
		{
			message: "codex websockets: upstream disconnected session=gptload-codex-ws-9f2 auth=1 url=wss://chatgpt.com/backend-api/codex/responses reason=error err=token=secret-value",
			want:    "codex websockets: upstream disconnected session=gptload-codex-ws-9f2 auth=1 url=wss://chatgpt.com/backend-api/codex/responses reason=error",
			class:   "upstream_error",
		},
	} {
		entry := &logrus.Entry{Logger: logrus.StandardLogger(), Message: test.message, Data: logrus.Fields{}}
		if err := NewLogHook().Fire(entry); err != nil {
			t.Fatal(err)
		}
		if entry.Message != test.want {
			t.Fatalf("message = %q, want %q", entry.Message, test.want)
		}
		if entry.Data["error_class"] != test.class {
			t.Fatalf("error_class = %v, want %s", entry.Data["error_class"], test.class)
		}
		if test.code != 0 && entry.Data["ws_close_code"] != test.code {
			t.Fatalf("ws_close_code = %v, want %d", entry.Data["ws_close_code"], test.code)
		}
	}
}

func TestWebsocketLogHookLeavesOtherLogsUntouched(t *testing.T) {
	for _, message := range []string{
		"codex websockets: upstream disconnected session=other-session auth=1 url=wss://example.test reason=closed err=token=secret",
		// 没有 err= 的那一条本来就不含正文。
		"codex websockets: upstream disconnected session=gptload-codex-ws-9f2 auth=1 url=wss://chatgpt.com/backend-api/codex/responses reason=closed",
		"unrelated log line",
	} {
		entry := &logrus.Entry{Logger: logrus.StandardLogger(), Message: message, Data: logrus.Fields{}}
		if err := NewLogHook().Fire(entry); err != nil {
			t.Fatal(err)
		}
		if entry.Message != message || len(entry.Data) != 0 {
			t.Fatalf("hook touched %q: %q %v", message, entry.Message, entry.Data)
		}
	}
}
