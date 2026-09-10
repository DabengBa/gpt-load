package codex

import (
	"testing"

	"github.com/sirupsen/logrus"
)

func TestWebsocketLogHookRemovesRawCloseErrorBody(t *testing.T) {
	for _, test := range []struct {
		message string
		want    string
		class   string
		code    int
	}{
		{
			message: "codex websockets: upstream disconnected session=gptload-codex-ws-9f2 err=websocket: close 1006 (abnormal closure): unexpected EOF",
			want:    "codex websockets: upstream disconnected session=gptload-codex-ws-9f2",
			class:   "websocket_closed",
			code:    1006,
		},
		{
			message: "codex websockets: upstream disconnected session=gptload-codex-ws-9f2 err=token=secret-value",
			want:    "codex websockets: upstream disconnected session=gptload-codex-ws-9f2",
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
		"codex websockets: upstream disconnected session=other-session err=token=secret",
		"codex websockets: upstream disconnected session=gptload-codex-ws-9f2",
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
