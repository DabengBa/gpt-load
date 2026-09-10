package codex

import (
	"strconv"
	"strings"

	"github.com/sirupsen/logrus"
)

// codexWSSessionIDPrefix 必须与 vendored cpaembedded 里生成会话 ID 的前缀一致。
const codexWSSessionIDPrefix = "gptload-codex-ws-"

const codexWSDisconnectedPrefix = "codex websockets: upstream disconnected session="

// LogHook 移除 pinned SDK 在通知 lifecycle 之前写入的原始断连正文。
// CPA v7.2.151 会在 lifecycle 之前把 CloseError 正文写进默认 logger；这里只处理
// 本封装会话的那一条日志，不修改日志级别、输出或其他执行器的日志。
// 按 dev 合同在运行时注册，不在 vendored 包用 init() 注册全局 hook。
type LogHook struct{}

// NewLogHook 返回可注册到 logrus 的 hook。
func NewLogHook() logrus.Hook { return LogHook{} }

func (LogHook) Levels() []logrus.Level { return []logrus.Level{logrus.InfoLevel} }

func (LogHook) Fire(entry *logrus.Entry) error {
	if entry == nil || !strings.HasPrefix(entry.Message, codexWSDisconnectedPrefix+codexWSSessionIDPrefix) {
		return nil
	}
	message, rawError, found := strings.Cut(entry.Message, " err=")
	if !found {
		return nil
	}
	if entry.Data == nil {
		entry.Data = logrus.Fields{}
	}
	entry.Message = message
	entry.Data["error_class"] = "upstream_error"
	// 只保留 Gorilla CloseError 的数值关闭码；未知格式同样不保留正文。
	if detail, ok := strings.CutPrefix(rawError, "websocket: close "); ok {
		if end := strings.IndexAny(detail, " :"); end >= 0 {
			detail = detail[:end]
		}
		if code, err := strconv.Atoi(detail); err == nil && code >= 1000 && code < 5000 {
			entry.Data["ws_close_code"] = code
			entry.Data["error_class"] = "websocket_closed"
		}
	}
	return nil
}
