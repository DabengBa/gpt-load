package codex

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"time"

	cpaembedded "github.com/router-for-me/CLIProxyAPI/v7/gptload-embedded/embedded"
)

// WSSessionOptions 固定调用者已选择的凭据和出站代理。
// CredentialID 是调度器已选定凭据的身份，与 HTTP 执行路径一致使用十进制凭据 ID。
// ProxyURL 必须是 direct 或具体代理 URL；环境代理模式不受支持，调用方必须拒绝。
type WSSessionOptions struct {
	CredentialID    string
	Credential      Credential
	ProxyURL        string
	TurnTimeout     time.Duration
	MaxRequestBytes int
	MaxEventBytes   int
}

// WSTurnResult 的 Usage 保留上游 JSON，缺失时为 nil，不伪造零用量。
// DispatchState 取值与 execution.DispatchState 一致：not_sent / maybe_sent。
type WSTurnResult struct {
	ResponseID    string
	Status        string
	Usage         json.RawMessage
	Headers       http.Header
	DispatchState string
}

// WSError 暴露稳定分类和发送证据，错误文本不包含上游正文与凭据。
type WSError struct {
	Code          string
	UpstreamCode  string
	HTTPStatus    int
	DispatchState string
	cause         error
}

func (err *WSError) Error() string { return "codex websocket: " + err.Code }
func (err *WSError) Unwrap() error { return err.cause }

// WSSession 独立于既有 HTTP Executor，不注册到数据面或全局生命周期。
type WSSession struct{ bridge *cpaembedded.CodexWSSession }

// NewWSSession 创建独立句柄，首轮 ExecuteTurn 才建立上游连接。
func NewWSSession(options WSSessionOptions) (*WSSession, error) {
	bridge, err := cpaembedded.NewCodexWSSession(cpaembedded.CodexWSSessionOptions{
		CredentialID: options.CredentialID, Credential: credentialToBridge(options.Credential),
		ProxyURL:    options.ProxyURL,
		TurnTimeout: options.TurnTimeout, MaxRequestBytes: options.MaxRequestBytes, MaxEventBytes: options.MaxEventBytes,
	})
	if err != nil {
		return nil, wsErrorFromBridge(err)
	}
	return &WSSession{bridge: bridge}, nil
}

// ExecuteTurn 同步执行一轮原生 Responses 生成。
// emit 顺序接收原生 JSON 事件，须及时返回并响应 ctx；nil 表示忽略事件。
// 调用者负责保存响应 ID，并通过同一 Session 串行发起续接。
func (s *WSSession) ExecuteTurn(ctx context.Context, payload json.RawMessage, emit func(context.Context, json.RawMessage) error) (WSTurnResult, error) {
	var bridge *cpaembedded.CodexWSSession
	if s != nil {
		bridge = s.bridge
	}
	result, err := bridge.ExecuteTurn(ctx, payload, emit)
	return WSTurnResult{
		ResponseID: result.ResponseID, Status: result.Status, Usage: result.Usage,
		Headers: result.Headers, DispatchState: result.DispatchState,
	}, wsErrorFromBridge(err)
}

// Close 幂等关闭本 Session，不影响其他 Session；活动 ExecuteTurn 随后退出。
func (s *WSSession) Close() error {
	if s == nil {
		return nil
	}
	return wsErrorFromBridge(s.bridge.Close())
}

func wsErrorFromBridge(err error) error {
	if err == nil {
		return nil
	}
	var failure *cpaembedded.CodexWSError
	if errors.As(err, &failure) {
		return &WSError{
			Code: failure.Code, UpstreamCode: failure.UpstreamCode, HTTPStatus: failure.HTTPStatus,
			DispatchState: failure.DispatchState, cause: failure.Unwrap(),
		}
	}
	// 未知失败按已发送处理：宁可禁止重放，也不能凭空声明未发送。
	return &WSError{Code: "internal_error", DispatchState: "maybe_sent"}
}
