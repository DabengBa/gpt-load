# Codex WebSocket Session 合同

上游提交：`7d80a981 feat(codex): 添加独立上游 WebSocket Session 封装 (#612)`

本文只冻结合同。合同定了才允许写代码，代码只做合同里写了的事。

## 基线

- 目标分支：`plan/main-ahead-integration`（基于 `dev`）
- 依赖：`github.com/router-for-me/CLIProxyAPI/v7 v7.2.151`，其 `internalexecutor.CodexWebsocketsExecutor` 与
  `sdk/cliproxy/executor` 的 `ExecutionLifecycle`、`WebSocketResponseObserver`、`WithUpstreamAttemptTracker`、
  `UpstreamAttempted`、`WithRequiredUpstreamWebsocket`、`IsUpstreamWebsocketReplayRequired` 均已存在，不需要换版本。
- 当前 `dev` 的 Codex 执行是 HTTP：`internal/execution/cpa/codex_provider.go` →
  `internal/subscription/providers/codex` → `cpaembedded.CodexHTTPExecutor`，固定官方端点、无自定义 `BaseURL`。

## 1. 定位：显式句柄，不进数据面

- 新增 `codex.WSSession`（`internal/subscription/providers/codex/websocket.go`）+ vendored
  `cpaembedded.CodexWSSession`（`third_party/cpaembedded/embedded/codex_websocket.go`）。
- 首版**不注册**到 `provideradapter`/`execution.Executor`、不在网关请求路径上被调用、不替换 HTTP executor、
  不加 UI、不加配置项、不加数据库字段。
- 禁止 HTTP fallback：WS 失败不回落到 `CodexHTTPExecutor`。
- 禁止业务请求重放：断连、半响应、上游关闭都不重发同一 turn。
- 只服务原生 OpenAI Responses 生成（`model` + 可选 `previous_response_id`）。带 `type`/`stream_id`/
  `conversation`/`background=true` 的请求直接拒绝，不静默丢弃语义。

## 2. 凭据身份合同

- Session 的凭据由**调用方**给定，Session 自己不选择、不刷新、不写回。
- 身份三元组来自 `execution.CredentialSnapshot`：`ID`、`Version`、`IdentityGeneration`，在 `NewWSSession` 时快照，
  运行期不可变。
- 传给 CPA 的 `credentialID` 必须用与 `internal/execution/cpa/adapter.go` 相同的构造：
  `strconv.FormatUint(uint64(spec.Credential.ID), 10)`。
- 构造后删除 `refresh_token`，Session 不持有刷新能力；token 过期由调用方决定重新准备凭据并新建 Session。
- 凭据 replacement（同一分组 `identity_generation` 变化）后，调用方**必须** `Close` 旧 Session 并新建；
  不允许 Session 内部隐式跨凭据迁移。
- `ID == 0` 或凭据校验失败 → `invalid_session_options`，`DispatchState = not_sent`。

## 3. Dispatch 与 Replay Safety 合同

- 直接复用 `execution.DispatchState`，不新增并行枚举：`not_sent` / `maybe_sent`。
  （`DispatchLocal` 不适用：任何成功 turn 都真实触达上游。）
- 判定口径：`bound && UpstreamAttempted(turnCtx)` → `maybe_sent`，否则 `not_sent`。
  - `not_sent`：选项非法、代理非法、请求体非法/过大、不受支持的请求、ctx 已取消、session 已关闭、
    session 正忙、未 started 却带 `previous_response_id`、SDK 预处理失败且未触及上游传输边界。
  - `maybe_sent`：连接已绑定且已触及上游传输边界之后的一切失败——超时、取消、上游错误、半响应、
    `response.failed`/`response.incomplete`、上游关闭、事件过大、事件消费失败。
- `ResponseStarted` 由"是否至少向调用方交付了一个原生事件"决定。交付过事件时 dispatch 必须是 `maybe_sent`
  （与 `internal/execution/validation.go` 的 `responseStarted` 校验一致）。
- Replay Safety：WebSocket 路径没有任何"处理前拒绝"的证据来源，一律 `execution.ReplaySafetyUnknown`。
  接入数据面后，WS 分支不得因此把同一逻辑请求切到另一个候选。

## 4. Proxy 生命周期合同

- `ProxyURL` **必填且非空**：`"direct"` 或具体 URL。
- 禁止空字符串：CPA 的 WS dialer 在 `auth.ProxyURL` 为空时回落到 `http.ProxyFromEnvironment`，
  会绕开 `outboundproxy.Effective` 的来源追踪（`internal/outboundproxy` 的 `Source`）。
- 不支持 `ModeEnvironment`：dev 只能判断"进程有环境代理"，无法把它解析成具体 URL；
  构造时按 `invalid_proxy` 拒绝，不做猜测式回退。
- 允许的协议由调用方负责：接入数据面时必须取 `proxySettingsForAttempt` 的结果
  （当前只会产出 `direct`、`http://`、`socks5://`）。vendored 层只强制显式代理且无 path/query，
  `ModeEnvironment` 在到达 Session 之前就由调用方拒绝。
- 代理在 `NewWSSession` 冻结，整个 Session 生命周期不变，不接受 per-turn 覆盖。
- 拒绝带 path/query/fragment 的代理 URL。
- 注意：WS 拨号不读 `cliproxy.roundtripper` context，代理只能通过 `auth.ProxyURL` 生效。
  dev HTTP 路径的 `authWithoutProxyURL` 约定**不适用**于 Session。

## 5. 生命周期与并发合同

- 单飞：同一 Session 同时只允许一个 turn；并发调用返回 `session_busy`（`not_sent`），不排队、不串行化。
- `previous_response_id` 非空且 Session 尚未成功完成过首轮 → `continuation_requires_session`（`not_sent`）。
- turn 超时默认 5 分钟，调用方可覆盖；超时按 `timeout` 分类并使 Session 失效。
- 取消、超时、事件消费失败、上游终态失败、响应不完整 → Session 失效，后续 turn 返回 `session_closed`。
- `Close` 幂等，可从事件回调内调用；关闭只影响本 Session。
- 事件大小上限在 SDK 读取之后生效，不是帧级内存上限；这一点必须写进代码注释，不能让调用方误以为是背压。

## 6. 错误与脱敏

- `WSError` 暴露 `Code`、`UpstreamCode`、`HTTPStatus`、`DispatchState`，错误文本不含上游正文、凭据、地址、
  代理密码。`UpstreamCode` 只接受 `[A-Za-z0-9_.-]` 且不超过 128 字符，否则丢弃。
- 首版**不定义** `Code → execution.ErrorKind/FailureHint` 映射表：Session 还没进数据面，
  此时建表是猜测。接入数据面时按真实 code 再定。
- 已知缺口：CPA 在通知 lifecycle 之前会用默认 logger 输出 `codex websockets: upstream disconnected ... err=<正文>`。
  处理动作放在 dev 运行时（与 `runtime.go` 里已有的 `redact.NewHook` 同类位置），
  **不在** `third_party/cpaembedded` 里用 `init()` 注册全局 hook。接入数据面前必须关闭这个缺口。

## 7. 与上游 7d80a981 的差异（不是 cherry-pick）

1. 去掉 `BaseURL` / `ResolveCodexAPIEndpoints`：dev 的 Codex 执行固定官方端点（L3 专题 4 未落地），
   WS 同样固定官方端点，不引入自定义上游。
2. `ProxyURL` 必填、禁止环境代理回落：dev 的代理是显式策略 + 来源追踪，不能让 vendored dialer 自己读环境。
3. 不做 `codex_websocket_log.go` 的 `init()` 全局 logrus hook（见第 6 节）。
4. `WSSession` 的 `DispatchState` 直接沿用 `execution.DispatchState` 的字符串值域，不新增 `WSNotSent/WSMaybeSent`。

## 8. 证据

编译：`third_party/cpaembedded` 与主 module 的 `go build ./...`。

行为一：`third_party/cpaembedded/embedded/codex_websocket_test.go`（本地 httptest WebSocket server）

1. 选项拒绝：空凭据 ID、非法凭据、`ProxyURL` 为空、`ProxyURL` 带 path、负超时 → 对应 code + `not_sent`
2. 已取消 ctx → `canceled`（`errors.Is(context.Canceled)`）+ `not_sent`
3. `Close` 后复用 → `session_closed` + `not_sent`
4. 未 started 且带 `previous_response_id` → `continuation_requires_session` + `not_sent`
5. 并发 turn → `session_busy` + `not_sent`（第一个 turn 仍在飞行中）
6. 两轮续接复用同一条连接：首轮 `completed`、有 `ResponseID`、`Usage` 是上游原始 JSON、`maybe_sent`、
   握手响应头保留；连接数为 1
7. 发送后断连：server 收帧后发一个 `response.created` 即关闭 → 错误非 nil + `maybe_sent`，Session 失效
8. 上游 `error` 事件：`UpstreamCode=rate_limit_exceeded` + `maybe_sent`，错误文本不含上游正文

行为二：`internal/subscription/providers/codex/websocket_test.go`

1. 已取消 ctx → `canceled` + `not_sent`；`Close` 后复用 → `session_closed` + `not_sent`
2. 未解析的代理（空、`environment`、带 path）→ `invalid_proxy` + `not_sent`
3. HTTP executor 未被改动：新建 `NewExecutor()` 后其内部 bridge 仍可用

测试直接把已固定的 `auth.Attributes["base_url"]` 指向本地假上游；生产入口仍然固定官方端点，
不为测试引入自定义上游选项。

## 未决问题

- 数据面接入路径（哪个 operation、凭据 replacement 时如何回收 Session、request log 如何记 attempt）
  不在本专题范围，等合同与证据完成后单独定。
