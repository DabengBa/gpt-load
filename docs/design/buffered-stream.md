# Buffered Stream

## 功能边界

HTTP/SSE 流式交付由单一策略入口固定，不再有 `buffered_stream` 开关、Group 覆盖、默认值或回滚路径。网关继续以 SSE 从上游接收流，但在协议终态、JSON 结构和事件生命周期全部验证完成前，不向客户端释放模型 payload。验证期间客户端只会收到网关注释心跳；成功后网关按原事件顺序集中释放已验证的事件。该模式不是非流式上游，也不提供断点续传、Exactly-once 或跨请求恢复。

固定交付矩阵（`internal/gateway/stream_policy.go` 的单一入口判定）：

| 客户端协议 | 流式 operation | 交付 |
| --- | --- | --- |
| OpenAI Chat Completions | chat completions | 强制 buffered |
| OpenAI Responses | `create` | 强制 buffered |
| Anthropic Messages | messages | 强制 buffered |
| Gemini | 生成型 operation | live exception：实时透传，不进入 buffered 验证、释放和重放窗口 |
| OpenAI Images | generate / edit | live exception：同上 |
| 其它协议或 operation | 任意 | dispatch 前拒绝 |

被拒绝的流式请求在 dialect 解析成功后、scheduler 选候选和 provider dispatch 之前返回 HTTP 400：已知协议的不支持 operation 返回 `streaming_operation_unsupported`，未知客户端协议返回 `streaming_protocol_unsupported`。malformed 请求仍优先返回既有 `invalid_protocol_request`。此前的协议专用拒绝错误码已删除。

Gemini 与 OpenAI Images 是明确的实时例外：它们保持原有实时流行为及统一 health judge，不产生 buffered heartbeat、spool 或 release gate，也不获得 buffered 重放/重试许可。非流式请求不进入本策略，行为不变。

## 交付流程

1. 请求完成鉴权、校验和首次准入后，每个 attempt 先完成本地 pre-dispatch 校验。只有收到 dispatch proof（`forwardStream` 的 `StreamEventReady` 或 capture writer 首次写入）后，网关才提交 `text/event-stream` 响应并发送 `: keep-alive\n\n`；本地校验失败保留原 HTTP 状态和错误码，不提交 buffered heartbeat。
2. 上游每个 attempt 继续使用原有流式执行、转换、脱敏和 usage 处理。网关每 15 秒发送一次注释心跳；心跳不是模型首字节，也不刷新上游 first-byte 或 stream-idle 计时器。
3. 每个 attempt 独立暂存并验证事件。Chat 必须有合法的 choice 终态和 `[DONE]`；Responses 区分 `response.completed`、`response.incomplete` 和 `response.failed`；Anthropic 必须有 `message_stop`，且每个已开始的 content block 都必须闭合。
4. 合法 EOF 和协议终态验证成功后，才开始一次不可逆的 payload release。释放从成功 attempt 开始，失败 attempt 的内容不会泄漏给客户端。
5. 已提交 heartbeat 但尚未释放 payload 时，只有明确的上游断流、idle timeout、半帧或协议错误，并且请求被证明没有供应商副作用，才允许按既有候选预算重试。对于同样满足 buffered 重放资格、但在收到任何上游响应前发生的 transport/timeout，网关也会按“结果未知”切换候选；这会承担上游已接受请求而再次执行的重复计费或副作用风险。payload release 开始后禁止重试，即使下游部分写入后失败也不会重放。上游状态可重试（408、429、5xx）却没有带回任何可分类证据的失败按 `fallback.missing_evidence_retry` 同样获得候选切换许可，不再直接终止请求；该分支按 `record_credential_failure` 计入凭据连续失败，达到 `blacklist_threshold` 后该凭据被拉黑，并由凭据检测探活恢复；它同样受尝试预算与 payload release 约束，已释放 payload 或已提交的非缓冲实时流既不再重放也不再计入凭据失败。带回可分类证据的上游 HTTP 失败（408、429、5xx）同样获得候选切换许可，规则记为 `buffered_stream.retry_before_release_upstream_status`，分类、作用域与 effect 保留证据的结论（5xx 落在 group 作用域并 skip group，429 沿用凭据冷却），不再退化成无证据分支。非 2xx 结束的 buffered attempt 不再被记成正常结束：end reason 为 `upstream_failed`，attempt 的错误码取执行层证据（如 `upstream_invalid_key`），请求日志按失败记录而不是成功。`retry_count` 是一次请求的上游尝试总次数上限，来自系统设置并在请求开始时冻结（跨分组共享，切换分组不重置）：0 与 1 都表示只尝试一次，2 表示可以换一次候选，默认 5；分组设置里的 `retry_count` 已退役，读取时容忍存量值、写入被拒绝、界面不展示，且任何一次保存分组都会把库里的残留删掉。

HTTP 响应头由请求级输出 owner 串行管理，包含 `Cache-Control: no-cache, no-transform` 和 `X-Accel-Buffering: no`，并移除不适用的 `Content-Length`。心跳提交 HTTP 后不会再补写上游成功响应头。

## 资源限制

每个 attempt 的默认限制是：

- 内存暂存阈值：1 MiB
- 单次响应硬上限：32 MiB
- 进程内暂存总预算：256 MiB，按已接收字节计，内存和临时文件合计

超过内存阈值后使用 Go 标准库临时文件。临时目录权限为 `0700`，文件权限为 `0600`；Unix 创建后解除文件名链接，仅保留打开的 fd。预算预留、写入、读回、取消、失败和关闭都会释放资源。超过硬上限、预算耗尽、磁盘写失败或读回失败都终止请求，不降级为实时透传，也不把本地资源错误当作可重试的供应商错误。暂存内容可能以明文存在受限临时卷中，运维应使用受限或加密的临时文件系统。

请求总 deadline 在第一次实际 forward 前按选定 Group 的 Request timeout 冻结，覆盖候选等待、所有 attempt、验证、重试和 payload release。每次下游写仍有 30 秒上限，并受总 deadline 约束；慢客户端不会让请求永久保活。

## 重试与计费风险

缓冲重试只对明确证明为无副作用的请求新增许可。OpenAI Responses 的存储、`previous_response_id`、conversation/continuity、background、prompt/resource reference、供应商工具或未知语义都不会获得这项许可；未知工具字段和未知顶层语义按保守规则拒绝重放。客户端自行执行的明确 `function`/`custom` 工具描述可以符合资格，但网关不会替请求关闭 store 或删除工具字段。

强制 buffered 意味着符合资格的生成在上游中途失败后可能被另一个候选重新执行，因此可能产生重复 token、重复请求或重复计费；该功能不承诺幂等。受保护请求可以完整缓冲，但失败时只返回失败，不新增 buffered 重试。HTTP 200、心跳或已观察到的 response ID 都不等于模型成功，也不会被伪造为成功终态。

建立 SSE 后的终失败使用客户端协议的流内错误：OpenAI 使用 `data.error` 或 Responses `event:error`，Anthropic 使用 `event:error`；保留已观察到的 Responses response ID，不回放失败 attempt 内容。供应商错误、协议错误、超时和下游写失败会分别记录 attempt 状态、usage、可见字节、暂存峰值、spill 状态和 release 起点，但不会记录响应原文。

## 固定策略与发布注意事项

流式交付没有开关：三个生成协议始终 buffered，Gemini/Images 始终 live exception，不存在“关掉开关回到实时路径”的操作。发布前应确认反向代理不缓冲注释心跳、临时卷容量充足，并让客户端超时覆盖完整缓冲与释放窗口。遇到 SDK 超时、代理不转发注释、临时卷空间不足或费用异常时，只能修复代理、容量或超时配置，或升级版本；没有关闭开关的逃生路径。生产网关、反向代理和目标客户端的超时仍需在灰度环境单独确认。

## 本地兼容性证明

`internal/gateway/testdata/buffered_stream_sdk_smoke.py` 启动仅监听 `127.0.0.1` 的本地 SSE fixture，不连接真实供应商。它使用隔离 Python 环境中的官方 OpenAI 与 Anthropic SDK，逐一验证协议与场景矩阵：Chat Completions 为 success、incomplete、合并后的 `error`；Responses 为 success、incomplete、failed、stream error；Anthropic Messages 为 success、incomplete、合并后的 `error`。Chat success 同时要求 SDK 观察到 `finish_reason:stop` 且 wire 含 `[DONE]`；Responses success/incomplete/failed 分别要求对应事件和 `response.status`；Anthropic success 要求完整的 message/block 生命周期。Chat 与 Anthropic 的 failed 和 stream-error 在该 fixture 中发送相同 wire 语义，因此不虚构两项独立覆盖。异常只接受对应 SDK 的明确 API 错误类型，并结合 fixture wire 记录判断。执行命令、SDK 版本和每条结果保存在本单元的 `evidence/validation-attempt-2.md` 与原始日志中。

该 smoke 只证明官方 SDK 对本地协议字节的兼容性，不证明 GPT-Load handler、鉴权、路由转换、缓冲、重试或真实 heartbeat/release。网关行为另由实际 gateway 输出驱动的 Go 集成测试证明；两类证据不能互相替代。
