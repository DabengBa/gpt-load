# Buffered Stream

## 功能边界

`buffered_stream` 让网关继续以 SSE 从上游接收流，但在协议终态、JSON 结构和事件生命周期全部验证完成前，不向客户端释放模型 payload。验证期间客户端只会收到网关注释心跳；成功后网关按原事件顺序集中释放已验证的事件。该模式不是非流式上游，也不提供断点续传、Exactly-once 或跨请求恢复。

功能默认关闭。全局设置和 Group 设置都使用既有稀疏 override：Group 未覆盖时继承全局值，显式覆盖可开启或关闭，恢复继承后使用全局值。保存期间控件禁用；保存失败保留草稿并显示失败反馈，重载后以服务端持久化值为准。关闭开关只影响新请求，已开始的请求不会在途切换模式。

当前支持客户端 SSE 协议：

- OpenAI Chat Completions
- OpenAI Responses
- Anthropic Messages

Gemini、Images、音频流、非 SSE 请求不进入 buffered stream。对不支持的客户端协议，网关在 dispatch 前返回 `buffered_stream_unsupported_protocol`，不会静默降级为普通实时流。非流式请求行为不变。

## 交付流程

1. 请求完成鉴权、校验和首次准入后，网关提交 `text/event-stream` 响应，并发送 `: keep-alive\n\n`。
2. 上游每个 attempt 继续使用原有流式执行、转换、脱敏和 usage 处理。网关每 15 秒发送一次注释心跳；心跳不是模型首字节，也不刷新上游 first-byte 或 stream-idle 计时器。
3. 每个 attempt 独立暂存并验证事件。Chat 必须有合法的 choice 终态和 `[DONE]`；Responses 区分 `response.completed`、`response.incomplete` 和 `response.failed`；Anthropic 必须有 `message_stop`，且每个已开始的 content block 都必须闭合。
4. 合法 EOF 和协议终态验证成功后，才开始一次不可逆的 payload release。释放从成功 attempt 开始，失败 attempt 的内容不会泄漏给客户端。
5. 已提交 heartbeat 但尚未释放 payload 时，只有明确的上游断流、idle timeout、半帧或协议错误，并且请求被证明没有供应商副作用，才允许按既有候选预算重试。payload release 开始后禁止重试，即使下游部分写入后失败也不会重放。

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

开启 buffered stream 意味着符合资格的生成在上游中途失败后可能被另一个候选重新执行，因此可能产生重复 token、重复请求或重复计费；该功能不承诺幂等。受保护请求可以完整缓冲，但失败时只返回失败，不新增 buffered 重试。HTTP 200、心跳或已观察到的 response ID 都不等于模型成功，也不会被伪造为成功终态。

建立 SSE 后的终失败使用客户端协议的流内错误：OpenAI 使用 `data.error` 或 Responses `event:error`，Anthropic 使用 `event:error`；保留已观察到的 Responses response ID，不回放失败 attempt 内容。供应商错误、协议错误、超时和下游写失败会分别记录 attempt 状态、usage、可见字节、暂存峰值、spill 状态和 release 起点，但不会记录响应原文。

## 灰度与回滚

建议先在单个 Group 开启并观察请求日志、失败 attempt、重复计费和客户端超时。回滚时先关闭 Group override 或恢复继承，再按需关闭全局 `buffered_stream`；关闭只阻止新请求进入缓冲模式，已在途请求继续完成其冻结的模式。遇到 SDK 超时、代理不转发注释、临时卷空间不足或费用异常时，应立即关闭开关并保留旧的实时流路径。生产网关、反向代理和目标客户端的超时仍需在灰度环境单独确认。

## 本地兼容性证明

`internal/gateway/testdata/buffered_stream_sdk_smoke.py` 启动仅监听 `127.0.0.1` 的本地 SSE fixture，不连接真实供应商。它使用隔离 Python 环境中的官方 OpenAI 与 Anthropic SDK，逐一验证协议与场景矩阵：Chat Completions 为 success、incomplete、合并后的 `error`；Responses 为 success、incomplete、failed、stream error；Anthropic Messages 为 success、incomplete、合并后的 `error`。Chat success 同时要求 SDK 观察到 `finish_reason:stop` 且 wire 含 `[DONE]`；Responses success/incomplete/failed 分别要求对应事件和 `response.status`；Anthropic success 要求完整的 message/block 生命周期。Chat 与 Anthropic 的 failed 和 stream-error 在该 fixture 中发送相同 wire 语义，因此不虚构两项独立覆盖。异常只接受对应 SDK 的明确 API 错误类型，并结合 fixture wire 记录判断。执行命令、SDK 版本和每条结果保存在本单元的 `evidence/validation-attempt-2.md` 与原始日志中。

该 smoke 只证明官方 SDK 对本地协议字节的兼容性，不证明 GPT-Load handler、鉴权、路由转换、缓冲、重试或真实 heartbeat/release。网关行为另由实际 gateway 输出驱动的 Go 集成测试证明；两类证据不能互相替代。
