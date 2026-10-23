---
description: "Enable and inspect raw Gateway and provider communication captures, including model probes and credential tests, with explicit retention and evidence boundaries."
kind: technical
topic: debug-capture
code:
  paths:
    - internal/debugcapture
    - internal/gateway/capture.go
    - internal/container/debug_capture.go
    - internal/container/debug_capture_unix.go
    - internal/control/probe_capture.go
    - internal/execution/http_observer.go
    - third_party/bifrost-core/providers/utils/http_observer.go
---
# Debug Communication Capture

Debug communication capture is an opt-in operational tool for investigating gateway and provider integration behavior. On Unix runtimes, set `DEBUG_CAPTURE_ENABLED=true` to enable it; it is disabled by default.

## Retention

The capture store uses the existing configured database and remains separate from `request_logs`. Captures are retained for a fixed 4-hour expiry, then startup and periodic cleanup remove expired rows and chunks automatically. Each runtime admits at most 64 concurrent capture sessions; each attempt retains at most 8 MiB of raw body data and 128 pending persistence events. Exceeding a limit fails the affected capture without changing the data-plane response.

The capture can contain complete observed sensitive values, including `Authorization`, `Cookie`, API keys, request bodies, response bodies, and provider credentials passed through the supported observation boundary. Protect `AUTH_KEY`, the database, database backups, and exported archives. Do not enable this feature on an untrusted or shared deployment unless that exposure is intentional.

## Administrator API

These endpoints require the management `AUTH_KEY` and are administrator-only. Data-plane access keys cannot use them.

- `GET /api/debug-captures`
- `GET /api/debug-captures/:capture_id`
- `GET /api/debug-captures/:capture_id/download`

The list endpoint accepts `request_id`, `access_key_id`, `protocol`, `operation`, `state`, `created_after_ms`, `created_before_ms`, `offset`, and `limit`. The default page size is 50 and the maximum is 200. `created_after_ms` is inclusive and `created_before_ms` is exclusive.

The detail response contains session metadata, lifecycle state, errors, and ordered logical attempts. The download endpoint streams a ZIP containing a JSON manifest, per-attempt metadata, and raw request/response header and body parts. Download responses are marked `no-store`.

The `/api/health` response includes `debug_capture` with enabled/running state, fixed retention seconds, active/completed/failed counts, sweep counters, and timestamps. These state counts describe unexpired database rows, not the runtime's in-memory admission slots. When all 64 slots are occupied, the component reports `error: "session_limit_reached"`, including when expired rows have already disappeared from the counts. The error clears when admission becomes available again. If the optional store cannot provide counts, the component reports `error: "counts_unavailable"` while the broader health response remains available.

Feature-specific capture storage and adapters are Unix-only. Windows retains the public application types and no-op container assembly so the unrelated Windows binary continues to compile; Windows does not install a capture store or Gateway factory and does not provide raw capture.

## Capture boundary

A session starts at the Gateway data-plane boundary for `/v1` and `/v1beta` namespaces, before authentication and early route termination. It records the client request that is actually observed, final client response writes, material write/read/close/flush/hijack outcomes, logical retry attempts, and preparation failures.

Bifrost-backed HTTP provider adapters pass a real per-request observer through the supported execution context. Their observed upstream request and response headers, bodies, termination events, and callback errors are recorded in the corresponding logical attempt. When debug capture is enabled, model-probe and credential-test operations under `/api` use the same observer and store path; each dispatched probe has a session keyed by its `request_id` and a logical attempt containing provider, channel, model, route, sequence, and retry/fallback metadata. Probe capture remains disabled when the explicit opt-in is off.

The feature records only data that the Gateway or a supported Bifrost HTTP observer actually observes. It does not claim to capture TLS or socket wire bytes, HTTP transfer framing already removed by the transport, HTTP trailers not exposed through the observer contract, or data that was never observed before cancellation, timeout, process failure, or connection close.

Capture persistence is isolated from the data plane. Storage, queue, callback, or cleanup failures do not change the response returned to the client. The affected capture is instead marked failed or remains non-terminal, and the runtime health section reports cleanup state and active/completed/failed counts where available.

Gateway finalization checks cancellation against the request context saved before the data-plane lifecycle adds its own child context. Normal lifecycle release must not turn a complete successful capture into a cancellation failure. Genuine client cancellation and server shutdown still propagate through the saved context.

The saved context is `request.Context()` as bound when the capture is created, not a context the data-plane lifecycle derives afterwards.

Provider observation sources register before emitting callbacks and unregister after their completion callback returns. Gateway capture waits for registered observations and admitted callbacks; an executor result alone does not establish that a transport observation started. Once the sources and callbacks have drained, absence of a response completion is recorded as `no_observation_source` and capture finalization fails explicitly, releasing its runtime admission instead of waiting forever. The event detail distinguishes a source that never started from one that started and ended without completion. Neither condition establishes whether request or response bytes were exchanged. This event identifies missing raw evidence; it is not a Provider response termination observed on the wire. A first-token abort already fired before the net/http send completes its registered observation with `no_response`, without issuing a Provider request.

The lifecycle contract is optional on both sides. `internal/execution/http_observer.go` declares `HTTPObservationLifecycle`, and the fork's observer state in `third_party/bifrost-core/providers/utils/http_observer.go` begins one source per real send before any callback can be emitted and ends it after the final completion callback returns, including the fired-abort path in `DoHTTPRequest`. Neither call may block. `internal/gateway/capture.go` implements it for the Gateway; `internal/control/probe_capture.go` implements it for probes.

Regression evidence for this lifecycle lives in `internal/gateway/capture_lifecycle_test.go`, `internal/gateway/capture_abort_evidence_test.go`, `internal/gateway/capture_test.go`, `internal/debugcapture/runtime_test.go`, `internal/container/debug_capture_lifecycle_test.go`, and `third_party/bifrost-core/providers/utils/http_observer_test.go`.

## Model probes and credential tests

Model probes and credential tests use the same opt-in store and administrator
query/download APIs. Each probe uses its execution `request_id` and attempt
identity to associate the observed provider headers, body bytes, and termination
events with the capture. These captures describe provider communication; they do
not add a Gateway client request/response pair or a structured request-log row.
A probe capture can be located directly with the debug capture `request_id`
filter, without first finding a request-log entry.

`internal/control/probe_capture.go` supplies the shared observer through the
execution context. `internal/container/debug_capture.go` adapts it to the capture
store. Capture metadata includes provider, channel, upstream model, route mode,
and attempt sequence. A failed provider call and a failed capture are separate:
a provider error may still have a complete capture, while a capture write error
marks the attempt/session failed without replacing the probe's execution result.

Provider observations register a lifecycle. An observer registers its lifecycle when
created and unregisters after the final queued callback completes. Probe
finalization waits for registered observations and in-flight writes before
completing the capture; a probe that never starts an observer does not wait for
one. This drain has no independent deadline: if a registered response body is
never consumed or closed, finalization can continue waiting. It does not imply
that unobserved bytes have been captured.

The evidence boundary remains the bytes and events actually observed. A
`completed` capture with no response body cannot establish that a provider
returned a complete response; inspect parts and termination metadata together.
Regression evidence lives in `internal/control/probe_capture_test.go`,
`internal/container/debug_capture_test.go`.

## 空完成排查顺序

先读取 structured request log，确认请求的 `error_code=upstream_empty_completion`、HTTP status、usage 和每个 attempt 的错误分类与决策；request log 是索引和结果记录，不能替代 Provider 原始 body。再使用管理员 debug capture API 按同一 `request_id` 查询全部 capture，并按 attempt sequence 关联原始 request/response：

```text
GET /api/logs/<request-id>
GET /api/debug-captures?request_id=<request-id>
GET /api/debug-captures/<capture-id>/download
```

完整的 raw capture 可以核对实际观察到的 response body、usage、finish reason 和 message 字段是否真的缺失，从而区分空完成与 `upstream_content_filter`。capture 的 metadata、解析结果或只有部分 body 都不是完整 raw 证据；capture 查询时仍为 `active`、已为 `failed`、已过期或已不存在，缺少应有的 response parts，或终止信息不完整时，必须标记证据不完整，不能把 metadata-only 结果当作完整 raw 证据。

raw capture 只记录 Gateway、支持的 provider observer 或 transport reader 实际观察到的字节和事件。它不声称是 TLS/socket wire capture，也不补齐已经被 transport 移除的 framing、未暴露的 trailers，或在取消、超时、进程失败、连接关闭前从未观察到的数据。

## 快照名额耗尽排查顺序

`/api/health` 的 `debug_capture.error` 为 `session_limit_reached` 时，先区分“名额被未释放的捕获占用”和“只是数据库行数看起来很多”。

1. 读取 health：`running` 为 `true`、`error` 为 `session_limit_reached`，而 `active` 可能为 `0`。`active/completed/failed` 统计的是未过期的数据库行，不是内存中的准入槽位；名额耗尽与过期清理已完成可以同时发生。
2. 用 `GET /api/debug-captures?state=active` 查看仍未终结的会话。如果 `active` 为 `0` 却仍报名额耗尽，说明槽位被运行期持有的会话占用，而不是被持久化行占用。
3. `no_observation_source` 会终结 attempt，因此该会话通常是 `failed` 而不是 `active`；只用 `state=active` 查不到它。名额异常时同时查询 `state=failed`（必要时再查 `completed`），对每个会话按 `GET /api/debug-captures/:capture_id` 检查 attempt。
4. `termination` 中含有 `no_observation_source` 表示该 attempt 没有收到 Provider 完成回调，事件 detail 会写明是“来源从未启动”还是“来源启动后未完成”。这只能说明无法据此证明一次完整的请求/响应对换，不等于该 attempt 没有记录任何响应字节：已完成观察的 body、header part 可能仍然存在。必须查看实际记录的 parts 与终止信息，并按「空完成排查顺序」判定证据是否完整。
5. 如果会话长期停在 `active` 且其 attempt 没有 `no_observation_source`，先确认该会话对应的请求是否仍在处理中（`request_id` 关联 structured request log）。请求未结束时等待其终结，不要通过重启掩盖。
6. 修复后用一个新请求验证：新的 capture 应能创建并终结，且 `error` 字段清空。已过期的历史会话无法补回原始正文，结构化 request log 不能替代 raw body。

名额泄漏的回归证据是 `internal/debugcapture/runtime_test.go`（健康状态报告耗尽与释放后清除）与 `internal/gateway/capture_test.go`（无观察来源、提前取消、in-flight 回调排空后均不留下 pending），端到端为 `internal/container/debug_capture_lifecycle_test.go`。
