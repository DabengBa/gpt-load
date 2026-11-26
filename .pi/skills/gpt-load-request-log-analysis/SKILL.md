---
name: gpt-load-request-log-analysis
description: "在 DabengBa/gpt-load 项目内下载、校验并分析 request logs 与 Debug Communication Capture；要求所有 Provider、所有 attempt 都有未经修改的原始请求/响应证据。"
---

# GPT-Load Request Log Analysis

本 skill 只适用于当前项目：`DabengBa/gpt-load`。

## 触发条件

- 下载或分析 Hostinger 上 GPT-Load 的请求日志；
- 解释 Provider retry、fallback、protocol error、SSE 错误或 token/cost 结果；
- 检查某个 request ID 的所有 Provider attempt；
- 验证一次运行是否满足「所有 Provider、所有 attempt 都有未经修改的原始请求/响应证据」的证据要求；
- 设计或评审 raw request/response capture 功能。

## 证据要求

1. 所有 Provider、所有 attempt（首次、retry、fallback、credential refresh、最终失败）都要有 raw evidence：原始 request headers/body、原始 response status/headers/body（以 transport reader 观察点为准，不把 TLS/socket 或已被 transport 移除的 framing 冒充 raw）、SSE 原始 framing、终止信息（EOF/close/timeout/cancellation/read error）。
2. 非流式响应也保存原始 JSON/body。解析后的 JSON/事件/错误分类只是分析索引，不能替代 raw bytes。
3. 任一 attempt 缺 raw evidence，结论必须是「证据不完整」，不能把 metadata-only attempt 或最终成功 attempt 当作完成。
4. 运行时事实：raw capture 是 opt-in（Unix 环境 `DEBUG_CAPTURE_ENABLED=true` 才启用，默认关闭），session 24 小时过期、异步清理；retention 不等于物理删除（DB pages/backups/exports 不做安全擦除）。capture 缺失时先区分「未启用采集」与「采集失败/过期」，两者都不能冒充证据。深度约束见 `references/capture-runtime.md`。
5. evidence 边界：只能证明 captured bytes；`no_observation_source` 表示排空后无 completion，不能据 executor result 断言 wire 交换。

## 执行流程

1. 确认仓库上下文（`pwd`、`git branch --show-current`、`git status --short`），必须在 `/mnt/projects/repos/gpt-load` 内工作；不要把 raw artifact 写入 tracked 文件。
2. 先取 structured request log：`GET /api/logs/<request-id>`，保存完整响应到本地 `tmp/` 并生成摘要（attempt sequence、Provider/channel、credential、status、retry decision、error code、usage/cost）。不要把 structured log 误称为 raw log。
3. 查询全部 debug captures：`GET /api/debug-captures?request_id=<request-id>`（管理员 `AUTH_KEY`，data-plane access key 不可用），不假设只有一条；记录 capture ID/state/created/expires/error/attempt 列表/是否有 raw parts。不存在、过期或 failed/incomplete 一律标记证据缺失；不自动改配置、重启或部署。
4. 下载 raw ZIP 到本地 `tmp/`（`GET /api/debug-captures/<capture-id>/download`）；不写入 tracked path、不提交、不上传。终端与报告只输出脱敏后的长度、hash、状态和事件摘要，不打印 Authorization/Cookie/API key/完整 prompt/完整响应。
5. 按 attempt 建证据矩阵（对照表与不完整判据见 `references/evidence-matrix.md`）。
6. 在 raw 副本上分析原始 SSE framing（hash/长度、按原始 framing 拆帧、event name、JSON 可解析性、terminal event 与 terminal 后事件、非法帧），把帧序号与 Gateway protocol error 对齐；不得用重新序列化的输出替换 raw artifact。
7. 输出结论：区分 request log 证明了什么、raw capture 证明了什么、每个 attempt 是否有完整 raw evidence、缺哪些 part、能否定位 offending frame、是否满足证据要求。只有全部 attempt 完整时才能输出「满足」，否则输出「证据不完整」与缺口列表。

## 修改边界

除非用户明确要求实施修复，否则不得修改 Provider 轮询/retry/fallback、Gateway stream parser、capture 存储与 retention、部署配置或无关业务代码。发现缺口先给出文件、代码路径、受影响 Provider/channel、attempt 类型和验证方案，等待实施确认。

## 当前运行时验证失败的处理

capture 不存在、failed/incomplete、某 Provider/attempt 缺 raw part 时按证据缺失处理；不得通过假设开关、metadata-only attempt 或最终成功 attempt 推断满足要求。报告列出受影响 capture、attempt、缺少的 part、终止事件和部署/运行时验证结果。
