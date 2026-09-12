---
name: gpt-load-request-log-analysis
description: "在 DabengBa/gpt-load 项目内下载、校验并分析 request logs 与 Debug Communication Capture；要求所有 Provider、所有 attempt 都有未经修改的原始请求/响应证据。"
---

# GPT-Load Request Log Analysis

本 skill 只适用于当前项目：`DabengBa/gpt-load`。

## 触发条件

用户要求以下任一事项时使用：

- 下载或分析 Hostinger 上 GPT-Load 的请求日志；
- 解释 Provider retry、fallback、protocol error、SSE 错误或 token/cost 结果；
- 检查某个 request ID 的所有 Provider attempt；
- 验证 GPT-Load 是否满足“所有 Provider、所有 attempt 都保存原始 SSE”的设计要求；
- 设计或评审 raw request/response capture 功能。

## 不可妥协的设计契约

1. **所有 Provider** 都必须覆盖，包括普通 Provider、Codex、Claude、Antigravity、Grok 以及后续新增 channel。
2. **所有 attempt** 都必须覆盖，包括首次请求、retry、fallback、credential refresh、重试前后以及最终失败的 attempt。
3. 每个 attempt 必须保存：
   - 原始 request headers；
   - 原始 request body；
   - 原始 response status 和 headers（若收到）；
   - 未经解析、规范化、截断或重编码的原始 response body（以 transport reader 观察点为准，不将 TLS/socket 或已被 transport 移除的 transfer framing 冒充 raw）；
   - SSE 原始 framing：`event:`、`data:`、注释行、空行和原始换行符；
   - EOF、close、timeout、cancellation、read error 等终止信息。
4. 非流式 Provider 响应也必须保存原始 JSON/body；不能只保存 SSE。
5. 解析后的 JSON、事件类型和错误分类只能作为分析索引，不能替代 raw bytes。
6. 默认必须保存 raw capture；不得引入 `--raw`、`DEBUG_CAPTURE_ENABLED` 或类似开关来决定是否保存。管理员 API 只控制谁可以读取，不控制是否记录。实现应始终安装 capture store 和上游 observer。
7. 任一 attempt 缺少 raw evidence 时，分析结果必须是“未满足设计”，不能把 metadata-only attempt 当作完成。

## 两类数据必须分开

### Structured request log

`request_logs` 是统计和业务结果记录，通常通过：

```text
GET /api/logs/:request_id
```

它可用于确认：

- 最终状态和 HTTP status；
- 首字节时间、总耗时；
- attempt 数量、路由和 retry 决策；
- error code、failure category 和 error summary；
- token、cost、pricing receipt。

它**不能**用于重建 Provider 原始 SSE。

### Raw Debug Communication Capture

Raw evidence 必须通过管理员 API 查询和下载：

```text
GET /api/debug-captures?request_id=<request-id>
GET /api/debug-captures/<capture-id>
GET /api/debug-captures/<capture-id>/download
```

这些接口只接受管理 `AUTH_KEY`/管理员身份，data-plane access key 不得使用。

ZIP 预期包含：

```text
manifest.json
attempts/<attempt-id>/metadata.json
attempts/<attempt-id>/parts/request.headers
attempts/<attempt-id>/parts/request.body
attempts/<attempt-id>/parts/response.headers
attempts/<attempt-id>/parts/response.body
```

### Retention

Raw capture 与 `request_logs` 分开存储。raw capture 按现有服务端策略自动清理；当前固定 retention 为 12 小时，并由启动清理和周期性清理共同处理。清理必须只影响 debug capture session、attempt 和 chunks，不得删除 request log、统计、token、cost 或 retry 记录。

## 执行流程

### 1. 确认仓库上下文

```bash
pwd
git branch --show-current
git status --short
```

必须在 `/mnt/projects/repos/gpt-load` 内工作。不要修改全局 pi skill，也不要把 raw artifact 写入仓库追踪文件。

### 2. 先获取 structured request log

按 request ID 请求：

```text
GET /api/logs/<request-id>
```

保存完整 API response 到本地 `tmp/`，并生成单独的摘要报告。摘要必须保留 attempt sequence、Provider/channel、credential、status、retry decision、error code、error summary 和 usage/cost 状态。

不要把 structured request log 误称为 raw log。

### 3. 查询全部 debug captures

使用管理员 API 按 request ID 查询所有 capture，不假设只有一条结果。记录每条 capture 的：

- capture ID；
- state；
- created/expires 时间；
- error；
- attempt 列表及 sequence；
- 是否存在 raw request/response parts。

如果 capture 不存在、capture 已过期或 capture state 为 failed/incomplete，必须在报告中标记为证据缺失。不要自动修改配置、重启服务或部署。

### 4. 下载 raw ZIP

通过管理员 download endpoint 下载每个相关 capture。raw 文件只保存到本地 `tmp/`，不得写入 Git tracked path、提交、上传或复制到公共 issue。

不要在终端或摘要报告中打印 Authorization、Cookie、API key、完整 prompt 或完整响应。分析时读取 raw 文件，但输出脱敏后的长度、hash、状态和事件摘要。

### 5. 按 attempt 建立证据矩阵

对每一个 request-log attempt 建立一行，并关联 debug-capture attempt。至少检查：

| 证据 | 必须存在 |
|---|---|
| logical attempt ID / sequence | 是 |
| Provider/channel | 是 |
| request headers/body | 是 |
| response status/headers | 收到响应时必须有 |
| raw response body | 收到响应数据时必须有 |
| terminal/EOF/close/error metadata | 是 |
| retry/fallback outcome | 是 |

以下任一情况都算 raw capture 不完整：

- 只有 attempt metadata，没有 response body；
- 只有解析后的 JSON，没有原始 SSE framing；
- 只保存最终成功 attempt，缺少失败或被 retry 的 attempt；
- 只保存 client response，没有对应 Provider response；
- provider 返回空 body、非法 JSON、错误 event、terminal 后追加帧，但原始字节未保存；
- capture 因 cancellation、timeout、process failure 或 connection close 提前结束，且未明确记录已保存范围。

### 6. 分析原始 SSE，不修改原文

分析器必须在 raw body 的副本上执行：

- 保留原始字节 hash 和长度；
- 按原始 SSE framing 拆帧；
- 记录 event name、data 长度、JSON 是否可解析、JSON `type`；
- 检查 terminal event、terminal 后事件、EOF、空帧和非法帧；
- 把原始帧序号与 Gateway 的 protocol error 对齐。

不得用 `json.Marshal`、换行归一化或字段重排后的输出替换 raw artifact。

### 7. 输出结论

报告必须明确区分：

1. request log 证明了什么；
2. raw capture 证明了什么；
3. 每个 Provider attempt 是否有完整 raw evidence；
4. 哪些 attempt 缺失哪些 part；
5. 是否能定位具体 offending SSE frame；
6. 当前结果是否满足“所有 Provider、所有 attempt 保存原始 SSE”的设计契约。

只有全部 attempt 的 raw evidence 完整时，才能输出“满足设计”。否则必须输出“未满足设计”以及缺口列表。

## 修改边界

本 skill 本身用于检查、下载、分析和形成方案。

除非用户明确要求实施修复，否则不得擅自修改：

- Provider 轮询、retry、fallback 或错误处理逻辑；
- Gateway stream parser 或 protocol classifier；
- capture 存储模型和 retention；
- 部署配置、环境变量或运行中的服务；
- 与 raw logging 无关的业务代码。

发现缺口时，先给出文件、代码路径、受影响 Provider/channel、受影响 attempt 类型和验证方案，再等待实施确认。

## 当前运行时验证失败的处理

当前实现按设计默认保存 raw capture，并由所有 HTTP Provider adapter 接入上游 observer。运行分析时若 capture 不存在、状态为 failed/incomplete、某个 Provider/channel 或实际 attempt 缺少 raw part，必须按证据缺失处理；不得通过假设开关未打开、metadata-only attempt 或最终成功 attempt 推断满足设计。报告中应列出受影响的 capture、attempt、缺少的 part、终止事件和部署/运行时验证结果。
