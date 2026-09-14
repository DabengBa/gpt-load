# 设计方案：协议感知的手动模型测活

状态：实现基线
关联：`internal/control/model_probe.go`、`internal/control/credential_probe.go`

## 1. 目标与边界

模型测活是一次由用户明确发起的、面向单个 `(group, model)` 目标的真实上游生成请求。它用于观察某个目标和本次选中的凭据是否能够完成一次可用的文本生成，不是流量健康判决器。

- 分组模型页和调度中心共用同一个 `POST /api/model-probe` 原语。
- 批量只是多个目标；前端按 8 个目标顺序分块发送，服务端单次最多接受 64 个目标，服务端内部并发上限为 4。
- 每个目标只执行一次由 Provider contract 指定的协议和路由，不跨协议、不得切换到 embeddings/rerank，也不做协议回退。
- 测活结果只观测：不改变可用性、冷却、拉黑、失败计数、调度权重、恢复状态、用量或成本统计。
- 不存在后台、定时、挂载触发、Tab 切换触发或可见性变化触发的测活；请求只由显式点击产生。
- 请求日志使用已有的 `operation=probe` 和 `request_id`。日志写入仍是尽力而为，不能把请求 ID 当作日志一定已落库的保证。

测活不覆盖新建分组草稿：只有已保存并编译到分组视图中的模型才可测。凭据页的「测试连接」是另一个显式入口，针对用户点选的单个凭据。

## 2. Provider contract

`ProviderBinding.ProbeContract` 是代码拥有的权威测活合同，至少声明：

- 单一的 client protocol（`Protocol`）；
- 该协议允许的最低输出预算（`MinOutputTokens`）。

它不直接声明最终的 upstream wire protocol，也不声明 native 或 converted route mode。分组目标构建时先取得 contract，再由 route resolver 按目标模型解析唯一的 route；route 决定 native/converted，converted route 再决定最终 upstream wire shape。若 contract、模型或 route 不可用，返回 `inconclusive`，不尝试另一个协议。

`ProviderKind` 只负责选择 Provider adapter，不能用来推断测活协议。

当前生成协议的请求形状如下：

| Provider protocol | 请求字段 | 输出预算 | 说明 |
|---|---|---|---|
| `openai-responses` | `input`、`max_output_tokens` | contract 预算，原生 Responses 最低为 16 | 原生 Responses wire 不发送 Chat 的 `messages` 或 `max_tokens` |
| `openai-completions` | `messages`、`max_tokens`（兼容网关）或 SDK 对应的 Chat 预算字段 | contract 预算 | converted route 的最终 upstream wire 按目标上游的 Chat 形状发送 |
| `anthropic` | `messages`、Anthropic 的 `max_tokens` | contract 预算 | 使用 Anthropic 原生响应形状 |
| `gemini` | `contents`、`generationConfig.maxOutputTokens` | contract 预算 | 使用 Gemini 原生生成请求形状 |

上表描述的是常见的 wire 形状，不把 client protocol 与最终 wire 混为一谈：native Responses route 使用 `input` 和 `max_output_tokens`；若 route resolver 选择 converted route（包括把 Responses 转为 Chat 的 route），最终 upstream 请求使用 `messages` 及该 Chat route 的预算字段（例如 `max_tokens`）。因此 `MinOutputTokens` 是 contract 的预算下限，不能替换成对所有最终 wire 的统一字段或统一数值。

## 3. 低成本真实问题与答案判定

所有生成型测活使用如下短问题：

> What is 2 + 2? Please answer briefly.

它不是固定的连通性 `ping`，也不是只检查 HTTP 状态。通过必须同时满足：

1. HTTP 响应属于成功范围；
2. 响应能够按 contract 指定协议解析；
3. 响应包含非空、可用的生成文本。

协议专用提取规则：

- Chat 检查 `choices` 中的 message content 或 completion text；
- Responses 检查 `output` / `output_text` 中的文本内容；
- Anthropic 检查 `content` 文本块；
- Gemini 检查非 `thought` part 的文本，只有面向用户的文本才算答案。

文本判断与结果分类必须区分：

| 证据 | outcome | reason |
|---|---|---|
| 合法协议响应，含非空生成文本 | `passed` | `null` |
| 合法协议响应，但文本为空或不存在 | `failed` | `no_answer` |
| 2xx 但不是所选协议的合法响应形状，或无法解析 | `failed` | `invalid_response` |
| 凭据无效、模型不可用 | `failed` | 对应 reason |
| 限流、超时、上游错误、协议/请求不兼容或无法归因 | `inconclusive` | 对应 reason |
| 目标不存在、无可调度凭据 | `inconclusive` | `target_unavailable` / `no_schedulable_credential` |

`passed` 没有 reason；前端必须分别展示「无答案」「响应无效」「协议/请求不兼容」「上游错误」和「无法判断」，不能将所有非 2xx 或空文本合并成同一条文案。

## 4. 手动入口与批量行为

### 4.1 单目标

- 分组模型页只对已保存模型显示可用的测活动作；草稿模型提示先保存。
- 调度中心行内动作明确对应当前 `(group, model)`；停用分组需要用户确认后才发送。
- 凭据页「测试连接」只对用户点击的 credential 发起一次请求。
- 任何入口均保留显式点击确认，不在页面初始化、刷新或路由变化时发送请求。

### 4.2 批量

- 批量范围是当前可见行，去重键为 `(group_id, model)`。
- 前端以 8 为分块大小顺序发送；停止只阻止后续分块，已发出的请求和已返回结果保留。
- 服务端单次请求上限为 64；前端不通过扩大请求或并发请求绕过该上限。
- 结果弹窗保留通过/失败/无法判断汇总、每行 reason、使用的凭据、协议、route mode、状态码、耗时和 request ID。

## 5. 日志、副作用与手动恢复

测活日志记录 `operation=probe`，并明确标记为不适用的 usage/pricing。测活不会调用健康 mutation，也不会因为 `failed` 或 `inconclusive` 自动冷却、拉黑或恢复凭据。

凭据测试成功时，后端可以返回 `can_restore` 和一次性的 `restore_proof`。前端只有在结果为 `passed`、后端允许恢复且用户再次点击确认后，才调用独立的 restore endpoint。恢复边界如下：

- 测活本身永远不恢复凭据；
- 非 `passed`、缺少 proof 或 proof 过期时不显示恢复动作；
- restore proof 绑定凭据身份、当前状态、目标协议、route mode 和输出预算；
- 恢复请求失败或状态冲突时保留测试结果，并要求用户重新测试或手动刷新；
- restore 成功后只更新该凭据的明确状态，不把测活结果扩展为整个分组的健康结论。

## 6. 前端合同

前端资源层对模型和凭据结果使用严格字段投影。reason 枚举至少覆盖：

- `passed`（由 `reason=null` 表示）；
- `no_answer`；
- `invalid_response`；
- `probe_incompatible`（显示为协议/请求不兼容）；
- `upstream_error`；
- `inconclusive` outcome 及其限流、超时、未知等原因。

模型测活弹窗和凭据测试弹窗都只能读取后端返回结果，不自行推断协议、不重试、不发起第二个请求。现有批量进度、停止、日志深链、点击确认和响应式布局保持不变。

## 7. 验收清单

- 前端和设计文档不再包含已退役的运行时键或分组测活字段。
- `pnpm --dir web run type-check` 通过。
- `pnpm --dir web run lint`、`format` 和 `build` 按 package scripts 执行并记录结果。
- `git diff --check` 通过。
- 手动确认模型单测、凭据测试、批量分块、停止后保留结果、失败 reason 展示及显式 restore 边界；没有页面自动请求。
