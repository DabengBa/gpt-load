# 参数覆盖规则语义

## 规则结构与 UI 映射

完整结构：

```json
{ "match": { "protocol": "...", "model": "..." }, "remove": ["/some/path"], "set": { "some_parameter": "value" } }
```

Web UI 拆成三个区域：「匹配」填 `match.protocol`/`match.model`；「参数」JSON 框只填 `set` 对象；「添加删除路径」填 `remove` 的 JSON Pointer。不能把外层数组或含 `match`/`remove`/`set` 的完整 JSON 粘进「参数」框（会报「至少添加一个参数」或 JSON 校验错误）。

## 三个模型名称

- `client_model`：客户端实际请求的模型名，`match.model` 匹配它；
- `alias`：模型别名配置；
- `id`：分组发往 Provider 的上游模型名。

规则匹配客户端模型名，不匹配上游 `id`（例：客户端请求 `test`、上游 `qwen3.8-flash` 时 `"match": {"model": "test"}`）。

## 实际客户端协议

`match.protocol` 必须等于 request log 的客户端 `protocol`，不凭 endpoint/Provider 名推断：`openai-completions`（`/v1/chat/completions`）与 `openai-responses`（`/v1/responses`）是两个不同匹配值，同一分组同一模型也不交叉命中；`anthropic`/`gemini` 按日志原值。

## 覆盖发生在协议转换前

参数覆盖先作用于客户端协议请求，之后才路由和协议转换：Responses 请求的 `reasoning.effort` 不是顶层 `reasoning_effort`；Provider 转换层可能改字段名；报错中的字段不一定会原样出现在客户端请求里；最终以上游 raw request body 为准。Qwen/DashScope 类供应商 `reasoning_effort` 与 `thinking_budget` 互斥，只能选一个，且需先确认供应商接受哪种字段与模型支持哪些值。

## 已验证规则示例（Qwen Responses）

```json
{
  "match": { "protocol": "openai-responses", "model": "test" },
  "remove": ["/reasoning", "/reasoning_effort"],
  "set": { "thinking_budget": 16384 }
}
```

语义：删除 Responses 客户端请求中的 `reasoning` 对象与可能的顶层 `reasoning_effort`，只设 `thinking_budget`。`16384` 只是该场景预算值，不是通用默认。Provider 明确要求 `reasoning_effort` 时用相反策略（删预算字段只留一个合法 `reasoning_effort`）；依据官方契约或 raw 证据选择，绝不同时 set 两个。

## 排查顺序

```text
request ID → request log 的客户端协议/模型 → 分组模型 client_model 与 upstream id
→ raw upstream request → 最小单字段覆盖 → 新 request ID 验证
```

已踩过的坑：规则写 `openai-completions` 而真实日志是 `openai-responses` → 不命中；上游请求是 `reasoning.effort`，转换层报 `reasoning_effort` 与 `thinking_budget` 冲突；改为真实协议 + 按 `client_model` 匹配 + 删 `/reasoning` 只设 `thinking_budget` 后成功。
