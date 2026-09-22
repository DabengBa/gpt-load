---
name: gpt-load-hostinger-parameter-override
description: "在 Hostinger 上为 GPT-Load 分组定位并修正 Provider 参数冲突；重点处理客户端协议、客户端模型别名、参数覆盖顺序与 request log/raw capture 验证。"
---

# GPT-Load Hostinger 参数覆盖

本 skill 只适用于当前项目 `DabengBa/gpt-load`，用于在已授权的 Hostinger 实例上定位并修正单个 Provider 分组的请求参数冲突。

## 触发条件

用户要求以下任一事项时使用：

- 直接在 Hostinger 上修改 GPT-Load 分组配置；
- 修复 Provider 报错，例如 `reasoning_effort` 与 `thinking_budget` 同时设置；
- 根据 request ID 反查实际客户端协议、客户端模型名、上游模型名和最终上游请求；
- 验证参数覆盖是否命中并真正改变了上游请求。

## 不可妥协的安全边界

1. 只使用用户已授权的 SSH 目标；本项目自建实例默认是 `vps-kl`，先读取本机 SSH 配置确认，不要猜服务器。
2. 绝不打印、写入 skill 或报告以下内容：SSH 私钥、`AUTH_KEY`、上游 API Key、`Authorization`、Cookie、完整 prompt、完整 raw request/response body。
3. 只输出脱敏后的分组 ID、名称、channel、客户端模型名、上游模型名、Base URL、状态码、字段存在性、长度和 hash。
4. 配置修改通过 GPT-Load 管理 API 完成；不要直接改 SQLite、Docker volume 或容器内配置文件。
5. 参数配置成功后不需要重启容器。除非用户另行明确要求，不要部署、重启、切换镜像或修改代码。
6. 写入前先读取目标分组的完整 `overrides`；更新时保留无关配置。若必须替换整个 `overrides`，确认没有并发修改，并保留旧值以便回滚。

## 关键经验

### 1. UI 的“参数”框不是完整规则编辑器

参数覆盖规则的完整结构是：

```json
{
  "match": { "protocol": "...", "model": "..." },
  "remove": ["/some/path"],
  "set": { "some_parameter": "value" }
}
```

但 Web UI 将它拆成三个区域：

- “匹配”控件填写 `match.protocol` 和 `match.model`；
- “参数” JSON 框只填写 `set` 对象，例如 `{ "thinking_budget": 16384 }`；
- “添加删除路径”填写 `remove` 中的 JSON Pointer。

不能把外层数组或包含 `match`、`remove`、`set` 的完整 JSON 粘贴进“参数”框，否则会出现“至少添加一个参数”或 JSON 校验错误。

### 2. 必须区分三个模型名称

同一个分组可能同时有：

- `client_model`：客户端实际请求的模型名，也是 `parameter_overrides.match.model` 应匹配的值；
- `alias`：模型别名配置；
- `id`：分组发往 Provider 的上游模型名。

规则匹配客户端模型名，不匹配上游 `id`。例如客户端请求 `test`，上游模型是 `qwen3.8-flash`，规则必须使用：

```json
"match": { "model": "test" }
```

不能因为 Provider 名称是 `qwen3.8-flash` 就把匹配模型写成 `qwen3.8-flash`。

### 3. 必须使用实际的客户端协议

规则中的 `match.protocol` 必须等于 request log 中的客户端 `protocol`，不是凭 endpoint 或 Provider 名称推断：

- `openai-completions`：`/v1/chat/completions`；
- `openai-responses`：`/v1/responses`；
- `anthropic`、`gemini` 等按日志中的协议原值填写。

`openai-completions` 和 `openai-responses` 是两个不同的匹配值。即使同一个客户端模型、同一个分组和同一个 Provider，也不会交叉命中。

### 4. 参数覆盖发生在协议转换前

参数覆盖先作用于客户端协议请求，之后 GPT-Load 才执行路由和协议转换。因此：

- OpenAI Responses 请求中的 `reasoning.effort` 不是顶层 `reasoning_effort`；
- Provider 转换层可能把它转换成另一种字段；
- 供应商报错中出现的 `reasoning_effort` 和 `thinking_budget`，不一定会原样出现在 GPT-Load 收到的客户端请求里；
- 最终判断必须以 raw capture 中的上游 request body 为准。

对于 Qwen/DashScope 类供应商，`reasoning_effort` 与 `thinking_budget` 只能选择一个。不要同时 set 两者；需要先确定供应商接受哪一种字段和模型支持哪些值。

## 标准执行流程

### 1. 确认工作区和 SSH 入口

```bash
pwd
git branch --show-current
git status --short
rg -n '^(Host|HostName|User|Port|IdentityFile)\\s' ~/.ssh/config
```

只确认目标和仓库状态，不把私钥内容输出。当前自建实例文档位于 `docs/deployment.md`，默认目标为：

```text
SSH target: vps-kl
container: gpt-load
local API: http://127.0.0.1:3001
```

### 2. 只读确认远端健康和分组

```bash
ssh vps-kl 'sudo -n docker ps --format "{{.Names}}\\t{{.Image}}\\t{{.Status}}"'
ssh vps-kl 'curl -fsS --max-time 10 http://127.0.0.1:3001/health'
```

读取管理员 API 时，令牌只留在远端 shell 变量中，不打印：

```bash
ssh vps-kl 'bash -s' <<'REMOTE'
set -eu
auth=$(sudo -n docker exec gpt-load sh -c '"'"'cat /app/data/auth.key'"'"')
curl -fsS -H "Authorization: Bearer $auth" \
  http://127.0.0.1:3001/api/groups > /tmp/gpt-load-groups.json
# 只输出脱敏后的 id/name/channel_id，不输出凭据或完整响应。
python3 - /tmp/gpt-load-groups.json <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
items = (payload.get("data") or {}).get("items") or []
for item in items:
    print(json.dumps({key: item.get(key) for key in
        ("id", "name", "channel_id", "connection_type", "enabled")},
        ensure_ascii=False, separators=(",", ":")))
PY
rm -f /tmp/gpt-load-groups.json
REMOTE
```

如果没有找到明确目标，不要按名称猜测并写入；继续读取模型和设置，或请求用户确认。

### 3. 读取目标分组的模型和当前设置

对候选分组读取：

```text
GET /api/groups/<group_id>
GET /api/groups/<group_id>/models
GET /api/groups/<group_id>/settings
```

只保留以下非敏感字段：

- 分组 ID、名称、channel ID；
- `params.base_url` 或非敏感 endpoint；
- 每个模型的 `id`、`alias`、`client_model`、`alias_enabled`；
- 当前 `overrides`，确认是否已有规则。

优先通过本机 `curl` 调用远端宿主机 `127.0.0.1:3001`。如果在容器内使用 BusyBox `wget`，注意它通常不支持 `PUT` 的 `--method=PUT` 选项；写入阶段使用宿主机 `curl -X PUT`，不要因为 BusyBox 报错而重复发送或误判服务故障。

### 4. 根据 request log 确定命中条件

如果用户提供 request ID，先使用已有的 `gpt-load-request-log-analysis` skill 下载并分析：

```bash
scripts/fetch-hostinger-request-log.sh <request-id>
```

至少确认：

- 顶层 `protocol`；
- `client_model`；
- 实际 `group_id`、`channel_id` 和 `upstream_model`；
- attempt 的 `route_mode` 和 `upstream_protocol`；
- Provider 错误文本和状态码。

不能只根据用户截图中的模型名或 Provider 名称写规则。

### 5. 用 raw capture 判断字段形状

从 debug capture ZIP 中读取副本，不在终端打印完整内容。每个 attempt 至少检查：

```text
attempts/<attempt-id>/parts/request.headers
attempts/<attempt-id>/parts/request.body
attempts/<attempt-id>/parts/response.headers
attempts/<attempt-id>/parts/response.body
attempts/<attempt-id>/metadata.json
```

记录：

- request/response 字节长度和 SHA-256；
- response HTTP 状态；
- 顶层字段集合；
- `reasoning_effort`、`thinking_budget`、`reasoning.effort` 等目标字段的存在性和值；
- EOF、close、timeout、cancellation 或 read error 终止事件。

如果 capture 是 `failed`/`incomplete`，或者缺少必要 raw part，必须报告 `raw evidence incomplete`。不能把 metadata-only attempt 当作完整证据，也不能因为最终请求成功就声称 capture 完整。

### 6. 生成最小规则

只保留一个供应商接受的推理控制字段。对于本次已验证的 Qwen Responses 场景，正确规则为：

```json
{
  "match": {
    "protocol": "openai-responses",
    "model": "test"
  },
  "remove": [
    "/reasoning",
    "/reasoning_effort"
  ],
  "set": {
    "thinking_budget": 16384
  }
}
```

这条规则的语义是：删除 Responses 客户端请求中的 `reasoning` 对象和可能存在的顶层 `reasoning_effort`，再只设置 `thinking_budget`。`16384` 仅是本次场景选择的预算值；不要把它当成所有模型的通用默认值。

如果目标 Provider 明确要求 `reasoning_effort`，则使用相反策略：删除预算字段，只保留一个合法的 `reasoning_effort`。必须依据 Provider 官方契约或 raw 请求/响应证据选择，不能同时 set 两个字段。

### 7. 写入分组设置

写入前先保存旧 `overrides`，只修改目标分组：

```bash
ssh vps-kl 'bash -s' <<'REMOTE'
set -eu
payload='{"overrides":{"parameter_overrides":[{"match":{"protocol":"openai-responses","model":"test"},"remove":["/reasoning","/reasoning_effort"],"set":{"thinking_budget":16384}}]}}'
auth=$(sudo -n docker exec gpt-load sh -c '"'"'cat /app/data/auth.key'"'"')
curl -fsS --max-time 15 -X PUT \
  -H "Authorization: Bearer $auth" \
  -H 'Content-Type: application/json' \
  --data-binary "$payload" \
  http://127.0.0.1:3001/api/groups/<group_id>/settings
REMOTE
```

实际操作中把 `<group_id>` 和规则中的协议/客户端模型替换成第 3、4 步确认的值。不要把 API 响应原样输出；只打印 `code`、分组 ID、分组名称和脱敏后的 `overrides`。

注意：`PUT /settings` 中提交的 `overrides` 是本次保存的配置集合。若目标分组已有无关 overrides，必须先合并保留，不得只发送新规则导致其他设置丢失。

### 8. 立即读取并验证

写入成功后重新读取：

```bash
curl -fsS --max-time 15 \
  -H "Authorization: Bearer $auth" \
  http://127.0.0.1:3001/api/groups/<group_id>/settings
curl -fsS --max-time 10 http://127.0.0.1:3001/health
```

确认：

- `code == 0`；
- 返回的协议和模型匹配预期；
- 旧的错误规则已被替换，而不是与新规则并存；
- 新规则的 remove path 和 set 字段准确；
- 服务仍为 healthy。

### 9. 用新的 request ID 做行为验证

配置读取成功不等于规则命中。要求用户使用相同的客户端模型名和协议重新请求，并提供新的 request ID。然后重复日志和 raw capture 分析，确认：

1. request log 的客户端 `protocol`/`client_model` 与规则一致；
2. 上游 raw request 只包含供应商选择的一个推理参数；
3. Provider 不再返回互斥参数错误；
4. 最终状态、attempt、retry/fallback 结果正常；
5. 若要验证 raw logging 设计，所有 attempt 都有完整 request/response parts 和终止事件。

## 本次故障的可复用结论

本次初始规则使用了：

```json
"protocol": "openai-completions"
```

但真实 request log 是：

```json
"protocol": "openai-responses"
```

因此规则没有命中。原始上游请求显示的是 `reasoning.effort`，供应商转换层随后报告 `reasoning_effort` 与 `thinking_budget` 冲突。将规则改为真实协议、按客户端模型 `test` 匹配、删除 `/reasoning` 并只设置 `thinking_budget` 后请求成功。

这说明排查顺序必须是：

```text
request ID
→ request log 的客户端协议/模型
→ 分组模型的 client_model 与 upstream id
→ raw upstream request
→ 最小单字段覆盖
→ 新 request ID 验证
```

## 回滚

如果新请求仍失败或出现非预期行为：

1. 不要继续叠加规则；
2. 读取当前设置并与写入前保存的旧 `overrides` 对比；
3. 通过同一个 `PUT /api/groups/<group_id>/settings` 恢复旧 `overrides`；
4. 重新 GET 设置并检查 `/health`；
5. 使用新的 request ID 验证回滚结果。

本 skill 不自动回滚、不自动重启、不自动部署。