# 设计方案：模型测活与凭据测试

状态：实现中，等待最终集成验收
关联：`docs/design/model-route-central-scheduling.md`、`internal/control/model_probe.go`、`internal/control/credential_probe.go`

## 1. 目标

1. 管理员明确点击「测活」或「测试连接」时，对目标最多发送一次真实上游请求。
2. 测活使用分组已经声明的 Provider 探测协议，不从数据面协议顺序推断，也不跨协议回退。
3. 只有协议正确、可解析且包含非空可见生成文本的响应才算通过；HTTP 2xx 本身不是证据。
4. 探测流量只用于诊断和请求日志，不进入正常用量、成本、凭据健康计数或熔断统计。
5. 普通请求仍按现有失败策略重新拉黑凭据或路由条目；黑名单到期由本地定时任务释放，不再通过上游存活探测恢复。

## 2. API

### 2.1 模型测活

```text
POST /api/model-probe
{ "targets": [ { "group_id": 12, "model": "gpt-4o" } ] }

{
  "results": [{
    "group_id": 12,
    "group_name": "openai-main",
    "model": "gpt-4o",
    "outcome": "passed" | "failed" | "inconclusive",
    "reason": null | "invalid_credential" | "model_unavailable" | "rate_limited"
             | "timeout" | "upstream_error" | "probe_incompatible"
             | "unknown" | "target_unavailable"
             | "no_schedulable_credential",
    "protocol": "openai-completions" | "openai-responses" | "anthropic" | "gemini" | null,
    "route_mode": "native" | "converted" | null,
    "status_code": 200 | null,
    "latency_ms": 412 | null,
    "credential_id": 33 | null,
    "credential_label": "sk-…c3d4" | null,
    "recovered": true | false,
    "log_id": "8f1b…" | null,
    "tested_at_ms": 1760000000000
  }]
}
```

- 请求目标数为 1–64；重复的 `(group_id, model)` 在服务端去重，结果保持去重后的输入顺序。
- 每个目标只解析一个 `ResolvedTarget.ProbeRoute(model)`。分组、模型、协议契约不满足时返回 `inconclusive`，不发送请求，`log_id` 为 `null`。
- 可调度凭据按现有候选规则选择 ID 最小者。没有可调度凭据时返回 `no_schedulable_credential`，不改变任何运行态。
- 单目标执行错误只影响该目标，不失败整批；服务端并发上限为 4。
- `recovered=true` 只表示本次成功探测实际清除了被测路由条目的当前失败状态；否则为 `false`。

### 2.2 凭据测试

```text
POST /api/groups/:group_id/credentials/:credential_id/test
→ { outcome, model, protocol, latency_ms, reason, recovered, log_id, tested_at_ms }
```

- 订阅凭据继续禁止手动测试。
- 测试成功后，只有凭据身份、版本、指纹、失败代次、冷却视图和当前目标签名都与本次读取一致时，才清除该凭据的黑名单/失败状态。
- `recovered` 由同一次操作直接返回；前端不再发起第二次「测试通过后恢复」请求。
- 失败和无法判断结果不写入健康或恢复状态。成功但状态已被其他操作改变时，返回 `recovered=false`，不覆盖更新后的状态。

## 3. 探测协议与证据

Provider 通过 `ProviderBinding.ProbeContract` 声明单一生成式探测协议和最低输出预算；`ResolvedTarget.ProbeRoute(model)` 只返回该 Provider 对该模型支持的单一路由。Provider 类型用于选择 Bifrost 适配器，协议用于选择探测线协议，两者不互相替代。

当前协议矩阵：

| Provider/渠道契约 | 探测协议 |
|---|---|
| 官方 OpenAI | `openai-responses` |
| OpenAI Compatible | `openai-completions` |
| Anthropic | `anthropic` |
| Gemini、Vertex Gemini | `gemini` |
| 没有生成式协议契约的订阅渠道 | 不支持 |

探测提示为 `What is 2 + 2? Please answer briefly.`，所有生成式探测共用 contract 声明的 128 输出预算——思考型模型可能先把小预算全部花在内部推理上而不产出可见文本，因此预算必须明显高于各家最小输出限制。协议请求字段由最终 wire 语义决定：

- Chat 使用必需的 `max_tokens=128`；
- 原生 Responses 使用 `/responses`、`input` 和 `max_output_tokens=128`；
- Anthropic、Gemini 使用各自适配器要求的字段，并同样携带 128 预算与同一探测提示；
- converted 路由保留目标适配器的请求语义，不将 native 字段强行写入转换后的协议。

`internal/execution/bifrost/probe.go` 按最终客户端协议检查响应并记录答案/响应形状证据：

- Completions 只接受 `choices[].message.content` 的非空字符串或非空文本 part；
- Responses 只接受 `output[].content[].text` 的非空字符串；
- Anthropic 只接受 `content[].text`；
- Gemini 只接受 `candidates[].content.parts[].text`。

- Gemini 的 `thought: true` part 是内部思考，不算用户可见答案；因此 thought-only 响应是合法的 `no_answer`，不能误判为协议损坏。
- Native passthrough 的响应先按 `Content-Encoding` 解码并受统一响应体上限约束，再进行协议提取；压缩响应不能绕过证据检查。

严格的 wire carrier 检查仍拒绝跨协议响应；空数组、`null`、空白文本、错误 carrier、错误协议响应、畸形 JSON 和仅有 2xx 状态都不能通过。生成文本只作为执行层证据，不回传原文，也不写入日志；失败日志保留经过脱敏的上游错误摘要。

## 4. 日志与统计隔离

已执行的手动探测产生一个 UUID `request_id`，作为响应中的 `log_id`，并写入既有请求日志，`operation=probe`。没有真正执行上游请求的目标不伪造日志 ID。日志写入沿用 best-effort 语义。

探测日志满足以下约束：

- `access_key_id=0`，表示控制面操作；
- 有且只有一次实际执行的 attempt；
- `action=terminate`、`effect=none`，因为探测不提交数据面判决效果；
- 通过响应的 `model_consistency=unknown`，失败响应使用 `not_applicable`；
- 用量和定价使用 `not_applicable`，不生成用量 journal；
- `operation=probe` 在用量、成本、分组请求数、凭据窗口统计和健康成功率查询中全部排除。

诊断日志可以显示探测曾使用某凭据；这不等于正常用量或健康计数。

## 5. 运行态恢复

### 5.1 本地定时释放

运行时只启动黑名单本地释放维护，不启动验证 worker、验证 ticker 或自动上游探测。`blacklist_release_seconds` 默认 `3600`，由系统设置页编辑。

凭据和模型路由条目分别保存 `BlacklistReleaseAt`，与 `CooldownUntil` 独立：

- 普通请求触发黑名单时，在同一个注册表变更中记录失败、黑名单和释放截止时间；
- 释放维护只访问本地注册表，截止时间到达后清除黑名单和当前失败状态；
- 凭据不是 ready 状态时，即使截止时间到达也保持黑名单，避免无效或需要重新授权的凭据重新进入流量；
- 冷却状态不被释放维护强行清除，仍按自身截止时间处理；
- 模型路由条目和凭据的失败版本/代次用于拒绝过期恢复视图。

健康接口对凭据黑名单返回：

```json
{ "automatic": true, "mode": "scheduled_release", "at_ms": 1760003600000 }
```

冷却凭据仍返回 `mode= cooldown_expiry` 和 `cooldown_until_ms`。健康页、凭据页和调度中心显示本地释放时间，不再显示等待验证探测的文案。

### 5.2 显式成功恢复

成功的显式凭据测试只恢复被测试的凭据；成功的模型测活只恢复被测试的 `(group, entry)`。恢复在同一协调器内完成，并且返回值只在注册表实际发生清除时为 `true`。

检查顺序包含：

1. 当前分组仍存在且目标模型仍存在；
2. 当前 Provider、协议、路由模式、模型、代理和 Header 规则签名仍匹配；
3. 凭据 identity/version/fingerprint/encrypted value 和 failure generation 仍匹配；
4. 条目 failure version 仍匹配；
5. 当前仍是被测的失败状态。

任一检查失败都不清除新状态。成功恢复清除当前问题状态，但保留滚动统计桶，避免把历史窗口整体重置。

## 6. 前端行为

- 分组模型页只允许对已保存模型行测活；调度中心对当前可见行提供单行和批量测活，批量按钮显示真实行数并按 8 个目标一块顺序发送。
- 弹窗展示协议、路由模式、结果、原因、凭据、耗时、`recovered` 和请求 ID；请求 ID可复制并深链到日志详情。
- 凭据测试弹窗直接展示 `recovered`，不再显示 `can_restore`、`restore_proof` 或第二个恢复按钮。
- 订阅凭据不显示测试动作。
- 页面挂载、刷新、标签切换、可见性变化和健康轮询只读取控制面 API，不发送上游探测。
- 所有支持的 locale（`en-US`、`zh-CN`、`ja-JP`）同步移除已删除的验证设置、验证模型和验证恢复文案。

## 7. 配置删除

`validation_interval` 和 `validation_model` 已从运行时设置、分组 API、状态快照、loader、存储模型和前端设置中直接删除。不添加迁移、loader 忽略或兼容回退；升级前由部署操作者清理旧持久化数据。

## 8. 验收证据

后端验收至少覆盖：

1. 协议矩阵、单目标最多一次请求、严格生成文本证据和错误 carrier 拒绝；
2. 探测失败/无法判断不变更健康、冷却、黑名单、恢复、用量或成本；
3. 成功恢复的身份/版本/指纹/失败代次/条目版本和目标签名冲突；
4. 本地黑名单释放、冷却独立性、非 ready 凭据保持黑名单；
5. `operation=probe` 日志和所有统计查询隔离；
6. 无后台验证 worker/ticker，且正常请求失败仍能重新拉黑。

前端验收使用仓库已有脚本：

```text
pnpm --dir web type-check
pnpm --dir web build
pnpm --dir web lint
pnpm --dir web format
pnpm --dir web verify:health-projection
```
