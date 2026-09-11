# 设计方案：模型测活（Model Liveness Probe）

状态：待评审
关联：`docs/design/model-route-central-scheduling.md`（调度中心）、`internal/control/credential_probe.go`（凭据测活）

## 1. 需求与目标

### 1.1 需求陈述

1. 分组维护（分组详情 · 模型 Tab）每个模型行增加「测活」按钮，中央弹窗显示结果并给出 log id。
2. 调度中心（监控 · 调度中心）每个模型行增加「测活」按钮，中央弹窗显示结果并给出 log id。
3. 调度中心增加「批量测活」按钮，中央弹窗逐个展示每个分组模型的测活结果，各带 log id。

### 1.2 目标

- **一个后端原语**：对 `(分组, 模型)` 发起一次最小真实上游请求，返回三态结论 + 证据 + log id。
- **批量 = 多目标**：批量不是新机制，只是同一端点的 N 个目标；前端不复用两条代码路径。
- **log id 就是请求日志的 `request_id`**：结果写入既有请求日志（`operation=probe`），弹窗可一键跳到日志详情，不新建"测活历史"表。
- **零副作用**：测活只观测，不改调度权重、不冷却、不拉黑、不恢复、不计入用量统计与凭据健康计数。

### 1.3 明确不做

| 不做 | 原因 |
|---|---|
| 「新建分组」流程里的测活 | 分组未落库：无分组 ID、无凭据池、无编译后的路由目标。新建完成进入分组详情即可测活。 |
| 定时/后台自动测活、测活历史表 | 日志页就是历史；定时探活是另一个产品（健康巡检）。 |
| 针对指定凭据的模型测活 | 已存在：凭据页「测试连接」（`POST /groups/:gid/credentials/:cid/test`）。 |
| 全库批量（所有已保存分组 × 所有模型） | 无界成本。v1 只测"当前可见行"。 |
| 测活失败自动冷却/拉黑凭据 | 测活是人的一次观察，不是流量结论；会污染熔断统计。 |
| 新增哈希、冻结契约、baseline、开关配置 | 无具体失败场景支撑。 |

## 2. 现状与可复用点

| 能力 | 位置 | 复用方式 |
|---|---|---|
| 最小真实探测执行（native route、`max_tokens=1`） | `internal/control/credential_probe.go` `credentialProbeExecutor.Probe`；`internal/execution/bifrost/executor.go` `newProbeRequest` | 完全复用，只替换探测目标与凭据来源 |
| 探测目标构建（协议选择 + 回退协议） | `internal/control/validation.go` `buildGroupValidationTarget` / `validationProtocol` / `validationProbeNeedsProtocolFallback` | 改为可显式传模型的变体 |
| 结果分类（passed/failed/inconclusive + reason） | `credential_probe.go` `classifyCredentialProbeResult` | 直接复用，reason 词表不扩展语义 |
| 失败判决（用于日志证据） | `internal/health` `JudgeExecution` → `Decision{Category,Origin,Scope,...}` | 复用取 Category/Origin/Scope，**丢弃 Effect/Cooldown** |
| 可调度凭据枚举 | `internal/state/registry.go` `CollectCredentialCandidates` | 直接复用（已过滤运行态可用 + 认证就绪，按 (group,id) 稳定排序） |
| 凭据可读标识 | `internal/control/request_logs.go` `Service.CredentialLabels` | 直接复用（返回掩码） |
| 请求日志写入 | `internal/telemetry/requestlog.go` `RequestLogSink`，由 `requestlog.Service` 实现，容器已注册（`internal/container/container.go`） | 控制面注入 sink，`Emit` 一条 `operation=probe` 记录 |
| 日志详情按 ID 直取 | `web/src/features/monitor/LogDetailDrawer.vue`（按 `request_id` 拉详情）+ `monitorLocation({tab:'logs', selected_request_id})` | 弹窗 "查看日志" 直接深链 |
| 日志 UI 已识别 `probe` | `web/src/app/resources/request-logs.ts` `RequestLogOperation` / `operations`；`web/src/i18n/locales/zh-CN/monitor.ts` `logs.operation.probe='健康探测'` | 前端零改动即可渲染（该枚举当前无生产写入方） |
| 中央弹窗范式 | `web/src/features/groups/credentials/CredentialTestDialog.vue`（`AppDialog` + `InlineFeedback` + `dl`） | 同构新建组件 |
| 行内容器 | `web/src/features/models/ModelAliasEditor.vue` 唯一具名槽 `third-column`；调度中心行 `SchedulePanelDetail.vue` `schedule-cell--breaker` | 不新增槽位，把按钮放进现有列 |

### 2.1 关键既有约束（决定了日志方案）

1. `models.RequestLogAttempt` 有 DB 约束 `group_id > 0`、`credential_id > 0`；`action`、`failure_category` 是闭集，且与 web 端枚举一致（`web/src/app/resources/request-logs.ts`）。
2. `internal/requestlog/mapper.go` `validateFrozenObservation`：**只要请求级带归因（group/credential/channel/attempt_sequence 非零），就必须恰好有一条匹配的 attempt**（`AttemptSequence=1`），且请求级与 attempt 级 `upstream_model`、pricing 身份一致。→ "有 group/credential 的日志" 必然 `attempt_count=1`。
3. `internal/requestlog/worker.go`：`AttemptCount == 0` 的行是"只审计、不进任何用量统计"的观察行；有 attempt 的行会进入 `UsageAggregationJournal` → `UsageStat`，并经 `writeRequestLogBatch` 进入 `CredentialAttemptStat`。
4. 网关只写数据面流量；控制面（模型发现、凭据校验、凭据测活）目前**不写任何请求日志**。

## 3. 关键决策

### D1 单一原语：`(group, model)` 探测；批量是同端点的 N 个目标

新增一个非变更型 POST（先例：`POST /api/route/inspect`），不带分组路径参数，天然支持跨分组批量：

```
POST /api/model-probe
{ "targets": [ {"group_id": 12, "model": "gpt-4o"}, ... ] }   // 1..64
→ 200
{ "results": [ {
    "group_id": 12, "group_name": "openai-main", "model": "gpt-4o",
    "outcome": "passed" | "failed" | "inconclusive",
    "reason": null | "invalid_credential" | "model_unavailable" | "rate_limited"
            | "timeout" | "upstream_error" | "probe_incompatible" | "unknown"
            | "no_schedulable_credential" | "target_unavailable",
    "protocol": "openai_completions", "route_mode": "native",
    "status_code": 200, "latency_ms": 412,
    "credential_id": 33, "credential_label": "sk-…c3d4",
    "log_id": "8f1b…" | null,
    "tested_at_ms": 1760000000000
} ] }
```

- 单行按钮发 1 个目标；批量按钮发 N 个目标。响应形状相同，前端只有一条渲染/投影路径。
- 请求级只做**形状**校验（目标数 1..64、`group_id>0`、`model` 非空），形状错误 → 400。
- 逐目标问题**不失败整批**，而是落成该目标的结果：`target_unavailable`（分组不存在/未加载/订阅类型分组/模型不在该分组模型列表）、`no_schedulable_credential`（分组存在但当前无可调度凭据）、`probe_incompatible`（该模型在当前分组无可用探测路由）、`unknown`（内部错误）。
- `log_id` 非空 **当且仅当**确实执行了一次上游尝试（无论 `dispatch_state` 是 `not_sent` 还是 `maybe_sent`）；上表的三个"未执行"原因返回 `log_id: null`。
- **未执行目标的形状（实现补充，原设计未写）**：`target_unavailable` / `no_schedulable_credential` / `probe_incompatible` / `unknown` 的 `outcome` 一律为 `inconclusive` —— 没有任何上游观测，不能宣称「通过」或「未通过」；`log_id`/`status_code`/`latency_ms`/`credential_id`/`credential_label` 为 `null`，而 `no_schedulable_credential` 与 `probe_incompatible` 仍回传已解析出的 `protocol` / `route_mode`（前端需要区分"没法测"与"分组/模型根本不存在"）。

### D2 凭据选择：当前可调度凭据中 ID 最小者，且结果回传凭据身份

- 用 `CollectCredentialCandidates([]uint{groupID}, nil, now)`，取第一个：确定性、可复现、与调度器对"可调度"的定义一致（运行态可用 + 认证就绪）。
- 不引入假访问密钥、不调用调度器权重抽样：测活是"这个分组现在能不能出活"，不是流量占比抽样。
- 响应必须回传 `credential_id` + 掩码标识，弹窗明示"本次使用凭据 X"；否则结论会被误读为整个分组的健康度。
- 池为空 → `no_schedulable_credential`（不改任何运行态，引导用户去凭据页）。

### D3 log id = 请求日志 `request_id`，并把 `operation=probe` 排除出用量/健康统计

- 探测执行本来就生成了一个 `requestID`（`newOperationID`，UUIDv4）；**直接把它当日志行主键**，一个 id 贯穿执行、日志、UI，不新增标识概念。
- 日志行形态：`operation=probe`、`access_key_id=0`（控制面流量，非访问密钥流量）、`attempt_count = 该目标实际执行的协议尝试数（1 或 2）`、`usage_state/cost_state/pricing_completeness=not_applicable`、`estimated_cost=0`。**每个已执行的协议各写一行 attempt**，`attempt.sequence` 按实际执行顺序（协议回退时 1、2），`Usage.AttemptSequence` 指向最终返回的那一次；`validateFrozenObservation` 只要求「恰一条 attempt 匹配归因」，2 行 attempt 不冲突。不回退时 `attempt_count=1`；触发 embeddings/rerank 回退时 `attempt_count=2`。
- 字段映射（实现必须照此写，否则撞 DB/前端闭集约束）：

| 探测结果 | `status` | `model_consistency` | `attempt.failure_category` | `attempt.action` | `attempt.effect` / `retry_directive` | `error_code` |
|---|---|---|---|---|---|---|
| passed | `success` | `unknown`（**必须显式给**：probe 不读上游回报模型，零值 `""` 会被 `validateModelObservation` 拒绝，整行日志被丢弃） | `ok` | `terminate` | `none` | `''` |
| failed / inconclusive | `error` | 不设（`status != success` 时 `normalizeModelObservation` 会强制归零为 `not_applicable`） | `health.Decision.Category` 经 health→telemetry **单源映射**后的 `telemetry.FailureCategory` | `terminate` | `none`（**不落地判决效果**，因为什么都没改） | 探测 `reason` |

  **身份不变量（缺一即 `mapEvent` 拒绝整行，日志与 `log_id` 一起消失）**：`event.UpstreamModel == attempt.UpstreamModel == event.Usage.Pricing.UpstreamModel == 探测 model`（`validateFrozenObservation` 的 boundModel 三方比较）；`event.UpstreamReportedModel == ""`；`stream=false` 且 `first_response_ms=null`（`FirstResponseMs` 只在有 stream 时合法）；探测 model 值必须先 `TrimSpace` 并满足 `validRawModel`（≤255 字节、UTF-8、无控制字符）。

  `dispatch_state` 只可能出现 `not_sent` / `maybe_sent`：`local` 的唯一生产者是 `internal/execution/cpa/adapter.go` 的本地 token 计数，受 `countTokensOperation(OperationCountTokens | OperationResponsesInputTokens)` 门禁，而 probe 用 `OperationProbe`、订阅分组在目标构建阶段即被排除，所以 **probe 的 attempt 不可能是 `local`**，本次不为它扩枚举。（`web/src/app/resources/request-logs.ts` 的闭集缺 `'local'` 是独立的既有缺陷：订阅分组的一次 `count_tokens` 请求会让该行日志详情投影抛错；另立小任务，不在本方案内。）

  `attempt.failure_origin/scope` 取 `Decision.Origin/Scope`；`attempt.upstream_request_id` 取 `result.UpstreamRequestID`；`dispatch_state` / `response_started` / `upstream_protocol` / `status_code` 取 `result`。`error_code` 用测活自己的 reason 词表（与凭据测活一致、且未被 web 端 i18n 映射，纯机器字段），**不是**网关的 `upstream_*` 词表——两套词表分别属于控制面观测与数据面流量；统一是另一个任务。
- **用量隔离（本方案唯一的既有不变量改动，2 行）**：`internal/requestlog/worker.go` `buildUsageAggregationJournals` / `buildUsageStatDeltas` 在既有 `AttemptCount == 0` 判据旁增加 `operation == probe` 判据。
  - `CredentialAttemptStat` **不需要额外改动**：`writeRequestLogBatch` 只对"有 pending journal 的请求 ID"应用 attempt 统计，probe 行不产生 journal，自动被排除。
  - 因此：probe 行只出现在日志页（可查、可深链），不会出现在用量/成本/首页统计与凭据24h成功率里。
- 写入是 best-effort（沿用日志系统语义：worker 未运行/队列满可能丢弃）。`log_id` 仍返回真实 UUIDv4；文档与实现都按"尽力送达"处理，不为此加同步写库路径。
- **凭据测活（`TestGroupCredential`）一并接入同一写日志函数**（已确认）：所有测活都有 `request_id`，口径统一，`CredentialProbeResponse` 新增 `log_id`（与 model-probe 响应同名；否则前端 `web/src/app/resources/credentials.ts` 的白名单投影会静默丢弃它），凭据页展示该 ID 并提供日志深链。代价是既有路径开始产生日志行，见 §6.6。

### D4 无副作用

- 不写注册表运行态：不冷却、不拉黑、不恢复、不更新失败计数；`Decision.Effect` / `CooldownUntil` 只读不落地。
- 不写 `access_key` 配额、不写 `CredentialAttemptStat`、不进 `UsageStat`。
- 不包装 `auditMutation`（与凭据测活一致：非变更操作）。

### D5 可测活的模型范围 = 该分组已保存模型

- 目标模型必须存在于运行时快照 `state.GroupView.Models[].ID`（即分组维护页已保存的行；调度中心行的 `entry.model_id` 天然满足）。
- 分组维护页里**未保存**的行（草稿新增/改名的行）按钮禁用并提示"先保存"：未落库模型不会被编译进路由目标，测了也是假结论。
- 订阅类型分组、以及该模型在 `ResolvedTarget.ModeForModel(proto, OperationProbe, model)` 下不可探测 → `probe_incompatible`。

### D6 批量：前端分块，服务端只保证单请求有界

- 服务端：单请求 ≤64 目标，内部并发上限 4（用带缓冲 channel 做信号量；`golang.org/x/sync v0.22.0` 虽已在 `go.mod`，此处不需要它，不提升为直接依赖），每个目标由该分组的超时配置约束。
- 前端：把批量目标切成 8 个一块**顺序**发送，逐块追加结果，弹窗显示 `已完成 n/N`；「停止」= 不再发送后续块（已收到的结果保留），不需要服务端取消语义。
- 顺序追加 + 每块独立请求 ⇒ 中途失败/停止不会丢已完成的结果，也不会让一次 HTTP 请求长时间挂着。

## 4. 后端设计

### 4.1 涉及文件

| 文件 | 改动 |
|---|---|
| `internal/control/model_probe.go`（新） | `ModelProbeRequest/Response`、`Service.ProbeGroupModels`、逐目标执行与批量编排、日志写入 |
| `internal/control/credential_probe.go` | `credentialProbeExecution` 扩展为回传 `requestID` + 已执行 attempts（现状只有 `result/latency/protocol`）；`TestGroupCredential` 复用同一写日志函数并在响应里回传 `log_id` |
| `internal/health` 或 `internal/telemetry` | 把 `internal/gateway/request_log.go` 里未导出的 `telemetryFailureCategory`（`health.FailureCategory` → `telemetry.FailureCategory`）下沉为单源函数（两处都无环：已核实 `internal/health` 不 import `internal/telemetry`），probe 与网关共用 |
| `internal/control/validation.go` | `buildGroupValidationTarget` 增加显式模型参数（`buildGroupProbeTarget(group, model)`；`model==""` 保留"ValidationModel → Models[0]"语义），更新现有 2 处调用 |
| `internal/control/service.go` | `NewService` 增加 `requestLogSink telemetry.RequestLogSink` 参数（容器已注册该类型，自动注入） |
| `internal/control/http_routes.go` | 新增 `control.model-probe.run` = `POST /model-probe`（无 `auditMutation`） |
| `internal/requestlog/worker.go` | 用量聚合排除 `operation=probe`（2 处判据） |
| `internal/control/*_test.go`、`internal/requestlog/model_probe_log_test.go`（新） | control fixture 注入记录型 sink；日志行约束与用量隔离的端到端证据落在 `internal/requestlog`（control 夹具的 `requestLogs/usageStats` 传 `nil` 且不跑 worker，证明不了 `mapEvent` 与 DB CHECK），见 §7 |

### 4.2 执行流程（单目标）

1. `captureProbeContext(ctx)`：取快照 + 注册表（读锁与现有一致）。
2. 分组解析：不存在/未加载/订阅类型 → `target_unavailable`。
3. 模型解析：`model ∈ group.Models[].ID` 否则 `target_unavailable`。
4. 目标构建：`buildGroupProbeTarget(group, model)`，失败 → `probe_incompatible`。
5. 凭据选择：`CollectCredentialCandidates([]uint{groupID}, nil, now)` 取首个 —— 它返回的 `CredentialMeta` 只有 `ID/GroupID/Version/IdentityGeneration`，没有密钥材料，必须再按 ID 调 `SnapshotGroupCredentialEntriesExact(groupID, []uint{id})` 才能构造 `state.CredentialRef`（范式见 `credential_probe.go`）；空 → `no_schedulable_credential`。注意该函数只判「运行态可用 + 认证就绪」，**不判模型/路由可用性**，所以该 reason 的语义是「凭据层无可选」，不能读作「这个分组出不了活」。
6. 执行：`credentialProbeExecutor.Probe(ctx, group, target, ref)`（协议 + 回退协议、native、无 wire 字段），返回 `requestID` + 已执行 attempts（每个已执行协议一项）。
7. 分类：`classifyCredentialProbeResult`。
8. 写日志：`Emit(telemetry.RequestEvent{...})`；`request_id` = step 6 的 `requestID`；attempts 取 step 6 的列表（1 或 2 行，不是固定 1）。
9. 返回结果项。

批量：对目标列表去重后，用信号量（上限 4）并发跑步骤 1–9，按输入顺序组装结果；单目标内部错误 → `unknown`，不影响其他目标。

## 5. 前端设计

### 5.1 新增文件

- `web/src/app/resources/model-probe.ts`：`runModelProbe(client, targets, signal)` + DTO 投影（严格投影，枚举用 `projectEnum`）。
- `web/src/features/models/use-model-probe.ts`：共享状态机（`pending` / `results` / `error` / `stop` / 分块循环），两个页面各持有一个实例。
- `web/src/features/models/ModelProbeDialog.vue`：中央弹窗。`results.length === 1` 渲染详情 `dl`（分组/模型/结果/说明/协议/路由模式/状态码/耗时/凭据/请求 ID/时间），`>1` 渲染列表 + 顶部汇总（通过 N · 未通过 M · 无法判断 K）+ 进度/停止；每行请求 ID 提供复制与「查看日志」。字段名沿用日志页的「请求 ID」（已确认），不自造「日志 ID」叫法。
- 文案一律经 `labels` prop 注入（仓库既有范式：`ModelAliasEditorLabels` / `SchedulePanelDetailLabels`），弹窗组件内不写死 i18n key。

### 5.2 接入点

| 位置 | 改动 |
|---|---|
| `web/src/features/groups/models/GroupModelsTab.vue` | `#third-column` 槽内加「测活」按钮；`savedModelIDs` 不含的行禁用 + 提示；托管 `ModelProbeDialog`；日志深链 `monitorLocation({tab:'logs', selected_request_id: logId})` |
| `web/src/features/monitor/SchedulePanelDetail.vue` | 行内 `schedule-cell--breaker` 加「测活」按钮（与「恢复」并列），emit `probe`(group_id, entry.model_id)；表头区加「批量测活」，emit `probe-all`(去重后的可见行) |
| `web/src/features/monitor/SchedulePanel.vue` | 承接上述 emit，托管 `ModelProbeDialog`（与既有 `recovered`/`saved` emit 范式一致） |
| `web/src/features/groups/credentials/CredentialTestDialog.vue` + `web/src/app/resources/credentials.ts` | 凭据页「测试连接」展示响应里的 `log_id`（`credentialTestResultFields` 加 `log_id`）并提供日志深链 |
| `web/src/i18n/locales/{zh-CN,en-US,ja-JP}/group.ts`、`monitor.ts` | 按钮、弹窗字段、9 个 reason 文案、批量汇总、成本提示（"将对 N 个分组模型各发起一次真实上游请求，可能产生少量费用"） |

调度中心行的语义（已确认）：行 = `(分组, 上游模型 entry)`，正是后端原语的目标。

**批量范围 = 当前可见行**，去重为 `(group_id, model)` 集合：

- 可见行 = 当前详情查询返回、且通过模式过滤后被渲染的那些行：`SchedulePanelDetail.vue` 的 `rows` = `detail.groups[].entries` 按模式（`all` / `primary` / `fallback`）过滤后的结果。可见性由四个上下文共同决定：**外模型下拉（一次只选一个模型）× 访问密钥下拉（候选范围随密钥变化）× 模式过滤器 × 协议/操作**。
- 不可见行 = (a) 其它外模型的行；(b) 同一模型同一密钥下被模式过滤器排掉的行（选"主路由"就看不到回退行）；(c) 其它访问密钥上下文下的行。
- 因此批量是"所见即所测"的切片，不是"该模型在全系统的全部候选"。按钮文案必须把数量写出来：**「测活当前 N 行」**，并在确认文案里明示"将对 N 个分组模型各发起一次真实上游请求"，避免用户把 N 误读为全量。

## 6. 边界与风险

1. **有成本**：每次测活 = 一次真实最小请求（chat 侧 `max_tokens=1`）。UI 必须在按钮与批量确认处明示；不做静默批量。
2. **结论只代表被选中的那个凭据**：弹窗明示凭据身份；分组内多凭据异构（不同上游账号）时不能推广为分组结论。
3. **日志两类失败必须分开**：(a) **确定性失败**——映射字段不合规（成功探测缺 `model_consistency`、`upstream_model` 三方不一致、model 值未规范化）会让 `mapEvent` 拒绝整行，每次都复现，属实现缺陷，必须在 §3 D3 的约束测试里消灭；(b) **尽力送达**——弹窗的文案按"日志 ID（可在日志页查询）"表述，不做"必然存在"的承诺。
4. **日志页会出现 `operation=probe` 行**：用量统计已隔离；日志列表当前不支持按 `operation` 过滤（`internal/control/request_logs.go` `parseRequestLogQuery` 白名单无该字段）。如需按"健康探测"筛日志，属于另一个小任务（后端白名单 + 前端筛选表单），不在本方案内。
5. **并发**：批量上限 64 目标 × 并发 4，单目标受分组超时约束；服务端 `WriteTimeout=0`，前端分块保证单请求时长可控。
6. **既有行为变化（可裁剪）**：凭据测活开始产生日志行与 log id（§3 D3 末段）。

## 7. 验收

后端（`go test -race -count=1 ./internal/...`）：

1. 单目标通过/失败/无法判断三态 + reason 映射正确（表驱动，复用现有 fake executor 模式）。
2. `buildGroupProbeTarget` 显式模型：命中分组模型列表；未列出的模型 → `target_unavailable`；不可探测 → `probe_incompatible`。
3. 凭据选择确定性与空池：多个可调度凭据取 ID 最小；全部黑名单/冷却 → `no_schedulable_credential`，`log_id=null`。
4. **日志行约束（端到端）**：在 `internal/requestlog` 用真实 `requestlog.Service` 灌入 probe 事件，断言 `operation=probe`、`attempt_count ∈ {1,2}`、每个已执行协议恰一行 attempt（`sequence` 连续、恰一条匹配 `Usage.AttemptSequence`）、`action=terminate`、`effect=none`、`failure_category` 落闭集、`model_consistency` 按 §3 D3 表，且 `mapEvent`/`validateFrozenObservation` 不报错、sqlite CHECK 全过。
5. **用量隔离（回归证明）**：同一批内混合 probe 行与普通行，`buildUsageAggregationJournals` 只为普通行产出 journal；`UsageStat` / `CredentialAttemptStat` 无 probe 贡献。
6. 批量：目标数 >64 → 400；并发上限被信号量约束（fake executor 统计在飞数 ≤4）；单目标内部错误不影响其他目标与结果顺序。
7. 路由注册：`control.model-probe.run` 出现在 `control.group-credentials` 白名单测试同款断言里；非 admin 主体访问被拒（`accessKeyControlRoutes` 未放宽）。

前端（`pnpm --dir web run type-check / lint / format / build`）：

8. DTO 投影对 9 种 reason 与 `log_id: null` 都通过严格投影。
9. 分组维护：草稿行按钮禁用；已保存行可测。调度中心：行内单测 + 表头批量，批量分块追加、停止后保留已完成结果。
10. 请求 ID 深链落到日志详情抽屉（按 `request_id` 直取）；完整路径 = `Emit 日志 → request_id → monitorLocation({tab:'logs', selected_request_id})`。

### 行为变化清单（本次交付会改变既有行为的两处）

- 凭据测活（`POST /groups/:gid/credentials/:cid/test`）响应新增请求 ID，并在日志页新增 `operation=probe` 行（已确认接受）。
- 日志页新增 `operation=probe` 行类型：用量/成本/首页统计与凭据24h成功率不受影响（§4.1 的两行隔离 + §7.5 证明）。

## 8. 已确认决策（评审通过）

1. **调度中心的"模型行"** = 详情表格行（分组 × 上游模型）。批量作用于当前可见行（§5.2）。
2. **命名**：按钮用「测活」（用户口径）；弹窗字段沿用日志页的「请求 ID」，不新增"日志 ID"叫法；日志内 `operation=probe` 显示为"健康探测"（现成 i18n）。
3. **凭据测活同步获得请求 ID**：`TestGroupCredential` 复用同一日志写入路径（§3 D3 末段、§6.6），响应回传 `log_id`，并在凭据页展示该 ID 与日志深链（§5.2）。
4. **批量范围** = 当前可见行，按钮文案带真实数量「测活当前 N 行」。更大范围（该模型下所有分组 / 全部分组全部模型）不在 v1，需另立成本上限与进度交互。

## 9. 实现补充（交付时记录）

1. **未执行目标的 outcome = `inconclusive`**（见 §3 D1 末段）：已实现为 `probeWithoutExecution`，与 `probe_incompatible` 在凭据测活里的既有口径一致。
2. **文案注入改为 `useI18n` 直用**：`ModelProbeDialog.vue` 按 `CredentialTestDialog.vue` 的现有范式直接 `t('monitor.modelProbe.*')`，不用 labels prop。设计 §5.1 原本要求 labels prop（其在仓库里的先例 `ModelAliasEditorLabels` / `SchedulePanelDetailLabels` 都是槽内子组件），而本弹窗是独立浮层，且批量文案需要由子级持有的可见行数量做插值。三语文案已齐（zh-CN / en-US / ja-JP）。
3. **前后端字段集合的双向钉法**：TS 侧 `assertNoSecretLikeFields(record, modelProbeResultFields)` 在运行期要求响应 key 集合**恰好**等于清单，投影返回对象用 `satisfies ModelProbeResultDto`，于是“DTO 新增字段但投影没跟上”会在 `type-check` 失败；Go 侧 `TestModelProbeResponseContract` 钉住同一份清单。这比“`Exclude<...> extends never` 类型别名”更直接：不引入额外类型声明，也不会被 `--max-warnings=0` 的 unused 规则误伤。
4. **3A 的必要性已被实现证实**：`web/src/app/resources/credentials.ts` 的 `credentialTestResultFields` 是精确 key 白名单，因此后端给凭据测活响应加 `log_id` 后，前端不同步更新就会对**每一次**测试连接响应抛 `InvalidResponseError`（不是可选优化）。凭据页展示请求 ID 与日志深链已实现于 `CredentialTestDialog.vue` + `GroupCredentialsTab.vue`。
5. **凭据页入参白名单**：`GroupCredentialsTab.vue` 把 `credentialTestResult` 显式收窄后才传给弹窗（`restore_proof` 永不进组件或 DOM），`log_id` 是显式新增的一行，而不是透传。
6. **宿主并发限制**：U001/U002 的 Worker 与 Reviewer 均为串行派发（宿主一次消息只允许一个工具调用），已在裁决日志披露；依赖顺序与写集合隔离本身不受影响。
