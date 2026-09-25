# 架构评审最终方案 — gpt-load（含评审反馈复查）

> 第一版：`docs/analysis/architecture-review-briefing.md`（本文件已合并评审意见并定稿）。
> 可视化报告：`%TEMP%\architecture-review-20260925-1630.html`。
> 复查日期：2026-09-25。所有论断均附文件/行号或提交号证据。

## 1. 项目介绍

**gpt-load** 是自托管 AI 网关：应用侧只需要一个 base URL 和一个 AccessKey，
上游供应商、账号、凭据、模型与路由策略全部在管理面配置。本仓库为
`DabengBa/gpt-load`（上游 `tbphp/gpt-load` 的维护分叉），默认集成分支 `dev`。

- **技术栈**：Go 1.27 · gin · gorm + SQLite（glebarez）· bifrost/core 执行适配 · dig；
  前端 Vue 3 · reka-ui · vue-query · Vite + pnpm · Playwright e2e；单二进制内嵌 UI。
- **领域模型**（`docs/design/unified-group-scheduling-plan.md` 为产品规则）：
  `Group ── 1 Credential ── Route Entries(weight, priority, circuit-breaker)`；
  `AccessKey + Protocol + Operation + ExternalModel → 候选 Route Entries`。
- **两个面**：数据面 `gateway`（准入→调度→转发→重试→观测）；
  管理面 `control`（30k LOC，85 文件的资源 CRUD/运维 API）。

## 2. 评审反馈复查结果

### 2.1 #3「删 800 行」→ 修正为约 300–400 行，证据如下

幂等/非幂等**成对存在**的只有 4 对（非 6 个方法都有双写）：

| 普通入口 | 幂等入口 | 重复段 |
|---|---|---|
| `CreateAccessKey` (access_keys.go:270) | `CreateAccessKeyIdempotent` (access_key_idempotency.go:36) | 校验+normalize ~50 行 + tx 体 ~45 行，逐行同构 |
| `CreateGroup` | `CreateGroupIdempotent` (group_idempotency.go:50) | 同构 |
| `ConnectGroupCredentials` | `ConnectGroupCredentialsIdempotent` (credential_connections.go:66) | 同构 |
| `ImportGroupCredentials` | `ImportGroupCredentialsIdempotent` (group_idempotency.go:229) | 同构 |

`RotateAccessKey`、`CopyGroup` 只有幂等入口，无重复可删。**修正估计：约 300–400 行
可收敛，不是 800 行。** 评审质疑成立。

**合并边界（回答评审问题）**：能挂到现有 `Mutate func(tx)` seam 上的是
**mutation core**（校验→tx 写→metadata 映射），不动 stage machine 契约。
两路径的发布后语义**刻意不同、不对齐也不应对齐**：

- 普通路径 `writeConfig`（service.go:405）：提交后失败走内联
  `recoverCommittedRuntime` 即时恢复，返回 control-operation 错误；
- 幂等路径 `executeIdempotentOperation`（idempotency_operation.go:157）：
  把进度持久化进 `ControlOperation`（`last_completed_stage`），支持重放与
  `can_reconcile` 续跑。

**结论**：只共享 mutation core；发布机制保留两套，这是设计差异而非债。

### 2.2 #1 Admission 边界——钉死清单

**进 Admission**（"能不能走、去哪"）：快照/AccessKey 上下文消费、配额准入、
RPM 限流、models 端点特判、方言解析、Content-Encoding+body 解码、
`InspectRequest`、流模式评估、`scheduler.Query` 拼装、凭据引用捕获、
响应绑定、亲和性解析。

**留在 Handler**（"怎么收发"）：gin 绑定与生命周期、WebSocket 分支、
requestID 头、抓包元数据边界（`captureFromContext` 生命周期）、
HTTP 写回。

**关键事实降低契约风险**：`reason.go` 已是集中式 `{Status, Code, Message}`
枚举 + `writeReason` 单一映射（L231），12 处拒绝点只是选 reason，
**RejectReason 可直接复用 `reason` 类型**——状态码/body 契约不变，
现有 `*_contract_test` 不会因抽取而变严/变松。

**recorder 处理**：8 处 `recorder.set*` 不是"留在 handler"，而是把
`metadata`→recorder 的映射收敛为 Admission 返回 `Dispatch` 时的一次性填充
（`Dispatch` 携带 operation/stream/reasoning/usage 标志），观测句柄仍由
handler 注入。避免"大函数搬家"。

### 2.3 #2 ForwardInput（25 字段）分类——在 #1 后落地

| 类别 | 字段 | 归属 |
|---|---|---|
| Attempt 上下文（随循环变） | AttemptID, AttemptSequence, Group, APIKey, CredentialSecrets, Credential, ChannelID, UpstreamModelID, TargetConfig, Proxy, ProxyFingerprint, ForceCredentialRefresh | `Attempts` 循环内部，不进跨层接口 |
| PreparedRequest 缓存（按 group/entry） | Request, ExternalModel, RouteRequirement, ResponsesStorePreference, ResponsesStoreDowngraded, RouteMode, Operation | prepareRequest 产物 |
| 副作用句柄 | OnStreamReady, OnFirstResponse, OnResponse, ContinuityKey, BufferedStream, ObserveUsage, RequestID, ClientProtocol, Dialect | 注入式 observer/策略，独立接口 |

`Attempts` 模块的对外接口只需要：`Dispatch`（#1 产物）+ iterator + 3 个
observer 回调 → 返回终止结果。25 字段是编排状态压在适配器入参的症状。

### 2.4 #4 前置验证任务（P3 门槛）

评审要求先验证内存分页假设，证据已在历史中：

- `810029cb`：分组模型页曾因全库回填拖慢首载 → **规模压力已在边缘出现**；
- `09e81fce`：Hostinger 上 SQLite DELETE journal 导致读写互斥 →
  WAL+读连接池是对"读多写少"的刻意设计。

**P3 前置任务**（进试点前必须完成）：对 AccessKey/Group/Credential 列表在
典型自托管规模（AccessKey ≤1k、Group ≤100）与压力规模（10×）各做一次
全表扫描延迟粗测，结论写入 `.docs/adr/`：

- N 小 → 声明式读模型保留内存过滤（收敛三件套仍有价值）；
- N 有压力 → SQL 下推，必须保留 `withReadSnapshot` 一致性语义。

### 2.5 #5 结论收紧

**在未出现第二条写路径或写锁回潮证据前，不拆 worker。** requestlog 集中写是
SQLite 写锁修复的刻意形态（`09e81fce`/`f78d20f1`）。仅允许命名/包边界归位，
且必须附带基准对比（写延迟、锁等待）方可合并。此条作为已关闭决定记录，
后续评审不再重复提出。

### 2.6 新增战略意见复查——三成立、一需走 ADR、一已是现状

| 建议 | 复查结论 | 处置 |
|---|---|---|
| **凭据升级为资源池、路由与凭据解耦** | **与已定型产品规则冲突**：`models/group.go:51` 用 `uniqueIndex:idx_credentials_group` 在 schema 层强制 1 Group=1 Credential；`unified-group-scheduling-plan.md` 明确删除了多 Key/多层权重。多凭据调度**已存在但粒度是 Group**——同一 channel 建 N 个 group 即得轮询/冷却/熔断。 | 不进重构计划。若确有"单逻辑上游 N 凭据"的产品需求，属于 ADR 级领域模型变更，单独立项，不混入本次深化。 |
| **调度决策可观测/可解释** | **已部分存在**：`scheduler/inspect.go` 逐候选 `ReasonCode`（disabled/filtered/no-route/websocket 等 9 类），Dispatch Center 可查。缺口：请求级"为什么没选某候选"未持久化进 request log（日志只有 `failure_category`/`dispatch_state`）。 | 单独立项小特性：失败请求可关联当次 inspect 决策快照。非重构候选。 |
| **SQLite 热路径/事务边界** | **已是现状**：内存 `state.ConfigSnapshot` + `writeMu` 单写者 + WAL 读池 + requestlog 批写。 | 写成 `.docs/tech/` 不变量文档（"数据面不直写 SQLite"），无需新开发。 |
| **安全与密钥生命周期 P0/P1** | **已部分存在**：本地加密（platform/encryption）、AccessKey 轮换（access_key_rotation.go）、脱敏（platform/redact）、CIDR 过滤、RPM/成本限额、变更审计（mutation_audit.go）、认证事件。 | 差距在审计覆盖面与 AccessKey 作用域粒度——建议作为独立加固工作流立项，不属于本轮架构深化。 |

## 3. 最终方案

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **P0 · #3** | 4 对写路径各自提取 `mutate<Resource>(tx, normalized) (result, error)`；普通入口经 `writeConfig` 调用，幂等入口经 `Mutate` seam 调用同一函数。**不动** stage machine 与发布语义。 | 两入口共享同一函数指针；现有幂等重放测试不改断言通过；删除 ~300–400 行 |
| **P1 · #1** | 抽 `Admission` 模块：`Admit(ctx, RequestContext) → (Dispatch, *reason)`；边界按 §2.2 清单；`reason` 类型直接复用为 RejectReason；recorder 由 Dispatch 一次性填充。 | `Handle` 降至 ~40 行 gin 管线；新增 Admit 单测覆盖全部 12 类拒绝；HTTP 契约测试零改动通过 |
| **P2 · #2** | `Attempts` 编排模块接管选择循环；`ForwardInput` 按 §2.3 三类拆分（attempt ctx / prepared / observers），适配器入参预计降至 ~10 字段。 | 重试/重放策略可脱离 gin 测试；prepared-request 缓存失效规则集中在模块内 |
| **P3 · #4** | 先做 §2.4 规模粗测并写 ADR；再以 AccessKey 列表为试点做声明式读模型。 | ADR 记录内存过滤 vs SQL 下推决策；试点资源三件套收敛为一份声明 |
| **关闭 · #5** | 不拆写路径；如未来触碰 requestlog worker，仅允许命名归位+基准对比。 | 记入 `.docs/adr/` 防重复提议 |
| **文档** | "数据面不直写 SQLite / 单写者 / 读快照"写成 `.docs/tech/` 不变量；#5 与凭据池决定入 `.docs/adr/`。 | `pnpm --dir web run docs:check` 通过 |

**明确不做**：凭据资源池（产品规则已定，改则走 ADR 单独立项）、
请求级决策持久化与审计加固（独立特性/安全工作流，另行排期）、
前端架构（本轮无证据）。

## 4. 风险清单

1. **SQLite 单写者是不可突破约束**——所有深化不得新增第二条写路径或绕过
   `writeMu`/`dbtx`。
2. **发布语义双轨是有意的**——合并 mutation core 时禁止"顺手统一"
   `writeConfig` 与 stage machine。
3. **HTTP 拒绝契约冻结**——`reason` 枚举与 `writeReason` 映射对外是线上契约，
   只允许搬运不允许改文案/状态码。
4. **#4 存在规模不确定性**——未做粗测前不动 collection 读路径。
