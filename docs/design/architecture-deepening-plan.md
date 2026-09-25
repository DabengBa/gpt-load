# 架构深化改造计划

> 来源：`docs/analysis/architecture-review-briefing.md`（评审定稿）。
> 本文档是实施依据：每个阶段给出目标形态、改动清单、不变量与验收标准。
> 词汇约定：module / interface / depth / seam / adapter / leverage / locality；
> 用「删除测试」判定深浅——删掉应集中复杂度，而非搬家。

## 0. 全局不变量（任何阶段不得破坏）

1. **SQLite 单写者**：管理面写只走 `Service.writeMu` + `dbtx`；数据面写只走
   requestlog worker。禁止新增第二条写路径。
2. **发布语义双轨是有意的**：`writeConfig`（提交后内联恢复）与幂等
   `ControlOperation` stage machine（持久化进度、可重放）保留两套，不合并。
3. **HTTP 拒绝契约冻结**：`gateway/reason.go` 的 `reason{Status,Code,Message}`
   与 `writeReason` 映射是线上契约，只搬运不修改。
4. **凭据粒度已定**：schema `uniqueIndex:idx_credentials_group` 强制
   1 Group = 1 Credential；多凭据调度发生在 Group 粒度。本计划不触碰该规则。
5. 现有测试（尤其 `*_contract_test`、幂等重放测试）不改断言即须通过。

## P0 — 合并管理面双写路径（先做）

**问题**：4 对资源写入口各自重复 validate→normalize→tx→map 流水线
（每对 ~95 行同构），修一处漏一处；digest 与 mutation 可静默漂移。

**目标形态**：每个资源一个 `mutate<Resource>(tx, normalized) (result, error)`
core。普通入口经 `writeConfig` 调用；幂等入口经已有 `Mutate func(tx)` seam
调用同一函数，外加 digest + stage plan。

**改动清单**（4 对，`internal/control/`）：

| 资源 | 普通入口 | 幂等入口 | 共享 core 落点 |
|---|---|---|---|
| access-key create | `access_keys.go` `CreateAccessKey` | `access_key_idempotency.go` `CreateAccessKeyIdempotent` | `access_keys.go` |
| group create | `group_create.go` `CreateGroup` | `group_idempotency.go` `CreateGroupIdempotent` | `group_create.go` |
| credential connect | `credential_connections.go` `ConnectGroupCredentials` | 同文件 `ConnectGroupCredentialsIdempotent` | `credential_connections.go` |
| credential import | `group_idempotency.go` 侧 `ImportGroupCredentials` | `ImportGroupCredentialsIdempotent` | 就近归并 |

**normalize 前置**：幂等入口需要 normalized 值算 digest，普通入口也做同样
normalize → normalize 块随 mutation core 一起共享（core 输入为 normalized
结构体，调用方只负责 HTTP DTO → normalized）。

**不改**：digest 构造、stage plan、ControlOperation 持久化、恢复路径、
`writeConfig`/`withControlTransaction`/`writeCredentialConfig` 本身。

**验收**：`go build ./...`；`go test ./internal/control/...` 全绿；
两入口经同一函数指针；净删 ~300–400 行。

## P1 — 数据面 Admission 模块

**问题**：`gateway/handler.go` `Handle`（L579 起 ~280 行）内联全部准入逻辑，
唯一测试面是完整 gin 请求。

**目标形态**：`Admit(ctx, RequestContext) → (Dispatch, *reason)`。

**边界（钉死）**：

- **进 Admission**：快照/AccessKey 消费、配额准入（`accessquota` 决策）、
  RPM 限流、models 端点特判、方言解析、Content-Encoding+body 解码、
  `InspectRequest`、流模式评估、`scheduler.Query` 拼装、凭据引用捕获、
  响应绑定、亲和性解析。
- **留 Handler**：gin 绑定/生命周期、WebSocket 分支、requestID 头、
  抓包（`captureFromContext`）生命周期、HTTP 写回（`writeReason` 等）。
- **recorder**：8 处 `set*` 收敛为 Dispatch 携带的 metadata 一次性填充；
  recorder 实例由 handler 创建注入（它是观测句柄，不进 Admission 决策）。

**RejectReason = 现有 `reason` 类型**，不新建枚举。

**验收**：`Handle` 降至 ~40 行；新增 `admission_test.go` 脱离 gin 覆盖
12 类拒绝路径；全部 `*_contract_test` 零改动通过。

## P2 — Attempts 编排模块 + ForwardInput 瘦身

**前置**：依赖 P1 的 `Dispatch` 产物。

**问题**：`executeAttempts`（handler.go:1023，12 参数 ~400 行嵌套闭包）；
`ForwardInput` 25 字段承载编排状态。

**ForwardInput 三分**：

| 类别 | 字段 | 归宿 |
|---|---|---|
| attempt 上下文 | AttemptID, AttemptSequence, Group, APIKey, CredentialSecrets, Credential, ChannelID, UpstreamModelID, TargetConfig, Proxy, ProxyFingerprint, ForceCredentialRefresh | `Attempts` 循环内部 |
| prepared-request（按 group/entry 缓存） | Request, ExternalModel, RouteRequirement, ResponsesStorePreference, ResponsesStoreDowngraded, RouteMode, Operation | prepareRequest 产物 |
| 副作用句柄/策略 | OnStreamReady, OnFirstResponse, OnResponse, ContinuityKey, BufferedStream, ObserveUsage, RequestID, ClientProtocol, Dialect | 注入 observer + 策略标志 |

**目标形态**：`Attempts.Run(dispatch, iterator, observers) → terminal result`；
适配器入参降至 ~10 字段。

**验收**：重试/重放/缓冲流策略可脱离 gin 单测；prepared-request 缓存失效
规则集中在模块内；现有转发行为测试通过。

## P3 — collection 读模型（先测量，后试点）

**门槛任务（先做）**：对 AccessKey/Group/Credential 列表在典型规模
（AccessKey ≤1k、Group ≤100）与 10× 压力规模做全表扫描延迟粗测，
结论写 `.docs/adr/`：

- N 小 → 声明式读模型保留内存过滤，收敛三件套（`*_http`/`*_collection`/`*_query`）；
- N 有压力 → SQL 下推，保留 `withReadSnapshot` 一致性语义。

**试点**：AccessKey 列表为唯一试点资源，验证「资源声明字段/过滤器/排序键 →
共享读模块执行」是否成立，成立后再推广。

## 已关闭项（防重复提议）

- **不拆 requestlog worker**（#5）：单写者是写锁修复的刻意形态；仅允许
  命名归位 + 基准对比。
- **凭据资源池**：与 schema 级产品规则冲突，需走 ADR 单独立项。
- **请求级决策持久化 / 审计加固**：独立特性/安全工作流，另行排期。

## 验证基线

- `go build ./...`
- `go test ./internal/control/... ./internal/gateway/...`
- 文档变更：`pnpm --dir web run docs:check`

## 实施进度

- **P0 已完成**：4 对写入口共享 mutation core（access-key create、group create、
  credential connect、import）；幂等 group-create 顺带补回缺失的 `entry_id` 处理。
- **P1 已完成**：新增 `internal/gateway/admission.go`——
  `admitRequest(ctx, admissionInput) → admissionOutcome{dispatch | rejection | cancelled}`。
  `requestDispatch` 冻结「去哪」的全部产物；`admissionRejection` 把拒绝表达为值
  （reason + Retry-After + 响应头 + quotaDecision + affinity 观测），
  `completeAdmissionRejection` 单一映射回 HTTP，拒绝契约不变。
  `Handle` 只保留 gin 绑定、WebSocket 分支、requestID/capture 生命周期、
  recorder 生命周期与 defer；`recorder.observeDispatch` 一次性填充 9 处观测字段。
  新增 `internal/gateway/admission_test.go`（10 个用例脱离 gin 覆盖拒绝/取消/dispatch）。
- **P2a 已完成**：新增 `internal/gateway/attempts.go`——`attemptLoop` struct
  承载原 `executeAttempts` 的全部编排状态（deferred 桶、refreshRetry、
  prepared-request 缓存、参数覆写失败簿记）。`preparedRequestCache`、
  `deferredAttempts`、`credentialRefreshRetry` 成为具名类型；候选拉取收敛为
  `nextCandidate()` 三态（yield/skip/exhausted）；配额迟准入收敛为
  `admitQuota()`；终止回退收敛为 `completeExhausted()`（优先级：
  providerError > response > transport > conversion > 静态拒绝）。
  `executeAttempts` 签名 12 参 → 5 参（复用 P1 的 `requestDispatch`）。
- **P2b 已完成**：`ForwardInput` 25 字段按生命周期拆成 4 个嵌入组
  （`attemptTarget` 候选选择 / `attemptIdentity` 尝试标识 /
  `preparedRoute` 缓存产物 / `attemptEffects` 副作用句柄与策略），
  字段提升使全部消费点 `input.X` 零改动；两处构造点（attempts.go、
  websocket_turn.go）按组重写。
- **P3 前置已完成**：`internal/storage/models/access_key_collection_scale_test.go`
  实测 collection 捕获路径——1k keys × 0 logs ~6ms，1k × 500k logs ~20ms，
  10k × 500k ~154ms（线性，无悬崖；唯一缩放项是逐行 `MAX` 索引探测）。
  结论落 `.docs/adr/0001-collection-read-model-scale.md`：保留内存过滤，
  不做 SQL 下推。
- **P3 试点已完成**：新增 `internal/control/collection.go` 共享读模块
  （参数解析脚手架、严格正整数解析、分页数学、Unicode fold）；
  AccessKey 三件套收敛到共享原语，`model_price.go` 顺带改用共享 fold。
  group/credential/project_model 留作推广候选（group 的 fold/分页与
  access-key 逐字相同，credential 的 parsePositiveInt 契约略异）。
