# Main Ahead 提交评估（全量专题版）

- 评估基线：`879f244c38e46fc172f977335a2d68012f8f20d4`（`Merge pull request #39 from DabengBa/feat/model-aliases`）
- 生成时间：`2026-09-11T14:45:00+08:00`
- 对比目标：`upstream/main@33ec6685503a038c8fea9d62ff1631f38ed281bd`（`fix(monitor): #625`）

## 基线

| 指标 | 值 |
|---|---|
| `dev vs upstream/main` | `139 30`（dev 独有 139，上游独有 30） |
| 上游独有提交 | `#587`–`#625`（不含已入 dev 的 `#589`、`#598`、`#607`、`#609`、`#612`、`#616`、`#617`、`#619`） |
| 双向冲突预演 | 116 路径（web/src 28、internal/control 21、internal/gateway 15、internal/state 11 等） |
| 迁移号段 | `dev` 的 `0010`–`0012` 永久占住上游同名号码段，任何上游新迁移必须从 `0013` 起追加 |

## 当前合同（与本文分析判断的边界）

- 每个分组只有一个上游凭据；普通第二凭据创建、导入、连接必须原子拒绝。
- 权重、优先级、熔断只属于模型 Route Entry；不恢复分组权重或凭据权重。
- 调度中心负责入口级路由、候选排序和运行态；不恢复旧 scheduler、Registry 或前端调度 API。
- 请求重试预算是系统级单一来源（`snapshot.Settings.RetryCount`），分组 `retry_count` 已退役。
- `(group_id, entry_id)` 只用于后端定位和 mutation，不进入用户可见的调度文案或筛选维度。
- 监控筛选的权威查询状态是 `range` preset；`from_ms`/`to_ms` 查询合同（上游 `#594`）未引入。
- 迁移台账只允许追加，编号必须与注册表位置连续（见「迁移编号重排方案」）。
- 存储层只支持 SQLite 与 PostgreSQL；MySQL 支持已移除。
- 交付面只覆盖 Linux 与 Docker。
- 无按模型冷却运行态：不引入上游 `#599` 的 credential+model cooldown。Codex 的 `usage_limit_reached` 固定按凭据范围上报。
- `previous_response_id` 续接由运行态归属表管辖：按 access key 隔离，收窄到原凭据身份。
- Responses WebSocket 是同一端口上的 `GET /v1/responses` 升级；开关 `responses_websocket_enabled`。
- Codex 身份头固定：`User-Agent` 与 `Version` 不能被覆盖/清空，`Originator` 仍按显式规则。
- CPA 固定 `v7.2.157`（根模块与 `third_party/cpaembedded` 一致）。

---

## 专题分析：22 个上游未移植提交

所有提交按业务关联分组。每组的结论以 `dev` 现有合同为判断标准。评估维度：

- **实现价值**：解决用户真实痛点还是内部重构
- **用户意义**：终端用户是否感知
- **功能变化**：引入/删除/修改了哪些能力
- **合并难度**：与 dev 合同/架构的冲突程度（L1=简单cherry-pick，L3=无法移植）

---

### 专题 A：凭据全量操作与批量导入（2 提交 + 1 扩展）

| 提交 | 主题 | 文件/行数 |
|---|---|---|
| `e888fe60` | `#587` feat(credentials): 统一全量操作并支持五千条密钥导入 | 15 文件，+746/-78 |
| `96d3e0d5` | `#596` feat(subscription): 支持多格式订阅凭据导入 | 17 文件，+2705/-1 |
| `6da82242` | `#579` feat(subscription): support custom upstream URLs | 15 文件，+293/-19 |

**分析**：三个主题本质都是凭据/订阅操作增强。`#587` 补充全选/批量启用禁用/五千条导入；`#596` 扩展导入格式支持（Claude/Codex/importfile 模板）；`#579` 允许 channel 级自定义上游 URL（与 `dev` 的「Codex 固定官方端点」合同冲突）。

**用户意义**：批量操作能减少运维工作量，但现有单条操作已满足日常使用。自定义上游 URL 与 `dev` 安全合同矛盾（`dev` 要求 Codex/Grok 固定官方端点，信道不能随意指向外部 URL）。

**合并难度**：L3。`#579` 直接违反 `dev` 合同第 6 节「环境代理按合同拒绝」。批量导入涉及大量控制面代码，且 `dev` 的控制面凭据/分组模型已有显著差异。

**结论**：**已放弃**（与已有前置结论一致）。不纳入范围。

---

### 专题 B：调度中心加权轮询与凭据权重（3 提交）

| 提交 | 主题 | 行数 |
|---|---|---|
| `9cb3f986` | `#591` feat(scheduler): 实现全局加权轮询与连续分配限制 | 15 文件，+943/-46 |
| `e0bfa07e` | `#597` feat(scheduler): 统一凭据权重并调整模型错误重试 | 15 文件，+200/-73 |
| `d4699dd2` | `#599` feat(health): 支持凭据按模型冷却与统一恢复 | 16 文件，+432/-22 |

**分析**：`#591` 在 `internal/scheduler/fair.go` 新增 `FairScheduler` 加权轮询 + `SchedulingCheckpoint`；`#597` 统一凭据权重（删除 `autoweight.go`）；`#599` 新增 `internal/control/model_cooldown.go` + 运行态模型冷却判定。这三者相互依赖：`#591` 是 `#597` 的前置，`#599` 是被 `dev` 合同排除的模型冷却运行态。

**与 dev 的冲突**：
- `dev` 的 `scheduler.Query` 没有 `AllowedCredentialRefs` 和 `Iterator.ChargeReplay`，所有续接/WS 适配都避开了这两条路径
- `#599` 的 model-level cooldown 与 `dev` 第 9 条合同「无按模型冷却运行态」直接冲突
- `#591` 的迁移文件 `0010_model_cooldown` 与 `dev` 的 `0010_single_credential_per_group` 号码冲突

**用户意义**：加权轮询在大量凭据同质时改善分布，但 `dev` 的「一分组一凭据」模式使该调度层功能实际无消费者。

**合并难度**：L3。即使忽略合同冲突，这三套代码与 `dev` 的调度层、检查点、凭据模型都有结构性差异，无法 cherry-pick。

**结论**：**已放弃**（与已有前置结论一致）。专题 2（#591）+ 专题 3（#597, #599）均不纳入范围。

---

### 专题 C：用量统计与监控时间筛选（5 提交）

| 提交 | 主题 | 行数 |
|---|---|---|
| `33fb54bf` | `#588` fix(monitor): 修正最近一小时用量与成本统计 | 14 文件，+775/-26 |
| `175949e2` | `#594` feat(usage): 支持自定义时间统计并统一时间筛选 | 15 文件，+983/-359 |
| `d417b7de` | `#608` feat(monitor): 统一日志与用量的时间筛选交互 | 14 前端文件，+666/-620 |
| `a628fe82` | `#621` fix(monitor): 更新日志筛选快捷时间 | 2 前端文件，+30/-2 |
| `33ec6685` | `#625` fix(monitor): 统一筛选操作与跳转的快捷时间行为 | 5 前端文件，+78/-25 |

**分析**：这五个提交构成用量/监控时间筛选的完整演变链。`#594` 引入 `from_ms`/`to_ms` 双轨查询合同（后端 + 前端 `usage-filters.ts`），`#608` 在其上统一日志与用量的前端时间筛选组件。`#588` 修正最近一小时统计边界。`#621`/`#625` 是后续的 UI 快捷时间优化。

**与 dev 的冲突**：
- `dev` 的监控筛选合同是 `range` preset（只有预设时间段，没有 `from_ms`/`to_ms`）
- 后端 `internal/requestlog/usage_query.go` 在 `dev` 是简化版（无 `from_ms`/`to_ms` 路径）
- `dev` 的 `UsageTab.vue`/`MonitorView.vue`/`LogsTab.vue` 已在多次 PR（#28, #31, #37）中被频繁修改，冲突面大

**用户意义**：自定义时间统计（`#594`）对用量分析有用，但 `dev` 已有 range preset 覆盖了 80% 使用场景。`#621`/`#625` 单独看是很小的前端优化（快捷时间操作），但依赖 `#594`/`#608` 的时间筛选架构。

**合并难度**：
- `#594`/`#608`：L3，查询合同冲突 + 后端架构差异
- `#621`/`#625`：L2（纯前端，但组件路径在 `dev` 有差异，需要手动适配）

**结论**：
- `#588`/`#594`/`#608`：**已放弃**（与已有「专题 5 不纳入范围」一致）
- `#621`/`#625`：**低价值挂起**。纯 UI 优化，不解决用户实际问题，只有在前端时间筛选架构统一后才有意义。

---

### 专题 D：冷却管理展示（1 提交）

| 提交 | 主题 | 行数 |
|---|---|---|
| `fb18dc9e` | `#606` refactor(cooldown): 精简模型冷却展示与管理接口 | 30 文件，+207/-323 |

**分析**：控制面 `model_cooldown.go` 展示重构 + 健康页/凭据页的冷却展示简化。硬前置：操作的文件 `model_cooldown.go` 由 `#599` 新增（`dev` 不存在），且 `runtime_observation.go` 在 `dev` 也有自己的冷却展示版本。

**用户意义**：模型冷却展示优化——但 `dev` 没有模型冷却运行态，展示层无数据可展示。

**合并难度**：L3。前置不存在（`#599` 已放弃），无法独立落地。

**结论**：**已放弃**（与已有一致）。依赖专题 3，而专题 3 已判不纳入范围。

---

### 专题 E：访问密钥分发（1 提交 + 1 已放弃）

| 提交 | 主题 | 行数 |
|---|---|---|
| `a97578fe` | `#590` feat(access-keys): 完善访问密钥分发与用量管理 | 15 文件，+413/-9 |
| `de6d2d44` | `#611` feat(access-keys): 支持自定义密钥并统一编辑与生成交互 | 已放弃，见上轮评估 |

**分析**：`#590` 和 `#611` 相辅相成，但 `#611` 已被放弃（迁移 + 约束校验 + 脱敏前缀长期维护成本高）。`#590` 单独引入用量查询增强等能力，但前置依赖 `#611` 的密钥生成交互变更。

**合并难度**：L3。没有 `#611` 的前置密钥交互变更就无法落地分发流程。

**结论**：**已放弃**。无 `#611` 则 `#590` 无落地起点。

---

### 专题 F：全局请求重试预算（1 提交）✅ 已完成

| 提交 | 主题 | 行数 |
|---|---|---|
| `a4255546` | `#598` feat(gateway): 统一全局请求重试预算 | 19 文件，+278/-68 |

已在 `dev` 随 `3274e32c` 同步完成（见 Plan PR #28）。保留两处本地差异：
- `retryAttemptLimit(retryCount) = max(retryCount, 1)`（尝试总次数，默认 5，上游为 `retryCount + 1`）
- `fallback.missing_evidence_retry` replay 许可（`internal/health/execution_judge.go`）
- 前端的 `retry_count` 分组设置字段已从 UI 移除

**结论**：**已完成**。

---

### 专题 G：网关健壮性与协议修复（2 提交）⭐ 值得移植

| 提交 | 主题 | 行数 | 非测试文件 |
|---|---|---|---|
| `1898d8ee` | `#603` fix(gateway): 修复协议查询参数与零输出边界 | 7 文件 +569/-0 | `internal/dialect/anthropic.go`, `internal/execution/bifrost/executor.go`, `internal/provideradapter/registry.go` |
| `8688dc80` | `#620` fix(gateway): 兼容 WS 请求中的 stream 布尔参数 | 5 文件 +108/-10 | `README*`, `internal/gateway/websocket_turn.go` |

#### `#603` 分析

**内容**：
- 修复 Anthropic 协议查询参数处理（`internal/dialect/anthropic.go`）
- 零输出边界处理：`first output` 场景下 `ContentLength <= 0` 时不应报错（`executor.go`）
- Provider adapter 注册新增 `ZeroOutputRequest` 接口（`provideradapter/registry.go`）

**用户意义**：中等。零输出边界影响所有协议，当模型返回空内容时（如仅 tool call 无文本）当前可能报错。查询参数修复是 Anthropic 特定。

**合并难度**：L2。改动集中在执行层和 adapter 层，与 `dev` 的 bifrost 执行层架构对齐（`dev` 的 `internal/execution/bifrost/` 已有大量改动，但零输出边界是新增逻辑）。`internal/provideradapter/registry.go` 在 `dev` 有变化但结构相似。

**价值判断**：**值得移植**。零输出边界是所有协议共用的稳健性修复，Anthropic 查询参数是补全遗漏的场景。

#### `#620` 分析

**内容**：
- 上游 stats code：`websocket_turn.go` 在第 9 行的 `queryParams` 解析中兼容 `stream` 布尔参数（Responses API 允许传 `stream: true` 但 WS 连接本身已经隐含流式，需静默忽略）
- `README*` 三语更新

**与 dev 的关系**：`dev` 已有 WS 数据面（通过 `#616` 移植），`internal/gateway/websocket_turn.go` 在 `dev` 是 `1b90e572` 版本的。上游的修复是增量改动。

**用户意义**：低。WS 传入 `stream=true` 不会导致功能异常，只是参数校验时可能报 warning。绝大多数客户端不会同时传 `stream` 和 WS 连接。

**合并难度**：L2。`websocket_turn.go` 在 `dev` 有本地适配差异（身份比对、重放不记账等），但 `stream` 布尔兼容是独立于这些差异的改动。

**价值判断**：**低价值挂起**。不影响功能可靠性的参数兼容修复，优先级低。

---

### 专题 H：执行层 UA 规则统一（1 提交）⭐ 值得移植

| 提交 | 主题 | 行数 |
|---|---|---|
| `a895ba25` | `#622` fix(execution): 统一普通渠道出站 UA 规则 | 4 文件 +214/-0 |

**内容**：新增 `internal/execution/bifrost/user_agent.go`，把出站 UA 从硬编码分散在各个 executor 统一到 `UserAgent` 函数，在 executor 和 WS 两条路径中调用。新增 `user_agent_test.go` 覆盖所有渠道。

**与 dev 的关系**：`dev` 目前没有 `user_agent.go`，UA 处理分散。`dev` 的 `internal/execution/bifrost/executor.go` 已在多次移植中被修改（`#607` rerank、`#616` WS、`#619` CPA），但该新增文件不冲突。`websocket.go` 的一行改动（调用 `UserAgent()`）在 `dev` 的 WS 路径中已存在 UA 逻辑，需确认是否对齐。

**用户意义**：低（运维层面）。统一 UA 规则便于上游服务日志分析和请求溯源，用户不直接感知。

**合并难度**：L1。新增文件 + 两处调用点，与 `dev` 现有代码无结构性冲突。`user_agent_test.go` 需要与 `dev` 的渠道列表对齐（新增 Antigravity 等）。

**价值判断**：**值得移植**。低成本（纯新增）+ 运维价值（统一 UA）。

---

### 专题 I：依赖升级（1 提交）✅ 已完成

| 提交 | 主题 | 行数 |
|---|---|---|
| `2bdc1058` | `#589` chore(deps): 升级官方 fasthttp 并移除临时替换 | 4 文件 +26/-12 |

`dev` 已有 `75a0d361`（同内容不同哈希）。fasthttp `v1.73.0` → `v1.74.0`，移除 `tbphp/fasthttp` 临时 replace。

**结论**：**已完成**。fasthttp 版本在 `dev` 与上游一致（`v1.74.0`）。

---

### 专题 J：前端 Web 修复（1 提交）⭐ 值得移植

| 提交 | 主题 | 行数 |
|---|---|---|
| `f0bd09b3` | `#602` fix(web): 兼容 HTTP 密钥复制并统一失败弹窗 | 14 文件 +316/-75 |

**内容**：剪贴板 API 在 HTTP 下不可用时的回滚交互：新增 `CopyFallbackDialog.vue`（只读单行输入框 + 手动选中复制），统一 `CopyButton.vue`/`CopyChip.vue`/`GatewayConnection.vue` 等多处的复制逻辑，三语文案。新增 `use-clipboard-copy.ts` composable。

**与 dev 的关系**：纯前端 UI 组件，`web/src/components/ui/` 和 `web/src/lib/clipboard.ts` 在 `dev` 中不受影响。`GatewayConnection.vue` 在 `dev` 有变化但复制逻辑独立。`GroupCredentialRecord.vue` 和 `ModelUpstreamDrawer.vue` 在 `dev` 已有各自的分组凭据/模型抽屉交互，新增复制兼容的 diff 很小。`AccessKeyCollection.vue` 在 `dev` 的访问密钥页有自己的交互，但复制兜底是插在 slot 中的纯新增。

**用户意义**：中等。HTTPS 环境不受影响，但 HTTP 部署下用户无法复制密钥——这个问题在 `dev` 的部署场景（自建实例反代：Caddy 强制 HTTPS → gptl 内部 HTTP）中**不暴露**，但本地开发环境（HTTP）中会触发。

**合并难度**：L1（前端）。纯 UI 组件新增 + 调用点修改，无后端依赖。

**价值判断**：**值得移植**。低风险 + 提升 HTTP 部署兼容性。但优先级不高（生产环境 HTTPS 不受影响）。

---

### 专题 K：新功能类（3 提交）⭐ 部分值得移植

| 提交 | 主题 | 行数 | 非测试文件 |
|---|---|---|---|
| `7ed01e66` | `#605` feat(images): 支持 Antigravity 与 Gemini 共享生图转换 | 19 文件 +1084/-8 | 13 个非测试文件 |
| `c55312e1` | `#623` feat(groups): 支持选择测试协议和模型并修复禁用分组测试 | 31 文件 +891/-114 | 20+ 非测试文件 |
| `b64a3898` | `#624` feat(ultrafast): 支持请求层级转发与独立计价 | 22 文件 +515/-34 | 8 个非测试文件 |

#### `#605` 分析

**内容**：把 Antigravity 渠道的生图转换复用给 Gemini 渠道，新增 `internal/execution/geminiimage/convert.go`、`antigravity_provider.go` 增加 images 字段转换、`bifrost/capabilities.go` 扩展生图能力、`bifrost/executor.go` 和 `passthrough.go` 增加 Gemini image 路径。

**与 dev 的关系**：`dev` 已有 `internal/execution/geminiimage/convert.go`（与上游一致），但 `bifrost/executor.go` 和 `bifrost/passthrough.go` 在 `dev` 已在多次移植中被重写（rerank、WS、image 路径）。`internal/channel/modules/antigravity.go` 和 `gemini.go` 在 `dev` 是稳定版本。`internal/execution/cpa/adapter.go` 和 `antigravity_provider.go` 在 `dev` 的 vendored CPA 模块中是同步的（v7.2.157）。

**用户意义**：中等。Antigravity 用户可以在 Gemini 模型上使用生图功能（目前该转换仅 Antigravity 独享），扩展了渠道能力。

**合并难度**：L2-L3。`executor.go`/`passthrough.go` 是 dev 的高冲突文件；`cpa/adapter.go` 在 vendored 模块内，需要 CPA v7.2.157 已存在的 API 对齐。如果 `dev` 的 geminiimage 路径与上游没有功能缺口，这部分可能是增量改动而非结构性冲突。

**价值判断**：**中等价值挂起**。功能扩展有实际意义，但执行层冲突面大，需在实际移植时评估 diff 范围。

#### `#623` 分析

**内容（完整上游）**：分组测活时支持选择协议（API/Responses）和模型，并修复禁用分组时的测试问题。涉及：`credential_probe.go` 扩展 `ProbeRequest` 协议/模型字段；`group_settings.go` 新增 `protocol` 和 `model` 设置；快照编译与存储层增加 validation protocol 迁移（`0013_validation_protocol.go`）；前端 `GroupTestFields.vue` 协议/模型选择 UI；`CredentialTestDialog.vue` 扩展。

**采用决策**：只取模型选择 + 禁用分组修复，跳过协议选择。测活使用分组已有协议。

**保留范围**：
- `CredentialProbeRequest.Model` 可选字段 + `normalizeValidationModel` 校验
- `captureCredentialProbe` 的模型覆盖逻辑
- `compileDisabledGroupProbe`（禁用分组也可测活）
- 前端模型输入（`GroupTestFields.vue` 去掉协议选择器）

**跳过范围**：
- `CredentialProbeRequest.Protocol` 及所有协议校验/覆盖
- `GroupSettingsResponse` 的 `ValidationProtocol`/`ValidationProtocols` 字段
- `group_settings.go` 的协议发现代码（`registry.Resolve`、`protocols` 构建）
- `0013_validation_protocol.go` 迁移
- `loader.go`/`snapshot.go` 协议相关改动
- 前端协议选择 UI

**与 dev 的关系**：`dev` 在 PR #32/#37 已实现测活闭环。模型选择让用户在测试时指定特定模型而不是仅靠默认。禁用分组修复解决了一个实际问题：分组关闭后无法测活。

**用户意义**：高。禁用分组可测活直接修复了一个使用 bug。模型选择让验证更灵活。

**合并难度**：L2。切除协议选择后，`credential_probe.go` 的增量收敛到 Model 可选参数 + 禁用分组路径；无迁移、无协议发现代码、无 `group_settings.go`/`loader.go`/`snapshot.go` 的协议改动。

**价值判断**：**最值得移植**。与 `dev` 现有功能直接互补，切除协议选择后冲突范围可控。

#### `#624` 分析

**内容**：「请求层级转发」指上游 OpenAI Responses API 支持 `service_tier` 参数选择计价层级（standard/fast），由用户请求中的 `ultrafast` 或 `service_tier` 字段触发。涉及：CPA `codex_service_tier.go`（新增 service tier 枚举与转发）、`pricing/types.go` 扩展 pricing Mode、`model_price.go` 价格编辑区分 standard/fast、`ModelUpstreamDrawer.vue` 前端价格编辑 UI、「计费模式」指示器 `PricingModeIndicator.vue`、三语文案。

**与 dev 的关系**：`dev` 的 `pricing/types.go` 已有 `ModeStandard`/`ModeFast` 枚举但 `ModeFast` 未被使用。`control/model_price.go` 在 `dev` 是简化版（无 service tier）。`third_party/cpaembedded/embedded/` 没有 `codex_service_tier.go`。前端价格编辑在 `dev` 通过 PR #39 新增了 `price_id` 跳转，但 `ModelUpstreamDrawer.vue` 的价格编辑器 UI 路径一致。

**用户意义**：中等。ultrafast 是 OpenAI 的潜在付费功能，`dev` 的自建实例接入的是前向兼容 API，该功能在当前上游链路下是否真正可用取决于转发的上游是否支持。

**合并难度**：L2-L3。新增 CPA 文件 +价格编辑扩展 +定价类型扩展 +前端 UI。`pricing/types.go` 的冲突可接受；`model_price.go` 在 `dev` 是简版，需要扩展；CPA 文件新增无冲突但需与 CPA v7.2.157 对齐。

**价值判断**：**中等价值挂起**。功能在自建场景下尚未验证实际可用性，存在不确定性。

---

## 专题级别汇总

| 专题 | 提交数 | 主题 | 结论 |
|---|---|---|---|
| A | 3 | 凭据全量操作与批量导入 | ❌ 已放弃（专题 1） |
| B | 3 | 调度加权轮询与凭据权重 | ❌ 已放弃（专题 2+3） |
| C | 5 | 用量统计与监控时间筛选 | ❌ 已放弃（专题 5）；#621/#625 低价值挂起 |
| D | 1 | 冷却管理展示 | ❌ 已放弃（依赖专题 3） |
| E | 2 | 访问密钥分发与用量管理 | ❌ 已放弃 |
| F | 1 | 全局请求重试预算 | ✅ 已完成 |
| G | 2 | 网关健壮性与协议修复 | ✅ `#603` 值得移植；`#620` 低价值挂起 |
| H | 1 | 执行层 UA 规则统一 | ✅ 值得移植（L1） |
| I | 1 | fasthttp 依赖升级 | ✅ 已完成 |
| J | 1 | 前端 Web 密钥复制修复 | ✅ 值得移植（L1） |
| K | 3 | 新功能（生图/测活增强/ultrafast） | `#623` 子集（模型+禁用分组修复）最值得移植；`#605`/`#624` 中等价值挂起 |

---

## 已合入/已完成提交清单（与上游不同哈希但内容已落地）

| 上游提交 | # | 主题 | 本地哈希 |
|---|---|---|---|
| `2bdc1058` | #589 | 升级 fasthttp | `75a0d361` |
| `a4255546` | #598 | 全局重试预算 | `3274e32c`（Plan #28） |
| `fe4b6ac1` | #607 | 原生文本重排序协议 | `23ea2bd1` |
| `7cbd2e67` | #609 | 响应状态续接路由 | `c58d0297` |
| `7d80a981` | #612 | Codex WS Session 封装 | `2994c96e`（PR #28） |
| `f091528b` | #616 | Responses WebSocket 接入 | `1b90e572` |
| `0c9d1888` | #617 | 身份头覆盖与会话兼容 | `fe080bf9`（被 #619 取代） |
| `abc6483d` | #619 | 升级 CPA 并固定版本声明 | `4474b214` |

## 执行顺序（更新）

```
优先级 1: #623 子集（模型选择 + 禁用分组修复）—— 与 dev 现有测活功能直接互补，跳过协议选择
优先级 2: #622（UA 规则统一）—— L1 低成本
优先级 3: #603（协议查询与零输出）—— 网关稳健性
优先级 4: #602（密钥复制兼容）—— L1 UI 修复
--- 以上为确定有价值的移植 ---
挂起:    #620（WS stream 布尔兼容）
挂起:    #605（Gemini 共享生图转换）
挂起:    #624（ultrafast 请求层级转发）
挂起:    #621/#625（监控快捷时间 UI 优化）—— 仅在时间筛选架构统一后有意义
```

## 迁移编号重排方案

> 现状：`#611` 已放弃，当前没有需要新迁移的上游提交。`#623` 引入的 `0013_validation_protocol.go`
> 是第一个可能需要移植的上游迁移。保留本节用于未来指导。

### 冲突事实

`dev` 与 `upstream/main` 对 `0010`–`0012` 三号码段赋予完全不同的语义：

| 编号 | `dev` | `upstream/main` |
|---|---|---|
| `0010` | `0010_single_credential_per_group` | `0010_model_cooldown` |
| `0011` | `0011_usage_latency` | `0011_custom_access_keys` |
| `0012` | `0012_debug_captures` | `0012_access_key_mask_prefix` |

`0001`–`0009` 号码一致，但 `dev` 在这 6 个文件里删掉了全部 MySQL 分支。上游 `0010`–`0012` 不能按原文复制。

### 重排映射

| 上游 | `dev` 目标 | 文件 | 重命名符号 |
|---|---|---|---|
| `0010_model_cooldown` | `0013` | `0013_model_cooldown.go` | `Up0010`→`Up0013`；删除全部 MySQL 分支；专题 B 已放弃，暂无消费者 |
| `0011_custom_access_keys` | `0014` | `0014_custom_access_keys.go` | 同上路线；专题 E 已放弃，暂无消费者 |
| `0012_access_key_mask_prefix` | `0015` | `0015_access_key_mask_prefix.go` | 同上路线；专题 E 已放弃，暂无消费者 |
| `0013_validation_protocol` | `0016` | `0016_validation_protocol.go` | 专题 K `#623` 引入；若移植则需此映射 |

## 结论

1. **上游 `main` 整体合并不成立**：双向 116 冲突路径 + 迁移号码全碰撞 + 多处合同不兼容。
2. **30 个上游独有提交中 8 个已落地到 `dev`**（不同哈希），**10 个已明确放弃**（专题 1/2/3/4/5 + `#611` + `#614` + `#606` + `#608`），**1 个已有**（`#589`）。
3. **确定性值得移植 4 个**：`#623`（测活增强，最高优先级）、`#622`（UA 规则，L1）、`#603`（零输出边界，L2）、`#602`（密钥复制，L1）。
4. **低价值或不确定挂起 5 个**：`#620`（WS stream 兼容）、`#605`（生图共享）、`#624`（ultrafast）、`#621`/`#625`（监控 UI 优化）。
5. **本地与上游的实质差异集中在四块**，均有代码注释或设计文档兜底：
   - 续接绑定用凭据 ID + 身份比对而不是 `AllowedCredentialRefs`
   - WS 刷新重放不记账
   - `usage_limit_reached` 按凭据范围上报
   - 模型级冷却运行态整体不存在
6. **下一步建议**：从优先级 1 的 `#623` 开始移植，这是与 `dev` 现有测活功能增益最大的提交。