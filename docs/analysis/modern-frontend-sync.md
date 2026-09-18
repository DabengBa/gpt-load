# Modern 前端同步评估（#668 系列）

## 基线

- 更新时间：`2026-09-18`
- `upstream/main`：`37cd99fb`（#686）；本批评估对象：#660、#668、#671、#672、#681、#683 及关联项 #669、#675、#678、#680、#682、#684、#686。
- 分叉统计：`merge-base = 0ddc41d8`（#585，09-07）；此后 upstream +66 / local +249，其中 `web/` 双方各 34 / 83 个提交。
- 规模对比：modern 前端 304 文件 / 约 5.0 万行（vue+ts+css）；本地 classic 311 文件 / 约 8.6 万行；上游 classic 299 文件。
- 关键事实：上游将 **modern 设为默认前端**，classic 需管理员在设置页主动选择（`web/src/shared/frontend/preference.ts`，`gpt-load.frontend.v2 = 'classic'` 才回退）。access_key 身份被有意挡在 classic 之外。
- 策略沿用：不整体合并 `upstream/main`，按 `dev` 合同行为级移植。

## 一、信息架构对比

| 页面 | fork classic（现路径） | upstream modern | 差异要点 |
|---|---|---|---|
| 登录 | `features/auth/LoginView.vue` | `features/auth/LoginView.vue` + `LoginMascot.vue`（#672） | modern 多吉祥物交互与三语样式 |
| 首页 `/` | `features/home/`：概要、GatewayConnection、订阅账号卡、Spend、Attention、Welcome | `features/home/`：ConnectClientList/Terminal、HomeRouteTool、HomeTrend、Accounts、Attention；`#route-inspector` hash 挂巡检 | modern 重"接入引导+趋势"；fork 重运营摘要 |
| 分组 | `GroupsView` + `GroupDetailView`（credentials/models/settings 三 Tab） | `GroupsView`（高级筛选 #671）+ `GroupDetailView` + `GroupCreatePanel`（API 密钥/订阅/并行账号三流程） | modern 创建引导更重；详情页结构相近 |
| 凭据 | `groups/credentials/`：TestDialog、BatchBar、ApiKeyEditor、SubscriptionAccountCard | `features/groups/`：CredentialCard 族（QuotaRows/QuotaTrend/Trends/WindowUsage/RoutingMeta/PlanBadge/OutcomeSummary/TestDialog/DetailPanel） | modern 凭据卡片信息量更大（用量/额度趋势）；fork 有测活入口与亲和信息 |
| 访问密钥 | `AccessKeyCollection`：cost-limit、scope、rotate、policy | `AccessKeysView`：quota editor、restrictions、handoff、draft | 能力各有独有项，契约见 §四 |
| 日志 | 独立 `/logs`（`features/logs/LogsView`）+ monitor 内嵌 | `/monitor/logs`：`LogColumnPicker`、`log-redacted-export`、`LogPricingReceipt`、模型一致性过滤（#675）、指标（#686） | 都有独立日志页；modern 表格工具更强 |
| 监控 | `/monitor` 多 Tab：健康/日志/巡检/用量/**调度中心** | 拆为 `/monitor/usage`、`/monitor/health` 两个一级页面；巡检挂在首页 hash | IA 不同；fork 调度中心无对应物 |
| 模型 | `ModelsView`：ModelTree、AliasEditor、DiscoveryDrawer、PriceStatus、**Probe 弹窗**、SpecSheet、UpstreamDrawer | `ModelsView`：ModelCard、DetailPanel、PriceBadge、SelectionDialog、SourceBadges | fork 有测活/别名/上游抽屉；modern 有价格徽标与来源标记 |
| 模型价格 | 独立 `features/model-prices/`：Matrix、SlotsEditor、ResetDialog | `inspector/ModelPriceEditor` + `ModelPricingDetails` | 都有价格编辑，形态与契约不同（fork `price_id`） |
| 设置 | 分区：BrowserAccess/Connection/DataMaintenance/**Reliability**/Routing/SystemInfo | `SettingsView`：HeadersEditor、NumberField、SystemInfo、**FrontendPicker** | fork 有黑名单释放等独有键；modern 有前端切换 |
| 导入 | 完整 `ImportView`：ChannelPresetPicker、connection-json、credential-analysis、ImportRecovery | 仅 `ImportGuide.vue`；`/import` 重定向到 `groups?import=1` | **modern 没有等价导入流程**，是大缺口 |
| 巡检 | monitor `InspectorTab` | `inspector/RouteInspector`（首页 hash） | 都有，落点不同 |
| 参数规则 | 分组 overrides 内 | `config/ParameterRulesEditor` 独立组件 | modern 抽成独立编辑器 |

## 二、fork 独有能力 → modern 覆盖情况

| fork 能力 | fork 位置 | modern 等价物 | 判定 |
|---|---|---|---|
| 模型测活（手动 probe） | `features/models/ModelProbeDialog.vue`、`ModelProbeScopeDialog.vue`、`use-model-probe.ts`；`groups/models/GroupModelsTab.vue` 行内入口；凭据页测活请求 ID 深链；`/api/model-probe` | `CredentialTestDialog`（语义为"测试协议+模型"，非测活流水线） | **缺失**：modern 无测活工作流与 `/api/model-probe` 消费方 |
| 调度中心（行内/批量测活、调度详情） | `features/monitor/SchedulePanel*.vue`；`/api/model-route/schedule*` | 无 | **缺失** |
| 黑名单定时释放 + 手动恢复 | `settings/ReliabilitySettingsSection.vue`、凭据 `auth_state` 模型、`/api/settings` 释放超时键 | health 页 blacklist 展示（上游 liveness 语义） | **语义冲突**：合同不同，非简单补 UI |
| 脱敏亲和筛选 | `features/monitor/request-log-affinity.ts`、`LogsAdvancedFilterDrawer`；`/api/logs?affinity_key=`；响应 `affinity_source/state/key` | 日志展示 `affinity_hit/affinity_kind`（上游 #635 词表），无亲和过滤参数 | **词表不同**：字段与参数名不一致 |
| 缓存命中率展示 | request-logs `cache_*` 字段 + 展示 | `cache_read/write_*` 字段 + `cache_present` 过滤 | **基本对齐**（同源字段），展示形式不同 |
| reasoning status 兼容开关 | 分组 overrides（`group_settings_http_reasoning_status_test.go`） | 无 | **缺失** |
| 分组编辑切上游保留凭据 | PATCH `channel_id`/`provider_url` | 上游 PATCH 无此字段 | **契约互斥** |
| 供应商官网链接 | 分组 `provider_url` | 无 | **缺失** |
| 日期时间选择器 | `AppDateTimeRangePicker.vue` 779 行（reka RangeCalendar + TimeField，`551b8b07`） | `AppDateTimeRangePicker.vue`（#660 版，767 行） | **平行实现**，同问题两套方案 |
| 多渠道凭据导入简化 | `features/import/` 全流程 | `ImportGuide` + 重定向 | **缺失（大）** |
| 用量成本监控 | monitor `UsageTab` + `HomeSpend` | `features/usage/` 整页（Composition/Rank/Trend/MetricGraphic） | **modern 更强** |
| 独立日志页 | `/logs` | `/monitor/logs` | 都有，路径不同 |
| e2e | `web/e2e/request-log-filters.spec.ts`（亲和） | 无 modern e2e | fork 独有测试 |

## 三、modern 独有能力（fork 无）

- 双前端切换：`settings/FrontendPicker.vue` + `shared/frontend/preference.ts` + `catalog.ts`。
- 登录吉祥物：`LoginMascot.vue` + 3 个 webp（约 385KB）。
- 凭据用量/额度趋势：`CredentialTrends`、`CredentialQuotaTrend`、`CredentialWindowUsage`（#678/#684 持续强化，依赖 `/api/groups/:id/credentials/:cid/quota-history`）。
- 分组创建引导：`GroupCreatePanel`（API 密钥/订阅/并行账号三流程）。
- 日志增强：列选择器、脱敏导出（`log-redacted-export`）、`LogPricingReceipt`、`model_consistency` 过滤（#675）、日志指标（#686）。
- 分组高级筛选：`AppAdvancedFilters` + `/api/modern/credentials/options`（#671）。
- 接入引导：`HomeRouteTool`、`ConnectTerminal`、ConnectFields。
- 设计系统：`styles/tokens.css` + 71 个 `components/ui/` 组件；i18n 三语按 feature 分模块（en-US/ja-JP/zh-CN 完整）。

## 四、后端契约差异（modern 落地的前置条件）

### 4.1 modern 专属端点（fork 缺失，需移植）

| 端点 | 上游实现 | fork 适配障碍 |
|---|---|---|
| `GET /api/modern/groups` | `modern_groups.go` → `captureGroupCollectionRecords` | 读 `record.Enabled/Weight/ModelNames`、`CredentialCounts.ModelCooldown`；fork 投影缺这些字段（无 model cooldown / weight_manual），多 `ClientModelCount/ProviderURL` → **需扩展投影或裁剪字段** |
| `GET /api/modern/groups/usage` | `modern_group_usage.go` | requestlog 聚合，预计可移植，需核对 SQLite 查询 |
| `GET /api/modern/groups/:gid/credentials` `GET .../:cid` | `modern_credentials.go` → `ListGroupCredentials`/`GetCredentialDetail`/`enrichCredentialActivityIDs` | 依赖凭据集合管线（fork `credential_collection_helpers.go` 有本地改动） |
| `GET /api/modern/credentials/options` | `modern_credential_options.go`（#671）→ `credentialFilterKey` 解密 canonical | 依赖 `credential_observations` 密文管线，fork 有同名文件 |
| `GET /api/groups/:gid/credentials/:cid/quota-history` | #678 | 新增，依赖 subscription/requestlog |
| `POST /api/credential-stages/import-batch` | 上游批量导入 | fork 是 `/api/credential-stages/import`，**路径与流程不同**，需二选一或并存 |

### 4.2 共享端点的语义分歧（modern 直接连 fork 后端会出问题）

| 端点 | 分歧 | modern 跑在 fork 上的后果 |
|---|---|---|
| `GET /api/logs` | fork 对 query 参数做白名单校验（`request_logs.go` `allowed` map），不认识 `model_consistency` → **400**；响应 fork 用 `affinity_source/state/key`，上游用 `affinity_kind` | modern 发模型一致性过滤即报错；亲和列空白 |
| `GET /api/health` | 上游组计数含 `model_cooldown`、问题凭据含 `weight`（fork 无此概念）；上游 quota/expiring 列表含 `identity`（fork 后端未发） | modern health 页字段缺失/恒零；不影响可用性但信息不全 |
| `PATCH /api/groups/:id/settings` | fork `DisallowUnknownFields`；上游发 `validation_protocol`/`validation_model`/`weight_manual` | **400 拒绝**；fork 的 `channel_id`/`provider_url` 上游不发（切上游无 UI） |
| `GET /api/settings` | 键集合不同：fork 有黑名单释放/reasoning 等键 | modern 设置页无入口，键值仍在但不可编辑 |
| `GET /api/home*`、`/api/home/subscription-accounts` | 双方都有实现；上游 #671/#684 改过 identity/趋势字段 | 需逐字段 diff 审计（**未确认项**） |
| `GET /api/access-keys` | fork cost-limit/scope/rotate vs 上游 quota/restrictions | 字段互有独有，modern 访问密钥页丢 fork 能力 |
| `GET /api/model-prices*` | fork `price_id` 契约 vs 上游 sync/catalog | 需 diff 审计（**未确认项**） |

### 4.3 webui 服务端改动（小）

- `internal/webui/page_routes.go`：合并 `modern_page_routes.json`（#668 改动约 20 行，易移植）。
- `internal/webui/server.go`：CSP 为 reka `SelectViewport`/`ComboboxViewport` 注入的 `<style>` 加两个 sha256 放行。fork 现有组件（含 `AppSelect`）不用 Viewport，**仅 modern 需要**。
- `Dockerfile`：#681 COPY `modern_page_routes.json`（一行）。

## 五、逐项取舍清单

| 项 | 内容 | 依赖 | 与 fork 的冲突点 | 判定建议 | 工作量 |
|---|---|---|---|---|---|
| #660 | 日期时间选择器优化 + AppCombobox + `home.go` 按密钥算模型范围 | 无 | fork 已用 `551b8b07` 自研同组件（779 行）；`home.go` 与 `7987c759` 调度重构冲突 | **前端放弃；后端 scopedHomeModelNames 低价值挂起**（如要网关接入引导再评） | 移植 ≈ 重写 |
| #668 | modern 前端整套 + classic 搬家 + `shared/http` + modern_* 端点 + health 加 identity | 无（但内含上游前 48 个提交的语境） | 330 rename 按上游 classic 度量；fork 新增文件不随迁会成孤儿；health.go/home.go/group_collection.go 实质冲突 | **不直接 cherry-pick**；走 §六 路径决策 | 见 §六 |
| #671 | identity 分组/凭据筛选 + `modern_credential_options.go` | #668 | 前端价值在 modern；classic 仅 3 处 i18n | **挂起**（随 modern 决策）；classic i18n 可单独挑 | 后端 ~116 行 + 管线适配 |
| #672 | 登录吉祥物 | #668 | 纯 modern，无 classic 落点 | **放弃**（除非采纳 modern；即便采纳也可裁剪） | — |
| #681 | Dockerfile COPY modern manifest | #668 | 无 | **条件采纳**（modern 落地时顺手） | 1 行 |
| #683 | 单位缩写标准化 | #668（路径） | 无；fork 同字符串存在 | **已落地**（本工作树）：`format.ts` `locale`→`'en-US'`；zh-CN `context`/`unit`、ja-JP 两处统一为 `1M Tokens` | 已完成 |
| #669 | 日志保留失效资源的历史引用 | #668 | 后端 + 双端；fork 日志栈已定制 | **后端部分评估移植**（`internal/control`+`internal/requestlog`）；前端部分随结构决策 | 中 |
| #675 | `model_consistency` 过滤 | #668 | fork `/api/logs` 白名单会拒绝该参数 | **后端可独立移植**（白名单+过滤逻辑），前端字段 fork 已有 | 小-中 |
| #682 | classic 接受 health identity 字段 | #668 health.go | 无 | **与 health.go identity 部分捆绑移植**：fork `quotaCredentialFields`/`expiringResetCreditFields` 加 `'identity'`（4 行） | 极小 |
| #678/#680/#684/#686 | 凭据趋势、tooltip、订阅洞察、日志指标 | #668 + #675 | 全部落在 modern | **挂起** | — |

## 六、路径方案

### A. 全量采纳 modern（设为默认或可选）

前置工作（§四全部）：移植/适配 5+2 个 modern 端点、扩展 group 投影（Weight/ModelNames/ModelCooldown 或裁剪）、`/api/logs` 参数并集、groups PATCH 字段并集或裁剪 modern 控件、home/model-prices 字段审计、webui manifest+CSP+Dockerfile。
 fork 功能移植（§二缺失项）：测活工作流、调度中心、导入流程、亲和筛选、reasoning 开关、切上游、供应商链接、黑名单释放 UI —— **每个都是独立功能开发**，不是合并。
结论：工作量是"重做一批已完成的 fork 功能"，仅当决定长期跟住 upstream 前端主线时划算。

### B. 结构对齐但不采纳 modern（**已落地于本工作树**）

已完成机械搬家：`web/src/*` → `web/src/frontends/classic/`、`web/src/api/*` → `web/src/shared/http/`、`web/src/api/control/` → `frontends/classic/api/control/`；vite alias `@`→`frontends/classic`、加 `@shared`；tsconfig paths 同步；原 `main.ts` 改为 `frontends/classic/bootstrap.ts`（`export bootstrap()`），新 `src/main.ts` 仅启动 classic 并复刻上游本地化启动失败 UI；新增 `shared/preferences/locale.ts`（上游原文），`shared/http/types.ts` 与 `i18n/index.ts` 的 `AppLocale`/`supportedLocales` 改从该处取得（与上游一致）；`app/page-routes.ts` manifest 相对路径加深两级；5 个验证脚本与 `connection-json.test.ts` 路径同步。
收益：此后上游 classic 修复（#683、#682、#669 的 classic 部分及今后所有 `frontends/classic/` 改动）可直接 cherry-pick，不再手工映射路径。
验证：`pnpm type-check`/`build`/`lint`/`test:connection-json`/`verify:health-projection`/`verify:request-log-affinity`/`verify-log-format`/`verify-group-settings-effort` 全部通过。
注意：`internal/webui/page_routes.json` 未变；`shared/frontend/*`（前端选择机制）与 `data-frontend` 标记未引入（无 modern 场景）；上游 classic 中 `auth-session.ts`/`FrontendSettingsSection.vue` 引用 `@shared/frontend` 的"回新版"语义不适用于本 fork，未来拣选相关文件需剔除。

### C. 维持现状

不动结构；上游前端修复全部手工路径映射。短期零成本，长期每次同步都付映射费，且 modern 系修复永久跳过。

## 七、建议

1. **本轮立即做**：#683 classic 部分（§五，约 10 分钟）；顺手评估 #675 后端、#682+health.go identity 这组小项。
2. **建议排期做**：方案 B（结构对齐）。它把"是否上 modern"与"能否继续低成本吃上游 classic 修复"解耦——对齐后两个方向都留着。
3. **modern 本体挂起**：上游 #668 后仍在密集迭代（#678/#680/#684/#686 全是 modern 修复），且 §二 缺失清单意味着采纳=重做 fork 功能。若未来要采纳，以更晚快照为基线，先完成 §4.1/4.2 后端适配清单，再按 §二 逐项排移植。
4. 若走 B 之后再评 modern，可将 modern 设为**可选非默认**（改 `preference.ts` 默认值为 classic），避免 fork 功能在默认界面回退。
