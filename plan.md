# 改进模型与别名功能

## Review

评审对象：`feat/model-aliases`（2e98eda / 7fe02fa / ad4617d）对比 dev（6bb935d5）。
评审类型：完整最终评审。仓库无该主题的 `spec.md` / `plan.md`、无 `.docs/db`（Doc ID gate N/A），范围按交付主题描述建立。

### 已确认发现

- [x] Review: `preserveGroupModelFields` 的精确匹配分支未检查 `assigned`，当请求顺序为「同 ID 带别名行在前、无别名行在后」时同一已保存条目仍被分配给两条新行，生成重复 `entry_id`，被 `ValidateModelRouteEntries` 以 duplicate entry_id 拒绝保存——即 2e98eda 声称修复的失败形态仍可复现 (severity: medium; scope: 2e98eda / 分组模型保存; evidence: internal/control/group_models.go:264-280; proof: 修复前 `go test -run TestPreserveGroupModelFieldsAssignsEachSavedEntryOnce` 失败 `duplicate entry_id "e000000000001"`；把 `assigned` 守卫统一应用到精确/回退匹配后 `go test ./internal/control/ ./internal/state/ -count=1` 全部通过)
- [x] Review: 2e98eda 的 bug 修复未留下任何自动化回归证据（该 commit 仅改 `group_models.go`，+11/-1） (severity: medium; scope: Proof 义务; evidence: `git show 2e98eda9 --stat`; proof: 新增 `TestPreserveGroupModelFieldsAssignsEachSavedEntryOnce`，覆盖原报告场景「一个无别名已保存行 + 多条新别名行」及顺序敏感用例，修复前失败、修复后通过)
- [x] Review: `PriceID` 断言自证——`want.Items[0].PriceID = result.Items[0].PriceID` 且 fixture 价格行 `ID=0`，对任何实现都成立；另两处 `want.Items[index].PriceID = item.PriceID` 使 `DeepEqual` 忽略该字段，新契约（有价格行→返回 ID，无价格行→字段省略）实际无证明 (severity: low; scope: ad4617d / Proof; evidence: internal/control/group_models_test.go:119、group_models_test.go:448、internal/control/server_test.go:1668; proof: fixture 行改为 `ID: 41` 并断言 `&41`；`TestGetGroupModels…` 改为按 (channel, model) 查库断言精确主键；HTTP DTO 测试断言 `price_id` 非空并随 JSON 往返；新增 `TestMapGroupModelsResponseOmitsPriceIDWithoutPriceRow` 断言无价格行时 JSON 不含 `price_id`；全部通过)
- [x] Review: 价格列的 `v-else-if="item.pricing_status === 'pending' && item.id"` 与 `v-else` 渲染完全相同的 `ModelPricingStatus`（含重复 labels 对象），属多余分支 (severity: low; scope: 7fe02fa / complexity; evidence: web/src/features/groups/models/GroupModelsTab.vue:766-781; proof: 删除后 `pnpm run type-check`/`lint`/`format` 全绿，`v-else` 已覆盖「pending 且无 price_id」的同一渲染)
- [x] Review: `v-if="… && item.price_id"` 依赖 JS 真值，`price_id=0` 会被判假（后端 `*uint` + `omitempty` 对非 nil 指针会输出 `0` 而非省略） (severity: low; scope: ad4617d / 前端契约; evidence: web/src/features/groups/models/GroupModelsTab.vue:757; proof: 改为 `item.price_id !== undefined`（与仓库既有写法一致），type-check/lint/prettier 通过；生产 `price_id` 为自增主键 ≥1，实际不可达，属防御性修正)

### 零发现 / 已核验项

- `mapGroupModelsResponse` 在 `rows[key]` 缺失时无行为变化：`modelPriceRows` 是 `map[Identity]*models.ModelPrice`，旧代码把 nil 交给 `resolvePricingStatus(nil)` 得 `pending`，新代码用 `priceExists` 守卫后结构体初值同为 `PricingStatusPending`（internal/control/pricing_status.go:18）。
- 安全/数据暴露：`price_id` 是模型价格行自增主键，同一控制面 API 已通过 upstream detail `price.id` 暴露同类标识（ModelsView.vue:248），未新增密钥、PII 或跨租户信息。
- 作用域纪律：改动限于 8 个文件（Go DTO/映射 + 前端类型/模板/CSS），无额外抽象、配置项或未使用灵活性；`createModelDraft` 写入的 `price_id` 确被 GroupModelsTab 第三列插槽读取（ModelAliasEditor 透传 draft 行），非死字段。
- 深链真实可用：`modelsLocation({selected_price_id})` → models-route.ts:50 → ModelsView.vue:64 `drawerOpen` → ModelUpstreamDrawer 内 `useModelPriceEditor`/`ModelPriceSlotsEditor` 即价格编辑器。
- CSS 悬停/聚焦描边可靠：`.group-models__pricing-link:hover .status-badge` 依赖祖先作用域 ID 传递；Vue 3.5.42 runtime `setScopeId` 沿 `parentComponent.parent` 递归（runtime-core.esm-bundler.js:5830-5855），StatusBadge 根 span 确实携带父作用域属性，无需 `:deep()`。
- 文档漂移：docs/ 与三份 README 未枚举 `GroupModelsResponse` 字段（`pricing_status` / `price_id` / `model_prices` 均无命中），不存在与实现的语义分歧；本主题无 spec/plan 与 Doc ID 库，无文档更新义务。
- 字体 token：`--text-label-xs: 10.5px`（web/src/styles/tokens.css:59）存在且为仓库通用标签字号，替代 10.8px 仅 0.3px 视觉差，属 7fe02fa 声明的统一化，非回归。
- 备注（未记发现）：回退匹配命中「同 ID 无别名已保存条目」时，权重/优先级随该条目继承给首条匹配的新行，结果顺序敏感，但这是既有启发式；修复前该场景直接保存失败，无可用行为被破坏。

### 未解决证据缺口

- 前端无自动化行为面（仓库无 vitest；`web/node_modules` 原为指向自身的循环符号链接）：本次临时 `pnpm install --frozen-lockfile` 后运行 type-check/lint/prettier，验证完成后已恢复该被跟踪的符号链接。链接渲染、跳转与描边样式仅经静态检查与源码/框架源码阅读，未做浏览器实测。
- 重复 `entry_id` 修复只有单元级 proof，未补 HTTP 级保存回归用例（`state` 校验的 duplicate entry_id 拒绝是单元断言直接覆盖的前提）。

- [x] Review complete —— 5 条已确认发现全部修复并复验；验证命令与结果：`gofmt -l internal/control/`（无输出）、`go vet ./internal/control/ ./internal/state/`（无输出）、`go test ./internal/control/ ./internal/state/ -count=1`（ok / ok）、`pnpm run type-check`、`pnpm run lint`、`pnpm run format`（全绿）。
