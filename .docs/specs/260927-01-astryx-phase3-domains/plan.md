# Astryx 迁移 Phase 3(运营域)Plan

> **For agentic workers:** REQUIRED SKILL: Use `delivery-workflow` to implement this task list end to end. During behavior-changing implementation or bug fixes, also use `test-driven-development`.

Source: `spec.md`
Doc IDs: none(`.docs/db` parity 清单只被消费、不改内容)

约束性约定(Phase 2 确立,全切片适用):mutable external controller 必须暴露 memoized snapshot + `useSyncExternalStore`,渲染期只读 `snapshot.*`;Vue watch 语义转 render-phase adjustment + post-commit effect;敏感操作状态机下沉 `shared/controllers`。

## Tasks

### Task A: access-keys 域 —— `/access-keys` 整页迁移 + flag
- [ ] **Done**
- **Scope:** `shared/routing/access-key-collection-route.ts`(由 classic 78 行 codec 抽取,classic 改 re-export);astryx `features/access-keys/`(AccessKeysView + Create/Edit/Rotate dialogs + PolicyFields + ScopeEditor + OperationFeedback);`internal/webui/page_routes.json` flag;`e2e/astryx-access-keys.spec.ts`。共享面:`shared/domain/access-keys/*` 已就绪直接消费;`shared/control/resources/access-keys.ts` 已存在。
- **Proof:** `astryx-access-keys.spec.ts` —— canonical query、集合渲染、create/edit 校验与保存、rotate 确认链、scope/policy 编辑、access_key 降维(若有);astryx+classic 双项目回归。
- **PM:** 打开 `/access-keys`(astryx cookie)→ 列表、创建/编辑/轮换对话框、scope 与 policy 编辑与 classic 一致;脏表单离开有确认。
- **Evidence:** (待填)

### Task B: monitor 宿主 —— route codec + MonitorView + HealthTab
- [x] **Done**
- **Scope:** `shared/routing/monitor-route.ts`(classic 311 行:health/usage/inspector 三 tab 的 canonical query + groups expanded 等);`MonitorView.tsx`(tab chrome + lazy surface + refresh 委托);`MonitorSectionHeading`;`HealthTab` + `HealthSummaryStrip` + `GroupHealthCollection` + `HealthProblemCollection` + `AccessKeyCostLimitHealth`;`router.tsx` 接线。
- **Proof:** `astryx-monitor.spec.ts` 第一批 —— `?tab=` canonical(默认 health 不落 URL)、health tab 渲染、groups 展开、tab 切换导航契约;双项目回归。
- **PM:** `/monitor` 默认 health tab;`?tab=health&groups=expanded` 直达展开态;tab 切换走 query 不刷新文档。
- **Evidence:** `astryx-monitor.spec.ts` 6/6(canonical→`?tab=health`、health 六区块渲染、problem/blocked/折叠组、`groups=expanded` 深链+折叠回写、tab 切换 SPA 导航、refresh 重发、access_key 降维至 usage 且不发 `/api/health`);`monitor` flag 已开;canonical 实际语义为 `tab=health` 显式落 URL(忠于 classic codec,plan 注记"不落 URL"不准确)。tsc/astryx eslint 净。

### Task C: UsageTab —— 用量统计 tab
- [x] **Done**
- **Scope:** `UsageTab.tsx`(810)+ `UsageFilterForm` + `UsageSummary` + `UsageBarChart`(SVG 手写,不引图表库)+ `UsageDistribution` + `UsageBreakdownTable` + `PricingModeIndicator`;`usage-filters`/`usage-bar-chart` 已 shared 直接消费;用量资源 queryOptions 缺口补抽。
- **Proof:** `astryx-monitor.spec.ts` usage 段 —— `?tab=usage` 渲染、筛选表单、汇总/分布/明细表、柱状图交互等价;时间范围 query 契约。
- **PM:** `/monitor?tab=usage` 展示 summary/bar/distribution/breakdown;筛选变更走 canonical query。
- **Evidence:** `astryx-monitor.spec.ts` usage 段 4/4(render 全区块含 quality/distribution/breakdown、range Selector+metric radio 写 canonical query 并重发、filter panel 经 `panel=filters` apply 落 `upstream_model`/`group_id`、access_key 隐藏 group/credential 字段);astryx 全套 99/99;tsc×3+eslint 净。

### Task D: InspectorTab —— 巡检 tab
- [x] **Done**
- **Scope:** `InspectorTab.tsx`(1586,本域最大文件)+ `InspectorForm`;巡检资源 queryOptions/mutation 缺口补抽。
- **Proof:** `astryx-monitor.spec.ts` inspector 段 —— `?tab=inspector` 渲染、表单校验、运行/结果呈现、错误态。
- **PM:** `/monitor?tab=inspector` 可发起巡检并看到结果与错误提示,与 classic 一致。
- **Evidence:** `astryx-monitor.spec.ts` inspector 段 4/4(表单提交落 canonical query 并发 `POST /api/route/inspect`、`run=1` 深链挂载即巡检、空表单校验拦截、500 → 错误态+retry);双 watch 语义以 render-adjustment + liveRef 镜像保留(owner/abort/同请求去重/not-routable 聚焦);route-meta monitor 补 `settings` 命名空间(`routeStrategies` 跨域键,与 group-detail 先例一致);InspectorLedger 复用 ledger overflow 模式;tsc×3+eslint 净,monitor spec 14/14。

### Task E: logs 全量 parity —— LogsTab + 筛选/详情抽屉,替换 spike 子集
- [x] **Done**
- **Scope:** `shared/domain/monitor/log-filters.ts` 化(classic 496 行);`logs-route.ts` codec 复核(现有 spike 契约扩展为全量:筛选/cursor/选中日志);`LogsTab.tsx`(1420)全量 + `LogsFilterForm` + `LogsAdvancedFilterDrawer`(590)+ `LogDetailDrawer`(1222,DetailPanel)+ `LogRouteIdentity` + `LogProtocolConversion` + `RequestLogHealthCard`;替换 spike `LogsView` 为全量 host。cursor 分页语义不变。
- **Proof:** `astryx-logs.spec.ts`(扩或替换 spike spec)—— 筛选/高级抽屉/cursor 翻页/列密度/详情抽屉/时间范围;既有 spike 用例不回归。
- **PM:** `/logs` 展示完整日志表;筛选、翻页、行详情与 classic 一致;`?tab=`/时间范围 canonical。
- **Evidence:** `astryx-request-log.spec.ts` 16/16(全行渲染与列头/路由身份行内筛选+维护与 provider 链接/data-tone 慢响应与模型映射/组与模型 selector 搜索/applied chips+高级抽屉/cursor next/prev/limit 提交/失败 transition 回滚 URL 且保留旧行/首载错误态+retry/同签名 apply 直 refetch/非法 affinity_key URL 全面拦截/access_key 七列降维/transition 骨架/legacy `tab=logs` 归一化/卡片布局 cell 标签);既有 `astryx-log-detail`+`astryx-log-time-range` 9/9;`logs-route.test.ts` 13/13 codec 单测(node --test + `@shared` 别名 loader,`test:logs-route` 脚本);astryx 全量 118/119(唯一失败为 access-keys spec 首个导航 vite 冷编译超时,已改用 `waitUntil: 'commit'`+90s 时限修复);tsc/astryx eslint 净。经典语义修正:日期 preset 只写草稿不自动提交、transition 失败 banner 为瞬态(URL 回滚后 isError 回落)、reset 显式序列化 `from_ms`/`to_ms`。

### Task F: schedule 域 —— `/schedule` 整页
- [ ] **Done**
- **Scope:** `ScheduleView` + `SchedulePanel` + `SchedulePanelDetail`(1252);schedule 资源缺口补抽。
- **Proof:** `astryx-schedule.spec.ts` —— 面板列表、detail 抽屉、启用/禁用与执行操作、错误态;双项目回归。
- **PM:** `/schedule` 展示调度面板;详情抽屉与启停操作与 classic 一致。
- **Evidence:** (待填)

### Task G: flag + 切片收尾
- [ ] **Done**
- **Scope:** `page_routes.json` —— `access-keys`/`monitor`/`schedule` `astryx: true`(随各域完成逐项开启,本任务只核对终态);plan Review(三评审面);方案文档 Phase 3 outcome + PROJECT_HISTORY + brief 追溯;wrap-up(过程文件删除 + `detect_stage complete`);WSL2 真机 CSP 矩阵收尾。
- **Proof:** `check_wrap_up.py` + `detect_stage.py --expect-stage complete`;真机 go-csp 全量;astryx+classic 全量回归。
- **PM:** manifest 终态正确;文档与历史反映交付。
- **Evidence:** (待填)

## Review

- [ ] Review complete

## 进行中

(空 —— planning 完成,进入 Task A)

## Notes

- astryx `/logs` 现为 Phase 1 spike 子集(587 行);Task E 将其升级为全量 host,spike spec 语义并入。
- `monitor-route.ts` 里 `?tab=logs` 返回 usage query(classic 行为)——迁移时保留该怪癖,不改语义。
- access-keys 的 domain 层(`shared/domain/access-keys/*`)与资源(`shared/control/resources/access-keys.ts`)已抽取完毕;剩余 shared 缺口 = collection route codec。
- WSL2 真机构建沿用 `~/gpt-load` 克隆 + rsync 工作区同步 + `GPT_LOAD_ORIGIN`(Phase 2 已验证)。
