# Spec: Astryx 迁移 Phase 3(运营域:access-keys / monitor / logs 全量 / schedule)

## 意图与核心流程

一句话:在 Phase 2 已交付的域迁移范式(shared 接缝 + memoized snapshot + render-phase adjustment)上,按方案文档「Phase 3: Operational domains」把 `access-keys`(约 4.2k)与 `monitor` 家族(约 13.3k:`/monitor` 三 tab + `/logs` 全量 parity + `/schedule`)迁移到 Astryx 前端。

- 触发条件:Phase 2 已归档(detect_stage=complete);方案文档 Phase 2 outcome 确立的三条约定为强制约束;用户指示"请继续完成剩余所有任务"。
- 主路径:逐域迁移,顺序 `access-keys` → `monitor` 宿主(health/usage/inspector tab)→ `/logs` 全量 parity(替换 Phase 1 spike 子集)→ `schedule`。每域独立完成「迁移 → manifest flag → e2e 双项目 → 评审」。
- 成功状态:四路由域在 astryx flag 下达到 classic parity;`/logs` 从 spike 子集升级为完整 LogsTab;classic 保持默认可达、功能完整;两前端 Playwright 项目各自全绿。

## 范围 / 不做范围

本次要做:

1. **access-keys 域**:`/access-keys` 整页——集合路由契约(`access-key-collection-route.ts` → shared)、AccessKeysView、create/edit/rotate dialogs、PolicyFields、ScopeEditor、OperationFeedback;`shared/domain/access-keys/` 的 create/edit/rotate/patch/presenter/scope/options 已抽取,直接消费。
2. **monitor 宿主与 tabs**:`/monitor` —— `monitor-route.ts`(311)抽为 shared codec(health/usage/inspector 三 tab 的 canonical query);MonitorView + tab chrome(AppTabs 等价)+ lazy surface;HealthTab + HealthSummaryStrip + GroupHealthCollection + HealthProblemCollection + AccessKeyCostLimitHealth;UsageTab + UsageFilterForm + UsageSummary + UsageBarChart + UsageDistribution + UsageBreakdownTable + PricingModeIndicator;InspectorTab + InspectorForm。
3. **logs 全量 parity**:`/logs` —— 完整 LogsTab(server 驱动筛选/cursor 分页/列密度)+ LogsFilterForm + LogsAdvancedFilterDrawer + LogDetailDrawer + LogRouteIdentity + LogProtocolConversion + RequestLogHealthCard;`log-filters.ts`(496)抽为 shared;替换 Phase 1 spike 子集 LogsView。
4. **schedule 域**:`/schedule` —— ScheduleView + SchedulePanel + SchedulePanelDetail(1252)。
5. **每域同步**:manifest `astryx: true`(`/logs` 已 flag,本切片升级为全量 parity)、e2e 覆盖、shared 层缺口补齐(monitor-route / log-filters / logs-route 的 parse/serialize/canonical)。

不做(明确推迟):

- Phase 4:`groups`/`group-detail`/`import`;Phase 5 默认前端切换;Phase 6 classic 删除。
- 产品行为、后端 API 契约、页面语义变化;`.docs/db` parity 清单文档(model-test-alias、usage-timing-metrics、monitor-navigation-shortcuts、dispatch-reasoning-policy)不改内容。
- 超出 parity 的视觉重设计;不引入新依赖。

## 边界规则 / 验收

### 域级 definition of done(逐域,沿用 Phase 2)

- manifest 条目 `astryx: true`;该域 e2e spec 在 `astryx` 项目全绿、`classic` 项目不回归。
- URL query 契约一致:parse/serialize/canonical 与 classic 相同(默认 tab/range 不落 URL、非法值剔除、重复键、`resetScroll:false` 于 query-only 导航)。
- React Compiler 纪律:mutable external controller 必须暴露 memoized snapshot + `useSyncExternalStore`;渲染期只读 `snapshot.*`;Vue watch 语义转 render-phase adjustment + post-commit effect。
- 敏感操作(访问密钥 reveal/copy/rotate)走 shared controller,身份守卫 + abort + route/unmount 失效链完整。
- 密度/键盘/a11y 与 classic 同量级;overlay 一律 DetailPanel/AlertDialog;不可见 label 仅限密度敏感处且可访问名必须存在。
- CSP:迁移后 `go-csp.spec.ts` 真机矩阵(WSL2 build + `GPT_LOAD_ORIGIN`)对该路由仍零违规。

### 共存期边界(沿用)

- `gpt-load.frontend` cookie 仅选择静态文档;feature-freeze 规则不变。
- 浏览器状态键冻结;monitor/logs 的 filter draft 若落在 localStorage/sessionStorage,键名与语义不得变。
- 回滚路径:单域出问题 = 该域 manifest flag 回退。

## 架构 / 约束(增量于方案文档)

- 组件复用优先序:既有 astryx `components/`(DetailPanel、ChannelIcon、CredentialHealthBar、RelativeInstant)→ Astryx 公共 API → swizzle(预算全局 ≤3,当前 0)。
- 图表:UsageBarChart 若 classic 用 canvas/SVG 手写,astryx 侧优先等价手写 SVG(不引图表库)。
- 长列表:LogsTab cursor 分页语义与 classic 相同(不以页码分页替代)。
- 编辑器型 dialogs(create/edit access-key)沿用 settings/models 确立的 draft-controller + unsaved-guard(`useBlocker` resolve 语义 + AlertDialog)模式。
- i18n:`access-keys`/`monitor` namespace 已有 locale 文件;`verify:astryx-i18n` 保持对齐。
- 仓库契约不变:tsc×3、eslint、ICU、theme-build、cookie 校验;依赖精确版本 + 7 天规则;`gptl-providers-models.html` 不动。

## 数据 / 集成

- `page_routes.json`:本 slice 结束时 `access-keys`、`monitor`、`schedule` 置 `astryx: true`(`/logs` 已 flag)。
- 共享资源:`shared/control/resources/{access-keys,request-logs}.ts` 已存在;monitor/usage/health/schedule 的资源缺口按域补抽。
- e2e:每域 `astryx-<domain>.spec.ts`;`/logs` 的既有 spike spec 语义并入全量 spec 或替换。

## 验证

| 门槛 | 命令 / 检查 | 时机 |
| --- | --- | --- |
| 类型/构建/lint | `vue-tsc -p tsconfig.app.json`、`tsc -p tsconfig.astryx.json`、`tsc -p tsconfig.node.json`、`eslint` | 每改动 |
| 单域 e2e | `playwright test --project=astryx e2e/astryx-<domain>.spec.ts` | 每域 |
| 全量 e2e | `--project=astryx` + `--project=classic` 全套 | 每域完成、切片收尾 |
| CSP(真机) | WSL2 `make build` → `GPT_LOAD_ORIGIN` 矩阵 | 切片收尾 |
| i18n | `verify:i18n-icu`、`verify:astryx-i18n` | 每域 |
| 契约 | `test:astryx-routes`、`test:search-codec` 等 node 单测 | 每域 |
| 密度/a11y | `astryx-density.spec.ts` + 手工浏览器复核 | 每域 |
| 语义文档 | `doc-compiler.js check`(若动 `.docs/db`) | 改动时 |
