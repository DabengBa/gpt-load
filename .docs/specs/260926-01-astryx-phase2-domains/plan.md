# Plan: Astryx 迁移 Phase 2(低耦合域)

Spec:`spec.md`(同目录)。方案文档:`.docs/tech/astryx-migration-plan.md`「Phase 2」。

域顺序:`settings` → `models` → `home`(model-prices 嵌入段随宿主域落地)。每域独立完成「实现 → flag → e2e → 评审记录」;单域回滚=manifest flag 回退。

## C: settings 域(约 2.9k classic LOC,6 section + controller + scrollspy)

- [ ] C1 shared 补齐(行为中立,classic 不动):
  - `shared/routing/settings-route.ts`:`section` query 契约(parse/serialize/canonical;默认 `routing` 不落 URL;非法值剔除)。classic `settings-route.ts` 改为 re-export。
  - `shared/lib/section-navigation.ts`:scrollspy 纯 DOM 控制器(subscribe/getSnapshot + selectSection + topOffset + 底部吸附规则),classic `use-section-navigation.ts` 改为薄适配。
  - `shared/lib/transient-flag.ts`:定时置位/清除(saved feedback 1.6s)。
  - `shared/controllers/settings-draft.ts`:`use-settings-controller` 的框架无关核(draft/patch/dirty/valid/pending/failed/savedAt/updateDraft/discard/saveAll + abort/reentrancy owner 守卫 + `hasLocalEdits` 挂点)。classic 侧暂保留现有 composable(本 slice 不改 classic 行为);astryx 侧 `useSyncExternalStore` 适配。
- [ ] C2 astryx 页面骨架 `features/settings/`:
  - `SettingsView.tsx`:176px SectionNav + content 双栏(≤860px 单列)、`<h1 id="settings-title">`、Skeleton/QueryFeedback/AsyncRefreshIndicator 等价物。
  - `SectionNav` 组件(vertical scrollspy nav;own component,非 swizzle)、粘性 save bar(dirty 计数/save/discard/savedAt)、validation banner(`role=alert` + 聚焦跳转 `settingTarget` 语义)、discard `Dialog`(列 changed labels)。
  - 深链:`?section=` canonical 化 + 首屏 settle 后 instant scroll(classic 的 120ms 双次补滚语义)。
- [ ] C3 routing / connection / reliability 三 section(timeout/retention/capacity 校验、proxy 子状态继承自 C2 的 draft 接口)。
- [ ] C4 browser-access section(606 行最重:header_rules / cors / response_header_rules 编辑器 + valid/invalid-edits 上抛 + reset-key 语义)。
- [ ] C5 data-maintenance + system-info(只读系统信息卡 + 检查更新链接)。
- [ ] C6 unsaved-guard:`useBlocker` + Astryx Dialog(`shouldBlockFn` resolve 语义 = Phase 1 评审更正后的合同;`allowRouteUpdate` 同 route 放行)。manifest `/settings` `astryx: true`。
- [ ] C7 域收尾:`astryx-settings.spec.ts`(section 切换 URL、draft→save→patch 提交、validation banner 聚焦、discard 对话框、unsaved 拦截、proxy 区段)、双项目回归、i18n/density 检查、WSL2 真机 CSP 复跑。

## D: models 域(约 4.7k classic LOC;model-prices 嵌入上游抽屉)

范围界定(域启动核查):`ModelProbeDialog`/`ModelDiscoveryDrawer`/`ModelAliasEditor` 的消费者是 `GroupModelsTab`/`SchedulePanel`(group-detail/monitor 域),不在 `/models` 页;本域 = `ModelsView` + `ModelTree` + `ModelUpstreamDrawer`(含 model-prices 编辑全套)。

- [ ] D1 shared 补齐(行为中立):
  - `shared/routing/models-route.ts`:`ModelsRouteState` + parse/serialize/canonical(`q`/`group_status`/`pricing_status`/`page`/`selected_price_id`,默认值不落 URL)。classic `models-route.ts` 改 re-export。
  - `shared/controllers/model-price-editor.ts`:`useModelPriceEditor` 的框架无关核(draft/errors/pending/failure/changed/canSave/allNull/unpricedConfirmOpen + addTier/removeTier/requestSave/confirmUnpricedSave/cancel/confirmDiscardSwitch + row(id,updated_at_ms) watch 语义 + abort/reentrancy)。classic composable 改薄适配;astryx 走 `useSyncExternalStore`。
  - `shared/controllers/model-price-sync.ts`:`useModelPriceSync` 的框架无关核(pending/failed/succeeded/run + abort + invalidation + transient flag)。classic 改薄适配。
- [ ] D2 astryx 集合页 `features/models/`:
  - `router.tsx` `modelsSearch`(sparse,parse→serialize)+ `routeViews` 注册。
  - `ModelsView.tsx`:sync 按钮 + succeeded/failed 提示;status 行(client/upstream 计数、pending 链接跳 `pricing_status=pending`、unit、catalog badge + fetch 时间);filter bar(search 250ms debounce + group_status/pricing_status Selectors + reset);skeleton 首屏/transition(`useCollectionLoading`);error/stale feedback;empty/no-results;DS `Pagination`;access_key 只读降维(无 group_status 筛选、无 sync、无 drawer、无 open 动作)。
  - `ModelTree.tsx`:grid+subgrid 树表(client 行 + upstream 行 + rail 伪元素),价格 4 列 + fast ⚡ tooltip、route_groups chips(≤2 +N tooltip)、status badge、open 按钮;≤860px 卡片布局。rail/hover 用 stylex variant 表达(无后代选择器)。
  - `ModelPriceStatusBadge.tsx`(method/match_source → icon+tone+sourceDetail tooltip)。
- [ ] D3 `ModelUpstreamDrawer`(`DetailPanel`,ADR-0002):
  - `DetailPanel` 加可选 `titleAdornment`(DialogHeader `endContent`,放 CopyChip)。
  - detail query(enabled=open&&priceId)+ skeleton/error;meta 条(status badge + updatedAt);identity dl(channel icon+name / model_id code);sharedImpact warning;`ModelSpecSheet`(catalog_reference 或 noCatalog);`ModelPriceMatrix` + fast schedule `ModelPriceSlotsEditor`;associations 列表(group `RouteLink` → unflagged `/groups/:id` 文档级跳转);footer:reset dialog + cancel/save。
  - close 语义:requestClose→confirmDiscardSwitch→cancel→route nav 摘除 `selected_price_id`;editor ref 暴露 confirmDiscardSwitch/discardChanges 给 view(筛选/翻页/切 upstream 前先确认)。
  - `ModelPriceResetDialog.tsx`(AlertDialog + trigger + failure InlineFeedback + invalidation)。
- [ ] D4 flag + e2e + 域收尾:manifest `/models` `astryx:true`;`astryx-models.spec.ts`(canonical query、筛选/搜索/翻页、drawer 开合、price 编辑校验/保存/reset、access_key 降维);双项目回归;i18n/density 检查。

## E: home 域(约 4.7k;域启动时细化)

- [ ] E1 home route 契约 + 页面骨架 + welcome/summary/attention。
- [ ] E2 gateway connection + access-key 只读 + spend + subscription 卡片 + release link。
- [ ] E3 flag + e2e + 域收尾(同 C7 门槛)。

## F: 切片收尾

- [ ] F1 plan Review(三评审面:architecture/correctness/frontend)。
- [ ] F2 方案文档 Phase 2 outcome 回写 + PROJECT_HISTORY + 语义文档 traceability。
- [ ] F3 wrap-up(process files 删除、`detect_stage --expect-stage complete`)。

## 进行中

(空)

## Review

(待填)

## Notes

- settings-patch 已是 shared(`shared/domain/settings/settings-patch.ts`);`shared/control/resources/settings.ts` 的 queryOptions/mutation 就绪。
- classic `useUnsavedChanges` 走 router guard;astryx 侧接线点 = `useBlocker` + shared `UnsavedChangesController`(resolve-with-promise 语义)。
- 密度:settings 表单控件沿用 theme token;长表单 section 分隔线/scroll-margin 照 classic 实测值。
