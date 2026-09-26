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

## D: models 域(约 4.7k;域启动时细化)

- [ ] D1 models route 契约 + 集合主体(模型树/筛选/分组)。
- [ ] D2 discovery / upstream drawer(走 `DetailPanel`,ADR-0002)。
- [ ] D3 probe dialogs + alias editor + model-prices 嵌入段。
- [ ] D4 flag + e2e + 域收尾(同 C7 门槛)。

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
