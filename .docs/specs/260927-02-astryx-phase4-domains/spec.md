# Spec: Astryx 迁移 Phase 4(最高耦合域:group-detail + import)

## 意图与核心流程

一句话:按方案文档「Phase 4: Highest-coupling domains」把最后两条 classic 持有的路由 `/groups/:id`(三 tab:credentials/models/settings,约 9.2k 行 classic 切片)与 `/import`(new/existing 双模式,约 5.7k 行)迁移到 Astryx 前端。完成后除组内 detail 链路外 classic 仅剩 Phase 5/6 的切换与删除对象。

- 触发条件:Phase 3 已收官(`8cbd2125`,`/access-keys`、`/monitor`、`/logs`、`/schedule` 已 flag 且终审通过),用户指示进入 Phase 4 立项。
- 主路径:shared codec 抽取 → 叶片组件(依赖序)→ group-detail 三 tab + 宿主 → flag flip;import operation/recovery 层 → import 页面 → flag flip;收尾评审 + 文档 + wrap-up。
- 成功状态:`/groups/:id` 与 `/import` 在 astryx flag 下达到 classic parity;`/import` 的 re-auth 恢复链路在 astryx 文档内端到端可用;`gpt-load.import-reauth-draft` 浏览器状态键语义不漂移;classic 保持默认可达且两 Playwright 项目各自全绿。

## 范围 / 不做范围

本次要做:

1. **shared 抽取**:`group-route.ts` → `shared/routing/group-detail-route.ts`(`tab=credentials|models|settings`、credential 子 query:`q`/`credential_status`/`page`/`page_size∈{20,50,100}`/`expanded_credential_ids`;model 子 query:`panel=discovery`/`discovery_q`/`discovery_filter`);`import-route.ts` → `shared/routing/import-route.ts`(`mode=new|existing`、`group_id` 强制 existing、`model_q`、discovery 三参数)。两文件落 shared 后 classic 侧改一行 re-export,astryx `router.tsx` 补 `validateSearch`。
2. **import recovery 接线修复(flag 前置条件)**:astryx `app/services.ts` 的 `importRecovery` 现用 `localStorage`,classic `bootstrap.ts:57` 用 `sessionStorage`——`/import` 未翻转时该分歧不可见;翻转前必须改为 `sessionStorage`。astryx `onUnauthorized`(services.ts:66-71)未调 `captureForUnauthorized`,翻转前必须接线(对应 classic `app/unauthorized.ts`)。
3. **叶片组件(astryx 侧缺口,按依赖序)**:`ChannelPresetPicker`(684,被 `NewGroupImport` 与 `GroupSettingsBaseForm` 双消费)、`CredentialTextarea`(267)、`ImportConnectionSection`(395)、`ImportOperationNotice`(71)、`ParameterOverrideRulesEditor`、`ModelAliasEditor`(963,本切片最大独立组件)、`ModelDiscoveryDrawer`(455)、`ModelPricingStatus`(15)。
4. **group-detail 域**:`GroupDetailView`(470:summary query + tab 路由态 + `#group-header-actions` Teleport→React portal)+ `GroupHeader`(241)+ `GroupTabs`(53);credentials 栈(`GroupCredentialsTab` 1681 + `GroupCredentialRecord` 256 + `GroupCredentialBatchBar` 67 + `GroupApiKeyEditor` 239 + `CredentialTestDialog` 187 + `SubscriptionAccountCard` 1976 + `SubscriptionCredentialStager` 1202);models 栈(`GroupModelsTab` 1013 + `GroupModelSyncDialog` 238,复用既有 astryx `ModelProbeDialog`/`use-model-probe`);settings 栈(`GroupSettingsTab` 1111 + `GroupSettingsBaseForm` 401 + `GroupDeleteDialog` 168 + `GroupInUseFeedback` 71,复用既有 `HeaderRulesEditor`/`ProxyOverrideControl`/setting-chrome)。
5. **import 域**:`import-operation.ts`(144,AbortController/幂等键/克隆 payload/stale 守卫/`dispose`)+ `import-operation-owner.ts`(118,三稳定操作 + 代际守卫)移植为 React hook/service;`ImportView`(182)+ `NewGroupImport`(1922,recovery draft getter 优先 stable-op payload + unsaved guard + 成功后 `navigate(groupDetail)`)+ `ExistingGroupImport`(591,`defineExpose` 改为提升状态)。
6. **flag 翻转与边界清理**:`page_routes.json` `group-detail`、`import` 置 `astryx: true`(各域完成后逐项);`GroupsView.tsx` 复制成功后的 `window.location.assign` 改 SPA 导航;`astryx-selection.spec.ts` 的 `UNFLAGGED_PATH = '/import'` 与 `astryx-routing.spec.ts` 的 classic-handoff 断言改用仍 flag 的路由或调整语义;classic `GroupCredentialsTab`/`GroupModelsTab` 外跳 `logsLocation` 的语义由 astryx 侧等价 RouteLink 承接。

不做(明确推迟):

- Phase 5(默认前端翻转)与 Phase 6(classic 删除)。
- 产品行为、后端 API 契约、页面语义变化;`.docs/db` 语义文档不改内容。
- 超出 parity 的视觉重设计;不引入新依赖。
- `/groups` 集合页已在 Phase 1 spike 迁移(`GroupsView.tsx`),不在本切片。

## 边界规则 / 验收

### 域级 definition of done(沿用 Phase 2/3)

- manifest 条目 `astryx: true`;该域 e2e spec 在 `astryx` 项目全绿,`classic` 项目回归不破。
- URL query 契约一致:parse/serialize/canonical 化与 classic 相同(默认值不落 URL、非法值剔除、`group_id` 出现即强制 `mode=existing`、existing 模式只序列化 `mode`+`group_id`);`resetScroll:false` 用于 query-only 导航。
- 导航移交:`RouteLink` flag 边界策略不变;翻转后既有 astryx 内 `/groups/:id?tab=models|credentials` 硬编码深链(`ModelTree.tsx:383`、`HealthProblemCollection.tsx:283`、`ModelUpstreamDrawer.tsx:535`、`InspectorTab.tsx:153`)自动升级为 SPA 导航且 query 值必须经新 codec round-trip。
- 密度/键盘/a11y:DetailPanel/overlay 焦点约定不回归;窄屏布局与 classic 等价。
- CSP/性能:`go-csp.spec.ts` 对两条新 flag 路由零违规(真机矩阵重跑)。

### Phase 4 特有验收

- **re-auth 恢复端到端**:astryx `/import` 编辑中遇到 401 → 清会话跳登录 → 重新登录(admin)→ draft 从 `sessionStorage` 恢复;非 admin principal 或目标非 import 路由时 draft 被清除(classic `LoginView` 语义);change-key/登出路径 `recovery.clear()`。
- **credential staging**:OAuth popup(`window.open`)、轮询定时器、ready 过期、stage 状态机(`pending_authorization/exchanging/ready/consumed/failed/cancelled/expired/outcome_unknown`)与 shared `credential-stages.ts` 契约一致;同一 stager 组件被 `NewGroupImport` 与 `GroupCredentialsTab` 双消费。
- **import 双模式**:`mode` 切换、`group_id` 深链、稳定操作(payload 克隆 + 幂等键 + 代际守卫)、断线/401 恢复、`in_use` 反馈与删除 typed confirmation。
- **unsaved guard**:import 两模式与 group-detail settings/credentials 草稿态的离开确认语义同 classic(`useBlocker` + resolve 合同)。

## 架构 / 约束(增量于方案文档)

- **Teleport → portal**:`GroupDetailView` 的 `#group-header-actions` Teleport 转 React `createPortal` 或布局 prop(以 `page title` 区已建模式为准);焦点与滚动锚点不漂移。
- **v-model → 受控 props**:`SubscriptionCredentialStager` 的 `update:modelValue` 转 `value`/`onChange`;`defineExpose`(ExistingGroupImport 暴露状态给 ImportView)改为状态提升或 ref 句柄——选最小改动者。
- **定时器纪律**:interval 时钟(quota 倒计时)、poll 调度、`URL.revokeObjectURL` 延迟回收全部走 effect cleanup;不可在 render 期创建。
- **稳定操作层**:`import-operation`/`import-operation-owner` 为框架无关逻辑 + 前端清理注册缝;astryx 侧按 `use-model-probe`/`gateway-actions` 先例做 hook 封装,清理挂 ephemeral-state cleaner。
- **draft 指纹**:SchedulePanelDetail 的 `\u0000` fieldKey 先例沿用(直接写字面量,不改写为转义,保持与 classic 字节级一致——若复用该模式)。
- **每域 flag 翻转时机**:沿用 Phase 3 惯例——group-detail flag 随三 tab 全量就位的提交翻转;import flag 随 recovery 修复 + 双模式页面就绪的提交翻转;域 e2e 在翻转后运行(切片内中间态不外发)。

## 数据 / 集成

- `page_routes.json`:本 slice 结束时 `group-detail`、`import` 置 `astryx: true`。
- shared 资源/领域层已就绪(Phase 0):`control/resources/{groups,credentials,credential-stages,channels,model-probe,model-route-schedule}.ts`、`domain/import/{model-draft,connection-json,credential-analysis,subscription-error-presenter}.ts`、`domain/groups/{settings/group-settings-patch,models/model-diff,credentials/credential-failure-presenter}.ts`、`controllers/import-recovery.ts`;切片内仅需补 codec 抽取与 recovery 接线。
- e2e:新增 `astryx-group-detail.spec.ts`、`astryx-import.spec.ts`;codec 单测沿用 `register-shared-alias` loader(monitor-route 需 `transform-types` 的先例已立);既有 `astryx-groups.spec.ts`(集合页)作回归基线。
- WSL2 真机矩阵:`~/gpt-load` rsync 同步 + `go build` + `GPT_LOAD_ORIGIN`(Phase 2/3 已验证,本机 Chrome 125 位于 `~/.toolchain/chrome-125`,puppeteer cache 亦有 125/131/147)。

## 验证

| 门槛 | 命令 / 检查 | 时机 |
| --- | --- | --- |
| 类型/构建/lint | node24:`vue-tsc -p tsconfig.app.json`、`tsc -p tsconfig.astryx.json`、`tsc -p tsconfig.node.json`;`eslint` | 每改动 |
| codec 单测 | `test:group-detail-route`(新增)、`test:import-route`(新增)、既有 `test:*` 回归 | codec 抽取后 |
| 单域 e2e | `playwright test --project=astryx e2e/astryx-group-detail.spec.ts` / `astryx-import.spec.ts` | 每域 |
| 全量 e2e | `--project=astryx` + `--project=classic` 全套 | flag 翻转后、切片收尾 |
| re-auth 链 | e2e:401 → login → draft 恢复(sessionStorage 断言) | import 翻转前 |
| CSP(真机) | WSL2 `go build` → `GPT_LOAD_ORIGIN` 三项目矩阵(含新 flag 路由) | 切片收尾 |
| i18n | `verify:i18n-icu`、`verify:astryx-i18n` | 每域 |
| 契约 | `test:astryx-routes`、`test:search-codec`、`test:channel-contract`、`test:connection-json` | 每域 |
| 语义文档 | `doc-compiler.js check/build` | 收尾 |

## Doc ID 契约

- 无新增 Doc ID:本切片为 parity 迁移,不新增用户可见语义。
- 复用 parity checklist:`feature.model-test-alias`(models tab)、`feature.monitor-navigation-shortcuts`(group 深链)、`feature.dispatch-reasoning-policy`(若 reasoning 控件随 settings/models tab 落地)。
- `feature.frontend-preview-switch` 的「flag 路由渲染 Astryx shell」契约不变。

## 参考资料

- 方案文档:`.docs/tech/astryx-migration-plan.md`(Phase 4 小节、Per-domain DoD、Phase 2/3 先例)
- 切片摸底:经典侧清单 22 文件 9,219 行(groups)+ 15 文件 5,670 行(import);`git` 工作区实测
- recovery 分歧证据:`web/src/frontends/classic/bootstrap.ts:56-61`(sessionStorage)vs `web/src/frontends/astryx/app/services.ts:87-92`(localStorage);`classic/app/unauthorized.ts` 的 `captureForUnauthorized` vs astryx `services.ts:66-71` 未接线
- 深链证据:`astryx/features/models/ModelTree.tsx:382-384`、`monitor/HealthProblemCollection.tsx:283`、`models/ModelUpstreamDrawer.tsx:535`、`monitor/InspectorTab.tsx:153`
- codec 原文:`classic/features/groups/group-route.ts`(132)、`classic/features/import/import-route.ts`(78)
- 既有 astryx 叶片:`astryx/components/{ChannelIcon,CopyButton,CredentialHealthBar,DetailPanel,HeaderRulesEditor,ProxyOverrideControl,SectionNav,StickySaveBar,setting-chrome}`;`features/models/{ModelProbeDialog,ModelProbeScopeDialog,use-model-probe,ModelPriceStatusBadge}`
- 旧会话历史(决策上下文):`C:\Users\walkl\AppData\Roaming\devin\cli\summaries\history_9d750c6a22c94423.md`
