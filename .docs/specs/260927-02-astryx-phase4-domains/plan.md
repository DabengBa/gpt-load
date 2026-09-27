# Astryx 迁移 Phase 4(group-detail + import)Plan

> **For agentic workers:** REQUIRED SKILL: Use `delivery-workflow` to implement this task list end to end. During behavior-changing implementation or bug fixes, also use `test-driven-development`.

Source: `spec.md`
Doc IDs: none(复用 parity checklist:`feature.model-test-alias`、`feature.monitor-navigation-shortcuts`、`feature.frontend-preview-switch`)

## Tasks

### Task A: shared route codec + import recovery 接线修复

- [ ] **Done**
- **Scope:** `shared/routing/group-detail-route.ts`(新,自 `classic/features/groups/group-route.ts` 132 行抽取:`tab=credentials|models|settings`、credential 段 `q`/`credential_status`/`page`/`page_size∈{20,50,100}`/`expanded_credential_ids`、model 段 `panel=discovery`/`discovery_q`/`discovery_filter`);`shared/routing/import-route.ts`(新,自 `classic/features/import/import-route.ts` 78 行:`mode=new|existing`、`group_id` 强制 existing、`model_q`、discovery 三参数;existing 只序列化 `mode`+`group_id`);classic 两文件改一行 re-export;`astryx/app/router.tsx` 两路由补 `validateSearch`;`astryx/app/services.ts`:`importRecovery` 改 `sessionStorage`(对齐 classic `bootstrap.ts:57`)+ `onUnauthorized` 接 `captureForUnauthorized`(对齐 `classic/app/unauthorized.ts`);单测 `web/scripts/group-detail-route.test.ts` + `import-route.test.ts`(alias loader,`transform-types` 先例见 `test:monitor-route`)。
- **Proof:** 新 codec 单测全绿(round-trip、默认值不落 URL、非法值剔除、`group_id` 强制 existing、expanded_credential_ids 去重排序);`tsc`×3 + eslint;recovery 接线由 Task G 的 e2e 兜底验证(sessionStorage 断言)。
- **PM:** 手工 `/groups/1?tab=models` 与 `/import?group_id=1` 在 classic 下 canonical 化行为不漂移(git diff classic 侧仅 re-export);astryx 侧 `sessionStorage.getItem('gpt-load.import-reauth-draft')` 为 recovery 唯一后端。
- **Evidence:** `test:group-detail-route` 9/9、`test:import-route` 6/6;tsc×3 + eslint 全绿。`services.ts`:`importRecovery` 改 sessionStorage + boot `sweep()`(对齐 classic bootstrap:67)+ `onUnauthorized` 按 classic 序 capture→bypass→clear→navigate(`onSessionCleared` 改返回 navigate promise,`.finally` 内 consumeBypass)。router.tsx:validateSearch 三元链收敛为 `searchValidators` map(同构),`groupDetail`/`import` 接线。

### Task B: 共享叶片组件批 1(表单与选择器原语)

- [ ] **Done**
- **Scope:** astryx 叶片:`ChannelPresetPicker`(classic `features/import/ChannelPresetPicker.vue` 684,双消费:NewGroupImport + GroupSettingsBaseForm)、`CredentialTextarea`(267,粘贴/分析输入,复用 `shared/domain/import/credential-analysis`)、`ImportConnectionSection`(395)、`ImportOperationNotice`(71)、`ParameterOverrideRulesEditor`(classic `app/` 内规则编辑器)、`ModelPricingStatus`(15)。按 spec「叶片先行」原则,全部落 `frontends/astryx/components/` 或 `features/<domain>/`(双消费件放 `components/`,单域件随域)。
- **Proof:** `tsc`×3 + eslint;组件级断言由各宿主 e2e 兜底;`verify:astryx-i18n` key 对齐。
- **PM:** 组件在宿主域落地后由该域 e2e 驱动可见(ChannelPresetPicker 的可访问名/搜索/选中回执与 classic 一致)。
- **Evidence:** (待填)

### Task C: 模型编辑叶片(alias editor + discovery drawer)

- [ ] **Done**
- **Scope:** `ModelAliasEditor`(classic 963 行,`feature.model-test-alias` 语义所有者——六字符 test alias 只读呈现、新建态无 alias、编辑态沿用)、`ModelDiscoveryDrawer`(455,`panel=discovery`/`discovery_q`/`discovery_filter` 路由态,复用既有 astryx `ModelProbeDialog`/`use-model-probe`)。落 `frontends/astryx/features/groups/`(models tab 专属)。
- **Proof:** `tsc`×3 + eslint;`feature.model-test-alias` 验收清单在 Task F models tab e2e 中逐条断言。
- **PM:** models tab 内 alias 编辑与 discovery 抽屉的交互顺序、CTA 优先级、错误态与 classic 相同。
- **Evidence:** (待填)

### Task D: 订阅 staging 层(stager + account card)

- [ ] **Done**
- **Scope:** `SubscriptionCredentialStager`(classic 1202:`v-model` → 受控 `value`/`onChange`;`window.open` OAuth popup;poll/`setTimeout` 调度与 ready 过期走 effect cleanup;deep watch → render 期调节;stage 状态机契约对齐 `shared/control/resources/credential-stages.ts` 的七态)+ `SubscriptionAccountCard`(1976:quota 进度、reset-credit、re-authorize、outcome_unknown;interval 时钟 `onBeforeUnmount` 清理语义)。两者落 `features/import/` 或 `components/`(双消费:NewGroupImport + GroupCredentialsTab;account card 另被 home 迷你卡参照但无共享)。
- **Proof:** `tsc`×3 + eslint;stage 状态机由 Task E/G e2e(mock 轮询序列)断言。
- **PM:** staging 流程:发起授权 → popup → poll → ready/失败呈现;account card 的倒计时与 outcome_unknown 呈现与 classic 一致。
- **Evidence:** (待填)

### Task E: group-detail credentials tab 栈

- [ ] **Done**
- **Scope:** `GroupCredentialsTab`(1681:route query 筛选/分页/`expanded_credential_ids` 深链、批量操作、凭证测试、staging 挂载 `SubscriptionCredentialStager`、导出 `URL.revokeObjectURL` 延迟回收、`logsLocation` 外跳→RouteLink)+ `GroupCredentialRecord`(256)+ `GroupCredentialBatchBar`(67)+ `GroupApiKeyEditor`(239,reveal/copy 走既有 `gateway-actions` 先例)+ `CredentialTestDialog`(187)。
- **Proof:** `tsc`×3 + eslint;组件断言由 Task F e2e 兜底。
- **PM:** credential 筛选/分页/展开态写回 URL;批量条、测试对话框、行内编辑器与 classic 一致。
- **Evidence:** (待填)

### Task F: group-detail 宿主 + models/settings tab + flag 翻转

- [ ] **Done**
- **Scope:** `GroupDetailView`(470:summary query + tab 路由 + Teleport→portal)+ `GroupHeader`(241)+ `GroupTabs`(53)+ `GroupModelsTab`(1013,复用 Task C 叶片 + `ModelSyncDialog` 238 + 既有 probe 组件)+ `GroupSettingsTab`(1111:`group-settings-patch` 草稿/补丁语义 + unsaved guard)+ `GroupSettingsBaseForm`(401)+ `GroupDeleteDialog`(168,typed confirmation)+ `GroupInUseFeedback`(71);`page_routes.json` `group-detail` `astryx: true`;`GroupsView.tsx:599` 的 `window.location.assign` 改 SPA `navigate`;`astryx-group-detail.spec.ts`。
- **Proof:** `astryx-group-detail.spec.ts` 全绿(tab URL 态、credential 深链、model alias/test alias 呈现、settings 草稿补丁、删除 typed confirm、in-use 反馈、`/logs?` 外跳);astryx 全量回归不破;classic 回归不破;tsc×3 + eslint;`page_routes_test.go` 绿。
- **PM:** `/groups/1` 三 tab 全部 astryx 渲染;`/groups` 行名/深链 SPA 直达;monitor/models 域内 `/groups/:id?tab=*` 链接升级 SPA。
- **Evidence:** (待填)

### Task G: import 域(operation 层 + 双模式页面 + flag 翻转)

- [ ] **Done**
- **Scope:** `import-operation.ts`(144)+ `import-operation-owner.ts`(118)移植为 React hook/service(ephemeral-state cleaner 注册、代际守卫、`dispose`);`ImportView`(182,mode 切换 + `watch(route.query)`→渲染调节)+ `NewGroupImport`(1922,recovery draft getter 优先 stable-op payload、unsaved guard、成功 `navigate(groupDetail)`)+ `ExistingGroupImport`(591,defineExpose→状态提升);`page_routes.json` `import` `astryx: true`;`astryx-import.spec.ts`(含 re-auth 恢复链:401 → login → sessionStorage draft 恢复;非 admin/非 import 目标清除);`astryx-selection.spec.ts` `UNFLAGGED_PATH` 与 `astryx-routing.spec.ts` classic-handoff 断言更新(改用 `/(not-found 或未 flag 残留)` 语义——若全路由均已 flag 则改为验证「cookie+flag 全匹配」与 unknown-path 回退)。
- **Proof:** `astryx-import.spec.ts` 全绿(mode 切换、`group_id` 深链、稳定操作幂等、staging 轮询、恢复链);astryx/classic 全量回归;`page_routes_test.go` + selection 测试绿;tsc×3 + eslint。
- **PM:** `/import` 双模式 astryx 渲染;401 后重登恢复草稿;`AuthGate`「换 key」与登出路径清 draft。
- **Evidence:** (待填)

### Task H: 切片收尾(评审 + 文档 + wrap-up + 真机 CSP)

- [ ] **Done**
- **Scope:** plan Review(三评审面);`astryx-migration-plan.md` Phase 4 outcome;`PROJECT_HISTORY.md`;brief 变更历史;过程文件删除;WSL2 `go build` → `GPT_LOAD_ORIGIN` 三项目 CSP 矩阵(含两条新 flag 路由)。
- **Proof:** 真机 go-csp 全量;astryx+classic 全量回归;`docs:check`/`docs:build`。
- **PM:** manifest 终态:`/login`/`/` `/groups` `/groups/:id` `/import` `/models` `/settings` `/access-keys` `/monitor` `/logs` `/schedule` 全部 `astryx: true`(仅剩 Phase 5/6 遗留面)。
- **Evidence:** (待填)

## Review

- [ ] Review complete

## 进行中

(空 —— planning 完成,进入 Task A)

## Notes

- 本机 pnpm v10/node v22 不满足 `engines`(pnpm≥12/node≥24);一律用 `~/.toolchain/node24`(PATH 前置)直跑 `node_modules` 脚本,Playwright 用 `./node_modules/.bin/playwright.cmd`。
- Chrome 125 实机二进制:`~/.toolchain/chrome-125/chrome-win64/chrome.exe`(`GPT_LOAD_CHROME_125_EXE`);puppeteer cache 另有 125/131/147。
- WSL2 链路:`rsync -az --delete --exclude .git --exclude node_modules /mnt/e/'Wechat work'/gpt-load/ ~/gpt-load/` → `go build` → `HOST=0.0.0.0` 运行(localhostForwarding 需 0.0.0.0 而非 127.0.0.1)→ Windows 侧 `GPT_LOAD_ORIGIN=http://localhost:<port>`。
- `GroupCredentialsTab` 挂载 `features/import/SubscriptionCredentialStager`、`GroupSettingsBaseForm` 引用 `features/import/ChannelPresetPicker`——叶片必须先于宿主落地(Task B/D 是 E/F 的前置)。
- 冷启动:astryx spec 首个 `/route` 导航用 `waitUntil:'commit'` + 90s 测试时限(vite 冷编译先例,Phase 3 已验)。
