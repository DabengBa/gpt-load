# Astryx Migration Foundation Plan

> **For agentic workers:** REQUIRED SKILL: Use `delivery-workflow` to implement this task list end to end. During behavior-changing implementation or bug fixes, also use `test-driven-development`.

Source: `spec.md`
Authoritative detail: `.docs/tech/astryx-migration-plan.md`(下称「方案文档」;各任务引用其小节,实现细节以它为准)
Doc IDs: `feature.frontend-preview-switch`(新增,Task B6)

## Part A — Phase 0:框架无关共享层抽取(仅 classic,行为中立)

每步独立可交付:现有 `build`/`lint`/`format`/`verify:*`/`test:*`/e2e 全绿,运行时行为不变。Proof 以现有套件为回归护栏(refactor,非新行为,TDD 不适用);新增脚本类任务按 TDD 先写失败证明。

### Task A0: 本地工具链就位
- [x] **Done**
- **Scope:** 仓库外工具链目录(如 `~/.toolchains/`);不改任何 repo 文件。
- **Proof:** `node -v` ≥ 24.11、`pnpm -v` ≥ 12.1,且 `pnpm --dir web run type-check` 在该 PATH 下退出 0。
- **PM:** 终端执行 `pnpm --dir web run type-check` -> 不再报 engines 错误。
- **Notes:** portable Node zip + 其内 `npm i -g pnpm@12`;按会话 PATH 前缀使用。若 pnpm 12 拒读 lockfile v9(--frozen-lockfile),先 `pnpm install` 迁移 lockfile 并记录该 diff。
- **Evidence:** `C:\Users\walkl\.toolchain\node24`(Node v24.21.0 LTS Krypton + pnpm 12.5.1,2026-09-18 发布满足 7 天规则;12.6.0/12.7.0 太新跳过)。`pnpm run type-check` 退出 0;pnpm 12 直接复用 lockfile v9(“Lockfile is up to date”),无需迁移。本机 curl 出网受限(SSL exit 35),改用 Node `fetch` 下载 portable zip;后续命令均前缀 `PATH=/c/Users/walkl/.toolchain/node24:$PATH`。

### Task A1: 删除 8 个零引用组件
- [x] **Done**
- **Scope:** `web/src/frontends/classic/components/{charts/TrendChart.vue,charts/trend-chart.ts,config/ProxyConfigEditor.vue,config/ProxyScopeIndicator.vue,layout/PageSection.vue,ui/MobileRecordCard.vue,ui/OperationNotice.vue,ui/SecretValue.vue,ui/StatFigure.vue}`;先 `rg` 复核零引用再删。
- **Proof:** `pnpm --dir web run build`、`lint` 绿;`rg` 无残留引用。
- **PM:** `rg -n "TrendChart|ProxyConfigEditor|ProxyScopeIndicator|PageSection|MobileRecordCard|OperationNotice|SecretValue|StatFigure" web/src` -> 0 命中。

### Task A2: client-context 移回 classic + shared 禁框架导入守卫
- [x] **Done**
- **Scope:** `web/src/shared/http/client-context.ts` → `web/src/frontends/classic/app/api-client-context.ts`;全部注入方改导入路径;`web/eslint.config.mjs` 对 `src/shared/**` 加 `no-restricted-imports`(ban `vue`、`vue-router`、`vue-i18n`、`@tanstack/vue-query`、`reka-ui` 及 `../frontends/**` 相对回引)。
- **Proof:** lint 绿;`rg -n "from 'vue'|vue-router|vue-i18n|@tanstack/vue-query|reka-ui" web/src/shared` -> 0 命中;build 绿。
- **PM:** 起 dev,登录后任一页正常 -> inject 链未断。

### Task A3: control 协议/类型 + query-keys + invalidation → `shared/control/`
- [x] **Done**
- **Scope:** `classic/api/control/*` → `shared/control/`;`classic/app/query-keys.ts`、`classic/app/resources/invalidation.ts` → `shared/control/`;classic 原路径保留 re-export。
- **Proof:** build、lint、全部 `verify:*`、`test:*` 绿(re-export 使 ssrLoadModule 路径仍解析)。
- **PM:** 无用户可见变化;`rg` 确认 `shared/control` 为唯一实现处。

### Task A4: 23 个 resource 文件去 Vue 化 → `shared/control/resources/`
- [x] **Done**
- **Scope:** `classic/app/resources/*.ts` → `shared/control/resources/*.ts`;fetcher 改纯参数,返回 `{ queryKey, queryFn }`;classic 侧薄 wrapper 用 `computed`/`toValue` 保持现有调用签名。13 个文件去 `vue`(`MaybeRefOrGetter`)、14 个去 `@tanstack/vue-query` 依赖。
- **Proof:** `verify:*` 全绿(经 wrapper 仍命中被测函数);`rg -n "vue|@tanstack" web/src/shared` -> 0;type-check 绿。
- **PM:** 任意列表页(group/access-keys)筛选分页行为不变。

### Task A5: 纯 lib + 无框架 features 逻辑 → `shared/lib`、`shared/domain/<domain>`
- [x] **Done**
- **Scope:** `classic/lib/*`(11 个)→ `shared/lib/`;方案文档认定的 55 个无框架 `features/**/*.ts`(约 7.3k 行,含 `subscription-error-presenter.ts`)→ `shared/domain/<domain>/`;classic 保留 re-export。
- **Proof:** build、lint、verify、test、e2e 绿;guard 无命中。
- **PM:** 无可见变化;`rg` 抽查移动文件的导入方全部解析到 shared。

### Task A6: 路由规则 → `shared/routing/`(safeRedirect 连测试一起搬)
- [x] **Done**
- **Scope:** `pagePathMatches`、`safeRedirect`、`decodedPathSegments`、query 规整(散于 `app/router.ts`、`route-query.ts`、`route-locations.ts`、`page-routes.ts`)→ `shared/routing/`;`safeRedirect` 既有用例原样迁移为 node:test/共享测试。
- **Proof:** 搬移后的 safeRedirect 测试在 shared 上绿;build、lint 绿。
- **PM:** `/login?redirect=` 合法/非法跳转行为不变(e2e 或手测)。

### Task A7: 6 个控制器 → `shared/controllers/`(subscribe/getSnapshot)
- [x] **Done**
- **Scope:** `features/preferences/theme.ts`、`app/toast.ts`、`app/unsaved-changes.ts`、`features/import/import-recovery.ts`、`app/ephemeral-state.ts`、`features/auth/auth-session.ts` 核心 → `shared/controllers/`;classic 侧 `ref`+订阅 adapter 保持 API;React 侧留给 Phase 1 用 `useSyncExternalStore`。
- **Proof:** build、lint、e2e(主题切换、toast、未保存拦截、import 恢复路径)绿;guard 无命中。
- **PM:** 切换主题/触发 toast/脏表单跳转拦截 -> 行为不变。

### Task A8: locale catalogs → `shared/i18n/locales/`
- [x] **Done**
- **Scope:** `classic/i18n/locales/**` → `shared/i18n/locales/**`;`classic/i18n` loader/context 改从 shared 导入;catalog 内容与命名空间结构不变。
- **Proof:** build 绿;三语言界面冒烟。
- **PM:** 偏好里切 zh-CN/en-US/ja-JP -> 文案全量切换无 raw key。

### Task A9: ICU 兼容改写 + `verify:i18n-icu`(新行为 → TDD)
- [x] **Done**
- **Scope:** `shared/i18n/locales/*/import.ts` 4 个 key × 3 locale:3 个凭据 JSON 示例改为 `{example}` 插值、`callbackPlaceholder` 的 `<port>`/`<端口>` 改为 `{port}`;对应 Vue 调用点改传值;新增 `web/scripts/verify-i18n-icu.mjs`(devDep `@formatjs/icu-messageformat-parser`,精确版本 ≥7 天)解析全部消息 + 跨 locale key 对齐;并入 `lint`。
- **Proof:** 先写脚本验证旧消息必失败(red),改写后转绿;受影响 UI 文本渲染一致。
- **PM:** 导入页查看凭据示例与 callback 占位文案 -> 与改前一致。

### Task A10: verify 脚本与 contract 测试改指 `shared/`
- [x] **Done**
- **Scope:** `web/scripts/verify-*.mjs`(8 个)、`scripts/*.test.ts` 的 `ssrLoadModule`/路径断言从 `frontends/classic/...` 改指 `shared/...` 真实实现处;保留 classic wrapper 仅为兼容,验证以 shared 为准。
- **Proof:** 全部 `verify:*`、`test:*` 绿,且日志可见加载路径为 `shared/`。
- **PM:** `pnpm --dir web run verify:group-collection` 输出 PASS。

### Task A11: e2e 选择器语义化改写
- [x] **Done**
- **Scope:** `web/e2e` 7 个 spec 的 149 处 `locator()` BEM class 定位 → `getByRole`/`getByLabel`/`getByText`;语义不足处给 classic 组件补 `data-testid`(React 端沿用同键)。
- **Proof:** 全量 Playwright 绿;`rg -n "locator\(['\"]\." web/e2e` -> 0。
- **PM:** `pnpm --dir web exec playwright test` 全绿。

### Task A12: tech 文档 `code.paths` 改指 shared
- [x] **Done**
- **Scope:** `.docs/tech/{model-test-alias.md,reasoning-policy.md,usage-accounting.md,billing-failure-attention/plan.md}` 中指向已迁出 classic 路径的 `code.paths`。
- **Proof:** `pnpm --dir web run docs:check`、`docs:build` 绿。
- **PM:** 无可见变化;diff 仅 `code.paths`。

### Task A13: Phase 0 出口复核
- [x] **Done**
- **Scope:** 全量回归 + 架构不变量终检。
- **Proof:** `pnpm --dir web run build && pnpm --dir web run lint && pnpm --dir web run format` + 全部 `verify:*`/`test:*`/e2e 绿;`rg "vue|vue-router|vue-i18n|@tanstack/vue-query|reka-ui" web/src/shared` -> 0;`rg "locator\(['\"]\." web/e2e` -> 0。
- **PM:** 手工走查 login → groups → group-detail → monitor → settings 主路径,行为与基线一致。
- **Exit audit(2026-09-25, 本机 node24 工具链执行):**
  - `vue-tsc + tsc --noEmit`:PASS;`vite build`:PASS(3235 modules)。
  - `eslint --max-warnings=0`:PASS;`verify:i18n-icu`:PASS(8123 messages)。
  - `verify:*`(group-collection/health-projection/request-log-affinity/reasoning-contract/model-test-alias/log-format)+ `test:*`(connection-json/channel-contract/group-collection.test/request-log-affinity.test):全绿。
  - Playwright 58/58 PASS(3.1m)。
  - `shared/` 框架导入扫描:0;`e2e` class locator 扫描:0。
  - `prettier --check` 在本机 CRLF 检出上基线即红(autocrlf),非本次改动引入;CI LF 环境不受影响。
  - PM 手工走查:待用户执行(清单见上)。
  - 修复项:verify-reasoning-contract fixture 移除已失效的 `fallback` 字段;route-query 注释去 vue-router 字面量以保持零匹配扫描。

## Part B — Phase 1:脚手架、共存与 spike(go/no-go gate)

顺序可微调;新增行为类任务按 TDD 先写失败证明(Go 测试、verify 脚本、契约测试)。

### Task B1: 依赖落地(精确版本,无代码变化)
- [x] **Done**
- **Scope:** `web/package.json` + `web/pnpm-workspace.yaml`;`pnpm --dir web add` 精确版本:`react`/`react-dom` 19.3.0、`@astryxdesign/core` 0.6.2、`@astryxdesign/theme-neutral` 0.6.2、`@stylexjs/stylex` 0.19.1;dev:`@stylexjs/unplugin` 0.19.1、`@vitejs/plugin-react` 6.x、`@rolldown/plugin-babel`、`@babel/core`、`babel-plugin-react-compiler` 1.0.x、`@tanstack/react-router` 1.170.x、`@tanstack/react-query` 5.103.x、`react-intl` 12.1.x、`lucide-react` 1.48.x、`@astryxdesign/cli`、`@formatjs/icu-messageformat-parser`、 `@stylexjs/eslint-plugin` 0.19.1、`eslint-plugin-react-hooks` 7.x;两处 `overrides` 重推导对齐。7 天规则逐包核对发布时间。
- **Proof:** `pnpm --dir web install --frozen-lockfile` 成功;`build`/`lint`/e2e 对 classic 仍绿。
- **PM:** lockfile diff 只新增预期包。
- **Notes:** `react-intl`/`intl-messageformat` 与 `@tanstack/*` 放 dependencies 还是 devDeps 按现有分区惯例(运行时包→dependencies,构建/校验工具→devDependencies)。`@astryxdesign/cli` 的 optional peers 不装。

### Task B2: Vite 双入口 + React 编译链 + ESLint 分区
- [x] **Done**
- **Scope:** `web/astryx.html`(head 契约同 `index.html`:theme-bootstrap.js、favicon)、`src/frontends/astryx/main.tsx`(占位 shell)、`vite.config.ts`(双 input、`target:'chrome125'`、browserslist、插件顺序 stylex→vue→react(include 限定 astryx)→babel(compiler preset 排除 classic/shared)→tailwind→selector 占位)、`tsconfig.astryx.json`(`jsx:react-jsx`)、`type-check` 扩为三 tsconfig(`tsconfig.app.json` 排除 `frontends/astryx/**`)、eslint flat 按目录 scope、`@app`/`@shared` 别名。
- **Proof:** `pnpm --dir web run build` 同时产出 `dist/index.html` 与 `dist/astryx.html` 及各自 assets;`type-check` 覆盖新 tsconfig;lint 绿。spike(d) 的编译链验证并入此任务:compiler+StyleX 双 Babel 输出正确、Fast Refresh 保状态、新代码编译器 lint 零 error。
- **PM:** `vite dev` 直开 `/astryx.html` -> 渲染占位 shell,无 CSP/控制台错误。

### Task B3: 静态主题 `gptload.theme.ts` + `entry.css` 层序
- [x] **Done**
- **Scope:** `src/frontends/astryx/theme/gptload.theme.ts`(`defineTheme` extends `neutralTheme`:accent `['#1c4f6e','#6fb2d6']`、canvas `#eeede9/#0b0d10`、status 色对、radius 6/7/10 显式 token、`typography.scale.base=13.5`、字体=classic 系统栈、`--size-element-*`=30/34/38)→ `pnpm run theme:build` 产物 `gptload.theme.css`/`gptload.js`/`gptload.d.ts`/`gptload.variants.d.ts` 提交;`entry.css` 声明 `@layer reset, astryx-base, astryx-theme` + 三 `@import`;vite `useCSSLayers:{before,prefix:'app'}` 输出 `app.priority*`;`theme-preference.ts` 偏好 store(只读写 `gpt-load.theme` storage,复用 shared `themeStorageKey`/`isTheme`),`<Theme mode>` 为 `data-theme` 唯一运行时所有者;lint 链并入 `verify-theme-build.mjs`(spawn `astryx theme build --check`,stdin ignore 防挂)。
- **Proof:** `theme:check` PASS(重建零 diff);`e2e/astryx-theme.spec.ts` 5/5 PASS:light canvas rgb(238,237,233)/dark rgb(11,13,16)/system 无 `data-theme` 且随 `prefers-color-scheme` 切换;computed-style 断言 Button radius theme=7px(neutral 默认 8px)> xstyle=2px;控件高 34px±1、`main` 正文 13.5px;`build`/`lint`/`type-check` 绿,全量 e2e 63/63。
- **PM:** dev 下切三种主题模式刷新 -> 无 flash(theme-bootstrap.js 预置 + `<Theme>` 同步同值);shell 已渲染主题色 Button 供观感对比。
- **Notes:** 修复 `verify-i18n-icu.mjs` 挂起 —— `createServer` 改为 `configFile:false`(catalog 为自足 TS,不需插件;B2 新增插件使 `server.close()` 不再返回)。babel/react 插件 include 收窄至 `*.[jt]sx?`(原正则误匹配 `entry.css`)。26/32/42px 与 setting 26px 等缺口留待 B9 组件覆盖;`--radius-*` 固定阶梯无法表达 6/7/10,用显式 token。

### Task B4: Go 侧 manifest v2 + 双 index 选择(TDD:先失败 Go 测试)
- [x] **Done**
- **Scope:** `internal/webui/page_routes.json`(`version:2`;真实 manifest 暂不标 `"astryx"` 旗标——选路机制在 flag 落上前对生产惰性)、`page_routes.go`(`pageRouteManifestVersion=2`、`pageRoute.Astryx`)、`shared/routing/page-routes.ts`(`version!==2`、`routeFields`+`astryx` boolean 校验、`PageRouteEntry.astryx`)、`server.go`(`astryxIndex` 缺失即 nil、`frontendPreference` 读 `gpt-load.frontend`、`indexFor`/`indexForNotFound`、`serveIndexWithStatus` 收 document 参数)、`http_routes.go`(按页绑定 `s.servePage(page)`)、新增 4 个 Go 用例。
- **Proof(RED→GREEN):** 先加用例后失败(version gate 拒绝 v2、cookie+flag 不选 astryx、404 fallback 不选 astryx)→ 实现 → `go test ./internal/webui -run 'TestParsePageRoute|TestEmbeddedPageRoute|TestServer'` 16/16 PASS;`go vet`/`go build ./internal/webui` 净;web `type-check`/`lint`/e2e 63/63 全绿(v2 manifest 在 classic 运行时 strict parse 生效)。
- **环境基线(先于本改动,非本次引入):** `internal/webui` 的 docker/workflow/semver 合同测试在本机全红(无 docker、fork 缺 `.github/actions/web-ci`、release 工作流形状不同——HEAD 验证同败);`go build ./internal/...` 在 Windows 失败(`platform/securefile` 仅 unix 文件);gofmt/prettier 报红为 CRLF 检出基线。
- **PM:** 同 URL 双文档 curl 验证需 `make build` 完整二进制,Windows 无法编译 securefile——随 CI/Linux 环境执行;无 cookie 路径已被 `TestServer*` 断言与现状一致。

### Task B5: dev selector 插件 + Playwright 三项目矩阵
- [x] **Done**
- **Scope:** `vite.config.ts` 的 `frontendSelectorDevPlugin`(`configureServer` 中间件:仅 GET+`accept:text/html`;cookie `gpt-load.frontend===astryx` 才生效;已标 flag 的 manifest 路由或未知名路径重写 `req.url='/astryx.html'`,未标 flag 路由恒 classic——与 `server.go` `indexFor`/`indexForNotFound` 语义一致;路由匹配复用 shared `pageRouteEntries`+`pagePathMatches`,`page_routes.json` 给 `settings` 打 `"astryx":true` 作 demo 旗标);`tsconfig.node.json` 补 `resolveJsonModule`;`playwright.config.ts` 三项目:`classic`(无 cookie,默认行为)/`astryx`(`astryx-*.spec.ts`,storageState 预置 opt-in cookie)/`go-csp`(spawn `make build` 产物或 `GPT_LOAD_BINARY`,无二进制则 skip——Windows 无法编译 securefile,随 CI/Linux 跑);新增 `e2e/astryx-selection.spec.ts`(cookie×flag×fallback 7 用例矩阵,请求级断言文档标记 `/src/main.ts` vs `/src/frontends/astryx/main.tsx`)、`e2e/go-csp.spec.ts`(起停二进制、断言 `default-src 'self'`/`object-src 'none'`、零 `securitypolicyviolation`、零 console error,两文档分别以 `/assets/index` `/assets/astryx` 标记判别)。
- **Proof:** `astryx-selection` 7/7 PASS;全量 e2e **70 passed + 2 skipped**(go-csp 在本机按设计 skip)= classic 58 + astryx 12;`tsc -p tsconfig.node.json` 净;`eslint . --max-warnings=0` 净;`vite build` 绿(双 HTML + `astryx-*.js`/`astryx-*.css` 产物);`go test ./internal/webui -run '…'`(page_routes/server/frontend 相关)**全绿**(flag 加入真实 manifest 后 Go 解析与选路用例不受影响;预存 docker/workflow 失败与 B4 相同基线)。
- **PM:** dev 下 `gpt-load.frontend=astryx` cookie + `/settings` -> astryx 文档;清 cookie -> classic;未 flag 路由恒 classic。

### Task B6: 前端切换控件 + `feature.frontend-preview-switch` Doc ID
- [x] **Done**
- **Scope:** `shared/controllers/frontend-preference.ts`(`frontendCookieName`/`FrontendPreference`/`isFrontendPreference`/`readFrontendPreference`/`frontendPreferenceCookie`);`PreferencesControl.vue` 两形态(compact popover + inline)新增 Interface 分段控件(`--pair` 两列),选中即写 cookie + `location.reload()`;`vite.config.ts` dev selector 改用 shared `readFrontendPreference`;`scripts/verify-frontend-cookie.mjs` 跨语言契约(server.go 字面量 == shared 常量 + vite/组件消费 shared 模块)并入 lint 链;`.docs/db/features/frontend-preview-switch.md` 落地(trigger→action→contract→boundaries);三语言 catalog `shell.frontend*` 键。
- **Proof:** `verify-frontend-cookie` PASS;`docs:check` 5 docs 净;`verify:i18n-icu` 8,132 消息 PASS;type-check/eslint 净;e2e **71 passed + 2 skipped**(astryx-selection 8/8 含真往返:login 页 Preview 单选 → cookie=astryx → reload → flag 路由取 astryx 文档);`vite build` 绿。
- **PM:** `/login` 偏好面板选 Preview -> 写 cookie + reload;`/settings` 等 flag 路由进新 shell,未 flag 路由保持 classic;astryx 侧反向控件随 B9。
- **Doc IDs:** `feature.frontend-preview-switch`。
- **Notes:** 全量跑中曾出现 vite 依赖优化重载(compiler-runtime 晚优化)导致 8 个用例瞬时失败,缓存热后复跑全绿——非代码回归。

### Task B7: TanStack Router 装配(manifest 适配 + 护栏 + 滚动/标题/播报)
- [x] **Done**
- **Scope:** `shared/routing/route-meta.ts` 单源 meta 表(titleKey/requiresAuth/adminOnly/primaryNav/messageNamespaces,`pageRouteMetaFor` fail-fast),classic `router.ts` 改为查表(运行时 meta 不变);`shared/i18n/namespaces.ts` 承载 `MessageNamespace` 联合,classic `i18n/index.ts` re-export;`astryx/app/`:`route-adapter.ts` 纯函数(`:id`→`$id`,零运行时依赖保证 node:test 可加载)、`services.ts`(queryClient retry:false + `createApiClient` + `createAuthSession` + unsavedChanges,ref 打破 apiClient↔session 构造环,`onSessionCleared` 回接 router)、`router.tsx`(rootRoute+manifest 子路由,`staticData{pageName,meta}`,`beforeLoad` 依序:尾斜杠 `notFound()`(TSR 'preserve' 仍匹配 `/x/`,补 canonical 检查对齐 classic strict)→adminOnly→requiresAuth→`/login?redirect=…`;`trailingSlash:'preserve'`+`caseSensitive:true`+`scrollRestoration`;`HeadSync`(titleKey→document.title 占位,B8 换 t())+`RouteAnnouncer` aria-live+`<html lang>`)、`pages.tsx`(LoginPageStub 真实 login()+safeRedirect 跳转;RoutePageStub/NotFoundPageStub)、`safe-redirect.ts`(`getMatchedRoutes` 适配 shared `safeRedirectTarget`,blocklist=login+not-found)、`useUnsavedGuard` 桥接 useBlocker;`main.tsx` 重接 Theme+QueryClientProvider+Services+RouterProvider。动态路由数组放弃字面量 routeId 类型,导航一律 `href`(生成树不注册文件路由)。
- **Proof:** `test:astryx-routes` node:test 4/4(覆盖、$param、meta 完备、auth 契约);`e2e/astryx-routing.spec.ts` 5/5(未登录 `/settings`→`/login?redirect=%2Fsettings`+表单、`/groups/42` 参数路由、尾斜杠与 `/SETTINGS` 大小写 → not-found、直连 `/login` 仍 classic);astryx 项目 18/18、classic 58/58 回归绿;type-check×3/eslint/ICU/theme/cookie 契约全绿;`vite build` 绿(astryx bundle 545.9kB 含 TSR+react-query)。
- **PM:** cookie=astryx 访问 `/settings` 未登录 -> 同文档内跳 `/login?redirect=…`,表单可用;`/settings/`、`/SETTINGS` -> not-found stub。

### Task B8: react-intl 运行时 + Astryx 组件文案 + `verify:astryx-i18n`
- [ ] **Done**
- **Scope:** `shared/i18n` loader:嵌套 catalog 拍平点路径 key(与 classic key 一致)、按路由 `beforeLoad` 懒加载命名空间、每命名空间先合 en-US 再叠当前语言、`IntlProvider`+命令式 `createIntl` 实例、`useT()` helper、`onError` dev/test 抛 `MISSING_TRANSLATION`/`FORMAT_ERROR`、生产记日志;`MessageId` 类型由 en-US 拍平 catalog 派生并测 tsc 成本(不行则退回 string+parity 脚本,记入 gate);`InternationalizationProvider` 接同 locale;`astryx/locales/{zh-CN,ja-JP}.json` 对齐 `@astryxdesign/core/locales/en.json`;`verify:astryx-i18n` key 对齐脚本。
- **Proof:** `verify:i18n-icu`+`verify:astryx-i18n` 绿;astryx 项目在三语言渲染 shell 无 MISSING/FORMAT 错误(spike f 证据并入)。
- **PM:** 新 shell 切三语言 -> 界面与 Astryx 组件文案同切。

### Task B9: shell 对等(AppShell/AuthGate/login/not-found/偏好/密度)
- [ ] **Done**
- **Scope:** `AppShell` 导航(含 admin-only 项、principal type)、`AuthGate`、`login`、`not-found`、偏好面板(主题、locale、返回-classic 控件)、路由播报接入;根级 `SizeContext`/`Table` density + theme 覆盖达成方案文档「Theme and tokens」全部密度指标;`login`/`not-found` 路由首批 `astryx:true`。
- **Proof:** astryx 项目 e2e:login→shell→logout、导航 admin 项可见性、404 页;密度实测表逐项 ≤±1px(gate #6 输入)。
- **PM:** cookie=astryx 登录后四页走查 -> 布局/密度与 classic 一致。

### Task B10: spike(a) `groups` 集合页(Table,1,000 行,服务端驱动)
- [ ] **Done**
- **Scope:** `frontends/astryx/features/groups/` 集合页:`Table` + `sortable`/`filtering`/`pagination`/`stickyColumns` 等插件的受控 `*State`(ADR-0001 服务端分页,禁 `paginateData`);typed search 承载筛选参数;键盘导航;路由标 `astryx:true`。同步产出 spike(e) typed-search 对 `logs`/`group-detail` 的 parity 验证记录。
- **Proof:** 改写后的 collection e2e 在 astryx 项目绿;1,000 行下交互正常(gate #3)。
- **PM:** astryx 前端打开 `/groups` -> 排序/筛选/分页/键盘走查与 classic 一致。

### Task B11: spike(b) 日志详情层(Dialog/BottomSheet/侧栏 swizzle 决策)
- [ ] **Done**
- **Scope:** 以 `logs` 域日志详情为对象试 `Dialog`/`BottomSheet`/swizzled 侧栏三种手段,选定最少 swizzle 方案;焦点陷阱、Esc、焦点归还、窄屏回退。
- **Proof:** e2e 断言焦点行为与 Esc;swizzle 计数与原因记录(gate #5 输入)。
- **PM:** 日志行打开详情 -> Tab 困在层内、Esc 关闭、焦点回行。

### Task B12: spike(c) `DateRangeInput` 日志时间过滤(三语言)
- [ ] **Done**
- **Scope:** `logs` 时间范围过滤用 Astryx `DateRangeInput`/`DateTimeInput`,接 shared 查询规整;zh-CN/ja-JP 组件文案来自 B8 本地 catalog。
- **Proof:** astryx 项目 e2e 在三语言下设置/清除时间范围,结果集正确。
- **PM:** ja-JP 下打开时间过滤 -> Astryx 文案为日语,过滤生效。

### Task B13: gate 度量与记录
- [ ] **Done**
- **Scope:** 汇总七条门槛证据:Chromium 125(独立 playwright 项目,`executablePath` 指 chrome@125)+ 最新 Chrome 的 Go-CSP 零违规报告;三模式无 flash 记录;1,000 行集合 e2e 结果;新旧 shell 首屏 JS+CSS gzip 对比实测;swizzle 清单(≤3);密度 ±1px 全表;`verify:i18n-icu`/`verify:astryx-i18n` 输出与三语言 MISSING/FORMAT 扫描。证据写入本文件 `## Review` 上方的 gate 记录小节。
- **Proof:** 证据齐全;每条门槛有对应命令输出或测量表。
- **PM:** 用户据证据做 go/no-go;no-go 则停并记录原因、删脚手架。

## Review

- [ ] Review complete
