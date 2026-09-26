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
- [x] **Done**
- **Scope:** `shared/i18n/catalogs.ts` 承载 core/namespace loader 表 + `flattenMessages` + `catalogLoader`,classic `i18n/index.ts` 去重引用;`shared/i18n/message-ids.ts` 由 8 个 en-US catalog `import type` 递归出 `MessageId` 联合(tsc 无可见增量:3.6s→3.6s,保留严格类型不回退 string);`PageRouteMeta.titleKey` 收窄为 `MessageId`(双端 router 同受益);`shared/preferences/locale.ts` 导出 `localeStorageKey`。`astryx/app/i18n.tsx`:`createAppI18n`(getBrowserLocale→core+en-US 合并→`subscribe`/`getSnapshot`/`setLocale`/`ensureNamespaces`,pending 去重 + requestedLocale 防乱序,镜像 classic 语义)、`emit()` 同步 `<html lang>` 与命令式 `getIntl()`、`onError` dev/test throw/生产 console、`AppI18nProviders`(IntlProvider+`InternationalizationProvider` 同 locale,`messages={'zh-CN','ja-JP'}` 用上游 shipped catalog)、`useT()`(MessageId+PrimitiveType values→string)。router `beforeLoad` 在 auth 守卫后 `ensureNamespaces(meta.messageNamespaces)`;`HeadSync` 用 t() 翻译 titleKey;stub 页面(login/route/not-found)全部走真实 catalog key。`services.i18n` 注入 services(apiClient getLocale 接 controller);main.tsx 启动序对齐 classic:先 `await createAppI18n()` 再建 services/router。
- **Proof:** `verify:i18n-icu` 8132 条 PASS;`verify:astryx-i18n` PASS(en/zh-CN/ja-JP 370 key,0 extra;zh-CN/ja-JP 各缺 122 条属上游翻译滞后,per-key en 回退,WARN 不 fail);`astryx-i18n.spec.ts` 4/4(三语言 `/settings` 标题/h1/`html lang` 断言 + reload 切语言);astryx 项目 22/22、classic 58/58 回归;TSC×3/eslint/build 全绿;lazy catalog 在生产构建中仍按 locale×namespace 分 chunk。
- **PM:** cookie=astryx 访问 `/settings`,三语言下 h1/标题/`html lang` 随 `gpt-load.locale` 切换,无 MISSING/FORMAT。

### Task B9: shell 对等(AppShell/AuthGate/login/not-found/偏好/密度)
- [x] **Done**
- **Scope:** `astryx/app/shell/`:`Shells.tsx`(`PublicShell`/`AuthedShell`:AuthGate 包顶栏,桌面 nav 按 principal_type 过滤 adminOnly 项,access_key 附加 `Access key · Read-only` 徽章并隐藏 import 动作;skip-link、移动 nav 走偏好弹层;logout 对 import 页走 `bypassNext`+先跳后清)、`AuthGate.tsx`(validating/locked/network/invalid-response 四态卡,access_key 命中 adminOnly 回 home,invalid-response 聚焦 retry)、`LoginView.tsx`(全对等:intro rail、reveal、required/whitespace 校验、invalid/locked(含 countdown)/network/invalid-response 反馈、`?redirect=`+`?help=auth` 规范化、import 草稿恢复)、`NotFoundView.tsx`(404 面板+请求路径+回首页/上一页)、`PreferencesControl.tsx`(Popover+三段式 theme/locale/frontend,frontend 写 `gpt-load.frontend` cookie+reload;`dialogLabel` 消 a11y 警告)、`BrandMark`/`use-countdown`。路由层:`login` 在 manifest 标 `astryx:true`;尾斜杠 canonical 检查上移至 root `beforeLoad`,not-found 渲染经 `ShellOutlet` 的 `matches[].status==='notFound'||_notFound` 判定落到 PublicShell(否则 AuthGate 匿名卡会吞掉 404);`LinkProvider` 把 astryx 组件内链接接到 TSR;`services.ts` 补 `importRecovery`;`entry.css` 引入 classic `tokens.css` 作 `classic-tokens` 层(5 个语义色重名属有意对齐),`useCSSLayers.before` 同步。`useT` 的 `formatMessage` 收 PrimitiveType values 保 `string` 返回;`FlatKeys` 补数字键(`capabilities.1.*`)。
- **Proof:** `astryx-shell.spec.ts` 5/5:login→shell→sign-out 往返(auth-key 清除回 `/login`)、access_key 在 `/settings` 被 adminOnly 弹回 `/` 且仅见 Home/Models/Monitor/Request logs(无 import、带只读徽章)、404 显示请求路径并回首页、偏好面板 theme→`data-theme=dark`/locale→zh-CN+`html lang`/frontend→`classic` cookie+classic 文档、密度实测 topbar 54px/padding 30px/import 动作 30px/shell 13.5px 全部 ≤±1px;`astryx-routing.spec.ts` 断言换成真实视图锚点(标题文本/表单 label,`data-route` stub 标记随 stub 移除),文档选择断言改为 `/groups` classic vs `/login` astryx;`astryx-theme.spec.ts` 迁到 `/login`,precedence 断言改为运行时 `document.styleSheets` 层序(`reset→astryx-base→astryx-theme→classic-tokens→app.priority*`,IconButton 半径 7px vs neutral 8px)加 `--size-element-md` 34px/正文 13.5px。astryx 项目 27/27、classic 58/58 绿;tsc×2/eslint(0)/ICU 8132/astryx-i18n(370 key,upstream lag WARN)/theme-build/cookie 契约全绿;`vite build` 绿(astryx bundle 734kB/216kB gzip,>500kB 警告留待 B13 gate 汇总)。修 Mock:`/api/auth/session` 需要 `{code:0,message,data}` 信封而非裸 payload(此前 AuthGate 未启用故未暴露)。
- **PM:** cookie=astryx 登录后 `/settings`/`/login`/404 走查 -> 布局/密度与 classic 一致;偏好面板可切主题/语言/返回经典。

### Task B10: spike(a) `groups` 集合页(Table,1,000 行,服务端驱动)
- [x] **Done**
- **Scope:** `frontends/astryx/features/groups/GroupsView.tsx`:Astryx `Table` 全受控(`useTableSortable` 5 值服务端 sort 枚举映射列+方向、`useTablePagination` 纯受控无 `paginateData`、stickyColumns;filtering 插件为列头绑定形态不匹配 classic 工具栏契约故未用,筛选走 Selector/状态 chip);typed search 由 `validateSearch`+shared `parseGroupCollectionRouteQuery` 承载(search/status/channel_type/sort/page/page_size),URL 为唯一真源;搜索 300ms 去抖、筛选/排序变更重置 page=1、越界页按响应 total_pages 纠偏(服务端回声请求页+空 items,客户端导航修正);React Query 承载 collection+channels;乐观启停+`invalidateGroupCollections`、copy 导航、行动作(enable/disable/copy/detail)、通道图标、凭据健康条、空/无结果态、summary chips。基建:`use-debounced-action`/`use-visible-refetch`/`collection-loading` React port、`ToastHost`(shared toast controller→服务)、`components/ChannelIcon`+`CredentialHealthBar`(React port);channel-icons 注册表+全部 svg/webp 资产从 `frontends/classic/assets` 提升到 `shared/assets/`(classic ChannelIcon 改引,`gateway-clients.ts` 注释同步)。路由层:`/groups` manifest 标 `astryx:true`;`routeViews` 注册表替掉 login 单例判断。**TSR 边界关键修复**:默认 codec JSON 解码 query(`?page=2`→number 2)且 `search.strict` 会把已验证 search 回写——shared query 契约原只收 string 导致 `scalarRouteQuery` 丢弃 number、`page` 恒回落 1、navigate 静默无请求;修复为自定义 `parseSharedRouteSearch`/`stringifySharedRouteSearch`(手写 decodeURIComponent 保 vue-router 语义:`+` 不转空格、重复 key→数组)+`SharedRouteQueryValue` 接受有限 number(validateSearch 幂等性要求,TSR 会用已验证值复验)。`vite.config.ts` `optimizeDeps.entries` 加 `astryx.html` 且 `include` 钉全部 `@astryxdesign/core/*` 深路径+`react/compiler-runtime`/`jsx-runtime`/`jsx-dev-runtime`(babel/plugin-react 注入,crawl 不可见)→消除运行中重优化导致 `.vite/deps` 重建、in-flight 页面 404 的复发性 flake。
- **Proof:** `astryx-groups.spec.ts` 7/7:summary chip 103/22/25、search 去抖入 URL+服务端参数、status chip/channel_type Selector、受控 sort(Selector+列头双向)、服务端分页+`?page=99` 越界纠偏到末页、键盘可达搜索/行动作、1,000 行 gate(交互正常+确认服务端分页非客户端切片)、空/无结果态+reset filters 双路径。E2E fixture 修正两处:`makeGroups` 索引映射(`Array.from` 传元素非索引曾产 `id:null`/`Group 0NaN`)、disabled 组凭据全 disabled(投影不变式);mock 改为镜像 Go 契约(越界回声请求页+空 items,非钳位)。全量 e2e 92 通过+2 skipped、冷缓存零 pre-transform error;tsc×3/eslint(0)/全部 verify 脚本绿。
- **PM:** cookie=astryx 打开 `/groups` -> 搜索/筛选/排序/分页/键盘走查与 classic 一致,越界 deep-link 自动纠偏。

### Task B11: spike(b) 日志详情层(Dialog/BottomSheet/侧栏 swizzle 决策)
- [x] **Done**
- **Scope:** `/logs` manifest 标 `astryx:true`;`frontends/astryx/features/logs/LogsView.tsx`:服务端驱动日志列表(shared `requestLogsQueryOptions`,行内 `View details` 动作),`?selected_request_id=` 为唯一选中态真源(`navigate` 写参/清参),详情走 `requestLogDetailQueryOptions` 服务端拉取(不信行载荷),展示 status/请求摘要/client→upstream model/attempt count/attempt chain。**决策见 `.docs/adr/0002-astryx-detail-layer-primitive.md`:选 Astryx `Dialog`,0 swizzle**——`position={{end:0,top:0}}`+xstyle 给抽屉外壳(全高、无圆角、左缘边线、`min(92vw,520px)`、≤520px 全幅、滑入动画+reduced-motion);`BottomSheet` 否决(底部锚定手势/snap 是移动形态,抑制 handle/手势比 xstyle 重);swizzled 侧栏否决(reka AppDrawer 本身即为 swizzle,无必要复刻)。新组件 `components/DetailPanel.tsx` 为可复用 AppDrawer 对应物(`title`/`subtitle`/`dismissible`→`purpose` info|required/`footer`)。三个实现要点:① 组件必须常驻、`isOpen` 由 URL 参驱动——条件渲染卸载会跳过 Dialog 的 trigger 捕获/焦点归还生命周期;② `LayoutContent` 带 `data-autofocus`+`tabIndex=-1`——Dialog 在 `showModal` 后只认此标记做自动对焦(组件级 autofocus 在 dialog 可见前 commit 被静默丢弃);③ Chromium 原生 modal `<dialog>` 在 tab 序边界把焦点静默落到 `<body>`(无 focus 事件),`focusin` 重定向抓不到——同 reka 哨兵机制在 Dialog `onKeyDown` 拦截 Tab/Shift+Tab 做硬收容,另留 `focusin` 守卫兜事件化逃逸;cleanup 先于 Dialog isOpen→false effect 跑,不劫持关闭时的焦点归还。`search-codec.ts` 从 `router.tsx` 抽出独立模块(B10 codec 复用,断 router↔view 环)。
- **Proof:** `astryx-log-detail.spec.ts` 6/6:行动作开面板+URL 深链(`selected_request_id` uuid 形)、详情含 attempt chain+upstream model、焦点硬困(activeElement 全程不出 dialog,Tab/Shift+Tab 边界实测)、Esc 关+清参+焦点归还触发行、scrim 点击关+清参、`?selected_request_id=` 直达开层、480px 视口全幅(实测宽 480)。swizzle 计数 0(gate #5 输入)。astryx 项目 40/40、classic 58/58 绿;vue-tsc/tsc×2/eslint(0) 绿。
- **PM:** cookie=astryx 打开 `/logs` -> 行动作开右侧面板,Tab 困于层内、Esc/scrim 关闭、焦点回触发行,窄屏全幅。

### Task B12: spike(c) `DateRangeInput` 日志时间过滤(三语言)
- [x] **Done**
- **Scope:** `/logs` 加 `LogTimeRangeFilter`(LogsView 内):`DateTimeInput`×2(from/to,`hasSeconds`+`hourFormat=24h`+`hasClear`+`size=sm`)对应 classic picker 的 from/to 字段,草稿值为同一 `YYYY-MM-DDTHH:mm:ss` 本地串;8 个 `dateTimePresets` 快捷 chip(`resolveDateTimePreset` 直写 URL,同 classic shortcut 净效果);`<form>` 承载 Enter-to-apply。`DateRangeInput` 否决:仅日粒度,无法表达 classic 的时分秒精度与 `now` 端点预设。**语义镜像 classic `log-filters.ts`**:draft 编辑不发请求,Apply 校验(`errors.dateTime`/`errors.range` 逐字段 status,from≥to 阻断)后写 `from_ms`/`to_ms`,Reset 删参回落 `defaultLogRange()`;URL 无参/非法时 `parseLogRangeMs` 返 undefined → 始终带默认窗(now−24h..now+24h)请求,与 classic `parseAppliedLogFilterState` 一致。shared `request-log-route.ts` 增 `parseLogRangeMs`(双参规范化整数+from<to)+`defaultLogRange`;draft 经 render-adjust 从 appliedRange 重同步(useMemo 键 raw 参数字串防每渲染重播种默认窗)。`optimizeDeps.include` 钉 `@astryxdesign/core/DateTimeInput`。
- **Proof:** `astryx-log-time-range.spec.ts` 3/3(fixture 加 `installRequestLogRangeRoutes`:行 `completed_at_ms` 按安装时刻 now−30m/−2d/−10d,mock 按 from_ms/to_ms 真过滤):en-US 首载默认窗(实测 span≈48h)仅 recent 行→键入 ISO 日期+HH:mm:ss 无请求→Apply 后 `from_ms`/`to_ms` 等于键入本地毫秒、2 行(old-model 隐藏)→Reset 删参回默认;zh-CN 断言 `开始时间`/`结束时间`/`选择日期`/`打开日历`/`应用`/`快捷时间范围` 均中文,'7d' chip 直写 ≈7d 窗→2 行;ja-JP 断言 `開始時刻`/`終了時刻`/`日付を選択`/`カレンダーを開く`,from>to 时 Apply 阻断零请求+`終了時刻は開始時刻より後である必要があります。` 入 assertive live region。astryx 43/43、classic 58/58、tsc×3/eslint(0) 绿。
- **PM:** 三语言下改时间范围 -> 字段/占位/日历开关/预设均为该语言;Apply 生效、Reset 回落默认窗、反向区间被拒。

### Task B13: gate 度量与记录
- [x] **Done**
- **Scope:** 汇总七条门槛证据:Chromium 125(独立 playwright 项目,`executablePath` 指 chrome@125)+ 最新 Chrome 的 Go-CSP 零违规报告;三模式无 flash 记录;1,000 行集合 e2e 结果;新旧 shell 首屏 JS+CSS gzip 对比实测;swizzle 清单(≤3);密度 ±1px 全表;`verify:i18n-icu`/`verify:astryx-i18n` 输出与三语言 MISSING/FORMAT 扫描。证据写入本文件 `## Review` 上方的 gate 记录小节。
- **Proof:** 见下方 `### Gate 记录`;每条门槛有对应命令输出或测量表。
- **PM:** 用户据证据做 go/no-go;no-go 则停并记录原因、删脚手架。

### Gate 记录(B13)

1. **浏览器矩阵 CSP 零违规**:Playwright 新增两项目 `chromium-125` 与 `chrome-latest`,与 `go-csp` 同跑 `go-csp.spec.ts`;`chromium-125` 经 `e2e/browser-executables.ts` 解析(`GPT_LOAD_CHROME_125_EXE` env → `~/.cache/puppeteer` 探针,未配置则 skip 而非悄悄回落 bundled)注入 `launchOptions.executablePath`,`chrome-latest` 用 `channel:'chrome'`。spec 清扫面扩为 manifest 驱动:classic `/` + 全部 `astryx:true` 路由(`/login`、`/groups`、`/logs`、`/settings`),每文档断言 CSP 头、`securitypolicyviolation` 事件数=0、console error=0。**结果 15/15**:bundled Chromium + Chrome for Testing 125.0.6422.78 + 系统 Chrome 153.0.8010.53。**方法学注记**:完整二进制在 Windows 开发机不可编译(`internal/platform/securefile`、`internal/catalog` 仅 linux 实现,spec 原有注释即载明),证据由新增 `internal/webui/cmd/webui` harness 产出——同一 `webui.NewServer`+`httproute.Registry.Bind` 链服务同一 embed dist,页面路由/资产/CSP/cookie 选择与生产一致;真实二进制仍在 Unix CI 跑同一 spec。`GPT_LOAD_BINARY` env 兼容两者。
2. **三模式无 flash**:`astryx-no-flash.spec.ts` 4/4 —— init-script 在 `DOMContentLoaded` 捕获 `data-theme`(light→`light`、dark→`dark`、system→attr 缺失),证明 `theme-bootstrap.js` 于初始解析期(首绘前)落值而非应用 JS 所为;另断言 bootstrap 为 head 内非 async/defer/module 阻塞脚本。`theme-bootstrap.js` 零改动。
3. **1,000 行集合**:`astryx-groups.spec.ts` "1,000-group collection stays interactive (gate #3)" PASS(滚动/筛选交互正常,断言走服务端分页而非客户端切片)。
4. **首屏 JS+CSS gzip**(`scripts/measure-first-screen.mjs`,`internal/webui/dist/.vite/manifest.json` 驱动:entry 静态闭包 + 入口级 `dynamicImports` 的静态闭包——classic `import('./bootstrap')` 为必经 boot split 计入;astryx 侧同规则计入其 3 个 DS 条件动态块(Tooltip/BottomSheet/MenuBottomSheet),为保守上界):

   | entry | JS gzip | CSS gzip | total | files |
   | --- | --- | --- | --- | --- |
   | index(classic) | 138.4 kB | 11.5 kB | 149.9 kB | 49 |
   | astryx | 390.7 kB | 38.5 kB | 429.2 kB | 38 |
   | delta | +252.3 kB | +27.0 kB | **+279.3 kB** | |

   解释:React 19+DS 运行时+react-intl 路由壳全量先于路由视图加载,较 Vue 壳重;路由视图仍 code-split。>500kB 原始体积警告与本表同源,留 Phase 3 汇总。
5. **Swizzle 清单:0(≤3)**。`frontends/astryx/components/` 全部四件皆为公共 API 组合:`DetailPanel`(Dialog+Layout 组合 + xstyle 外壳,ADR-0002)、`ChannelIcon`/`CredentialHealthBar`(域组件 React port,shared assets)、`ChannelIcon.css`。无任何 DS 内部 fork 或 dist 路径导入。
6. **密度 ±1px 全表**(`astryx-density.spec.ts` 4/4;token 级读 themed scope 计算值,渲染级实测):
   - token:control sm/md/lg **30/34/38**、text body/supporting/label/large/h3/h2 **13.5/12/11.5/16/16/22**、radius inner/element/container **6/7/10**、spacing-1 **4px** — 全中。
   - 渲染:IconButton **34px**(±1 内)、shell 正文 **13.5px**、radius **7px**、topbar 54px/padding 30px/import 30px(B9 shell 断言复跑仍绿)。
   - **触达目标(新修)**:classic `≤860px` 时 shell 动作区升为 `--touch-target` 44px——主题 `adaptations.widthBreakpoints.md=861`+`when.width.below:'md'`→`components.button.base.min/minWidth=44px` 镜像之(IconButton 渲染为 Button,一键覆盖 preferences trigger 与 import action);差异记录:astryx 为组件级提升,classic 为选择器级,窄屏下覆盖面更宽。实测 480px 视口 trigger 44px。
   - **集合行(新修)**:`--collection-row-height` 48px 在 classic 为声明未用 token(真实面:logs ledger 52px、组卡片 96px);astryx `Table` `density="compact"` 实测 **39px** → 按机制表换 `density="balanced"` 实测 **47px**,对 48px 声明目标 ±1 内。
   - **无对应项(记录非断言)**:classic `--control-lg` 42px(astryx 无 element-xl 挡位,已迁移面无该控件)、`--setting-control-height` 26px(settings 行内控件不属 Phase 1 面)、`--text-label-xs` 10.5px(astryx 最小 label 挡位 11.5px)。
7. **i18n**:`verify:i18n-icu` **PASS**(8132 条);`verify:astryx-i18n` **PASS**(370 key ×3 locale;zh-CN/ja-JP upstream lag 122/370 → en 回落,WARN 非 MISSING_TRANSLATION/FORMAT_ERROR);`astryx-i18n.spec.ts` 4/4(zh-CN/en-US/ja-JP 渲染+`html lang`+切换)。

**回归尾**:`vite build` 绿(target chrome125);`theme:check` 零 diff;tsc(app/astryx/node)/eslint(0)/契约测试 21/21(astryx-routes、channel-contract、connection-json)绿;astryx **51/51**、classic **58/58**、CSP 矩阵 **15/15**。`internal/webui` Go 测试在 stash 基线上同现失败(workflow/docker YAML 契约用例对本 fork 的 CI 文件为预存漂移),与 B13 无涉;harness 包 `cmd/webui` 编译+vet 净。

**go/no-go 判断**:七条门槛均有可复核证据;惟 #1 的生产二进制路径在本机以等价 handler 链 harness 代替(真实二进制走 Unix CI),#6 有三项 classic 指标无 astryx 对应面(已列明)。其余全数达标。

## Review

- [ ] Review complete
