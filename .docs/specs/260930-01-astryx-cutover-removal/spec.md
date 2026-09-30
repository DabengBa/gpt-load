# Spec: Astryx 默认化与 classic 删除(Phase 5+6 合并交付)

## 意图与核心流程

- **意图**:把 Astryx 从"cookie 可选预览"翻转为唯一管理前端,并删除
  `frontends/classic` 及全部共存基建(前端选择机制、双入口构建、Vue 工具链、
  双 Playwright 项目)。
- **参与者/触发条件**:浏览器对页面路由发起 GET 请求(接受 HTML);部署方
  `make build` 产出单前端 dist。
- **主路径**:
  1. 任意已注册页面路由(`page_routes.json` 11 条)→ Go 返回唯一内嵌
     `index.html`(Astryx 文档,200)。
  2. 未知路径 → 同一文档,404 状态,Astryx not-found 视图。
  3. 文档加载 → `src/frontends/astryx/main.tsx` → TanStack Router 接管全部
     页面路由。
  4. 不再存在任何前端选择分支:无 cookie 读取、无第二文档、无 manifest flag。

## 范围 / 不做范围

### 范围

**A. Go 服务端单文档化**(`internal/webui/`)

- `server.go`:删除 `astryxIndex` 字段与加载分支、`frontendCookieName`、
  `frontendPreference`、`indexFor`、`indexForNotFound`;`index` 成为唯一文档;
  `serveIndexWithStatus` 移除 `Vary: Cookie`(响应不再随 cookie 变化)。
- `page_routes.go` / `page_routes.json`:删除 `astryx` 字段(全部恒真),
  manifest `version` 2 → 3,`pageRoute.Astryx` 字段删除。
- `server_test.go` / `page_routes_test.go`:选择矩阵用例改写为"始终返回
  index",manifest fixture 去掉 `astryx` 字段、版本号更新。

**B. web 构建单入口**(`web/`)

- `index.html` 成为唯一入口:`<script>` 指向 `/src/frontends/astryx/main.tsx`;
  删除 `astryx.html`。
- `vite.config.ts`:删除 `frontendSelectorDevPlugin`、`@vitejs/plugin-vue`、
  `@tailwindcss/vite`;`rollupOptions.input` 收敛为单 `index.html`;
  `optimizeDeps.entries` 单入口;删除 `@` alias(指向 classic);删除
  `cssInjectionTarget`(单 css 产物无需按文件名分流);`useCSSLayers.before`
  中 `classic-tokens` 层名随 tokens 迁移改名为 `tokens`(见 D)。
- 删除 `src/main.ts`(classic bootstrap 入口)。

**C. classic 树与依赖删除**

- 删除 `src/frontends/classic/`(约 2.4MB / 164 个 `.vue`)整树,包括其内部对
  `@shared` codec 的 re-export shim。
- `package.json` 移除:`vue`、`vue-i18n`、`vue-router`、`reka-ui`、
  `@lucide/vue`、`@tanstack/vue-query`(deps);`@vitejs/plugin-vue`、
  `vue-tsc`、`eslint-plugin-vue`、`@vue/eslint-config-typescript`、
  `@tailwindcss/vite`、`tailwindcss`(devDeps)。
- `pnpm-workspace.yaml`:删除 `vue-demi` allowBuilds 条目。
- `tsconfig`:删除 `tsconfig.app.json`(vue-tsc 用);保留 `tsconfig.astryx.json`
  作为唯一应用配置(覆盖 `src/frontends/astryx` + `src/shared`);根
  `tsconfig.json` references 收敛为 astryx + node;`type-check` 脚本相应收敛。
- `eslint.config.mjs`:删除 `eslint-plugin-vue` / `vueTsConfigs` 接入、`.vue`
  文件规则、restricted-import 中的 vue 系正则。

**D. astryx 侧共存残留收敛**

- `src/frontends/astryx/app/shell/PreferencesControl.tsx`:删除 Interface
  (frontend)段与 `updateFrontend`;移除 i18n 键 `shell.frontend`、
  `shell.frontendClassic`、`shell.frontendPreview`(三套 catalog +
  `message-ids` 同步删除,`verify:i18n-icu` 守护 parity)。
- `src/shared/controllers/frontend-preference.ts`:删除;`lint` 脚本摘除
  `verify-frontend-cookie.mjs` 并删除该脚本(跨语言契约对象消失)。
- `src/shared/routing/page-routes.ts`:删除 `astryx` 字段(`routeFields`、
  校验、`pageRouteEntries` 元素类型);`astryxRoutePaths` 收敛为全量。
- `src/frontends/astryx/app/route-link.tsx`:`isAstryxNavigable` 删除——全部
  页面路由与未知路径均由 astryx 承担,SPA 导航对所有内部 href 恒安全;
  调用点(`LoginView.tsx`、`pages.tsx` 的 unflagged 守卫、`router.tsx` 的
  `astryxRoutePaths` 过滤)直接收敛。
- **密度 token 迁移**:`src/frontends/astryx/entry.css` 当前
  `@import '../classic/styles/tokens.css' layer(classic-tokens)`——tokens 是
  astryx 密度契约来源,不可随 classic 删除。迁至
  `src/frontends/astryx/theme/tokens.css`,层名 `classic-tokens` → `tokens`
  (`entry.css` @layer 列表 + vite `useCSSLayers.before` 同步);
  `gptload.theme.ts` 注释中的 classic 路径引用同步更新。

**E. e2e 收敛**

- `playwright.config.ts`:删除 `classic` project;`astryx` project 移除
  cookie storageState(默认即 astryx)。
- 删除 8 个 classic 项目 spec:`model-test-alias`、`request-log-affinity`、
  `request-log-display`、`request-log-page-polish`、`request-log-readability`、
  `request-log-route-layout`、`schedule-editing`、`schedule-routing`。
- 改写含共存断言的 astryx spec:`astryx-selection.spec.ts`(cookie 选择语义
  → 默认即 astryx,无 cookie)、`astryx-shell.spec.ts`(frontend 段用例删除)、
  `astryx-routing.spec.ts` / `astryx-visual-audit.spec.ts` / `go-csp.spec.ts`
  中 cookie 种子与双文档断言收敛。

**F. 文档**

- Doc ID `feature.frontend-preview-switch`:机制消失,删除
  `.docs/db/features/frontend-preview-switch.md`,重建 `dist/`。
- `.docs/db/features/model-test-alias.md` 中 "classic" 措辞更新为 astryx 语义;
  `.docs/tech/model-test-alias.md` 的 `code.paths` 改指 astryx 路径。
- `.docs/tech/astryx-migration-plan.md`:补 Phase 5+6 outcome。
- 新增 `.docs/adr/0003-single-frontend.md`(或同等序号):记录"删除回退窗口、
  单前端化"这一难逆转决策及用户确认。
- `PRODUCT.md` / `README.md` 中如有 classic/Vue 表述同步更新。

### 不做范围

- 不保留任何形式的 cookie 回退、URL 参数选择或双文档构建(用户明确合并交付,
  放弃一个发布周期的回退窗口;回退手段为版本回滚)。
- 不改 API 契约、后端业务逻辑、鉴权流程。
- 不做 astryx 视觉/交互重设计。
- 不引入新依赖;依赖只减不增。
- `shared/` 下仍被 astryx 消费的模块不动;只删 classic 独占消费者
  (`frontend-preference.ts`)。

## 边界规则 / 验收

| 场景 | 预期 |
| --- | --- |
| 无 cookie GET 任一页面路由 | 200 + astryx index(唯一文档) |
| `gpt-load.frontend=classic` cookie 残留 | 忽略,仍返回 astryx index |
| 未知路径 | 404 + astryx index(客户端 not-found) |
| `astryx.html` 构建产物 | 不存在;dist 只有 `index.html` 单文档 |
| `pnpm install` 后依赖树 | 无 vue / vue-i18n / vue-router / reka-ui / tailwindcss 等 |
| 偏好面板 | 无 Interface 段;主题/语言段照常 |
| `type-check`/`lint`/`format`/`build` | 全绿;lint 不再跑 `verify-frontend-cookie` |
| Playwright | astryx + go-csp + chromium-125 + chrome-latest 项目全绿;无 classic 项目 |
| `make test`(Go) | 单文档断言更新后全绿 |
| `docs:check`/`docs:build` | 绿;`frontend-preview-switch` Doc ID 移除 |

## 架构 / 约束

- 选择机制是**删除**不是翻转:不实现"默认 astryx + cookie=classic 回退"的
  中间态;合并交付直接到单文档终态。
- `indexCSP`、安全响应头、`Cache-Control: no-cache`(文档)/`immutable`
  (assets)不变;仅 `Vary: Cookie` 随 cookie 语义消失而移除。
- manifest 仍承担"哪些路径返回 SPA 文档"的职责,`pageRouteShape` 去重与路径
  校验保留;仅删 `astryx` 字段并 bump `version` 到 3。
- astryx SPA 现在必须承接**所有** manifest 路径与未知路径——`router.tsx` 的
  路由表即 manifest 全集,not-found 由 astryx 渲染。
- 浏览器基线 Chrome 125+ 不变(`browserslist`/`build.target` 保留)。
- CSP 下主题必须保持静态构建(`astryx theme build` + `verify-theme-build`),
  不得引入运行时样式注入。
- tokens 迁移只改文件位置与层名,**不改任何 token 值**(密度契约冻结,
  `astryx-density.spec.ts` ±1px 门继续守护)。

## 数据 / 集成

- 无 API/schema/存储变更。
- 浏览器残留的 `gpt-load.frontend` cookie(max-age 1 年)被忽略,自然过期;
  服务端不写不读。
- `internal/webui/dist` 由 `pnpm --dir web run build` 重新生成;Go
  `//go:embed all:dist` 契约不变(单 index)。
- `gpt-load.import-reauth-draft` 等浏览器状态键不变(astrx sessionStorage
  恢复链已交付)。

## 验证

```powershell
pnpm --dir web install          # 依赖收敛后
pnpm --dir web run build        # type-check + vite build,单 index 产物
pnpm --dir web run lint         # eslint + i18n-icu + theme-build
pnpm --dir web run format
pnpm --dir web run test:search-codec / test:logs-route / ... # 全部 codec 单测
pnpm --dir web run verify:astryx-i18n / verify:*.mjs         # shared 校验脚本
# Playwright:astryx / go-csp / chromium-125 / chrome-latest 全项目
make test                        # Go 单测(server/page_routes 更新后)
pnpm --dir web run docs:check && pnpm --dir web run docs:build
```

真机证据(沿用 Phase 4 模式):WSL2 构建 Linux 二进制 → `GPT_LOAD_BINARY`
+ `GPT_LOAD_ORIGIN` 起服务 → go-csp 三浏览器矩阵;curl 验证无 cookie /
`cookie=classic` / `cookie=astryx` 三种请求头均返回同一 astryx 文档。

## Doc ID 契约

- `feature.frontend-preview-switch`:**退役删除**——该 Doc ID 绑定的用户可见
  控件(Interface 选择)与偏好持久化机制整体消失;删除
  `.docs/db/features/frontend-preview-switch.md` 并重建 `.docs/db/dist/`。
- 其余 Doc ID 不变;凡 `code.paths`/prose 引用 `frontends/classic` 的文档
  (`tech/model-test-alias.md` 等)改指 astryx 对应路径。
- `pnpm --dir web run docs:check` 验证语义库一致性。

## 参考资料

- `.docs/tech/astryx-migration-plan.md`(Phase 5/6 定义、Verification 门)
- `.docs/tech/briefs/astryx-cutover-removal.md`(用户确认的合并决策)
- `.docs/db/features/frontend-preview-switch.md`(待退役契约)
- `internal/webui/server.go`(`frontendCookieName`/`indexFor`/`astryxIndex`)
- `internal/webui/page_routes.go` + `page_routes.json`(manifest v2)
- `web/vite.config.ts`(双入口/selector 插件/`classic-tokens` 层)
- `web/src/shared/controllers/frontend-preference.ts`(cookie 契约 TS 侧)
- `web/src/frontends/astryx/entry.css`(tokens 导入)
- `web/playwright.config.ts`(项目划分)
- `web/package.json`、`web/pnpm-workspace.yaml`、`web/eslint.config.mjs`
- `web/e2e/astryx-shell.spec.ts`(frontend 段用例)
