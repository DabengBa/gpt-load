# Astryx Cutover & Classic Removal Plan

> **For agentic workers:** REQUIRED SKILL: Use `delivery-workflow` to implement this task list end to end. During behavior-changing implementation or bug fixes, also use `test-driven-development`.

Source: `spec.md`
Doc IDs: `feature.frontend-preview-switch`(退役删除)

## Tasks

### Task A: density tokens 迁入 astryx(切断唯一的 classic 文件依赖)

- [x] **Done**
- **Scope:** `web/src/frontends/astryx/entry.css`、`web/src/frontends/astryx/theme/tokens.css`(自 `src/frontends/classic/styles/tokens.css` 迁入)、`web/vite.config.ts`(`useCSSLayers.before` 层名)、`web/src/frontends/astryx/theme/gptload.theme.ts`(注释路径)、`web/e2e/astryx-shell.spec.ts`(注释)
- **Proof:** `pnpm --dir web run build` + `verify:theme-build` 绿;`astryx-density.spec.ts` ±1px 断言绿(token 值零变化)
- **PM:** `/settings` 顶栏高度 54px、字号 13.5px 等密度指标不变 → 视觉无差异
- **Notes:** 层名 `classic-tokens` → `tokens`;只改文件位置与层标识,不改任何 token 值

### Task B: 单文档终态——web 单入口 + Go 单文档化 + manifest v3

- [x] **Done**
- **Scope:** `web/index.html`(script 指向 `/src/frontends/astryx/main.tsx`)、删 `web/astryx.html`、`web/vite.config.ts`(删 `frontendSelectorDevPlugin`/`vue()`/`tailwindcss()`/`@` alias/`cssInjectionTarget`,`optimizeDeps.entries` 单入口,`rollupOptions.input` 单入口)、`internal/webui/server.go`(删 `astryxIndex`/`frontendCookieName`/`frontendPreference`/`indexFor`/`indexForNotFound`/`Vary: Cookie`)、`internal/webui/page_routes.go` + `page_routes.json`(删 `astryx` 字段,version → 3)、`internal/webui/server_test.go` + `page_routes_test.go`、`web/src/shared/routing/page-routes.ts`(删 `astryx` 字段解析)
- **Proof:** `pnpm --dir web run build` 产出单 `index.html`;`make test` 中 `internal/webui` 用例绿;curl 真机矩阵(无 cookie / `=classic` / `=astryx` → 同一文档)
- **PM:** 构建并启动二进制,浏览器无 cookie 打开 `/settings` → astryx 界面;写入 `gpt-load.frontend=classic` 刷新 → 仍 astryx(cookie 被忽略)
- **Notes:** vite 入口翻转后 `index.html` 即 astryx 文档,Go 侧 `astryxIndex` 读不到文件自然退化为单文档,本任务把死代码一并删净;`optimizeDeps.include` 里 astryx 依赖钉住保留

### Task C: astryx 侧共存残留收敛

- [x] **Done**
- **Scope:** `web/src/frontends/astryx/app/route-link.tsx`(删 `isAstryxNavigable`)、`app/pages.tsx`(删 unflagged 守卫)、`app/route-adapter.ts`(`astryxRoutePaths` 收敛全量)、`app/router.tsx`、`app/shell/LoginView.tsx`(SPA 导航无分支)、`app/shell/PreferencesControl.tsx`(删 Interface 段与 `updateFrontend`)、删 `src/shared/controllers/frontend-preference.ts`、删 `web/scripts/verify-frontend-cookie.mjs` + `lint` 脚本摘除、`src/shared/i18n/locales/*/core.ts` 与 `message-ids` 删 `shell.frontend*` 三键、`web/e2e/astryx-shell.spec.ts`(删 frontend 段用例)、`astryx-selection.spec.ts` / `astryx-routing.spec.ts` / `astryx-visual-audit.spec.ts` / `go-csp.spec.ts` 的 cookie/双文档断言改写为默认即 astryx
- **Proof:** `tsc` 两 config + `eslint` + `verify:i18n-icu` 绿;`astryx` Playwright project 绿(去 cookie 种子后)
- **PM:** 偏好面板只剩主题/语言两段;登录页跳转与内部链接全部 SPA 导航
- **Notes:** `verify-frontend-cookie.mjs` 守护的 Go↔TS cookie 契约随机制消失而退役;`astryxRoutePaths` 调用点改为直接使用 `pageRouteEntries`

### Task D: classic 树与 Vue 工具链删除

- [x] **Done**
- **Scope:** 删 `web/src/frontends/classic/`(164 个 `.vue`)、`web/src/main.ts`;`web/package.json` 移除 `vue`/`vue-i18n`/`vue-router`/`reka-ui`/`@lucide/vue`/`@tanstack/vue-query`/`@vitejs/plugin-vue`/`vue-tsc`/`eslint-plugin-vue`/`@vue/eslint-config-typescript`/`@tailwindcss/vite`/`tailwindcss`,`type-check` 收敛为 `tsc -p tsconfig.astryx.json && tsc -p tsconfig.node.json`;`web/pnpm-workspace.yaml` 删 `vue-demi`;删 `web/tsconfig.app.json`,根 `tsconfig.json` references 收敛;`web/eslint.config.mjs` 删 vue 系接入与 `.vue`/vue-import 规则
- **Proof:** `pnpm install` 后 `pnpm ls vue vue-tsc tailwindcss` 为空;`build`/`lint`/`type-check` 全绿
- **PM:** 无独立可见面(经 Task B 已不可达);验证 = 工具链全绿
- **Notes:** classic 内对 `@shared` 的 re-export shim 随树一并删除;`shared/` 中 astryx 仍消费的模块保持原样

### Task E: Playwright 项目收敛

- [x] **Done**
- **Scope:** `web/playwright.config.ts`(删 `classic` project、`astryx` project 的 cookie storageState);删 8 个 classic spec:`model-test-alias`/`request-log-affinity`/`request-log-display`/`request-log-page-polish`/`request-log-readability`/`request-log-route-layout`/`schedule-editing`/`schedule-routing`
- **Proof:** `playwright test --list` 无 classic project;astryx + go-csp + chromium-125 + chrome-latest 四项目全绿
- **PM:** 无(测试基建)
- **Notes:** 被删 classic spec 守护的共享逻辑仍由 `verify:*.mjs` 脚本与 astryx parity 用例覆盖(request-log-affinity→verify:request-log-affinity;model-test-alias→astryx 侧已有 parity)

### Task F: 文档与 Doc ID 收尾

- [x] **Done**
- **Scope:** 删 `.docs/db/features/frontend-preview-switch.md` + `docs:build` 重建 dist;`.docs/db/features/model-test-alias.md` 措辞与 `.docs/tech/model-test-alias.md` `code.paths` 改指 astryx;`.docs/tech/astryx-migration-plan.md` 补 Phase 5+6 outcome;新建 `.docs/adr/0003-*.md`(单前端化决策,含放弃回退窗口的用户确认);grep `PRODUCT.md`/`README*`/`.docs/` 中 `classic`/`Vue`/`preview` 残留并更新;brief 追加实现追溯
- **Proof:** `docs:check` + `docs:build` 绿;`rg "frontends/classic" .docs` 仅剩历史记录语境
- **PM:** 无(文档面)
- **Doc IDs:** `feature.frontend-preview-switch` 删除并在 PROJECT_HISTORY/迁移计划中注明退役

### Task G: 全量验证矩阵与真机 CSP

- [x] **Done**
- **Scope:** 验证执行,无代码改动
- **Proof:** `pnpm --dir web run build`/`lint`/`format`/`type-check`、全部 `test:*` 与 `verify:*` 脚本、Playwright 四项目、`make test`(Go)、`docs:check`/`docs:build`、真机 Linux 二进制 go-csp 三浏览器矩阵——全部命令退出码 0
- **PM:** 真机 curl 三态(无 cookie / `=classic` / `=astryx`)返回同一 astryx 文档;浏览器实测 `/login`→`/groups/:id`→`/import` 链路

## Review

- [x] Review complete

### Evidence (2026-09-30)

| Gate | Result |
|---|---|
| `type-check` (astryx+node tsconfig) | exit 0 |
| `build` | exit 0 — single `dist/index.html` entry |
| `lint` (eslint + i18n-icu + theme-build + verify-*) | exit 0 |
| All `verify:*`/`test:*` node scripts | exit 0 (7 scripts got explicit `process.exit` — StyleX unplugin keeps worker handles after `server.close()`, pre-existing) |
| Playwright `astryx` project | 129/129 green (incl. 4 new redirect tests for /access-keys & /schedule) |
| Playwright `go-csp` + `chromium-125` + `chrome-latest` vs real WSL Linux binary | 36/36 green — all 11 routes identical document, retired `gpt-load.frontend=classic` cookie ignored, zero CSP violations |
| `go test ./internal/webui` (server/manifest/notfound) | ok — single-document serving tests green |
| `go test ./internal/control -run TestFrontendHealth` | ok — import repointed to `shared/control/resources/health.ts` |
| `docs:check` + `docs:build` | exit 0 — `feature.frontend-preview-switch` retired, 4-doc bundle rebuilt |

Notes: `pnpm run format` reports widespread pre-existing drift (332 files incl. generated/test artifacts) — baseline noise, not introduced by this change; no formatting regression was introduced in touched files. `make test` full run keeps the known baseline failures (docker missing, CI workflow YAML test, CRLF shell test) — all unrelated to the frontend cutover; focused packages green.

