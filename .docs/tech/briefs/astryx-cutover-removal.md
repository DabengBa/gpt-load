---
created: 2026-09-30
source: 用户在 Phase 4 wrap-up 完成后的直接指示
confirmed: 2026-09-30
last_updated: 2026-09-30
---
# Brief: Astryx Cutover & Classic Removal

## User Original Request

> 请继续完成phase5和phase6

## Background & Motivation

Astryx 迁移 Phase 0–4 已交付(2026-09-30 wrap-up `e413188e`):全部 11 条
`page_routes.json` 路由已置 `astryx: true`,astryx e2e 134/134、codec 单测
15/15、真机 CSP 矩阵 36/36。迁移计划(`astryx-migration-plan.md`)剩余两个
阶段:

- **Phase 5 Cutover**:默认文档翻转为 astryx,classic 通过
  `gpt-load.frontend=classic` cookie 保留一个发布周期作为回退通道。
- **Phase 6 Deletion**:删除 `frontends/classic`、Vue/Tailwind 工具链、
  Go 侧文档选择代码、manifest flag、dev selector 插件与 classic
  Playwright 项目。

## Intent Domains

### 合并交付(用户确认)

- 用户期望:Phase 5 与 Phase 6 合并为一次交付——Astryx 直接成为唯一前端,
  classic 与全部共存基建(cookie 选择机制、双 html 入口、双 tsconfig、双
  Playwright 项目)在同一交付中移除。**放弃计划中的一个发布周期回退窗口**,
  回退手段退化为 git/版本回滚。
- 变更历史:
  - 2026-09-30 用户指示"请继续完成phase5和phase6";在切流窗口选择题上选择
    "合并成一个 spec";在偏好控件命名题上选择"Phase 6 时再说"——因合并交付后
    控件随 classic 一并删除,命名问题自然消解。
- 实现追溯:
  - 规格目录 `.docs/specs/260930-01-astryx-cutover-removal/`(归档时删除过程
    文件)。
  - 代码路径:`internal/webui/`(单文档化、manifest v3)、`web/vite.config.ts`
    (单入口)、`web/index.html`、`web/src/frontends/astryx/`、`web/src/shared/`、
    `web/e2e/`、`web/package.json`、`web/pnpm-workspace.yaml`、
    `web/eslint.config.mjs`、`.docs/db/features/frontend-preview-switch.md`。
  - 上游规格:`.docs/tech/astryx-migration-plan.md` Phase 5/6 小节。
  - 交付记录(2026-09-30):
    - `f2b41446` Task A:density tokens 迁入 `astryx/theme/tokens.css`(`tokens`
      层),密度值零变化,`astryx-density` ±1px 断言绿。
    - `81f45345` Task B–E:Go 单文档化(去 cookie 选择与 `Vary: Cookie`)、
      manifest v3 删 `astryx` 字段、单 `index.html` 入口、`astryx.html`/
      dev selector/`frontend-preference.ts`/`verify-frontend-cookie.mjs`/
      Interface 偏好段/`isAstryxNavigable` 全删;classic 树与 Vue/Tailwind
      工具链删除(`typescript-eslint` 接替解析);classic Playwright 项目与
      8 个 classic spec 移除,`e2e` 脚本统一入口。
    - 验证:类型/构建/lint 绿;`internal/webui` 单文档契约测试绿(含残留
      cookie 忽略用例);`internal/control` health 契约测试改指
      `shared/control/resources/health.ts` 后绿;verify:* 与 test:* 全绿;
      astryx Playwright 129/129(两处既有 spec 断言修正:cache token
      details 按钮名冲突 + adminOnly 重定向等待窗)。
    - 附注:7 个 `verify-*.mjs` 补显式 `process.exit`——`@stylexjs/unplugin`
      在 `server.close()` 后残留常驻 handle,曾把门脚本挂死。
  - ADR:`.docs/adr/0003-astryx-cutover-without-fallback-window.md`。

### 约束(继承迁移全程约束)

- 用户可见 URL、登录态、locale、主题不变;唯一可见差异是默认渲染 astryx 且
  无任何途径回到 classic。
- CSP 不变;Go 内嵌部署契约不变;`make build`/`make test` 命令契约不变。
- 浏览器基线 Chrome 125+ 不变。
- 残留的 `gpt-load.frontend` cookie 被忽略(机制移除,不解释旧值)。
