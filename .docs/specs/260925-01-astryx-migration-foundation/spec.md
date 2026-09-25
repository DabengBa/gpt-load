# Spec: Astryx 迁移地基(Phase 0 + Phase 1)

## 意图与核心流程

一句话:在 classic Vue 前端内完成框架无关共享层抽取(Phase 0),然后搭起 React 19 + Astryx + StyleX 新前端脚手架,与 classic 在同一批 URL 上共存,跑完 7 项 spike 并产出 go/no-go 门槛证据(Phase 1)。

- 触发条件:用户指令"根据文档,继续完成开发工作";技术细节的唯一权威来源是 `.docs/tech/astryx-migration-plan.md`(下称"方案文档"),本 spec 只做范围与验收契约,不复述其实现细节。
- 主路径:Phase 0 的每一步行为中立、独立可交付(全部现有验证保持绿色)→ Phase 1 建立 `astryx.html` 第二入口、Go 双 index 选择、开发期 selector、Playwright 双项目矩阵 → shell 对等(导航/鉴权/登录/404/偏好/路由播报)→ 7 项 spike → 按 7 条门槛度量并记录证据 → 用户在 gate 处决定 go/no-go。
- 成功状态:门槛证据齐备;`dev` 上 classic 仍是默认前端且行为不变;新前端通过 cookie 选择性可达。

## 范围 / 不做范围

本次要做:

1. **Phase 0(仅 classic,不引入任何 React 依赖)**:按方案文档「Phase 0 Shared-Layer Extraction」表逐行执行——control API 类型与协议、query keys 与 invalidation、资源 fetcher/DTO(去掉 `vue`/`@tanstack/vue-query` 依赖,改为纯参数对象)、纯 lib 与 55 个无框架 `features/**/*.ts`、6 个控制器改为 `subscribe`/`getSnapshot` 外部 store、`client-context.ts` 移回 classic、路由规则(`pagePathMatches`/`safeRedirect`/`decodedPathSegments`/query 规整,连同测试)、locale catalogs 移到 `shared/i18n`、12 条 ICU 不兼容消息改写为 `{example}`/`{port}` 插值并新增 `verify:i18n-icu`、verify 脚本改读 `shared/`、149 处 e2e BEM class 定位器改写为语义定位、删除 8 个零引用组件、4 份 tech 文档 `code.paths` 改指 `shared/`。
2. **Phase 1(脚手架 + 共存 + spike + gate)**:按方案文档「Coexistence Architecture」「Target Architecture」「Phased Delivery → Phase 1」执行——`astryx.html` 入口与 `main.tsx` providers 链(Query/Router/react-intl/Astryx Theme + InternationalizationProvider/toast)、`gptload.theme.ts` 静态主题构建并提交产物、`entry.css` 层序;Go 侧 `page_routes.json` v2(`astryx` 标志位)+ 双 index 选择 + `gpt-load.frontend` cookie;Vite dev selector 插件;Playwright `classic`/`astryx`/Go-CSP 三项目矩阵;shell 对等(AppShell、AuthGate、login、not-found、偏好面板含前端切换控件、路由播报);spike (a)-(g) 全做;产出 gate 度量记录。
3. **执行环境前置**:本机 Node 22.22 / pnpm 10.28 不满足 `web/package.json` engines(≥24.11 / ≥12.1)。实现开始前在仓库外配置独立工具链(portable Node 24 + pnpm 12),不改 `engines`、不改 CI、不改用户全局环境。

不做(明确推迟):

- Phase 2-6:任何业务域页面迁移、默认前端切换、classic 及其工具链(Tailwind/plugin-vue/vue-tsc/Vue ESLint)删除。gate 通过、用户批准、`.docs/adr` 记录决定后另行立项。
- 产品行为、后端 API 契约、页面语义的任何变化;`.docs/db` 语义文档内容不改。
- 超出 Astryx 采用所必须的视觉重设计;不加载任何 webfont(系统字体栈保持)。
- 迁移 ADR:本 slice 结束于 gate 度量,ADR 是 gate 之后的工作。

## 边界规则 / 验收

### Phase 0 出口(逐条可验证)

- `pnpm --dir web run build`、`lint`、`format`、全部 `verify:*`、`test:*`、现有 e2e 全绿,行为不变(行为中立承诺:每步单独可回滚)。
- `web/src/shared/**` 内没有任何文件 import `vue`、`vue-router`、`vue-i18n`、`@tanstack/vue-query`、`reka-ui`;用 ESLint `no-restricted-imports` 强制。
- 12 条不兼容消息改写完成;`verify:i18n-icu` 解析三种语言全部消息并校验 key 对齐,并入 `lint`。
- e2e 不再含 BEM class 定位器(`locator('.x__y')` 形态清零,语义不足处才用 `data-testid`)。

### Phase 1 go/no-go 门槛(方案文档七条,全部必须成立)

1. Go 二进制托管的构建在 Chromium 125 与最新 Chrome 上渲染 shell 与全部 spike,CSP 零违规。
2. `theme-bootstrap.js` 不变前提下,light/dark/system 三种模式刷新无主题闪烁。
3. spike 表格承载 1,000 行,且改写选择器后的现有 collection e2e 流程通过。
4. 新 shell 首屏 JS+CSS(gzip)对 classic shell 实测并记录。
5. 所需行为不超过 3 个 swizzle 组件;每个 swizzle 记录原因与取自的 Astryx 版本。
6. 「Theme and tokens」表中每项密度指标与 classic 实测差 ≤±1px,且只能通过 theme 输入、`SizeContext`、`Table` density、theme 组件覆盖达成,禁止 call-site `xstyle` 调密度。
7. `verify:i18n-icu` 通过;shell 在 zh-CN/en-US/ja-JP 渲染无 `MISSING_TRANSLATION`/`FORMAT_ERROR`。

### 共存期边界

- `gpt-load.frontend` cookie(`Path=/; SameSite=Strict`)只能在两个内嵌静态文档间选择;不得参与鉴权、API 路由、文件路径。任何非 `classic|astryx` 值按 classic 处理。
- 无 `astryx.html` 的构建行为与今天完全一致(`astryxIndex == nil` 路径)。
- feature-freeze:域未迁移前新功能只进 classic;已迁移后只进新前端;域仍可达期间 bug fix 双端落地。
- 浏览器状态键(`gpt-load.auth-key`/`.locale`/`.theme`/`.import-reauth-draft`)冻结;新增共享键须加双端导入同一常量的契约测试。
- gate 失败处理:停止,记录原因,删除脚手架;Phase 0 成果保留。

## 架构 / 约束

- 双 HTML 入口同 URL:Go `internal/webui` 按 manifest `astryx` flag + cookie 选 index;SPA fallback 同规则;`Cache-Control`/CSP/`nosniff`/`DENY` 响应头两文档一致。无 `/v2` 前缀。
- `vite.config.ts`:`rolldownOptions.input` 双 input;`target: 'chrome125'` + `browserslist`(`chrome >= 125`,`edge >= 125`);插件顺序 stylex → vue → react(`include` 限定 `src/frontends/astryx/**`)→ babel(React Compiler preset,rolldown filter 排除 classic 与 shared)→ tailwindcss → dev selector。
- 别名:`@`→classic(不变),`@app`→astryx,`@shared`→shared。
- CSS:新入口零 unlayered 规则;层序 `reset → astryx-base → astryx-theme → app.*`;禁引 `classic/styles/*`。
- 主题:`defineTheme` extends `neutral`;`astryx theme build` 产物提交;`lint` 加 rebuild-diff 检查;`data-theme` 唯一所有者是 root `Theme`(与 `theme-bootstrap.js` 契约一致),不得出现第二个写 `data-theme` 的控制器;字体固定为 classic 系统栈。
- i18n:`IntlProvider` + 命令式 `createIntl` 实例(路由标题/presenter 用),locale 单一 store 驱动两者 + `<html lang>` + `Accept-Language`;catalog 保持嵌套 TS,loader 拍平为点路径 key;每个命名空间先合 en-US 再叠当前语言(复现 `fallbackLocale`);`onError` 在 dev/test 抛错、生产记日志;Astryx 组件文案用 `InternationalizationProvider` + 本地 zh-CN/ja-JP catalog + `verify:astryx-i18n` key 对齐检查。
- 路由:TanStack Router code-based;由 `page_routes.json` 适配生成(`:id`→`$id`);`validateSearch` 承载 typed search;`trailingSlash: 'preserve'` + 尾斜杠路径显式 404;`caseSensitive: true`;滚动行为镜像 classic `scrollBehavior`;`beforeLoad` 承载 `requiresAuth`/`adminOnly`/命名空间懒加载;`safeRedirect` 先搬测试再启用。
- React Compiler:只编译 `frontends/astryx/**`;默认不手写 `useMemo`/`useCallback`/`memo`;`eslint-plugin-react-hooks` 编译器诊断按 error;`"use no memo"` 必须附上游 issue 链接。
- 仓库契约:`build` 脚本字面量保持 `pnpm run type-check && vite build`(`type-check` 内部扩为 vue-tsc + 两个 tsc);`tsconfig.app.json` 排除 `src/frontends/astryx/**`;ESLint flat config 按目录 scope(Vue 规则只罩 classic);CI 命令集与顺序不破坏(改动则同步 `workflow_test.go`);依赖全部精确版本 + 7 天规则,`overrides` 两处同步重推导;`internal/webui/dist` 产物不入库。
- Go 侧:`page_routes.go` 与 `app/page-routes.ts` 两个 strict parser 同步升 v2;`page_routes_test.go`/`server_test.go`/`workflow_test.go` 增加对应 case。
- 依赖清单(按方案文档「Stack mapping」固定):`react`/`react-dom` 19.3.0、`@astryxdesign/core` 0.6.2(2026-09-30 后可升 0.6.3)、`@astryxdesign/theme-neutral` 同版本、`@stylexjs/stylex`/`@stylexjs/unplugin` 0.19.1、`@tanstack/react-query` 5.103.x、`@tanstack/react-router` 1.170.x、`react-intl` 12.1.x、`lucide-react` 1.48.x、`@vitejs/plugin-react` 6.x、`@rolldown/plugin-babel` + `@babel/core` + `babel-plugin-react-compiler` 1.0.x;dev:`@astryxdesign/cli`、`@formatjs/icu-messageformat-parser`、`@stylexjs/eslint-plugin` 0.19.1、`eslint-plugin-react-hooks` 7.x。添加一律 `pnpm --dir web add` 精确版本。

## 数据 / 集成

- `page_routes.json`:`version: 1 → 2`,路由条目新增可选 `"astryx": true`;双端 strict parser 同改;Phase 1 结束时仅 shell 路由(`/`、`login`、not-found 路径及 spike 覆盖的 `groups`/`logs`/`group-detail` 中实际迁移的项)标 true。
- 浏览器存储:上节冻结键不变;新增 `gpt-load.frontend` cookie 契约测试。
- 共享层目录:`web/src/shared/`(http、control、domain、i18n、controllers、routing),无框架依赖;classic 与新前端各自 thin adapter。
- Astryx 本地 catalog:`astryx/locales/zh-CN.json`、`ja-JP.json`,形状对齐 `@astryxdesign/core/locales/en.json`。
- 向后兼容:老二进制 + 新 `page_routes.json` 的组合不会出现(parser 随二进制一起发布);新代码读 v1 manifest 时按无 flag 处理。classic 在本 slice 全程保持默认可达、功能完整。

## 验证

执行前置:本地先备齐满足 engines 的工具链(Node ≥24.11、pnpm ≥12.1),否则下列命令无法运行。

| 门槛 | 命令 / 检查 | 时机 |
| --- | --- | --- |
| 类型/构建 | `pnpm --dir web run build`(内含 type-check) | 每次改动 |
| Lint/格式 | `pnpm --dir web run lint`、`pnpm --dir web run format` | 每次改动 |
| 共享层逻辑 | `verify:group-collection`、`verify:health-projection`、`verify:request-log-affinity` 及其余 `verify-*.mjs`;`test:connection-json`、`test:channel-contract` | 触碰 `shared` 的改动 |
| 共享层依赖禁则 | `no-restricted-imports` 命中数 = 0 | 每次改动(ESLint) |
| i18n | `verify:i18n-icu`(并入 lint)、`verify:astryx-i18n` | 触碰 catalog / 每次 Astryx 升级 |
| E2E(dev) | 全量 Playwright,`classic` 与 `astryx` 双项目 | 每个域/Phase 1 各 spike |
| E2E(Go+CSP) | `make build` 起二进制,CSP 项目监听 `securitypolicyviolation` | Phase 1 收尾、gate 度量 |
| 最低浏览器 | Playwright Chromium 125 项目(`executablePath` 指向 chrome@125) | gate 度量 |
| Go 契约 | `make test`(含 `internal/webui` 三测试) | 触碰 `internal/webui`、`web/package.json`、CI 的改动 |
| 主题产物 | 重跑 `astryx theme build` 并 diff | 触碰主题的改动 |
| 文档 | `pnpm --dir web run docs:check`、`docs:build` | 每次改 `code.paths` |
| Gate 证据 | 包体积记录、密度 ±1px 表、CSP 报告、spike 结论,全部写入 plan.md 执行记录 | Phase 1 结束 |

完成判据:Phase 0 出口全绿 + Phase 1 七条门槛度量齐备并写入执行记录;go/no-go 由用户依据证据决定,不属于本 spec 的交付义务。

## Doc ID 契约

- 新增计划 ID `feature.frontend-preview-switch`:归属 `.docs/db/features/frontend-preview-switch.md`,绑定点是 classic 偏好面板中的前端切换控件与新 shell 的"返回 classic"控件(`gpt-load.frontend` cookie 写入处)。该 ID 生命周期与共存期一致,Phase 6 删除时同步处置文档。实现期先建最小 stub,正文随 Phase 1 控件落地补齐。
- 无新增 `page.*`/`term.*` ID:URL 不变,不产生新页面语义。
- 已确认 `.docs/db` 语义文档无 code 绑定(方案文档 Corrections 表),本 slice 不会造成既有 Doc ID 漂移;义务是更新 4 份 tech 文档(`model-test-alias.md`、`reasoning-policy.md`、`usage-accounting.md`、`billing-failure-attention/plan.md`)中指向 classic 的 `code.paths`,随 Phase 0 搬运逐处改指 `shared/`。
- 验证:`docs:check` + `docs:build` 必须全绿。

## 参考资料

- `.docs/tech/astryx-migration-plan.md` — 全部技术细节的唯一权威来源(Baseline Inventory、Target Architecture、Coexistence Architecture、Phase 0 表、翻译规则、组件映射、Router、Internationalization、Toolchain、Phased Delivery、Verification、Risks)。
- `.docs/tech/briefs/astryx-frontend-migration.md` — 用户意图与确认记录。
- `.docs/adr/0001-collection-read-model-scale.md` — 集合服务端分页约束(Table 必须用受控 `*State`)。
- `.docs/db/features/{monitor-navigation-shortcuts,model-test-alias,dispatch-reasoning-policy,usage-timing-metrics}.md` — 后续域迁移阶段的验收清单来源(本 slice 仅 spike 涉及)。
- `internal/webui/{server.go,http_routes.go,page_routes.go,page_routes.json}`、`web/package.json`、`web/vite.config.ts`、`web/src/frontends/classic/styles/tokens.css`、`web/public/theme-bootstrap.js`、`web/e2e/`、`web/scripts/` — 方案文档实测所引的仓库路径。
- npm registry 发布日期与 peer 依赖、Astryx/StyleX/Vite/TanStack/react-intl 官方文档:引用事实已在方案文档内逐条标注日期(2026-09-25)。
- 推断项(Inference):spec 切片边界(Phase 0+1 作为首个交付单元)是对"继续完成开发工作"的解释——方案文档自身的 go/no-go gate 即天然决策点;若用户意图是一次性批准全量迁移,需扩大 scope 后重新批准。
