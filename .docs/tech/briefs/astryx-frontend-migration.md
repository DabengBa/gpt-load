---
created: 2026-09-25
source: 用户在迁移方案评审会话中的原始指令与逐项选型确认
confirmed: 2026-09-25
last_updated: 2026-09-25
---
# Brief: Astryx Frontend Migration

## User Original Request

> 阅读后，结合项目实际情况，结合网络迁移经验，更新并详细化这份迁移文档。

随后：

> 根据文档，继续完成开发工作。

## Background & Motivation

管理界面当前是 Vue 3 + Reka UI + scoped BEM CSS(约 17.6k 行手写样式)。用户要求把前端迁到 React 19 + Astryx(`@astryxdesign/core`)+ StyleX,同时保持现有行为、路由、鉴权、多语言、Go 内嵌部署与紧凑视觉密度不变。迁移必须先产出可行性证据(go/no-go gate),再决定是否继续全量迁移。

用户在 2026-09-25 逐项确认了选型(均记录于 `.docs/tech/astryx-migration-plan.md` 的 Confirmed choices):

- 浏览器基线:Chromium/Chrome 125+;Safari/Firefox 尽力兼容、不卡发布
- 视觉密度:贴近 classic 现有紧凑尺寸
- 路由:TanStack Router(code-based)
- React Compiler:开启(稳定 Babel 集成)
- 应用 i18n:react-intl
- 主题基底:`neutral` 加 GPT-Load 覆盖

## Intent Domains

### 迁移执行

- 用户期望:按 `.docs/tech/astryx-migration-plan.md` 推进开发;第一阶段交付 Phase 0(框架无关共享层抽取)与 Phase 1(React/Astryx 脚手架、同 URL 共存、shell 对等、7 项 spike、go/no-go 门槛证据),在 gate 处由用户决定是否进入 Phase 2+ 域迁移。
- 当前状态:**Phase 0 + Phase 1 + Phase 2(settings/models/home)+ Phase 3(access-keys/monitor/logs/schedule)已交付(2026-09-27)**;Phase 4(groups + import)按方案继续。
- 变更历史:
  - 2026-09-25 方案文档扩写完成(149 → ~900 行),记录实测基线与全部技术选型。
  - 2026-09-25 用户确认五项开放选型与 neutral 主题基底,文档内不再有待决项。
  - 2026-09-25 用户指示"根据文档，继续完成开发工作",brainstorming 判定首个可交付切片为 Phase 0 + Phase 1(到 go/no-go gate 为止)。
  - 2026-09-26 Phase 1 全部 spike 完成,七条门槛证据齐备,用户审阅后回复 "go"。
  - 2026-09-26 final-review 发现 17 项全部处置(路由移交、sparse validateSearch、codec parity、resetScroll、`Vary: Cookie` 等),评审关闭。
  - 2026-09-27 Phase 2 三域交付:settings(e9a8b0bf)、models 含 model-prices 嵌入(7f6e2b1c)、home(805b0f95);切片终审收敛死字段并关闭(1b8d0a18),方案文档回写 Phase 2 outcome(3dffa83b)。
  - 2026-09-27 Phase 3 交付:access-keys(3a019b90)、monitor 宿主 + health/usage(781ef55b)、inspector(ef97aaa7)、logs 全量 parity(ac540c16)、schedule 调度中心(2fdd51c6);终审补 monitor-route codec 单测 12 条,真机 CSP 30/30 三浏览器矩阵全绿;方案文档回写 Phase 3 outcome,过程文件删除,wrap-up 提交 8cbd2125。
  - 2026-09-27 Phase 4 立项摸底:group-detail(9,219 行)+ import(5,670 行)为最后两条 classic 路由;发现 astryx `importRecovery` 误接 localStorage(classic 用 sessionStorage)且 `onUnauthorized` 未接 `captureForUnauthorized`——列为 flag 前置修复项。
  - 2026-09-30 Phase 4 交付:codec 抽取 + recovery 修复(641f3950);detail 宿主 + settings/models/credentials 三 tab + import 双模式 + staging 层(057a7833)——rebase fork/dev 后同步上游 f78071f8 可读性语义(eefa9bc3、cff1b699);终审修掉 flag 翻转遗留的四处 document 导航;astryx e2e 134/134 + classic readability 7/7 + codec 15/15;manifest 终态 11/11 astryx flag,仅剩 Phase 5 默认切换与 Phase 6 删除。
- 实现追溯:
  - 规格目录 `.docs/specs/260925-01-astryx-migration-foundation/`(归档时已删过程文件;gate 证据与评审记录已并入 `.docs/tech/astryx-migration-plan.md` Phase 1 小节)。
  - Phase 2 规格目录 `.docs/specs/260926-01-astryx-phase2-domains/`(归档时已删过程文件;域分解、三评审面记录与真机证据并入 `.docs/tech/astryx-migration-plan.md` Phase 2 outcome 与本条追溯)。
  - Phase 3 规格目录 `.docs/specs/260927-01-astryx-phase3-domains/`(归档时已删过程文件;同上并入 Phase 3 outcome)。
  - 分支 `docs/astryx-migration-plan`;关键提交:`55add5be`(groups 集合)、`ef6b2779`(log detail layer + ADR-0002)、`d0362813`(日志时间范围)、`c28a1c2c`(门槛证据)、`d1570ac6`(方案文档回写)、`f1e722cf`(final-review 修正批)、`e9a8b0bf`(settings 域)、`7f6e2b1c`(models 域)、`805b0f95`(home 域)、`1b8d0a18`(Phase 2 切片终审)。
  - 代码路径:`web/src/frontends/astryx/`、`web/src/shared/`(routing/lib/control/controllers/domain)、`internal/webui/`(manifest v2、文档选择、`cmd/webui` CSP harness)、`web/e2e/astryx-*.spec.ts`。
  - ADR:`.docs/adr/0001-collection-read-model-scale.md`(集合读模型)、`0002-astryx-detail-layer-primitive.md`(detail overlay 原语)。
  - Phase 2 确立的约束性约定(记录在方案文档 Phase 2 outcome,Phases 3-4 适用):mutable controller 一律 memoized snapshot + `useSyncExternalStore`;Vue watch 语义转 render-phase adjustment + post-commit effect;敏感操作状态机下沉 shared controller(`gateway-actions`)。

### 约束(贯穿全部阶段)

- 用户可见 URL、登录态、locale、主题、`gpt-load.import-reauth-draft` 等浏览器状态键在共存期冻结;新增 `gpt-load.frontend` cookie 仅用于选择静态文档,不得影响鉴权、API 路由或文件路径。
- CSP 不变:`style-src-elem 'self'` 排除运行时样式注入,主题必须 `astryx theme build` 静态产出。
- Go 内嵌部署与 CI 命令契约(`workflow_test.go` 钉住的命令集)不得破坏。
- 依赖固定精确版本且遵守 7 天规则。

## Non-Goals

- 已完成切片不做 Phase 3-6(operational/highest-coupling 域、默认前端切换、classic 删除);这些阶段按方案文档逐阶段立项。
- 不改产品行为、后端 API 契约、页面语义;`.docs/db` 语义文档继续作为验收清单。
- 不做超出 Astryx 采用本身的视觉重设计。
- 不写迁移 ADR(go/no-go gate 之后才有难以逆转的决定可记录)。
