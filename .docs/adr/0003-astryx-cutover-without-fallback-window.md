---
description: Astryx 切换决策——合并 Phase 5/6 一次交付，放弃一个发布周期的 cookie 回退窗口
kind: adr
topic: astryx-migration
status: accepted
date: 2026-09-30
---

# ADR-0003：合并切换与删除一次交付，cookie 回退窗口为零

## 背景

迁移计划（`.docs/tech/astryx-migration-plan.md`）原设计将收官拆为两个
阶段、相隔一个发布周期：

- Phase 5：默认文档切到 Astryx，`gpt-load.frontend=classic` cookie
  仍可显式回退；
- Phase 6：一个发布周期后删除 classic 与全部共存基建。

cookie 回退窗口的用途是给运维一条「不改版本、秒级回退」的逃生通道，
覆盖切换后发现致命回归的场景。

用户在 Phase 4 收官后明确指示"继续完成 phase5 和 phase6"，并在澄清
环节选择了**合并立项、一次交付**——即不存在只切换不删除的中间发布。

## 决策

- Phase 5 与 Phase 6 合并为单个 spec
  （`260930-01-astryx-cutover-removal`），直接交付单文档终态：
  Go 服务对每个页面路由与未匹配路径返回同一份 Astryx `index.html`，
  不读取 `gpt-load.frontend` cookie，不下发 `Vary: Cookie`。
- 选择机制整体**删除**而非翻默认：cookie 契约模块、校验脚本、偏好面板
  Interface 段、`isAstryxNavigable`、manifest `astryx` 字段（v3 起
  变为非法字段）、dev selector 中间件、classic Playwright 项目全部
  移除。
- **回退手段退化为版本回滚**：切换后若发现致命回归，恢复方式是回滚
  部署版本，而非运维侧写 cookie。

## 取舍

| 方案 | 回退成本 | 共存成本 | 结论 |
|---|---|---|---|
| 分两阶段 + cookie 窗口（原设计） | 秒级（写 cookie） | 双前端、选择基建再维护一个周期 | 否决（用户要求合并） |
| 合并交付（选定） | 分钟级（版本回滚） | 当期即清零 | **选定** |
| 仅翻转默认、保留机制 | 秒级 | 共存代码永久残留 | 否决（删除面被推迟成遗留债） |

## 后果

- 正面：classic 树（2.4MB / 164 个 `.vue`）、Vue/Tailwind 工具链、
  12 个依赖、双入口构建、文档选择中间件当期全部清零，迁移无尾巴。
- 代价：切换即刻生效、无灰度。缓解：manifest v2→v3 之前 11 条路由已
  全部经 astryx e2e + 真机 CSP 矩阵验证；路由/认证/本地化语义由
  Phase 0–4 parity 测试钉死。
- 不可逆性：发布后再引入 classic 需从版本回滚或重建代码——属于显式
  接受的 trade-off。
