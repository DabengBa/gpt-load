# L5 Three-Space Architecture & Exclusion Matrix

面向 **L5 暗黑工厂（Zero-Human-In-The-Loop）**：无会话历史、短生命周期、高频冷启动（Fresh-Context Spawn）的并发 Agent 与编排器。完全剔除面向人类阅读的散文与教程。

## 1. 四条铁律

1. **工具绝对无关**：100% 采用 AAIF 通用开放标准（`AGENTS.md`、`SKILL.md`），不依赖绑定单一工具或特定编排守护进程的专有方言。
2. **代码自描述排除（Zero Redundancy）**：凡源码、类型系统、Schema、静态分析能表达的内容，严禁写入任何文档。
3. **临时过程与永久沉淀物理隔离**：开发过程文档（临时 Spec、执行草稿、排错轨迹）严格存放在独立 Git Worktree 的 `.tmp/`（由 `.gitignore` 强制忽略，永不入库，PR 合并 / 销毁 Worktree 后自动消失）；仅通过 PR 合并前蒸馏闭环把无法由代码反推的决策与不变量提取至永久知识层。
4. **纯确定性隔离，不做指令安全防御冗余**：仅在业务代码与独立验收系统之间保留物理盲区（防止 Builder Agent 修改测试或过拟合），不增设针对 Markdown 的提示词注入审计流。

## 2. 代码自描述准入/排除边界矩阵

| 信息维度 | 严禁写入文档（一律删除） | 机器替代真值源 |
| :--- | :--- | :--- |
| 目录结构与文件列表 | 手写目录树、模块清单 | `fd` / `rg` / LSP 符号检索 |
| 代码规范与风格 | 命名、缩进、函数长度约定 | Linter + AST 自定义可自动修复报错规则 |
| 接口与数据模型 | API 列表、字段表、表结构说明 | OpenAPI / Protobuf / GraphQL / SQL Migration / 强类型定义 |
| UI 样式与组件契约 | 色号表、间距规范、Props 说明 | Design Tokens + 组件契约清单 + 静态检查规则 |
| 依赖与环境清单 | 运行时/库版本文字描述 | 包管理器清单（`package.json` / `Cargo.toml` 等）+ Lockfile |
| 实时开发进度 | `TODO.md`、`CHANGELOG.md`、任务复选框 | Dolt 图状态库（Beads）+ Git 历史 |

## 3. 全局物理拓扑：三空间分离架构

```text
[空间 A: 主代码仓库 (Production Repo)] —— 永久版本化机器记忆 (Builder & Auditor 可读)
├── .gitignore                       # 强制包含 `.tmp/` 与 `**/.tmp/`，阻断过程文件入库
├── AGENTS.md                        # 全局路由与执行基元 (硬上限 <= 120行 / 1500 tokens)
├── .agents/
│   └── skills/                      # 程序性记忆 (AAIF SKILL.md 标准，按需懒加载)
│       ├── l5-doc-governance/       # 文档体系创建、预算检查、蒸馏打包与熵减治理 Skill
│       └── <domain-skill>/SKILL.md  # 项目按需沉淀的领域多步操作流 Skill
├── .knowledge/                      # 永久语义知识层 (仅存代码无法自描述之沉淀，按路径条件加载)
│   ├── decisions/                   # 结构化 ADR (因果权衡、被否决方案与约束)
│   ├── invariants/                  # 非类型化硬不变量 (EARS + RFC 2119 断言)
│   └── quirks/                      # 外部依赖 / 运行环境未文档化暗坑
├── .beads/                          # (或独立挂载) Dolt 版本化 SQL 图状态机
└── packages/<subpkg>/
    └── AGENTS.md                    # 子包局部覆盖指令 (仅覆写局部构建/验证命令)

--------------------------------------------------------------------------------------

[空间 B: Worktree 瞬时过程空间 + 图状态机] —— 过程文档随 PR 湮灭，状态进图库
├── <worktree>/.tmp/                 # 未纳入 Git！存放临时 Specs、拆解计划、调试轨迹、蒸馏载荷
└── Dolt Graph Plane (Beads)         # 跨并发 Agent 的 DAG 依赖边 (blocks / discovered-from / 租约)

--------------------------------------------------------------------------------------

[空间 C: Clean-Room 独立验证仓 (Holdout & DTU Repo)] —— Builder Agent 不可见 ("The Wall")
├── scenarios/*.yaml                 # 库外留存高阶行为验收场景 (多轮通过率门禁 + 单行症状输出)
└── digital-twins/                   # 外部三方服务高保真行为克隆契约
```

- **Space A (Production Repo)**: `AGENTS.md` (root + optional sub-package overrides) is the minimal boot contract & routing index; `.agents/skills/*/SKILL.md` lazy-loaded procedural SOPs per `agentskills.io`; `.knowledge/{decisions,invariants,quirks}/*.md` holds permanent semantic knowledge strictly invisible in AST/types.
- **Space B**: `<worktree>/.tmp/` untracked ephemeral development specs, task breakdowns, scratchpads, debug logs — MUST NEVER be committed or merged into main. Execution DAG (e.g. Dolt-backed Beads) carries concurrent task dependency graph (`blocks`, `discovered-from`, leases). Never track tasks in Markdown checklists.
- **Space C**: physically unreadable to Builder Agents ("The Wall" prevents reward hacking); executed exclusively by independent Evaluator Agents returning single-line symptom diagnostics on failure. The wall is a separate repository with no builder visibility; no script enforces it.

## 4. 核心文档层与预算

1. **`AGENTS.md`（L0 常驻运行契约）**：任意 Agent 冷启动后唯一全量加载的引导入口。只放构建 / Lint / 测试 / 治理命令、`.knowledge/` 路由索引、`.tmp/` 隔离铁律和 Fresh-Context 审计规则。根文件 `<= 120` 行且 `<= 1,500` 估算 Tokens；Monorepo 子包仅在需要覆写构建或验证命令时放置局部 `AGENTS.md`。
2. **`.agents/skills/*/SKILL.md`（程序性记忆 — 渐进式披露）**：常态仅暴露 `name` + `description`（L1）供路由；命中触发条件后加载 `SKILL.md` 正文（L2，`<= 500` 行）；执行步骤直接调用 `scripts/` 确定性脚本（仅 stdout / 退出码进入上下文），复杂规范按需读取 `references/`。调度时机直接接入仓库标准 CI 或工程脚本（如 `Makefile`）。
3. **`.knowledge/`（永久语义记忆层）**：`decisions/ADR-XXXX.md` 元数据含 `id`、`status`、`supersedes`、`affects_symbols`（精确到文件路径与符号名），正文 `Context` / `Decision` / `Rejected` / `Constraint`；`invariants/INV-XXX.md` 元数据含 `id`、`paths`（Glob 路径路由，处理无关目录时不加载），正文强制 EARS + RFC 2119；`quirks/QUIRK-XXX.md` 元数据含 `id`、`external_service`、`verified_date`，正文仅 `Symptom` 与 `Invariant Action`。

估算方式：`estimate_tokens(text) = int(cjk / 1.4 + other / 3.8)`。skill `name` 必须等于目录名、匹配 `^[a-z0-9]+(-[a-z0-9]+)*$` 且 ≤64 字符；`description` 非空且 ≤1024 字符。

## 5. 无人值守熵减 (Entropy Control)

- **Dangling Symbol GC**：每次 PR 合并校验 `.knowledge/decisions/` 与映射后的既有 owner 文档中 `affects_symbols` 引用的文件与符号是否因重构被删除，失效即告警并触发维护 Agent 归档为 `superseded`。
- **Token 预算硬拦截**：根 `AGENTS.md` 或任意 `SKILL.md` 超出预算时 CI 直接阻断，倒逼内容向代码静态规则或模块化 Skill 迁移。
- **`.tmp/` 逃逸检测**：CI 验证 `.tmp/` 处于 `.gitignore` 忽略链中且无任何文件被错误追踪。

## 6. 已有 durable owner 体系时的落地映射

若目标仓库已定义权威文档 owner，则沿用其路径、frontmatter 契约与既有校验入口，MUST NOT 新建平行 `.knowledge/` 树，MUST NOT 叠加第二套门禁。L5 的排除矩阵、预测误差过滤、单一 owner、预算与 `.tmp/` 隔离仍然全部适用；路由表里的 L5 落点按下表替换：

| L5 落点 | 已有 owner 体系中的对应落点 |
| :--- | :--- |
| `.knowledge/decisions/ADR-XXXX.md` | 真实权衡类 ADR owner（本仓 `.docs/adr/NNNN-slug.md`） |
| `.knowledge/invariants/INV-XXX.md` | 实现 / 架构 / 运维 owner（本仓 `.docs/tech/`） |
| `.knowledge/quirks/QUIRK-XXX.md` | 实现 / 运维 owner 中的外部异常条目（本仓 `.docs/tech/`） |
| `.agents/skills/<skill-name>/SKILL.md` | 仓库既有 skill 根目录（本仓 `.agents/skills/`） |
| `scenarios/*.yaml` | 独立 Holdout 仓的空间 C |
| `<worktree>/.tmp/` | 目标仓库既有 `.tmp/`（本仓 `.gitignore` 已忽略） |

本仓沿用 `bun run docs:check`（`cd web && node ../.docs/db/scripts/doc-compiler.js check --root ../.docs/db && node ../.docs/scripts/doc-integrity-check.js --repo ..`），检索入口 `bun run docs:tree|index|search|show`，维护现有 owner 见 [文档维护流程](../workflows/route-and-write.md)。写入 `.docs/adr/`、`.docs/tech/` 时须同时满足其检索 frontmatter（`description`、`kind`、`topic`）与 ADR 文件名规则 `^\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*\.md$`；L5 的 `id` / `status` / `affects_symbols` / `paths` 字段继续保留。`check_doc_budgets.py` 负责预算、skill 契约、`.tmp/` 卫生，以及 `.knowledge/` 与映射后 owner 文档中 `affects_symbols` 的符号 GC。
