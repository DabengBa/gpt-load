---
name: l5-doc-governance
description: "Guide and constrain documentation updates at the end of every development task: discard facts inferable from code, distill non-inferable decisions, invariants and quirks into durable owners, keep .tmp/ process docs out of git, and enforce AGENTS.md/SKILL.md token budgets. Use when wrapping up a change, before commit or PR merge, or when asked to update docs after coding."
license: Apache-2.0
compatibility: "Requires git >= 2.38 and python3 >= 3.10. Tool-agnostic across agentskills.io and AAIF compliant agents."
metadata:
  author: l5-dark-factory
  version: "2.1.0"
  standard: agentskills.io
---

# L5 Pure-Agent Documentation Governance

每次开发工作收尾（提交 / 合并 PR 前）使用。目标仓库的 `AGENTS.md` 与既有文档 owner 规则优先于本 skill；面向无会话历史、短生命周期的 Fresh-Context Agent，只保留代码无法反推的知识。

## 收尾四步

1. 打包：`python3 .agents/skills/l5-doc-governance/scripts/prepare_distill_bundle.py --worktree . --base-ref origin/dev --out .tmp/distill-bundle.json`。只把 stdout 与 bundle 路径带入上下文。
2. 裁决：零历史上下文读取该 bundle 与 `references/distillation-rubric.md`；逐条过可预测性门禁，能由代码/类型/测试反推的直接丢弃，其余按路由表写入 durable owner。
3. 验证：`python3 .agents/skills/l5-doc-governance/scripts/check_doc_budgets.py --repo .`。目标仓库已有文档校验入口时一并复用（本仓 `bun run docs:check`），只处理脚本列出的 `errors`。
4. 清理：提交沉淀变更；`.tmp/` 保持未跟踪，合并 / 销毁 worktree 后过程文档随之消失。

skill 安装在其他根目录时，把命令前缀换成该 skill 的实际路径。

## 硬约束

- MUST NOT 把目录树、API/字段表、Schema、UI 规格、依赖版本、`TODO.md` / `CHANGELOG.md` / 进度写进文档；见 `references/architecture-and-boundaries.md` 的排除矩阵。
- MUST 让过程文档只存在于 `<worktree>/.tmp/`，MUST NOT 提交；`.gitignore` 必须忽略 `.tmp/` 与 `**/.tmp/`。
- MUST 写入前查 `existing_knowledge_index` / `existing_owner_files`：同一事实只更新一个 owner，不新建平行说明。目标仓库已有 durable owner 体系时沿用其路径与现有校验入口；无体系时用 `scripts/init_l5_docs.py` 建立 `.knowledge/` 三空间。
- MUST 保持预算：根 `AGENTS.md` ≤120 行 / ≤1500 估算 tokens，每个 `SKILL.md` ≤500 行。超出就把流程移入 `references/` / `scripts/`，或编码成静态检查。
- MUST NOT 增设 Markdown 提示词注入审计流；只在业务代码与独立验收系统之间保留物理盲区（空间 C 对 Builder Agent 不可见，失败只返回单行症状）。
- MUST NOT 把脚本源码或整个 `.knowledge/` 目录读进上下文；脚本只返回 stdout 与退出码。

## 按需读取

| 本次收尾遇到 | 读取 |
| :--- | :--- |
| 判断哪些内容该删、该留、路由到哪类文档 | `references/distillation-rubric.md` |
| 三空间拓扑、四条铁律、排除矩阵、已有 owner 的落地映射 | `references/architecture-and-boundaries.md` |
| ADR / INV / QUIRK / Holdout 字段与模板 | `references/formats-and-schemas.md` |
| 初始化新仓、日常熵减、悬空符号与预算超限处理 | `references/workflows.md` |
| 治理文档自身的写法约束 | `AGENTS.md`、`.docs/AGENTS.md` |

## 脚本

| 命令 | 作用 | 输出 |
| :--- | :--- | :--- |
| `scripts/init_l5_docs.py --repo . --with-templates` | 建目录、`.gitignore` 规则、模板、可选 holdout | JSON：created / gitignore_added / holdout_created |
| `scripts/prepare_distill_bundle.py --worktree . --base-ref origin/dev --out .tmp/distill-bundle.json` | 收集 `.tmp/` 文档、分支 diff、现有知识索引 | JSON 摘要 + bundle 文件 |
| `scripts/check_doc_budgets.py --repo .` | 预算、frontmatter、`.tmp/` 卫生、悬空符号（含映射后 owner） | JSON：status / errors / warnings / budgets / counts + 退出码 |

## 验收

按 `CHEATSHEET.md` 逐项确认；只报告本次实际改动与脚本结论。
