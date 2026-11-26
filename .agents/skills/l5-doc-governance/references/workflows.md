# Workflows

三个工作流：新仓初始化（一次性）、收尾蒸馏（每次开发工作）、日常熵减（PR / 定时）。详细步骤只在本文件；`SKILL.md` 只保留四步路由。

## Workflow 1: Repository Initialization

仅在目标仓库**没有** durable owner 体系时执行；已有体系时按 `architecture-and-boundaries.md` 第 6 节映射，MUST NOT 新建平行 `.knowledge/`。

1. Execute deterministic scaffolding script (uses templates from `assets/`):
   ```bash
   python3 .agents/skills/l5-doc-governance/scripts/init_l5_docs.py --repo <REPO_ROOT> --with-templates --init-holdout <OPTIONAL_HOLDOUT_PATH>
   ```
   Creates `.knowledge/{decisions,invariants,quirks}/`, `.agents/skills/`, adds `.tmp/` and `**/.tmp/` to `.gitignore`, copies `assets/AGENTS.template.md` when no root AGENTS file exists (`--force` overwrites), and creates `scenarios/`, `digital-twins/`, `seeds/` in the Holdout path when `--init-holdout` is given.
2. Replace `<BUILD_CMD>`, `<LINT_AND_TYPECHECK_CMD>`, and `<UNIT_TEST_CMD>` placeholders in `AGENTS.md` by inspecting project manifests (`package.json`, `Cargo.toml`, `pyproject.toml`, `Makefile`, etc.).
3. Wire CI / git hooks (in the repo's standard CI workflow or Makefile) to invoke `check_doc_budgets.py` on PR gates and `prepare_distill_bundle.py` prior to PR merge. 已有统一入口时复用该入口，不叠加第二套 hook。
4. Verify `.tmp/` and `**/.tmp/` are ignored by git and zero process files are staged.

## Workflow 2: Wrap-Up / Pre-PR Knowledge Distillation

Run inside the active worktree at the end of every development task, before PR merge / `<worktree>/.tmp/` vanishes:

1. Collect `<worktree>/.tmp/` artifacts, branch diff, and current owner index into a single JSON bundle:
   ```bash
   python3 .agents/skills/l5-doc-governance/scripts/prepare_distill_bundle.py --worktree <WORKTREE_ROOT> --base-ref origin/dev --out .tmp/distill-bundle.json
   ```
   Only stdout and the bundle path enter the agent context; the script excludes `distill-bundle.json` itself.
2. Spawn a fresh-context Distiller step reading ONLY `.tmp/distill-bundle.json` and `references/distillation-rubric.md`.
3. Apply the **Predictability Gate**: if a future zero-history agent can infer the fact from the merged code/types/tests alone, **DISCARD IT**. Otherwise route it:
   - Causal tradeoff / rejected alternative → `.knowledge/decisions/ADR-XXXX.md` (`assets/decision.template.md`)
   - Non-typable runtime/domain rule (EARS + RFC 2119) → `.knowledge/invariants/INV-XXX.md` (`assets/invariant.template.md`)
   - External API / OS / toolchain anomaly → `.knowledge/quirks/QUIRK-XXX.md` (`assets/quirk.template.md`)
   - Reusable multi-step CLI/tool procedure → `.agents/skills/<skill-name>/SKILL.md`
   - New end-to-end behavioral assertion → propose scenario update to Space C Holdout repository
4. Run Workflow 3 validation, commit durable-owner / `.agents/skills/` updates to the PR branch, and allow `<worktree>/.tmp/` to vanish upon merge/prune.

## Workflow 3: Daily Maintenance, Budget Gate & Entropy GC

Execute on every PR gate and scheduled maintenance cycle, and as the final step of every wrap-up:

1. Run deterministic budget, frontmatter, `.tmp/` hygiene, and dangling symbol audit:
   ```bash
   python3 .agents/skills/l5-doc-governance/scripts/check_doc_budgets.py --repo <REPO_ROOT>
   ```
   Use `--check-skill <SKILL_DIR>` to validate one standalone skill; defaults `--agents-max-lines 120`, `--agents-max-tokens 1500`, `--skill-max-lines 500`.
2. Resolve any non-zero exit code strictly:
   - **Dangling `affects_symbols` or orphaned `paths`** (both `.knowledge/` and mapped owner docs): Inspect git log for the refactored symbol. If the constraint still applies under a new symbol name, update frontmatter; if obsolete, set `status: superseded` or delete the entry.
   - **Budget exceeded (`AGENTS.md` > 120 lines / 1500 est. tokens, or `SKILL.md` > 500 lines)**: Extract procedural blocks into `.agents/skills/*/SKILL.md` or `references/` and replace inline rules with compiler/linter diagnostics whenever expressible statically.
   - **Tracked files under `.tmp/`**: Unstage immediately (`git rm -r --cached <path>/.tmp`) and verify `.gitignore`.
   - **`SKILL.md` name/description invalid**: rename the directory to match `name`, or fix `description` (≤1024 chars, non-empty).
3. Target repos with an existing doc-validation entry: run that entry too (本仓 `bun run docs:check`); do not create a second gate.

## Fresh-Context 审计

收尾前 MUST 对磁盘现状做一次零历史审计：假定没有会话记忆，仅凭 `AGENTS.md`、命中的 `SKILL.md`、按路由加载的 owner 文档与代码/类型/测试，能否重建本次结论。能重建 → 删除新增文档；不能 → 保留并完成蒸馏。MUST NOT 用 Markdown 记录任务进度或执行历史（含 `TODO.md`、`CHANGELOG.md`）；状态进 Dolt 图状态库与 Git 历史。
