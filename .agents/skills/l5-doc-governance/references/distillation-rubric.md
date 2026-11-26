# Pre-PR Distillation Rubric (Prediction Error Filtering)

Run ONLY with fresh context reading `.tmp/distill-bundle.json` and this file. The distiller has zero session history; MUST NOT read the whole `.knowledge/` directory, MUST NOT load owner files whose `paths` frontmatter does not match the diff.

## 1. Predictability Gate (First Filter)

对 `.tmp/` 过程文档与失败日志中的每条候选信息提问（原文）：

> **"若仅看合入后的代码、类型和单元测试，未来无记忆 Agent 能否反推出此结论？"**

English form from the source report:

> *"Given ONLY the merged codebase, type definitions, schemas, linters, and unit tests, could a zero-history agent reliably deduce this constraint or avoid this mistake?"*

- **YES / 能反推** → **REJECT / 直接丢弃**。不污染 `.knowledge/`，随 `.tmp/` 消亡。
- **NO / 不能反推** → **DISTILL / 蒸馏**，按下面的路由表写入 durable owner。

## 2. Routing Decision Tree

1. **Rejected Alternative or Non-Obvious Tradeoff?**
   - Signal: Ephemeral docs debated Option A vs Option B, benchmarks rejected an intuitive approach, or code contains a counter-intuitive workaround that a future refactorer might "simplify" back into a bug.
   - Target: `.knowledge/decisions/ADR-XXXX.md` (MUST bind `affects_symbols`). Template: `../assets/decision.template.md`.
2. **Untypable Cross-Boundary Invariant?**
   - Signal: Concurrency lock ordering, idempotency window, backward-compatibility window, or cross-module coupling prohibition not enforceable by static types/linters.
   - Target: `.knowledge/invariants/INV-XXX.md` using EARS (`WHEN ... IF ... THEN ... MUST/MUST NOT ...`) and `paths` globs. Template: `../assets/invariant.template.md`.
3. **External Dependency / Toolchain Anomaly?**
   - Signal: Third-party service out-of-order behavior, OS/runtime bug, undocumented rate limit, or SDK quirk discovered via retry failures.
   - Target: `.knowledge/quirks/QUIRK-XXX.md`. Template: `../assets/quirk.template.md`.
4. **Repeatable Multi-Step Procedure?**
   - Signal: Sequence of 3+ CLI/tool/environment operations verified to succeed deterministically.
   - Target: `.agents/skills/<skill-name>/SKILL.md`.
5. **New End-to-End Business Journey?**
   - Target: Draft Holdout scenario proposal for Space C (`scenarios/*.yaml`) **without copying internal implementation identifiers**. Evaluator returns `on_fail_emit: ONE_LINE_SYMPTOM_ONLY`.

目标仓库已有 durable owner 体系时，按 `architecture-and-boundaries.md` 第 6 节替换落点；路由判断本身不变。

## 3. Deduplication & Compaction

- Check `existing_knowledge_index` and `existing_owner_files` in the bundle. Update an existing entry rather than creating a fragment if the same `affects_symbols` or `external_service` is already covered. 同一事实只保留一个 owner。
- If a rule from `.knowledge/invariants/` was successfully encoded into an AST linter rule or type constraint during this PR, **DELETE that invariant file in the same commit**.
- 超预算内容移入 `references/` / `scripts/`，或改成编译器 / Linter 诊断，不靠删约束来过门禁。

## 4. Pre-PR 蒸馏闭环

1. **载荷打包**：`prepare_distill_bundle.py` 采集 `<worktree>/.tmp/` 全部临时文档（单文件 12000 字符上限，扩展名 `.md/.txt/.yaml/.yml/.json/.log`）、PR 分支相对基线的 `git diff`（24000 字符上限，含未提交工作区 diff）、现有 `.knowledge/` 索引与既有 owner 文件列表。
2. **Fresh-Context 蒸馏裁决**：零历史 Agent 只读取 bundle 与本 rubric，逐条执行可预测性测试并按路由表沉淀。
3. **验证与合并销毁**：`check_doc_budgets.py` 门禁通过 → 提交 `.knowledge/` / `.agents/skills/`（或映射后的 owner）变更 → 合并 PR → 销毁 Worktree，`.tmp/` 彻底消失，主干工作区保持绝对干净。
