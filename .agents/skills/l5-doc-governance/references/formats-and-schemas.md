# L5 Documentation Formats & Schemas

All `.knowledge/` artifacts MUST begin with strict YAML frontmatter and use RFC 2119 keywords (`MUST`, `MUST NOT`, `SHOULD`) with zero narrative prose. Templates live in `../assets/`.

## 1. Decision Record (`.knowledge/decisions/ADR-XXXX.md`)
- Required frontmatter:
  - `id`: `"ADR-XXXX"`
  - `status`: `"active"` | `"superseded"`
  - `supersedes`: `null` | `"ADR-YYYY"`
  - `affects_symbols`: list of `"relative/file/path#SymbolName"` (checked by GC)
  - `created_at`: `YYYY-MM-DD`
- Body sections: `Context`, `Decision`, `Rejected`, `Constraint`.
- Template: `../assets/decision.template.md`.

## 2. Invariant (`.knowledge/invariants/INV-XXX.md`)
- Required frontmatter:
  - `id`: `"INV-XXX"`
  - `paths`: non-empty list of repo-relative globs (e.g., `["services/billing/**"]`)
  - `enforcement`: `"static-lint"` | `"holdout-eval"` | `"fresh-audit"`
- Body format: EARS syntax (`WHEN ... IF ... THEN ... MUST / MUST NOT ...`).
- Template: `../assets/invariant.template.md`.

## 3. Quirk (`.knowledge/quirks/QUIRK-XXX.md`)
- Required frontmatter:
  - `id`: `"QUIRK-XXX"`
  - `external_service`: identifier string (e.g., `"stripe-webhooks"`, `"postgres-16"`)
  - `verified_date`: `YYYY-MM-DD`
- Body sections: `Symptom`, `Invariant Action`.
- Template: `../assets/quirk.template.md`.

## 4. Clean-Room Holdout Scenario (`scenarios/<id>.yaml` in Space C)
- Required keys: `scenario_id`, `satisfaction_policy` (`runs`, `min_pass`, `min_aggregate_score`), `setup` (`digital_twins`, `initial_state`), `journey_assertions`, `feedback_projection` (`on_fail_emit: ONE_LINE_SYMPTOM_ONLY`).
- MUST NOT copy internal implementation identifiers into the Holdout repo.

## 5. Root Boot Contract & Skill Envelope

- `AGENTS.md` template: `../assets/AGENTS.template.md`. Replace `<BUILD_CMD>`, `<LINT_AND_TYPECHECK_CMD>`, `<UNIT_TEST_CMD>` by inspecting project manifests (`package.json`, `Cargo.toml`, `pyproject.toml`, `Makefile`). Frontmatter: `scope: root`, `schema_version: 2.1`, `token_budget: 1500`.
- `SKILL.md` frontmatter: `name` (equals directory name, `^[a-z0-9]+(-[a-z0-9]+)*$`, ≤64 chars), `description` (non-empty, ≤1024 chars); L1 exposes only `name` + `description`, L2 body ≤500 lines.
- 精确预算：根 `AGENTS.md` ≤120 行 / ≤1500 估算 tokens；`SKILL.md` ≤500 行；diff 载荷 24000 字符；单个 `.tmp/` 文档 12000 字符；git ≥2.38；python3 ≥3.10；治理套件版本 `2.1.0`。
- `check_doc_budgets.py` 必检：ADR `id/status/affects_symbols`、INV `id/paths`（0 匹配仅告警）、QUIRK `id/external_service`、`.knowledge/` 与映射后 owner 文档中的悬空文件/符号、被追踪的 `.tmp/` 文件、`.gitignore` 忽略链。EARS 正文、`supersedes/created_at/verified_date/enforcement` 取值与 Holdout YAML 不由脚本校验，由本文件与 Fresh-Context 审计承担。

## 6. 已有 owner 体系的 frontmatter 叠加

写入目标仓库既有 owner 时，MUST 同时满足其 frontmatter 契约，MUST NOT 用 L5 字段替换掉它。本仓 `.docs/adr/`、`.docs/tech/` 需要 `description`、`kind`（`technical` | `workflow` | `adr`）、`topic`，`description` 不得与标题相同；ADR 文件名须匹配 `^\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*\.md$`。L5 的 `id` / `status` / `affects_symbols` / `paths` 等字段保留在同一 frontmatter 内，供 `check_doc_budgets.py` 的符号 GC 与预测误差过滤使用；脚本只对显式声明 `affects_symbols` 且 `status` 未置 `superseded` 的 owner 文档做此检查，不向其余 owner 文档强加 L5 必填字段。
