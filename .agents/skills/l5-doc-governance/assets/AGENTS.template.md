---
scope: root
schema_version: 2.1
token_budget: 1500
---

# Execution Primitives
- Build: `<BUILD_CMD>`
- Static Check: `<LINT_AND_TYPECHECK_CMD>`
- Unit Test: `<UNIT_TEST_CMD>`
- Doc Budget & GC Audit: `python3 .agents/skills/l5-doc-governance/scripts/check_doc_budgets.py --repo .`

# Context Routing (Load ONLY on match)
- Runtime & domain invariants: `.knowledge/invariants/*.md` (filter by `paths` frontmatter)
- Rejected approaches & causal ADRs: `.knowledge/decisions/*.md` (filter by `affects_symbols`)
- External dependency quirks: `.knowledge/quirks/*.md`
- Executable SOPs: `.agents/skills/*/SKILL.md` (load L2 body ONLY on trigger match)

# L5 Operational Constraints
- Ephemeral process documents MUST reside exclusively in `<worktree>/.tmp/` (untracked by git).
- MUST NOT mirror directory trees, API tables, schemas, or type specs into `.knowledge/`.
- MUST execute fresh-context audit against disk state before completion; MUST distill permanent knowledge before PR merge discards `<worktree>/.tmp/`.
