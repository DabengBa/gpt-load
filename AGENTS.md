---
scope: root
schema_version: 2.1
token_budget: 1500
---
# Repository Identity

- Canonical repository: `DabengBa/gpt-load`
- Canonical URL: `https://github.com/DabengBa/gpt-load`
- Canonical remote: `origin` → `https://github.com/DabengBa/gpt-load.git`
- Default integration branch for this checkout: `dev`
- `upstream` may point to `tbphp/gpt-load` for reference only. Do not treat `tbphp/gpt-load` as the repository being maintained, and do not merge or push there unless the user explicitly requests it.

## Git Rules

- PR discovery, merge, fetch, and push operations target `DabengBa/gpt-load` by default.
- README repository links, clone commands, and ownership statements must identify `DabengBa/gpt-load`.
- Keep links to `tbphp/gpt-load` only when they intentionally refer to upstream release/container artifacts, and label that relationship explicitly.
- Before merging, verify the target repository and base branch with `gh pr view -R DabengBa/gpt-load`.

## Commands

- `make dev` / `make build` / `make test` / `make check` (deployment regression without Docker/SSH)
- `go test ./...` · `go vet ./...`
- `pnpm --dir web run build` · `lint` · `type-check` · `test` · `e2e`
- Documentation validation: `bun run docs:check`
- Doc governance: `python3 .agents/skills/l5-doc-governance/scripts/check_doc_budgets.py --repo .`

## Documentation Owner Routing

Read `.docs/AGENTS.md` first to locate the current document owner.

- `.docs/db/` — product semantics (validated by the semantic compiler).
- `.docs/tech/` — implementation, architecture, operations, calculation facts (incl. INV/QUIRK entries).
- `.docs/adr/` — hard-to-reverse decisions with explicit trade-offs.
- `.agents/skills/` — deterministic multi-step operational procedures.
- `docs/analysis/` — dated analysis snapshots; verify against current owners and code before acting.

## Temporary Artifacts

- All process documents (drafts, scratch, migration notes, probes) live under `.tmp/` (and `**/.tmp/`), which MUST stay untracked and git-ignored. Never commit them.

## Hostinger Deployment

- Follow `.agents/skills/gpt-load-operations/SKILL.md` for requests to push and update the Hostinger GPT-Load service.
- Run `scripts/deploy.sh [branch]` locally after validation and pushing the intended commits; the default branch is `dev`.
- Build images locally and upload them over SSH. Hostinger only loads images and runs Compose; do not build images or invoke BuildKit on the server.
- Verify the deployed image/version, local and public health endpoints, and behavior relevant to the change before reporting success.
