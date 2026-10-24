---
created: 2026-09-25
source: The user's original instruction and per-item choice confirmations in the migration-plan review session
confirmed: 2026-09-25
last_updated: 2026-09-25
---
# Brief: Astryx Frontend Migration

## User Original Request

> 阅读后，结合项目实际情况，结合网络迁移经验，更新并详细化这份迁移文档。

("After reading it, combine the project's actual situation and general migration
experience, then update and expand this migration document.")

Followed by:

> 根据文档，继续完成开发工作。

("Following the document, continue with the development work.")

## Background & Motivation

The management UI was Vue 3 + Reka UI + scoped BEM CSS (roughly 17.6k lines of
hand-written styles). The user required migrating the frontend to React 19 +
Astryx (`@astryxdesign/core`) + StyleX while keeping existing behavior, routing,
authentication, localization, Go-embedded deployment, and the compact visual
density unchanged. The migration had to produce feasibility evidence (a go/no-go
gate) before a full migration was approved.

On 2026-09-25 the user confirmed each choice, all recorded in the Confirmed
choices section of `.docs/tech/astryx-migration-plan.md`:

- Browser baseline: Chromium/Chrome 125+; Safari/Firefox best effort, not
  release-blocking
- Visual density: match the compact metrics of the then-current Vue frontend
- Routing: TanStack Router (code-based)
- React Compiler: enabled (stable Babel integration)
- App i18n: react-intl
- Theme base: `neutral` plus GPT-Load overrides

## Intent Domains

### Migration execution

- What the user wanted: work through `.docs/tech/astryx-migration-plan.md`,
  delivering Phase 0 (framework-neutral shared-layer extraction) and Phase 1
  (React/Astryx scaffolding, same-URL coexistence, shell parity, 7 spikes,
  go/no-go gate evidence) first, with the user deciding at the gate whether to
  enter Phase 2+ domain migration.
- Status: **Phase 0 + Phase 1 + Phase 2 (settings/models/home) + Phase 3
  (access-keys/monitor/logs/schedule) shipped (2026-09-27)**; Phase 4 (groups +
  import) follows the plan.
- Change history:
  - 2026-09-25 the plan document was expanded (149 → ~900 lines), recording the
    measured baseline and every technical choice.
  - 2026-09-25 the user confirmed five open choices and the neutral theme base,
    leaving no open decisions in the document.
  - 2026-09-25 the user said "following the document, continue with the
    development work"; brainstorming scoped the first deliverable as Phase 0 +
    Phase 1 up to the go/no-go gate.
  - 2026-09-26 all Phase 1 spikes finished, the seven gate evidence items were
    complete, and the user reviewed and replied "go".
  - 2026-09-26 final review found 17 items, all handled (route handoff, sparse
    `validateSearch`, codec parity, `resetScroll`, `Vary: Cookie`, and others);
    the review was closed.
  - 2026-09-27 Phase 2 shipped three domains: settings (`e9a8b0bf`), models with
    embedded model prices (`7f6e2b1c`), and home (`805b0f95`); the slice review
    collapsed dead fields and closed (`1b8d0a18`), and the plan document recorded
    the Phase 2 outcome (`3dffa83b`).
  - 2026-09-27 Phase 3 shipped: access-keys (`3a019b90`), the monitor host with
    health and usage (`781ef55b`), inspector (`ef97aaa7`), full logs parity
    (`ac540c16`), and the schedule dispatch center (`2fdd51c6`); the final review
    added 12 monitor-route codec unit tests, the real-binary CSP matrix passed
    30/30 across three browsers, the plan recorded the Phase 3 outcome, process
    files were deleted, and wrap-up commit `8cbd2125` followed.
  - 2026-09-27 Phase 4 scoping: group-detail (9,219 lines) and import (5,670
    lines) were the last two classic routes; astryx `importRecovery` was wrongly
    wired to localStorage (classic used sessionStorage) and `onUnauthorized` was
    not connected to `captureForUnauthorized`, so both became pre-flag fixes.
  - 2026-09-30 Phase 4 shipped: codec extraction plus the recovery fix
    (`641f3950`); the detail host with settings/models/credentials tabs, dual
    import modes, and the staging layer (`057a7833`) — after rebasing onto
    fork/dev, upstream `f78071f8` readability semantics were synced
    (`eefa9bc3`, `cff1b699`); the final review fixed four leftover document
    navigations from the flag flip; astryx e2e 134/134, classic readability 7/7,
    codec 15/15; the manifest ended with 11/11 astryx flags, leaving only the
    Phase 5 default flip and Phase 6 deletion.
- Implementation trail:
  - Spec directory `.docs/specs/260925-01-astryx-migration-foundation/`
    (process files deleted during archive; gate evidence and review records were
    merged into the Phase 1 section of `.docs/tech/astryx-migration-plan.md`).
  - Phase 2 spec directory `.docs/specs/260926-01-astryx-phase2-domains/`
    (process files deleted during archive; domain breakdown, review records, and
    real-binary evidence merged into the Phase 2 outcome and this trail).
  - Phase 3 spec directory `.docs/specs/260927-01-astryx-phase3-domains/` (same
    handling).
  - Branch `docs/astryx-migration-plan`; key commits: `55add5be` (groups
    collection), `ef6b2779` (log detail layer + ADR-0002), `d0362813` (log time
    range), `c28a1c2c` (gate evidence), `d1570ac6` (plan writeback), `f1e722cf`
    (final-review fixes), `e9a8b0bf` (settings domain), `7f6e2b1c` (models
    domain), `805b0f95` (home domain), `1b8d0a18` (Phase 2 slice review).
  - Code paths: `web/src/frontends/astryx/`, `web/src/shared/`
    (routing/lib/control/controllers/domain), `internal/webui/` (manifest v2,
    document selection, `cmd/webui` CSP harness), and `web/e2e/astryx-*.spec.ts`.
  - ADRs: `.docs/adr/0001-collection-read-model-scale.md` (collection read
    model) and `0002-astryx-detail-layer-primitive.md` (detail overlay
    primitive).
  - Binding conventions established in Phase 2 (recorded in the plan's Phase 2
    outcome, applied to Phases 3–4): a mutable controller always publishes a
    memoized snapshot consumed through `useSyncExternalStore`; Vue `watch`
    semantics port to a render-phase adjustment plus a post-commit effect; and
    sensitive-operation state machines move down into a shared controller
    (`gateway-actions`).

### Constraints (across all phases)

- User-visible URLs, session state, locale, theme, and browser state keys such as
  `gpt-load.import-reauth-draft` were frozen during coexistence; the new
  `gpt-load.frontend` cookie only selected a static document and could not affect
  authentication, API routing, or file paths.
- CSP is unchanged: `style-src-elem 'self'` rules out runtime style injection, so
  every theme must be produced statically by `astryx theme build`.
- The Go-embedded deployment and CI command contract (the command set pinned by
  `workflow_test.go`) must not break.
- Dependencies are pinned to exact versions under the seven-day rule.

## Non-Goals

- Completed slices do not take on Phases 3–6 (operational and
  highest-coupling domains, the default frontend flip, classic deletion); those
  phases are scoped one at a time by the plan.
- No product behavior, backend API contract, or page semantics change; `.docs/db`
  semantic docs remain the acceptance checklist.
- No visual redesign beyond what adopting Astryx itself implies.
- No migration ADR until the go/no-go gate has produced a hard-to-reverse
  decision to record.
