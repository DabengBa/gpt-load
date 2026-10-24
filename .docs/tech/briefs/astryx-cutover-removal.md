---
created: 2026-09-30
source: Direct user instruction after the Phase 4 wrap-up
confirmed: 2026-09-30
last_updated: 2026-09-30
---
# Brief: Astryx Cutover & Classic Removal

## User Original Request

> 请继续完成phase5和phase6

("Continue and finish phase 5 and phase 6.")

## Background & Motivation

Astryx migration Phases 0–4 had shipped (2026-09-30 wrap-up `e413188e`): all 11
`page_routes.json` routes carried `astryx: true`, the astryx e2e suite passed
134/134, codec tests 15/15, and the real-binary CSP matrix 36/36. Two phases
remained in the migration plan (`astryx-migration-plan.md`):

- **Phase 5 Cutover**: flip the default document to astryx while keeping a
  one-release fallback channel through the `gpt-load.frontend=classic` cookie.
- **Phase 6 Deletion**: delete `frontends/classic`, the Vue/Tailwind toolchain,
  the Go document-selection code, the manifest flag, the dev-selector plugin,
  and the classic Playwright project.

## Intent Domains

### Combined delivery (user-confirmed)

- What the user wanted: Phase 5 and Phase 6 delivered as one change — Astryx
  becomes the only frontend, and classic plus all coexistence infrastructure
  (cookie selection, the second HTML entry, the second tsconfig, the second
  Playwright project) is removed in the same delivery. This **gives up the
  planned one-release fallback window**; the fallback degrades to git/version
  rollback.
- Change history:
  - 2026-09-30 the user said "continue and finish phase 5 and phase 6"; on the
    cutover-window question they chose "merge into a single spec"; on the
    preference-control naming question they chose "decide at Phase 6" — which
    resolved itself because the control is deleted with classic.
- Implementation trail:
  - Spec directory `.docs/specs/260930-01-astryx-cutover-removal/` (process
    files deleted during archive).
  - Code paths: `internal/webui/` (single document, manifest v3),
    `web/vite.config.ts` (single entry), `web/index.html`,
    `web/src/frontends/astryx/`, `web/src/shared/`, `web/e2e/`,
    `web/package.json`, `web/pnpm-workspace.yaml`, `web/eslint.config.mjs`.
  - Upstream spec: the Phase 5/6 sections of
    `.docs/tech/astryx-migration-plan.md`.
  - Delivery record (2026-09-30):
    - `f2b41446` Task A: density tokens moved into
      `astryx/theme/tokens.css` (the `tokens` layer) with zero density change;
      the `astryx-density` ±1px assertions stayed green.
    - `81f45345` Tasks B–E: Go single-document serving (cookie selection and
      `Vary: Cookie` removed), manifest v3 dropped the `astryx` field, single
      `index.html` entry, and `astryx.html`, the dev selector,
      `frontend-preference.ts`, `verify-frontend-cookie.mjs`, the Interface
      preference segment, and `isAstryxNavigable` were all deleted; the classic
      tree and the Vue/Tailwind toolchain were removed (`typescript-eslint`
      took over parsing); the classic Playwright project and its 8 specs were
      removed and `e2e` became a single entrypoint.
    - Verification: type-check, build, and lint green; the `internal/webui`
      single-document contract test green (including the leftover-cookie case);
      the `internal/control` health contract test repointed to
      `shared/control/resources/health.ts` and green; all `verify:*` and
      `test:*` scripts green; astryx Playwright 129/129 (with two existing spec
      assertions fixed: a cache-token-details button-name collision and an
      adminOnly redirect wait window).
    - Note: seven `verify-*.mjs` scripts got an explicit `process.exit` —
      `@stylexjs/unplugin` left a live handle after `server.close()` and had
      hung the gate script.

### Constraints (inherited from the whole migration)

- User-visible URLs, session state, locale, and theme are unchanged; the only
  visible difference is that astryx renders by default with no path back to
  classic.
- CSP is unchanged; the Go-embedded deployment contract is unchanged; the
  `make build` / `make test` command contract is unchanged.
- The Chrome 125+ browser baseline is unchanged.
- A leftover `gpt-load.frontend` cookie is ignored: the mechanism is gone and
  old values are not interpreted.
