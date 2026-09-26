# Project History

Durable record of shipped changes that altered the project's architecture,
deployment shape, or long-lived conventions. Each entry links to owning docs
(`.docs/tech/`, `.docs/adr/`, `.docs/db/`) rather than duplicating their
content. Process files are deleted after wrap-up; this file is the surviving
narrative.

## 2026-09-26 — Astryx migration: Phase 0 + Phase 1 (foundation, coexistence, spikes)

The management UI's migration from Vue 3 to React 19 + Astryx + StyleX
delivered its first slice on branch `docs/astryx-migration-plan`:

- **Dual-document coexistence:** `page_routes.json` v2 carries an `astryx`
  flag per route; Go serves `index.html` or `astryx.html` per flag +
  `gpt-load.frontend` cookie (`Vary: Cookie`); the Vite dev selector plugin
  mirrors the same rules. Classic stays the default; unknown paths fall back
  per the cookie.
- **Astryx shell parity:** TanStack Router (code-based tree generated from
  the manifest), react-intl, TanStack Query, React Compiler, the built
  `neutral`-derived theme at classic density, AuthGate/login/not-found/
  preferences, and the route announcer.
- **Spikes:** groups collection on `Table` (server-driven sort/filter/page),
  log detail on the `DetailPanel` primitive (ADR-0002), log time-range filter
  on `DateTimeInput` across three locales.
- **Go/no-go gate:** all seven criteria passed; evidence is recorded in
  `.docs/tech/astryx-migration-plan.md` (Phase 1 outcome). Notable numbers:
  CSP 15/15 across Chromium 125 / Chrome 153; first-screen gzip +279.3 kB vs
  classic; zero swizzles.
- **Contracts established:** `RouteLink` flag-boundary navigation policy, the
  sparse `validateSearch` rule, the vue-router-parity search codec, and the
  `resetScroll: false` query-navigation rule — all documented in the
  migration plan and enforced by `web/scripts/search-codec.test.ts` and the
  `astryx-*` Playwright specs.

Owner doc: `.docs/tech/astryx-migration-plan.md`. ADRs:
`0001-collection-read-model-scale`, `0002-astryx-detail-layer-primitive`.
Phase 2+ domain migrations follow the same plan.
