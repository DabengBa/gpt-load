# Project History

Durable record of shipped changes that altered the project's architecture,
deployment shape, or long-lived conventions. Each entry links to owning docs
(`.docs/tech/`, `.docs/adr/`, `.docs/db/`) rather than duplicating their
content. Process files are deleted after wrap-up; this file is the surviving
narrative.

## 2026-09-27 — Astryx migration: Phase 2 (low-coupling domains)

The second slice migrated `settings`, `models` (with the model-prices editor
embedded), and `home` onto the Astryx frontend on branch
`docs/astryx-migration-plan`, flagged in `page_routes.json` alongside the
Phase 1 pilot routes:

- **Shared seams:** every cross-frontend contract now lives in `shared/` —
  route query codecs (`routing/*-route.ts`, classic files become re-exports),
  stateful controllers (`controllers/settings-draft`, `model-price-*`,
  `home-statistics`, `gateway-actions`), and domain pure functions
  (`domain/settings`, `domain/home/subscription-quota`, …). Classic keeps its
  own composables where behavior is Vue-specific.
- **React Compiler discipline (durable):** mutable shared controllers publish
  a memoized snapshot consumed via `useSyncExternalStore`; render code reads
  `snapshot.*` and never calls `controller.get*()`. This was root-caused in
  the models drawer (compiler-freeze on stable-reference method calls) and is
  now a binding convention for Phases 3–4. Vue `watch` semantics port to
  render-phase adjustments + post-commit effects.
- **Sensitive operations:** gateway reveal/copy/quick-import runs through a
  framework-free `gateway-actions` controller — operation-identity guard,
  abort on key/client/config change and unmount, route-change invalidation,
  and ephemeral-state cleaner registration all preserved.
- **Evidence:** astryx e2e 79/79, classic e2e 58/58, go-csp 14/14 against a
  real WSL2-built linux binary (`GPT_LOAD_ORIGIN`), ICU 8132 messages, tsc×3
  + eslint clean. Commits: `e9a8b0bf` (settings), `7f6e2b1c` (models),
  `805b0f95` (home), `1b8d0a18` (slice review).

Owner doc: `.docs/tech/astryx-migration-plan.md` (Phase 2 outcome). Phase 3
(`access-keys`, then `monitor`/`schedule`/`logs`) follows the same plan.

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
