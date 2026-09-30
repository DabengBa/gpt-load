# Project History

Durable record of shipped changes that altered the project's architecture,
deployment shape, or long-lived conventions. Each entry links to owning docs
(`.docs/tech/`, `.docs/adr/`, `.docs/db/`) rather than duplicating their
content. Process files are deleted after wrap-up; this file is the surviving
narrative.

## 2026-09-30 — Astryx migration: Phases 5+6 cutover and classic deletion

The migration closed out on branch `docs/astryx-migration-plan` by
combining the Phase 5 default flip and Phase 6 deletion into a single
delivery (user-approved; ADR-0003) — Astryx is now the only management
frontend and the one-release cookie fallback window was deliberately
skipped. Rollback is version rollback.

- **Single document:** the Go server returns the same embedded
  `index.html` for every page route and every unknown browser path,
  ignoring leftover `gpt-load.frontend` cookies; `Vary: Cookie` is gone.
  `page_routes.json` v3 drops the `astryx` field and the shared/Go parsers
  now reject it as an unknown field.
- **Classic and coexistence infra deleted:** `web/src/frontends/classic`
  (2.4MB, 164 .vue), `web/main.ts`, `astryx.html`, the Vite dev selector
  plugin, `frontend-preference.ts`, `verify-frontend-cookie.mjs`, the
  Interface preference segment, `isAstryxNavigable`, and the Vue/Tailwind
  toolchain (vue, vue-i18n, vue-router, vue-demi, @tanstack/vue-query,
  reka-ui, @lucide/vue, plugin-vue, vue-tsc, eslint-plugin-vue,
  @vue/eslint-config-typescript — `typescript-eslint` now supplies the TS
  parser). Density tokens moved to `astryx/theme/tokens.css` first.
- **Playwright consolidation:** the classic project and its eight specs are
  removed; remaining projects are `astryx`, `go-csp`, `chromium-125`,
  `chrome-latest`. Doc ID `feature.frontend-preview-switch` retired.
- **Evidence:** astryx e2e 129/129, real Linux binary CSP matrix 36/36
  (go-csp + chromium-125 + chrome-latest across all 11 routes and the
  retired-cookie case), `internal/webui` server/manifest tests green,
  health contract test repointed to `shared/control/resources/health.ts`,
  all `verify:*`/`test:*` scripts green, tsc + eslint clean.
  `verify-*.mjs` scripts got an explicit `process.exit` — the StyleX
  unplugin leaves worker handles after `server.close()`.
  Commits: `f2b41446`, `81f45345`, `b97aff51`.

Owner doc: `.docs/tech/astryx-migration-plan.md` (Phases 5+6 outcome).
ADR: `0003-astryx-cutover-without-fallback-window`. The Vue→React
migration is complete; no classic frontend or coexistence code remains.

## 2026-09-30 — Astryx migration: Phase 4 (group detail + import)

The fourth slice migrated `/groups/:id` (unified settings+models editor,
credentials management tab, header portals) and `/import` (dual-mode
new-group/existing-group flows, subscription credential staging, and the
sessionStorage re-auth recovery chain) onto the Astryx frontend on branch
`docs/astryx-migration-plan`. Every manifest route now carries
`astryx: true`; only the Phase 5 default flip and Phase 6 classic deletion
remain.

- **Shared codecs:** `shared/routing/group-detail-route` (tab +
  credential/model query segments) and `shared/routing/import-route`
  (mode, `group_id` deep link, discovery params) extracted with classic
  re-exports; bare `/groups/:id` deliberately keeps the unified-editor
  default — the dead `GroupTabs.vue` normalizer would have hijacked it to
  the management tab.
- **Import operation owner:** `useStableImportOperation` ported as a
  module-scoped store with idempotency keys, a generation guard, and
  ephemeral-cleaner registration; `captureForUnauthorized` +
  `sessionStorage` draft restore give the 401 → login → draft-recovery
  chain exact classic parity.
- **Pending-transition guard (new precedent):** while TanStack commits a
  navigation the outgoing route still renders with the incoming location —
  every canonicalization/correction effect across all astryx views now
  bails when `pathname` leaves its own route, fixing the
  import→detail bounce. Programmatic success navigations go through the
  unsaved-changes bypass because the React dirty flag has not flushed yet
  where classic's synchronous guard already saw converged state.
- **Evidence:** astryx e2e 134/134 (import 6/6 incl. recovery chain +
  request-log display parity), classic readability 7/7, codec tests 15/15,
  log-format 49 cases, tsc + eslint clean. Upstream `f78071f8` readability
  semantics (protocol labels, layered route/protocol rows, zoned
  timestamps, labeled cache rates) were synced into the astryx table.
  Commits: `641f3950`, `057a7833`, `eefa9bc3`, `cff1b699`.

Owner doc: `.docs/tech/astryx-migration-plan.md` (Phase 4 outcome). Phase 5
cutover (default flip) follows the same plan.

## 2026-09-27 — Astryx migration: Phase 3 (operational domains)

The third slice migrated `access-keys`, the `monitor` host (health, usage,
and inspector tabs), `/logs` at full parity (replacing the Phase 1 spike
subset), and the `/schedule` dispatch center onto the Astryx frontend on
branch `docs/astryx-migration-plan`:

- **Shared codec extraction:** every route codec both frontends consume now
  lives in `shared/routing/` (`monitor-route`, `logs-route`,
  `access-key-collection-route`) plus `shared/domain/monitor/log-filters` and
  `shared/control/mutation-outcome`; the classic files are one-line
  re-exports, so query semantics cannot drift between frontends.
- **Logs parity:** full filter surface + advanced drawer, cursor pagination
  with rollback-on-failure, applied-filter chips, detail drawer with deep
  links, route identity affordances, access-key scoping, and responsive card
  layout — all asserted by a 16-test parity suite.
- **Schedule state machine:** draft serialization/hydration
  (`schedule_draft`), revision-aware PATCH with conflict preservation,
  optimistic toggles, probe-one/probe-all flows, and recovery actions ported
  faithfully; `SchedulePanelDetail` keeps classic's `#priority-N` input ids
  via a native-input primitive because Astryx `TextInput` overwrites caller
  ids.
- **New binding precedents:** TanStack navigation helpers need explicit
  `null` clear sentinels — passing `undefined` re-triggers
  `param = currentRoute` defaults and silently no-ops (the access-keys drawer
  bug); watch→effect conversions express guarded draft sync as render-phase
  adjustment plus queued post-commit emits.
- **Evidence:** astryx e2e 126/126, codec unit tests 25/25 (`logs-route` 13 +
  `monitor-route` 12, added in final review to close a proof gap), real
  linux-binary CSP 30/30 across bundled Chromium, Chrome 125, and system
  Chrome covering all newly flagged routes, tsc×3 + eslint clean, Go
  frontend-selection tests green. Commits: `3a019b90`, `781ef55b`,
  `ef97aaa7`, `ac540c16`, `2fdd51c6`.

Owner doc: `.docs/tech/astryx-migration-plan.md` (Phase 3 outcome). Phase 4
(`groups` + `import`, the highest-coupling domains) follows the same plan.

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
