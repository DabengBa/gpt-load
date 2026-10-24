---
description: Astryx cutover decision — ship Phase 5 and Phase 6 together and give up the one-release cookie fallback window
kind: adr
topic: astryx-migration
status: accepted
date: 2026-09-30
---

# ADR-0003: Combine cutover and deletion into one delivery, leaving a zero-length cookie fallback window

## Context

The migration plan (`.docs/tech/astryx-migration-plan.md`) originally split the
close-out into two phases one release apart:

- Phase 5: flip the default document to Astryx while the
  `gpt-load.frontend=classic` cookie still allowed an explicit rollback;
- Phase 6: delete the classic frontend and all coexistence infrastructure one
  release later.

The cookie window existed to give operations an escape hatch — a second-level
rollback without changing the deployed version — for a fatal regression found
after the switch.

After Phase 4 closed, the user instructed "continue with phase 5 and phase 6"
and, when asked to clarify, chose a **single combined delivery**. A middle
release that switches without deleting therefore does not exist.

## Decision

- Phase 5 and Phase 6 merged into one spec
  (`260930-01-astryx-cutover-removal`), delivering the single-document end state
  directly: the Go server returns the same Astryx `index.html` for every page
  route and every unmatched path, never reads the `gpt-load.frontend` cookie,
  and never emits `Vary: Cookie`.
- The selection mechanism is **deleted** rather than defaulted: the cookie
  contract module, the verification script, the Interface segment of the
  preferences panel, `isAstryxNavigable`, the manifest `astryx` field (illegal
  from v3), the dev-selector middleware, and the classic Playwright project are
  all removed.
- **Rollback degrades to version rollback**: a fatal regression found after the
  switch is recovered by rolling back the deployed version, not by an operator
  writing a cookie.

## Trade-offs

| Option | Rollback cost | Coexistence cost | Outcome |
|---|---|---|---|
| Two phases with a cookie window (original design) | seconds (write a cookie) | two frontends plus selection infrastructure maintained for another release | rejected (the user required a combined delivery) |
| Combined delivery (chosen) | minutes (version rollback) | cleared in the same release | **chosen** |
| Flip the default only, keep the mechanism | seconds | coexistence code remains forever | rejected (deletion would be deferred into permanent debt) |

## Consequences

- Positive: the classic tree (2.4 MB, 164 `.vue` files), the Vue/Tailwind
  toolchain, 12 dependencies, the dual-entry build, and the document-selection
  middleware are all cleared in the same release, so the migration leaves no
  tail behind.
- Cost: the switch is immediate and uncanaried. Mitigation: before the manifest
  v2→v3 change, all 11 routes had passed the astryx e2e suite and the real-binary
  CSP matrix, and routing, authentication, and localization semantics were
  pinned by the Phase 0–4 parity tests.
- Irreversibility: reintroducing the classic frontend after release means
  rolling back the version or rebuilding the code, which is an explicitly
  accepted trade-off.
