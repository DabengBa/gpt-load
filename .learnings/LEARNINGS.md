# Project Learnings

## [LRN-20261107-001] best_practice

**Logged**: 2026-11-07T10:24:10+08:00
**Priority**: medium
**Status**: pending
**Area**: tests

### Summary
Raw capture retention changes require one contract check across expiry, cleanup observability, and persisted child rows.

### Details
The final review found that operational documentation used `last_sweep_at` while the management JSON contract exposes `last_sweep_at_ms` and `last_failure_at_ms`. It also found that saying a capture is retained for 24 hours can imply physical deletion at that boundary, even though cleanup is asynchronous and database pages, backups, and exported archives are not securely erased. The regression test needed to wait for both parent/child row deletion and the cleanup counters because the cleanup worker updates counters after the database transaction returns.

### Suggested Action
When changing raw-capture lifecycle behavior, verify wire field names against response structs and contract tests, describe expiry separately from best-effort deletion, and make asynchronous tests wait for both durable state and post-cleanup observability state.

### Metadata
- Source: user_feedback
- Related Files: `.docs/tech/debug-communications.md`, `internal/control/health.go`, `internal/debugcapture/runtime.go`, `internal/debugcapture/runtime_test.go`
- Tags: debug-capture, retention, observability, async-cleanup, testing
- Pattern-Key: debugcapture.expiry-observability-contract
- Recurrence-Count: 1
- First-Seen: 2026-11-07
- Last-Seen: 2026-11-07
- See Also: none

### Context Maintenance
- Decay-Type: reality
- Decay-Rate: medium (health field names and cleanup lifecycle may change with the API or storage implementation)
- Authority: current response structs, contract tests, runtime code; Owner: debug-capture technical owner
- Revalidation: run the debug-capture tests and docs check after lifecycle or health-contract changes; recheck when field names, retention, or cleanup scheduling changes
- Validation: verified; Last-Validated: 2026-11-07; Evidence: focused Go tests, race test, docs check, and final review
- Disposition: retain; Application: applied

---

## [LRN-20261105-001] best_practice

**Logged**: 2026-11-05T02:55:00Z
**Priority**: low
**Status**: pending
**Area**: infra

### Summary
Read the installed `aislop` command help before selecting a change-scope flag.

### Details
The repository's `aislop` version is `0.20.0`. Its supported native change
scope is `--changes --base <ref>` for both `aislop scan` and `aislop ci`.
Selecting the range from the current help output avoids assuming a flag name
or comparison semantics that may have changed between tool versions. When the
branch HEAD equals the target base and the PR changes are still only in the
worktree, temporarily staging those tracked changes ensures the Git-based
`--changes --base` scope includes them; reset the index immediately afterward.
This was validated against the `dev` integration base with both commands.

### Suggested Action
Before a future `aislop` change review, run `aislop --version`,
`aislop scan --help`, and `aislop ci --help`; then use the installed command's
documented scope flags and record the base ref in the review result. If the
changes are unstaged and the branch has no commit delta from the base, stage
only the intended files for the scan and restore the original index state.

### Metadata
- Source: user_feedback
- Related Files: .learnings/LEARNINGS.md
- Tags: aislop, change-scope, tooling, validation
- Pattern-Key: tooling.scope-flag-from-help
- Recurrence-Count: 2
- First-Seen: 2026-11-05
- Last-Seen: 2026-11-07
- Promotion: not eligible; recurrence is 1 and there is no evidence from 2 tasks within 30 days

### Context Maintenance
- Decay-Type: dependency
- Decay-Rate: fast (the installed `aislop` version or supported flags may change)
- Authority: `aislop 0.20.0` `--help` output; Owner: unknown
- Revalidation: run `aislop --version`, `aislop scan --help`, and `aislop ci --help` before each use; if the version or scope flags differ, update this learning and use the newly documented form; if the command is unavailable, report the missing tool and do not infer its flags
- Validation: verified; Last-Validated: 2026-11-07; Evidence: `aislop scan --changes --base dev --json .` and `aislop ci --changes --base dev --format json .` completed successfully after temporarily staging the 7 intended worktree files; the index was reset afterward
- Disposition: retain; Application: applied

---

## [LRN-20261107-002] best_practice

**Logged**: 2026-11-07T11:38:00Z
**Priority**: medium
**Status**: pending
**Area**: frontend

### Summary
Validate route-scoped localization and optional response fields, then verify mobile layout in a real browser.

### Details
The groups route crashed when its collection component requested a message from
the monitor catalog, because route chunks load only their registered catalogs.
An announcement source projection also rejected an older response and a test
fixture when the newly added diagnostic field was absent. Separately, compact
CSS did not prove that a mobile save bar stopped covering the form: actual
375px browser geometry showed the sticky bar still overlapped content until it
was moved into document flow. Hiding a wrapping helper note reduced its height
from about 152px to 58px.

### Suggested Action
When adding localized UI, use a message in the owning feature's route catalog
and exercise the actual route. When extending response projections, default
optional fields at the compatibility boundary and test old payloads. For
responsive layout changes, inspect a real rendered mobile viewport and element
geometry instead of relying only on styles or mocked route tests.

### Metadata
- Source: task_observation
- Related Files: `web/src/frontends/astryx/features/groups/GroupCollectionTable.tsx`, `web/src/shared/control/resources/announcements.ts`, `web/src/frontends/astryx/components/StickySaveBar.tsx`
- Tags: i18n, compatibility, responsive-layout, browser-verification
- Pattern-Key: frontend.route-catalog-compatibility-and-rendered-layout
- Recurrence-Count: 1
- First-Seen: 2026-11-07
- Last-Seen: 2026-11-07
- Promotion: not eligible; one task only and below the recurrence threshold

### Context Maintenance
- Decay-Type: mixed
- Decay-Rate: medium (route catalog ownership and response contracts can change)
- Authority: current route catalog registration, API projection contracts, and browser-rendered layout; Owner: feature maintainers
- Revalidation: on each related change, inspect route-specific catalog wiring and response optionality; repeat mobile browser geometry checks when layout or sticky positioning changes
- Validation: verified; Last-Validated: 2026-11-07; Evidence: Astryx route tests, older announcement source fixture compatibility, and Chromium 156 at 375×812
- Disposition: retain; Application: applied

---

## [LRN-20261105-002] best_practice

**Logged**: 2026-11-05T05:32:00Z
**Priority**: low
**Status**: pending
**Area**: frontend

### Summary
Inspect Knip's report structure and verify consumers before removing reported symbols.

### Details
Knip 5.88.1 JSON reports `issues` as an array of per-file records, not an
iterable map. Reports may include dynamic entry points and exports used inside
their own module. A clean archive scan also differed from the live worktree,
so compare reports from the same command and environment and disclose limits.

### Suggested Action
Inspect the installed version's JSON output, compare findings by file, kind
and name, and check dynamic and internal consumers before deleting code.

### Metadata
- Source: error
- Related Files: web/package.json
- Tags: knip, dead-code, tooling
- Pattern-Key: tooling.knip-report-review
- Recurrence-Count: 1
- First-Seen: 2026-11-05
- Last-Seen: 2026-11-05
- Promotion: not eligible; one task only

### Context Maintenance
- Decay-Type: dependency
- Decay-Rate: fast (tool output and project entry points can change)
- Authority: Knip 5.88.1 reports and current source; Owner: unknown
- Revalidation: inspect current report shape and consumers on each audit; if evidence is unavailable, retain candidates rather than deleting them
- Validation: verified; Last-Validated: 2026-11-05; Evidence: paired live-worktree reports and source review
- Disposition: retain; Application: applied

---

## [LRN-20261107-003] best_practice

**Logged**: 2026-11-07T21:56:00Z
**Priority**: medium
**Status**: pending
**Area**: frontend

### Summary
After an uncertain write to a shared settings surface, re-read the server and
merge per stable server identity (`entry_id`) so untouched remote fields survive
a retry.

### Details
The group model editor sends reasoning/enabled edits together with the whole
model list, but a failed write may still have committed. The editor therefore
refetches and three-way merges instead of discarding either side:
`web/src/frontends/astryx/features/groups/models/GroupModelsTab.tsx:624-649`
keys the previous saved rows and the refreshed remote rows by `entry_id`, reuses
the previous draft `key` for rows whose `entry_id` matches
(`web/src/frontends/astryx/features/groups/models/GroupModelsTab.tsx:631-635`), and then takes the remote value only for fields
where the draft still equals the old baseline:
`reasoning_effort: (item.reasoning_effort ?? '') === (original.reasoning_effort ?? '') ? remote.reasoning_effort : item.reasoning_effort`
(same shape for `enabled`). The retry payload then comes from
`normalizedModels(draft, saved)` at `web/src/frontends/astryx/features/groups/models/GroupModelsTab.tsx:702`, which diffs against
the refreshed baseline (`web/src/shared/domain/groups/models/model-diff.ts:51-63`)
and omits unedited fields, so the retry cannot clobber remote state.

The non-obvious coupling: the diff at `web/src/shared/domain/groups/models/model-diff.ts:53` looks the baseline up by
draft `key`, so the merge is only correct because the refresh reuses `key` via
`entry_id`. Reassigning keys on refresh would silently compare against the wrong
row and resend stale fields. Rows without `entry_id` (new, unsaved rows) have no
remote counterpart and keep their local values.

### Suggested Action
When an editor performs a whole-collection write of shared fields, treat every
failure as "possibly committed": refetch, key both sides by the server's stable
identity, keep user edits, adopt remote values for untouched fields, and rebuild
the save payload as a diff against the refreshed baseline. Cover both directions
(edited field retried, untouched field preserved) with a test.

### Metadata
- Source: conversation
- Related Files: web/src/frontends/astryx/features/groups/models/GroupModelsTab.tsx, web/src/shared/domain/groups/models/model-diff.ts, web/scripts/group-model-schedule-draft.test.ts, web/e2e/astryx-group-model-schedule.spec.ts
- Tags: optimistic-update, three-way-merge, entry_id, draft-editor, failed-write
- Pattern-Key: frontend.uncertain-write-entry-id-merge
- Recurrence-Count: 1
- First-Seen: 2026-11-07
- Last-Seen: 2026-11-07
- Promotion: not eligible; recurrence is 1 within a single task

### Context Maintenance
- Decay-Type: reality
- Decay-Rate: medium (the merge lives in one editor; the pattern stays valid while failed writes can commit, but these files are actively refactored)
- Authority: current source in this branch plus its regression tests; Owner: frontend/group editor maintainers (inferred from file ownership, not asserted)
- Revalidation: re-read `web/src/frontends/astryx/features/groups/models/GroupModelsTab.tsx:624-649` and `web/src/shared/domain/groups/models/model-diff.ts:51-63` and run `pnpm test` (`web/scripts/group-model-schedule-draft.test.ts`) when either file changes; invalidating triggers are a refresh that stops reusing `key`/`entry_id`, or a save path that no longer diffs against a refreshed baseline; if the source is unavailable, treat the rule as unverified rather than assuming the merge holds
- Validation: verified (unit level); Last-Validated: 2026-11-07; Evidence: `node --import ./scripts/register-typescript.mjs --import ./scripts/register-shared-alias.mjs --test scripts/group-model-schedule-draft.test.ts` passed 4/4, including "replacement sends only schedule fields changed relative to baseline, including new rows"; the recovery-retains-remote E2E case in `web/e2e/astryx-group-model-schedule.spec.ts:99-125` was read but not executed here
- Disposition: retain; Application: applied

---
