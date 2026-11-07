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
