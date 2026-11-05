# Project Learnings

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
or comparison semantics that may have changed between tool versions. This was
validated against the `dev` integration base with both commands.

### Suggested Action
Before a future `aislop` change review, run `aislop --version`,
`aislop scan --help`, and `aislop ci --help`; then use the installed command's
documented scope flags and record the base ref in the review result.

### Metadata
- Source: user_feedback
- Related Files: .learnings/LEARNINGS.md
- Tags: aislop, change-scope, tooling, validation
- Pattern-Key: tooling.scope-flag-from-help
- Recurrence-Count: 1
- First-Seen: 2026-11-05
- Last-Seen: 2026-11-05
- Promotion: not eligible; recurrence is 1 and there is no evidence from 2 tasks within 30 days

### Context Maintenance
- Decay-Type: dependency
- Decay-Rate: fast (the installed `aislop` version or supported flags may change)
- Authority: `aislop 0.20.0` `--help` output; Owner: unknown
- Revalidation: run `aislop --version`, `aislop scan --help`, and `aislop ci --help` before each use; if the version or scope flags differ, update this learning and use the newly documented form; if the command is unavailable, report the missing tool and do not infer its flags
- Validation: verified; Last-Validated: 2026-11-05; Evidence: `aislop scan --changes --base dev --json .` and `aislop ci --changes --base dev --human .` both completed successfully
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
