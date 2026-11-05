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
