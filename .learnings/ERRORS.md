# Project Errors

## [ERR-20261107-001] debugcapture_test_insertion

**Logged**: 2026-11-07T10:24:10+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
A patch inserted a new runtime test inside an existing function and caused a Go syntax error.

### Error
```text
internal/debugcapture/runtime_test.go:131:83: missing ',' before newline in argument list
```

### Context
- Added the disabled-runtime raw-chunk cleanup test near `TestRuntimeDisabledStillCleansExpiredSessions`.
- The existing `StartSession` call was left without its closing `})`, so the new function declaration appeared before the prior function was closed.
- The focused package test failed during setup before executing behavior.

### Suggested Fix
After inserting a test beside an existing function, inspect the surrounding function boundary and run the focused package compile/test immediately before adding more assertions.

### Metadata
- Reproducible: yes
- Related Files: `internal/debugcapture/runtime_test.go`
- See Also: none

### Resolution
- **Resolved**: 2026-11-07T09:43:33+08:00
- **Commit/PR**: pending
- **Notes**: Restored the missing function boundary, ran gofmt, and the intended red test then failed on the pre-fix cleanup behavior; subsequent focused and race tests passed.

---

## [ERR-20261105-001] knip_report_comparison

**Logged**: 2026-11-05T05:32:00Z
**Priority**: low
**Status**: resolved
**Area**: frontend

### Summary
The initial Knip report comparator assumed the wrong JSON structure.

### Error
```text
TypeError: object is not iterable (cannot read property Symbol(Symbol.iterator))
```

### Context
- Comparing Knip 5.88.1 JSON reports; `issues` contains per-file records.

### Suggested Fix
Inspect the JSON first, then iterate each record and its finding arrays.

### Metadata
- Reproducible: yes
- Related Files: web/package.json
- See Also: LRN-20261105-002

### Resolution
- **Resolved**: 2026-11-05T05:32:00Z
- **Commit/PR**: not applicable
- **Notes**: Corrected comparator produced the paired report delta.

---

## [ERR-20261105-002] shell_inspection_commands

**Logged**: 2026-11-05T05:32:00Z
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
Shell quoting and an option-like printf format broke inspection commands.

### Error
```text
SyntaxError: Unexpected token '.'
printf: --: invalid option
```

### Context
- A Node package-script inspection had incorrect shell quoting; a rule inventory used a printf format beginning with dashes.

### Suggested Fix
Use a safely quoted JSON reader and `printf --` for option-like formats.

### Metadata
- Reproducible: yes
- Related Files: web/package.json

### Resolution
- **Resolved**: 2026-11-05T05:32:00Z
- **Commit/PR**: not applicable
- **Notes**: Retried package inspection with Python and inspected the rules successfully.

---
