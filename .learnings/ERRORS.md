# Project Errors

## [ERR-20261107-001] route_scoped_catalog_and_optional_projection

**Logged**: 2026-11-07T11:38:00Z
**Priority**: medium
**Status**: resolved
**Area**: frontend

### Summary
Groups route failed after a scroll-hint message was added to the monitor catalog.

### Error
```text
MissingTranslationError: [@formatjs/intl Error MISSING_TRANSLATION] Missing message: "monitor.scrollHint" for locale "zh-CN", using id as fallback.
```

### Context
- The collection table renders on `/groups`, whose route-scoped catalog does
  not include the monitor catalog. A separate announcement source test fixture
  also omitted the newly optional `last_error`, and the projection initially
  treated absence as a malformed response.

### Suggested Fix
Use the owning `groups.collection` message key and align all supported locale
catalogs. Default the optional `last_error` field to an empty string at the
projection boundary, preserving older API payloads and fixtures.

### Metadata
- Reproducible: yes
- Related Files: `web/src/frontends/astryx/features/groups/GroupCollectionTable.tsx`, `web/src/shared/control/resources/announcements.ts`
- See Also: LRN-20261107-001

### Resolution
- **Resolved**: 2026-11-07T11:38:00Z
- **Commit/PR**: pending
- **Notes**: Added `groups.collection.scrollHint` in all three catalogs and
  backward-compatible projection default; 19 announcement/site-info browser
  tests passed.

---

## [ERR-20261107-002] assumed_command_targets

**Logged**: 2026-11-07T11:38:00Z
**Priority**: low
**Status**: resolved
**Area**: tooling

### Summary
Several verification commands used an incorrect script, build path or browser project name.

### Error
```text
pnpm run verify-astryx-i18n: missing script
go build ./cmd/gpt-load: package path does not exist
Playwright project "chromium": project not found
```

### Context
- The repository uses `verify:astryx-i18n`; the Go binary entry point is the
  repository root (`go build .`); and the relevant Playwright project is
  `astryx`. These are repository-specific command contracts.

### Suggested Fix
Inspect `package.json`, Go module entry points and Playwright config before
choosing command names or targets; do not infer conventional paths.

### Metadata
- Reproducible: yes
- Related Files: `web/package.json`, `web/playwright.config.ts`, `main.go`
- See Also: none

### Resolution
- **Resolved**: 2026-11-07T11:38:00Z
- **Commit/PR**: pending
- **Notes**: Retried with the repository's actual script, root build target and
  `astryx` project. These failures did not alter source files.

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
