---
description: Race detection scope and its third-party SQLite coverage trade-off in pull request and release workflows
kind: adr
topic: ci-race-testing
id: ADR-0005
status: active
supersedes: null
created_at: 2026-11-02
affects_symbols:
  - internal/webui/workflow_test.go#TestBranchAndReleaseWorkflowsRunRaceInParallelGates
---
# ADR-0005: Keep the pull-request race exception for third-party SQLite code while release stays fully instrumented

## Context

SQLite-heavy tests made the fully instrumented pull-request race job expensive,
while the release workflow still needs complete race coverage.

## Decision

The pull-request job keeps `-race` for project code and disables instrumentation
for `modernc.org/...` dependencies. The release job keeps the fully instrumented
command. The exception is scoped to pull-request cost and does not change
release builds or application runtime behavior.

## Rejected

- Removing race coverage entirely: faster, but it hides project-code races.
- Applying the dependency exception to releases: cheaper, but it weakens the
  artifact that users receive.

## Constraint

This trade-off accepts that dependency-only races can be missed in pull
requests; release coverage remains the backstop. `TestBranchAndReleaseWorkflowsRunRaceInParallelGates`
pins both workflow commands independently.
