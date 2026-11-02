---
description: Race detection scope and its third-party SQLite coverage trade-off in pull request and release workflows.
kind: technical
topic: ci-race-testing
code:
  paths:
    - .github/workflows/ci.yml
    - .github/workflows/release.yml
    - internal/webui/workflow_test.go
---

# CI Race Testing

## Responsibility

The race jobs exercise the Go application and internal packages with the Go race detector. The pull request and release workflows maintain separate commands and cache scopes.

## Scope And Constraints

The pull request race job disables race instrumentation for `modernc.org/...` dependencies while keeping `-race` enabled for project code. This reduces the cost of SQLite-heavy tests but can miss races involving uninstrumented dependency code. The release race job retains its existing fully instrumented command; this exception does not alter release builds or application runtime behavior.

## Implementation And Verification

The pull request job runs `go test -race -gcflags='modernc.org/...=-race=false' -count=1 -timeout=15m . ./internal/...`. The release job runs the same package set without that compiler exception. `TestBranchAndReleaseWorkflowsRunRaceInParallelGates` checks both workflow commands independently.
