---
description: "Classify upstream insufficient-balance failures as a distinct billing category, cool a credential only after consecutive billing failures using the published blacklist_release_seconds window, and surface it as a Home attention alert."
kind: technical
topic: failure-classification
code:
  paths:
    - internal/execution/contracts.go
    - internal/channel/failure_class.go
    - internal/health/execution_judge.go
    - internal/state/registry.go
    - internal/gateway/handler.go
    - internal/storage/migrations/0023_billing_failure_category.go
    - web/src/shared/domain/home/attention.ts
    - web/src/frontends/astryx/features/home/HomeAttention.tsx
---

# Billing Failure Classification And Home Attention

> Current scope: built-in subscriptions and the embedded CPA adapter have been
> removed. The CPA provider fixes in the older review below are history only and
> describe neither a current dependency nor a current execution path. Billing
> classification for the standard API and the Home attention alert remain.

## Current Implementation And Verification Surface

This section locates the current implementation and the reusable verification
entry points in today's source. It does not claim that these tests were run now.
The pass records, Windows limits, and Vue toolchain in the older review below
belong to that environment and are not evidence for the current React/shared
implementation.

| Boundary | Current implementation | Current verification entry and status |
|---|---|---|
| Billing signal classification | `internal/execution/contracts.go`, `internal/channel/failure_class.go` | `internal/execution/contracts_test.go`, `internal/channel/failure_class_test.go`; paths checked only, not run here |
| Cooldown parameter and effect application | `internal/health/execution_judge.go` (`DecisionContext.BillingCooldown`), `internal/gateway/handler.go` (reuses `blacklistReleaseDeadline`) | `internal/health/execution_judge_billing_test.go`, `internal/gateway/billing_streak_test.go` |
| Consecutive-two threshold and streak state | `internal/state/registry.go` (`BillingStreak`, `RecordBillingFailure`, `ResetBillingStreak`), `internal/gateway/handler.go` (`billingStreakThreshold`) | `internal/state/weights_test.go`, `internal/gateway/billing_streak_test.go` |
| Committed safety constraint | `internal/health/execution_judge.go` (`constrainCommittedDecision`) | `internal/health/execution_judge_billing_test.go` |
| Billing persistence constraints | `internal/storage/migrations/0023_billing_failure_category.go` | `internal/storage/migrations/0023_billing_failure_category_test.go`; paths checked only, not run here |
| Home aggregation and display | `web/src/shared/domain/home/attention.ts`, `web/src/frontends/astryx/features/home/HomeAttention.tsx` | the current `web/package.json` `type-check` (`tsc`) and `lint` (ESLint and others) scripts; not run here, and not proof of UI behavior |

The shared aggregation counts cooldown credentials per group and takes the
earliest cooldown end, and the React `HomeAttention.tsx` renders that aggregate.
`vue-tsc` is not a verification entrypoint for the current implementation.

## Cooldown Threshold And Duration (current implementation)

- **Threshold (two consecutive failures):** `health` still only produces the
  intent "switch candidate plus cool the credential" and a deadline; `gateway`
  hands that billing cooldown to the registry's `RecordBillingFailure`, which
  does counting → threshold check → cooldown write → streak reset **inside one
  lock acquisition**, so counting and cooling are atomic and concurrent failures
  produce exactly one real cooldown. Only two **consecutive** billing failures on
  the same credential cool it; the first billing failure only switches candidate
  to save the request and the credential stays schedulable.
- **Interruption:** any success or other outcome ends the streak.
  `applyGroupDecisionEffectForEntry` calls `resetBillingStreakForResult` for every
  non-billing decision (covering the `EffectNone` upstream-4xx and model-scope
  early-return branches), and success goes through `recordCredentialSuccess`; the
  registry's `ClearFailure`/`Recover`/`IncrFailure`/`RecordFailureWithBlacklist`
  and the cooldown writes that actually land also clear the streak.
- **Reset after firing:** starting a cooldown
  (`SetCooldownWithChange`/`SetCooldownWithChangeIfVersion`) or
  `RecordBillingFailure` landing a cooldown ends the streak immediately, so the
  next billing failure after the window starts again from 1.
- **Duration:** `judge` computes the deadline from
  `DecisionContext.BillingCooldown`, and `gateway` fills it with the same
  `handler.blacklistReleaseDuration()` — that is, the system breaker release
  parameter `blacklist_release_seconds`, identical to the blacklist release
  window. The fixed 24h constant is gone. When nothing is injected,
  `normalizeDecisionContext` uses that parameter's factory default
  (`state.DefaultRuntimeSettings().BlacklistReleaseSeconds`) instead of falling
  back to the rate-limit cooldown, which keeps `Decision.Validate`'s requirement
  that a cooldown carry a deadline satisfiable.
- **Concurrency safety:** `RecordBillingFailure` validates the credential version
  (`expectedVersion`, where 0 means unrestricted) before counting, so a stale
  version advances neither the streak nor the cooldown; both counting and cooling
  happen under the registry lock.
- **Retry instruction unchanged:** a billing decision's `Retry`/`RuleID`
  (`billing.insufficient_balance`) and the committed safety constraint
  (`constrainCommittedDecision`) behave as before.

## Historical Review (original review record)

The original review text below is preserved with its proofs, environment limits,
and completion checkmarks. It is a historical record, not a re-verification
result or an open task list. The review carried no date, so no date or historical
success evidence is invented. File line numbers and toolchain names in it only
locate the evidence of that time.

Second reviewer: the Claude Code CLI was unavailable (proxy auth 401) and the
shared script `invoke-claude-readonly-review.ps1` was missing, so only the primary
review ran.

### Confirmed findings, all fixed

- [x] Review: CPA providers wrote hints directly and bypassed `neutralFailureHint` — antigravity `INSUFFICIENT_G1_CREDITS_BALANCE` was classified rate_limited, grok 402 was classified candidate_unavailable, and the claude/codex billing marker was taken by the generic branch first (severity: high; scope: classification chain; evidence: internal/execution/cpa/{antigravity,claude,codex,grok}_provider.go; proof: each provider got an InsufficientBalanceSignal branch ahead of the generic one, and two tests changed verdict and passed)
- [x] Review: the `request_log_attempts` CHECK constraint `chk_request_log_attempt_failure_category` did not include `billing`, so billing attempts were rejected on write (severity: high; scope: persistence; evidence: internal/storage/models/request.go + migrations/0006; proof: migration 0023 rebuilds the table and widens the constraint, with tests covering preserved data, writable billing, still-rejected illegal categories, and idempotency)
- [x] Review: `fixedErrorSummaries` lacked `upstream_insufficient_balance`, so a log without a provider summary decayed to the generic "Upstream request failed." (severity: medium; scope: log observability; evidence: internal/gateway/request_log.go:884; proof: the catalog was completed and linux vet compiled)
- [x] Review: the effect-stripping gate for web-search operations did not exempt billing and would strip the then-fixed 24h credential cooldown (severity: medium; scope: effect application; evidence: internal/gateway/execution_boundary.go:133; proof: billing joined the account-state exceptions at the same level as auth/invalid_key)
- [x] Review: the existing test `generic payment required switches candidate` still asserted the old 402 → http_4xx_retry expectation (severity: low; scope: tests; evidence: execution_judge_v2_test.go; proof: it now asserts billing.insufficient_balance and the health package tests pass)

### Unresolved evidence gaps

- `internal/{storage,requestlog,control,gateway,execution/bifrost,execution/cpa}` could not run tests on that Windows machine (`securefile`/`catalog` have linux-only variants). `GOOS=linux go vet` verified that every related package, test files included, compiled; runtime verification depended on CI or a Linux environment.
- The frontend had no unit-test facility; the proof surface was `vue-tsc` type-check plus eslint, both passing.

### Boundaries that were re-checked

- Billing decision precedes the 401/403/429/4xx status branches (consistent across the channel, bifrost, cpa, and judge layers); `quota_exceeded`/`resource_exhausted`/`usage_limit_reached` still classify as rate_limited (pinned by tests).
- Decisions still pass through `constrainOperationReplay`/`constrainCommittedDecision` after `decisionForExecutionCategory`, so the committed-response safety constraint cannot be bypassed (tested).
- Cooldowns only extend: `SetCooldownWithChange` semantics guarantee a later, shorter rate-limit cooldown cannot shorten a billing window (the window length now comes from `blacklist_release_seconds`).
- Home Attention exposes only group, count, and earliest re-probe time, and never credential identity; `credential_status=cooldown` is a valid route filter value.
- `attentionTotal` counts billing by credential; row caps and summary behavior are unchanged.

- [x] Review complete
