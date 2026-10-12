---
description: "How successful requests learn, take over, and preserve credential affinity bindings, including the health-bypass takeover boundary and its single-process concurrency guarantee."
kind: technical
topic: affinity-binding
code:
  paths:
    - internal/affinity/cache.go
    - internal/gateway/affinity.go
    - internal/gateway/attempts.go
    - internal/gateway/websocket_turn.go
    - internal/scheduler/scheduler.go
---
# Affinity Binding

## Scope

- **Owner:** credential affinity binding semantics: observation, binding
  writes, takeover of health-unavailable bindings, cache CAS and eviction
  behavior, and the request success boundary that gates learning.
- **Authoritative for:** when a request may write a binding, when a healthy
  successful fallback may replace an existing binding, which results never
  learn a binding, and what concurrency guarantee the cache CAS provides.
- **Excludes:** provider attempt performance classification and feedback
  status, owned by [Provider Feedback](provider-feedback.md); durable request
  log persistence; scheduler selection order and weighting (this document only
  owns the health-bypass observation the scheduler exposes).

## Health Bypass And Takeover

Health exclusion is observed by the scheduler, not inferred from the selection.
`internal/scheduler/scheduler.go:Iterator.PreferredHealthBypassed` reports
whether the last `Next()` selected a different credential because runtime
health excluded every otherwise eligible, positively weighted route of the
preferred affinity credential. The observation belongs to the request's
iterator, resets before every `Next()` call including exhaustion, and is set
only when a different credential is actually selected. It requires a positive
`Snapshot()` observation of the preferred credential (auth-ready, runtime
health known) and respects all request-specific gates: allowed credential IDs,
skipped groups, static route pools, configured cardinality, entry weights, and
already-tried pairs. Any otherwise eligible healthy route of the preferred
credential prevents the observation. Sources without a positive snapshot
observation never report a bypass.

The gateway reads the observation immediately after `Next()` and retains the
positive evidence for the request (once true it stays true across later
attempts). A healthy successful fallback may take over an existing binding
when the original binding's credential was health-bypassed. The takeover is
validated at write time: the selected credential identity (ID, group,
identity generation), its current auth readiness, its current runtime
availability, and the selected entry's current availability are checked under
the existing binding's stripe lock immediately before the cache CAS. A
credential that recovered between selection and write is therefore allowed to
write; the captured health-bypass permission is not revoked by recovery during
the flight.

The health-bypass observation is qualified by the frozen bound identity:
positive health evidence counts only when the runtime view supplying it
matches the bound target's group, credential ID, and identity generation.
A same-ID replacement in cooldown cannot authorize migration of the original
binding. Identity generation is distinct from the credential version used for
ordinary secret refresh.

## Request-Specific Exclusions

Bindings are never learned for requests that carry `previous_response_id`.
Many other standalone requests do not carry that field while resolving to the
same affinity key; those requests participate normally and may learn or
refresh the binding. The scheduler's request-specific gates (allowed IDs,
skipped groups, tried pairs, cardinality) also suppress the bypass
observation, so a restricted request cannot take over a binding that ordinary
scheduling of the same key would reach.

## Healthy Final Success Boundary

Both HTTP and WebSocket paths learn bindings only on a healthy final result,
decided by the same gate (`affinityResultSuccessful`): no error and no
execution error, not a local or not-sent dispatch, not an error before commit,
2xx status code, final feedback status not faulty, and a final health decision
of category OK with upstream origin. Failures and faulty performance never
learn bindings even when the response was successfully delivered to the
client. Successful results classified normal, slow, or unassessed still learn;
a faulty-classified result does not.

WebSocket turns record affinity only after `ExecuteTurn` returns and the final
feedback/error validation passes, on a committed, clean, completed turn — the
callback that previously recorded inside the stream before the final result
was removed. A terminal delivery followed by a late error no longer writes a
binding.

## Concurrency: Original-Observation CAS

Binding writes use the existing cache CAS (`RecordSuccess`) under a per-key
stripe lock. Each request holds the observation from its own lookup; the first
completion wins and later completions of older observations are rejected by the
version-and-target comparison. This gives the C-before-B guarantee: when C and
B both hold a found observation of A and C completes first, C's binding stands
and B's stale completion cannot overwrite it.

Two gaps are explicitly closed and one is explicitly out of scope:

- **Eviction/expiry guard:** a stale *found* observation whose key is no
  longer resident at CAS time (LRU-evicted or TTL-expired) is rejected, never
  reinserted, because a newer binding may exist outside the snapshot. Miss
  inserts (`found == false`) are unchanged.
- **Missing found observation:** a request bound to an existing target whose
  original lookup observation is missing (`boundTarget` valid but
  `observation` not found) cannot write affinity at all, closing the
  read-through eviction gap on the gateway side.
- **Out of scope:** the guarantee is one Handler/process with the in-memory
  cache and per-key stripe lock. The durable store `Upsert` does not implement
  a distributed CAS, and there is no serialization of an initial miss across
  an eviction window. No new schema, configuration, or distributed protocol is
  introduced.

## Recovery Semantics

Credential recovery during an in-flight request does not retroactively change
that request's behavior: a health-bypassed fallback that completes
successfully takes over the binding even if the preferred credential recovered
in the meantime, because the write-time check only requires the selected
credential to be currently available. The captured health-bypass permission is
an observation frozen at selection time and is retained regardless of later
health changes to the preferred credential. The selected fallback target is
different: it must satisfy live availability at write time — matching
identity, auth readiness, and current credential and entry health. A selected
credential that became unavailable after selection is rejected at write time.

## Binding Points

- Code binding: `internal/scheduler/scheduler.go:Iterator.PreferredHealthBypassed`
- Code binding: `internal/gateway/affinity.go:requestAffinity.mayRecordSuccess`
- Code binding: `internal/gateway/affinity.go:affinityResultSuccessful`
- Code binding: `internal/gateway/affinity.go:Handler.affinitySelectionAvailable`
- Code binding: `internal/affinity/cache.go:Cache.RecordSuccess`

## Testing

Regression coverage lives in `internal/scheduler/preferred_health_test.go`
(observation contract), `internal/gateway/affinity_takeover_test.go` (HTTP/WS
takeover, exclusions, final-success boundary, C-before-B concurrency,
durable preservation), and `internal/affinity/cache_test.go`
(stale found observation does not reinsert).
