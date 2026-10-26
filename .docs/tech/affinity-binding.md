---
description: "How successful requests learn, take over, and preserve exact upstream-route affinity bindings, including the health-bypass takeover boundary and its single-process concurrency guarantee."
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

- **Owner:** affinity binding semantics: the remembered target identity,
  observation, binding writes, takeover of health-unavailable bindings, cache
  CAS and eviction behavior, durable recovery, and the request success
  boundary that gates learning.
- **Authoritative for:** what a binding remembers, when a request may write a
  binding, when a healthy successful fallback may replace an existing binding,
  which results never learn a binding, and what concurrency guarantee the
  cache CAS provides.
- **Excludes:** provider attempt performance classification and feedback
  status, owned by [Provider Feedback](provider-feedback.md); durable request
  log persistence; scheduler selection order and weighting (this document only
  owns the health-bypass observation the scheduler exposes).

## Target Identity

A binding remembers the exact upstream route that a provider attempt actually
used, not just the credential that owned it:

- group ID;
- credential ID;
- credential identity generation;
- route entry ID;
- upstream model ID (an empty value is legitimate for operations that address
  no model, such as a Responses resource read).

Entry ID is mandatory. A target without a route entry is invalid and is never
written. The whole target value is the comparison unit: the scheduler prefers a
route only when group, credential, identity generation, entry and model all
match, and capacity overflow for a bound request is granted only when the
resolved route equals the binding.

No-model Responses routes use the group-level `resource` entry identity with
an empty upstream model. They do not borrow an arbitrary configured model's
entry. A resource or extension request with an extracted affinity signal can
therefore learn and recover its successful route, including after hot-cache
eviction; requests without a signal do not learn a binding.

Route identity is what makes a same-credential sibling model reachable: a
credential may serve several entries and upstream models behind one public
alias, and the binding must select the one that succeeded rather than whichever
model sorts first. The scheduler's ordinary candidate ordering is unchanged
for every other request.

`affinity_hit` is a request-log observation of the first attempt, not a
separate binding rule: it is recorded only when that attempt matches the
bound group, credential, entry and upstream model. A same-credential sibling
model is therefore a miss for this field even though it inherited the
credential. The flag is monotone — once a request starts on the bound route it
stays set after a later takeover fallback in the same request.

The credential-only `resolved_affinity_credential_id` diagnostic is unchanged:
it still records the credential that the binding resolved to, independent of
whether the first attempt actually used the bound route.

## Health Bypass And Takeover

Health exclusion is observed by the scheduler, not inferred from the selection.
`internal/scheduler/scheduler.go:Iterator.PreferredHealthBypassed` reports
whether the last `Next()` selected a different route because runtime health
excluded the otherwise eligible, positively weighted bound route.

The observation is route-scoped: it reports exclusion of the exact bound route,
so an unavailable bound entry is
recognized as a bypass even when the same credential still has a healthy
sibling model. Removing an entry or changing its model is a configuration
invalidation, not a health bypass: the gateway clears the resolved target while
retaining the original cache observation, allowing the next healthy success to
replace the stale binding through the same CAS.

The observation belongs to the request's iterator, resets before every `Next()`
call including exhaustion, and is set only when the bound route is not the
selected one. It requires a positive `Snapshot()` observation of the preferred
credential (auth-ready, runtime health known) and continues to respect every
request-specific gate: allowed credential IDs, skipped groups, static route
pools, configured cardinality, entry weights, and already-tried pairs. A
healthy bound route prevents the observation. Sources without a positive
snapshot observation never report a bypass, and filtering, disabled groups,
capacity denial and downstream failures never produce one.

The gateway reads the observation immediately after `Next()` and retains the
positive evidence for the request (once true it stays true across later
attempts). A healthy successful fallback may take over an existing binding
when the bound route was health-bypassed, including when the fallback is
another entry or upstream model of the same credential. The takeover is
validated at write time by `Handler.affinitySelectionAvailable`: the selected
credential identity (ID, group, identity generation), its current auth
readiness, its current runtime availability, and the selected entry's current
availability are checked under the existing binding's stripe lock immediately
before the cache CAS. A
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

Dispatch-row capacity admission is owned by [Schedule concurrency](schedule-concurrency.md). Capacity overflow does not itself qualify as a health bypass or allow a binding takeover; only the valid binding and health observations defined below can do so.

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
version-and-target comparison, where the compared target is now the full route
identity. This gives the C-before-B guarantee: when C and B both hold a found
observation of A and C completes first, C's binding stands and B's stale
completion cannot overwrite it.

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

Bindings persist in `system_settings` under the existing affinity key prefix
and carry the full route identity, so a recovered binding still names the entry
and upstream model that succeeded. There is no compatibility surface for
earlier credential-only records: a stored row without a route entry is read as
a plain miss (never an error, never a delete) and the next success replaces it.
Corrupted JSON and rows with a missing credential identity still fail closed, so
an unreadable durable authority is never mistaken for "no binding". No schema,
column, or migration is added.

A required durable lookup/decode failure is returned as HTTP 503
`affinity_store_unavailable` with `Retry-After: 1`; WebSocket turns return the
same error code. It is not reported as `configuration_changed`, because a
storage read failure does not establish that configuration changed. The data
plane emits a rate-limited `affinity_binding_lookup_failed` diagnostic with the
request ID, lookup phase, masked affinity key and underlying error. It never
includes the raw client cache key or stored binding payload. Both transports
remain fail closed and dispatch no provider attempt.

Route changes are reconciled against the published execution route catalog,
scoped to the key's client protocol, operation and client model, rather than a
stored deletion list. When the bound entry is removed or its upstream model
changes, or the entry no longer serves the key's client model or test alias,
affinity reports `target_unavailable`, schedules normally, and lets
the next healthy success rewrite the row. The entry ID alone cannot preserve a
binding across a model change. Disabled or zero-weight entries still exist;
these exclusions do not authorize a takeover. Request-specific restrictions
also preserve the original binding.

`target_unavailable` is also the state reported when the bound group is no
longer present in the snapshot, or when the stored credential identity is no
longer in the request's allowed credential set with a matching group and
identity generation. Those cases leave the durable row untouched for the same
reason: losing a temporary permission or an in-flight config revision must not
destroy a binding that would be valid again.

Credential recovery during an in-flight request does not retroactively change
that request's behavior: a health-bypassed fallback that completes successfully
takes over the binding even if the preferred credential recovered in the
meantime, because the write-time check only requires the selected credential to
be currently available. The captured health-bypass permission is an observation
frozen at selection time and is retained regardless of later health changes to
the preferred credential. The selected fallback target is different: it must
satisfy live availability at write time — matching identity, auth readiness,
and current credential and entry health. A selected credential that became
unavailable after selection is rejected at write time.

## Binding Points

- Code binding: `internal/scheduler/scheduler.go:Iterator.PreferredHealthBypassed`
- Code binding: `internal/scheduler/scheduler.go:Iterator.isPreferredRoute`
- Code binding: `internal/gateway/affinity.go:requestAffinity.mayRecordSuccess`
- Code binding: `internal/gateway/affinity.go:affinityResultSuccessful`
- Code binding: `internal/gateway/affinity.go:Handler.affinitySelectionAvailable`
- Code binding: `internal/gateway/request_log.go:requestRecorder.observeAffinityHitComparison`
- Code binding: `internal/affinity/cache.go:Cache.RecordSuccess`
- Code binding: `internal/storage/affinity_store.go:decodeAffinityBinding`

## Testing

Regression coverage lives in `internal/scheduler/preferred_health_test.go` and
`internal/scheduler/preferred_route_test.go` (observation and exact-route
selection contract), `internal/gateway/affinity_route_test.go` (same-credential
two-model learning over HTTP and the WebSocket turn, health takeover,
`affinity_hit` route matching, and real compiled no-model extension routes
across durable recovery), `internal/gateway/affinity_takeover_test.go`
(takeover, exclusions, final-success boundary, C-before-B concurrency, durable
preservation), `internal/storage/affinity_store_route_test.go` (route round
trip, obsolete credential-only payload as a miss), and
`internal/affinity/cache_test.go` / `internal/affinity/target_test.go`
(target validity, stale found observation does not reinsert).
