---
description: "Enable and inspect raw Gateway and provider communication captures, including model probes and credential tests, with explicit retention and evidence boundaries."
kind: technical
topic: debug-capture
code:
  paths:
    - internal/debugcapture
    - internal/gateway/capture.go
    - internal/container/debug_capture.go
    - internal/container/debug_capture_unix.go
    - internal/control/probe_capture.go
    - internal/execution/http_observer.go
    - third_party/bifrost-core/providers/utils/http_observer.go
---
# Debug Communication Capture

Debug communication capture is an opt-in operational tool for investigating gateway and provider integration behavior. On Unix runtimes, set `DEBUG_CAPTURE_ENABLED=true` to enable it; it is disabled by default.

## Retention

The capture store uses the existing configured database and remains separate from `request_logs`. Captures are retained for a fixed 24-hour expiry, then startup and periodic cleanup remove expired rows and chunks automatically. Each runtime admits at most 64 concurrent capture sessions; each attempt retains at most 8 MiB of raw body data and 128 pending persistence events. Exceeding a limit fails the affected capture without changing the data-plane response.

The capture can contain complete observed sensitive values, including `Authorization`, `Cookie`, API keys, request bodies, response bodies, and provider credentials passed through the supported observation boundary. Protect `AUTH_KEY`, the database, database backups, and exported archives. Do not enable this feature on an untrusted or shared deployment unless that exposure is intentional.

## Administrator API

These endpoints require the management `AUTH_KEY` and are administrator-only. Data-plane access keys cannot use them.

- `GET /api/debug-captures`
- `GET /api/debug-captures/:capture_id`
- `GET /api/debug-captures/:capture_id/download`

The list endpoint accepts `request_id`, `access_key_id`, `protocol`, `operation`, `state`, `created_after_ms`, `created_before_ms`, `offset`, and `limit`. The default page size is 50 and the maximum is 200; `limit=0`, an unknown `state`, a non-canonical id, or a window whose start is not before its end is rejected. `created_after_ms` is inclusive and `created_before_ms` is exclusive.

`state` accepts `active`, `completed`, `failed`, and `expired`, but the list only reads rows whose `expires_at_ms` is still in the future, ordered by creation time. A session that already expired is not listed under `state=expired`; that value currently returns nothing. Use the structured request log for history beyond the fixed 24-hour retention.

The detail response contains session metadata, lifecycle state, errors, and ordered logical attempts. The download endpoint streams a ZIP containing a JSON manifest, per-attempt metadata, and raw request/response header and body parts. Download responses are marked `no-store`.

The `/api/health` response includes `debug_capture` with enabled/running state, fixed retention seconds, active/completed/failed counts, sweep counters, and timestamps. These state counts describe unexpired database rows, not the runtime's in-memory admission slots. When all 64 slots are occupied, the component reports `error: "session_limit_reached"`, including when expired rows have already disappeared from the counts. The error clears when admission becomes available again. If the optional store cannot provide counts, the component reports `error: "counts_unavailable"` while the broader health response remains available.

Feature-specific capture storage and adapters are Unix-only. Windows retains the public application types and no-op container assembly so the unrelated Windows binary continues to compile; Windows does not install a capture store or Gateway factory and does not provide raw capture.

## Capture boundary

A session starts at the Gateway data-plane boundary for `/v1` and `/v1beta` namespaces, before authentication and early route termination. It records the client request that is actually observed, final client response writes, material write/read/close/flush/hijack outcomes, logical retry attempts, and preparation failures.

Bifrost-backed HTTP provider adapters pass a real per-request observer through the supported execution context. Their observed upstream request and response headers, bodies, termination events, and callback errors are recorded in the corresponding logical attempt. When debug capture is enabled, model-probe and credential-test operations under `/api` use the same observer and store path; each dispatched probe has a session keyed by its `request_id` and a logical attempt containing provider, channel, model, route, sequence, and retry/fallback metadata. Probe capture remains disabled when the explicit opt-in is off.

The feature records only data that the Gateway or a supported Bifrost HTTP observer actually observes. It does not claim to capture TLS or socket wire bytes, HTTP transfer framing already removed by the transport, HTTP trailers not exposed through the observer contract, or data that was never observed before cancellation, timeout, process failure, or connection close.

Capture persistence is isolated from the data plane. Storage, queue, callback, or cleanup failures do not change the response returned to the client. The affected capture is instead marked failed or remains non-terminal, and the runtime health section reports cleanup state and active/completed/failed counts where available.

Gateway finalization checks cancellation against the request context saved before the data-plane lifecycle adds its own child context. Normal lifecycle release must not turn a complete successful capture into a cancellation failure. Genuine client cancellation and server shutdown still propagate through the saved context.

The saved context is `request.Context()` as bound when the capture is created, not a context the data-plane lifecycle derives afterwards.

Provider observation sources register before emitting callbacks and unregister after their completion callback returns. Gateway capture waits for registered observations and admitted callbacks; an executor result alone does not establish that a transport observation started. Once the sources and callbacks have drained, absence of a response completion is recorded as `no_observation_source` and capture finalization fails explicitly, releasing its runtime admission instead of waiting forever. The event detail distinguishes a source that never started from one that started and ended without completion. Neither condition establishes whether request or response bytes were exchanged. This event identifies missing raw evidence; it is not a Provider response termination observed on the wire. A first-token abort already fired before the net/http send completes its registered observation with `no_response`, without issuing a Provider request.

The lifecycle contract is optional on both sides. `internal/execution/http_observer.go` declares `HTTPObservationLifecycle`, and the fork's observer state in `third_party/bifrost-core/providers/utils/http_observer.go` begins one source per real send before any callback can be emitted and ends it after the final completion callback returns, including the fired-abort path in `DoHTTPRequest`. Neither call may block. `internal/gateway/capture.go` implements it for the Gateway; `internal/control/probe_capture.go` implements it for probes.

Regression evidence for this lifecycle lives in `internal/gateway/capture_lifecycle_test.go`, `internal/gateway/capture_abort_evidence_test.go`, `internal/gateway/capture_test.go`, `internal/debugcapture/runtime_test.go`, `internal/container/debug_capture_lifecycle_test.go`, and `third_party/bifrost-core/providers/utils/http_observer_test.go`.

## Model probes and credential tests

Model probes and credential tests use the same opt-in store and administrator
query/download APIs. Each probe uses its execution `request_id` and attempt
identity to associate the observed provider headers, body bytes, and termination
events with the capture. These captures describe provider communication; they do
not add a Gateway client request/response pair or a structured request-log row.
A probe capture can be located directly with the debug capture `request_id`
filter, without first finding a request-log entry.

`internal/control/probe_capture.go` supplies the shared observer through the
execution context. `internal/container/debug_capture.go` adapts it to the capture
store. Capture metadata includes provider, channel, upstream model, route mode,
and attempt sequence. A failed provider call and a failed capture are separate:
a provider error may still have a complete capture, while a capture write error
marks the attempt/session failed without replacing the probe's execution result.

Provider observations register a lifecycle. An observer registers its lifecycle when
created and unregisters after the final queued callback completes. Probe
finalization waits for registered observations and in-flight writes before
completing the capture; a probe that never starts an observer does not wait for
one. This drain has no independent deadline: if a registered response body is
never consumed or closed, finalization can continue waiting. It does not imply
that unobserved bytes have been captured.

The evidence boundary remains the bytes and events actually observed. A
`completed` capture with no response body cannot establish that a provider
returned a complete response; inspect parts and termination metadata together.
Regression evidence lives in `internal/control/probe_capture_test.go`,
`internal/container/debug_capture_test.go`.

## Empty-completion triage order

Start with the structured request log: confirm `error_code=upstream_empty_completion`, the HTTP status, the usage numbers, and each attempt's error classification and decision. The request log is an index and a result record, and it never substitutes for the Provider's raw body. Then query every capture for the same `request_id` through the administrator debug capture API and pair raw request/response bytes by attempt sequence:

```text
GET /api/logs/<request-id>
GET /api/debug-captures?request_id=<request-id>
GET /api/debug-captures/<capture-id>/download
```

A complete raw capture is what proves whether the observed response body, usage, finish reason, and message fields are genuinely absent, and therefore separates an empty completion from `upstream_content_filter`. Capture metadata, a parsed result, or a partial body is never complete raw evidence: a capture that is still `active`, already `failed`, expired, or gone, that lacks the response parts it should have, or that carries incomplete termination information must be marked as incomplete evidence. A metadata-only result must not be treated as a complete raw capture.

Raw capture records only the bytes and events that the Gateway, a supported provider observer, or a transport reader actually observed. It is not a TLS or socket wire capture, and it does not reconstruct framing already removed by the transport, trailers that were never exposed, or data that was never observed before a cancellation, timeout, process failure, or connection close.

## Capture-slot exhaustion triage order

When `/api/health` reports `debug_capture.error` as `session_limit_reached`, first separate "the slots are held by captures that have not been released" from "the database merely looks full".

1. Read health: `running` is `true`, `error` is `session_limit_reached`, and `active` may be `0`. The `active`/`completed`/`failed` numbers count unexpired database rows, not in-memory admission slots, so slot exhaustion and a finished expiry sweep can coexist.
2. Query `GET /api/debug-captures?state=active` for sessions that have not terminated. If `active` is `0` while the error persists, the slots are held by runtime-held sessions rather than by persisted rows.
3. The session may already be terminated: an attempt that is cancelled, drains its in-flight callbacks, and still sees no completion event finishes as a failed attempt, so its session becomes `failed` and `state=active` will not find it. When the slot count looks wrong, also query `state=failed` (and `completed` if needed) and inspect each session's attempts through `GET /api/debug-captures/:capture_id`.
4. A `termination` carrying `no_observation_source` is a Gateway-side finalization verdict: every registered observation source and admitted callback drained, yet no response completion event appeared. It only means the attempt cannot prove a full request/response exchange; it does not mean the attempt recorded no response bytes, because projected body and header parts may still exist. Inspect the recorded parts and the termination metadata, then judge evidence completeness by the empty-completion order above.
5. If a session stays `active` for a long time and its attempt has no `no_observation_source`, first confirm through the structured request log (joined by `request_id`) that the request is still being processed. Wait for it to finish; do not mask the condition by restarting the process.
6. After a fix, verify with a new request: the new capture must be created and terminate, and the `error` field must clear. Expired historical sessions cannot regain their raw bodies, and a structured request log never substitutes for a raw body.

Regression evidence for slot leaks is `internal/debugcapture/runtime_test.go` (health reports exhaustion and clears it after release) and `internal/gateway/capture_test.go` (no observation source, early cancellation, and in-flight callback drain all leave no pending work), with `internal/container/debug_capture_lifecycle_test.go` covering the end-to-end path.
