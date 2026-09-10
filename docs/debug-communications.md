# Debug Communication Capture

Debug communication capture is an opt-in operational tool for investigating gateway and provider integration behavior. It is disabled by default.

## Enablement and retention

Set `DEBUG_CAPTURE_ENABLED=true` before starting GPT-Load. The setting is process configuration and takes effect after restart. The capture store uses the existing configured database and remains separate from `request_logs`.

When capture is disabled, no new sessions are admitted; existing unexpired captures remain available to management administrators until their fixed expiry. Startup cleanup still removes expired rows and chunks even when the feature is disabled, and enabled deployments also run periodic cleanup.

The capture can contain complete observed sensitive values, including `Authorization`, `Cookie`, API keys, request bodies, response bodies, and provider credentials passed through the supported observation boundary. Protect `AUTH_KEY`, the database, database backups, and exported archives. Do not enable this feature on an untrusted or shared deployment unless that exposure is intentional.

## Administrator API

These endpoints require the management `AUTH_KEY` and are administrator-only. Data-plane access keys cannot use them.

- `GET /api/debug-captures`
- `GET /api/debug-captures/:capture_id`
- `GET /api/debug-captures/:capture_id/download`

The list endpoint accepts `request_id`, `access_key_id`, `protocol`, `operation`, `state`, `created_after_ms`, `created_before_ms`, `offset`, and `limit`. The default page size is 50 and the maximum is 200. `created_after_ms` is inclusive and `created_before_ms` is exclusive.

The detail response contains session metadata, lifecycle state, errors, and ordered logical attempts. The download endpoint streams a ZIP containing a JSON manifest, per-attempt metadata, and raw request/response header and body parts. Download responses are marked `no-store`.

The `/api/health` response includes `debug_capture` with enabled/running state, fixed retention seconds, active/completed/failed counts, sweep counters, and timestamps. If the optional store cannot provide counts, the component reports `error: "counts_unavailable"` while the broader health response remains available.

Feature-specific capture storage and adapters are Unix-only. Windows retains the public application types and no-op container assembly so the unrelated Windows binary continues to compile; enabling capture on Windows does not install a capture store or Gateway factory.

## Capture boundary

A session starts at the Gateway data-plane boundary for `/v1` and `/v1beta` namespaces, before authentication and early route termination. It records the client request that is actually observed, final client response writes, material write/read/close/flush/hijack outcomes, logical retries and refresh attempts, and preparation failures.

CPA integrations for Codex, Claude, Antigravity, and Grok pass a real per-request observer through the supported execution context. Their observed upstream request and response headers, bodies, trailers, EOF, close, and callback errors are recorded in the corresponding logical attempt.

The feature records only data that the Gateway or a supported CPA observer actually observes. It does not claim to capture Bifrost's underlying HTTP transport, TLS or socket wire bytes, data never read or written, or bytes lost before observation because of cancellation, timeout, process failure, or connection close.

Capture persistence is isolated from the data plane. Storage, queue, callback, or cleanup failures do not change the response returned to the client. The affected capture is instead failed or left explicitly incomplete, and the runtime health section reports cleanup state and active/completed/failed counts where available.
