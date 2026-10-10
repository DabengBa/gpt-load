# Security Policy

## Supported versions

| Version line | Status |
|---|---|
| `2.0.x` | Pre-release security-supported candidate |
| `1.4.x` | Maintained security support line |

The repository's 2.0 candidate receives security fixes while its first public release is being prepared. This support status does not assert release readiness or public availability. No `v2.0.0` tag, GitHub Release, public binary, or public container image is confirmed here; verify availability from actual public assets.

## Debug communication capture

Debug communication capture is an opt-in operational tool. On Unix runtimes it is disabled by default and is enabled with `DEBUG_CAPTURE_ENABLED=true`. It writes observed application-layer request and response headers and bodies to the separate debug-capture store in plaintext, including sensitive `Authorization`, `Cookie`, and API-key values. Captures are retained for a fixed 4 hours and cleaned automatically.

Only authenticated management administrators can list, inspect, or download captures through `GET /api/debug-captures`, `GET /api/debug-captures/:capture_id`, and `GET /api/debug-captures/:capture_id/download`. Do not expose these endpoints or exported ZIP files to access-key users or untrusted operators. Protect `AUTH_KEY`, the database, backups, and downloaded archives as sensitive material. Capture storage failures are isolated from gateway request results; a capture may be marked failed or incomplete when observation or persistence was not completed.

The supported boundary is what Gateway, CPA, and Bifrost HTTP integrations actually observe at the application layer, including observed logical attempts and material outcomes. This feature does not claim to capture TLS/socket wire bytes, HTTP transfer framing already removed by the transport, HTTP trailers not exposed through the observer contract, or data that was never observed before cancellation, timeout, process failure, or connection close.

## Reporting a vulnerability

The reporting status for this repository is documented in its own
[Security policy](https://github.com/DabengBa/gpt-load/security/policy).

GitHub Private Vulnerability Reporting is currently **disabled** for
`DabengBa/gpt-load` (the repository API returned `enabled: false` during the
2026-10-10 governance check). No private reporting channel has been verified.
The policy link is informational, not an available private submission channel.

Please do not open a public issue or post undisclosed vulnerability details in
public discussions or community chat. This policy does not offer a public-issue
fallback or claim that a private report can currently be submitted.

## Channel credential exposure boundary

Channel credentials and any derived secret fragments must not appear in data-plane response headers or bodies, request logs, or the request-log management API. Request-log attempts identify the selected credential only by its internal `credential_id`, together with the non-secret `channel_id`.

The authenticated Group credential list is the sole approved mask exception: it may return a safe identifier mask so an operator can identify a credential in the provider console. Plaintext reveal is a separate authenticated, explicitly invoked endpoint and must never be logged or returned by collection, health, usage, or request-log APIs.
