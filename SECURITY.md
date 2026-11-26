# Security Policy

## Supported Versions

Security support follows the assets actually published for this repository; no version line is promised beyond what is visibly released.

## Reporting a Vulnerability

GitHub Private Vulnerability Reporting is **disabled** for `DabengBa/gpt-load` (`enabled: false`, confirmed by a governance check on 2026-10-10). The security policy link shown by GitHub is informational only: there is currently **no working private reporting channel** and no public-issue fallback is offered. Do not disclose vulnerabilities in public issues, discussions, or pull requests.

## Debug Communication Capture

Debug communication capture is disabled by default and opt-in. When enabled, captured plaintext may contain Authorization headers, cookies, and API keys. Operating constraints, retention semantics, and the evidence boundary are owned by [`.agents/skills/gpt-load-operations/references/debug-capture.md`](.agents/skills/gpt-load-operations/references/debug-capture.md).

## Channel Credential Exposure Boundary

- Channel credentials and derived secrets MUST NOT appear in data-plane response headers or bodies, request logs, or request-log management APIs.
- An attempt identifies its credential only by internal `credential_id` plus a non-sensitive `channel_id`.
- The authenticated Group credential list is the single masking exception, and it exposes only masked security identifiers.
- Plaintext reveal is a separate, explicitly invoked, authenticated endpoint; revealed values are never recorded or returned by collection, health, usage, or request-log APIs.
