---
description: "Site access security and compatibility invariants for provider site information: root confirmation, credential handling, snapshot consistency, and compatibility boundaries."
kind: technical
topic: group-site-information
id: INV-003
status: active
paths:
  - internal/control/site_info.go
  - internal/storage/models/site_info.go
  - web/src/frontends/astryx/features/groups/**
enforcement: fresh-audit
relations:
  related:
    - adr/0004-group-site-info-independent-tables.md
code:
  paths:
    - internal/control/site_info.go
    - internal/control/site_info_http.go
    - internal/control/site_info_sync.go
    - internal/storage/models/site_info.go
---
# Group Site Information Invariants

## Invariants

- The provider website scheme and host define the site root. Changing a configured root SHALL require explicit confirmation before the browser submits the access token to that root.
- URL validation MUST reject unsupported schemes, credentials embedded in the URL, and invalid new ports, while preserving safe reads and unrelated edits for legacy values.
- Site requests authenticate with `Authorization` and `New-Api-User`. Credentials SHALL be encrypted at rest and MUST NOT be projected back to the client.
- Cross-origin redirects MUST NOT be followed. Persisted display errors MUST be sanitized: no request credentials or response bodies.
- The effective ratio is scoped to the authenticated user and the matched token's effective group. A failed authenticated group lookup MUST NOT fall back to an anonymous pricing view; provider account balance is distinct from an individual token's remaining quota.
- Failures SHALL retain the last successful balance, ratio, and announcements where possible. Announcement entries and source states are read from one database snapshot so the page MUST NOT combine refresh generations.
- The retry action SHALL mark the source pending and wake the synchronizer; it MUST NOT perform a synchronous provider request or promise completion.
- New fields are opt-in for the updated frontend so older clients retain the response shape. An announcement source `last_error` is optional at the projection boundary.
- Access tokens MUST NOT appear in copy-group payloads or import recovery storage.
- A local mocked provider verifies the browser workflow but MUST NOT be treated as compatibility proof for every live New API deployment.

## Boundary

Storage layout and migration numbering are owned by
`adr/0004-group-site-info-independent-tables.md`. Endpoint and field tables
are excluded — code is the source of truth.
