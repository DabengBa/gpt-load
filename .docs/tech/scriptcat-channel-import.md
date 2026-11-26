---
description: "ScriptCat handoff invariants: single-claim delivery, origin binding, expiry windows, and residual credential exposure."
kind: technical
topic: scriptcat-channel-import
id: INV-004
status: active
paths:
  - tools/scriptcat/gpt-load-channel-import.user.js
  - web/src/shared/domain/import/scriptcat-payload.ts
  - web/src/frontends/astryx/features/import/**
enforcement: fresh-audit
relations:
  related:
    - db/features/scriptcat-credential-import.md
code:
  paths:
    - tools/scriptcat/gpt-load-channel-import.user.js
    - web/src/shared/domain/import/scriptcat-payload.ts
    - web/src/frontends/astryx/features/import/ImportView.tsx
---
# ScriptCat Channel Import Handoff Invariants

## Invariants

- Delivery SHALL be single-claim: read, validation, expiry check, origin check, `sessionStorage` write, and transfer deletion all run inside one exclusive Web Lock so two tabs serialize and the second finds nothing.
- There MUST be NO fallback claim path. When Web Locks or browser storage are unavailable the claim throws, nothing is written to `sessionStorage`, and the user imports manually; no local-storage marker, retry loop, or compatibility record may emulate the lock.
- The destination SHALL accept a transfer only when its recorded target origin equals the current origin. Credentials, Base URL, group name, and provider website MUST NOT appear in the destination URL.
- The transfer expires 60 seconds after write in the source store; the consumed `sessionStorage` record is removed at a 15-minute TTL and only while the stored value is still the one written. A payload older than 15 minutes or with a future `createdAt` MUST be discarded.
- Consumption MUST remove the record before parsing; the first non-null result stops the poll and the record is gone for every later consumer. Anything failing the exact record shape MUST be discarded.
- The userscript manager store is local to the browser profile and is NOT encrypted by GPT-Load. If the destination tab never opens, a plaintext transfer can remain; sharing or syncing that store while a transfer is pending moves the credential outside the intended profile.

## Boundary

Install and adaptation steps are owned by `tools/scriptcat/README.md`;
operator-visible import behavior by `db/features/scriptcat-credential-import.md`.
Payload field lists are excluded — code is the source of truth.
