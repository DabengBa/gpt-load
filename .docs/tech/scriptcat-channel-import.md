---
description: "How a ScriptCat userscript hands one API key from a provider key page to GPT-Load's new-group import form: single-claim delivery, origin binding, expiry, consumption and draft prefill."
kind: technical
topic: scriptcat-channel-import
relations:
  related:
    - db/features/scriptcat-credential-import.md
code:
  paths:
    - tools/scriptcat/gpt-load-channel-import.user.js
    - web/src/shared/domain/import/scriptcat-payload.ts
    - web/src/frontends/astryx/features/import/ImportView.tsx
---
# ScriptCat Channel Import Handoff

## Scope

- **Owner:** the one-use handoff that moves an API key, Base URL, group name and
  provider website from a provider key page into the `/import` new-group form,
  including claim, origin binding, expiry, consumption and draft prefill.
- **Authoritative for:** where each credential copy lives, how a delivery is
  claimed exactly once, which TTL applies, and what the import form does with a
  consumed payload.
- **Excludes:** what operators see and may edit, owned by
  [ScriptCat credential import](../db/features/scriptcat-credential-import.md);
  group creation, channel parameters and credential validation, owned by the
  import feature itself; browser or userscript-manager installation steps,
  which stay in `tools/scriptcat/README.md`.

## Delivery Path

The userscript runs in two roles. On a supported provider key page it renders a
panel and, on submit, stores the credential, Base URL, group name, target
origin and the current page origin under `pending-import:<transferId>` in the
userscript manager's shared value store. The transfer expires 60 seconds after
it is written. It then opens
`<origin>/import?mode=new&scriptcat_handoff=<transferId>` in a new tab and
reuses the last accepted GPT-Load origin as the default for the next transfer.

On the GPT-Load origin the same script reads the transfer named by the
non-secret `scriptcat_handoff` parameter. A transfer is accepted only when its
recorded target origin equals the current origin; credentials, Base URL, group
name and provider website never appear in the destination URL.

## Single Claim

Claiming runs inside
`navigator.locks.request('gpt-load.scriptcat-import-payload:<transferId>')`.
The read, the record validation, the expiry check, the origin check, the
`sessionStorage` write and the deletion of the transfer entry all happen while
that exclusive lock is held, so two tabs opened from the same link serialize:
the first claims and deletes the transfer, and the second finds nothing and
renders a plain manual form.

There is no fallback claim path. When Web Locks or browser storage are
unavailable the claim throws, nothing is written to `sessionStorage`, the
receiver's failure is silent, and the user imports manually. No local-storage
marker, retry loop or compatibility record is used to emulate the lock.

## Payload Record And Expiry

The claimed record is written to the page's own `sessionStorage` under
`gpt-load.scriptcat-import-payload` as `version: 2` with `createdAt`, `apiKey`,
`baseURL`, `groupName` and `providerUrl`. Consumption requires those fields,
rejects a record with any additional field, requires a `createdAt` that is not
in the future and not older than 15 minutes, a non-empty API key, Base URL and
group name within their length bounds, and an http or https provider URL. Any
other content is discarded.

Two independent timers bound the credential, and neither survives its document:

- The source tab schedules deletion of the `GM` transfer entry 60 seconds after
  writing it, while the source page is still open. If the destination has not
  claimed it by then, the entry is gone and the handoff must be repeated; if the
  source tab closes first, the entry persists until a later transfer attempt or
  a matching destination page removes it as expired.
- The destination claim schedules removal of the `sessionStorage` record when
  its 15-minute TTL elapses, only while the stored value is still the one it
  wrote. A full reload interrupts that timer, and the userscript reschedules it
  when it starts on `/import` again: an expired record is removed immediately,
  an unexpired one gets a new timer for its remaining lifetime.

Creating a transfer from the source page sweeps expired `GM` entries before it
writes the new one, and a destination claim deletes a malformed or expired entry
it is asked to claim.

## Consumption And Prefill

`consumeScriptcatImportPayload` reads the record, removes it before parsing,
and returns a validated payload or `null`. The import view consumes at mount,
on the `gpt-load:scriptcat-import` event, and through a short bounded poll while
the page stays open; the first non-null result stops the poll and the record is
gone for every later consumer.

A consumed payload becomes a new-group draft with `channel_id` empty,
the payload Base URL as the URL parameter, the group name,
the provider website and the API key as credentials. Nothing is submitted.
Selecting a channel keeps the imported Base URL when the channel exposes a URL
parameter, and a draft the operator has already edited is not replaced by the
payload. A recovered re-authentication draft keeps an empty `channel_id`, so an
import draft survives the login round trip.

## Residual Exposure

The userscript manager store is local to the browser profile and is not
encrypted by GPT-Load. If the destination tab never opens, the plaintext
transfer can remain until a later transfer attempt or a matching destination
page removes it. Sharing or syncing that store while a transfer is pending
moves the credential outside the intended profile.

## Binding Points

- Code binding: `tools/scriptcat/gpt-load-channel-import.user.js`
- Code binding: `web/src/shared/domain/import/scriptcat-payload.ts:consumeScriptcatImportPayload`
- Code binding: `web/src/frontends/astryx/features/import/ImportView.tsx`
- Code binding: `web/src/frontends/astryx/features/import/NewGroupImport.tsx`
- Code binding: `web/src/shared/controllers/import-recovery.ts`

## Testing

`web/scripts/scriptcat-import-payload.test.ts` covers record validation,
expiry, single consumption and rejection of foreign or malformed records.
`web/e2e/astryx-scriptcat-widget.spec.ts` drives the real userscript with a
`GM` bridge: widget visibility, panel fields, transfer creation, receiver
prefill, and the concurrent double-tab case that must deliver exactly one
payload under the browser's exclusive lock.
`web/e2e/astryx-import-credentials.spec.ts` covers the prefilled import form,
Base URL retention across channel selection, and the untouched-user-draft rule.
