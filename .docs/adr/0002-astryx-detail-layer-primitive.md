---
description: Astryx detail-layer primitive decision — right-side Dialog panel versus BottomSheet or a swizzled sidebar
kind: adr
topic: astryx-migration
status: accepted
date: 2026-09-25
---

# ADR-0002: Build the detail layer on the Astryx `Dialog` as a right-side panel; introduce neither `BottomSheet` nor a swizzled sidebar

## Context

Phase 1 spike (b) had to pick one primitive for the log detail layer out of
`Dialog`, `BottomSheet`, and a swizzled sidebar. The choice constrains every
later detail surface (logs, group detail, editing forms), and swapping the
primitive after the components above it have settled is expensive, so this is a
hard-to-reverse decision.

The reference implementation was the retired Vue `AppDrawer` (a fully
customized reka-ui `Dialog` turned into a right-side panel: focus trap, Escape,
scrim dismissal, focus return, a `dismissible` guard, and full width at ≤520px).

## Options

| Option | Focus trap / Escape / focus return | Panel shell | Swizzles | Outcome |
|---|---|---|---|---|
| `Dialog` `position={{end:0}}` | native `<dialog>` provides all of it | xstyle: full height, no radius, left border, full width at ≤520px | 0 | **chosen** |
| `BottomSheet` | same native dialog lifecycle | bottom-anchored with snap points and gestures — a mobile form whose gesture suppression is heavy override work | 1+ | rejected |
| Swizzled sidebar (hand-rolled reka equivalent) | must be implemented | fully custom | 1 (and it copies a pattern that had already become maintenance debt) | rejected |

## Three measured facts about the native `<dialog>`

1. **Permanent mounting:** the Dialog captures `document.activeElement` on the
   rising edge of `isOpen` and returns focus on the falling edge; conditional
   unmounting skips that lifecycle. A host must drive it with
   `isOpen={selected !== undefined}`, never `{selected && <Panel/>}`.
2. **The `data-autofocus` marker:** component-level autofocus committed before
   `showModal` is silently dropped, and the Dialog only focuses the first
   `[data-autofocus]` descendant after it opens. The detail layer puts that
   marker on `LayoutContent` (the scroll region, matching reka focusing the
   content root).
3. **Focus escaping silently at the edges:** a native Chromium modal moves focus
   to `<body>` at the tab-order boundary without dispatching a focus event, so a
   `focusin` redirect cannot catch it. Reka's sentinels behave the same way, so
   `Dialog onKeyDown` intercepts Tab and Shift+Tab as a hard containment
   (implemented inside `DetailPanel`, with a `focusin` guard left in place for
   evented escapes).

## Decision

- Detail-layer primitive = Astryx `Dialog` positioned on the right, wrapped by
  `DetailPanel` (`title`, `subtitle`, `dismissible` mapped to `purpose`
  info|required, `footer`).
- Swizzles for this spike: **0** (an input to gate #5).
- The focus contract is pinned by `web/e2e/astryx-log-detail.spec.ts`:
  `activeElement` never leaves the dialog, Escape and scrim dismissal work,
  focus returns to the invoking row, the `?selected_request_id=` deep link opens
  the panel directly, and 480px is full width.
