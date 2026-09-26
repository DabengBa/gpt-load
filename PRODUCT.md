# Product

## Register

product

## Users
Operators and administrators of a self-hosted GPT-Load instance — the people who configure LLM upstream groups, manage credentials and access keys, and monitor request logs. They work inside an admin console during triage and configuration sessions, often on dense tables of channels, groups, and log records. The job to be done on any screen is operational: find a record, inspect it, change a setting, verify the result.

## Product Purpose
GPT-Load is a self-hosted LLM gateway; this UI is its management plane. It exists to make gateway operations — group/channel configuration, key management, scheduling, request-log forensics — fast and legible for a technical operator. Success looks like: every operational task reachable in a few clicks, dense data surfaces that stay readable, and a UI that never hides state or surprises during production work.

## Brand Personality
Compact, professional, utilitarian. The interface is a tool, not a showcase — it favors information density, predictable layout, and restrained chrome over marketing flourish.

## Anti-references
- Marketing-style landing page patterns (hero sections, oversized type, decorative whitespace) — this is a console.
- Consumer-app visual noise: gradients-as-decoration, playful illustrations, oversized cards that sacrifice row density.
- Heavy Material-Design-style elevation and padding that cut data density.
- The visual target is explicitly the existing classic Vue UI: deviations from its density, spacing rhythm, and information hierarchy are defects, not improvements.

## Design Principles
- Parity over novelty: a migrated surface is correct when it matches the classic behavior and density, not when it looks newer.
- Density is a feature: compact control heights, tight rows, and small meta text exist so operators scan more data per screen.
- Server truth in the URL: collection state (filters, pagination, selection) lives in the query string so views are shareable and reload-safe.
- Progressive disclosure: detail overlays and filter drawers reveal depth on demand without leaving the collection context.
- Keyboard and screen-reader paths are first-class: every control is named, focus is managed in overlays, and landmarks stay correct.

## Accessibility & Inclusion
WCAG AA baseline. Focus is trapped and restored in overlays, every interactive element carries an accessible name (visible or visually-hidden label), contrast meets the audited tokens, and reduced-motion is respected. Two primary principals — admin and read-only access_key — get capability-scoped surfaces.
