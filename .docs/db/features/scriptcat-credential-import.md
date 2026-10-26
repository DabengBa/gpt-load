---
id: feature.scriptcat-credential-import
type: feature
name: ScriptCat credential import
related: []
---
# ScriptCat Credential Import

## ID Explanation

`feature.scriptcat-credential-import` owns the administrator-visible behavior of
carrying an API key from a provider key page into the new-group import form: the
entry control, the values that arrive, what the form pre-fills, and how the
administrator finishes or abandons the import. The handoff storage, claim
protocol and expiry rules belong to
[ScriptCat Channel Import Handoff](../../tech/scriptcat-channel-import.md).

## Purpose

Copying a key out of a provider's console and pasting it into GPT-Load is
error-prone. An administrator who keeps an API key page open can hand the key,
its Base URL, a group name and the provider website directly to the import form
and only review and submit there.

## User-Visible Contract

- On a supported provider key page, a `导入到 GPT-Load` control opens a panel for
  the API key, group name, Base URL and GPT-Load address. The control is not
  shown on unrelated pages or on a login screen.
- The panel pre-selects the key visible in the page text or the current
  selection, the site root as Base URL, the page origin as provider website, and
  the last GPT-Load address used.
- Activating the panel opens GPT-Load's new-group import page in a new tab with
  the new-group mode already selected. The API key is never placed in the URL.
- The import form arrives pre-filled with the key as credentials, the group
  name, the Base URL and the provider website. No group, credential or request
  is created until the administrator submits the form.
- The administrator can edit every pre-filled value, choose a different
  channel, or abandon the import. Choosing a channel keeps the imported Base URL
  when that channel has a URL parameter.
- A handoff is delivered at most once. Opening the same handoff link in several
  tabs fills the form in one tab and leaves the others as ordinary manual
  imports.
- If the browser does not provide the required locking or storage, the form is
  not pre-filled and the administrator enters the values manually. That
  fallback is not guaranteed to be announced: the destination may fail
  silently, and the source may already hold the transfer it created.
- Signing in after the redirect keeps an import draft that has no channel
  selected yet.

## Boundaries

- This feature owns the entry control, the pre-filled draft and the
  administrator's review step.
- The userscript owns source-page detection, panel fields and the handoff
  protocol; GPT-Load accepts no credential without a form submission.
- The import form owns channel selection, parameter validation, model
  discovery and submission; group lifecycle and credential storage are
  unchanged by this feature.
