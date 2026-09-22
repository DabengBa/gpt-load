---
id: feature.model-test-alias
type: feature
name: Provider-specific model test aliases
---
# Provider-Specific Model Test Aliases

## ID Explanation

`feature.model-test-alias` owns the operator-visible short names that select
one configured provider and model route for testing. It covers the code shown
in group model maintenance, how an operator uses it in a client request, and
the attribution visible after that request; persistence and route-index
mechanics belong to the technical model-routing owner.

## Purpose

When several groups expose the same client model, an ordinary request may be
distributed across multiple provider routes. A test alias gives an operator a
stable, recognizable way to exercise one specific group/model entry without
changing the public client alias or removing the other routes from normal
traffic.

## User-Visible Contract

- Every saved group model row has a server-generated six-character test alias
  made only of lowercase letters and digits. The alias is unique across the
  configured model entries and does not reuse an ordinary upstream or client
  model name.
- The classic group model editor shows the alias as read-only code. A newly
  added unsaved row shows that its alias will be assigned after saving; the
  alias is not an editable field or a keyboard focus stop.
- A client can send the test alias as its model name. The request resolves to
  the one group/model route entry that owns that alias instead of joining the
  route pool for a shared ordinary model name. Normal access-key and group
  visibility filters still apply.
- The test alias is included in visible model listings when its route is
  visible to the caller, so an operator can discover it through the same model
  surface used for other models.
- Keeping the same saved model entry preserves its alias across reads, updates,
  restarts, and repeated idempotent writes. Copying a group assigns the copied
  entries different aliases so the two groups remain independently testable.
- Request history keeps both identities: the requested test alias remains the
  client model, while the actual upstream model and group/route attribution
  remain visible for diagnosis, usage, and cost reporting.
- The alias remains visible and usable on narrow screens without horizontal
  overflow, while model editing controls retain their normal keyboard order.

## Acceptance Workflows

### Test one provider/model route

- **Role and purpose:** An administrator wants to verify one provider/model
  combination while another group exposes the same ordinary client model.
- **Entry and action:** Open group model maintenance, copy the read-only test
  alias for the target row, and send a request using that value as `model`.
- **Expected result:** The request reaches the target group/model entry only;
  the response remains usable, and request history identifies the test alias as
  the client model and the real upstream model as the route target.
- **Failure signal:** A request using the test alias must not fan out to another
  entry sharing the ordinary client model or attribute usage to the short code
  as if it were the provider's model.

### Preserve the server-owned alias while editing models

- **Role and purpose:** An administrator wants to change a public alias or
  model list without losing the code used by an existing test client.
- **Entry and action:** Edit an existing model row, save the group model list,
  and reopen the model editor.
- **Expected result:** The save payload contains only editable model fields;
  the server returns the same test alias for the retained entry. A newly added
  entry receives its own alias after the write completes.
- **Failure signal:** The UI must not allow a submitted `test_alias` value to
  overwrite a server allocation, and a retained row must not receive a new
  alias merely because another editable field changed.

## Boundaries

- This feature owns test-alias discovery, single-entry route selection, and
  operator-facing attribution.
- Ordinary client aliases continue to own shared traffic routing; test aliases
  do not replace them, alter weights, or disable fallback behavior for ordinary
  requests.
- The feature does not change provider model IDs, provider pricing rules,
  credentials, request-log retention, or access-key permissions.
- A test alias is a routing identifier, not a secret or an authorization
  bypass. Its visibility remains subject to the caller's existing route and
  group filters.
