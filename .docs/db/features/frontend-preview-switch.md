---
id: feature.frontend-preview-switch
type: feature
name: Frontend preview switch
related: []
---
# Frontend Preview Switch

## ID Explanation

`feature.frontend-preview-switch` owns the user-visible control that lets an
operator opt into the React/Astryx frontend preview during coexistence, and
the preference it persists. It stops at writing the selection and reloading;
it does not own which routes serve the new frontend (that is the server-side
route manifest), the preview shell's content, or removal of the classic
frontend.

## Purpose

While both frontends ship, operators need an explicit, reversible way to try
the new interface without changing URLs, bookmarks, or deployment config.
The switch keeps the same origin and routes; only the served document and
rendered shell change for routes that have opted into Astryx.

## User-Visible Contract

- The preferences panel offers an "Interface" choice between Classic and
  Preview, available both on the login screen and inside the signed-in shell.
- Choosing Preview stores the `gpt-load.frontend=astryx` preference cookie
  and reloads the page.
- After opting in, routes flagged for the new frontend render the Astryx
  shell; unflagged routes and unknown paths continue to render the classic
  shell until they are migrated.
- Choosing Classic restores the classic frontend everywhere.
- The preference is a browser cookie, so it follows the browser, not the
  account; other browsers and sessions keep their own choice.
- If the preview build is not present in a deployment, the server falls back
  to the classic document and the page still loads.

## Acceptance Workflows

### Opt into the preview

- **Role and purpose:** An administrator wants to check the new interface
  before it becomes the default.
- **Entry and action:** Open the preferences panel, choose Preview under
  Interface, and let the page reload.
- **Expected result:** The preference cookie is stored; flagged routes then
  render the Astryx shell at the same URLs.
- **Failure signal:** If the reload renders the classic shell on a flagged
  route, the cookie was not written or the deployment lacks the preview
  build; no error screen appears.

## Boundaries

- The route manifest and the Go server own which paths serve which document.
- The classic preferences panel owns the control's placement, wording, and
  reload behavior; the Astryx shell owns its own return-to-classic control.
- Authentication, localization, and API behavior are unchanged by the
  switch; this feature only selects the document a route serves.
