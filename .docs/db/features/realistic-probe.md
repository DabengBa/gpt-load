---
id: feature.realistic-probe
type: feature
name: Model liveness probing
---
# Model Liveness Probing

## ID Explanation

`feature.realistic-probe` owns the preset real-request liveness workflow for a
specified group and model, including its result and recovery boundaries.

## Purpose

Operators can send a preset real upstream request to check a specific group and
model, without supplying an access key or writing a request body. The model
probe is the single liveness-testing workflow; there is no separate editable
actual-request workspace.

## User-Visible Contract

- From model maintenance or the dispatch center, run a single probe or a batch
  of probes. Each target identifies the group and model to test. A disabled
  group can be explicitly tested after confirmation; that does not make it
  eligible for ordinary routing.
- The management service selects a credential from the specified group and
  uses a code-owned protocol preset. Supported data protocols retain their own
  probe templates. The group's matching parameter overrides and model reasoning
  setting are applied to model probes.
- A single result puts the outcome, next step and scope first, with technical
  details expandable. A request-log ID is shown and linked when available;
  logging is best effort, so an unavailable ID does not prove no request was
  made. The single-target dialog can probe the same group, model and selected
  protocol again; a disabled group requires confirmation each time, and
  cancelling preserves the previous result without sending a request.
  A successful probe can restore the tested unhealthy route only while
  its relevant configuration and existing recovery guards still match; no
  reported recovery does not establish whether recovery was needed.
- This is an authenticated management operation, not a data-plane client
  request. It does not validate a client's access key, quota, stream or entire
  application workflow. Probe accounting is not ordinary client usage or cost;
  an upstream provider may nevertheless charge for the request.

## Acceptance Workflows

### Check a group model

- **Role and purpose:** An operator checks whether a model is available in a
  specific group.
- **Entry and action:** Select a saved model and run Probe; confirm first if the
  group is disabled.
- **Expected result:** A preset request uses a credential from that group, and
  the result and any request-log link are shown.
- **Failure signal:** A failed or inconclusive result shows its reason and a
  cautious next step. A log ID may be unavailable even after an attempt.

## Boundaries

Probes do not reproduce arbitrary client payloads, streaming, access-key
authorization or quotas. Credential validation and model tracing remain
separate operations.

Related pages: [Groups](page.groups), [Dispatch Center](page.dispatch-center).
Test-alias behavior is owned by [Model Test Aliases](feature.model-test-alias).
