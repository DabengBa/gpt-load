---
id: feature.realistic-probe
type: feature
name: Actual request testing
---
# Actual Request Testing

## ID Explanation

`feature.realistic-probe` owns the interactive request workspace opened from
model maintenance and the dispatch center. It describes what an operator can
conclude from one real request, separately from the existing basic diagnostic
probe and recovery actions.

## Purpose

Operators need to test the protocol, client model, parameters and streaming mode
their client actually uses. A fixed diagnostic prompt alone cannot establish
that the same request will be routed and delivered successfully.

## User-Visible Contract

- A saved model row can open Actual request with its saved client alias or model
  ID. Its initial template uses a supported generation protocol. The dispatch
  center supplies its selected external model and generation protocol.
- The operator can choose Chat Completions, Responses, Anthropic Messages or
  Gemini, edit native request JSON and select streaming. Replacing a protocol
  template requires confirmation because it replaces the edited JSON.
- The operator supplies an access key for this request. The browser workspace
  does not save it and clears it on close. When raw debug communication capture
  is enabled, it can record the request headers, including the access key.
  Management sign-in does not substitute for data-plane access.
- Requests use normal routing and permissions, parameter overrides, quotas,
  usage accounting and health effects. Opening from a group does not pin that
  group or bypass eligibility. Editing to a test alias changes the client model
  and therefore can change model permission and parameter-rule matches.
- Results distinguish running, complete, incomplete, failed and canceled. An
  HTTP 200 or partial streamed text alone is not a complete result. Token-budget
  or filtering stops remain incomplete, with their terminal reason visible.
- A complete tool-call response is valid delivery; the workspace does not run
  the tool. Completion establishes protocol delivery, not answer quality or
  successful completion of a multi-step client workflow.
- Results show HTTP status, first observed output time when available, total
  elapsed time, terminal reason, a bounded response preview and any request ID.
  A request ID opens the existing request-log detail to inspect routing and
  attempts after the request stops. A successful fallback does not change the
  failed attempt in that log.
- Missing or delayed logs do not establish success or failure. Early rejection
  may have no request ID. The preview is not a raw communication capture.
- Editing request inputs clears the previous result. Stop cancels local receipt;
  closing also cancels receipt and prevents a late result from replacing a new
  workspace result. Cancellation does not guarantee that upstream work or cost
  has stopped.

## Acceptance Workflows

### Reproduce a client's request

- **Role and purpose:** An operator investigates a model that passes a basic
  probe but fails in a client.
- **Entry and action:** Open Actual request from a saved model or selected
  dispatch model; choose the client's protocol, paste its JSON, supply the
  access key, and send.
- **Expected result:** The same client model, native parameters and stream mode
  go through normal routing. The result reports delivered completion and links
  to the request's actual attempts when a request ID is available.
- **Failure signal:** A rejected request shows failure and its response preview;
  a broken stream shows incomplete delivery even after text arrived. The
  operator can stop receipt or edit and try again.

## Boundaries

Basic probes remain available for fixed, low-cost diagnostics and recovery,
including existing data-protocol probes. Actual request testing covers the four
generation protocols above. It does not provide arbitrary remote endpoints,
custom client headers, saved keys, automatic log replay, batch execution, tool
execution or a separate dispatch policy.

Related pages: [Groups](page.groups), [Dispatch Center](page.dispatch-center).
Test-alias behavior is owned by [Model Test Aliases](feature.model-test-alias).
