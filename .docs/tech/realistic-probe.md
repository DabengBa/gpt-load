---
description: "Browser request execution and response observation for actual-use model testing, alongside fixed diagnostic probes."
kind: technical
topic: model-probing
relations:
  related:
    - db/features/realistic-probe.md
    - tech/model-test-alias.md
code:
  paths:
    - web/src/shared/probe/realistic-probe.ts
    - web/src/frontends/astryx/features/models/RealisticProbeDialog.tsx
    - internal/execution/bifrost/gateway_probe.go
---
# Actual Request Testing

## Responsibility

The workspace observes a single browser-originated request through the existing
data plane. It has no new control API, backend executor, persisted source marker
or routing override. Product semantics belong to
[Actual Request Testing](../db/features/realistic-probe.md).

`GroupModelsTab` uses the saved client alias/ID and the first supported native
generation route, falling back to a converted route, as an editable template.
This is not the channel's basic probe contract. `SchedulePanel` supplies the
selected external model and protocol. Data-only routes keep their existing
diagnostic controls.

## Execution And Identity

`runRealisticProbe` sends the original JSON string using native `fetch` to fixed
root-relative, same-origin paths:

| Protocol | Path |
| --- | --- |
| Chat Completions | `/v1/chat/completions` |
| Responses | `/v1/responses` |
| Anthropic Messages | `/v1/messages` |
| Gemini | `/v1beta/models/{model}:generateContent` or `:streamGenerateContent?alt=sse` |

Non-Gemini JSON owns `model` and `stream`. Gemini model and stream mode select
the path; the body retains its native shape. Validation checks JSON object,
model/path, stream type and header-safe key; the gateway/upstream owns the rest
of the native request schema. Unknown native fields are preserved.

The supplied key is sent in `Authorization: Bearer`, with `credentials: omit`
and `redirect: error`. The management `ApiClient` is not used. The dialog holds
the key only in component state, and clears it on close. No URL, browser storage
or copied result is created for the key. Returned previews mask exact and
JSON-escaped matches of the supplied key before publishing or truncating, and
withhold a possible trailing key prefix on interruption. Previews remain
potentially sensitive response content. If raw debug communication capture is
enabled, its client-request headers can include the supplied access key; its
storage, administrator access and retention follow the separate
[debug capture contract](debug-communications.md).

The ordinary gateway remains authoritative for access filters, client-model
parameter-rule matching, upstream mapping, retries, fallback, billing and health.
The group from which the workspace was opened is not a routing constraint.
Using a test alias changes the requested model; see
[Model Test Alias Routing](model-test-alias.md). This is a normal generation
operation, not `OperationProbe`, so ordinary usage accounting applies.

## Response Observation

The helper incrementally decodes UTF-8 and SSE across transport chunk boundaries.
It requires protocol completion rather than treating HTTP 200 as success:

- Chat needs completed choices and `[DONE]`; requested choices are tracked by
  index. `stop`, `tool_calls` and `function_call` are complete finish reasons;
  tool calls also require a name and arguments for each observed function.
- Responses distinguishes `response.completed`, `response.incomplete` and
  `response.failed`, validates the terminal response, and retains incomplete
  reasons. A valid reasoning-only completion can have no first-output timestamp.
- Anthropic needs content and `message_stop` with a complete stop reason; tool
  input deltas must follow a valid tool block and form a JSON object. Accumulated
  tool input is bounded to 1 Mi characters across the stream.
- Gemini aggregates requested candidates and waits for EOF; normal completion
  requires content and `STOP` on every observed/requested candidate, with no
  unfinished trailing event.

Non-stream JSON receives equivalent protocol checks. Tool-call output is
accepted without executing tools. Budget/filter stops and malformed, unknown
or unfinished replies are incomplete. HTTP/protocol errors are failed. An
interruption without a complete terminal result is incomplete; abort is canceled.
These are local delivery observations, not overrides of gateway health outcomes.

Preview storage is bounded to 128 Ki characters, one SSE event to 256 Ki
characters, and non-stream parsing to 1 Mi characters. Candidate state is
bounded to 1024 entries. Oversized parsing input produces an incomplete result;
preview truncation alone does not change protocol completion. This preview
omits SSE framing and is not equivalent to debug communication capture.

The dialog uses `AbortController` plus an invocation counter to suppress late
updates. Inputs and log navigation are disabled during execution. Closing/unmounting aborts the
reader; editing after completion clears its result. Native browser/network
headers and SDK multi-step behavior are not reproduced.

## Request Logs And Diagnostics

`X-GPTLoad-Request-ID` is accepted for navigation only in canonical UUIDv4 form.
Early authentication/route failures may have no ID. Existing
`selected_request_id` navigation opens the request-log detail with actual attempt
attribution; log availability does not gate the delivery result. Debug captures
remain subject to their own enablement, admission and retention rules.

The separate fixed Responses diagnostic now emits a user-message list with
`input_text` through `responsesProbePayload`, shared by native and multiprotocol
execution. Its prompt, 128-token budget, non-stream/store flags and diagnostic
recovery/accounting behavior are retained.

## Verification

- `web/scripts/realistic-probe.test.ts`: controlled native fetch and streamed
  response cases, terminal/partial/error/cancel, tools, bounds, native fields,
  secret handling and multi-candidate aggregation.
- `web/e2e/astryx-realistic-probe.spec.ts`: both entries, saved client alias,
  request result/log navigation, key lifetime, late response isolation, stream
  stop, protocol replacement, keyboard and 390/1280-pixel layout.
- `TestMultiProtocolGatewayProbesUseSelectedProtocol` and
  `TestNativeResponsesProbeUsesInputAndMaxOutputTokens`: diagnostic wire body.

Local controlled tests establish implementation behavior; they do not certify
an external provider, a deployment under an arbitrary subpath or every client's
headers and workflow.
