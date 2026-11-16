---
description: "Native non-streaming Responses handling when a provider labels completed JSON as an event stream."
kind: technical
topic: responses-unary-compatibility
relations:
  related:
    - tech/realistic-probe.md
    - tech/modeltrace-fingerprint.md
code:
  paths:
    - internal/execution/bifrost/passthrough.go
    - internal/execution/bifrost/gateway_probe.go
---
# Native Responses Unary Compatibility

## Responsibility and boundary

The Bifrost passthrough executor handles a provider that returns a completed
Responses JSON object with `Content-Type: text/event-stream` for a non-streaming
request. This compatibility applies when both client and upstream protocols are
OpenAI Responses; it does not define streaming or protocol conversion behavior.
Model liveness probes and ModelTrace have separate request and result semantics:
see [Model Liveness Probing](realistic-probe.md) and
[ModelTrace Fingerprint Detection](modeltrace-fingerprint.md).

## Handling and constraints

For a successful native Responses create, the executor decodes the body within
the unary limit and corrects the content type to `application/json` only if the
decoded body is a structurally valid completed response: `object` is `response`,
`status` is `completed`, `output` is an array of valid typed items, and `error`
is absent or null. This includes validated shell and apply-patch tool calls.
When correcting it, the executor removes representation metadata and extracts
usage if the SDK did not supply it. Model alias rewriting can still run later.
Malformed, incomplete, error-bearing, actual SSE, non-success, or undecodable
responses are not relabeled by this compatibility branch; normal downstream
handling and safety checks still apply.

For successful native Responses probe and ModelTrace operations, a valid
completed JSON body is accepted even with an SSE content type. Otherwise an
SSE-labeled body follows the existing terminal-event normalization path; an
invalid or undecodable stream fails as an upstream protocol probe response.
The normalized utility result is JSON, and usage is extracted when needed.

## Verification

`internal/execution/bifrost/responses_test.go` covers normal create responses,
negative cases and gzip; `internal/execution/bifrost/probe_sse_test.go` covers
probe and ModelTrace results; `internal/execution/bifrost/responses_mislabeled_test.go`
covers tool-item validation. These are local fixtures, not live-provider proof.
