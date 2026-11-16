---
description: "The multi_protocol channel's gateway target, declared native routes and default probe boundary."
kind: technical
topic: multi-protocol-channel
relations:
  related:
    - db/pages/groups.md
    - tech/realistic-probe.md
code:
  paths:
    - internal/channel/modules/multi_protocol.go
    - internal/channel/multi_protocol_module_test.go
    - internal/execution/bifrost/multi_protocol_runtime_test.go
---
# Multi-Protocol Channel

## Responsibility and constraints

The `multi_protocol` channel binds a gateway using the existing
`multi_protocol_gateway` provider kind. It requires a normalized `base_url` and
a non-empty API key; it has no provider catalog ID. Configure the gateway root
or deployment prefix rather than an API version path: the adapter appends the
operation's protocol path to the configured base. A version-looking path is
not stripped or rejected: for example, `/v1` produces `/v1/v1/chat/completions`
for Chat, and `/team/v1beta` produces `/team/v1beta/v1beta/...` for Gemini.
These can be valid deployment prefixes, so the import form warns rather than
blocking them. This next hop follows the
client protocol for declared routes; the gateway decides what happens beyond
that hop. Product-facing selection is described in [Groups](../db/pages/groups.md).

## Declared operations

The channel declares native routes for Chat Completions, Responses create and
compact, Anthropic and Gemini generation, OpenAI and Gemini embeddings, image
generation and edit, rerank, and OpenAI model listing. Responses create uses
upstream-managed storage; Responses retrieval/lifecycle and WebSocket support
are not declared. The route declarations, not a general promise that every
gateway supports every protocol, define GPT-Load's boundary.

The channel's default probe contract is OpenAI Chat Completions with a minimum
output budget of 128. Model probes can explicitly select supported data probe
routes where declared; a probe only establishes the tested operation. General
probe preparation, recovery and limitations belong to
[Model Liveness Probing](realistic-probe.md).

## Verification

- `internal/channel/multi_protocol_module_test.go` checks the fields, routes,
  default probe contract and unsupported lifecycle capabilities.
- `internal/control/multi_protocol_integration_test.go` checks registration and
  group creation through the management API.
- `internal/execution/bifrost/multi_protocol_runtime_test.go` checks local
  gateway paths and credentials for Chat, Responses, Anthropic, Gemini and a
  Chat probe. These fixtures do not certify an external gateway or provider.
