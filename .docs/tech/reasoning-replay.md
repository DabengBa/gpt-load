---
description: "Define reasoning-history serialization and tool pairing when Bifrost converts between Chat Completions and Responses."
kind: technical
topic: protocol-conversion
relations:
  related:
    - tech/reasoning-policy.md
    - tech/debug-communications.md
code:
  paths:
    - third_party/bifrost-core/schemas/mux.go
    - third_party/bifrost-core/schemas/responses.go
    - third_party/bifrost-core/providers/openai/chat.go
    - internal/execution/bifrost/reasoning_replay_test.go
    - internal/execution/bifrost/responses_to_chat_replay_test.go
---
# Reasoning History Replay

## Responsibility

Bifrost protocol conversion carries reasoning history and associated tool calls
between Chat Completions and Responses. This boundary is separate from the
[route-entry reasoning policy](reasoning-policy.md), which chooses request
effort rather than transforming conversation history. The local fork and its
upgrade procedure are recorded in
[UPSTREAM.md](../../third_party/bifrost-core/UPSTREAM.md).

## Architecture And Constraints

The root module requires Bifrost core v1.11.3 and replaces it with
`third_party/bifrost-core`. Its schema converters and provider serializers
determine the actual converted payload. Native passthrough does not repair
previously saved client history.

Protocol conversion preserves tool-call/result pairing, but does not promise
lossless reasoning-state transfer or encrypted reasoning portability between
providers. A provider's acceptance of a field remains a separate constraint.

## Core Implementation

### Chat Completions To Responses

Converted reasoning items omit `role` and always contain a `summary` array.
When the source has only raw reasoning, `summary` is `[]`; the original text
remains in `content` as `reasoning_text`. Existing real summaries and encrypted
content are preserved. Unary conversion and streaming added/done/terminal
output, including incomplete responses, use the same reasoning item shape.

A client can therefore replay newly converted items verbatim to a Responses
route without introducing the unsupported `role` or missing `summary` that
previously caused rejection. The converter does not rewrite history already
stored by a client.

### Responses To OpenAI-Compatible Chat Completions

Raw `reasoning_text` becomes assistant `reasoning_content`. The OpenAI Chat
message builder does not serialize Responses summaries or encrypted details,
and does not recast them as raw reasoning. Tool calls and results retain their
matching call IDs, arguments, and result content.

A destination that rejects `reasoning_content` can return HTTP 400. The adapter
propagates that error instead of silently dropping reasoning. Target-specific
changes require evidence of the target's behavior; removing reasoning globally
would break destinations that require it on tool-call turns.

For OpenAI-compatible Chat targets, function tool output text blocks are joined
in order into a string; an empty block array becomes an empty string. Results
containing non-text blocks cannot be represented without losing content and are
rejected before dispatch with `critical_semantic_loss`. This rejects the
converted candidate while preserving the original request for another route.
Best-effort filtering of unsupported tool declarations does not permit dropping
tool result images, files, or audio.

### Grok Prompt Cache Identity

The Grok bridge scopes continuity by credential and model, and uses the attempt
identity when no continuity is available. For OpenAI Chat and Responses input,
the embedded executor removes caller continuity fields and writes its stable
gateway identity to `prompt_cache_key` in both payload and original request.
CPA uses that key for the final request body and conversation header. The bridge
does not pass this caching identity as trusted execution session metadata, which
would enable automatic encrypted reasoning replay into later requests.
Other input formats retain their existing metadata behavior.

## Failure Boundaries And Verification

- `third_party/bifrost-core/schemas/reasoningreplay_test.go` checks serialized
  unary and streaming items, terminal completed/incomplete output, and Chat
  replay tool pairing.
- `internal/execution/bifrost/reasoning_replay_test.go` sends real converted
  unary and stream-done output to a strict local native Responses HTTP endpoint
  without repairing or dropping the captured reasoning item.
- `internal/execution/bifrost/responses_to_chat_replay_test.go` checks the
  actual Chat HTTP payload, summary/encrypted omission, tool pairing, and
  propagation of a strict destination's HTTP 400 rejection.

These local tests prove the covered serialization and pairing behavior; they
do not establish acceptance by every live provider. For an actual destination,
inspect all observed provider attempts through
[debug communication capture](debug-communications.md).

`internal/execution/bifrost/ahead_tool_output_test.go` verifies string-only tool
output serialization and rejection before dispatch. Grok two-turn unary and
stream captures are covered by
`third_party/cpaembedded/embedded/ahead_grok_cache_test.go`; credential, model,
and attempt scope checks also run in
`internal/execution/cpa/ahead_grok_scope_test.go`.
