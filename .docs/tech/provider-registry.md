---
description: "Own the published model catalog and bound Embeddings selection that GPT-Load derives from a Provider response, including which OpenRouter entries are dropped and why the raw body stays the wire authority."
kind: technical
topic: model-discovery
relations:
  related:
    - tech/provider-feedback.md
    - tech/debug-communications.md
    - tech/reasoning-replay.md
code:
  paths:
    - internal/execution/bifrost/list_models.go
    - internal/execution/bifrost/list_models_test.go
    - internal/execution/bifrost/embeddings.go
    - internal/execution/bifrost/embedding_wire_test.go
---
# Provider Model Registry And Embeddings Selection

## Scope

- **Owner:** the Bifrost-backed ListModels sanitizer and the Embeddings request
  marker.
- **Authoritative for:** which Provider models GPT-Load publishes to a client,
  how an Embeddings selection is described to the SDK preflight, and what
  remains upstream authority.
- **Excludes:** route selection and credential health
  ([Provider Feedback](provider-feedback.md)), raw capture boundaries
  ([Debug Communication Capture](debug-communications.md)), and protocol
  conversion of the same SDK ([Reasoning History Replay](reasoning-replay.md)).

## Responsibility

`ListModels` returns a Provider catalog to a client, and Embeddings forwards a
client selection to a Provider. Both cross the Bifrost SDK boundary with a typed
value, and in both cases the typed value is only the SDK's preflight input: the
published answer for a model list is the sanitized catalog, and the upstream
answer for an Embeddings call is the untouched raw body.

## Architecture And Constraints

The sanitizer never mutates the Provider response. It clones the response,
applies the Provider-specific catalog rule, strips the `provider/` id prefix so
clients see bare model ids, and clears extra fields and key statuses before
projection to the OpenAI, Anthropic, or Gemini client shape.

The Embeddings marker never becomes the request payload. `buildEmbeddingRequest`
sanitizes the client body, builds a typed marker from `input`, and sends that
sanitized body as `RawRequestBody` with per-request raw override enabled, so a
selection GPT-Load cannot represent in Go (for example token ids outside Go
integer bounds) is still forwarded verbatim.

## Core Implementation

### Published model catalog

`sanitizeListModelsResponse` publishes the whole catalog for every Provider
except OpenRouter. `withoutOpenRouterUnservedModels` narrows OpenRouter because
the SDK merges OpenRouter's decisions catalog
(`/v1/models?output_modalities=decisions`) into the same list, and no GPT-Load
route serves the native decisions endpoint.

- A model is published when at least one declared output modality is servable:
  `text`, `embedding`, or `embeddings`, compared case-insensitively after
  trimming surrounding space.
- A mixed entry such as `["decisions","text"]` is therefore published.
- An entry whose modalities are only unservable values is dropped.
- An entry with no architecture at all is dropped: the merged decision items
  decode without one, so it cannot be shown to be chat or embed.
- A non-OpenRouter catalog is published unchanged, including embedding-only
  models.

### Embeddings selection marker

`embeddingInputMarker` builds the typed input the SDK requires while the
sanitized client body is sent as `RawRequestBody`, so the marker never replaces
what the Provider receives. It is not a validator: only malformed bodies produce
an error, and every selection the item list cannot express is described by one
placeholder item.

The whole body must decode as a JSON object, and `input` must be present and
non-empty; those three failures return an error. A JSON string `input` decodes as
one text part. A non-string `input` must decode as a JSON array, otherwise the
marker returns an error. The first array element then selects the branch:

- `"` decodes the array as `[]string` into one text part. A decode failure
  returns an error.
- `[` decodes the array as `[][]int`, one token part per row.
- Any other first byte decodes the array as `[]int`, one token part.

A numeric token array, which is what clients send, therefore takes the default
branch. A token id beyond the Go integer range fails the `[]int` decode, or the
matrix decode in the `[` branch.

Two SDK rules decide which of those results can be expressed at all: an input
slice must be non-empty, and a text part must carry non-empty text while a token
part must carry at least one token id. Anything that would violate them — an
empty string, an empty array, an empty text within an array, an empty matrix row,
or a decode failure — is reported as the placeholder item instead: one text part
holding a single space, which satisfies both rules.

The placeholder is a description, not a request. Because the marker is only the
SDK's typed preflight and the raw body is the wire authority, a selection
described by the placeholder still reaches the Provider byte-for-byte unchanged.
GPT-Load therefore imposes no Go integer bound on token ids, and the destination
decides whether a selection it cannot represent locally is valid.

## Related Product Semantics And Binding Points

- Client protocol scope for Embeddings is described in the repository README
  scope section, not in this owner.
- Code binding:
  `internal/execution/bifrost/list_models.go:withoutOpenRouterUnservedModels`
- Code binding: `internal/execution/bifrost/list_models.go:servableOutputModality`
- Code binding: `internal/execution/bifrost/embeddings.go:embeddingInputMarker`
- Code binding: `internal/execution/bifrost/embeddings.go:enableEmbeddingWireFidelity`

## Operations, Calculation, And Failure Boundaries

- **Catalog degradation:** dropping an unclassifiable OpenRouter entry narrows
  the published list. It does not change routing, health, or any stored model
  selection; a group can still be configured with a model the list omits.
- **Embeddings preflight:** the marker either describes the selection or carries
  the placeholder. Both are accepted by the SDK, so a malformed body is the only
  local failure and every valid body reaches the Provider unchanged; no path
  rewrites the client request.
- **Raw authority:** response bytes are taken from the SDK's raw response, and
  the typed response data is cleared before projection and gateway processing,
  so a converted typed value never replaces the observed wire.

Regression coverage: `list_models_test.go` pins the servable-modality rule
(including the non-OpenRouter control), `embedding_wire_test.go`
pins byte-identical request bodies and raw responses for all four input shapes,
the marker's serialized shape, the placeholder accepted by the SDK preflight for
every undescribable selection, and the rejection cases,
`embeddings_test.go` proves through the real Runtime and SDK that each of those
selections still reaches the Provider,
`internal/dialect/openai_embeddings_test.go` pins the accepted token ids
(including one far beyond the Go integer range) and the refused mixed array, and
`runtime_manager_test.go:TestProductionRuntimeManagerUsesNativeProviderListModelsPaths`
proves the OpenRouter behavior through the production runtime, including that
the decisions query is actually issued.
