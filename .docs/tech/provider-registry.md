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

`embeddingInputMarker` accepts the four OpenAI Embeddings `input` shapes and
describes each as the single equivalent item of the SDK's item list: a string or
a text array becomes one text part, a token array becomes one token part, and a
token matrix becomes one token part per row. The marker never becomes the
request payload: the sanitized client body is sent as `RawRequestBody`, so the
marker cannot widen or narrow what reaches the Provider.

The marker's element dispatch has three branches, keyed on the first byte of the
first array element:

- `"` decodes the whole array as `[]string` and joins it into one text part.
- `[` decodes the array as `[][]int`, one token part per row.
- Any other first byte falls to the default branch, which decodes the whole
  array as `[]int`. A numeric token array — the shape clients actually send —
  takes this branch and becomes one token part.

The marker itself raises an error for only four shapes: a body that does not
decode as a JSON object, an absent `input` member, an `[`-led array that does not
decode as a token matrix, and an `input` whose value is neither a JSON string nor
a JSON array. A text-array element that is not a string is not caught here — a
mixed array such as `["x",1]` is already refused by the dialect validator in
`InspectRequest` before the marker runs, so the marker is not the gate that keeps
it out.

An empty marker is returned — not an error — whenever the selection carries no
part the item list can describe. That covers an empty string, an empty array, an
empty text array, and a token array or matrix that is empty or that fails to
decode into Go `int` at all, which includes a token id beyond the Go integer
range. The empty marker still occupies one slot, so the SDK's empty-input gate
passes and the attempt proceeds. The raw body, already accepted by the dialect
validator as a well-formed integer array, is then forwarded verbatim: GPT-Load
imposes no Go integer bound on token ids and leaves the token id's validity to
the destination.

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
- **Embeddings preflight:** the marker is a preflight description, not a
  validator. Shape errors it does report stop the attempt before dispatch;
  shapes it cannot describe degrade to an empty marker and still forward the raw
  body. Neither path rewrites the client request.
- **Raw authority:** response bytes are taken from the SDK's raw response, and
  the typed response data is cleared before projection and gateway processing,
  so a converted typed value never replaces the observed wire.

Regression coverage: `list_models_test.go` pins the servable-modality rule
(including the non-OpenRouter control), `embedding_wire_test.go` pins
byte-identical request bodies and raw responses for all four input shapes, the
marker's serialized shape, and its rejection cases,
`internal/dialect/openai_embeddings_test.go` pins the accepted token ids
(including one far beyond the Go integer range) and the refused mixed array, and
`runtime_manager_test.go:TestProductionRuntimeManagerUsesNativeProviderListModelsPaths`
proves the OpenRouter behavior through the production runtime, including that
the decisions query is actually issued.
