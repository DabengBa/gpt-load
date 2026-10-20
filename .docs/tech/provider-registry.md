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
token matrix becomes one token part per row. An empty selection produces an
empty marker, which satisfies the SDK's length gate; the raw body remains the
wire authority, so the marker cannot widen or narrow what is sent.

A body with no `input` member, a `null` or non-array/non-string `input`, or an
array whose first element is neither a string nor an array is rejected before
dispatch.

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
- **Embeddings preflight:** an unrepresentable input is rejected before dispatch
  instead of being rewritten, so the client sees a local validation failure
  rather than a silently altered upstream request.
- **Raw authority:** response bytes are taken from the SDK's raw response, and
  the typed response data is cleared before projection and gateway processing,
  so a converted typed value never replaces the observed wire.

Regression coverage: `list_models_test.go` pins the servable-modality rule
(including the non-OpenRouter control), `embedding_wire_test.go` pins
byte-identical request bodies and raw responses for every input shape plus the
marker's serialized shape and rejection cases, and
`runtime_manager_test.go:TestProductionRuntimeManagerUsesNativeProviderListModelsPaths`
proves the OpenRouter behavior through the production runtime, including that
the decisions query is actually issued.
