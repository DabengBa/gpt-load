# third_party/bifrost-core

Pinned local fork of the Bifrost Go SDK used by gpt-load.

| fact | value |
|---|---|
| module path | `github.com/maximhq/bifrost/core` |
| baseline version | `v1.8.4` (`version` file contains `1.8.4`) |
| upstream source of truth | `/home/allen/go/pkg/mod/github.com/maximhq/bifrost/core@v1.8.4` (module cache, mode `555`, never written) |
| baseline tree digest | sha256 of `sha256sum` over every `*.go` file, sorted by path: `d55de55664d0354795af5882c39a326292131a1ea7ab6ed4b154a225aecd4413` |
| root wiring | `go.mod`: `require github.com/maximhq/bifrost/core v1.8.4` + `replace github.com/maximhq/bifrost/core => ./third_party/bifrost-core` |
| nested go.mod / go.sum | byte-identical to upstream v1.8.4 (the dependency graph is unchanged; only the root module's MVS selection applies) |

## Why the fork exists

gpt-load must record the raw bytes of every real provider HTTP attempt (request
headers/body, response status/headers/body, termination) for the debug-capture
feature. Bifrost v1.8.4 exposes no request-scoped transport hook: `BifrostConfig`
/ `ProviderConfig` / `NetworkConfig` have no client or RoundTripper field, the
provider construction table is hardcoded, and the plugin hooks only see
structured requests/responses. A minimal local fork is the smallest maintainable
way to add a request-scoped observer without a global registry.

## Fork delta

`gptload-observer.patch` is the complete diff against the pristine v1.8.4 tree.
It was generated with `diff -ruN` and verified to reproduce this directory
exactly when applied to a fresh copy of the module cache.

Modified/added files:

| file | change |
|---|---|
| `providers/utils/http_observer.go` | **new**. `HTTPObserver` contract, optional `HTTPObserverTermination` extension, untyped string context keys (`gpt-load.http-observer`, `gpt-load.http-attempt-id`, `gpt-load.http-observer-state`), per-attempt `httpObserverState` with exactly-once completion, raw body reader wrappers, termination classification, request-only net/http body wrapper (never completes the response state), and teardown completion (`completeIncomplete` / `completePending`) |
| `providers/utils/http_observer_test.go` | **new**. Fork-level tests: unary, SSE byte fidelity, truncated stream, release drain tail, no-response, observer-panic isolation, redirect-chain limitation, idle timeout, cancellation, net/http request/response/eof lifecycle, early-return cancel/deadline/idle completion, 512 KiB error-body boundary |
| `providers/gemini/gemini.go` | materialize the streamed **error** body through the observer before `resp.Body()` on the streaming chat-completion error branch (that branch previously drained the raw stream and bypassed the observer) |
| `providers/gemini/http_observer_stream_error_test.go` | **new**. Proves the gemini streaming error body is observed byte-for-byte and that the parsed provider error is unchanged |
| `providers/utils/utils.go` | instrument `MakeRequestWithContext`, `MakeRequestWithContextFollowRedirects`, `DoStreamingRequest`, `DoHTTPRequest`; `DecompressStreamBody` gained a `context.Context` parameter and wraps the raw stream; `ReleaseStreamingResponse` drains through the same observer; idle-timeout / cancellation mark the semantic termination |
| `providers/utils/largeresponse.go` | observe the raw stream in `MaterializeStreamErrorBody` / `FinalizeResponseWithLargeDetection`; `LargeResponseReader.Close` drains through the observer |
| `providers/utils/passthrough_stream.go` | observe `rawBodyStream` argument 1 only (close/idle teardown still targets the unwrapped stream) |
| `providers/utils/utils_test.go` | follow the `DecompressStreamBody` signature change (passes `nil` where no observer is used) |
| `providers/{anthropic,azure,cohere,elevenlabs,gemini,huggingface,mistral,openai,replicate,sarvam,vllm}/*.go` | mechanical `DecompressStreamBody(ctx, resp)` call-site update |

The observer is purely observational: it never replaces `resp.BodyStream()`,
never buffers the request body, and every callback is panic-recovered so a
broken observer cannot change a provider request's result, retry, cancellation
or error classification.

### Continuation fixes (raw-provider-capture_20260912T212852+0800, U002)

Three reviewer-confirmed high findings were fixed inside this fork:

1. The net/http **request** body wrapper records request bytes only; `Close` no
   longer completes the response state (net/http closes the request body before
   the response is consumed). Response completion belongs to the response body
   reader / transport error.
2. Every stream teardown owner (cancellation, deadline, idle timeout,
   `ReleaseStreamingResponse` skip, `LargeResponseReader.Close` skip) now emits
   the exactly-once incomplete termination for a parser that returned early and
   left no pending Read. Completion is observation-only: no extra close, drain,
   delay or cancellation change.
3. `MaterializeStreamErrorBody` keeps its 512 KiB parser cap but drains the
   remaining stream (decompressed layer and raw layer) to the real transport
   termination, so raw observation is never truncated and never misses EOF.

### Present-but-unreachable boundary (redirect helper)

`MakeRequestWithContextFollowRedirects` calls fasthttp `Client.DoRedirects`,
whose per-hop loop (`doRequestFollowRedirects`) is unexported inside the
fasthttp dependency. A Bifrost-only fork therefore cannot emit one event per
intermediate 3xx hop; only the initial request and the final response are
observable. This is pinned by `TestRedirectChainOnlyInitialAndFinalAreObservable`.

Per the gpt-load plan decision **R-B** (2026-09-12), this helper is recorded as
**present-but-unreachable**: both call sites (`GeminiProvider.VideoDownload` and
the `GeminiProvider.Passthrough` `:download` branch) are unreachable from the
current production route registry, so no production Provider attempt uses it.
No second fasthttp fork is added and redirect semantics are unchanged. The
per-hop gap is deliberately not claimed as covered; if a future route makes
either call site reachable, either the plan must narrow the matrix or a pinned
`third_party/fasthttp` fork with an additive per-hop hook must be added.

## Upstream sync procedure

1. `go mod download github.com/maximhq/bifrost/core@<new version>` (module cache
   stays read-only).
2. Copy the new pristine tree over this directory (or start from a clean export).
3. Re-apply `gptload-observer.patch`; resolve conflicts in the six files above.
4. Update this file (version, digest), then run from the repository root:
   - `go test -race github.com/maximhq/bifrost/core/providers/utils`
   - `go test -race ./internal/execution/bifrost/`
   - `go build ./...`
5. Regenerate `gptload-observer.patch` and confirm it reproduces the tree.

## Rollback

1. Delete `third_party/bifrost-core`.
2. Remove the `replace github.com/maximhq/bifrost/core => ./third_party/bifrost-core`
   line from the root `go.mod`.
3. `go mod tidy` to restore the upstream `go.sum` entries.
4. `go build ./... && go test ./internal/execution/bifrost/` must pass, and
   `go list -m github.com/maximhq/bifrost/core` must resolve to the module cache.

Because the fork changes no dependency requirements, rollback is a single revert
of the `replace` line plus the directory removal.
