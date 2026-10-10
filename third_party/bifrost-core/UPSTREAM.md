# third_party/bifrost-core

Pinned local fork of the Bifrost Go SDK used by gpt-load.

| fact | value |
|---|---|
| module path | `github.com/maximhq/bifrost/core` |
| upstream version | `v1.11.3` (`version` file contains `1.11.3`) |
| upstream source | `github.com/maximhq/bifrost`, tag `core/v1.11.3`, commit `40d588d6b371d18d426dca935601e99723e9b0db` |
| root wiring | `go.mod`: `require github.com/maximhq/bifrost/core v1.11.3` + `replace github.com/maximhq/bifrost/core => ./third_party/bifrost-core` |
| nested go.mod / go.sum | byte-identical to upstream v1.11.3 (unchanged from v1.11.0; the root module's MVS selection applies) |

## Why the fork exists

gpt-load must record the raw bytes of every real provider HTTP attempt (request
headers/body, response status/headers/body, termination) for the debug-capture
feature. Bifrost v1.9.0 exposes no request-scoped transport hook: `BifrostConfig`
/ `ProviderConfig` / `NetworkConfig` have no client or RoundTripper field, the
provider construction table is hardcoded, and the plugin hooks only see
structured requests/responses. A minimal local fork is the smallest maintainable
way to add a request-scoped observer without a global registry.

## Fork delta

`gptload-observer.patch` is the complete source diff against pristine v1.11.3,
excluding this document and the patch itself. Apply it with `patch -p1` in a
fresh copy of the upstream module. Its historical name also covers the remaining
cache, tool, stream, and schema changes; Git is the history of those changes.

Modified/added files:

| file | change |
|---|---|
| `providers/utils/http_observer.go` | **new**. `HTTPObserver` contract, optional `HTTPObserverTermination` extension, untyped string context keys (`gpt-load.http-observer`, `gpt-load.http-attempt-id`, `gpt-load.http-observer-state`), per-attempt `httpObserverState` with exactly-once completion, raw body reader wrappers, termination classification, request-only net/http body wrapper (never completes the response state), and teardown completion (`completeIncomplete` / `completePending`) |
| `providers/utils/http_observer_test.go` | **new**. Fork-level tests: unary, SSE byte fidelity, truncated stream, release drain tail, no-response, observer-panic isolation, redirect-chain limitation, idle timeout, cancellation, net/http request/response/eof lifecycle, early-return cancel/deadline/idle completion, 512 KiB error-body boundary |
| `providers/gemini/gemini.go` | materialize the streamed **error** body through the observer before `resp.Body()` on the streaming chat-completion error branch (that branch previously drained the raw stream and bypassed the observer) |
| `providers/gemini/http_observer_stream_error_test.go` | **new**. Proves the gemini streaming error body is observed byte-for-byte and that the parsed provider error is unchanged |
| `providers/utils/utils.go` | instrument `MakeRequestWithContext`, `MakeRequestWithContextFollowRedirects`, `DoStreamingRequest`, `DoHTTPRequest`; `DecompressStreamBody` gained a `context.Context` parameter and wraps the raw stream; `ReleaseStreamingResponse` drains through the same observer and labels its v1.9.0 abandon paths (`ctx.Err()`, parked-after-finish); idle-timeout / cancellation mark the semantic termination |
| `providers/utils/largeresponse.go` | observe the raw stream in `MaterializeStreamErrorBody` / `FinalizeResponseWithLargeDetection` (v1.9.0 hoisted the stream setup, so the wrapper sits at the single read-path entry while teardown still targets the raw stream); `LargeResponseReader.Close` drains through the observer |
| `providers/utils/passthrough_stream.go` | observe `rawBodyStream` argument 1 only (close/idle teardown still targets the unwrapped stream) |
| `providers/utils/utils_test.go` | follow the `DecompressStreamBody` signature change (passes `nil` where no observer is used) |
| `providers/utils/streampoison_test.go` | same `DecompressStreamBody(ctx, resp)` signature follow for the v1.9.0 test |
| `providers/utils/fasthttpreplace_test.go` | v1.9.0 monorepo drift guard: skip `TestFasthttpVersionIsConsistentAcrossModules` when the core/framework/transports root is absent (vendored core-only tree); `repoRoot` refactored to a non-fatal `tryRepoRoot` |
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
2. Compare the currently documented upstream version with the actual local tree,
   excluding `UPSTREAM.md` and the patch. Do not use an old patch as the only
   record of local changes.
3. Three-way merge the old pristine version, the actual local tree, and the new
   pristine version. Review both conflicting and automatically merged overlaps.
   Retain required local behavior, adopt equivalent upstream fixes, and remove
   superseded local implementations. Do not overwrite the fork with a new tree.
4. Update the root require, this document, and the source version; run from the
   repository root so tests use the same module selection as production:
   - `go test -race github.com/maximhq/bifrost/core/providers/utils`
   - `go test -race ./internal/execution/bifrost/`
   - `go build ./...`
   Also run affected provider/schema tests and core dispatch regressions by
   their full `github.com/maximhq/bifrost/core/...` package paths. Root `./...`
   does not include tests in the nested module.
5. Regenerate `gptload-observer.patch` from the new pristine version and confirm
   it reproduces the source tree, excluding this document and the patch. If a
   later fix changes the fork, regenerate it again for the final source.

## Rollback

1. Delete `third_party/bifrost-core`.
2. Remove the `replace github.com/maximhq/bifrost/core => ./third_party/bifrost-core`
   line from the root `go.mod`.
3. `go mod tidy` to restore the upstream `go.sum` entries.
4. `go build ./... && go test ./internal/execution/bifrost/` must pass, and
   `go list -m github.com/maximhq/bifrost/core` must resolve to the module cache.

Because the fork changes no dependency requirements, rollback is a single revert
of the `replace` line plus the directory removal.

## Anthropic prompt-cache integration

GPT-Load opts Anthropic API runtimes into Bifrost prompt-cache injection. The fork
selects the latest cacheable Responses item only for Anthropic, including a
message-level marker on complete function-call outputs. Other providers retain
Bifrost's default first-block strategy. Explicit tool, input, and request markers
suppress injection; tool normalization precedes detection so embedded and
namespace tools participate. OpenAI prompt-cache keys are routing hints, not
Anthropic breakpoints. Injection copies the selected item per attempt.

Regression coverage: `promptcachedispatch_test.go`,
`providers/utils/promptcache_test.go`, and GPT-Load's
`internal/execution/bifrost/anthropic_cache_egress_test.go` exercise isolation,
explicit markers, and two complete tool turns through unary/stream HTTP egress.

## Chat-to-Responses reasoning replay

The current conversion behavior, failure boundaries, and regression coverage
are owned by [Reasoning History Replay](../../.docs/tech/reasoning-replay.md).

The v1.11.3 upgrade uses a three-way merge from pristine v1.11.0 and the actual
local fork. Observer and stream recovery changes remain. Upstream's tool-output
cache injection replaces the equivalent local target/helper; GPT-Load's
Anthropic latest-item selection remains. The MCP test plugin adopts upstream's
tool-name modifier while retaining the local Go syntax adjustments. Automatic
overlaps in observer utilities, Anthropic usage, and cache tests were reviewed.

Integration stream boundary: Azure keeps its preamble check. OpenAI checks the
preamble only when the existing request has SDK fallbacks or the current attempt
has SDK retry budget remaining. GPTLoad clears both, so created/in_progress is
delivered immediately and cancellation and HTTP 200 response.failed retain their
established response metadata. Explicit SDK fallback mocks still recover before
exposing failed-attempt startup events. No new recovery policy or flag is added.
