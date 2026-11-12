---
description: "Fixed-group model liveness probes using preset upstream requests."
kind: technical
topic: model-probing
relations:
  related:
    - db/features/realistic-probe.md
    - tech/model-test-alias.md
code:
  paths:
    - internal/control/model_probe.go
    - internal/control/credential_probe.go
    - internal/execution/bifrost/executor.go
    - internal/execution/bifrost/gateway_probe.go
    - web/src/frontends/astryx/features/models/ModelProbeDialog.tsx
---
# Model Liveness Probing

## Identity and execution

The authenticated management model-probe API accepts explicit group and model
targets. It selects a schedulable credential in the target group, resolves the
channel probe contract, and executes an `OperationProbe`. It does not route a
normal data-plane request or accept a client-provided request body. The group
models page and dispatch center expose the probe dialog, not the former editable
actual-request workspace. Product semantics are owned by
[Model Liveness Probing](../db/features/realistic-probe.md).

Code-owned presets cover the selected generation or supported data protocol.
The model probe applies matching group parameter rules using the external model
name, then model reasoning configuration using the same preparation as the
ordinary gateway. Explicit parameter changes are applied before conversion;
provider SDKs can add their own defaults. In particular, Bedrock conversion can
supply its ordinary 4096-token default when a limit is absent. Probe result
classification and execution identity remain distinct from normal operations.
Credential probes and ModelTrace retain their separate request semantics.

## Recovery, logs and limits

Results report per-target outcomes and link executed attempts to request logs.
Existing concurrent recovery guards remain in force; a model result must not
restore a route after its request-relevant parameter rules or model settings
change. Explicit probing of a disabled group does not make it routable. Probe
logs are not ordinary access-key usage or cost records; providers may charge.
Non-streaming probes do not test streaming, data-plane authentication or quota.

## Verification

- `internal/control/model_probe_test.go` and model-probe wire tests exercise
  target selection, request preparation and recovery with local test fixtures.
- `internal/execution/bifrost/probe_wire_test.go` and converted-path tests
  cover protocol construction and diagnostic response classification.
- `web/e2e/astryx-unified-model-probe.spec.ts` exercises the UI entry points.

Local fixtures do not certify an external provider or a deployment.
