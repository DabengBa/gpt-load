# Channel Selection Simplification

- **Change:** Fresh new-group imports favor OpenAI Compatible and expose a
  Follow client protocol gateway choice; existing channel presets remain
  searchable.
- **Why:** Make the common connection choices easier to find without removing
  existing presets, and distinguish an OpenAI-compatible API prefix from a
  multi-protocol gateway root.
- **Scope and current owners:** [Groups](../db/pages/groups.md),
  [Model Liveness Probing](../db/features/realistic-probe.md), and the
  [multi-protocol channel contract](../tech/multi-protocol-channel.md).
- **Evidence and delivery:** `web/e2e/astryx-import.spec.ts`,
  `internal/channel/multi_protocol_module_test.go`,
  `internal/control/multi_protocol_integration_test.go`, and
  `internal/execution/bifrost/multi_protocol_runtime_test.go`. Local work on
  `feat/channel-simplification`; not asserted to be deployed.
