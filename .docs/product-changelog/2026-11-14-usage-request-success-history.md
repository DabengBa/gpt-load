# Usage Request Success History

- **Change:** Monitor usage breakdown rows show request success history by
  time bucket, with expandable counts and empty intervals distinguished from
  failures.
- **Why:** Operators need to see when actual request outcomes changed, not
  just the success rate over an entire report period.
- **Scope:** Administrator and access-key monitor usage reports; history
  follows the report filters and access scope and is not an uptime or probe
  measurement. The Agent API does not request this history.
- **Current owners:** [product behavior](../db/features/usage-request-success-history.md)
  and [aggregation rules](../tech/usage-accounting.md).
- **Evidence and delivery:** Current working-tree implementation and tests in
  `internal/requestlog/usage_history.go`, `internal/requestlog/usage_history_test.go`,
  `internal/control/usage_test.go`, and `web/e2e/astryx-monitor.spec.ts`;
  delivery is not yet committed.
