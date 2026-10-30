# Usage Reporting By Request Model

- **Change:** The monitor usage report separates the client-requested model
  from the upstream model and supports exact request-model filtering.
- **Why:** Operators need to distinguish client model names even when requests
  reach the same upstream model.
- **Scope:** Monitor usage reports for administrators and access-key users;
  historical data without a request model remains unknown. The Agent API's
  upstream-model grouping is unchanged.
- **Current owners:** [product behavior](../db/features/usage-request-model.md)
  and [aggregation rules](../tech/usage-accounting.md).
- **Evidence and delivery:** [PR #220](https://github.com/DabengBa/gpt-load/pull/220),
  `web/e2e/astryx-monitor.spec.ts`, `web/scripts/monitor-route.test.ts`, and
  `internal/control/usage_test.go`.
