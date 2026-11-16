# Historical Request Log Filtering

- **Change:** Request-log group and client-model choices come from matching
  history and apply immediately when selected or cleared.
- **Why:** Operators need to find past requests even when groups or models are
  no longer in current configuration, without a separate Apply step.
- **Scope:** Administrator request-log filters; access-key filtering retains
  its existing scoped text-input workflow.
- **Current owners:** [product behavior](../db/features/request-log-filtering.md)
  and [facet query contract](../tech/request-log-facets.md).
- **Evidence and delivery:** Working-tree implementation and tests in
  `internal/requestlog/query_facets.go`, `internal/requestlog/facets_test.go`,
  `internal/control/request_log_facets_test.go`, and
  `web/e2e/astryx-log-facets.spec.ts`; not yet committed or deployed.
