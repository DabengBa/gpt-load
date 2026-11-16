---
description: Historical request-log group and client-model facet queries, access boundaries, filtering semantics, and failure behavior.
kind: technical
topic: request-log-filtering
code:
  paths:
    - internal/requestlog/query_facets.go
    - internal/requestlog/query_list.go
    - internal/control/request_logs.go
    - web/src/shared/control/resources/request-logs.ts
---
# Request Log Facets

## Responsibility

The request-log read path supplies historical group and client-model choices
for the [request-log filtering feature](../db/features/request-log-filtering.md).
It does not own log retention, usage aggregation, or current group configuration.

## Architecture And Constraints

`GET /api/logs/facets` is administrator-only. It uses the request-log list
query parser and validation, and returns the standard success envelope with
`data.groups` (`id`, optional `name`) and `data.client_models` (`value`).
Empty results are arrays; there are no counts. Valid `limit` and `cursor`
parameters are parsed but do not restrict the facet result. Request time
filters use the list's inclusive `from_ms` and exclusive `to_ms` bounds.

Each dimension uses all other applied list filters but excludes its own:

- Groups come from historical attempts with a positive group ID. Remaining
  attempt filters must match the same candidate attempt. They are grouped by
  ID; the name is `MAX(group_name)` over matching historical snapshots, not
  necessarily the latest or current name.
- Nonempty client models come from matching request logs. When attempt filters
  are present, at least one attempt of the request must satisfy all of them
  together. A request without attempts can contribute a model only when no
  attempt filter is active.

The two dimension queries execute separately, so concurrent writes need not
produce a shared snapshot. Neither query uses the list page cursor or limit;
the response is deduplicated and ordered by group ID or client model.

## Core Implementation And Failure Boundaries

- `internal/requestlog/query_facets.go:Facets` reuses request-level predicates
  and the same attempt-condition builder as the list in
  `internal/requestlog/query_list.go`. It does not introduce a migration or
  index for facets.
- `internal/control/request_logs.go:handleRequestLogFacets` rejects access-key
  principals, validates the query, and propagates database errors through the
  service error path. A reader without facet support returns an internal error.
- The web client requests facets for applied administrator filters without a
  cursor; on failure, it surfaces the option error instead of using current
  group configuration as historical evidence. Access-key sessions do not
  request this endpoint.
- The client model selector only offers historical values accepted unchanged
  by the existing route text validator: nonempty, trimmed, no Unicode control
  characters, and at most 255 UTF-8 bytes. Invalid historical values remain in
  the underlying logs and API result; they are not normalized into a different
  exact-match filter.
- `internal/requestlog/facets_test.go` covers historical data and matching;
  `internal/control/request_log_facets_test.go` exercises authorization and
  SQLite through HTTP. `web/e2e/astryx-log-facets.spec.ts` exercises the UI
  against mocked HTTP. These do not by themselves establish real browser-to-
  database end-to-end behavior or production-scale performance.
