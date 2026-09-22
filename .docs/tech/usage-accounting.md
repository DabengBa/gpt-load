---
description: "The durable request-log aggregation rules that preserve total duration and observed first-response metrics across hourly usage reports and API projections."
kind: technical
topic: usage-accounting
relations:
  related:
    - db/features/usage-timing-metrics.md
    - tech/agent-api.md
code:
  paths:
    - internal/requestlog/worker.go
    - internal/requestlog/usage_query.go
    - internal/storage/models/request.go
    - internal/storage/migrations/0022_usage_first_response.go
    - internal/control/usage.go
    - internal/agent/usage.go
    - web/src/frontends/classic/app/resources/usage.ts
---
# Usage Accounting

## Scope

- **Owner:** the request-log usage aggregation worker, durable usage-stat
  schema, query service, and usage response projections.
- **Authoritative for:** which request logs enter usage aggregates, how duration
  and first-response samples are counted, how averages are calculated, and how
  invalid or unavailable values are handled.
- **Excludes:** request-log retention, provider behavior, route selection, and
  the product-facing timing explanation owned by [Usage Duration And
  First-Response Metrics](../db/features/usage-timing-metrics.md).

## Responsibility

The usage pipeline turns eligible completed request logs into durable hourly
aggregates and serves those aggregates as report summaries, time-series
points, breakdown rows, and Agent API projections. It preserves totals and
sample counts separately so consumers can distinguish an observed zero from a
missing observation.

## Architecture And Constraints

Usage aggregation excludes rows with no upstream attempt, probe operations,
and standalone web-search operations. Each remaining row is assigned to the
hour containing its completion time and grouped by access key, channel, group,
credential, and upstream model.

Duration is available for every eligible row. First-response timing is
optional: a row contributes one first-response sample only when its persisted
`FirstResponseMs` value is present. Non-streaming requests and other rows
without that value remain valid usage rows with zero first-response samples.

## Core Implementation

- `internal/requestlog/worker.go` builds one journal delta per eligible request,
  adds duration for every delta, and adds first-response total/count only for
  observed values. Journal upserts and usage-stat updates carry both totals and
  sample counts.
- `internal/storage/migrations/0022_usage_first_response.go` adds non-null,
  zero-default first-response total/count columns to `usage_aggregation_journal`
  and `usage_stats` for SQLite, then validates columns, defaults, constraints,
  and stored values.
- `internal/requestlog/usage_query.go` sums timing totals and sample counts
  across buckets and dimensions. The average for a metric is its total divided
  by its corresponding sample count only when that count is positive.
- `internal/control/usage.go` and `internal/agent/usage.go` project timing
  totals and counts after rejecting negative values, unsafe integers, and
  sample counts larger than request counts. The web usage resource accepts the
  two timing sort fields and exposes both aggregate pairs.

## Related Product Semantics And Binding Points

- Product behavior is owned by [Usage Duration And First-Response
  Metrics](../db/features/usage-timing-metrics.md).
- Agent API projection details are owned by [Agent Control-Plane
  API](agent-api.md).
- Code binding: `internal/requestlog/worker.go:buildUsageStatDeltas`
- Code binding: `internal/requestlog/usage_query.go:queryUsage`
- Code binding: `internal/storage/migrations/0022_usage_first_response.go:Up0022`

## Operations, Calculation, And Failure Boundaries

- **Duration:** add each eligible row's non-negative `DurationMs` to
  `DurationMsTotal` and increment `DurationSampleCount` once.
- **First response:** when `FirstResponseMs` is present, add its non-negative
  value to `FirstResponseMsTotal` and increment
  `FirstResponseSampleCount` once. Otherwise add neither value nor sample.
- **Average:** return `total / sample_count` only when `sample_count > 0`;
  consumers render no first-response sample as undefined rather than zero.
- **Integrity:** checked integer addition rejects overflow; aggregate mapping
  rejects negative values, unsafe JavaScript integers, and sample counts above
  request counts. Query integrity checks reject corrupt persisted timing rows.
- **Migration:** existing rows receive zero defaults for the new first-response
  fields. They therefore carry no historical first-response observation unless
  a later request contributes one; the migration does not invent timing data.
- **Degradation:** missing first-response observations reduce the denominator
  only for that metric. They do not remove the request from duration, status,
  token, or cost aggregates.
