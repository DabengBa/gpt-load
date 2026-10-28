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
    - internal/storage/migrations/0027_success_usage_efficiency.go
    - internal/control/usage.go
    - internal/agent/usage.go
    - web/src/shared/control/resources/usage.ts
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

Usage aggregation excludes rows with no upstream attempt and `probe`
operations (`internal/requestlog/worker.go:usageAggregationExcluded`);
`modeltrace` challenges are real model requests and stay in the aggregates.
Each remaining row is assigned to the hour containing its completion time and
grouped by access key, channel, group, credential, and upstream model.

Request token and estimated-cost amounts include reported billed usage from
earlier failed attempts as well as the terminal attempt, each priced with its
own frozen snapshot. The request still contributes one request count and one
duration sample under its terminal-attempt dimensions. Thus a model/group cost
breakdown attributes the whole request amount to its terminal route; it is not
a per-provider invoice breakdown. Individual attempt receipts retain the
actual route, quantities and frozen prices for diagnosis.

The mapper checked-sums billed attempt quantities and amounts, excludes the
terminal attempt from the additional sum, and rejects overflow. Existing
request-idempotent transactional journals apply the combined totals once even
when the same event is persisted again. Missing usage is not reconstructed;
partial or unpriced evidence remains explicit. Runtime access-key quota
settlement uses the same individually priced attempt sum. Calculation details
are owned by [Frozen Model Pricing](frozen-model-pricing.md).

Duration is available for every eligible row. First-response timing is
optional: a row contributes one first-response sample only when its persisted
`FirstResponseMs` value is present. The recorder captures that value only for
streaming requests (`requestRecorder.recordFirstResponse` returns early
otherwise), so non-streaming requests and every other row without an
observation remain valid usage rows with zero first-response samples.

Successful-request token efficiency is tracked independently of all-request
output and duration. A request contributes to this metric only when it is
successful, usage is complete, output tokens are positive, and duration is
positive. Its success output and duration are added together and the sample
count is incremented once. The aggregate is valid only when the sample count
equals `success_count`, is positive, and success duration is positive; the
reported rate is `success_output_tokens * 1000 /
success_duration_ms_total` tokens per second. Otherwise it is unavailable.
Success-only historical values are not inferred from all-request aggregates.

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
  The monitor table displays both averages in seconds with one decimal place
  (`1234 ms` renders as `1.2 s`); `—` stands for a missing sample, not a
  zero-latency measurement.
- **Success efficiency:** complete success-only totals and sample counts are
  included in summary, series, and breakdown aggregates. The efficiency rate is
  output tokens per successful request divided by average duration per
  successful request, equivalently `success_output_tokens * 1000 /
  success_duration_ms_total`. It requires positive output, a sample count equal
  to `success_count`, and positive duration; unavailable rows sort after
  available rows in either direction.
- **Cache rate:** `cache_read_tokens / (uncached_input_tokens +
  cache_read_tokens)`; a zero denominator is unavailable and sorts last in
  either direction.
- **Breakdown sorting:** the monitor table sorts all displayed columns through
  the usage query before pagination, including total, failed, normal, slow,
  and faulty upstream attempt counts. Clicking a different column starts in
  descending order; clicking the active column reverses the direction and
  resets pagination to page one. The URL preserves the selected sort.
  `average_first_response_ms` is additionally sample-aware: rows without a
  first-response sample stay after rows with samples in both directions, so a
  row with no observation never ranks as the fastest or slowest.
- **Integrity:** checked integer addition rejects overflow; aggregate mapping
  rejects negative values, unsafe JavaScript integers, and sample counts above
  request counts. Query integrity checks reject corrupt persisted timing rows.
- **Migration:** existing rows receive zero defaults for the new first-response
  fields. They therefore carry no historical first-response observation unless
  a later request contributes one; the migration does not invent timing data.
- **Eligibility:** only `AttemptCount > 0` and non-`probe` rows produce usage
  deltas. Every other operation the execution contract defines is counted
  normally, so a new local-only operation must be added to that predicate
  explicitly if it should not appear in reports.
- **Degradation:** missing first-response observations reduce the denominator
  only for that metric. They do not remove the request from duration, status,
  token, or cost aggregates.
