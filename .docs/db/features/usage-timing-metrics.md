---
id: feature.usage-timing-metrics
type: feature
name: Usage timing and efficiency metrics
---
# Usage Timing And Efficiency Metrics

## ID Explanation

`feature.usage-timing-metrics` owns the user-visible timing and efficiency
measures in usage reports: total request duration, time to the first response,
cache rate, and successful-request token efficiency. It covers their meaning,
display, unavailable state, and sorting behavior; durable aggregation rules
and API projection constraints belong to the technical usage-accounting owner.

## Purpose

Operators need to distinguish a request that completes slowly from one that
takes a long time to produce its first response. The usage report makes both
signals available for the selected time range and current report scope without
pretending that an unobserved first response was instantaneous.

## User-Visible Contract

- Usage breakdowns expose average total duration and average first-response
  time alongside request, success, failure, token, and cost measures. Both
  averages are displayed in seconds with one decimal place.
- A first-response measurement exists only for streaming requests. A group
  that served only non-streaming requests therefore reports no first-response
  average at all, which is a real absence of observation rather than fast
  responses.
- Average duration is calculated from the total duration and its observed
  request sample count.
- Average first-response time is calculated only from requests with an
  observed first-response measurement. A row with no such observations shows
  `—`, not `0 s`.
- The breakdown exposes cache rate as a percentage of the input covered by
  cache reads, and token efficiency as output tokens per second for successful
  requests. These measures are shown on each applicable breakdown row.
- A measure shows `—` when the selected row has no usable observation for that
  measure. This is an unavailable state, not a zero value. The cache-rate and
  token-efficiency totals are also shown as `—` when the report does not offer
  a meaningful combined value.
- The breakdown can be sorted by any displayed measure, including cache rate
  and token efficiency. Selecting a different column starts with descending
  order; selecting the active column reverses the direction and returns to the
  first page. Rows with unavailable timing or efficiency measures remain after
  rows with available values in either direction.
- The timing measures follow the selected report range, grouping, filters, and
  access scope just like the other usage aggregates.

## Acceptance Workflows

### Compare completion time and time to first response

- **Role and purpose:** An administrator wants to determine whether a model is
  slow to start responding or slow to finish.
- **Entry and action:** Open the usage report for a selected range and compare
  the average duration and average first-response columns; sort either column
  in both directions.
- **Expected result:** Each row shows the appropriate millisecond average,
  rows with observed first-response samples sort by that average, and rows
  without observations remain visibly undefined and grouped after sampled
  rows.
- **Failure signal:** A row without a first-response observation must not show
  `0 ms` or move ahead of rows with measured response times merely because its
  sample count is zero.

### Compare cache coverage and successful-request efficiency

- **Role and purpose:** An administrator wants to compare how much input was
  served from cache and how quickly successful requests produced output.
- **Entry and action:** Open the usage report for a selected range, inspect the
  cache-rate and token-efficiency columns, and sort either column in both
  directions.
- **Expected result:** Applicable rows show a percentage for cache rate and a
  per-second value for successful-request token efficiency. Rows without a
  usable observation show `—`, and the selected sort order is reflected in the
  URL and table.
- **Failure signal:** An unavailable value must not be presented as zero or be
  ranked ahead of rows with an available value.

## Boundaries

- This feature explains report-level timing aggregates, not per-request trace
  details or provider latency guarantees.
- It does not change request routing, retries, streaming behavior, token
  accounting, cost calculation, or log retention.
- A missing first-response sample is a real absence of measurement, not a
  performance score or an inferred zero.
- Cache rate and successful-request token efficiency are report measures; the
  aggregation and validity rules that produce them belong to the technical
  usage-accounting owner.
