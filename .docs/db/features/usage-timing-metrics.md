---
id: feature.usage-timing-metrics
type: feature
name: Usage duration and first-response metrics
---
# Usage Duration And First-Response Metrics

## ID Explanation

`feature.usage-timing-metrics` owns the user-visible timing measures in usage
reports: total request duration and time to the first response. It covers their
meaning, display, and sorting behavior; durable aggregation rules and API
projection constraints belong to the technical usage-accounting owner.

## Purpose

Operators need to distinguish a request that completes slowly from one that
takes a long time to produce its first response. The usage report makes both
signals available for the selected time range and current report scope without
pretending that an unobserved first response was instantaneous.

## User-Visible Contract

- Usage breakdowns expose average total duration and average first-response
  time in milliseconds alongside request, success, failure, token, and cost
  measures.
- Average duration is calculated from the total duration and its observed
  request sample count.
- Average first-response time is calculated only from requests with an
  observed first-response measurement. A group with no such observations shows
  `—`, not `0 ms`.
- The breakdown can be sorted by either timing measure. Rows without a
  first-response sample remain after rows with samples in both sort directions;
  they are not treated as zero-latency rows.
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

## Boundaries

- This feature explains report-level timing aggregates, not per-request trace
  details or provider latency guarantees.
- It does not change request routing, retries, streaming behavior, token
  accounting, cost calculation, or log retention.
- A missing first-response sample is a real absence of measurement, not a
  performance score or an inferred zero.
