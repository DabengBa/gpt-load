---
id: feature.usage-request-success-history
type: feature
name: Usage request success history
---
# Usage Request Success History

## ID Explanation

`feature.usage-request-success-history` owns the user-visible history of
request outcomes on each monitor usage breakdown row. Aggregation and API
projection rules belong to [Usage Accounting](../../tech/usage-accounting.md).

## Purpose

Operators can see how actual request outcomes for a model and route changed
over the selected report period, rather than relying only on one overall
success rate.

## User-Visible Contract

- Each visible usage breakdown row has a history divided into the report's
  time buckets. The history follows the selected range, filters, row identity,
  and access scope of the report.
- Each bucket distinguishes all successful requests, a mixture of successes
  and failures, all failures, and no requests. Expanding the history shows the
  time interval and, when requests exist, the successful and total request
  counts and success percentage. The history can be opened with the keyboard.
- A bucket with no requests, including an attempt-only row, is shown as no
  requests, not as a failure or a zero-percent success rate.
- Administrator histories follow the full breakdown row identity. Access-key
  users see only their own usage, aggregated over route dimensions hidden from
  them; they cannot inspect another key's history.

## Acceptance Workflows

### Inspect a model's request outcomes over time

- **Role and purpose:** An operator wants to see when requests for a model had
  failures within the selected report period.
- **Entry and action:** Open the monitor usage report, apply the desired range
  and filters, and expand a row's request history, including by keyboard.
- **Expected result:** Each interval reports its request outcomes or no
  requests, with counts and a success percentage for intervals with requests.
  The history describes the same row and scope as the breakdown.
- **Failure signal:** An empty interval is presented as a failed request, or
  a row includes outcomes belonging to a different identity or access key.

## Boundaries

- This is history of eligible actual requests, not uptime, active probing, or
  a guarantee that a model is available when no requests were made.
- It does not change routing, retries, request accounting, or log retention.
- The Agent API does not request this monitor breakdown history.
