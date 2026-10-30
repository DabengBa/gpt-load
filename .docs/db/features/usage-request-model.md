---
id: feature.usage-request-model
type: feature
name: Usage reporting by request model
---
# Usage Reporting By Request Model

## ID Explanation

`feature.usage-request-model` owns the request-model dimension and filter in
the usage report. Aggregation and historical-data rules belong to
[Usage Accounting](../../tech/usage-accounting.md).

## Purpose

Operators can distinguish usage requested under different client model names,
even when those requests are served by the same upstream model.

## User-Visible Contract

- Usage breakdown rows show the client-requested model separately from the
  upstream model. A model unavailable in historical data is shown as unknown.
- The request-model filter matches the complete client model name exactly. It
  can be combined with the upstream-model filter and other report filters.
- Applying filters updates the report and preserves the request model in the
  URL; browser navigation and refresh retain that selection. Reset clears the
  filter. Invalid input cannot be applied.
- Access-key users see only usage within their own access-key scope, including
  when filtering by request model.
- Timing and efficiency measures retain their existing meanings when filtered;
  see [Usage Timing And Efficiency Metrics](feature.usage-timing-metrics).

## Acceptance Workflows

### Compare usage for a client model

- **Role and purpose:** An operator wants to isolate usage for a client model
  alias without confusing it with the provider's upstream model.
- **Entry and action:** Open the usage report, enter the request model, and
  apply it, optionally selecting an upstream model as well.
- **Expected result:** The report shows matching usage and identifies request
  and upstream models separately. Refreshing retains the filter.
- **Failure signal:** A partial model name matches another name, historical
  unknown values are presented as inferred models, or an access-key user sees
  another key's usage.

## Boundaries

- The filter describes usage reporting, not model routing or model aliases.
- Existing records without a client model remain unknown; they are not
  reconstructed from the upstream model.
- The Agent API retains its existing upstream-model grouping; it does not gain
  this monitor filter or dimension.
