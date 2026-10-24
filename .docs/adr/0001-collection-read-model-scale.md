---
description: Measured AccessKey collection read-path scale and the decision to keep in-memory filtering instead of pushing it into SQL
kind: adr
topic: collection-read-model
status: accepted
date: 2026-09-25
---

# ADR-0001: Keep the collection read model as full-snapshot capture plus in-memory filtering; do not push filtering into SQL yet

## Context

Every management collection read path repeats the same trio: `*_http` (query
parsing) → `*_collection` (`withReadSnapshot` full capture) → `*_query`
(in-memory filtering, sorting, pagination). Architecture review candidate #4
proposed collapsing that trio into a shared read model; the gate was to first
answer, with measurements, whether full capture plus in-memory filtering is
acceptable at the current scale.

## Measurement

Benchmark `internal/storage/models/access_key_collection_scale_test.go`
(`BenchmarkAccessKeyCollectionScale`) reproduces the complete query shape of
`captureAccessKeyCollectionRecords`: a full `Select` over `access_keys`, a
per-row correlated subquery on `request_logs` for `MAX(completed_at_ms)`, a
`Find` over the cost-rule table, and in-memory grouping and sorting.
Environment: Windows amd64, Ryzen 5 2600X, file-backed SQLite with WAL, a
4-connection pool.

| Scale | Latency/op | Note |
|---|---|---|
| 1k keys, 0 request_logs | ~6.0 ms | typical scale |
| 1k keys, 500k request_logs | ~19.9 ms | the correlated subquery probes through `idx_request_logs_access_completed_id` |
| 10k keys, 500k request_logs | ~153.6 ms | 10× stress, linear growth |

## Decision

1. **At typical scale (≤1k AccessKeys) full capture costs ~6–20 ms**, which is
   entirely acceptable for a management list; in-memory filtering, sorting, and
   pagination are microsecond-level work at that size and are not a cost center.
2. **The only term that grows with scale is the per-row correlated `MAX`
   probe** (~14 µs/row at 500k logs). `request_logs.access_key_id` already has
   the covering index
   `idx_request_logs_access_completed_id(access_key_id, completed_at_ms DESC, id DESC)`,
   so the probe is an O(log) index lookup rather than a full scan, and growth
   stays linear at 10× scale with no cliff.
3. **Decision:** keep the declarative read model with in-memory filtering. P3
   may pilot collapsing the trio, but must not make pushing filtering or
   pagination into SQL a prerequisite.
4. The `withReadSnapshot` consistency semantics must be preserved — whether the
   work happens in memory or in a future SQL pushdown, the capture is taken
   inside one read transaction.

## Re-measurement triggers (any one reopens this decision)

- Any AccessKey, Group, or Credential collection exceeds **10k rows** in
  measurement;
- measured collection P95 latency exceeds **200 ms**;
- `request_logs` grows enough that the per-row probe dominates latency (~14 µs
  per row today, ~150 ms at ten-thousand keys).

The preferred fix at that point is not a per-row subquery but a single
`LEFT JOIN + GROUP BY` aggregating `MAX(completed_at_ms)`, or demoting
`last_request_at_ms` to a write-side denormalized column — neither changes the
read-snapshot semantics.

## References

- Review report: `docs/analysis/architecture-review-briefing.md` (candidate #4)
- Implementation plan: `docs/design/architecture-deepening-plan.md` (P3 section)
- Read-path code: `internal/control/access_key_collection.go`, `withReadSnapshot` in `internal/control/service.go`
