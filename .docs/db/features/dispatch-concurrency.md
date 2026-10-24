---
id: feature.dispatch-concurrency
type: feature
name: Dispatch row concurrency limit
route: /schedule
related: [feature.dispatch-reasoning-policy]
---
# Dispatch Row Concurrency Limit

## ID Explanation

`feature.dispatch-concurrency` describes how an administrator sets a
concurrency cap on a single dispatch row, and how ordinary and affinity
requests are scheduled differently once that row is full. It does not define a
group-wide cap, a request rate, or queueing.

## Purpose

Control how ordinary new requests keep being assigned to one dispatch row,
while an existing valid affinity session keeps its original target.

## User-Visible Contract

- An administrator selects a model in Dispatch Center, opens that row's
  **Advanced** section, sets **Concurrency limit**, and saves.
- 0 means unlimited; a positive integer is the maximum number of ordinary
  in-flight requests. The input must be an integer from 0 to 2147483647, and a
  cleared input must be filled in before the row can be saved.
- The cap belongs to a specific dispatch row, not to a group. Different
  request entrypoints that use the same row share its capacity; other rows
  count separately.
- Once a row is full, an ordinary request is scheduled directly to another
  candidate that still matches the existing rules. When every candidate is
  full, the request fails immediately with a capacity error instead of waiting
  in a queue.
- A valid affinity request may still enter its bound target and count toward
  occupancy, so real concurrency can exceed the configured value. A request is
  a valid affinity only when the target actually resolved for it matches the
  binding exactly (same group, credential, identity generation, dispatch entry,
  and upstream model); the bound target must still satisfy the existing
  permission, health, and availability rules, and an affinity signal without a
  valid binding cannot exceed the limit.
- Being full does not by itself rebind an affinity or mark that row as faulty. A
  WebSocket request with a fixed target identity is not switched to another
  identity because of saturation; without a valid affinity exemption it returns
  the capacity error.
- An ordinary request holds its slot until execution ends, and a streaming
  request holds it until the stream ends. WebSocket counts active requests; an
  idle connection holds no capacity.
- Lowering the limit never interrupts in-flight requests, and the stored value
  is still shown after reopening or refreshing. A failed save or a
  configuration conflict keeps the draft, which is saved after the conflict is
  resolved.

## Acceptance Workflows

### Save and maintain one dispatch row's limit

- **Role and purpose:** An administrator wants one dispatch row to stop
  accepting ordinary new requests once two ordinary requests are in flight.
- **Entry and action:** Open Dispatch Center, select the model, enter 2 in the
  target row's advanced configuration, save, refresh, and reopen the advanced
  configuration.
- **Expected result:** That row still shows 2 and other rows are unaffected; the
  note that a valid affinity request may exceed the limit remains visible.
- **Maintenance and failure signal:** A negative number, a decimal, or an empty
  value cannot be saved; a save conflict does not lose the input. Setting 0 and
  saving removes the ordinary-request capacity limit.

## Boundaries

Concurrency means the number of simultaneously executing requests and is not
requests per minute. This feature provides per-dispatch-row control inside a
single service process. It does not provide a limit shared across instances and
does not promise strict upstream concurrency protection while affinity requests
exist.
