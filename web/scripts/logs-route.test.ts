import assert from 'node:assert/strict'
import test from 'node:test'

import {
  createLogFilterDraft,
  applyLogFilterDraft,
  defaultRequestLogFilters,
  parseAppliedLogFilterState,
  serializeAppliedLogFilters,
  validateLogFilterDraft,
} from '../src/shared/domain/monitor/log-filters.ts'
import {
  logsMonitorQuery,
  parseLogsMonitorState,
  parseSelectedRequestID,
  scopeAccessKeyLogFilters,
} from '../src/shared/routing/logs-route.ts'

// Codec parity suite for the shared logs route state — classic reads these
// same rules through its re-export adapters, so any drift here diverges both
// frontends.

const VALID_ID = 'aaaaaaaa-1111-4111-8111-111111111111'
const VALID_AFFINITY = '0123456789abcdef****fedcba9876543210'

test('default parse yields the 48h default window and limit 20', () => {
  const { filters, invalidAffinityKey } = parseAppliedLogFilterState({})
  assert.equal(filters.limit, 20)
  assert.ok(filters.from_ms !== undefined && filters.to_ms !== undefined)
  const span = filters.to_ms! - filters.from_ms!
  assert.ok(span > 47 * 60 * 60 * 1000 && span < 49 * 60 * 60 * 1000)
  assert.equal(invalidAffinityKey, undefined)
})

test('from >= to falls back to the default window', () => {
  const { filters } = parseAppliedLogFilterState({ from_ms: '200', to_ms: '100' })
  assert.ok(filters.from_ms! < filters.to_ms!)
})

test('unsupported limit values normalize to 20', () => {
  assert.equal(parseAppliedLogFilterState({ limit: '50' }).filters.limit, 50)
  assert.equal(parseAppliedLogFilterState({ limit: '37' }).filters.limit, 20)
  assert.equal(parseAppliedLogFilterState({ limit: '-1' }).filters.limit, 20)
})

test('enum and typed params parse; garbage is dropped', () => {
  const { filters } = parseAppliedLogFilterState({
    status: 'incomplete',
    retry_state: 'retried',
    stream: 'true',
    group_id: '12',
    final_status_code: '500',
    attempt_status_code: '429',
    model_consistency: 'mismatch',
    cost_min_nano_usd: '1500000000',
  })
  assert.equal(filters.status, 'incomplete')
  assert.equal(filters.retry_state, 'retried')
  assert.equal(filters.stream, true)
  assert.equal(filters.group_id, 12)
  assert.equal(filters.final_status_code, 500)
  assert.equal(filters.attempt_status_code, 429)
  assert.equal(filters.model_consistency, 'mismatch')
  assert.equal(filters.cost_min_nano_usd, '1500000000')

  const garbage = parseAppliedLogFilterState({
    status: 'bogus',
    group_id: '0',
    final_status_code: '1000',
    stream: 'yes',
    client_model: 'x'.repeat(300),
  }).filters
  assert.equal(garbage.status, undefined)
  assert.equal(garbage.group_id, undefined)
  assert.equal(garbage.final_status_code, undefined)
  assert.equal(garbage.stream, undefined)
  assert.equal(garbage.client_model, undefined)
})

test('invalid affinity key stays out of filters but surfaces as invalidAffinityKey', () => {
  const parsed = parseAppliedLogFilterState({ affinity_key: 'not-a-key' })
  assert.equal(parsed.filters.affinity_key, undefined)
  assert.equal(parsed.invalidAffinityKey, 'not-a-key')

  const valid = parseAppliedLogFilterState({ affinity_key: VALID_AFFINITY })
  assert.equal(valid.filters.affinity_key, VALID_AFFINITY)
  assert.equal(valid.invalidAffinityKey, undefined)
})

test('serializeAppliedLogFilters writes every set field as string', () => {
  const query = serializeAppliedLogFilters({
    ...defaultRequestLogFilters(),
    group_id: 3,
    stream: false,
    client_model: 'worker',
  })
  assert.equal(query.group_id, '3')
  assert.equal(query.stream, 'false')
  assert.equal(query.client_model, 'worker')
  assert.equal(query.limit, '20')
})

test('draft round-trip keeps cost USD<->nanoUSD and returns equal filters', () => {
  const filters = {
    ...defaultRequestLogFilters(),
    cost_min_nano_usd: '1500000000',
    retry_count_min: 1,
    affinity_key: VALID_AFFINITY,
  }
  const draft = createLogFilterDraft(filters)
  assert.equal(draft.cost_min_usd, '1.5')
  const applied = applyLogFilterDraft(draft)
  assert.equal(applied.cost_min_nano_usd, '1500000000')
  assert.equal(applied.retry_count_min, 1)
  assert.equal(applied.affinity_key, VALID_AFFINITY)
})

test('validateLogFilterDraft flags inverted ranges and malformed ids', () => {
  const draft = createLogFilterDraft()
  draft.from = '2030-01-02T00:00:00'
  draft.to = '2030-01-01T00:00:00'
  draft.request_id = 'not-a-uuid'
  draft.group_id = '0'
  draft.retry_count_min = '5'
  draft.retry_count_max = '2'
  draft.affinity_key = 'bogus'
  const errors = validateLogFilterDraft(draft)
  assert.equal(errors.to, 'monitor.logs.errors.range')
  assert.equal(errors.request_id, 'monitor.logs.errors.requestId')
  assert.equal(errors.group_id, 'monitor.logs.errors.positiveId')
  assert.equal(errors.retry_count_max, 'monitor.logs.errors.numericRange')
  assert.equal(errors.affinity_key, 'monitor.logs.errors.affinityKey')
})

test('selected_request_id accepts only UUIDv4', () => {
  assert.equal(parseSelectedRequestID({ selected_request_id: VALID_ID }), VALID_ID)
  // v1 shape and non-uuid are dropped.
  assert.equal(
    parseSelectedRequestID({ selected_request_id: 'aaaaaaaa-1111-1111-8111-111111111111' }),
    undefined,
  )
  assert.equal(parseSelectedRequestID({ selected_request_id: 'zzz' }), undefined)
  assert.equal(parseSelectedRequestID({ selected_request_id: [VALID_ID, VALID_ID] }), undefined)
})

test('cursor history parses bounded unique valid cursors', () => {
  const state = parseLogsMonitorState({ log_cursors: '["a","b-2","c_3"]' })
  assert.deepEqual(state.cursorHistory, ['a', 'b-2', 'c_3'])

  // duplicates, invalid chars, non-arrays, over-cap and oversized raw all reset
  assert.deepEqual(parseLogsMonitorState({ log_cursors: '["a","a"]' }).cursorHistory, [])
  assert.deepEqual(parseLogsMonitorState({ log_cursors: '["a b"]' }).cursorHistory, [])
  assert.deepEqual(parseLogsMonitorState({ log_cursors: '"a"' }).cursorHistory, [])
  const overCap = JSON.stringify(Array.from({ length: 51 }, (_, i) => `c${i}`))
  assert.deepEqual(parseLogsMonitorState({ log_cursors: overCap }).cursorHistory, [])
  const huge = JSON.stringify(Array.from({ length: 4 }, () => 'c'.repeat(400)))
  assert.deepEqual(parseLogsMonitorState({ log_cursors: huge }).cursorHistory, [])
})

test('logsMonitorQuery serializes detail, panel and cursors; clears when absent', () => {
  const query = logsMonitorQuery(
    { ...defaultRequestLogFilters(), group_id: 2 },
    {
      filtersOpen: true,
      cursorHistory: ['p1', 'p2'],
      selectedRequestID: VALID_ID,
      invalidAffinityKey: 'bad-key',
    },
  )
  assert.equal(query.panel, 'filters')
  assert.equal(query.log_cursors, '["p1","p2"]')
  assert.equal(query.selected_request_id, VALID_ID)
  // invalid affinity keys round-trip through the URL so reloads keep the blocked state
  assert.equal(query.affinity_key, 'bad-key')
  assert.equal(query.group_id, '2')

  const cleared = logsMonitorQuery(defaultRequestLogFilters())
  assert.equal(cleared.panel, undefined)
  assert.equal(cleared.log_cursors, undefined)
  assert.equal(cleared.selected_request_id, undefined)
})

test('serialized cursor history keeps at most the last 50 valid cursors', () => {
  const query = logsMonitorQuery(defaultRequestLogFilters(), {
    filtersOpen: false,
    cursorHistory: [...Array.from({ length: 55 }, (_, i) => `c${i}`), 'bad cursor'],
  })
  const parsed = JSON.parse(String(query.log_cursors)) as string[]
  assert.equal(parsed.length, 50)
  // slice(-50) keeps the newest — c55..c54 tail after filtering the bad cursor
  assert.equal(parsed.at(-1), 'c54')
  assert.equal(parsed.includes('bad cursor'), false)
})

test('scopeAccessKeyLogFilters strips every admin-only dimension', () => {
  const scoped = scopeAccessKeyLogFilters({
    ...defaultRequestLogFilters(),
    group_id: 1,
    channel_id: 'openai_compatible',
    credential_id: 2,
    upstream_model: 'x',
    access_key_id: 3,
    attempt_status_code: 500,
    failure_category: 'ok',
    error_code: 'e',
    retry_state: 'retried',
    retry_count_min: 1,
    retry_count_max: 2,
    affinity_key: VALID_AFFINITY,
    client_model: 'worker',
    status: 'success',
  })
  assert.equal(scoped.group_id, undefined)
  assert.equal(scoped.channel_id, undefined)
  assert.equal(scoped.credential_id, undefined)
  assert.equal(scoped.upstream_model, undefined)
  assert.equal(scoped.access_key_id, undefined)
  assert.equal(scoped.attempt_status_code, undefined)
  assert.equal(scoped.failure_category, undefined)
  assert.equal(scoped.error_code, undefined)
  assert.equal(scoped.retry_state, undefined)
  assert.equal(scoped.retry_count_min, undefined)
  assert.equal(scoped.retry_count_max, undefined)
  assert.equal(scoped.affinity_key, undefined)
  // allowed dimensions stay
  assert.equal(scoped.client_model, 'worker')
  assert.equal(scoped.status, 'success')
  assert.equal(scoped.limit, 20)
})
