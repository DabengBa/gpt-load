import assert from 'node:assert/strict'
import test from 'node:test'

import {
  normalizeAccessKeyMonitorQuery,
  normalizeMonitorQuery,
  parseScheduleMonitorState,
  scopeAccessKeyScheduleMonitorState,
  sameMonitorQuery,
  scheduleMonitorQuery,
  scopeAccessKeyUsageFilters,
  usageMonitorQuery,
} from '../src/shared/routing/monitor-route.ts'

// Codec parity suite for the shared monitor route state — classic reads these
// same rules through its re-export adapter, so drift here diverges both
// frontends. /schedule shares the schedule_* surface of the same query space.

test('monitor query canonicalizes to the usage filter surface', () => {
  // Retired tab/metric params and unknown keys are dropped in the canonical query.
  assert.deepEqual(normalizeMonitorQuery({ tab: 'health', junk: 'x' }), { range: '24h' })
  assert.deepEqual(normalizeMonitorQuery({ tab: 'usage', range: '7d', metric: 'cost' }), {
    range: '7d',
  })
})

test('access-key normalization scopes filters to the principal', () => {
  assert.deepEqual(normalizeAccessKeyMonitorQuery({ tab: 'logs' }), { range: '24h' })
  assert.deepEqual(normalizeAccessKeyMonitorQuery({ group_id: '3', upstream_model: 'gpt-4o' }), {
    range: '24h',
    upstream_model: 'gpt-4o',
  })
})

test('access-key scoping drops the group filter and rescales sort', () => {
  const scoped = scopeAccessKeyUsageFilters({
    range: '7d',
    group_id: 3,
    upstream_model: 'gpt-4o',
    breakdown_sort: 'group',
    breakdown_sort_direction: 'desc',
  })
  assert.equal(scoped.group_id, undefined)
  assert.equal(scoped.upstream_model, 'gpt-4o')
  assert.equal(scoped.breakdown_sort, 'model')
  assert.equal(scoped.breakdown_sort_direction, 'asc')
})

test('schedule draft round-trips with sorted rows and explicit null clears', () => {
  const state = {
    externalModel: 'gpt-5',
    selectedRow: '2:entry-b',
    sourceGroupId: 7,
    drafts: {
      '2:entry-b': { priority: null },
      '1:entry-a': { weight: 50, priority: 3 },
    },
  }
  const query = scheduleMonitorQuery(state)
  assert.equal(query.schedule_model, 'gpt-5')
  assert.equal(query.schedule_row, '2:entry-b')
  assert.equal(query.schedule_group, '7')
  // Sorted row keys make the serialized form stable for URL comparisons.
  assert.equal(
    query.schedule_draft,
    JSON.stringify({ '1:entry-a': { weight: 50, priority: 3 }, '2:entry-b': { priority: null } }),
  )
  assert.deepEqual(parseScheduleMonitorState(query), {
    externalModel: 'gpt-5',
    selectedRow: '2:entry-b',
    sourceGroupId: 7,
    selectedPriceID: undefined,
    drafts: {
      '1:entry-a': { weight: 50, priority: 3 },
      '2:entry-b': { priority: null },
    },
  })
})

test('schedule price selection round-trips as a positive integer id', () => {
  const query = scheduleMonitorQuery({ externalModel: 'gpt-5', selectedPriceID: 42, drafts: {} })
  assert.equal(query.selected_price_id, '42')
  assert.deepEqual(parseScheduleMonitorState(query), {
    externalModel: 'gpt-5',
    selectedRow: undefined,
    sourceGroupId: undefined,
    selectedPriceID: 42,
    drafts: {},
  })
  // A price deep link must survive parsing without any schedule_model — the
  // drawer can open standalone and be closed by the browser back button.
  assert.equal(parseScheduleMonitorState({ selected_price_id: '7' }).selectedPriceID, 7)
  assert.equal(scheduleMonitorQuery({ selectedPriceID: 7, drafts: {} }).selected_price_id, '7')
})

test('schedule price selection drops invalid ids', () => {
  for (const raw of ['0', '-1', 'abc', '1.5', ' 7', '', ['3'], 7]) {
    assert.equal(parseScheduleMonitorState({ selected_price_id: raw }).selectedPriceID, undefined)
  }
  assert.equal(scheduleMonitorQuery({ drafts: {} }).selected_price_id, undefined)
})

test('access-key scoping strips price and schedule context but keeps the model', () => {
  const scoped = scopeAccessKeyScheduleMonitorState({
    externalModel: 'gpt-5',
    selectedRow: '2:entry-b',
    sourceGroupId: 7,
    selectedPriceID: 42,
    drafts: { '2:entry-b': { weight: 50 } },
  })
  assert.deepEqual(scoped, { externalModel: 'gpt-5', drafts: {} })
  // Canonical serialization must drop the malicious deep-link params too.
  assert.deepEqual(scheduleMonitorQuery(scoped), { schedule_model: 'gpt-5' })
})

test('schedule draft drops malformed rows and out-of-range fields', () => {
  const drafts = parseScheduleMonitorState({
    schedule_draft: JSON.stringify({
      'no-prefix': { weight: 10 },
      '1:ok': { weight: 101, priority: 0 },
      '1:also-ok': { weight: 0 },
      '2:bad': 'not-an-object',
      '3:float': { weight: 1.5 },
    }),
  }).drafts
  assert.deepEqual(drafts, { '1:also-ok': { weight: 0 } })
})

test('schedule draft ignores non-object and oversized payloads', () => {
  assert.deepEqual(parseScheduleMonitorState({ schedule_draft: '[1,2]' }).drafts, {})
  assert.deepEqual(parseScheduleMonitorState({ schedule_draft: 'x'.repeat(20_001) }).drafts, {})
  // An empty draft map serializes to nothing so the param disappears.
  assert.deepEqual(scheduleMonitorQuery({ drafts: {} }), {})
})

test('schedule_row requires the group:entry shape', () => {
  assert.equal(parseScheduleMonitorState({ schedule_row: '12:model-x' }).selectedRow, '12:model-x')
  assert.equal(parseScheduleMonitorState({ schedule_row: 'bare' }).selectedRow, undefined)
  assert.equal(parseScheduleMonitorState({ schedule_row: '12:' }).selectedRow, undefined)
  assert.equal(parseScheduleMonitorState({ schedule_row: 'x:y' }).selectedRow, undefined)
  assert.equal(parseScheduleMonitorState({ schedule_group: '0' }).sourceGroupId, undefined)
  assert.equal(parseScheduleMonitorState({ schedule_group: '-3' }).sourceGroupId, undefined)
})

test('usage canonical query omits defaults', () => {
  const query = usageMonitorQuery({ range: '24h' })
  assert.equal(query.range, '24h')
  assert.equal(query.panel, undefined)
  assert.equal(query.breakdown_sort, undefined)
  assert.equal(query.breakdown_page, undefined)
  assert.equal(query.breakdown_page_size, undefined)
  // Retired filter params never serialize into the canonical query.
  assert.deepEqual(usageMonitorQuery({ range: '7d', group_id: 2 }), {
    range: '7d',
    group_id: '2',
  })
})

test('sameMonitorQuery requires identical string key sets', () => {
  assert.equal(sameMonitorQuery({ range: '24h' }, { range: '24h' }), true)
  assert.equal(sameMonitorQuery({ range: '24h' }, { range: '7d' }), false)
  assert.equal(sameMonitorQuery({ range: '24h', group_id: '1' }, { range: '24h' }), false)
  // Non-string leftovers (e.g. vue-router array values) never match.
  assert.equal(sameMonitorQuery({ range: '24h', extra: ['a', 'b'] }, { range: '24h' }), false)
})
