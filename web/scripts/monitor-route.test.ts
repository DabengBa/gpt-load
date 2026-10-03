import assert from 'node:assert/strict'
import test from 'node:test'

import {
  inspectorMonitorQuery,
  normalizeAccessKeyMonitorQuery,
  normalizeMonitorQuery,
  normalizeMonitorTab,
  parseInspectorMonitorState,
  parseScheduleMonitorState,
  parseUsageMonitorState,
  scopeAccessKeyScheduleMonitorState,
  sameMonitorQuery,
  scheduleMonitorQuery,
  scopeAccessKeyUsageFilters,
  usageMonitorQuery,
  healthMonitorQuery,
} from '../src/shared/routing/monitor-route.ts'

// Codec parity suite for the shared monitor route state — classic reads these
// same rules through its re-export adapter, so drift here diverges both
// frontends. /schedule shares the schedule_* surface of the same query space.

test('tab normalization falls back to health and canonicalizes per-tab queries', () => {
  assert.equal(normalizeMonitorTab('usage'), 'usage')
  assert.equal(normalizeMonitorTab('garbage'), 'health')
  assert.equal(normalizeMonitorTab(undefined), 'health')
  // Unknown params on the health tab are dropped in the canonical query.
  assert.deepEqual(normalizeMonitorQuery({ tab: 'health', junk: 'x' }), { tab: 'health' })
})

test('access-key normalization rewrites tab=logs to the usage query', () => {
  // Classic quirk: /monitor?tab=logs lands on usage for access-key principals.
  assert.deepEqual(normalizeAccessKeyMonitorQuery({ tab: 'logs' }), {
    tab: 'usage',
    range: '24h',
    metric: 'tokens',
  })
})

test('access-key scoping drops group/channel/credential and rescales sort', () => {
  const scoped = scopeAccessKeyUsageFilters({
    range: '7d',
    group_id: 3,
    channel_id: 'ch-1',
    credential_id: 9,
    breakdown_sort: 'group',
    breakdown_sort_direction: 'desc',
  })
  assert.equal(scoped.group_id, undefined)
  assert.equal(scoped.channel_id, undefined)
  assert.equal(scoped.credential_id, undefined)
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

test('inspector run=1 serializes only with the full protocol/model/key triple', () => {
  const base = { protocol: 'openai-completions', externalModel: 'm', accessKeyID: '4' }
  assert.equal(inspectorMonitorQuery({ ...base, run: true }).run, '1')
  for (const missing of [
    { ...base, protocol: undefined },
    { ...base, externalModel: undefined },
    { ...base, accessKeyID: undefined },
  ]) {
    assert.equal(inspectorMonitorQuery({ ...missing, run: true }).run, undefined)
  }
  // Parse side: run=1 without the triple also resolves to false.
  assert.equal(parseInspectorMonitorState({ run: '1' }).run, false)
  assert.equal(parseInspectorMonitorState({ run: '1', protocol: 'bogus' }).protocol, undefined)
})

test('inspector expanded_groups dedupes, sorts, and rejects duplicates', () => {
  assert.deepEqual(parseInspectorMonitorState({ expanded_groups: '9,3,3,1' }).expandedGroupIDs, [])
  assert.deepEqual(
    parseInspectorMonitorState({ expanded_groups: '9,3,1' }).expandedGroupIDs,
    [1, 3, 9],
  )
  const query = inspectorMonitorQuery({
    run: false,
    expandedGroupIDs: [9, 1, 9, 3],
  })
  assert.equal(query.expanded_groups, '1,3,9')
})

test('usage canonical query omits defaults and keeps panel/series state', () => {
  const query = usageMonitorQuery(
    { range: '24h' },
    { filtersOpen: true, seriesExpanded: true, metric: 'cost' },
  )
  assert.equal(query.tab, 'usage')
  assert.equal(query.range, '24h')
  assert.equal(query.metric, 'cost')
  assert.equal(query.panel, 'filters')
  assert.equal(query.series, 'expanded')
  assert.equal(query.breakdown_sort, undefined)
  assert.equal(query.breakdown_page, undefined)
  assert.equal(query.breakdown_page_size, undefined)
  assert.deepEqual(parseUsageMonitorState(query), {
    filtersOpen: true,
    seriesExpanded: true,
    metric: 'cost',
  })
})

test('health canonical query only carries the groups-expanded flag', () => {
  assert.deepEqual(healthMonitorQuery({ groupsExpanded: false }), { tab: 'health' })
  assert.deepEqual(healthMonitorQuery({ groupsExpanded: true }), {
    tab: 'health',
    groups: 'expanded',
  })
})

test('sameMonitorQuery requires identical string key sets', () => {
  assert.equal(sameMonitorQuery({ tab: 'health' }, { tab: 'health' }), true)
  assert.equal(sameMonitorQuery({ tab: 'health' }, { tab: 'usage' }), false)
  assert.equal(sameMonitorQuery({ tab: 'health', groups: 'expanded' }, { tab: 'health' }), false)
  // Non-string leftovers (e.g. vue-router array values) never match.
  assert.equal(sameMonitorQuery({ tab: 'health', extra: ['a', 'b'] }, { tab: 'health' }), false)
})
