import assert from 'node:assert/strict'
import test from 'node:test'

import {
  isCanonicalCredentialRouteQuery,
  normalizeGroupQuery,
  normalizeGroupTab,
  parseCredentialRouteQuery,
  parseCredentialRouteState,
  parseGroupModelsRouteQuery,
  parsePositiveId,
  serializeCredentialRouteQuery,
  serializeGroupModelsRouteQuery,
} from '../src/shared/routing/group-detail-route.ts'

// Codec parity suite for the shared group-detail route state — classic reads
// these same rules through its re-export adapter, so drift here diverges both
// frontends.

test('tab normalization falls back to credentials', () => {
  assert.equal(normalizeGroupTab('models'), 'models')
  assert.equal(normalizeGroupTab('settings'), 'settings')
  assert.equal(normalizeGroupTab('credentials'), 'credentials')
  assert.equal(normalizeGroupTab('bogus'), 'credentials')
  assert.equal(normalizeGroupTab(undefined), 'credentials')
})

test('credential query defaults: page 1, size 20, no q/status', () => {
  assert.deepEqual(parseCredentialRouteQuery({}), {
    page: 1,
    page_size: 20,
  })
})

test('credential query keeps valid page_size/status/q and drops the rest', () => {
  const filters = parseCredentialRouteQuery({
    page: '3',
    page_size: '50',
    credential_status: 'cooldown',
    q: '  sk-live  ',
  })
  assert.deepEqual(filters, {
    page: 3,
    page_size: 50,
    status: 'cooldown',
    q: 'sk-live',
  })
  // Invalid enum/page-size values normalize away.
  assert.deepEqual(parseCredentialRouteQuery({ credential_status: 'bogus', page_size: '37' }), {
    page: 1,
    page_size: 20,
  })
  assert.equal(parseCredentialRouteQuery({ credential_status: 'disabled' }).status, 'disabled')
})

test('expanded_credential_ids dedupe-sort; duplicates or junk drop entirely', () => {
  assert.deepEqual(
    parseCredentialRouteState({ expanded_credential_ids: '9,1,3' }).expandedCredentialIDs,
    [1, 3, 9],
  )
  assert.deepEqual(
    parseCredentialRouteState({ expanded_credential_ids: '9,1,1' }).expandedCredentialIDs,
    [],
  )
  assert.deepEqual(
    parseCredentialRouteState({ expanded_credential_ids: 'x' }).expandedCredentialIDs,
    [],
  )
})

test('credential serialization omits defaults and re-emits expanded ids', () => {
  assert.deepEqual(serializeCredentialRouteQuery({ page: 1, page_size: 20 }), {
    tab: 'credentials',
  })
  assert.deepEqual(
    serializeCredentialRouteQuery(
      { page: 2, page_size: 100, status: 'blacklisted', q: 'k' },
      { expandedCredentialIDs: [3, 1] },
    ),
    {
      tab: 'credentials',
      q: 'k',
      credential_status: 'blacklisted',
      page: '2',
      page_size: '100',
      expanded_credential_ids: '1,3',
    },
  )
})

test('canonical check matches the serialized form exactly', () => {
  const filters = { page: 1, page_size: 20 }
  assert.equal(isCanonicalCredentialRouteQuery({ tab: 'credentials' }, filters), true)
  assert.equal(isCanonicalCredentialRouteQuery({ tab: 'credentials', junk: 'x' }, filters), false)
  assert.equal(isCanonicalCredentialRouteQuery({}, filters), false)
})

test('models tab: discovery params only live while the panel is open', () => {
  assert.deepEqual(parseGroupModelsRouteQuery({ tab: 'models' }), {
    discoveryOpen: false,
    discoverySearch: undefined,
    discoveryFilter: 'unadded',
  })
  // discovery_* keys without panel=discovery are ignored on parse and absent
  // on serialize.
  assert.deepEqual(
    parseGroupModelsRouteQuery({
      tab: 'models',
      discovery_q: 'gpt',
      discovery_filter: 'all',
    }),
    { discoveryOpen: false, discoverySearch: undefined, discoveryFilter: 'unadded' },
  )
  assert.deepEqual(
    serializeGroupModelsRouteQuery({
      discoveryOpen: true,
      discoverySearch: 'gpt',
      discoveryFilter: 'all',
    }),
    { tab: 'models', panel: 'discovery', discovery_q: 'gpt', discovery_filter: 'all' },
  )
  assert.deepEqual(
    serializeGroupModelsRouteQuery({
      discoveryOpen: true,
      discoveryFilter: 'unadded',
    }),
    { tab: 'models', panel: 'discovery' },
  )
})

test('normalizeGroupQuery canonicalizes per tab and drops foreign params', () => {
  assert.deepEqual(normalizeGroupQuery({ tab: 'settings', junk: 'x' }), {
    tab: 'settings',
  })
  assert.deepEqual(normalizeGroupQuery({ tab: 'models', panel: 'discovery', q: 'leftover' }), {
    tab: 'models',
    panel: 'discovery',
  })
  // Unknown tab → credentials canonical query.
  assert.deepEqual(normalizeGroupQuery({ tab: 'bogus', page: '2' }), {
    tab: 'credentials',
    page: '2',
  })
})

test('parsePositiveId rejects non-positive and non-integer input', () => {
  assert.equal(parsePositiveId('12'), 12)
  assert.equal(parsePositiveId('0'), undefined)
  assert.equal(parsePositiveId('-3'), undefined)
  assert.equal(parsePositiveId('1.5'), undefined)
  assert.equal(parsePositiveId('abc'), undefined)
  assert.equal(parsePositiveId(undefined), undefined)
})
