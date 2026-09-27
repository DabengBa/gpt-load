import assert from 'node:assert/strict'
import test from 'node:test'

import {
  isCanonicalImportRouteQuery,
  parseImportRouteQuery,
  serializeImportRouteQuery,
} from '../src/shared/routing/import-route.ts'

// Codec parity suite for the shared import route state — classic reads these
// same rules through its re-export adapter, so drift here diverges both
// frontends.

test('bare /import parses as new mode with defaults', () => {
  assert.deepEqual(parseImportRouteQuery({}), {
    mode: 'new',
    panel: undefined,
    modelSearch: undefined,
    discoverySearch: undefined,
    discoveryFilter: 'unadded',
  })
})

test('group_id presence forces existing mode even when mode=new', () => {
  assert.equal(parseImportRouteQuery({ group_id: '7' }).mode, 'existing')
  assert.equal(
    parseImportRouteQuery({ mode: 'new', group_id: '7' }).mode,
    'existing',
  )
  assert.equal(parseImportRouteQuery({ group_id: '7' }).groupID, 7)
  assert.equal(parseImportRouteQuery({ mode: 'existing' }).mode, 'existing')
})

test('existing mode drops new-mode params on parse and serialize', () => {
  const state = parseImportRouteQuery({
    mode: 'existing',
    group_id: '4',
    model_q: 'gpt',
    panel: 'discovery',
    discovery_q: 'x',
    discovery_filter: 'all',
  })
  assert.deepEqual(state, { mode: 'existing', groupID: 4, discoveryFilter: 'unadded' })
  assert.deepEqual(serializeImportRouteQuery(state), { mode: 'existing', group_id: '4' })
  // Existing without a valid group id still serializes bare.
  assert.deepEqual(serializeImportRouteQuery({ mode: 'existing', discoveryFilter: 'unadded' }), {
    mode: 'existing',
  })
})

test('new mode round-trips model_q and the discovery panel state', () => {
  const state = parseImportRouteQuery({
    mode: 'new',
    model_q: '  gpt-5  ',
    panel: 'discovery',
    discovery_q: 'oai',
    discovery_filter: 'all',
  })
  assert.deepEqual(state, {
    mode: 'new',
    panel: 'discovery',
    modelSearch: 'gpt-5',
    discoverySearch: 'oai',
    discoveryFilter: 'all',
  })
  assert.deepEqual(serializeImportRouteQuery(state), {
    mode: 'new',
    model_q: 'gpt-5',
    panel: 'discovery',
    discovery_q: 'oai',
    discovery_filter: 'all',
  })
})

test('discovery params only survive while panel=discovery', () => {
  assert.deepEqual(
    parseImportRouteQuery({ discovery_q: 'x', discovery_filter: 'all' }),
    {
      mode: 'new',
      panel: undefined,
      modelSearch: undefined,
      discoverySearch: undefined,
      discoveryFilter: 'unadded',
    },
  )
  // A non-discovery panel value is dropped entirely.
  assert.equal(parseImportRouteQuery({ panel: 'bogus' }).panel, undefined)
})

test('canonical check compares the sparse serialized form', () => {
  const state = parseImportRouteQuery({})
  assert.equal(isCanonicalImportRouteQuery({ mode: 'new' }, state), true)
  assert.equal(isCanonicalImportRouteQuery({ mode: 'new', junk: '1' }, state), false)
  assert.equal(isCanonicalImportRouteQuery({}, state), false)
})
