import assert from 'node:assert/strict'
import test from 'node:test'
import {
  normalizeGroupQuery,
  parseGroupModelsRouteQuery,
  serializeGroupModelsRouteQuery,
} from '../src/shared/routing/group-detail-route.ts'

test('credential routes canonicalize to a singular tab and discard collection state', () => {
  assert.deepEqual(
    normalizeGroupQuery({
      tab: 'credentials',
      q: 'secret',
      page: '4',
      page_size: '100',
      credential_status: 'disabled',
      expanded_credential_ids: '1,2',
      junk: 'x',
    }),
    { tab: 'credentials' },
  )
})

test('models canonical route remains independent', () => {
  assert.deepEqual(
    parseGroupModelsRouteQuery({ tab: 'models', panel: 'discovery', discovery_q: 'gpt' }),
    {
      discoveryOpen: true,
      discoverySearch: 'gpt',
      discoveryFilter: 'unadded',
    },
  )
  assert.deepEqual(
    serializeGroupModelsRouteQuery({ discoveryOpen: true, discoveryFilter: 'all' }),
    {
      tab: 'models',
      panel: 'discovery',
      discovery_filter: 'all',
    },
  )
})
