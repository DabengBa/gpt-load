import assert from 'node:assert/strict'
import test from 'node:test'

import {
  isCanonicalSettingsRouteQuery,
  parseSettingsCredentialsRoute,
  parseSettingsRouteSection,
  serializeSettingsRouteQuery,
} from '../src/shared/routing/settings-route.ts'

// Codec parity suite for the settings route contract. The credentials section
// embeds the access-key collection + drawer params that used to live on the
// standalone /access-keys page.

test('section parsing defaults to routing and validates names', () => {
  assert.equal(parseSettingsRouteSection({}), 'routing')
  assert.equal(parseSettingsRouteSection({ section: 'credentials' }), 'credentials')
  assert.equal(parseSettingsRouteSection({ section: 'system' }), 'system')
  assert.equal(parseSettingsRouteSection({ section: 'bogus' }), 'routing')
  assert.equal(parseSettingsRouteSection({ section: ['system', 'routing'] }), 'routing')
})

test('routing serializes to a bare URL, other sections carry section=', () => {
  assert.deepEqual(serializeSettingsRouteQuery('routing'), {})
  assert.deepEqual(serializeSettingsRouteQuery('system'), { section: 'system' })
  assert.deepEqual(serializeSettingsRouteQuery('credentials'), { section: 'credentials' })
})

test('credentials state round-trips collection filters and the drawer', () => {
  const parsed = parseSettingsCredentialsRoute({
    q: 'prod',
    status: 'disabled',
    page: '3',
    action: 'edit',
    access_key_id: '7',
  })
  assert.deepEqual(parsed.collection, { q: 'prod', status: 'disabled', page: 3, page_size: 20 })
  assert.deepEqual(parsed.drawer, { mode: 'edit', accessKeyID: 7 })

  assert.deepEqual(
    serializeSettingsRouteQuery('credentials', parsed),
    {
      section: 'credentials',
      q: 'prod',
      status: 'disabled',
      page: '3',
      action: 'edit',
      access_key_id: '7',
    },
  )
})

test('create-mode drawer and defaults serialize sparsely', () => {
  const state = parseSettingsCredentialsRoute({ action: 'create' })
  assert.deepEqual(state.drawer, { mode: 'create' })
  assert.deepEqual(serializeSettingsRouteQuery('credentials', state), {
    section: 'credentials',
    action: 'create',
  })
})

test('credentials params are dropped for other sections', () => {
  assert.deepEqual(serializeSettingsRouteQuery('system', parseSettingsCredentialsRoute({})), {
    section: 'system',
  })
  assert.deepEqual(
    serializeSettingsRouteQuery('routing', {
      collection: { page: 2, page_size: 20 },
      drawer: { mode: 'edit', accessKeyID: 4 },
    }),
    {},
  )
})

test('canonicalization keeps credentials params only in that section', () => {
  const credentials = parseSettingsCredentialsRoute({ action: 'create' })
  assert.equal(
    isCanonicalSettingsRouteQuery({ section: 'credentials', action: 'create' }, 'credentials', credentials),
    true,
  )
  // Stray params on another section are non-canonical.
  assert.equal(
    isCanonicalSettingsRouteQuery({ section: 'system', action: 'create' }, 'system'),
    false,
  )
  // Missing credentials state on the credentials section is non-canonical too.
  assert.equal(
    isCanonicalSettingsRouteQuery({ section: 'credentials', action: 'create' }, 'credentials'),
    false,
  )
})
