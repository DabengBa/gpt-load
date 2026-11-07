import assert from 'node:assert/strict'
import test from 'node:test'
import { createApiClient } from '../src/shared/http/client.ts'
import {
  createGroup,
  importGroupCredentials,
  listGroupCollection,
  getGroupSettings,
  updateGroupSettings,
  projectGroupSummary,
} from '../src/shared/control/resources/groups.ts'

function clientReturning(data: unknown) {
  const requests: Array<{ path: string; body: unknown; method: string | undefined }> = []
  const client = createApiClient({
    fetch: async (input, init) => {
      requests.push({
        path: String(input),
        body: init?.body ? JSON.parse(String(init.body)) : null,
        method: init?.method,
      })
      return new Response(JSON.stringify({ code: 0, message: 'ok', data }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    },
    getAuthKey: () => 'test-key',
    getLocale: () => 'en-US',
    onUnauthorized: () => {},
  })
  return { client, requests }
}

test('create and configure send one credential and consume one identity', async () => {
  const { client, requests } = clientReturning({
    group_id: 1,
    group_name: 'Provider',
    credential_id: 9,
  })
  const base = {
    channel_id: 'openai',
    params: {},
    models: [],
    confirm_same_target: false,
  }
  assert.equal(
    (await createGroup(client, { ...base, credential: 'provider-key' }, 'create')).credential_id,
    9,
  )
  assert.deepEqual(requests[0]?.body, { ...base, credential: 'provider-key' })
  const configured = clientReturning({ group_id: 1, credential_id: 9 })
  await importGroupCredentials(configured.client, 1, { credential: 'key' }, 'configure')
  assert.deepEqual(configured.requests[0], {
    path: '/api/groups/1/credential',
    body: { credential: 'key' },
    method: 'POST',
  })
})

const summary = {
  id: 1,
  name: 'Provider',

  channel_id: 'openai',
  params: {},
  provider_url: null,
  service_status: 'unavailable',
  service_status_reason: 'no_available_credentials',
  credential_configured: false,
  credential_status: null,
  model_count: 1,
}

test('summary rejects an empty group carrying a disabled credential status', () => {
  assert.throws(() => projectGroupSummary({ ...summary, credential_status: 'disabled' }))
  assert.equal(projectGroupSummary(summary).credential_status, null)
})

test('actual summary fixture rejects retired pricing and collection fields', () => {
  assert.equal(projectGroupSummary(summary).name, 'Provider')
  assert.throws(() => projectGroupSummary({ ...summary, price_multiplier: '1' }))
  assert.throws(() => projectGroupSummary({ ...summary, credential_count: 1 }))
})

test('collection projects empty and configured states without credential counts', async () => {
  const { client, requests } = clientReturning({
    observed_at_ms: 1,
    summary: { total: 2, available: 1, unavailable: 1, disabled: 0 },
    items: [
      { ...summary, status: 'unavailable', client_model_count: 1 },
      {
        ...summary,
        id: 2,
        status: 'available',
        credential_configured: true,
        credential_status: 'available',
        client_model_count: 1,
      },
    ].map((item) => ({
      id: item.id,
      name: item.name,

      channel_id: item.channel_id,
      params: item.params,
      provider_url: item.provider_url,
      status: item.status,
      model_count: item.model_count,
      client_model_count: item.client_model_count,
      credential_configured: item.credential_configured,
      credential_status: item.credential_status,
      site_info: null,
    })),
    pagination: { page: 1, page_size: 100, total_items: 2, total_pages: 1 },
  })
  const result = await listGroupCollection(client, { sort: 'recent', page: 1, page_size: 100 })
  assert.equal(
    new URL(requests[0]!.path, 'http://localhost').searchParams.get('include'),
    'site_info',
  )
  assert.equal(result.items[0]?.credential_configured, false)
  assert.equal(result.items[1]?.credential_status, 'available')
  assert.equal('credential_counts' in (result.items[1] ?? {}), false)
})

test('settings reads and writes explicitly request the site extension', async () => {
  const { client, requests } = clientReturning({
    name: 'Provider',
    channel_id: 'openai',
    params: {},
    provider_url: null,
    enabled: true,
    overrides: {},
    effective: {
      first_byte_timeout: 120,
      request_timeout: 600,
      stream_idle_timeout: 300,
      blacklist_threshold: 3,
      header_rules: { set: {}, remove: [] },
      affinity_enabled: true,
      responses_websocket_enabled: true,
      responses_reasoning_status_filter_enabled: false,
    },
    proxy: {
      configured_mode: 'inherit',
      effective_mode: 'direct',
      effective_source: 'default',
      has_auth: false,
    },
    site_access: { user_id: '42', token_configured: false, site_root_url: '', snapshot: null },
  })
  assert.equal((await getGroupSettings(client, 1)).site_access.user_id, '42')
  assert.equal((await updateGroupSettings(client, 1, { enabled: true })).site_access.user_id, '42')
  assert.deepEqual(
    requests.map((request) => [
      request.method,
      new URL(request.path, 'http://localhost').searchParams.get('include'),
    ]),
    [
      ['GET', 'site_info'],
      ['PUT', 'site_info'],
    ],
  )
})
