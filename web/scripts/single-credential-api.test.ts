import assert from 'node:assert/strict'
import test from 'node:test'
import { createApiClient } from '../src/shared/http/client.ts'
import {
  createGroup,
  importGroupCredentials,
  listGroupCollection,
  projectGroupSummary,
} from '../src/shared/control/resources/groups.ts'
import { connectGroupCredential } from '../src/shared/control/resources/credential-stages.ts'

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
    connection_type: 'api_key' as const,
    params: {},
    models: [],
    confirm_same_target: false,
  }
  assert.equal(
    (await createGroup(client, { ...base, credential: 'provider-key' }, 'create')).credential_id,
    9,
  )
  assert.equal(
    (
      await createGroup(
        client,
        { ...base, connection_type: 'subscription', staged_credential_id: 'stage-one' },
        'subscribe',
      )
    ).credential_id,
    9,
  )
  assert.deepEqual(requests[0]?.body, { ...base, credential: 'provider-key' })
  assert.deepEqual(requests[1]?.body, {
    ...base,
    connection_type: 'subscription',
    staged_credential_id: 'stage-one',
  })
  const configured = clientReturning({ group_id: 1, credential_id: 9 })
  await importGroupCredentials(configured.client, 1, { credential: 'key' }, 'configure')
  assert.deepEqual(configured.requests[0], {
    path: '/api/groups/1/credential',
    body: { credential: 'key' },
    method: 'POST',
  })
})

test('connect sends exactly one stage to the singular resource', async () => {
  const { client, requests } = clientReturning({ group_id: 1, credential_id: 9 })
  assert.equal(
    (await connectGroupCredential(client, 1, 'stage-one', 'connect', 0)).credential_id,
    9,
  )
  assert.deepEqual(requests[0], {
    path: '/api/groups/1/credential/connect',
    body: { staged_credential_id: 'stage-one' },
    method: 'POST',
  })
})

for (const expectedCredentialID of [0, 99]) {
  test(`connect sends expected credential header ${expectedCredentialID}`, async () => {
    const headers: Headers[] = []
    const client = createApiClient({
      fetch: async (_input, init) => {
        headers.push(new Headers(init?.headers))
        return new Response(
          JSON.stringify({ code: 0, message: 'ok', data: { group_id: 1, credential_id: 9 } }),
        )
      },
      getAuthKey: () => 'test-key',
      getLocale: () => 'en-US',
      onUnauthorized: () => {},
    })
    await connectGroupCredential(client, 1, 'stage-one', 'connect', expectedCredentialID)
    assert.equal(headers[0]?.get('X-Credential-ID'), String(expectedCredentialID))
    assert.equal(headers[0]?.get('Idempotency-Key'), 'connect')
  })
}

test('connect rejects stage arrays before making a request', async () => {
  const { client, requests } = clientReturning({ group_id: 1, credential_id: 9 })
  await assert.rejects(
    connectGroupCredential(client, 1, ['one', 'two'] as unknown as string, 'connect', 0),
  )
  assert.equal(requests.length, 0)
})

test('connect rejects the retired plural result', async () => {
  const { client } = clientReturning({
    group_id: 1,
    credentials_added: 1,
    credentials_duplicated: 0,
  })
  await assert.rejects(connectGroupCredential(client, 1, 'stage-one', 'connect', 0))
})

const summary = {
  id: 1,
  name: 'Provider',

  channel_id: 'openai',
  connection_type: 'api_key',
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
  const { client } = clientReturning({
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
      connection_type: item.connection_type,
      params: item.params,
      provider_url: item.provider_url,
      status: item.status,
      model_count: item.model_count,
      client_model_count: item.client_model_count,
      credential_configured: item.credential_configured,
      credential_status: item.credential_status,
    })),
    pagination: { page: 1, page_size: 100, total_items: 2, total_pages: 1 },
  })
  const result = await listGroupCollection(client, { sort: 'recent', page: 1, page_size: 100 })
  assert.equal(result.items[0]?.credential_configured, false)
  assert.equal(result.items[1]?.credential_status, 'available')
  assert.equal('credential_counts' in (result.items[1] ?? {}), false)
})
