import assert from 'node:assert/strict'
import test from 'node:test'
import { createApiClient, type ApiClient } from '../src/shared/http/client.ts'
import { QueryClient } from '@tanstack/query-core'
import { controlQueryKeys } from '../src/shared/control/query-keys.ts'
import * as credentials from '../src/shared/control/resources/credentials.ts'

test('single credential actions call singular endpoints', async () => {
  const calls: string[] = []
  const stopped = new Error('request recorded')
  const client: ApiClient = {
    request: async (path, options) => {
      assert.equal(new Headers(options?.headers).get('X-Credential-ID'), '11')
      calls.push(path)
      throw stopped
    },
  }
  const actions = [
    ['restore', () => credentials.restoreCredential(client, 7, 11)],
    ['test', () => credentials.testCredentialConnection(client, 7, 11)],
    ['reveal', () => credentials.revealCredential(client, 7, 11)],
  ] as const
  for (const [action, invoke] of actions) {
    await assert.rejects(invoke, (error) => error === stopped)
    assert.equal(calls.pop(), `/api/groups/7/credential/${action}`)
  }
})

const item = {
  credential_id: 11,
  connection_type: 'api_key' as const,
  secret_version: 1,
  mask: 'test-mask',
  effective_status: 'available' as const,
  recent_success_count: 0,
  recent_failure_count: 0,
  consecutive_failure_count: 0,
  last_failure_category: 'ok' as const,
  last_status_code: null,
  cooldown_until_ms: null,
  recovery: { mode: 'none' as const, automatic: false, at_ms: null },
}

test('real HTTP client sends singular edit/delete contracts', async () => {
  const requests: { path: string; options: RequestInit }[] = []
  let response: unknown = item
  const client = createApiClient({
    fetch: async (input, options) => {
      requests.push({ path: String(input), options: options ?? {} })
      return new Response(JSON.stringify({ code: 0, message: 'ok', data: response }), {
        headers: { 'Content-Type': 'application/json' },
      })
    },
    getAuthKey: () => 'fixture-auth',
    getLocale: () => 'en-US',
    onUnauthorized: () => assert.fail('unexpected unauthorized'),
  })
  assert.deepEqual(
    await credentials.updateCredential(client, 7, 11, { credential: 'fixture-value' }),
    item,
  )
  assert.equal(requests.at(-1)?.path, '/api/groups/7/credential')
  assert.equal(requests.at(-1)?.options.method, 'PUT')
  assert.deepEqual(JSON.parse(String(requests.at(-1)?.options.body)), {
    credential: 'fixture-value',
  })
  response = null
  await credentials.deleteCredential(client, 7, 11)
  assert.equal(requests.at(-1)?.options.method, 'DELETE')
  assert.equal(requests.at(-1)?.path, '/api/groups/7/credential')
  for (const request of requests) {
    assert.equal(new Headers(request.options.headers).get('X-Credential-ID'), '11')
  }
  response = { credential: null }
  assert.deepEqual(await credentials.getCredentialDetail(client, 7), response)
  assert.equal(requests.at(-1)?.options.method, 'GET')
  assert.equal(requests.at(-1)?.path, '/api/groups/7/credential')
})

test('single credential update writes the detail cache without collection shape', async () => {
  const client = new QueryClient()
  await credentials.cacheCredentialItem(client, 7, item)
  assert.deepEqual(client.getQueryData(controlQueryKeys.groups.credentialsAll(7)), {
    credential: item,
  })
  client.clear()
})

test('empty group detail is nullable', async () => {
  assert.deepEqual(credentials.projectCredentialDetail({ credential: null }), {
    credential: null,
  })
})

test('actions project successful responses and reject malformed upstream results', async () => {
  let response: unknown = item
  const client = createApiClient({
    fetch: async () => new Response(JSON.stringify({ code: 0, message: 'ok', data: response })),
    getAuthKey: () => 'fixture-auth',
    getLocale: () => 'en-US',
    onUnauthorized: () => assert.fail('unexpected unauthorized'),
  })
  assert.deepEqual(await credentials.restoreCredential(client, 7, 11), item)
  response = {
    credential_id: 11,
    credential: { api_key: 'fixture-value' },
    revealed_at_ms: 1700000000000,
  }
  assert.deepEqual(await credentials.revealCredential(client, 7, 11), response)
  response = {
    outcome: 'passed',
    model: 'fixture-model',
    protocol: 'openai-completions',
    latency_ms: 1,
    reason: null,
    recovered: true,
    log_id: null,
    tested_at_ms: 1700000000000,
  }
  assert.deepEqual(await credentials.testCredentialConnection(client, 7, 11), response)
})

test('item mutations reject a response for a different credential', async () => {
  const client: ApiClient = { request: async () => ({ ...item, credential_id: 12 }) }
  for (const invoke of [
    () => credentials.updateCredential(client, 7, 11, { credential: 'fixture-value' }),
    () => credentials.restoreCredential(client, 7, 11),
  ])
    await assert.rejects(invoke, { name: 'InvalidResponseError' })
})
