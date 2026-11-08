import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  createModelDraft,
  normalizedModels,
  reconcileUncertainModels,
  sameModels,
  syncedModels,
} from '../src/shared/domain/groups/models/model-diff.ts'
import {
  cacheGroupModels,
  getGroupModels,
  replaceGroupModelsResource,
  updateGroupConfiguration,
} from '../src/shared/control/resources/groups.ts'
import { QueryClient } from '@tanstack/react-query'
import { controlQueryKeys } from '../src/shared/control/query-keys.ts'
import type { ApiClient } from '../src/shared/http/client.ts'

test('combined settings and models use one atomic request', async () => {
  const calls: unknown[] = []
  const api: ApiClient = {
    request: async (url, options) => {
      calls.push([url, options])
      throw new Error('stop')
    },
  }
  const settings = { channel_id: 'gemini' }
  const models = [{ id: 'm', alias: '', alias_enabled: false, reasoning_effort: null }]
  await assert.rejects(updateGroupConfiguration(api, 1, settings, models), /stop/)
  assert.deepEqual(calls, [
    [
      '/api/groups/1/configuration',
      { method: 'PUT', json: { settings, models }, signal: undefined },
    ],
  ])
})

test('uncertain sync adopts fresh server membership and routing', () => {
  const row = (id: string, entry_id: string, key: number) => ({
    id,
    entry_id,
    key,
    alias: '',
    alias_enabled: false,
    name: id,
    sources: [],
    pricing_status: 'pending' as const,
  })
  const old = [row('removed', 'e1', 1)]
  const remote = [{ ...row('added', 'e2', 2), weight: 9 }]
  const result = reconcileUncertainModels(old, old, remote, 'sync')
  assert.deepEqual(result.draft, remote)
  assert.equal(result.conflict, false)
})

test('uncertain new-row commit keeps its identity and later schedule edits', () => {
  const row = {
    id: 'new',
    key: 1,
    alias: '',
    alias_enabled: false,
    name: 'new',
    sources: [],
    pricing_status: 'pending' as const,
    reasoning_effort: 'high',
    enabled: false,
  }
  const remote = [{ ...row, key: 2, entry_id: 'e1', reasoning_effort: 'low', enabled: true }]
  const result = reconcileUncertainModels([], [row], remote, 'save')
  assert.equal(result.draft[0].entry_id, 'e1')
  assert.equal(result.draft[0].reasoning_effort, 'high')
  assert.equal(result.draft[0].enabled, false)
  assert.equal(result.conflict, true)
})

test('ambiguous new-row identity cannot be silently retried', () => {
  const row = {
    id: 'new',
    key: 1,
    alias: '',
    alias_enabled: false,
    name: 'new',
    sources: [],
    pricing_status: 'pending' as const,
  }
  const remote = [
    { ...row, key: 2, entry_id: 'e1' },
    { ...row, key: 3, entry_id: 'e2' },
  ]
  assert.equal(reconcileUncertainModels([], [row], remote, 'save').conflict, true)
})

test('uncertain save merges independent remote routing changes and keeps local edit', () => {
  const base = {
    id: 'm',
    entry_id: 'e1',
    key: 1,
    alias: '',
    alias_enabled: false,
    name: 'm',
    sources: [],
    pricing_status: 'pending' as const,
    reasoning_effort: '',
    enabled: true,
  }
  const result = reconcileUncertainModels(
    [base],
    [{ ...base, reasoning_effort: 'high' }],
    [{ ...base, enabled: false }],
    'save',
  )
  assert.equal(result.conflict, false)
  assert.equal(result.draft[0].reasoning_effort, 'high')
  assert.equal(result.draft[0].enabled, false)
})

test('committed new row is matched and can be safely saved again', () => {
  const row = {
    id: 'new',
    key: 1,
    alias: '',
    alias_enabled: false,
    name: 'new',
    sources: [],
    pricing_status: 'pending' as const,
    reasoning_effort: 'high',
    enabled: false,
  }
  const result = reconcileUncertainModels([], [row], [{ ...row, key: 2, entry_id: 'e1' }], 'save')
  assert.equal(result.conflict, false)
  assert.equal(result.draft[0].entry_id, 'e1')
  assert.equal(result.saved[0].key, row.key)
})

test('committed new row matches server defaults without losing identity', () => {
  const row = {
    id: 'new',
    key: 1,
    alias: '',
    alias_enabled: false,
    name: 'new',
    sources: [],
    pricing_status: 'pending' as const,
    reasoning_effort: 'high',
    enabled: false,
  }
  const result = reconcileUncertainModels(
    [],
    [row],
    [
      {
        ...row,
        key: 2,
        entry_id: 'e1',
        concurrency_limit: 0,
        weight: null,
        priority: null,
        circuit_breaker: null,
      },
    ],
    'save',
  )
  assert.equal(result.conflict, false)
  assert.equal(result.draft[0].entry_id, 'e1')
})

test('concurrent reorder is not silently rolled back after an uncertain save', () => {
  const make = (id: string, key: number) => ({
    id,
    entry_id: id,
    key,
    alias: '',
    alias_enabled: false,
    name: id,
    sources: [],
    pricing_status: 'pending' as const,
  })
  const a = make('a', 1)
  const b = make('b', 2)
  const result = reconcileUncertainModels([a, b], [{ ...a, enabled: false }, b], [b, a], 'save')
  assert.equal(result.conflict, true)
})

test('late model GET cannot overwrite successful replacement cache', async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const key = controlQueryKeys.groups.models(1)
  let resolve!: (value: { items: never[]; total: number; pending: number }) => void
  const get = client
    .fetchQuery({
      queryKey: key,
      queryFn: () =>
        new Promise<{ items: never[]; total: number; pending: number }>((done) => {
          resolve = done
        }),
    })
    .catch(() => undefined)
  const fresh = { items: [], total: 1, pending: 0 }
  await cacheGroupModels(client, 1, fresh)
  resolve({ items: [], total: 0, pending: 0 })
  await get
  assert.deepEqual(client.getQueryData(key), fresh)
  client.clear()
})

test('sync retains rows without sending unchanged routing settings', () => {
  const saved = [
    {
      key: 1,
      id: 'm',
      alias: '',
      alias_enabled: false,
      name: 'm',
      sources: [],
      pricing_status: 'pending' as const,
      entry_id: 'e1',
      weight: 3,
      priority: 2,
    },
  ]
  const [wire] = syncedModels(saved, { additions: [], removals: [] }, 'full')
  assert.equal(Object.hasOwn(wire, 'weight'), false)
  assert.equal(Object.hasOwn(wire, 'priority'), false)
})

test('schedule edits are dirty and reverting restores equality', () => {
  const saved = [
    {
      key: 1,
      id: 'model',
      alias: '',
      alias_enabled: false,
      name: 'model',
      sources: [],
      pricing_status: 'pending' as const,
      entry_id: 'e1',
      reasoning_effort: 'high',
      enabled: true,
    },
  ]
  const draft = [{ ...saved[0], reasoning_effort: '', enabled: false }]
  assert.equal(sameModels(draft, saved), false)
  assert.equal(sameModels(saved, saved), true)
})

test('schedule wire opt-in projects values', async () => {
  const urls: string[] = []
  const payload = {
    items: [
      {
        id: 'model',
        alias: '',
        alias_enabled: false,
        client_model: 'model',
        test_alias: 'abc123',
        concurrency_limit: 0,
        weight: null,
        priority: null,
        pricing_status: 'pending',
        entry_id: 'e1',
        reasoning_effort: 'high',
        enabled: false,
      },
    ],
    total: 1,
    pending: 1,
  }
  const api: ApiClient = {
    request: async (url) => {
      urls.push(url)
      return payload
    },
  }
  const response = await getGroupModels(api, 1)
  const saved = createModelDraft(response.items)
  assert.equal(saved[0].reasoning_effort, 'high')
  assert.equal(saved[0].enabled, false)
  await replaceGroupModelsResource(api, 1, { models: [] })
  assert.deepEqual(urls, [
    '/api/groups/1/models?include_schedule=true',
    '/api/groups/1/models?include_schedule=true',
  ])
})

test('replacement sends only schedule fields changed relative to baseline, including new rows', () => {
  const saved = [
    {
      key: 1,
      id: 'm',
      alias: '',
      alias_enabled: false,
      name: 'm',
      sources: [],
      pricing_status: 'pending' as const,
      entry_id: 'e1',
      reasoning_effort: 'high',
      enabled: true,
    },
  ]
  const draft = [
    { ...saved[0], reasoning_effort: '', enabled: false },
    { ...saved[0], key: 2, id: 'new', entry_id: undefined, reasoning_effort: 'low', enabled: true },
  ]
  assert.deepEqual(normalizedModels(draft, saved), [
    {
      id: 'm',
      alias: '',
      alias_enabled: false,
      entry_id: 'e1',
      reasoning_effort: null,
      enabled: false,
    },
    {
      id: 'new',
      alias: '',
      alias_enabled: false,
      weight: null,
      priority: null,
      entry_id: undefined,
      reasoning_effort: 'low',
    },
  ])
  assert.deepEqual(normalizedModels(saved, saved), [
    { id: 'm', alias: '', alias_enabled: false, entry_id: 'e1' },
  ])
})

test('replacement omits unchanged routing settings to preserve concurrent edits', () => {
  const saved = [
    {
      key: 1,
      id: 'm',
      alias: '',
      alias_enabled: false,
      name: 'm',
      sources: [],
      pricing_status: 'pending' as const,
      entry_id: 'e1',
      weight: 3,
      priority: 2,
      concurrency_limit: 5,
      circuit_breaker: null,
    },
  ]
  const [wire] = normalizedModels([{ ...saved[0], reasoning_effort: 'high' }], saved)
  for (const field of ['weight', 'priority', 'concurrency_limit', 'circuit_breaker']) {
    assert.equal(Object.hasOwn(wire, field), false, `${field} must not overwrite a concurrent edit`)
  }
})
