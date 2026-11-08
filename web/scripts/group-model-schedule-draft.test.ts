import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  createModelDraft,
  normalizedModels,
  sameModels,
} from '../src/shared/domain/groups/models/model-diff.ts'
import {
  cacheGroupModels,
  getGroupModels,
  replaceGroupModelsResource,
} from '../src/shared/control/resources/groups.ts'
import { QueryClient } from '@tanstack/react-query'
import { controlQueryKeys } from '../src/shared/control/query-keys.ts'
import type { ApiClient } from '../src/shared/http/client.ts'

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
      weight: null,
      priority: null,
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
    { id: 'm', alias: '', alias_enabled: false, weight: null, priority: null, entry_id: 'e1' },
  ])
})
