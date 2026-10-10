import assert from 'node:assert/strict'
import test from 'node:test'
import { QueryClient, QueryObserver } from '@tanstack/query-core'
import type { ApiClient } from '../src/shared/http/client.ts'
import {
  groupCollectionQueryOptions,
  groupOptionsQueryOptions,
  groupSummaryQueryOptions,
  groupSettingsQueryOptions,
  groupModelsQueryOptions,
} from '../src/shared/control/resources/groups.ts'

const summary = {
  id: 1,
  name: 'Cached',
  channel_id: 'openai',
  connection_type: 'api_key',
  params: {},
  provider_url: null,
  service_status: 'available',
  service_status_reason: null,
  credential_configured: false,
  credential_status: null,
  model_count: 0,
}
const collection = {
  observed_at_ms: Date.now(),
  items: [],
  summary: { total: 0, available: 0, unavailable: 0, disabled: 0 },
  pagination: { page: 1, page_size: 100, total_items: 0, total_pages: 0 },
}

for (const resource of ['collection', 'summary', 'options'] as const) {
  for (const trigger of ['mount', 'focus', 'reconnect'] as const) {
    test(`${resource}: stale ${trigger} refresh retains cached data`, async () => {
      const cache = new QueryClient({ defaultOptions: { queries: { retry: false } } })
      let calls = 0
      let resolve!: (value: unknown) => void
      const response = new Promise<unknown>((done) => {
        resolve = done
      })
      const api: ApiClient = {
        request: async () => {
          calls++
          return response
        },
      }
      const options =
        resource === 'collection'
          ? groupCollectionQueryOptions(api, {})
          : resource === 'summary'
            ? groupSummaryQueryOptions(api, 1)
            : groupOptionsQueryOptions(api)
      const data = resource === 'collection' ? collection : resource === 'summary' ? summary : []
      cache.setQueryData(options.queryKey, data, { updatedAt: Date.now() - 60_000 })
      // Heterogeneous DTOs share only the observer lifecycle under test.
      const observer = new QueryObserver(cache, {
        ...options,
        refetchOnMount: trigger === 'mount' ? options.refetchOnMount : false,
      })
      const unsubscribe = observer.subscribe(() => {})
      try {
        if (trigger === 'focus') cache.getQueryCache().onFocus()
        if (trigger === 'reconnect') cache.getQueryCache().onOnline()
        assert.equal(calls, 1)
        assert.deepEqual(observer.getCurrentResult().data, data)
        assert.equal(observer.getCurrentResult().isFetching, true)
        resolve(data)
        await observer.refetch({ cancelRefetch: false })
        assert.equal(observer.getCurrentResult().isFetching, false)
      } finally {
        resolve(data)
        unsubscribe()
        cache.clear()
      }
    })
  }
}

test('fresh prefetched summary is reused on mount, focus and reconnect', async () => {
  const cache = new QueryClient()
  let calls = 0
  const api: ApiClient = {
    request: async () => {
      calls++
      return summary
    },
  }
  const options = groupSummaryQueryOptions(api, 1)
  try {
    await cache.prefetchQuery(options)
    const observer = new QueryObserver(cache, options)
    const unsubscribe = observer.subscribe(() => {})
    cache.getQueryCache().onFocus()
    cache.getQueryCache().onOnline()
    assert.equal(calls, 1)
    assert.equal(observer.getCurrentResult().data?.name, 'Cached')
    unsubscribe()
  } finally {
    cache.clear()
  }
})

test('options retain forced mount refresh even while fresh', async () => {
  const cache = new QueryClient()
  let calls = 0
  const api: ApiClient = {
    request: async () => {
      calls++
      return []
    },
  }
  const options = groupOptionsQueryOptions(api)
  cache.setQueryData(options.queryKey, [])
  const observer = new QueryObserver(cache, options)
  const unsubscribe = observer.subscribe(() => {})
  try {
    await observer.refetch({ cancelRefetch: false })
    assert.equal(calls, 1)
  } finally {
    unsubscribe()
    cache.clear()
  }
})

for (const factory of [groupSettingsQueryOptions, groupModelsQueryOptions]) {
  test(`${factory.name}: old editable snapshots stay manual`, () => {
    const cache = new QueryClient()
    let calls = 0
    const api: ApiClient = {
      request: async () => {
        calls++
        throw new Error('unexpected refresh')
      },
    }
    const options = factory(api, 1)
    cache.setQueryData(options.queryKey, { draftSource: true }, { updatedAt: 1 })
    const observer = new QueryObserver(cache, options)
    const unsubscribe = observer.subscribe(() => {})
    try {
      cache.getQueryCache().onFocus()
      cache.getQueryCache().onOnline()
      assert.equal(calls, 0)
      assert.equal(observer.getCurrentResult().isFetching, false)
    } finally {
      unsubscribe()
      cache.clear()
    }
  })
}
