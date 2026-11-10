import assert from 'node:assert/strict'
import test from 'node:test'
import {
  QueryClient,
  QueryObserver,
  environmentManager,
  focusManager,
  timeoutManager,
} from '@tanstack/query-core'
import type { ApiClient } from '../src/shared/http/client.ts'
import {
  groupCollectionQueryOptions,
  groupOptionsQueryOptions,
  groupSummaryQueryOptions,
  groupSettingsQueryOptions,
  groupModelsQueryOptions,
} from '../src/shared/control/resources/groups.ts'
import { FakeClock, flush } from './fake-clock.ts'

const summary = {
  id: 1,
  name: 'Cached',
  channel_id: 'openai',
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

test('historical provider URLs with oversized ports remain readable', async () => {
  const api: ApiClient = {
    request: async () => ({ ...summary, provider_url: 'https://example.com:99999/path' }),
  }
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  try {
    const result = await cache.fetchQuery(groupSummaryQueryOptions(api, 1))
    assert.equal(result.provider_url, 'https://example.com:99999/path')
  } finally {
    cache.clear()
  }
})

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

const clock = new FakeClock()
// Installed before the first observer so query-core schedules on the virtual clock.
timeoutManager.setTimeoutProvider(clock.provider)
// Node has no `window`; without this query-core refuses to schedule intervals.
environmentManager.setIsServer(() => false)
focusManager.setFocused(true)

/** The backend resolver ticks once a minute, so a visible page must poll at least that often. */
const oneMinute = 60_000

const syncedSiteInfo = {
  status: 'ok',
  site_root_url: 'https://site.example',
  site_name: 'Site',
  remaining_amount: 12.5,
  currency_symbol: '$',
  effective_group: 'default',
  group_ratio: 1,
  fetched_at_ms: 1_700_000_000_000,
  last_error: '',
}

function collectionWith(fetchedAt: number, name: string) {
  return {
    observed_at_ms: fetchedAt,
    items: [
      {
        id: 1,
        name,
        channel_id: 'openai',
        params: {},
        provider_url: null,
        status: 'available',
        model_count: 0,
        client_model_count: 0,
        credential_configured: false,
        credential_status: null,
        site_info: { ...syncedSiteInfo, fetched_at_ms: fetchedAt },
      },
    ],
    summary: { total: 1, available: 1, unavailable: 0, disabled: 0 },
    pagination: { page: 1, page_size: 100, total_items: 1, total_pages: 1 },
  }
}

function settingsWith(fetchedAt: number, name: string) {
  return {
    name,
    channel_id: 'openai',
    params: {},
    provider_url: null,
    enabled: true,
    overrides: {},
    effective: {
      first_byte_timeout: 30,
      request_timeout: 60,
      stream_idle_timeout: 30,
      blacklist_threshold: 3,
      header_rules: { set: {}, remove: [] },
      affinity_enabled: false,
      responses_websocket_enabled: false,
      responses_reasoning_status_filter_enabled: false,
    },
    proxy: {
      configured_mode: 'inherit',
      effective_mode: 'direct',
      effective_source: 'default',
      has_auth: false,
    },
    site_access: {
      user_id: 'site-user',
      token_configured: true,
      site_root_url: 'https://site.example',
      snapshot: { ...syncedSiteInfo, fetched_at_ms: fetchedAt, announcement_count: 0 },
    },
  }
}

function observeWithClock<T extends { queryKey: readonly unknown[] }>(
  cache: QueryClient,
  options: T,
) {
  const observer = new QueryObserver(cache, { ...options, gcTime: Infinity })
  const unsubscribe = observer.subscribe(() => {})
  return { observer, unsubscribe }
}

test('a visible group list picks up a backend sync that lands mid-interval', async () => {
  let calls = 0
  const api: ApiClient = {
    request: async () => {
      calls++
      // The 60 minute backend sync has just written a newer snapshot; a client that
      // polled on its own 60 minute phase would not see it for almost an hour.
      return calls === 1
        ? collectionWith(1_700_000_000_000, 'before sync')
        : collectionWith(1_700_000_060_000, 'after sync')
    },
  }
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const { observer, unsubscribe } = observeWithClock(
    cache,
    groupCollectionQueryOptions(api, { page: 1 }),
  )
  try {
    await flush()
    assert.equal(calls, 1)
    assert.equal(observer.getCurrentResult().data?.items[0].name, 'before sync')
    // The observed cadence must be short enough to catch a sync tick that landed
    // between two polls; the old 60 minute interval mirrored the backend period.
    assert.equal(clock.intervals.at(-1), oneMinute)

    await clock.advance(oneMinute)
    assert.equal(calls, 2)
    assert.equal(observer.getCurrentResult().data?.items[0].name, 'after sync')
    assert.equal(
      observer.getCurrentResult().data?.items[0].site_info?.fetched_at_ms,
      1_700_000_060_000,
    )
  } finally {
    unsubscribe()
    cache.clear()
  }
})

test('an open group settings page picks up a backend sync without a reload', async () => {
  let calls = 0
  const api: ApiClient = {
    request: async () => {
      calls++
      return calls === 1
        ? settingsWith(1_700_000_000_000, 'before sync')
        : settingsWith(1_700_000_060_000, 'after sync')
    },
  }
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const { observer, unsubscribe } = observeWithClock(cache, groupSettingsQueryOptions(api, 1))
  try {
    await flush()
    assert.equal(calls, 1)
    assert.equal(
      observer.getCurrentResult().data?.site_access.snapshot?.fetched_at_ms,
      1_700_000_000_000,
    )
    assert.equal(clock.intervals.at(-1), oneMinute)

    await clock.advance(oneMinute)
    assert.equal(calls, 2)
    assert.equal(
      observer.getCurrentResult().data?.site_access.snapshot?.fetched_at_ms,
      1_700_000_060_000,
    )
    assert.equal(observer.getCurrentResult().data?.name, 'after sync')
  } finally {
    unsubscribe()
    cache.clear()
  }
})

test('a pending site info still follows the fast cadence and then slows to a minute', async () => {
  let calls = 0
  const api: ApiClient = {
    request: async () => {
      calls++
      if (calls === 1) {
        const pending = collectionWith(1_700_000_000_000, 'pending group')
        pending.items[0].site_info.status = 'pending'
        return pending
      }
      return collectionWith(1_700_000_060_000, 'synced group')
    },
  }
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const { observer, unsubscribe } = observeWithClock(
    cache,
    groupCollectionQueryOptions(api, { page: 1 }),
  )
  try {
    await flush()
    assert.equal(clock.intervals.at(-1), 2_000)

    // The first sync tick resolves the pending snapshot.
    await clock.advance(2_000)
    assert.equal(calls, 2)
    assert.equal(observer.getCurrentResult().data?.items[0].site_info?.status, 'ok')
    assert.equal(clock.intervals.at(-1), oneMinute)
  } finally {
    unsubscribe()
    cache.clear()
  }
})
