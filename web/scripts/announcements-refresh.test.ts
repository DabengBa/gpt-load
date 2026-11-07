import test from 'node:test'
import assert from 'node:assert/strict'
import {
  QueryClient,
  QueryObserver,
  environmentManager,
  focusManager,
  timeoutManager,
} from '@tanstack/query-core'

import {
  announcementsQueryOptions,
  announcementsRefetchIntervalMs,
  projectAnnouncementsResponse,
} from '@shared/control/resources/announcements'
import type { AnnouncementsResponseDto } from '@shared/control/types'
import type { ApiClient } from '../src/shared/http/client.ts'
import { FakeClock, flush } from './fake-clock.ts'

const clock = new FakeClock()
// Must be installed before the first QueryClient/QueryObserver creates a timer.
timeoutManager.setTimeoutProvider(clock.provider)
// Node has no `window`, so query-core would refuse to schedule refetch intervals.
environmentManager.setIsServer(() => false)
focusManager.setFocused(true)

function emptyFeed(): AnnouncementsResponseDto {
  return { items: [], sites: [] }
}

test('announcement source status survives projection and legacy feeds remain readable', () => {
  const legacy = projectAnnouncementsResponse({ items: [], sites: [] })
  assert.deepEqual(legacy.sources, [])
  const sources = [
    {
      group: { id: 7, name: 'group-7' },
      site_root_url: 'https://site.example',
      site_name: 'Site',
      announcement: { status: 'error', fetched_at_ms: 1000 },
      notice: { status: 'pending', fetched_at_ms: null },
    },
  ]
  assert.deepEqual(projectAnnouncementsResponse({ items: [], sites: [], sources }).sources, [
    {
      ...sources[0],
      announcement: { ...sources[0].announcement, last_error: '' },
      notice: { ...sources[0].notice, last_error: '' },
    },
  ])
})

function feedWith(content: string): AnnouncementsResponseDto {
  return {
    items: [
      {
        id: 'https://site.example#notice-1',
        site_root_url: 'https://site.example',
        site_name: 'Site',
        content,
        extra: '',
        level: '',
        source: 'notice',
        event_at_ms: 1_700_000_000_000,
        fetched_at_ms: 1_700_000_000_000,
        groups: [{ id: 7, name: 'group-7' }],
      },
    ],
    sites: [{ site_root_url: 'https://site.example', site_name: 'Site', count: 1 }],
  }
}

function observe(api: ApiClient) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const observer = new QueryObserver(client, {
    ...announcementsQueryOptions(api, {}),
    gcTime: Infinity,
  })
  const unsubscribe = observer.subscribe(() => {})
  return { client, observer, unsubscribe }
}

function contents(observer: QueryObserver): string[] {
  return (observer.getCurrentResult().data?.items ?? []).map((item) => item.content)
}

test('announcement polling stays on one interval for empty and populated feeds', () => {
  const options = announcementsQueryOptions({
    request: async () => emptyFeed(),
  } as unknown as ApiClient)

  // The read endpoint carries no sync status, so the client cannot tell a running
  // sync from an idle site. Neither an empty page nor a long empty period may slow
  // the cadence down, otherwise later sync results are invisible for a long time.
  assert.equal(options.refetchInterval, announcementsRefetchIntervalMs)
  assert.equal(options.refetchIntervalInBackground, false)
  assert.equal(options.refetchOnWindowFocus, 'always')
})

test('an announcement published before 1970 keeps its negative event time', () => {
  const feed = feedWith('announcement from 1960')
  feed.items[0].event_at_ms = -315_619_200_000

  // The backend stores RFC3339 event times, so a pre-1970 publication is a legal
  // negative epoch. Projecting it as an unsigned timestamp threw and blanked the page.
  const projected = projectAnnouncementsResponse(feed)
  assert.equal(projected.items[0].event_at_ms, -315_619_200_000)
  assert.equal(new Date(projected.items[0].event_at_ms).getUTCFullYear(), 1960)
  assert.equal(projected.items[0].content, 'announcement from 1960')
})

test('a page with a pre-1970 announcement renders the whole feed', async () => {
  let calls = 0
  const api: ApiClient = {
    request: async () => {
      calls++
      const feed = feedWith('announcement from 1960')
      return { ...feed, items: [{ ...feed.items[0], event_at_ms: -315_619_200_000 }] }
    },
  }
  const { client, observer, unsubscribe } = observe(api)
  try {
    await flush()
    assert.equal(calls, 1)
    // One negative timestamp must not fail projection and hide every other item.
    assert.equal(observer.getCurrentResult().status, 'success')
    assert.deepEqual(contents(observer), ['announcement from 1960'])
    assert.equal(observer.getCurrentResult().data?.items[0].event_at_ms, -315_619_200_000)
  } finally {
    unsubscribe()
    client.clear()
  }
})

test('an empty page kept open past 30 empty responses shows a new announcement', async () => {
  let calls = 0
  const api: ApiClient = {
    request: async () => {
      calls++
      // The periodic sync only produces the first announcement long after the
      // old bounded fast window had expired (31 empty reads).
      return calls <= 31 ? emptyFeed() : feedWith('announcement after 30 empty reads')
    },
  }
  const { client, observer, unsubscribe } = observe(api)
  try {
    await flush()
    for (let poll = 0; poll < 30; poll++) await clock.advance(announcementsRefetchIntervalMs)
    assert.equal(calls, 31)
    assert.deepEqual(contents(observer), [])
    // The 31st empty response is still polled quickly instead of parking for an hour.
    assert.equal(clock.intervals.at(-1), announcementsRefetchIntervalMs)

    await clock.advance(announcementsRefetchIntervalMs)
    assert.equal(calls, 32)
    assert.deepEqual(contents(observer), ['announcement after 30 empty reads'])
  } finally {
    unsubscribe()
    client.clear()
  }
})

test('a populated page picks up announcements from later periodic syncs', async () => {
  let calls = 0
  const api: ApiClient = {
    request: async () => {
      calls++
      return calls === 1
        ? feedWith('first announcement')
        : {
            items: [
              ...feedWith('first announcement').items,
              {
                ...feedWith('second announcement').items[0],
                id: 'https://site.example#notice-2',
                content: 'second announcement',
                event_at_ms: 1_700_000_100_000,
              },
            ],
            sites: feedWith('second announcement').sites,
          }
    },
  }
  const { client, observer, unsubscribe } = observe(api)
  try {
    await flush()
    assert.deepEqual(contents(observer), ['first announcement'])

    // The next hourly backend sync lands, and the open page reflects it without a reload.
    await clock.advance(announcementsRefetchIntervalMs)
    assert.equal(calls, 2)
    assert.deepEqual(contents(observer), ['first announcement', 'second announcement'])
  } finally {
    unsubscribe()
    client.clear()
  }
})
