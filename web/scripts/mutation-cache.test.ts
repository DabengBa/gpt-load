import test from 'node:test'
import assert from 'node:assert/strict'
import { QueryClient, QueryObserver, type QueryKey } from '@tanstack/query-core'
import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import { controlQueryKeys as k } from '@shared/control/query-keys'
import {
  invalidateGroupSettingsDependents,
  invalidateGroupModelDependents,
  clearGroupResourceCaches,
} from '@shared/control/resources/groups'
import {
  clearAuthenticatedClientState,
  createAuthSession,
  authSessionQueryKey,
} from '@shared/controllers/auth-session'
import { createApiClient } from '@shared/http/client'
import { ApiError } from '@shared/http/errors'

const collection = (page: number) => k.groups.collection({ sort: 'name', page, page_size: 100 })
const schedule = [
  k.modelRouteSchedule.index(),
  k.modelRouteSchedule.detail({ protocol: 'openai', external_model: 'test' }),
]
const catalog = [...k.models.all, 'collection', {}]
const prices = [...k.modelPrices(), 'collection', {}]
const unrelated = [
  k.groups.summary(8),
  k.groups.settings(8),
  k.groups.models(8),
  [...k.groups.credentialsAll(8), 'page', 2],
  k.logs.detail('request'),
  k.settings('en-US'),
  ['public'],
]

function seed(client: QueryClient, keys: readonly QueryKey[]) {
  for (const key of keys) client.setQueryData(key, { cached: true })
}
function assertState(client: QueryClient, keys: readonly QueryKey[], invalidated: boolean) {
  for (const key of keys)
    assert.equal(client.getQueryState(key)?.isInvalidated, invalidated, JSON.stringify(key))
}

for (const [name, helper, extra] of [
  [
    'settings',
    invalidateGroupSettingsDependents,
    [k.groups.models(7), [...k.groups.credentialsAll(7), 'page', 1]],
  ],
  ['models', invalidateGroupModelDependents, []],
] as const) {
  test(`${name} helper stales affected reads and leaves unrelated snapshots intact`, async () => {
    const client = new QueryClient()
    const affected = [
      collection(1),
      collection(2),
      k.groups.summary(7),
      k.groups.options(),
      k.home.base(),
      catalog,
      prices,
      ...schedule,
      ...extra,
    ]
    const untouched = [
      ...unrelated,
      k.groups.settings(7),
      ...(name === 'models'
        ? [k.groups.models(7), [...k.groups.credentialsAll(7), 'page', 1]]
        : []),
    ]
    try {
      seed(client, [...affected, ...untouched])
      await helper(client, 7)
      assertState(client, affected, true)
      assertState(client, untouched, false)
    } finally {
      client.clear()
    }
  })
}

test('settings helper refetches an active home consumer through its mocked request', async () => {
  const client = new QueryClient()
  let requests = 0
  const api = createApiClient({
    fetch: async () => {
      requests++
      return new Response(JSON.stringify({ code: 0, message: '', data: { revision: requests } }))
    },
    getAuthKey: () => 'test',
    getLocale: () => 'en-US',
    onUnauthorized() {},
  })
  const options = {
    queryKey: k.home.base(),
    staleTime: Infinity,
    queryFn: () => api.request('/api/home/base'),
  }
  let unsubscribe = () => {}
  try {
    await client.fetchQuery(options)
    const observer = new QueryObserver(client, options)
    unsubscribe = observer.subscribe(() => {})
    await invalidateGroupSettingsDependents(client, 7)
    assert.equal(requests, 2)
    assert.deepEqual(client.getQueryData(k.home.base()), { revision: 2 })
  } finally {
    unsubscribe()
    client.clear()
  }
})

for (const mutation of ['create', 'delete', 'importCredentials'] as const) {
  test(`${mutation} actual centralized plan covers filtered dependent reads`, async () => {
    const client = new QueryClient()
    const affected = [
      collection(1),
      collection(2),
      k.health(),
      k.home.base(),
      k.home.subscriptionAccounts(),
      ...schedule,
      ...(mutation === 'importCredentials'
        ? [k.groups.summary(7), [...k.groups.credentialsAll(7), 'page', 1]]
        : [k.groups.options(), catalog, prices]),
    ]
    try {
      seed(client, [...affected, ...unrelated])
      if (mutation === 'delete') {
        const removed = [
          k.groups.summary(7),
          k.groups.settings(7),
          k.groups.models(7),
          [...k.groups.credentialsAll(7), 'page', 2],
        ]
        seed(client, removed)
        clearGroupResourceCaches(client, 7)
        for (const key of removed) assert.equal(client.getQueryState(key), undefined)
      }
      await applyInvalidationPlan(
        client,
        mutation === 'importCredentials'
          ? mutationInvalidationPlans.group.importCredentials(7)
          : mutationInvalidationPlans.group[mutation],
      )
      assertState(client, affected, true)
      assertState(client, unrelated, false)
    } finally {
      client.clear()
    }
  })
}

for (const mode of ['logout', '401'] as const) {
  test(`${mode} cancels an abort-aware protected request and clears protected caches/mutations (verification)`, async () => {
    const client = new QueryClient()
    let aborted = false
    let started!: () => void
    const ready = new Promise<void>((resolve) => {
      started = resolve
    })
    const session = createAuthSession({
      queryClient: client,
      validate: async () => ({ authenticated: true, principal_type: 'admin' }),
    })
    const api = createApiClient({
      fetch: async (_url, init) => {
        if (_url === '/api/unauthorized')
          return new Response(JSON.stringify({ code: 'UNAUTHORIZED', message: 'Unauthorized' }), {
            status: 401,
          })
        return new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener(
            'abort',
            () => {
              aborted = true
              reject(new DOMException('Aborted', 'AbortError'))
            },
            { once: true },
          )
          started()
        })
      },
      getAuthKey: () => session.getAuthKey(),
      getLocale: () => 'en-US',
      onUnauthorized: () => session.clear(),
    })
    try {
      await session.login('test')
      seed(client, [collection(1), ['public']])
      client
        .getMutationCache()
        .build(client, { mutationKey: ['protected'], mutationFn: async () => 'done' })
      const pending = client
        .fetchQuery({
          queryKey: k.groups.summary(7),
          retry: false,
          queryFn: ({ signal }) => api.request('/api/groups/7', { signal }),
        })
        .catch((error) => error)
      await ready
      if (mode === 'logout') await clearAuthenticatedClientState(client)
      else await assert.rejects(api.request('/api/unauthorized'), ApiError)
      await pending
      assert.equal(aborted, true)
      assert.equal(client.getQueryCache().findAll({ queryKey: k.all }).length, 0)
      assert.equal(client.getQueryState(authSessionQueryKey), undefined)
      assert.equal(client.getMutationCache().getAll().length, 0)
      assert.deepEqual(client.getQueryData(['public']), { cached: true })
      if (mode === '401') assert.equal(session.getState().phase, 'anonymous')
    } finally {
      client.clear()
    }
  })
}
