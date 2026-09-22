import type { Page } from '@playwright/test'

// Fixtures for U003: the classic group model editor renders the server-owned
// read-only `test_alias`, keeps it out of the save payload, and stays usable
// at desktop/mobile widths without breaking keyboard focus order.

export const modelTestAliasGroupId = 1
export const rowOneTestAlias = 'a4g233'
export const rowTwoTestAlias = 'k9x2q7'

function envelope(data: unknown) {
  return { code: 0, message: 'OK', data }
}

function response(data: unknown, status = 200) {
  return {
    status,
    contentType: 'application/json',
    body: JSON.stringify(envelope(data)),
  }
}

export function groupModelItems() {
  return [
    {
      id: 'model-a',
      alias: '',
      alias_enabled: false,
      client_model: 'model-a',
      weight: 50,
      priority: 1,
      pricing_status: 'configured',
      entry_id: 'entry-1',
      test_alias: rowOneTestAlias,
    },
    {
      id: 'model-b',
      alias: 'worker-b',
      alias_enabled: true,
      client_model: 'worker-b',
      weight: 50,
      priority: 2,
      pricing_status: 'configured',
      entry_id: 'entry-2',
      test_alias: rowTwoTestAlias,
    },
  ]
}

function groupSummary() {
  return {
    id: modelTestAliasGroupId,
    name: 'alpha group',
    channel_id: 'openai',
    connection_type: 'api_key',
    params: {},
    provider_url: null,
    price_multiplier: '1',
    service_status: 'available',
    service_status_reason: null,
    credential_count: 1,
    model_count: 2,
  }
}

function groupSettings() {
  return {
    name: 'alpha group',
    channel_id: 'openai',
    connection_type: 'api_key',
    params: {},
    provider_url: null,
    price_multiplier: '1',
    enabled: true,
    overrides: {},
    effective: {
      first_byte_timeout: 30,
      request_timeout: 600,
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
  }
}

export type SavedModelBody = unknown

export async function installModelTestAliasRoutes(page: Page): Promise<SavedModelBody[]> {
  await page.addInitScript(() => {
    window.localStorage.setItem('gpt-load.auth-key', 'e2e-admin-key')
  })

  const savedModelBodies: unknown[] = []
  await page.route(
    (url) => url.pathname === '/api' || url.pathname.startsWith('/api/'),
    async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      const path = url.pathname

      if (path === '/api/auth/session') {
        await route.fulfill(response({ authenticated: true, principal_type: 'admin' }))
        return
      }
      if (path === '/api/channels') {
        await route.fulfill(response({ items: [], total: 0 }))
        return
      }
      if (path === `/api/groups/${modelTestAliasGroupId}`) {
        await route.fulfill(response(groupSummary()))
        return
      }
      if (path === `/api/groups/${modelTestAliasGroupId}/settings`) {
        await route.fulfill(response(groupSettings()))
        return
      }
      if (path === `/api/groups/${modelTestAliasGroupId}/credentials`) {
        await route.fulfill(
          response({
            observed_at_ms: 1_700_000_000_000,
            stats_window_seconds: 86_400,
            summary: { total: 0, available: 0, cooldown: 0, blacklisted: 0, disabled: 0 },
            items: [],
            pagination: { page: 1, page_size: 20, total_items: 0, total_pages: 0 },
          }),
        )
        return
      }
      if (path === `/api/groups/${modelTestAliasGroupId}/models`) {
        if (request.method() === 'PUT') {
          savedModelBodies.push(request.postDataJSON())
          await route.fulfill(response({ items: groupModelItems(), total: 2, pending: 0 }))
          return
        }
        await route.fulfill(response({ items: groupModelItems(), total: 2, pending: 0 }))
        return
      }
      if (path === '/api/groups/options') {
        await route.fulfill(response([]))
        return
      }
      if (path === '/api/access-keys/options') {
        await route.fulfill(response([{ id: 7, name: 'e2e access key', status: 'active' }]))
        return
      }

      await route.fulfill(response({}, 404))
    },
  )
  return savedModelBodies
}
