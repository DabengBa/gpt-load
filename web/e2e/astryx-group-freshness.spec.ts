import { expect, test } from '@playwright/test'

test('dirty settings and model drafts survive focus and reconnect', async ({ page }) => {
  test.setTimeout(90_000)
  // Shared-worktree artifacts and unrelated workers must not trigger Vite HMR
  // document reloads while this test holds intentionally unsaved local state.
  await page.routeWebSocket('**', (socket) => socket.close())
  const reads = { settings: 0, models: 0 }
  const base = {
    name: 'Freshness group',
    channel_id: 'openai',
    connection_type: 'api_key',
    params: {},
    provider_url: null,
  }
  const runtime = {
    first_byte_timeout: 30,
    request_timeout: 60,
    stream_idle_timeout: 30,
    blacklist_threshold: 3,
    header_rules: { set: {}, remove: [] },
    affinity_enabled: false,
    responses_websocket_enabled: false,
    responses_reasoning_status_filter_enabled: false,
  }
  await page.addInitScript(() => {
    localStorage.setItem('gpt-load.auth-key', 'e2e-freshness')
    localStorage.setItem('gpt-load.locale', 'en-US')
  })
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    let data: unknown = {}
    if (path === '/api/auth/session') data = { authenticated: true, principal_type: 'admin' }
    if (path === '/api/channels') data = { items: [], total: 0 }
    if (path === '/api/groups/1')
      data = {
        ...base,
        id: 1,
        service_status: 'available',
        service_status_reason: null,
        credential_configured: false,
        credential_status: null,
        model_count: 1,
      }
    if (path === '/api/groups/1/settings') {
      reads.settings++
      data = {
        ...base,
        enabled: true,
        overrides: {},
        effective: runtime,
        proxy: {
          configured_mode: 'inherit',
          effective_mode: 'direct',
          effective_source: 'default',
          has_auth: false,
        },
      }
    }
    if (path === '/api/groups/1/models') {
      reads.models++
      data = {
        items: [
          {
            id: 'gpt-4o',
            alias: 'original',
            alias_enabled: true,
            client_model: 'original',
            test_alias: 'abc123',
            weight: null,
            priority: null,
            pricing_status: 'pending',
          },
        ],
        total: 1,
        pending: 1,
      }
    }
    if (path === '/api/groups/1/credential') data = { credential: null, observation: null }
    await route.fulfill({ json: { code: 0, message: 'ok', data } })
  })
  await page.goto('/groups/1', { waitUntil: 'commit' })
  await expect(page.locator('#group-detail-title')).toBeVisible({ timeout: 60_000 })
  const name = page.getByRole('textbox', { name: /^Name(?: Name)?$/ })
  const alias = page.locator('input[aria-label="Client alias for gpt-4o"]')
  await expect(name).toBeVisible({ timeout: 60_000 })
  await name.fill('Unsaved settings')
  await alias.fill('unsaved-model')
  const before = { ...reads }
  // Settings and models are both mounted in the unified editor; browser
  // lifecycle events must not replace either draft with cached snapshots.
  await page.evaluate(() => {
    window.dispatchEvent(new Event('visibilitychange'))
    window.dispatchEvent(new Event('offline'))
    window.dispatchEvent(new Event('online'))
  })
  await expect(name).toHaveValue('Unsaved settings')
  await expect(alias).toHaveValue('unsaved-model')
  await expect(page.getByRole('button', { name: 'Save settings', exact: true })).toBeEnabled()
  expect(reads).toEqual(before)
})
