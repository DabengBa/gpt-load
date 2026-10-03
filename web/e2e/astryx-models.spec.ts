import { expect, test } from '@playwright/test'

test('authorized model catalog is read-only inside the dispatch center', async ({ page }) => {
  const managementRequests: string[] = []
  await page.addInitScript(() => localStorage.setItem('gpt-load.auth-key', 'catalog-key'))
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path.startsWith('/api/model-route/') || path.startsWith('/api/model-prices')) {
      managementRequests.push(path)
    }
    const data =
      path === '/api/auth/session'
        ? { authenticated: true, principal_type: 'access_key' }
        : {
            summary: {
              client_model_count: 1,
              upstream_model_count: 1,
              price_count: 1,
              pending_price_count: 0,
              unreferenced_price_count: 0,
            },
            catalog: {
              available: false,
              checked_at_ms: 0,
              successful_fetch_at_ms: 0,
              error_code: '',
            },
            items: [
              {
                client_model: 'authorized-model',
                protocols: ['openai-completions'],
                upstream_models: [
                  {
                    model_id: 'upstream-model',
                    alias_applied: true,
                    route_groups: [],
                    affected_groups: [],
                    catalog_reference: null,
                    price: {
                      id: 7,
                      channel_id: 'openai',
                      channel_name: 'OpenAI',
                      channel_mark: 'O',
                      channel_icon: 'openai',
                      model_id: 'upstream-model',
                      prices: { input: '2.5', output: '10', cache_read: null, cache_write: null },
                      mode_schedules: {},
                      pricing_status: 'configured',
                      method: 'user_set',
                      matched_provider_id: null,
                      match_source: null,
                      referenced: true,
                      reference_count: 1,
                      reference_group_count: 1,
                      context_tiers: [],
                      updated_at_ms: 0,
                      can_reset: false,
                      can_delete: false,
                    },
                  },
                ],
              },
            ],
            pagination: { page: 1, page_size: 10, total_items: 1, total_pages: 1 },
          }
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'OK', data }),
    })
  })
  await page.goto('/schedule')
  const catalog = page.getByTestId('schedule-read-only')
  await expect(catalog.getByText('authorized-model', { exact: true })).toBeVisible()
  await expect(catalog.getByText('openai-completions', { exact: true })).toBeVisible()
  await expect(catalog.getByText(/Input.*2.5/)).toBeVisible()
  await expect(catalog.getByText(/Output.*10/)).toBeVisible()
  await expect(catalog.getByText(/USD.*1M/)).toBeVisible()
  await expect(page.getByTestId('schedule-panel')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toHaveCount(0)
  expect(managementRequests).toEqual([])
})

test('retired models route renders not found instead of a compatibility redirect', async ({
  page,
}) => {
  await page.goto('/models')
  await expect(page.getByText('/models', { exact: true }).last()).toBeVisible()
  await expect(page.getByRole('heading', { name: 'This page does not exist' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Back to Home' })).toBeVisible()
  await expect(page).toHaveURL(/\/models$/)
})
