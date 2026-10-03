import { expect, test } from '@playwright/test'

test('access-key clears administrative context and never requests management APIs', async ({
  page,
}) => {
  const requests: string[] = []
  await page.addInitScript(() => localStorage.setItem('gpt-load.auth-key', 'core-key'))
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url())
    requests.push(url.pathname + url.search)
    const data =
      url.pathname === '/api/auth/session'
        ? { authenticated: true, principal_type: 'access_key' }
        : {
            summary: {
              client_model_count: 0,
              upstream_model_count: 0,
              price_count: 0,
              pending_price_count: 0,
              unreferenced_price_count: 0,
            },
            catalog: {
              available: false,
              checked_at_ms: 0,
              successful_fetch_at_ms: 0,
              error_code: '',
            },
            items: [],
            pagination: { page: 1, page_size: 10, total_items: 0, total_pages: 0 },
          }
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'OK', data }),
    })
  })
  await page.goto(
    '/schedule?schedule_model=allowed&selected_price_id=7&schedule_row=1%3Ax&schedule_group=1&schedule_draft=%7B%7D',
  )
  await expect(page.getByTestId('schedule-read-only')).toBeVisible()
  await expect.poll(() => new URL(page.url()).search).toBe('?schedule_model=allowed')
  await expect(page.getByRole('textbox', { name: 'External model' })).toHaveValue('allowed')
  await expect
    .poll(() => requests.some((url) => url.startsWith('/api/models?') && url.includes('q=allowed')))
    .toBe(true)
  expect(
    requests.filter(
      (url) => url.startsWith('/api/model-route/') || url.startsWith('/api/model-prices'),
    ),
  ).toEqual([])
  await expect(page.getByTestId('schedule-panel')).toHaveCount(0)
})

test('admin searches the model list and recovers an empty filter', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('gpt-load.auth-key', 'core-key'))
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    const data =
      path === '/api/auth/session'
        ? { authenticated: true, principal_type: 'admin' }
        : {
            items: ['worker', 'other'].map((external_model) => ({
              external_model,
              protocol: 'openai-completions',
              operation: 'chat_completion',
              candidate_count: 0,
              group_count: 0,
              cooled_candidates: 0,
              blacklisted_candidates: 0,
            })),
          }
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'OK', data }),
    })
  })
  await page.goto('/schedule')
  const search = page.getByRole('textbox', { name: 'External model' })
  await search.fill('worker')
  await expect(page.getByRole('button', { name: 'worker', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'other', exact: true })).toHaveCount(0)
  await search.fill('missing')
  await expect(page.getByRole('button', { name: 'worker', exact: true })).toHaveCount(0)
  await search.fill('')
  await expect(page.getByRole('button', { name: 'other', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'worker', exact: true }).click()
  await expect(page).toHaveURL(/schedule_model=worker/)
})
