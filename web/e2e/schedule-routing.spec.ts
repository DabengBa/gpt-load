import { expect, test, type Page } from '@playwright/test'

const adminKey = 'e2e-admin-key'
const accessKey = 'e2e-access-key'

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

async function installScheduleRoutingRoutes(
  page: Page,
  principal: 'admin' | 'access_key' = 'admin',
): Promise<URL[]> {
  const key = principal === 'admin' ? adminKey : accessKey
  await page.addInitScript((authKey) => {
    window.localStorage.setItem('gpt-load.auth-key', authKey)
  }, key)

  const scheduleIndexRequests: URL[] = []
  await page.route(
    (url) => url.pathname === '/api' || url.pathname.startsWith('/api/'),
    async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      const path = url.pathname

      if (path === '/api/auth/session') {
        const requestPrincipal =
          request.headers()['authorization'] === `Bearer ${accessKey}` ? 'access_key' : 'admin'
        await route.fulfill(response({ authenticated: true, principal_type: requestPrincipal }))
        return
      }
      if (path === '/api/model-route/schedule') {
        scheduleIndexRequests.push(url)
        await route.fulfill(
          response({
            items: [
              {
                external_model: 'worker',
                protocol: 'openai-completions',
                operation: 'chat_completion',
                candidate_count: 1,
                group_count: 1,
                has_fallback: false,
                cooled_candidates: 0,
                blacklisted_candidates: 0,
              },
            ],
          }),
        )
        return
      }
      if (path === '/api/access-keys/options') {
        await route.fulfill(response([{ id: 7, name: 'e2e access key', status: 'active' }]))
        return
      }
      if (path === '/api/groups/options') {
        await route.fulfill(response([]))
        return
      }
      if (path === '/api/channels') {
        await route.fulfill(response({ items: [], total: 0 }))
        return
      }

      await route.fulfill(response({}, 404))
    },
  )

  return scheduleIndexRequests
}

async function openDispatchCenter(page: Page): Promise<void> {
  await page.goto('/schedule')
  await page.locator('.desktop-nav').waitFor()
  await page.waitForLoadState('networkidle')
}

test.describe('dispatch center routing', () => {
  test('admin opens /schedule directly and the primary navigation highlights it', async ({
    page,
  }) => {
    const scheduleIndexRequests = await installScheduleRoutingRoutes(page, 'admin')
    await openDispatchCenter(page)

    await expect(page).toHaveURL(/\/schedule$/u)
    await expect(page.getByRole('heading', { name: 'Dispatch center' })).toBeVisible()
    await expect(page.locator('.schedule-panel')).toBeVisible()

    const mode = page.locator('.schedule-panel .app-select__trigger[aria-label="Mode"]')
    await expect(mode).toContainText('All candidates')
    const model = page.locator('.schedule-panel .app-select__trigger[aria-label="External model"]')
    await expect(model).toContainText('Select a model')
    expect(scheduleIndexRequests.length).toBeGreaterThan(0)

    const navLink = page.getByRole('link', { name: 'Dispatch center' }).first()
    await expect(navLink).toBeVisible()
    await expect(navLink).toHaveAttribute('aria-current', 'page')
  })

  test('/monitor keeps its own tabs without a dispatch center tab or tab links', async ({
    page,
  }) => {
    await installScheduleRoutingRoutes(page, 'admin')
    await page.goto('/monitor?tab=inspector')
    await page.locator('.app-tabs__bar').waitFor()

    await expect(page.locator('.app-tabs__trigger')).toHaveCount(3)
    await expect(page.locator('.app-tabs__trigger', { hasText: 'Dispatch center' })).toHaveCount(0)
    await expect(page.getByRole('tab', { name: 'Health' })).toBeVisible()
    await expect(page.getByRole('tab', { name: 'Usage & cost' })).toBeVisible()
    await expect(page.getByRole('tab', { name: 'Route inspector' })).toBeVisible()
    await expect(page.locator('a[href*="tab=schedule"]')).toHaveCount(0)

    // 旧的 Tab 深链不再渲染调度内容，只按既有非法 Tab 规则回到 health。
    await page.goto('/monitor?tab=schedule')
    await expect.poll(() => new URL(page.url()).searchParams.get('tab')).toBe('health')
    await expect(page.locator('.schedule-panel')).toHaveCount(0)
  })

  test('access key principal has no dispatch entry and cannot reach /schedule', async ({
    page,
  }) => {
    await installScheduleRoutingRoutes(page, 'access_key')
    await page.goto('/monitor?tab=usage')
    await page.locator('.desktop-nav').waitFor()

    await expect(page.getByRole('link', { name: 'Dispatch center' })).toHaveCount(0)
    await expect(page.locator('.desktop-nav a[href="/schedule"]')).toHaveCount(0)

    await page.goto('/schedule')
    await expect(page).toHaveURL(/\/$/u)
    await expect(page.locator('.schedule-panel')).toHaveCount(0)
  })
})
