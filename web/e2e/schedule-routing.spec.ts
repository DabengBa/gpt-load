import { expect, test, type Page } from '@playwright/test'

import { alphaGroupId, installScheduleRowTargetingRoutes } from './fixtures/schedule-row-targeting'

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

test.describe('group model row targeting', () => {
  test('the group models schedule link targets the source group entry row', async ({ page }) => {
    await installScheduleRowTargetingRoutes(page)
    await page.goto(`/groups/${alphaGroupId}`)

    const links = page.locator('.group-models__schedule-link')
    await expect(links).toHaveCount(2)
    // 未启用别名的行按模型 ID、启用别名的行按对外别名生成调度目标，并携带来源分组与 entry 定位。
    await expect(links.nth(0)).toHaveAttribute(
      'href',
      /schedule_model=model-a&schedule_group=1&schedule_row=1:entry-1/u,
    )
    await expect(links.nth(1)).toHaveAttribute(
      'href',
      /schedule_model=worker-b&schedule_group=1&schedule_row=1:entry-2/u,
    )

    await links.nth(1).click()
    await expect(page).toHaveURL(
      /\/schedule\?schedule_model=worker-b&schedule_group=1&schedule_row=1:entry-2$/u,
    )
    const selected = page.locator('.schedule-row--selected')
    await expect(selected).toHaveCount(1)
    await expect(selected).toContainText('worker-b')
    await expect(selected).toContainText('alpha group')

    // 前进/后退保留 URL 定位状态。
    await page.goBack()
    await expect(page).toHaveURL(new RegExp(`/groups/${alphaGroupId}`))
    await page.goForward()
    await expect(page).toHaveURL(/schedule_row=1:entry-2/u)
    await expect(page.locator('.schedule-row--selected')).toContainText('worker-b')
  })

  test('schedule_group resolves the first matching entry row after the detail loads', async ({
    page,
  }) => {
    await installScheduleRowTargetingRoutes(page)
    await page.goto('/schedule?schedule_model=worker&schedule_group=2')

    const selected = page.locator('.schedule-row--selected')
    await expect(selected).toHaveCount(1)
    await expect(selected).toContainText('worker')
    await expect(page).toHaveURL(/schedule_row=2:entry-2/u)

    // 刷新后 URL 状态继续恢复定位。
    await page.reload()
    await expect(selected).toHaveCount(1)
    await expect(selected).toContainText('worker')
  })

  test('an exact schedule_row scrolls the target row into the viewport', async ({ page }) => {
    await installScheduleRowTargetingRoutes(page)
    await page.goto('/schedule?schedule_model=worker&schedule_group=2&schedule_row=2%3Aentry-12')

    const selected = page.locator('.schedule-row--selected')
    await expect(selected).toHaveCount(1)
    await expect(selected).toContainText('model-l')
    const box = await selected.boundingBox()
    expect(box).not.toBeNull()
    expect(box!.y).toBeGreaterThanOrEqual(0)
    expect(box!.y + box!.height).toBeLessThanOrEqual(721)
  })

  test('a missing target row keeps the model context and clears the selection silently', async ({
    page,
  }) => {
    await installScheduleRowTargetingRoutes(page)
    await page.goto('/schedule?schedule_model=worker&schedule_group=9&schedule_row=9%3Aentry-99')

    await expect(page.locator('.schedule-row:not(.schedule-row--header)')).toHaveCount(10)
    await expect(page.locator('.schedule-row--selected')).toHaveCount(0)
    const model = page.locator('.schedule-panel .app-select__trigger[aria-label="External model"]')
    await expect(model).toContainText('worker')
    const query = new URL(page.url()).searchParams
    expect(query.get('schedule_group')).toBe('9')
    expect(query.get('schedule_row')).toBeNull()
  })
})
