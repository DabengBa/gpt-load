import { expect, test } from '@playwright/test'

const alpha = 'https://alpha.example'
const beta = 'https://beta.example'
const empty = 'https://empty.example'
const sites = [
  { site_root_url: alpha, site_name: 'Alpha', count: 1 },
  { site_root_url: beta, site_name: 'Beta', count: 1 },
]
const items = [
  {
    id: 'a',
    site_root_url: alpha,
    site_name: 'Alpha',
    content: 'Alpha maintenance',
    extra: '',
    level: '',
    source: 'notice',
    event_at_ms: 1700000000000,
    fetched_at_ms: 1700000000000,
    groups: [],
  },
  {
    id: 'b',
    site_root_url: beta,
    site_name: 'Beta',
    content: 'Beta release',
    extra: '',
    level: '',
    source: 'notice',
    event_at_ms: 1700000000000,
    fetched_at_ms: 1700000000000,
    groups: [],
  },
]

test('deep-linked empty site remains selected and matches the URL', async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem('gpt-load.auth-key', 'e2e-announcements')
    localStorage.setItem('gpt-load.locale', 'en-US')
  })
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url())
    const data =
      url.pathname === '/api/announcements'
        ? {
            sites,
            items: items.filter(
              (item) =>
                !url.searchParams.get('site') ||
                item.site_root_url === url.searchParams.get('site'),
            ),
          }
        : url.pathname === '/api/auth/session'
          ? { authenticated: true, principal_type: 'admin' }
          : {}
    await route.fulfill({ json: { code: 0, message: 'ok', data } })
  })
  await page.goto(`/announcements?site=${encodeURIComponent(empty)}`)
  await expect(page.getByText('No announcements')).toBeVisible()
  await expect(page.getByRole('combobox', { name: 'Site' })).toContainText(empty)
  await expect(page).toHaveURL(new RegExp(`site=${encodeURIComponent(empty)}`))
  await page.getByRole('combobox', { name: 'Site' }).click()
  await page.getByRole('option', { name: 'All sites' }).click()
  await expect(page).toHaveURL(/\/announcements$/)
  await expect(page.getByText('Alpha maintenance')).toBeVisible()
  await page.evaluate((value) => {
    history.pushState(history.state, '', `?site=${encodeURIComponent(value)}`)
    dispatchEvent(new PopStateEvent('popstate'))
  }, empty)
  await expect(page.getByRole('combobox', { name: 'Site' })).toContainText(empty)
  await expect(page.getByText('No announcements')).toBeVisible()
})

test('site choices stay available while filtering and search results match the request', async ({
  page,
}) => {
  await page.addInitScript(() => {
    localStorage.setItem('gpt-load.auth-key', 'e2e-announcements')
    localStorage.setItem('gpt-load.locale', 'en-US')
  })
  let release: (() => void) | undefined
  let blocked = false
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url())
    if (url.pathname === '/api/announcements') {
      if (url.searchParams.get('site') === alpha && !blocked) {
        blocked = true
        await new Promise<void>((resolve) => {
          release = resolve
        })
      }
      const site = url.searchParams.get('site')
      const q = (url.searchParams.get('q') ?? '').toLowerCase()
      return route.fulfill({
        json: {
          code: 0,
          message: 'ok',
          data: {
            sites,
            items: items.filter(
              (item) =>
                (!site || item.site_root_url === site) && item.content.toLowerCase().includes(q),
            ),
          },
        },
      })
    }
    return route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data:
          url.pathname === '/api/auth/session'
            ? { authenticated: true, principal_type: 'admin' }
            : {},
      },
    })
  })
  await page.goto('/announcements')
  await expect(page.getByText('Beta release')).toBeVisible()
  const selector = page.getByRole('combobox', { name: 'Site' })
  const search = page.getByRole('textbox', { name: 'Search' })
  await expect(search).toBeVisible()
  for (const name of ['Site', 'Search']) {
    const labels = page.locator('label').filter({ hasText: new RegExp(`^${name}$`) })
    for (const label of await labels.all()) {
      expect((await label.boundingBox())?.height ?? 0).toBeLessThanOrEqual(1)
    }
  }
  for (const width of [390, 1440]) {
    await page.setViewportSize({ width, height: 900 })
    await expect(selector).toBeVisible()
    await expect(search).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    )
  }
  await selector.click()
  await page.getByRole('option', { name: 'Alpha' }).click()
  await expect.poll(() => blocked).toBe(true)
  await expect(selector).toContainText('Alpha')
  await selector.click()
  await expect(page.getByRole('option', { name: 'Beta' })).toBeVisible()
  await page.keyboard.press('Escape')
  release?.()
  await expect(page.getByText('Alpha maintenance')).toBeVisible()
  await expect(page.getByText('Beta release')).toHaveCount(0)
  await expect(page).toHaveURL(new RegExp(`site=${encodeURIComponent(alpha)}`))
  await page.getByRole('textbox', { name: 'Search' }).fill('release')
  await expect(page.getByText('No announcements')).toBeVisible()
  await page.getByRole('textbox', { name: 'Search' }).fill('maintenance')
  await expect(page.getByText('Alpha maintenance')).toBeVisible()
})
