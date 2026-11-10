import { expect, test, type Route } from '@playwright/test'

// The announcement feed is written by the background site sync, and the read
// endpoint carries no sync status. A page opened before the first sync finishes
// therefore only updates if the client polls the feed itself. These specs pin
// that behavior in a real browser: an empty page kept open must turn into a
// populated list, and a `?site=` deep link must initialize the filter.

test.setTimeout(120_000)

const site = 'https://site.example'

function announcement(id: string, siteRoot: string, content: string) {
  return {
    id: `${siteRoot}#${id}`,
    site_root_url: siteRoot,
    site_name: 'Demo site',
    content,
    extra: '',
    level: '',
    source: 'notice',
    event_at_ms: 1_700_000_000_000,
    fetched_at_ms: 1_700_000_000_000,
    groups: [{ id: 1, name: 'Group 0001' }],
  }
}

async function mockShell(page: import('@playwright/test').Page, feed: () => unknown) {
  await page.addInitScript(() => {
    window.localStorage.setItem('gpt-load.auth-key', 'e2e-announcements')
    window.localStorage.setItem('gpt-load.locale', 'en-US')
  })
  const requests: string[] = []
  await page.route('**/api/**', async (route: Route) => {
    const url = new URL(route.request().url())
    let data: unknown = {}
    if (url.pathname === '/api/auth/session') {
      data = { authenticated: true, principal_type: 'admin' }
    }
    if (url.pathname === '/api/announcements') {
      requests.push(url.search)
      data = feed()
    }
    await route.fulfill({ json: { code: 0, message: 'ok', data } })
  })
  return requests
}

test('a kept-open announcement page picks up the first background sync', async ({ page }) => {
  let reads = 0
  const requests = await mockShell(page, () => {
    reads++
    // The backend finishes its first sync only after the page is already open.
    return reads === 1
      ? { items: [], sites: [] }
      : {
          items: [announcement('notice-1', site, 'First synced announcement')],
          sites: [{ site_root_url: site, site_name: 'Demo site', count: 1 }],
        }
  })

  await page.goto('/announcements', { waitUntil: 'commit' })
  await expect(page.locator('#announcements-title')).toBeVisible({ timeout: 60_000 })
  await expect(page.getByText('No announcements')).toBeVisible()

  // No reload, no focus event: the open page refetches on its own.
  await expect(page.getByText('First synced announcement')).toBeVisible({ timeout: 60_000 })
  expect(reads).toBeGreaterThan(1)
  expect(requests.every((search) => search === '?include=source_status')).toBe(true)
})

test('a site deep link initializes the announcement filter from the URL', async ({ page }) => {
  const requests = await mockShell(page, () => ({
    items: [
      announcement('notice-1', site, 'Filtered announcement'),
      announcement('notice-2', 'https://other.example', 'Other site announcement'),
    ],
    sites: [
      { site_root_url: site, site_name: 'Demo site', count: 1 },
      { site_root_url: 'https://other.example', site_name: 'Other site', count: 1 },
    ],
  }))

  await page.goto(`/announcements?site=${encodeURIComponent(site)}`, { waitUntil: 'commit' })
  await expect(page.locator('#announcements-title')).toBeVisible({ timeout: 60_000 })
  await expect(page.getByText('Filtered announcement')).toBeVisible()
  expect(requests.at(0)).toBe(`?include=source_status&site=${encodeURIComponent(site)}`)

  // The site selector reflects the deep-linked value rather than "All sites".
  await expect(page.getByRole('combobox', { name: 'Site' })).toContainText('Demo site')
})
