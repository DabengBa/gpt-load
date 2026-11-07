import { expect, test } from '@playwright/test'

test('shows independent source states alongside preserved announcements', async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem('gpt-load.auth-key', 'e2e-announcements')
    localStorage.setItem('gpt-load.locale', 'en-US')
  })
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url())
    const data =
      url.pathname === '/api/announcements'
        ? {
            sites: [],
            items: [
              {
                id: 'a',
                site_root_url: 'https://alpha.example',
                site_name: 'Alpha',
                content: 'Old announcement',
                extra: '',
                level: '',
                source: 'announcement',
                event_at_ms: 1700000000000,
                fetched_at_ms: 1700000000000,
                groups: [],
              },
            ],
            sources: [
              {
                group: { id: 1, name: 'Group A' },
                site_root_url: 'https://alpha.example',
                site_name: 'Alpha',
                announcement: { status: 'error', fetched_at_ms: 1700000000000 },
                notice: { status: 'pending', fetched_at_ms: null },
              },
            ],
          }
        : url.pathname === '/api/auth/session'
          ? { authenticated: true, principal_type: 'admin' }
          : {}
    await route.fulfill({ json: { code: 0, message: 'ok', data } })
  })
  await page.goto('/announcements')
  await expect(page.getByText('Old announcement')).toBeVisible()
  await expect(page.getByText('Announcement: Sync failed')).toBeVisible()
  await expect(page.getByText('Notice: Never synced')).toBeVisible()
})
