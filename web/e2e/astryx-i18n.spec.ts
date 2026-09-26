import { expect, test, type Page } from '@playwright/test'

// B8: react-intl runtime on the Astryx entry. The `astryx` project seeds the
// frontend-preference cookie; each test pins gpt-load.locale and asserts the
// shell renders that locale's catalog with <html lang> in sync and no
// react-intl error escaping (dev onError throws, so a missing key would
// surface as a pageerror).

const locales = [
  { tag: 'zh-CN', settingsTitle: '设置' },
  { tag: 'en-US', settingsTitle: 'Settings' },
  { tag: 'ja-JP', settingsTitle: '設定' },
] as const

async function seedAuth(page: Page): Promise<void> {
  await page.addInitScript((authKey) => {
    window.localStorage.setItem('gpt-load.auth-key', authKey)
  }, 'e2e-auth-key')
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/auth/session') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          code: 0,
          message: 'ok',
          data: { authenticated: true, principal_type: 'admin' },
        }),
      })
      return
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'ok', data: {} }),
    })
  })
}

for (const { tag, settingsTitle } of locales) {
  test(`shell renders ${tag} messages with <html lang> in sync`, async ({ page }) => {
    const intlErrors: string[] = []
    page.on('pageerror', (error) => intlErrors.push(error.message))
    page.on('console', (message) => {
      if (message.type() === 'error' && /i18n|MISSING|FORMAT/i.test(message.text())) {
        intlErrors.push(message.text())
      }
    })

    await seedAuth(page)
    await page.addInitScript((locale) => {
      window.localStorage.setItem('gpt-load.locale', locale)
    }, tag)

    await page.goto('/settings', { waitUntil: 'load' })
    const shell = page.locator('[data-testid="astryx-shell"]')
    await expect(shell).toBeVisible()
    await expect(page.locator('h1')).toHaveText(settingsTitle)
    await expect.poll(() => page.evaluate(() => document.title)).toBe(
      `${settingsTitle} · GPT-Load`,
    )
    expect(await page.evaluate(() => document.documentElement.lang)).toBe(tag)
    expect(intlErrors).toEqual([])
  })
}

test('switching locale re-renders shell text and updates <html lang>', async ({
  page,
}) => {
  await seedAuth(page)
  await page.goto('/settings', { waitUntil: 'load' })
  await expect(page.locator('h1')).toHaveText('Settings')

  // The locale controller is module-level on the entry; exercise the same
  // path the preferences control will use.
  await page.addInitScript(() => {
    window.localStorage.setItem('gpt-load.locale', 'zh-CN')
  })
  await page.reload({ waitUntil: 'load' })
  await expect(page.locator('h1')).toHaveText('设置')
  expect(await page.evaluate(() => document.documentElement.lang)).toBe('zh-CN')
})
