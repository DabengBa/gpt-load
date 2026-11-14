import { expect, test, type Page } from '@playwright/test'

test.setTimeout(90_000)

async function seedSession(page: Page, initialLocale = 'zh-CN') {
  await page.addInitScript((locale) => {
    localStorage.setItem('gpt-load.auth-key', 'e2e-auth-key')
    if (!localStorage.getItem('gpt-load.locale')) {
      localStorage.setItem('gpt-load.locale', locale)
    }
  }, initialLocale)
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    const data =
      path === '/api/auth/session'
        ? { authenticated: true, principal_type: 'admin' }
        : path === '/api/groups'
          ? []
          : path === '/api/settings'
            ? {
                values: {
                  route_strategy: 'native_first',
                  first_byte_timeout: 30,
                  request_timeout: 600,
                  stream_idle_timeout: 60,
                  retry_count: 2,
                  first_provider_attempts: 1,
                  blacklist_threshold: 3,
                  blacklist_release_seconds: 30,
                  header_rules: { set: {}, remove: [] },
                  response_header_rules: { set: {}, remove: [] },
                  cors: {
                    enabled: false,
                    allowed_origins: [],
                    allowed_methods: [],
                    allowed_headers: [],
                    exposed_headers: [],
                    allow_credentials: false,
                    max_age: 600,
                  },
                  affinity_enabled: true,
                  responses_websocket_enabled: false,
                  affinity_ttl: 300,
                  affinity_capacity: 100,
                  request_log_retention_days: 7,
                  models_dev_auto_sync_enabled: false,
                  proxy_config: {
                    configured_mode: 'inherit',
                    effective_mode: 'direct',
                    effective_source: 'default',
                    has_auth: false,
                  },
                },
                overrides: [],
                read_only: [],
              }
            : {}
    await route.fulfill({ json: { code: 0, message: 'ok', data } })
  })
}

test('rapid locale changes during navigation publish complete catalogs', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 900 })
  await seedSession(page, 'en-US')
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  let releaseCore = () => {}
  const coreGate = new Promise<void>((resolve) => {
    releaseCore = resolve
  })
  let releaseMonitor = () => {}
  const monitorGate = new Promise<void>((resolve) => {
    releaseMonitor = resolve
  })
  await page.route('**/shared/i18n/locales/zh-CN/core.ts*', async (route) => {
    await coreGate
    await route.continue()
  })
  try {
    await page.goto('/settings')
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('Settings')
    await page.getByRole('button', { name: 'Menu and preferences' }).click()
    await expect(page.locator('html')).toHaveAttribute('lang', 'en-US')
    await expect(page.getByRole('radio', { name: 'EN', exact: true })).toBeChecked()
    await expect(page.getByRole('radio', { name: '日本語', exact: true })).toHaveCount(0)
    await page.route('**/shared/i18n/locales/en-US/monitor.ts*', async (route) => {
      await monitorGate
      await route.continue()
    })
    const coreRequested = page.waitForRequest('**/shared/i18n/locales/zh-CN/core.ts*')
    await page.getByRole('radio', { name: '中文', exact: true }).locator('..').click()
    await coreRequested
    const monitorRequested = page.waitForRequest('**/shared/i18n/locales/en-US/monitor.ts*')
    await page.getByRole('dialog').getByRole('link', { name: 'Monitor', exact: true }).click()
    await monitorRequested
    // Navigation owns a new namespace while the pending locale owns the old page.
    releaseMonitor()
    await expect(page).toHaveURL(/\/monitor(?:\?|$)/)
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('Monitor')
    releaseCore()
    await expect(page.locator('html')).toHaveAttribute('lang', 'zh-CN')
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('监控')
    await expect(page.getByRole('dialog')).not.toBeVisible()
    expect(errors).toEqual([])
    expect(await page.evaluate(() => localStorage.getItem('gpt-load.locale'))).toBe('zh-CN')
  } finally {
    releaseCore()
    releaseMonitor()
  }
})

for (const width of [375, 768]) {
  test(`mobile navigation closes and restores trigger focus at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 })
    await seedSession(page)
    const intlErrors: string[] = []
    page.on('pageerror', (error) => intlErrors.push(error.message))
    page.on('console', (message) => {
      if (message.type() === 'error' && message.text().includes('MISSING_TRANSLATION')) {
        intlErrors.push(message.text())
      }
    })
    let releaseCatalog = () => {}
    const catalogGate = new Promise<void>((resolve) => {
      releaseCatalog = resolve
    })
    await page.route('**/shared/i18n/locales/en-US/group.ts*', async (route) => {
      await catalogGate
      await route.continue()
    })
    await page.goto('/settings')
    const trigger = page.getByRole('button', { name: '菜单与偏好设置' })
    await trigger.click()
    const dialog = page.getByRole('dialog')
    const catalogRequested = page.waitForRequest('**/shared/i18n/locales/en-US/group.ts*')
    await dialog.getByRole('link', { name: '分组', exact: true }).click()
    await catalogRequested
    releaseCatalog()
    await expect(page).toHaveURL(/\/groups$/)
    await expect(page.getByRole('heading', { level: 1, name: '分组', exact: true })).toBeVisible()
    await expect(dialog).not.toBeVisible()
    await expect(trigger).toBeFocused()
    expect(intlErrors).toEqual([])
    await expect(page).toHaveTitle('分组 · GPT-Load')
    await trigger.click()
    await expect(dialog.getByRole('link', { name: '分组', exact: true })).toHaveAttribute(
      'aria-current',
      'page',
    )
    await page.keyboard.press('Escape')
    await expect(dialog).not.toBeVisible()
    await expect(trigger).toBeFocused()
  })
}

for (const width of [1440, 375]) {
  test(`locale updates title, navigation and body without reload at ${width}px`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 900 })
    await seedSession(page)
    await page.goto('/settings')
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('设置')
    const documentToken = await page.evaluate(() => {
      const token = crypto.randomUUID()
      document.documentElement.dataset.testDocument = token
      return token
    })
    await page.getByRole('button', { name: '菜单与偏好设置' }).click()
    for (const locale of [
      { radio: 'EN', tag: 'en-US', title: 'Settings', body: 'Routing and scheduling' },
      { radio: '中文', tag: 'zh-CN', title: '设置', body: '路由与调度' },
      { radio: 'EN', tag: 'en-US', title: 'Settings', body: 'Routing and scheduling' },
    ]) {
      await page.getByRole('radio', { name: locale.radio, exact: true }).locator('..').click()
      await expect(page.locator('html')).toHaveAttribute('lang', locale.tag)
      await expect(page.getByRole('radio', { name: locale.radio, exact: true })).toBeChecked()
      await expect(page.getByRole('heading', { level: 1 })).toHaveText(locale.title)
      await expect(page).toHaveTitle(`${locale.title} · GPT-Load`)
      const nav = width > 860 ? page.getByTestId('desktop-nav') : page.getByRole('dialog')
      await expect(nav.getByRole('link', { name: locale.title, exact: true })).toBeVisible()
      await expect(page.getByRole('heading', { name: locale.body, exact: true })).toBeVisible()
      await expect(page.locator('html')).toHaveAttribute('data-test-document', documentToken)
      expect(await page.evaluate(() => localStorage.getItem('gpt-load.locale'))).toBe(locale.tag)
    }
    // A new document verifies persistence separately; no reload is used to update the UI above.
    await page.goto('/settings')
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('Settings')
    await expect(page.locator('html')).toHaveAttribute('lang', 'en-US')
  })
}
