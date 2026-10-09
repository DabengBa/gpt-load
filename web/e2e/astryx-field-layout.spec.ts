import { expect, test, type Page, type Locator } from '@playwright/test'

import { installRequestLogTableRoutes } from './fixtures/request-log-display'

test.setTimeout(90_000)

const channel = {
  channel_id: 'openai',
  name: 'OpenAI',
  mark: 'OA',
  icon: 'openai',
  search_terms: ['openai'],
  description: '',
  default_base_url: 'https://api.openai.com',
  notices: [],
  param_fields: [
    {
      key: 'base_url',
      label: 'Base URL',
      required: false,
      input_kind: 'url',
      sensitive: false,
      default_value: null,
    },
  ],
  credential_fields: [],
  connection: { type: 'api_key', credential_input: 'batch_text', authorization_methods: [] },
  capabilities: {
    model_discovery: false,
    quota_observation: false,
    credential_actions: [],
    outbound_proxy: true,
  },
  routes: [
    {
      client_protocol: 'openai-completions',
      operation: 'chat_completion',
      route_mode: 'native',
      model_dependent: false,
      possible_modes: [],
    },
  ],
  client_protocols: ['openai-completions'],
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

async function installFields(page: Page, theme: string) {
  await page.addInitScript((theme) => {
    localStorage.setItem('gpt-load.auth-key', 'e2e-field-layout')
    localStorage.setItem('gpt-load.locale', 'zh-CN')
    localStorage.setItem('gpt-load.theme', theme)
  }, theme)
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    const base = {
      name: 'Geometry group',

      channel_id: 'openai',
      connection_type: 'api_key',
      params: {},
      provider_url: null,
    }
    let data: unknown = {}
    if (path === '/api/auth/session') data = { authenticated: true, principal_type: 'admin' }
    if (path === '/api/channels') data = { items: [channel], total: 1 }
    if (path === '/api/groups/options') data = []
    if (path === '/api/groups/1')
      data = {
        ...base,
        id: 1,
        service_status: 'available',
        service_status_reason: null,
        credential_configured: false,
        credential_status: null,
        model_count: 0,
      }
    if (path === '/api/groups/1/settings')
      data = {
        ...base,
        enabled: true,
        overrides: { ...runtime, parameter_overrides: [] },
        effective: runtime,
        proxy: {
          configured_mode: 'inherit',
          effective_mode: 'direct',
          effective_source: 'default',
          has_auth: false,
        },
      }
    if (path === '/api/groups/1/models') data = { items: [], total: 0, pending: 0 }
    if (path === '/api/groups/1/credential') data = { credential: null, observation: null }
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'ok', data }),
    })
  })
}

test('U004 group basic settings submit without multiplier', async ({ page }) => {
  await installFields(page, 'light')
  const puts: Record<string, unknown>[] = []
  page.on('request', (request) => {
    if (request.method() === 'PUT' && new URL(request.url()).pathname === '/api/groups/1/settings')
      puts.push(request.postDataJSON())
  })
  await page.goto('/groups/1', { waitUntil: 'commit' })
  await expect(page.getByRole('textbox', { name: '分组名称' })).toBeVisible({ timeout: 60_000 })
  await expect(page.getByRole('textbox', { name: '价格倍率' })).toHaveCount(0)
  await page.getByRole('textbox', { name: '分组名称' }).fill('Updated group')
  await page.getByRole('button', { name: '保存设置', exact: true }).click()
  await expect.poll(() => puts.length).toBe(1)
  expect(puts[0]).toMatchObject({ name: 'Updated group' })
  expect(puts[0]).not.toHaveProperty('price_multiplier')
})

async function expectUnclipped(locator: Locator) {
  const clipped = await locator.evaluate((root) =>
    Array.from(root.querySelectorAll('*')).some((el) => {
      const style = getComputedStyle(el)
      return style.textOverflow === 'ellipsis' && el.scrollWidth > el.clientWidth + 1
    }),
  )
  expect(clipped, 'visible control text must not be ellipsized').toBe(false)
}

test('successful group delete navigates before stalled options refetch and survives reload', async ({
  page,
}) => {
  await installFields(page, 'light')
  let deleted = false
  // Keep options active: the detail page itself does not observe this query.
  await page.route('**/src/frontends/astryx/app/services.ts', async (route) => {
    const response = await route.fetch()
    const source = await response.text()
    expect(source).toContain('authRef.current = authSession;')
    await route.fulfill({
      response,
      body:
        `import { QueryObserver as DeleteTestObserver } from '/node_modules/.vite/deps/@tanstack_react-query.js';\n` +
        `import { groupOptionsQueryOptions as deleteTestOptions } from '/src/shared/control/resources/groups.ts';\n` +
        source.replace(
          'authRef.current = authSession;',
          'authRef.current = authSession; new DeleteTestObserver(queryClient, deleteTestOptions(apiClient)).subscribe(() => {});',
        ),
    })
  })
  let optionsBlocked = false
  await page.route('**/api/groups?*', async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        code: 0,
        message: 'ok',
        data: {
          observed_at_ms: 1_700_000_000_000,
          summary: { total: 0, available: 0, unavailable: 0, disabled: 0 },
          items: [],
          pagination: { page: 1, page_size: 100, total_items: 0, total_pages: 0 },
        },
      }),
    })
  })
  const stalledOptions = Promise.withResolvers<void>()
  await page.route('**/api/groups/1', async (route) => {
    if (route.request().method() !== 'DELETE') return route.fallback()
    deleted = true
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'ok', data: null }),
    })
  })
  await page.route('**/api/groups/options', async (route) => {
    if (!deleted || route.request().method() !== 'GET') return route.fallback()
    optionsBlocked = true
    await stalledOptions.promise
    await route.fallback()
  })
  try {
    await page.goto('/groups/1', { waitUntil: 'commit' })
    await expect(page.locator('footer[data-status]')).toBeVisible({ timeout: 60_000 })
    await page.getByRole('button', { name: '删除分组', exact: true }).click()
    const dialog = page.getByRole('dialog')
    await dialog.getByRole('textbox').fill('Geometry group')
    await dialog.getByRole('button', { name: /删除/ }).click()
    await expect.poll(() => optionsBlocked).toBe(true)
    await expect(page).toHaveURL(/\/groups$/)
    stalledOptions.resolve()
    await page.reload()
    await expect(page).toHaveURL(/\/groups$/)
    await expect(page.locator('#groups-title')).toBeVisible()
    await expect(page.locator('#group-detail-title')).toHaveCount(0)
  } finally {
    stalledOptions.resolve()
  }
})

test('failed group delete keeps detail URL and confirmation dialog', async ({ page }) => {
  await installFields(page, 'light')
  await page.route('**/api/groups/1', async (route) => {
    if (route.request().method() !== 'DELETE') return route.fallback()
    await route.fulfill({
      status: 500,
      contentType: 'application/json',
      body: JSON.stringify({ code: 'INTERNAL_ERROR', message: 'delete failed', data: null }),
    })
  })
  await page.goto('/groups/1', { waitUntil: 'commit' })
  await expect(page.locator('footer[data-status]')).toBeVisible({ timeout: 60_000 })
  await page.getByRole('button', { name: '删除分组', exact: true }).click()
  const dialog = page.getByRole('dialog')
  await dialog.getByRole('textbox').fill('Geometry group')
  await dialog.getByRole('button', { name: /删除/ }).click()
  await expect(dialog.getByRole('alert')).toContainText('无法删除分组。')
  await expect(page).toHaveURL(/\/groups\/1$/)
  await expect(dialog.getByRole('textbox')).toHaveValue('Geometry group')
})

for (const width of [320, 768, 1440]) {
  for (const theme of ['light', 'dark']) {
    for (const path of ['/import', '/groups/1']) {
      test(`task surface ${path} ${width} ${theme}`, async ({ page }, testInfo) => {
        const errors: string[] = []
        page.on('pageerror', (error) => errors.push(error.message))
        await page.setViewportSize({ width, height: 900 })
        await installFields(page, theme)
        await page.goto(path, { waitUntil: 'load' })
        const title = page.locator(
          path === '/import' ? '#import-page-title' : '#group-detail-title',
        )
        await expect(title).toBeVisible()
        const name = page.getByRole('textbox', { name: /^分组名称/ })
        await expect(name).toBeVisible()
        const titleSize = await title.evaluate((el) => parseFloat(getComputedStyle(el).fontSize))
        expect(titleSize).toBeLessThanOrEqual(24)
        const box = await name.boundingBox()
        expect(box).not.toBeNull()
        expect(box!.x).toBeGreaterThanOrEqual(12)
        expect(box!.x + box!.width).toBeLessThanOrEqual(width - 12)
        if (width === 320) expect(box!.width).toBeGreaterThanOrEqual(270)
        expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
          width,
        )
        await name.focus()
        await expect(name).toBeFocused()
        await page.screenshot({ path: testInfo.outputPath('task-surface.png'), fullPage: true })
        expect(errors).toEqual([])
      })
    }
  }
}

for (const width of [375, 1440]) {
  for (const theme of ['light', 'dark']) {
    test(`import field geometry ${width} ${theme}`, async ({ page }, testInfo) => {
      await page.setViewportSize({ width, height: 900 })
      await installFields(page, theme)
      await page.goto('/import', { waitUntil: 'commit' })
      await expect(page.getByRole('textbox', { name: '价格倍率', exact: true })).toHaveCount(0)
      const customURL = page.getByRole('switch').first()
      if (!(await customURL.isChecked())) await customURL.click()
      await page.screenshot({ path: testInfo.outputPath('import.png'), fullPage: true })
      for (const name of ['自定义上游地址', '供应商官网']) {
        const input = page.getByRole('textbox', { name: new RegExp(`^${name}`) })
        await expect(input).toBeVisible()
        const geometry = await input.evaluate((input) => {
          const wrapper = input.closest('.astryx-text-input') ?? input
          return {
            height: wrapper.getBoundingClientRect().height,
            border: getComputedStyle(wrapper).borderTopWidth,
          }
        })
        expect(geometry.height, name).toBeGreaterThanOrEqual(30)
        expect(parseFloat(geometry.border), name).toBeGreaterThan(0)
      }
    })
    test(`save bar geometry ${width} ${theme}`, async ({ page }, testInfo) => {
      await page.setViewportSize({ width, height: 900 })
      await installFields(page, theme)
      await page.goto('/groups/1', { waitUntil: 'commit' })
      const bar = page.locator('footer[data-status]')
      await expect(bar).toBeVisible({ timeout: 60_000 })
      await page.screenshot({ path: testInfo.outputPath('group-sticky.png') })
      await testInfo.attach('save-bar-geometry', {
        body: JSON.stringify(await bar.boundingBox()),
        contentType: 'application/json',
      })
      if (width === 375) expect((await bar.boundingBox())?.height).toBeLessThanOrEqual(140)
      await expectUnclipped(bar)
      await page.getByText('高级配置', { exact: true }).click()
      await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight))
      const lastField = page.locator('main input:visible, main textarea:visible').last()
      const fieldBox = await lastField.boundingBox()
      const barBox = await bar.boundingBox()
      expect(fieldBox).not.toBeNull()
      expect(barBox).not.toBeNull()
      expect((fieldBox?.y ?? 0) + (fieldBox?.height ?? 0)).toBeLessThanOrEqual(barBox?.y ?? 0)
      await page.screenshot({ path: testInfo.outputPath('group-bottom.png') })
    })
    test(`log date and shortcut geometry ${width} ${theme}`, async ({ page }, testInfo) => {
      await page.setViewportSize({ width, height: 900 })
      await installRequestLogTableRoutes(page)
      await page.addInitScript((theme) => {
        localStorage.setItem('gpt-load.locale', 'zh-CN')
        localStorage.setItem('gpt-load.theme', theme)
      }, theme)
      await page.goto('/logs', { waitUntil: 'commit' })
      // Manual date inputs were removed by design; the labelled filter form
      // owns the remaining time surface, so scope geometry checks to it.
      const filters = page.getByRole('form', { name: '请求日志筛选' })
      const shortcuts = filters.getByRole('group', { name: '快捷时间范围' })
      await expect(shortcuts).toBeVisible({ timeout: 60_000 })
      await page.screenshot({ path: testInfo.outputPath('logs.png'), fullPage: true })
      await expectUnclipped(shortcuts)
      expect(await shortcuts.evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(
        1,
      )
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth - innerWidth),
      ).toBeLessThanOrEqual(1)
    })
  }
}
