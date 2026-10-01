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
      price_multiplier: '1',
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
        credential_count: 0,
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
    if (path === '/api/groups/1/credentials') data = { items: [], total: 0 }
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'ok', data }),
    })
  })
}

async function expectUnclipped(locator: Locator) {
  const clipped = await locator.evaluate((root) =>
    Array.from(root.querySelectorAll('*')).some((el) => {
      const style = getComputedStyle(el)
      return style.textOverflow === 'ellipsis' && el.scrollWidth > el.clientWidth + 1
    }),
  )
  expect(clipped, 'visible control text must not be ellipsized').toBe(false)
}

for (const width of [375, 1440]) {
  for (const theme of ['light', 'dark']) {
    test(`import field geometry ${width} ${theme}`, async ({ page }, testInfo) => {
      await page.setViewportSize({ width, height: 900 })
      await installFields(page, theme)
      await page.goto('/import', { waitUntil: 'commit' })
      const multiplier = page.getByRole('textbox', { name: '价格倍率', exact: true })
      await expect(multiplier).toBeVisible({ timeout: 60_000 })
      const customURL = page.getByRole('switch').first()
      if (!(await customURL.isChecked())) await customURL.click()
      await page.screenshot({ path: testInfo.outputPath('import.png'), fullPage: true })
      for (const name of ['价格倍率', '自定义上游地址', '供应商官网']) {
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
      const from = page.getByRole('combobox', { name: '开始时间', exact: true })
      await expect(from).toBeVisible({ timeout: 60_000 })
      await page.screenshot({ path: testInfo.outputPath('logs.png'), fullPage: true })
      expect((await from.boundingBox())?.width).toBeGreaterThanOrEqual(180)
      const shortcuts = page.getByRole('group', { name: '快捷时间范围' })
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
