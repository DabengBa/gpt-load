import { expect, test } from '@playwright/test'

import { mockGroupModels } from './fixtures/group-models-mock'

test('channel loading failure preserves current value and can be retried', async ({ page }) => {
  await setup(page)
  let calls = 0
  await page.route('**/api/channels**', (route) => {
    calls += 1
    if (calls === 1) return route.fulfill({ status: 500, json: { code: 500, message: 'failed' } })
    return route.fallback()
  })
  await page.goto('/groups/1')
  const form = page.locator('#settings-general')
  const channel = form.getByRole('combobox', { name: 'Channel', exact: true })
  await expect(form.getByRole('alert')).toContainText('Unable to load channels')
  await expect(channel).toHaveValue('openai')
  await expect(channel).toBeDisabled()
  await form.getByRole('button', { name: 'Retry' }).click()
  await expect(channel).toBeEnabled()
  await expect(channel.locator('option:checked')).toHaveText('OpenAI')
  await expect(form.getByRole('alert')).toHaveCount(0)
})

async function setup(page: import('@playwright/test').Page) {
  await mockGroupModels(page)
  const runtime = {
    first_byte_timeout: 30,
    request_timeout: 60,
    stream_idle_timeout: 30,
    blacklist_threshold: 3,
    header_rules: { set: {}, remove: [] },
    affinity_enabled: false,
    responses_websocket_enabled: false,
    responses_reasoning_status_filter_enabled: false,
    parameter_overrides: [],
  }
  let settings = {
    name: 'Discovery Group',
    channel_id: 'openai',
    params: { base_url: 'https://upstream.example/v1' },
    provider_url: 'https://provider.example',
    enabled: true,
    site_access: { user_id: '', token_configured: false, site_root_url: '', snapshot: null },
    overrides: runtime,
    effective: {
      first_byte_timeout: 30,
      request_timeout: 60,
      stream_idle_timeout: 30,
      blacklist_threshold: 3,
      header_rules: { set: {}, remove: [] },
      affinity_enabled: false,
      responses_websocket_enabled: false,
      responses_reasoning_status_filter_enabled: false,
    },
    proxy: {
      configured_mode: 'inherit',
      effective_mode: 'direct',
      effective_source: 'default',
      has_auth: false,
    },
  }
  const requests: unknown[] = []
  let failNext = false
  let release: (() => void) | undefined
  let gate: Promise<void> | undefined
  await page.route('**/api/channels**', (route) =>
    route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data: {
          items: [
            {
              channel_id: 'openai',
              name: 'OpenAI',
              mark: 'OA',
              icon: 'openai',
              search_terms: [],
              description: '',
              default_base_url: 'https://api.openai.com',
              param_fields: [
                {
                  key: 'base_url',
                  label: 'URL',
                  input_kind: 'url',
                  required: true,
                  sensitive: false,
                  default_value: null,
                },
              ],
              credential_fields: [],
              connection: { credential_input: 'batch_text' },
              capabilities: { model_discovery: true, outbound_proxy: false },
              routes: [],
              client_protocols: [],
            },
            {
              channel_id: 'alternative',
              name: 'Alternative',
              mark: 'Alt',
              icon: 'openai',
              search_terms: [],
              description: '',
              default_base_url: 'https://alternative.example',
              param_fields: [
                {
                  key: 'base_url',
                  label: 'URL',
                  input_kind: 'url',
                  required: true,
                  sensitive: false,
                  default_value: null,
                },
                {
                  key: 'tenant',
                  label: 'Tenant',
                  input_kind: 'text',
                  required: false,
                  sensitive: false,
                  default_value: null,
                },
              ],
              credential_fields: [],
              connection: { credential_input: 'batch_text' },
              capabilities: { model_discovery: true, outbound_proxy: false },
              routes: [],
              client_protocols: [],
            },
          ],
          total: 2,
        },
      },
    }),
  )
  await page.route('**/api/groups/1/settings**', async (route) => {
    if (route.request().method() === 'PUT') {
      const body = route.request().postDataJSON()
      requests.push(body)
      if (gate) await gate
      if (failNext) {
        failNext = false
        return route.fulfill({ status: 500, json: { code: 500, message: 'failed' } })
      }
      settings = { ...settings, ...body }
    }
    return route.fulfill({ json: { code: 0, message: 'ok', data: settings } })
  })
  return {
    requests,
    fail: () => {
      failNext = true
    },
    hold: () => {
      gate = new Promise<void>((resolve) => {
        release = resolve
      })
    },
    release: () => {
      release?.()
      gate = undefined
    },
  }
}

for (const width of [390, 1440]) {
  test(`basic fields, provider link and switch at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 900 })
    const api = await setup(page)
    await page.goto('/groups/1')
    const form = page.locator('#settings-general')
    const channel = form.getByRole('combobox', { name: 'Channel', exact: true })
    const name = form.getByRole('textbox', { name: 'Name', exact: true })
    const upstream = form.getByRole('textbox', { name: 'Upstream URL' })
    const provider = form.getByRole('textbox', { name: 'Provider website' })
    const userId = form.getByRole('textbox', { name: 'User ID' })
    const token = form.getByLabel('Access token')
    await expect(provider).toBeVisible({ timeout: 60_000 })
    await expect(channel).toHaveValue('openai')
    await page.context().grantPermissions(['clipboard-read', 'clipboard-write'])
    await upstream.fill('https://draft.example/v1')
    await upstream.selectText()
    await upstream.press('Control+C')
    await expect
      .poll(() => page.evaluate(() => navigator.clipboard.readText()))
      .toBe('https://draft.example/v1')
    await upstream.fill('https://upstream.example/v1')
    await expect(userId).toBeVisible()
    await expect(token).toHaveAttribute('type', 'password')
    await expect(page.getByRole('heading', { name: 'Site access' })).toHaveCount(0)
    const link = form.getByRole('link', {
      name: 'Open provider website https://provider.example in a new tab',
    })
    await expect(link).toHaveAttribute('href', 'https://provider.example')
    await expect(link).toHaveAttribute('target', '_blank')
    await expect(link).toHaveAttribute('rel', 'noopener noreferrer')
    await expect(link).toHaveAttribute('title', /in a new tab/)
    await expect(provider).toHaveAccessibleDescription(/does not change where requests are sent/)
    await expect(
      page
        .locator('header')
        .filter({ has: page.locator('#group-detail-title') })
        .getByRole('link', { name: /Open provider website/ }),
    ).toHaveCount(0)
    const [a, b, c, d, e, f, g] = await Promise.all([
      name.boundingBox(),
      upstream.boundingBox(),
      provider.boundingBox(),
      link.boundingBox(),
      channel.boundingBox(),
      userId.boundingBox(),
      token.boundingBox(),
    ])
    if (width > 800) {
      expect(Math.abs(e!.y + e!.height / 2 - (a!.y + a!.height / 2))).toBeLessThan(3)
      expect(Math.abs(a!.y - b!.y)).toBeLessThan(3)
      expect(e!.x).toBeLessThan(a!.x)
      expect(a!.x).toBeLessThan(b!.x)
      expect(c!.y).toBeGreaterThan(b!.y)
      expect(Math.abs(c!.y - f!.y)).toBeLessThan(3)
      expect(Math.abs(f!.y - g!.y)).toBeLessThan(3)
      expect(c!.x).toBeLessThan(f!.x)
      expect(f!.x).toBeLessThan(g!.x)
    } else {
      expect(e!.y).toBeLessThan(a!.y)
      expect(a!.y).toBeLessThan(b!.y)
      expect(b!.y).toBeLessThan(c!.y)
      expect(c!.y).toBeLessThan(f!.y)
      expect(f!.y).toBeLessThan(g!.y)
    }
    expect(Math.abs(d!.y - c!.y)).toBeLessThan(c!.height + 5)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    )
    await page.screenshot({ path: testInfo.outputPath(`basic-clean-${width}.png`), fullPage: true })
    await channel.focus()
    await expect(channel).toBeFocused()
    await channel.selectOption('alternative')
    await expect(channel).toHaveValue('alternative')
    await expect(upstream).toHaveValue('https://upstream.example/v1')
    await expect(form.getByRole('textbox', { name: 'Tenant' })).toBeVisible()
    await channel.selectOption('openai')
    await expect(form.getByRole('textbox', { name: 'Tenant' })).toHaveCount(0)
    await provider.fill('javascript:alert(1)')
    await expect(link).toHaveAttribute('href', 'https://provider.example')
    await provider.fill('https://provider.example')
    const toggle = page.getByRole('switch', { name: 'Group enabled' })
    await expect(toggle).toBeChecked()
    await name.fill('Unsaved draft')
    await toggle.click()
    await expect.poll(() => api.requests.length).toBe(1)
    expect(api.requests[0]).toEqual({ enabled: false })
    await expect(toggle).not.toBeChecked()
    await expect(name).toHaveValue('Unsaved draft')
    api.fail()
    api.hold()
    await toggle.click()
    await expect.poll(() => api.requests.length).toBe(2)
    expect(api.requests[1]).toEqual({ enabled: true })
    await expect(toggle).toBeDisabled()
    await expect(toggle).not.toBeChecked()
    api.release()
    await expect(toggle).toBeEnabled()
    await expect(name).toHaveValue('Unsaved draft')
    await expect(page.getByText('Unable to update the Group settings.').first()).toBeVisible()
    await page.screenshot({ path: testInfo.outputPath(`basic-${width}.png`), fullPage: true })
  })
}
