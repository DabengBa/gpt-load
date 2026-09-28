import { expect, test } from '@playwright/test'

import { installRequestLogDisplayRoutes, openRequestLogs } from './fixtures/request-log-display.ts'

test.use({ timezoneId: 'Asia/Shanghai' })

test('tokens distinguish recorded zero, cache hits, partial usage and unavailable rates', async ({
  page,
}) => {
  await installRequestLogDisplayRoutes(page, (items) => [
    { ...items[0], input_tokens: '1000', cache_read_tokens: '980' },
    items[1]!,
    { ...items[2], usage_state: 'partial' },
    {
      ...items[3],
      usage_state: 'missing',
      cost_state: 'unpriced',
      pricing_completeness: 'unavailable',
      estimated_cost_nano_usd: '0',
    },
    {
      ...items[0],
      request_id: 'eeeeeeee-5555-4555-8555-555555555555',
      cache_write_5m_tokens: '50',
    },
    { ...items[0], request_id: 'ffffffff-6666-4666-8666-666666666666', input_tokens: '0' },
    {
      ...items[0],
      request_id: 'aaaaaaaa-7777-4777-8777-777777777777',
      usage_state: 'not_applicable',
      cost_state: 'not_applicable',
      pricing_completeness: 'not_applicable',
      estimated_cost_nano_usd: '0',
    },
  ])
  await openRequestLogs(page)
  const rows = page.locator('.logs-list__record')
  const hit = rows.nth(0).locator('.logs-list__cache-rate')
  await expect(hit).toContainText('Cache hit rate 98.0%')
  await expect(rows.nth(0).locator('.logs-list__token-line')).toHaveText(
    /1,000\s*Input\s*·\s*42\s*Output/u,
  )
  const tokenBox = (await rows.nth(0).locator('.logs-list__token-line').boundingBox())!
  expect((await hit.boundingBox())!.y).toBeGreaterThanOrEqual(tokenBox.y + tokenBox.height)
  await expect(rows.nth(1).locator('.logs-list__cache-rate')).toContainText('0.0%')
  await expect(rows.nth(2).locator('.logs-list__cache-state')).toHaveText('Cache data unavailable')
  await expect(rows.nth(3).locator('.logs-list__cache-state')).toHaveText('Cache data unavailable')
  const writeOnly = rows.nth(4).locator('.logs-list__cache-rate')
  await expect(writeOnly).toContainText('0.0%')
  await writeOnly.focus()
  await expect(page.locator('.app-tooltip__content')).toContainText('Cache write 5m 50')
  await expect(rows.nth(5).locator('.logs-list__cache-rate')).toContainText('Cache hit rate —')
  await expect(rows.nth(6).locator('.logs-list__cache-state')).toHaveText('Not applicable')
})

test('every row retains its local date with a full timezone timestamp on focus', async ({
  page,
}) => {
  await installRequestLogDisplayRoutes(page, (items) =>
    items.map((item, index) => ({
      ...item,
      completed_at_ms: Date.parse(index < 2 ? '2026-09-28T16:00:01Z' : '2026-09-28T15:59:59Z'),
    })),
  )
  await openRequestLogs(page)
  const dates = page.locator('.logs-list__time small')
  await expect(dates).toHaveText(['2026-09-29', '2026-09-29', '2026-09-28', '2026-09-28'])
  const time = page.locator('.logs-list__time time').nth(1)
  await expect(time).toHaveAttribute('datetime', '2026-09-28T16:00:01.000Z')
  await expect(time).toHaveText('00:00:01')
  await time.focus()
  await expect(page.locator('.app-tooltip__content')).toContainText('2026-09-29 00:00:01')
  await expect(page.locator('.app-tooltip__content')).toContainText('Asia/Shanghai')
})

test('route and model metadata occupy separate readable lines', async ({ page }) => {
  await page.setViewportSize({ width: 1503, height: 900 })
  await installRequestLogDisplayRoutes(page)
  await openRequestLogs(page)

  const row = page.locator('.logs-list__record').first()
  const group = row.locator('.log-route-identity__group')
  const credential = row.locator('.log-route-identity__credential')
  const groupBox = (await group.boundingBox())!
  const credentialBox = (await credential.boundingBox())!
  expect(credentialBox.y).toBeGreaterThanOrEqual(groupBox.y + groupBox.height)

  const modelBox = (await row.locator('.logs-list__model').boundingBox())!
  const protocol = row.locator('.logs-list__protocol')
  await expect(protocol).toHaveText('Completions')
  const protocolBox = (await protocol.boundingBox())!
  expect(protocolBox.y).toBeGreaterThanOrEqual(modelBox.y + modelBox.height)
  await protocol.focus()
  await expect(page.locator('.app-tooltip__content')).toContainText('openai-completions')

  const affinityBox = (await row.locator('.logs-list__affinity-key-cell').boundingBox())!
  expect(groupBox.width).toBeGreaterThan(affinityBox.width)
  await group.click()
  await expect(page).toHaveURL(/group_id=1/u)
  await page.locator('.log-route-identity__credential').first().click()
  await expect(page).toHaveURL(/credential_id=3/u)
})

for (const theme of ['dark', 'light'] as const) {
  test(`long names, contrast and keyboard details in ${theme} theme`, async ({ page }) => {
    await page.setViewportSize({ width: 1503, height: 900 })
    await page.addInitScript((value) => localStorage.setItem('gpt-load.theme', value), theme)
    await installRequestLogDisplayRoutes(page, (items) =>
      items.map((item, index) => ({
        ...item,
        client_model:
          index === 0 ? 'lead-production-client-model-with-long-alias' : item.client_model,
        credential_name: 'sk-redacted-credential-with-a-long-display-name',
        protocol: 'openai-responses',
        route_mode: 'converted',
        reasoning: { mode: 'enabled', effort: 'low', budget_tokens: null },
        cache_read_tokens: '118',
        affinity_hit: true,
        affinity_source: 'prompt_cache_key',
        affinity_state: 'hit',
        affinity_key: '0123456789abcdef****fedcba9876543210',
      })),
    )
    await page.route('**/api/groups/options', (route) =>
      route.fulfill({
        json: {
          code: 0,
          message: 'OK',
          data: [
            {
              id: 1,
              name: 'aiapi-production-routing-group-with-a-long-name',
              channel_id: 'openai_compatible',
              connection_type: 'api_key',
              params: { base_url: 'https://example.com/v1' },
              provider_url: null,
              enabled: true,
              models: [],
            },
          ],
        },
      }),
    )
    await openRequestLogs(page)
    const row = page.locator('.logs-list__record').first()
    const list = page.locator('.ledger-record-list')
    expect(
      await list.evaluate((element) => element.scrollWidth - element.clientWidth),
    ).toBeLessThanOrEqual(1)
    const contrast = await row.locator('.logs-list__time small').evaluate((element) => {
      const luminance = (color: string) => {
        const values = color
          .match(/\d+(?:\.\d+)?/gu)!
          .slice(0, 3)
          .map(Number)
          .map((value) => {
            const channel = value / 255
            return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4
          })
        return values[0]! * 0.2126 + values[1]! * 0.7152 + values[2]! * 0.0722
      }
      const foreground = luminance(getComputedStyle(element).color)
      let parent: HTMLElement | null = element as HTMLElement
      while (parent && getComputedStyle(parent).backgroundColor === 'rgba(0, 0, 0, 0)')
        parent = parent.parentElement
      if (!parent) throw new Error('Expected an opaque row ancestor')
      const background = luminance(getComputedStyle(parent).backgroundColor)
      return (Math.max(foreground, background) + 0.05) / (Math.min(foreground, background) + 0.05)
    })
    expect(contrast).toBeGreaterThanOrEqual(4.5)
    const details = row.getByRole('button', { name: 'View details' })
    await details.focus()
    await expect(page.locator('.app-tooltip__content')).toContainText('View details')
    const hint = row.locator('.logs-list__affinity')
    expect((await hint.boundingBox())!.width).toBeGreaterThanOrEqual(24)
    await hint.focus()
    await expect(page.locator('.app-tooltip__content')).toContainText('Affinity')
    await page.keyboard.press('Escape')
    await row.locator('.logs-list__model').focus()
    await expect(page.locator('.app-tooltip__content')).toContainText(
      'lead-production-client-model-with-long-alias',
    )
    await page.keyboard.press('Escape')
    await row.locator('.log-route-identity').focus()
    await expect(page.locator('.app-tooltip__content')).toContainText(
      'aiapi-production-routing-group-with-a-long-name',
    )
    await page.keyboard.press('Escape')
    await row.locator('.log-protocol-conversion').focus()
    await expect(page.locator('.app-tooltip__content')).toContainText('openai-responses')
    await expect(page.locator('.app-tooltip__content')).toContainText('openai-completions')
    await page.keyboard.press('Escape')
    await row.locator('.log-protocol-conversion').blur()
    await page.mouse.move(0, 0)
    await page.screenshot({
      path: `../.tmp/request-log-readability_20261002/units/U003/evidence/${theme}-desktop.png`,
      fullPage: true,
    })
    await details.focus()
    await page.keyboard.press('Enter')
    await expect(page).toHaveURL(/selected_request_id=/u)
    await expect(page.locator('.log-detail')).toBeVisible()
  })
}

test('mobile cards and intermediate widths keep all fields reachable', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await installRequestLogDisplayRoutes(page)
  await openRequestLogs(page)
  await expect(page.locator('.ledger-record-list__header')).toBeHidden()
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
  const row = page.locator('.logs-list__record').first()
  await expect(row.locator('.logs-list__protocol')).toBeVisible()
  await expect(row.locator('.logs-list__cache-rate')).toBeVisible()
  await expect(row.getByRole('button', { name: 'View details' })).toBeVisible()
  await page.screenshot({
    path: '../.tmp/request-log-readability_20261002/units/U003/evidence/mobile.png',
    fullPage: true,
  })
  await page.setViewportSize({ width: 1024, height: 900 })
  const list = page.locator('.ledger-record-list')
  await list.evaluate((element) => {
    element.scrollLeft = element.scrollWidth
  })
  await expect(row.getByRole('button', { name: 'View details' })).toBeInViewport()
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1024)
})

test('Chinese desktop labels keep cache and protocol metadata readable', async ({ page }) => {
  await page.setViewportSize({ width: 1503, height: 900 })
  await page.addInitScript(() => {
    localStorage.setItem('gpt-load.locale', 'zh-CN')
    localStorage.setItem('gpt-load.theme', 'dark')
  })
  await installRequestLogDisplayRoutes(page, (items) =>
    items.slice(0, 3).map((item, index) => ({
      ...item,
      input_tokens: '38417',
      output_tokens: '227',
      cache_read_tokens: index < 2 ? '37649' : '0',
      protocol: 'openai-responses',
      reasoning: { mode: 'enabled', effort: 'low', budget_tokens: null },
      affinity_key: '0123456789abcdef****fedcba9876543210',
    })),
  )
  await openRequestLogs(page)
  const row = page.locator('.logs-list__record').first()
  await expect(row.locator('.logs-list__token-line')).toHaveText(/38,417\s*输入\s*·\s*227\s*输出/u)
  await expect(row.locator('.logs-list__cache-rate')).toContainText('98.0%')
  await expect(row.locator('.logs-list__protocol')).toHaveText('Responses')
  expect(
    await row
      .locator('.logs-list__token-line')
      .evaluate((element) => element.scrollWidth - element.clientWidth),
  ).toBeLessThanOrEqual(1)
  await page.screenshot({
    path: '../.tmp/request-log-readability_20261002/units/U003/evidence/zh-dark-desktop.png',
    fullPage: true,
  })
})
