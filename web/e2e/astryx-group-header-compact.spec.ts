import { expect, test } from '@playwright/test'

import { mockGroupModels } from './fixtures/group-models-mock'

for (const width of [390, 1440]) {
  test(`group header has no upstream copy action at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 })
    await mockGroupModels(page)
    await page.route('**/api/groups/1', async (route) => {
      await route.fulfill({
        json: {
          code: 0,
          message: 'ok',
          data: {
            id: 1,
            name: 'Discovery Group',
            channel_id: 'openai',
            params: { base_url: 'https://upstream.example/v1' },
            provider_url: 'https://provider.example',
            service_status: 'available',
            service_status_reason: null,
            credential_configured: true,
            credential_status: 'available',
            model_count: 0,
          },
        },
      })
    })
    await page.goto('/groups/1')
    const header = page.locator('header').filter({ has: page.locator('#group-detail-title') })
    await expect(header).toBeVisible()
    await expect(header.getByRole('heading', { name: 'Discovery Group' })).toBeVisible()
    await expect(header.getByRole('link', { name: /Back to groups/i })).toBeVisible()
    await expect(header.getByText('OpenAI', { exact: true })).toHaveCount(0)
    await expect(header.getByText('https://upstream.example/v1', { exact: true })).toHaveCount(0)
    await expect(header.getByRole('link', { name: /Open provider website/ })).toHaveCount(0)
    await expect(header.getByRole('button', { name: /copy upstream address/i })).toHaveCount(0)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    )
    await page.screenshot({ path: test.info().outputPath(`compact-header-${width}.png`) })
  })
}
