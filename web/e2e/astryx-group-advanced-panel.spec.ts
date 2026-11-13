import { expect, test } from '@playwright/test'

import { mockGroupModels } from './fixtures/group-models-mock'

test('empty advanced panel stays hidden while settings load', async ({ page }) => {
  await mockGroupModels(page)
  let releaseSettings!: () => void
  const settingsGate = new Promise<void>((resolve) => {
    releaseSettings = resolve
  })
  await page.route('**/api/groups/1/settings?**', async (route) => {
    await settingsGate
    await route.fallback()
  })

  try {
    await page.goto('/groups/1', { waitUntil: 'domcontentloaded' })
    await expect(page.getByRole('status', { name: 'Loading Group settings' })).toBeVisible()
    const target = page.locator('#group-settings-advanced-target')
    await expect(target).toBeEmpty()
    await expect(target.locator('..')).toBeHidden()
  } finally {
    releaseSettings()
  }
})

test('failed settings hide the empty advanced panel and retry restores the separate portal', async ({
  page,
}) => {
  await mockGroupModels(page)
  let failSettings = true
  await page.route('**/api/groups/1/settings?**', async (route) => {
    if (failSettings) {
      await route.fulfill({ status: 500, contentType: 'application/json', body: '{}' })
    } else {
      await route.fallback()
    }
  })

  await page.goto('/groups/1', { waitUntil: 'domcontentloaded' })
  const error = page.getByRole('alert').filter({ hasText: 'Unable to load Group settings' })
  await expect(error).toBeVisible()
  const target = page.locator('#group-settings-advanced-target')
  await expect(target).toBeEmpty()
  await expect(target.locator('..')).toBeHidden()

  failSettings = false
  await error.getByRole('button', { name: 'Retry' }).click()
  const advanced = target.getByText('Advanced configuration', { exact: true })
  await expect(advanced).toBeVisible()
  const panel = target.locator('..')
  await expect(panel).toBeVisible()
  await expect(panel).toHaveAttribute('data-content-panel', '')
  await expect(
    page.locator('[data-content-panel]').filter({ has: page.locator('#settings-general') }),
  ).toHaveCount(1)
  await expect(panel.locator('#settings-general')).toHaveCount(0)
  await advanced.click()
  await expect(target.locator('#settings-runtime')).toBeVisible()
  await advanced.click()
  await expect(target.locator('#settings-runtime')).toBeHidden()
})
