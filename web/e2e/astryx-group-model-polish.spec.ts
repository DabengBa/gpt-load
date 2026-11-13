import { expect, test } from '@playwright/test'

import { mockGroupFingerprint } from './fixtures/group-fingerprint-mock'
import { mockGroupModels } from './fixtures/group-models-mock'

for (const width of [1440, 900, 375]) {
  test(`model controls retain alignment and keyboard semantics at ${width}px`, async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width, height: 900 })
    await mockGroupFingerprint(page)
    await page.goto('/groups/1?tab=models')
    const table = page.getByRole('table', { name: 'Current Group models' })
    const row = table.getByTestId('model-alias-editor__record').first()
    const alias = row.getByLabel('Client alias for gpt-5.4')
    const weight = row.getByLabel('Weight', { exact: true })
    const priority = row.getByLabel('Priority', { exact: true })
    const enabled = row.getByRole('switch', { name: 'Enable model gpt-5.4' })
    await expect(enabled).toBeChecked()
    await enabled.focus()
    await page.keyboard.press('Space')
    await expect(enabled).not.toBeChecked()
    await page.keyboard.press('Space')
    await expect(enabled).toBeChecked()
    await alias.focus()
    await expect(alias).toBeFocused()
    const focus = await alias.evaluate((el) => ({
      outline: getComputedStyle(el).outlineWidth,
      border: getComputedStyle(el).borderColor,
    }))
    expect(parseFloat(focus.outline)).toBeGreaterThanOrEqual(2)
    await page.keyboard.press('Tab')
    await expect(weight).toBeFocused()
    const sizes = await Promise.all([weight, priority].map((input) => input.boundingBox()))
    expect(Math.abs(sizes[0]!.width - sizes[1]!.width)).toBeLessThan(1)
    expect(sizes[0]!.width).toBeLessThanOrEqual(80)
    if (width > 860) {
      const header = table.getByRole('columnheader').nth(1)
      const [h, a] = await Promise.all([header.boundingBox(), alias.boundingBox()])
      expect(Math.abs(h!.x - a!.x)).toBeLessThan(1)
    }
    if (process.env.MODEL_POLISH_SCREENSHOTS) {
      await page.screenshot({ path: `/tmp/gpt-load-model-polish-${width}.png`, fullPage: true })
    }
    await weight.fill('-1')
    await expect(weight).toHaveAttribute('aria-invalid', 'true')
    const invalidBorder = await weight.evaluate((el) => getComputedStyle(el).borderColor)
    expect(invalidBorder).not.toBe('rgba(0, 0, 0, 0)')
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    )
    await page.screenshot({
      path: testInfo.outputPath(`model-polish-${width}.png`),
      fullPage: true,
    })
  })
}

test('new test alias is a pill and pending save disables row controls', async ({ page }) => {
  await mockGroupModels(page, { models: [] })
  let release!: () => void
  const waitForRelease = new Promise<void>((resolve) => {
    release = resolve
  })
  await page.route(/\/api\/groups\/1\/models(?:\?.*)?$/, async (route) => {
    if (route.request().method() === 'PUT') await waitForRelease
    await route.fallback()
  })
  await page.goto('/groups/1?tab=models')
  await page.getByRole('button', { name: 'Add a model', exact: true }).click()
  const row = page.getByTestId('model-alias-editor__record').first()
  const pending = row.getByTestId('group-models__test-alias').getByText('Assigned after save')
  await expect(pending).toBeVisible()
  expect(
    parseFloat(await pending.evaluate((el) => getComputedStyle(el).borderRadius)),
  ).toBeGreaterThan(0)
  await page.getByLabel('Model ID', { exact: true }).fill('new-model')
  await page.getByRole('button', { name: 'Save settings', exact: true }).click()
  try {
    await expect(row.getByRole('switch', { name: 'Enable model new-model' })).toBeDisabled()
    await expect(row.getByLabel('Client alias for new-model')).toBeDisabled()
  } finally {
    release()
  }
  await expect(row.getByRole('switch', { name: 'Enable model new-model' })).toBeEnabled()
  await expect(row.getByTestId('group-models__test-alias')).toHaveText('tst000')
})

test('multiple rows align, hover reveals controls, and dark mode preserves focus', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await mockGroupModels(page, {
    models: [{ id: 'model-a' }, { id: 'model-b', alias: 'long-alias', alias_enabled: true }],
  })
  await page.goto('/groups/1?tab=models')
  const rows = page.getByTestId('model-alias-editor__record')
  await expect(rows).toHaveCount(2)
  for (const label of ['Weight', 'Priority']) {
    const inputs = rows.getByRole('textbox', { name: label, exact: true })
    const boxes = await inputs.evaluateAll((elements) =>
      elements.map((el) => ({
        x: el.getBoundingClientRect().x,
        width: el.getBoundingClientRect().width,
      })),
    )
    expect(Math.abs(boxes[0].x - boxes[1].x)).toBeLessThan(1)
    expect(Math.abs(boxes[0].width - boxes[1].width)).toBeLessThan(1)
  }
  const aliases = rows.locator('[data-alias-input-index]')
  const aliasBoxes = await aliases.evaluateAll((elements) =>
    elements.map((el) => ({
      x: el.getBoundingClientRect().x,
      width: el.getBoundingClientRect().width,
    })),
  )
  expect(Math.abs(aliasBoxes[0].x - aliasBoxes[1].x)).toBeLessThan(1)
  expect(Math.abs(aliasBoxes[0].width - aliasBoxes[1].width)).toBeLessThan(1)
  const alias = aliases.first()
  await page.mouse.move(0, 0)
  await expect(alias).toHaveCSS('border-top-color', 'rgba(0, 0, 0, 0)')
  await alias.hover()
  await expect(alias).not.toHaveCSS('border-top-color', 'rgba(0, 0, 0, 0)')
  await page.evaluate(() => localStorage.setItem('gpt-load.theme', 'dark'))
  await page.reload()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
  await alias.focus()
  await expect(alias).toHaveCSS('outline-width', '2px')
  if (process.env.MODEL_POLISH_SCREENSHOTS) {
    await page.screenshot({ path: '/tmp/gpt-load-model-polish-dark.png', fullPage: true })
  }
})
