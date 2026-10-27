import { expect, test } from '@playwright/test'

import { mockGroupModels } from './fixtures/group-models-mock'

// Group route fields: the maintenance editor keeps weight/priority editable
// and states the effective value (an unset field routes as 1) instead of a
// read-only placeholder, and neither entrypoint paints a calculated share.

test.setTimeout(120_000)

test('maintenance shows editable effective route fields and no share presentation', async ({
  page,
}, testInfo) => {
  const { puts } = await mockGroupModels(page, {
    models: [
      // Unset route fields: routed as weight 1 / priority 1, persisted as null.
      { id: 'gpt-5.2', alias: 'shared', alias_enabled: true },
      // Explicit values, including the zero weight that cuts an entry out.
      { id: 'gpt-5.2-mini', alias: 'shared', alias_enabled: true, weight: 0, priority: 3 },
      { id: 'gpt-5.3' },
    ],
  })
  await page.goto('/groups/1?tab=models', { waitUntil: 'load' })
  const table = page.getByRole('table', { name: 'Current Group models' })
  const rows = table.locator('[data-testid="model-alias-editor__record"]')
  await expect(rows).toHaveCount(3, { timeout: 60_000 })

  const weight = (index: number) => rows.nth(index).locator(`[data-model-weight-index="${index}"]`)
  const priority = (index: number) =>
    rows.nth(index).locator(`[data-model-priority-index="${index}"]`)

  // Editable, not a read-only chip.
  await expect(weight(0)).toBeVisible()
  await expect(weight(0)).toBeEditable()
  await expect(priority(0)).toBeEditable()
  // An unset field reads as the effective default rather than a dash.
  await expect(weight(0)).toHaveValue('1')
  await expect(priority(0)).toHaveValue('1')
  // Explicit values are shown unchanged, zero included.
  await expect(weight(1)).toHaveValue('0')
  await expect(priority(1)).toHaveValue('3')

  // No calculated shares: no percentage text, no distribution bar.
  await expect(table.locator('i')).toHaveCount(0)
  for (const row of [rows.nth(0), rows.nth(1), rows.nth(2)]) {
    await expect(row).not.toContainText('%')
  }

  // Editing one entry leaves the untouched unset fields null in the payload.
  await weight(0).fill('9')
  await priority(0).fill('2')
  await page.getByRole('button', { name: 'Save settings', exact: true }).click()
  await expect.poll(() => puts.length).toBe(1)
  expect(puts[0]).toMatchObject({
    models: [
      { id: 'gpt-5.2', weight: 9, priority: 2 },
      { id: 'gpt-5.2-mini', weight: 0, priority: 3 },
      { id: 'gpt-5.3', weight: null, priority: null },
    ],
  })

  // Saved values survive a reload; the untouched entry still reads 1.
  await page.reload({ waitUntil: 'load' })
  await expect(rows).toHaveCount(3, { timeout: 60_000 })
  await expect(weight(0)).toHaveValue('9')
  await expect(priority(0)).toHaveValue('2')
  await expect(weight(1)).toHaveValue('0')
  await expect(weight(2)).toHaveValue('1')
  await page.screenshot({ path: testInfo.outputPath('route-fields-desktop.png'), fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await expect(weight(0)).toBeVisible()
  await expect(priority(0)).toBeVisible()
  await page.screenshot({ path: testInfo.outputPath('route-fields-mobile.png'), fullPage: true })
})

test('maintenance keeps the route-field validation gate', async ({ page }) => {
  await mockGroupModels(page, { models: [{ id: 'gpt-5.2' }] })
  await page.goto('/groups/1?tab=models', { waitUntil: 'load' })
  const rows = page.locator('[data-testid="model-alias-editor__record"]')
  await expect(rows).toHaveCount(1, { timeout: 60_000 })
  const weight = page.locator('[data-model-weight-index="0"]')
  const priority = page.locator('[data-model-priority-index="0"]')
  const save = page.getByRole('button', { name: 'Save settings', exact: true })
  const weightError = page.getByText('Weight must be an integer between 0 and 100 before saving')
  const zeroShareError = page.getByText(
    'Entries under one client name sum to weight 0; the model cannot receive traffic',
  )
  const priorityError = page.getByText('Priority must be an integer of at least 1 before saving')

  for (const invalid of ['-1', '101', '1.5']) {
    await weight.fill(invalid)
    await expect(weightError.first()).toBeVisible()
    await expect(save).toBeDisabled()
  }

  await weight.fill('100')
  await expect(save).toBeEnabled()

  // Weight 0 alone leaves the client name with no routable traffic.
  await weight.fill('0')
  await expect(zeroShareError.first()).toBeVisible()
  await expect(save).toBeDisabled()

  await weight.fill('1')
  await expect(save).toBeEnabled()

  for (const invalid of ['0', '-1', '1.5']) {
    await priority.fill(invalid)
    await expect(priorityError.first()).toBeVisible()
    await expect(save).toBeDisabled()
  }

  await priority.fill('1')
  await expect(save).toBeEnabled()
})
