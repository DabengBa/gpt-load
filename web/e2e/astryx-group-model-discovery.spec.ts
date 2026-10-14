import { expect, test, type Page, type Route } from '@playwright/test'

import { mockGroupModels } from './fixtures/group-models-mock'

// U003: the Group models discovery drawer must present live/pricing evidence
// through an always-visible legend (not tooltip-only icons) and a batch
// selection summary that states how many of the available candidates are
// selected. Confirmed candidates still append through the existing PUT.

test.setTimeout(120_000)

test('discovery drawer shows an evidence legend and a scoped selection summary', async ({
  page,
}) => {
  const { puts } = await mockGroupModels(page)
  await page.goto('/groups/1?tab=models&panel=discovery', { waitUntil: 'load' })
  const dialog = page.getByRole('dialog', { name: 'Get upstream models' })
  await expect(dialog).toBeVisible({ timeout: 60_000 })

  // Evidence is readable without hovering: the legend names both markers.
  const legend = dialog.getByLabel('Evidence legend')
  await expect(legend).toBeVisible()
  await expect(legend.getByText('Live discovery', { exact: true })).toBeVisible()
  await expect(legend.getByText('Price found from a matched provider', { exact: true })).toBeVisible()

  // Batch selection states how many of the available candidates are picked.
  await expect(dialog.getByText('0 of 2 available models selected')).toBeVisible()
  // Checkbox accessible names include the evidence markers rendered inline,
  // so select by the model name prefix instead of an exact match.
  await dialog.getByRole('checkbox', { name: /^GPT-5\.2 GPT-5\.2 gpt-5\.2/ }).click()
  await dialog.getByRole('checkbox', { name: /^GPT-5\.2 Mini GPT-5\.2 Mini gpt-5\.2-mini/ }).click()
  await expect(dialog.getByText('2 of 2 available models selected')).toBeVisible()

  await dialog.getByRole('button', { name: 'Add to current list' }).click()
  await expect(dialog).not.toBeVisible()
  // Confirmed candidates land in the draft; the unified editor only sends
  // the PUT when the page-level Save models action runs.
  await page.getByRole('button', { name: 'Save settings', exact: true }).click()
  await expect.poll(() => puts.length).toBe(1)
  expect(puts[0]).toMatchObject({
    models: [
      { id: 'gpt-5.2' },
      { id: 'gpt-5.2-mini' },
    ],
  })
})
