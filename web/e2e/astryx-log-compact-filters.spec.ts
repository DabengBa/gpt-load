import { expect, test } from '@playwright/test'

import { installRequestLogDisplayRoutes } from './fixtures/request-log-display'

test('request log controls use available desktop space without a standalone summary row', async ({
  page,
}, info) => {
  await page.setViewportSize({ width: 1440, height: 1000 })
  await installRequestLogDisplayRoutes(page)
  await page.goto('/logs', { waitUntil: 'commit' })
  await expect(page.getByTestId('logs-list__record')).toHaveCount(4, { timeout: 60_000 })

  const model = page.getByRole('button', { name: 'Client model' })
  const from = page.getByRole('combobox', { name: 'From', exact: true })
  const to = page.getByRole('combobox', { name: 'To', exact: true })
  const presets = page.getByRole('group', { name: 'Quick time ranges' })
  const summary = page.getByTestId('logs-result-summary')
  const pagination = page.getByRole('navigation', { name: 'Pagination' })
  await expect(summary).toHaveText('4 logs on this page · Newest first')
  await expect(presets).toContainText('Local timezone (local time)')
  await page.screenshot({ path: info.outputPath('desktop.png'), fullPage: true })
  const [modelBox, fromBox, toBox, presetsBox, summaryBox, paginationBox] = await Promise.all(
    [model, from, to, presets, summary, pagination].map((element) => element.boundingBox()),
  )
  expect(modelBox!.width).toBeLessThanOrEqual(240)
  expect(Math.abs(fromBox!.y - toBox!.y)).toBeLessThan(4)
  expect(presetsBox!.y).toBeLessThan(toBox!.y + toBox!.height)
  expect(summaryBox!.y).toBeGreaterThanOrEqual(paginationBox!.y - 2)
  expect(summaryBox!.y).toBeLessThan(paginationBox!.y + paginationBox!.height)
})

test('request log controls remain labelled and fit on a narrow screen', async ({ page }, info) => {
  await page.setViewportSize({ width: 375, height: 812 })
  await installRequestLogDisplayRoutes(page)
  await page.goto('/logs', { waitUntil: 'commit' })
  await expect(page.getByTestId('logs-list__record')).toHaveCount(4, { timeout: 60_000 })
  await expect(page.getByRole('combobox', { name: 'From', exact: true })).toBeVisible()
  await expect(page.getByRole('combobox', { name: 'To', exact: true })).toBeVisible()
  await expect(page.getByRole('group', { name: 'Quick time ranges' })).toBeVisible()
  await expect(page.getByTestId('logs-result-summary')).toBeVisible()
  const previous = await page.getByRole('button', { name: 'Previous' }).boundingBox()
  const next = await page.getByRole('button', { name: 'Next' }).boundingBox()
  expect(Math.abs(previous!.y - next!.y)).toBeLessThan(4)
  await page.screenshot({ path: info.outputPath('mobile.png'), fullPage: true })
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(375)
})
