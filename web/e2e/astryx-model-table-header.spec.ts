import { expect, test } from '@playwright/test'

import { mockGroupFingerprint } from './fixtures/group-fingerprint-mock'

test('model editor keeps wrapped headers centered with breathing room', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await mockGroupFingerprint(page)
  await page.goto('/groups/1?tab=models')
  const table = page.getByRole('table', { name: 'Current Group models' })
  const headers = table.getByRole('columnheader')
  await expect(headers.first()).toBeVisible({ timeout: 30_000 })
  await headers.first().evaluate((cell) => cell.append(document.createElement('br'), 'Second line'))
  const geometry = await headers.first().evaluate((cell) => {
    const range = document.createRange()
    range.selectNodeContents(cell)
    const text = range.getBoundingClientRect()
    const row = cell.parentElement!.getBoundingClientRect()
    const neighbor = cell.nextElementSibling!.getBoundingClientRect()
    const style = getComputedStyle(cell.parentElement!)
    return {
      top: text.top - row.top,
      bottom: row.bottom - text.bottom,
      neighborDelta: neighbor.top + neighbor.height / 2 - (row.top + row.height / 2),
      paddingTop: style.paddingTop,
      paddingBottom: style.paddingBottom,
    }
  })
  expect(geometry.top).toBeGreaterThanOrEqual(4)
  expect(geometry.bottom).toBeGreaterThanOrEqual(4)
  expect(geometry.paddingTop).toBe(geometry.paddingBottom)
  expect(Math.abs(geometry.neighborDelta)).toBeLessThanOrEqual(1)
})
