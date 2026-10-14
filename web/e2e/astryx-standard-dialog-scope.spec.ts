import { expect, test } from '@playwright/test'

import { mockGroupModels } from './fixtures/group-models-mock'

// Guard for the DetailPanel visual polish: the drawer geometry rules live in
// entry.css scoped to [data-detail-panel]. Standard Dialogs must keep the
// Astryx-native header (no clipping) and their own width prop.
test.setTimeout(120_000)

test('standard sync dialog keeps native header geometry and its own width', async ({ page }) => {
  await mockGroupModels(page)
  await page.goto('/groups/1?tab=models&panel=discovery', { waitUntil: 'load' })
  // Discovery runs automatically on load; wait for the drawer, then close it
  // (Sync enables once discovery completes).
  const discovery = page.getByRole('dialog', { name: 'Get upstream models' })
  await expect(discovery).toBeVisible({ timeout: 60_000 })
  // Discovery completes automatically; the Sync footer action then enables.
  // (The drawer may take a moment to mount on a cold dev server, so poll.)
  const syncButton = discovery.getByRole('button', { name: 'Sync', exact: true })
  await expect(async () => {
    if ((await syncButton.count()) === 0) {
      await page.goto('/groups/1?tab=models&panel=discovery', { waitUntil: 'load' })
      await expect(discovery).toBeVisible({ timeout: 30_000 })
    }
    await syncButton.click({ timeout: 5_000 })
  }).toPass({ timeout: 90_000 })
  const dialog = page.getByRole('dialog', { name: 'Sync models' })
  await expect(dialog).toBeVisible()
  await dialog.evaluate(async (node) => {
    await Promise.all(node.getAnimations().map((animation) => animation.finished))
  })

  const geo = await dialog.evaluate((node) => {
    const header = node.querySelector('.astryx-dialog-header') as HTMLElement
    const titleBlock = node.querySelector('.astryx-dialog-header-title-block') as HTMLElement
    const close = header.querySelector('button') as HTMLElement | null
    return {
      width: Math.round(node.getBoundingClientRect().width),
      titleBlockOverflow: getComputedStyle(titleBlock).overflow,
      closeRight: close ? Math.round(close.getBoundingClientRect().right) : null,
      drawerRight: Math.round(node.getBoundingClientRect().right),
      isDetailPanel: node.hasAttribute('data-detail-panel'),
    }
  })
  expect(geo.isDetailPanel).toBe(false)
  // Measure the border box after the entry transform finishes.
  expect(geo.width).toBe(520)
  // Native header: the DetailPanel clipping rules do not leak in (core
  // already ships min-width:0 on the title block; only overflow differs).
  expect(geo.titleBlockOverflow).not.toBe('hidden')
  // The close button still paints (its -8px bleed tucks into header padding
  // exactly like any standard Astryx dialog).
  expect(geo.closeRight).not.toBeNull()
  expect(geo.closeRight!).toBeLessThanOrEqual(geo.drawerRight)
})
