import { expect, test } from '@playwright/test'

import { mockGroupFingerprint } from './fixtures/group-fingerprint-mock'

for (const width of [1440, 375]) {
  for (const theme of ['light', 'dark']) {
    test(`model removal is distinct, accessible and keeps draft behavior at ${width}px ${theme}`, async ({
      page,
    }, testInfo) => {
      await page.setViewportSize({ width, height: 900 })
      await mockGroupFingerprint(page)
      await page.addInitScript((value) => localStorage.setItem('gpt-load.theme', value), theme)
      await page.goto('/groups/1?tab=models')
      await expect(page.locator('html')).toHaveAttribute('data-theme', theme)
      const row = page.getByTestId('model-alias-editor__record').first()
      const remove = row.getByRole('button', { name: 'Remove model gpt-5.4' })
      const fingerprint = row.getByRole('button', { name: 'Fingerprint', exact: true })
      await expect(remove.locator('svg.lucide-trash-2')).toBeVisible()
      const metrics = await remove.evaluate((element) => {
        const style = getComputedStyle(element)
        const root = getComputedStyle(document.documentElement)
        const box = element.getBoundingClientRect()
        return {
          color: style.color,
          background: style.backgroundColor,
          danger: root.getPropertyValue('--color-danger').trim(),
          width: box.width,
          height: box.height,
        }
      })
      const danger = await page.evaluate((value) => {
        const probe = document.createElement('span')
        probe.style.color = value
        document.body.append(probe)
        const color = getComputedStyle(probe).color
        probe.remove()
        return color
      }, metrics.danger)
      expect(metrics.color).toBe(danger)
      const luminance = (color: string) => {
        const channels = color
          .match(/\d+/g)!
          .slice(0, 3)
          .map((n) => Number(n) / 255)
        const linear = channels.map((c) =>
          c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4,
        )
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
      }
      const levels = [luminance(metrics.color), luminance(metrics.background)].sort((a, b) => b - a)
      expect((levels[0] + 0.05) / (levels[1] + 0.05)).toBeGreaterThanOrEqual(3)
      expect(metrics.width).toBeGreaterThanOrEqual(width <= 860 ? 44 : 32)
      expect(metrics.height).toBeGreaterThanOrEqual(width <= 860 ? 44 : 32)
      if (width > 860) {
        const [actionBox, removeBox] = await Promise.all([
          fingerprint.boundingBox(),
          remove.boundingBox(),
        ])
        expect(removeBox!.x - (actionBox!.x + actionBox!.width)).toBeGreaterThanOrEqual(16)
      }
      await remove.hover()
      await expect(remove).toHaveCSS('color', danger)
      await remove.focus()
      await expect(remove).toBeFocused()
      expect(
        parseFloat(await remove.evaluate((el) => getComputedStyle(el).outlineWidth)),
      ).toBeGreaterThanOrEqual(2)
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
        width,
      )
      await page.screenshot({
        path: testInfo.outputPath(`model-danger-${width}-${theme}.png`),
        fullPage: true,
      })
      await page.keyboard.press('Enter')
      await expect(row).toHaveCount(0)
      await expect(page.getByRole('button', { name: 'Save settings', exact: true })).toBeEnabled()
    })
  }
}

test('shared control and tag radii follow the Astryx theme scale', async ({ page }) => {
  await mockGroupFingerprint(page)
  await page.goto('/groups/1?tab=models')
  const row = page.getByTestId('model-alias-editor__record').first()
  const input = row.getByLabel('Client alias for gpt-5.4')
  const select = row.locator('select')
  const tag = row.getByText('Configured', { exact: true })
  const button = page.getByRole('button', { name: 'Add a model', exact: true })
  const pill = page.getByText('Available', { exact: true }).locator('..')
  const switchTrack = page.getByRole('switch', { name: 'Group enabled' }).locator('+ div')
  for (const control of [input, select, button])
    await expect(control).toHaveCSS('border-radius', '8px')
  await expect(tag).toHaveCSS('border-radius', '8px')
  await expect(pill).toHaveCSS('border-radius', '9999px')
  await expect(switchTrack).toHaveCSS('border-radius', '9999px')
  // A theme-scale change must reach both kit components and app-owned controls.
  await page.evaluate(() => {
    const themes = [...document.querySelectorAll<HTMLElement>('[data-astryx-theme="gptload"]')]
    const theme = themes[themes.length - 1]
    theme.style.setProperty('--radius-element', '10px')
    theme.style.setProperty('--radius-inner', '7px')
  })
  for (const control of [input, select, button])
    await expect(control).toHaveCSS('border-radius', '10px')
  await expect(tag).toHaveCSS('border-radius', '10px')
  await expect(pill).toHaveCSS('border-radius', '9999px')
  await expect(switchTrack).toHaveCSS('border-radius', '9999px')
})
