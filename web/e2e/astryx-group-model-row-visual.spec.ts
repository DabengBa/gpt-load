import { expect, test } from '@playwright/test'

import { mockGroupFingerprint } from './fixtures/group-fingerprint-mock'

for (const width of [1440, 900, 375]) {
  test(`test alias has its own aligned column at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 900 })
    await mockGroupFingerprint(page)
    await page.goto('/groups/1?tab=models')
    const table = page.getByRole('table', { name: 'Current Group models' })
    const row = table.getByTestId('model-alias-editor__record').first()
    const cells = row.getByRole('cell')
    await expect(cells).toHaveCount(6)
    await expect(cells.nth(1).getByLabel('Client alias for gpt-5.4')).toBeVisible()
    await expect(cells.nth(2).getByTestId('group-models__test-alias')).toHaveText('a4g233')
    await expect(cells.nth(3).getByLabel('Weight')).toBeVisible()
    await expect(cells.nth(4).getByLabel('Enable model gpt-5.4')).toBeChecked()
    await expect(cells.nth(4).getByRole('link', { name: 'Schedule' })).toBeVisible()
    await expect(cells.nth(5).getByRole('button', { name: 'Remove model gpt-5.4' })).toBeVisible()
    if (width > 860) {
      const headers = table.getByRole('columnheader')
      await expect(headers).toHaveText([
        'Model ID',
        'Client alias',
        'Test alias',
        'Weight / Priority',
        'Scheduling / pricing / actions',
        '',
      ])
      await expect(headers.nth(5)).toHaveAttribute('aria-label', 'Remove')
      const headerBoxes = await headers.evaluateAll((items) =>
        items.map((item) => {
          const box = item.getBoundingClientRect()
          return { x: box.x, width: box.width }
        }),
      )
      const cellBoxes = await cells.evaluateAll((items) =>
        items.map((item) => {
          const box = item.getBoundingClientRect()
          return { x: box.x, width: box.width }
        }),
      )
      for (let i = 0; i < headerBoxes.length; i++) {
        expect(Math.abs(headerBoxes[i].x - cellBoxes[i].x)).toBeLessThan(1)
        expect(Math.abs(headerBoxes[i].width - cellBoxes[i].width)).toBeLessThan(1)
      }
    } else {
      await expect(cells.nth(2)).toContainText('Test alias')
      const labelBox = await cells.nth(2).getByText('Test alias', { exact: true }).boundingBox()
      const valueBox = await cells.nth(2).getByTestId('group-models__test-alias').boundingBox()
      expect(labelBox!.y + labelBox!.height).toBeLessThanOrEqual(valueBox!.y)
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
        width,
      )
    }
    const aliasInput = cells.nth(1).getByLabel('Client alias for gpt-5.4')
    await aliasInput.focus()
    await expect(aliasInput).toBeFocused()
    await page.keyboard.press('Tab')
    await expect(cells.nth(3).getByLabel('Weight')).toBeFocused()
    await page.screenshot({
      path: testInfo.outputPath(`model-columns-${width}.png`),
      fullPage: true,
    })
  })

  test(`model row keeps coherent actions and numeric alignment at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 })
    await mockGroupFingerprint(page)
    await page.goto('/groups/1?tab=models')
    const row = page.getByTestId('model-alias-editor__record').first()
    await expect(row).toBeVisible()
    const actions = [
      row.getByRole('link', { name: 'Schedule' }),
      row.getByRole('button', { name: 'Probe', exact: true }),
      row.getByRole('button', { name: 'Fingerprint', exact: true }),
      row.getByRole('button', { name: 'Remove model gpt-5.4' }),
    ]
    const chrome = await Promise.all(
      actions.map((action) =>
        action.evaluate((element) => {
          const style = getComputedStyle(element)
          return { border: style.borderStyle, radius: style.borderRadius, height: style.height }
        }),
      ),
    )
    expect(chrome.every((style) => style.border === 'solid')).toBe(true)
    expect(new Set(chrome.map((style) => style.radius)).size).toBe(1)
    expect(new Set(chrome.slice(0, 3).map((style) => style.height)).size).toBe(1)
    expect(chrome[3].height).toBe(width === 375 ? '44px' : chrome[0].height)
    await expect(row.getByLabel('Weight')).toHaveCSS('font-variant-numeric', 'tabular-nums')
    await expect(row.getByLabel('Priority')).toHaveCSS('font-variant-numeric', 'tabular-nums')
    const badge = row.getByText('Configured', { exact: true })
    await expect(badge).toHaveCSS('color', 'rgb(26, 107, 63)')
    await expect(badge).toHaveCSS('background-color', 'rgb(230, 243, 236)')
    if (width === 375) {
      const viewport = await page.evaluate(() => document.documentElement.clientWidth)
      for (const action of actions) {
        const box = await action.boundingBox()
        expect(box && box.x >= 0 && box.x + box.width <= viewport).toBeTruthy()
      }
      await row.getByLabel('Client alias for gpt-5.4').focus()
      await expect(row.getByLabel('Client alias for gpt-5.4')).toBeFocused()
      await row.getByLabel('Client alias for gpt-5.4').fill('example-alias')
      const alias = await row.getByLabel('Client alias for gpt-5.4').boundingBox()
      expect(alias && alias.x >= 0 && alias.x + alias.width <= viewport).toBeTruthy()
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
        viewport,
      )
    }
  })
}
