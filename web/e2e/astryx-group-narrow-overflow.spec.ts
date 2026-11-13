import { expect, test, type Page } from '@playwright/test'

import { mockGroupFingerprint } from './fixtures/group-fingerprint-mock'

async function expectNoPageOverflow(page: Page) {
  const dimensions = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
    overflowing: Array.from(document.querySelectorAll('body *'))
      .filter(
        (element) =>
          element.getBoundingClientRect().right > document.documentElement.clientWidth + 1,
      )
      .slice(0, 15)
      .map(
        (element) =>
          `${element.tagName}.${element.className} ${Math.round(element.getBoundingClientRect().right)}`,
      ),
  }))
  expect(dimensions.scroll, JSON.stringify(dimensions)).toBeLessThanOrEqual(dimensions.viewport)
}

for (const width of [320, 375]) {
  for (const invalidWeight of [false, true])
    test(`group detail fits ${width}px ${invalidWeight ? 'with an invalid weight' : 'initially'}`, async ({
      page,
    }, testInfo) => {
      await page.setViewportSize({ width, height: 900 })
      await mockGroupFingerprint(page)
      await page.goto('/groups/1')
      const row = page.getByTestId('model-alias-editor__record').first()
      await expect(row).toBeVisible()
      if (invalidWeight) {
        await row.getByLabel('Weight', { exact: true }).fill('-1')
        await expect(row.getByLabel('Weight', { exact: true })).toHaveAttribute(
          'aria-invalid',
          'true',
        )
        await expect(
          page.getByRole('button', { name: 'Save settings', exact: true }),
        ).toBeDisabled()
      }
      await expectNoPageOverflow(page)
      const footer = page
        .locator('footer')
        .filter({ has: page.getByRole('button', { name: 'Save settings' }) })
      const children = await footer.locator(':scope > *').evaluateAll((elements) =>
        elements.map((element) => ({
          width: element.clientWidth,
          scroll: element.scrollWidth,
          text: element.textContent,
        })),
      )
      for (const child of children)
        expect(child.scroll, JSON.stringify(children)).toBeLessThanOrEqual(child.width)
      const boxes = await footer.locator(':scope > *').evaluateAll((elements) =>
        elements.map((element) => {
          const { left, top, right, bottom } = element.getBoundingClientRect()
          return { left, top, right, bottom }
        }),
      )
      for (let i = 0; i < boxes.length; i++) {
        for (let j = i + 1; j < boxes.length; j++) {
          const a = boxes[i],
            b = boxes[j]
          expect(
            a.right <= b.left || b.right <= a.left || a.bottom <= b.top || b.bottom <= a.top,
            JSON.stringify(boxes),
          ).toBe(true)
        }
      }
      if (invalidWeight) {
        const locate = page.getByRole('button', { name: 'Clear search and locate the first issue' })
        await expect(locate).toBeVisible()
        const box = await locate.boundingBox()
        expect(box && box.x >= 0 && box.x + box.width <= width).toBeTruthy()
      }
      await page.screenshot({
        path: testInfo.outputPath(`narrow-${width}-${invalidWeight}.png`),
        fullPage: true,
      })
      if (invalidWeight) {
        await page.getByRole('button', { name: 'Clear search and locate the first issue' }).click()
        await expect(row.getByLabel('Weight', { exact: true })).toBeFocused()
        await page.getByRole('button', { name: 'Discard', exact: true }).click()
        await expect(row.getByLabel('Weight', { exact: true })).toHaveValue('1')
        await expectNoPageOverflow(page)
      }
    })
}
