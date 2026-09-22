import { expect, test, type Page } from '@playwright/test'

import {
  installModelTestAliasRoutes,
  modelTestAliasGroupId,
  rowOneTestAlias,
  rowTwoTestAlias,
  type SavedModelBody,
} from './fixtures/model-test-alias'

// U003: the classic group model editor shows the server-owned read-only
// test alias without horizontal overflow at desktop/mobile widths, keeps
// keyboard focus order intact, and never sends test_alias back on save.

async function openGroupModelsEditor(page: Page): Promise<SavedModelBody[]> {
  const savedModelBodies = await installModelTestAliasRoutes(page)
  await page.goto(`/groups/${modelTestAliasGroupId}`)
  await expect(page.locator('.model-alias-editor__record')).toHaveCount(2)
  return savedModelBodies
}

async function assertNoHorizontalOverflow(page: Page): Promise<void> {
  const overflow = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }))
  expect(overflow.scrollWidth).toBeLessThanOrEqual(overflow.clientWidth)
}

test.describe('group model test alias', () => {
  test('renders the six-character test alias as read-only code at desktop width', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 720 })
    await openGroupModelsEditor(page)

    const codes = page.locator('.group-models__test-alias code')
    await expect(codes).toHaveCount(2)
    await expect(codes.nth(0)).toHaveText(rowOneTestAlias)
    await expect(codes.nth(1)).toHaveText(rowTwoTestAlias)

    for (const code of await codes.all()) {
      const readOnly = await code.evaluate((element) => ({
        tag: element.tagName.toLowerCase(),
        contentEditable: element.isContentEditable,
        focusable: (element.focus(), document.activeElement === element),
      }))
      expect(readOnly.tag).toBe('code')
      expect(readOnly.contentEditable).toBe(false)
      expect(readOnly.focusable).toBe(false)
    }

    await assertNoHorizontalOverflow(page)
  })

  test('keeps the table free of horizontal overflow at mobile width', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await openGroupModelsEditor(page)

    await expect(page.locator('.group-models__test-alias code').nth(0)).toHaveText(rowOneTestAlias)
    await assertNoHorizontalOverflow(page)
  })

  test('keyboard focus reaches the editable controls but never the test alias', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 720 })
    await openGroupModelsEditor(page)

    const firstToggle = page.locator('[data-alias-toggle-index="0"]')
    await firstToggle.focus()
    await expect(firstToggle).toBeFocused()

    const focusPath: string[] = []
    for (let step = 0; step < 8; step += 1) {
      await page.keyboard.press('Tab')
      const descriptor = await page.evaluate(() => {
        const active = document.activeElement
        if (active === null) return 'none'
        if (active.closest('.group-models__test-alias')) return 'test-alias'
        for (const attribute of ['data-alias-toggle-index', 'data-alias-input-index']) {
          if (active.hasAttribute(attribute))
            return `${attribute}=${active.getAttribute(attribute)}`
        }
        return `${active.tagName.toLowerCase()}:${active.textContent?.trim().slice(0, 24) ?? ''}`
      })
      focusPath.push(descriptor)
    }

    expect(focusPath).not.toContain('test-alias')
    // Row order is preserved: after row 0's controls, tab reaches row 1's toggle
    // and then its alias input. The test alias code is never a stop.
    const toggle1 = focusPath.indexOf('data-alias-toggle-index=1')
    const input1 = focusPath.indexOf('data-alias-input-index=1')
    expect(focusPath.join(' | '), 'tab stops after row 0 toggle').toContain(
      'data-alias-toggle-index=1',
    )
    expect(toggle1).toBeGreaterThanOrEqual(0)
    expect(input1).toBeGreaterThan(toggle1)
  })

  test('saving the model list never sends test_alias and keeps the server value', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 720 })
    const savedModelBodies = await openGroupModelsEditor(page)

    // Make the draft dirty through an existing editable control, then save.
    const aliasInput = page.locator('[data-alias-input-index="1"]')
    await aliasInput.fill('worker-b2')
    await page.getByRole('button', { name: 'Save settings' }).click()
    await expect(page.getByText('Saved successfully')).toBeVisible()

    expect(savedModelBodies).toHaveLength(1)
    const body = JSON.stringify(savedModelBodies[0])
    expect(body).not.toContain('test_alias')
    expect(body).not.toContain(rowOneTestAlias)
    expect(body).not.toContain(rowTwoTestAlias)

    // The save response restores the server-owned alias unchanged.
    const codes = page.locator('.group-models__test-alias code')
    await expect(codes.nth(0)).toHaveText(rowOneTestAlias)
    await expect(codes.nth(1)).toHaveText(rowTwoTestAlias)
  })
})
