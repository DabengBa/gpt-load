// Standalone prototype checks, not application E2E. No backend or dev server.
// Run: node --test docs/design/schedule-center-preview.test.mjs
// Screenshots: /tmp/gpt-load-schedule-preview/ (caller owns cleanup).
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { test } from 'node:test'
import { chromium } from '../../web/node_modules/@playwright/test/index.mjs'

const preview = new URL('./schedule-center-preview.html', import.meta.url).href

test('preview preserves every draft field and provides responsive recovery paths', async () => {
  const browser = await chromium.launch({ headless: true, channel: 'chrome' })
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } })
  const page = await context.newPage()
  const errors = []
  const measurements = []
  page.on('pageerror', (error) => errors.push(error.message))
  try {
    await page.goto(preview)
    await page.getByRole('heading', { name: '调度中心', exact: true }).waitFor()
    const first = page.locator('[data-candidate]').first()
    const initialHeight = (await first.boundingBox()).height
    assert.ok(initialHeight <= 84, `Normal desktop row must stay compact, received ${initialHeight}px`)
    await first.locator('[data-field="reasoning"]').selectOption('high')
    await page.getByTestId('save-bar').waitFor({ state: 'visible' })
    await first.getByRole('switch').click()
    assert.equal(await first.locator('[data-field="reasoning"]').inputValue(), 'high')
    await page.reload()
    assert.equal(await page.locator('[data-field="reasoning"]').first().inputValue(), 'high')
    await page.locator('[data-model="claude-sonnet-4-5"]').first().click()
    const guard = page.getByRole('dialog', { name: '切换前处理未保存修改' })
    await guard.waitFor()
    await guard.getByRole('button', { name: '留在当前模型' }).click()
    assert.equal(await page.locator('[data-field="reasoning"]').first().inputValue(), 'high')

    const priority = page.locator('[data-field="priority"]').first()
    await priority.fill('0')
    assert.equal(await priority.getAttribute('aria-invalid'), 'true')
    await page.getByText('请输入大于或等于 1 的整数。', { exact: true }).waitFor()
    assert.equal(await page.getByTestId('save-button').isDisabled(), true)
    await priority.fill('1')
    await page.getByTestId('save-button').click()
    await page.getByTestId('save-bar').waitFor({ state: 'hidden' })

    for (const width of [1920, 1440, 1366, 1200, 900, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 })
      const geometry = await page.evaluate(() => ({
        pageWidth: document.documentElement.scrollWidth,
        width: innerWidth,
        rowWidth: document.querySelector('[data-candidate]').scrollWidth,
        rowClientWidth: document.querySelector('[data-candidate]').clientWidth,
        rowHeight: document.querySelector('[data-candidate]').getBoundingClientRect().height,
      }))
      assert.ok(geometry.pageWidth <= width + 1, JSON.stringify(geometry))
      assert.ok(geometry.rowWidth <= geometry.rowClientWidth + 1, JSON.stringify(geometry))
      assert.ok(geometry.rowHeight <= (width >= 768 ? 100 : 240), JSON.stringify(geometry))
      measurements.push({ width, rowHeight: geometry.rowHeight })
      assert.equal(await page.locator('[data-candidate]').first().locator('[data-stat="success"]').isVisible(), true)
      if (width <= 900) {
        assert.equal(await page.getByTestId('model-sidebar').isVisible(), false)
        assert.equal(await page.getByTestId('model-picker').isVisible(), true)
      }
    }

    await page.getByTestId('model-picker').click()
    const picker = page.getByRole('dialog', { name: '切换对外模型' })
    await picker.getByRole('searchbox').fill('不存在')
    await picker.getByText('没有匹配的模型', { exact: true }).waitFor()
    await picker.getByRole('button', { name: '清空搜索' }).click()
    await picker.getByRole('button', { name: /claude-sonnet-4-5/ }).click()
    assert.equal(await page.getByTestId('current-model').innerText(), 'claude-sonnet-4-5')

    await page.locator('#prototype-tools > summary').click()
    await page.getByRole('button', { name: '重置演示' }).click()
    await page.getByRole('button', { name: '加载草稿示例' }).click()
    await page.getByTestId('conflict-toggle').check()
    await page.getByTestId('save-button').click()
    await page.getByRole('alert').filter({ hasText: '服务器配置已变化' }).waitFor()
    assert.equal(await page.locator('[data-field="reasoning"]').first().inputValue(), 'high')
    await page.getByTestId('conflict-toggle').uncheck()
    await page.getByTestId('save-button').click()
    await page.getByTestId('save-bar').waitFor({ state: 'hidden' })
    await page.getByRole('button', { name: '重置演示' }).click()

    await page.setViewportSize({ width: 390, height: 1000 })
    await page.getByRole('button', { name: '加载草稿示例' }).click()
    await page.getByTestId('model-picker').click()
    await picker.getByRole('button', { name: /claude-sonnet-4-5/ }).click()
    await guard.getByRole('button', { name: '保存并继续' }).click()
    await page.getByTestId('save-bar').waitFor({ state: 'hidden' })
    assert.equal(await page.getByTestId('current-model').innerText(), 'claude-sonnet-4-5')
    await page.getByTestId('model-picker').click()
    await picker.getByRole('button', { name: /^gpt-4\.1-mini/ }).click()
    assert.equal(await page.locator('[data-field="reasoning"]').first().inputValue(), 'high')
    await page.locator('[data-field="weight"]').first().fill('30')
    await page.getByTestId('model-picker').click()
    await picker.getByRole('button', { name: /claude-sonnet-4-5/ }).click()
    await guard.getByRole('button', { name: '放弃并继续' }).click()
    await page.getByRole('button', { name: '查看失效模型链接' }).click()
    await page.getByRole('heading', { name: '找不到这个对外模型' }).waitFor()
    await page.getByRole('button', { name: '重新选择模型' }).click()
    await picker.getByRole('button', { name: /^gpt-4\.1-mini/ }).click()
    assert.equal(await page.locator('[data-field="weight"]').first().inputValue(), '75')
    const candidate = page.locator('[data-candidate]').first()
    await candidate.getByRole('button', { name: /更多操作/ }).click()
    const advanced = candidate.locator('[popover]')
    await advanced.waitFor({ state: 'visible' })
    const menuBounds = await advanced.boundingBox()
    assert.ok(menuBounds.x >= 0 && menuBounds.x + menuBounds.width <= 390 && menuBounds.y >= 0 && menuBounds.y + menuBounds.height <= 1000, JSON.stringify(menuBounds))
    await advanced.locator('[data-field="concurrency"]').fill('-1')
    assert.equal(await page.getByTestId('save-button').isDisabled(), true)
    await page.keyboard.press('Escape')
    await page.getByRole('button', { name: /1 项需修正/ }).click()
    assert.equal(await advanced.isVisible(), true)
    await page.keyboard.press('Escape')
    await page.getByRole('button', { name: '放弃修改', exact: true }).click()
    const sizes = await page.locator('[data-candidate]').first().locator('button[role="switch"], select, input, .actions > button').evaluateAll((nodes) => nodes.filter(node => node.getClientRects().length).map((node) => ({ height: node.getBoundingClientRect().height, width: node.getBoundingClientRect().width })))
    assert.ok(sizes.every((size) => size.height >= 44 && size.width >= 44), JSON.stringify(sizes))
    await page.locator('[data-candidate="backup"]').getByRole('button', { name: '恢复', exact: true }).click()
    assert.equal(await page.locator('[data-candidate="backup"] .state').innerText(), '可用')
    await page.locator('[data-candidate]').first().getByRole('button', { name: /更多操作/ }).click()
    await page.locator('[data-candidate]').first().getByRole('button', { name: '价格', exact: true }).click()
    await page.getByRole('dialog', { name: '上游价格 · 示例' }).waitFor()
    await page.getByRole('dialog', { name: '上游价格 · 示例' }).getByRole('button', { name: '关闭', exact: true }).click()
    await page.getByRole('button', { name: '重置演示' }).click()
    const mobileHeight = (await page.locator('[data-candidate]').first().boundingBox()).height
    assert.ok(mobileHeight <= 240, `Mobile row must stay compact, received ${mobileHeight}px`)
    await page.locator('#prototype-tools > summary').click()

    await mkdir('/tmp/gpt-load-schedule-preview', { recursive: true })
    await page.evaluate(() => { document.querySelector('#toast').hidden = true })
    for (const [width, theme] of [[1440, 'light'], [900, 'light'], [390, 'light'], [390, 'dark']]) {
      await page.setViewportSize({ width, height: 1000 })
      await page.evaluate((theme) => { document.documentElement.dataset.theme = theme }, theme)
      await page.screenshot({ path: `/tmp/gpt-load-schedule-preview/${width}-${theme}.png`, fullPage: true })
    }
    await page.setViewportSize({ width: 1440, height: 1000 })
    await page.evaluate(() => { document.documentElement.dataset.theme = 'light' })
    await page.locator('#prototype-tools > summary').click()
    await page.getByRole('button', { name: '加载草稿示例' }).click()
    await page.locator('#prototype-tools > summary').click()
    await page.evaluate(() => { document.querySelector('#toast').hidden = true })
    await page.screenshot({ path: '/tmp/gpt-load-schedule-preview/1440-draft.png', fullPage: true })
    await page.locator('[data-model="claude-sonnet-4-5"]').first().click()
    await page.screenshot({ path: '/tmp/gpt-load-schedule-preview/switch-confirm.png', fullPage: true })
    await guard.getByRole('button', { name: '留在当前模型' }).click()
    await page.locator('#prototype-tools > summary').click()
    await page.getByRole('button', { name: '重置演示' }).click()
    assert.deepEqual(errors, [])
    console.log('Normal row heights:', JSON.stringify(measurements))
  } finally {
    await context.close()
    await browser.close()
  }
})
