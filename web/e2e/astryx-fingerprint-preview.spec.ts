import { expect, test } from '@playwright/test'
import { fingerprintSuccess, mockGroupFingerprint } from './fixtures/group-fingerprint-mock'

for (const viewport of [
  { name: 'desktop', width: 1440, height: 1100 },
  { name: 'mobile', width: 390, height: 844 },
]) {
  test(`capture actual fingerprint dialog: ${viewport.name}`, async ({ page }) => {
    await page.setViewportSize(viewport)
    await mockGroupFingerprint(page)
    await page.addInitScript(() => localStorage.setItem('gpt-load.locale', 'zh-CN'))
    let unavailable = false
    await page.route('**/api/modeltrace-probe', async (route) => {
      const result = fingerprintSuccess()
      result.group_name = 'OpenAI · 演示分组'
      result.analysis.candidates[1]!.model =
        'gpt-5.6-sol-long-model-name-for-responsive-wrapping-verification'
      result.challenges = [1, 2, 3].map((index) => ({
        ...result.challenges[0]!,
        id: `probe-${index}-demo`,
        expected_count: 300,
        latency_ms: 12000 + index * 420,
      }))
      result.analysis.used_outputs = 3
      result.analysis.candidates[1]!.probability = 0.088
      if (unavailable) {
        result.challenges[1] = {
          ...result.challenges[1]!,
          outcome: 'failed',
          reason: 'timeout',
          status_code: null,
        }
      }
      const data = unavailable
        ? { ...result, analysis: null, failure: 'analysis_unavailable' }
        : result
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ code: 0, message: 'ok', data }),
      })
    })
    await page.goto('/groups/1?tab=models')
    const button = page.getByRole('button', { name: '指纹检测', exact: true })
    await expect(button).toBeEnabled({ timeout: 60000 })

    await button.click()
    const dialog = page.getByRole('dialog')
    await expect(dialog.getByText('91.2%', { exact: true }).first()).toBeVisible()
    await dialog.screenshot({
      path: `../tmp/demos/fingerprint-polish-${viewport.name}-success.png`,
    })
    expect(await dialog.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true)
    await dialog.getByRole('button', { name: '关闭', exact: true }).last().click()
    unavailable = true
    await button.click()
    await expect(dialog.getByText(/检测服务无法分析/)).toBeVisible()
    await dialog.screenshot({
      path: `../tmp/demos/fingerprint-polish-${viewport.name}-failure.png`,
    })
    expect(await dialog.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true)
  })
}
