import { expect, test } from '@playwright/test'

test.setTimeout(90000)

for (const { width, status } of [
  { width: 1280, status: 'blacklisted' },
  { width: 390, status: 'cooldown' },
  { width: 320, status: 'blacklisted' },
  { width: 390, status: 'disabled' },
  { width: 1280, status: 'available' },
]) {
  test(`single credential ${status} at ${width}px`, async ({ page }) => {
    page.on('pageerror', (error) => console.error(error.message))
    await page.setViewportSize({ width, height: 900 })
    await page.addInitScript(() => localStorage.setItem('gpt-load.auth-key', 'fixture-auth'))
    let credential: Record<string, unknown> | null = {
      credential_id: 11,
      secret_version: 1,
      mask: 'fixture-mask',

      effective_status: status,
      recent_success_count: 0,
      recent_failure_count: 1,
      consecutive_failure_count: 1,
      last_failure_category: 'invalid_key',
      last_failure_summary:
        width === 1280
          ? 'Invalid channel credential'
          : width === 320
            ? 'Upstream returned HTTP 401: credential-rejected-with-a-very-long-unbroken-diagnostic-token'
            : 'Upstream returned HTTP 401',
      last_status_code: 401,
      cooldown_until_ms: status === 'cooldown' ? 2000000000000 : null,
      recovery:
        status === 'cooldown'
          ? { mode: 'cooldown', automatic: true, at_ms: 2000000000000 }
          : status === 'blacklisted'
            ? { mode: 'scheduled_release', automatic: true, at_ms: null }
            : { mode: status === 'disabled' ? 'manual' : 'none', automatic: false, at_ms: null },
    }
    const requests: string[] = []
    let restoreAttempts = 0
    await page.route('**/api/**', async (route) => {
      const path = new URL(route.request().url()).pathname
      if (path.includes('/credential') && route.request().method() !== 'GET')
        expect(route.request().headers()['x-credential-id']).toBe('11')
      requests.push(`${route.request().method()} ${path}`)
      let data: unknown = null
      if (path === '/api/auth/session') data = { authenticated: true, principal_type: 'admin' }
      else if (path === '/api/groups/7')
        data = {
          id: 7,
          name: 'Fixture group',
          channel_id: 'openai',
          params: {},
          provider_url: null,

          service_status: 'unavailable',
          service_status_reason: 'no_models',
          credential_configured: credential !== null,
          credential_status: credential?.effective_status ?? null,
          model_count: 0,
        }
      else if (path === '/api/groups/7/credential/test')
        data = {
          outcome: 'passed',
          model: 'fixture-model',
          protocol: 'openai-completions',
          latency_ms: 1,
          reason: null,
          recovered: true,
          log_id: null,
          tested_at_ms: 1700000000000,
        }
      else if (path === '/api/groups/7/credential/restore') {
        restoreAttempts++
        if (width === 1280 && status === 'blacklisted' && restoreAttempts === 1) {
          await route.fulfill({ status: 500, json: { code: 'ERROR', message: 'failed' } })
          return
        }
        credential = {
          ...credential,
          effective_status: 'available',
          cooldown_until_ms: null,
          recovery: { mode: 'none', automatic: false, at_ms: null },
        }
        data = credential
      } else if (path === '/api/groups/7/credential') {
        if (route.request().method() === 'DELETE') credential = null
        if (route.request().method() === 'PUT') {
          expect(route.request().postDataJSON()).toEqual({ credential: 'fixture-new-value' })
          credential = {
            ...credential,
            secret_version: 2,
            mask: 'updated-mask',
            recent_success_count: 42,
            recent_failure_count: 0,
            consecutive_failure_count: 0,
            last_failure_category: 'ok',
            last_failure_summary: undefined,
            last_status_code: null,
          }
          data = credential
        } else data = { credential }
      } else if (path === '/api/channels') data = { items: [] }
      else {
        await route.fulfill({
          status: 404,
          json: { code: 'NOT_FOUND', message: 'fixture unavailable' },
        })
        return
      }
      await route.fulfill({ json: { code: 0, message: 'ok', data } })
    })
    await page.goto('/groups/7?tab=credentials', { waitUntil: 'commit' })
    await expect(page.getByRole('button', { name: 'Copy key' })).toContainText('fixture-mask', {
      timeout: 60000,
    })
    const record = page
      .locator('article')
      .filter({ has: page.getByRole('button', { name: 'Copy key' }) })
    await expect(
      record.getByText(
        {
          blacklisted: 'Blacklisted',
          cooldown: 'Cooldown',
          disabled: 'Disabled',
          available: 'Available',
        }[status]!,
        { exact: true },
      ),
    ).toBeVisible()
    const metrics = record.locator('dl')
    await expect(metrics.getByText('0 success · 1 failure', { exact: true })).toBeVisible()
    await expect(metrics.getByText('Consecutive failures', { exact: true })).toBeVisible()
    if (status === 'blacklisted')
      await expect(metrics.getByText('Returns after the local blacklist delay')).toBeVisible()
    if (status === 'disabled')
      await expect(metrics.getByText('Enable the group to check credential status')).toBeVisible()
    await expect(metrics).toHaveCSS('font-variant-numeric', 'tabular-nums')
    // Each label and value reads inline; the group can wrap at narrow widths.
    const pairs = metrics.locator(':scope > div')
    await expect(pairs).toHaveCount(4)
    for (const pair of await pairs.all()) {
      const term = await pair.locator('dt').boundingBox()
      const value = await pair.locator('dd').boundingBox()
      expect(term && value && Math.abs(term.y - value.y)).toBeLessThanOrEqual(3)
    }
    if (width === 1280) {
      const first = await pairs.first().boundingBox()
      const last = await pairs.last().boundingBox()
      expect(first && last && Math.abs(first.y - last.y)).toBeLessThanOrEqual(3)
    }
    await expect(page.getByRole('button', { name: 'Update key', exact: true })).toHaveAttribute(
      'data-variant',
      'secondary',
    )
    expect(await record.evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true)
    await expect(metrics).toBeVisible()
    await expect(
      record.getByRole('button', { name: /credential details|More actions/i }),
    ).toHaveCount(0)
    await expect(record.getByRole('button', { name: 'Test connection' })).toHaveCount(0)
    await expect(record.getByRole('button', { name: 'Delete', exact: true })).toHaveCount(0)
    if (width === 1280) {
      await expect(page.getByText('Invalid channel credential', { exact: true })).toHaveCount(1)
      await expect(
        page.getByText('Invalid channel credential · Invalid channel credential'),
      ).toHaveCount(0)
    } else {
      await expect(page.getByText('Upstream returned HTTP 401')).toBeVisible()
      if (width === 320) {
        await expect(metrics.getByText(/very-long-unbroken-diagnostic-token/)).toBeVisible()
        expect(await metrics.evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true)
      }
    }
    await page.screenshot({ path: test.info().outputPath(`problem-${width}.png`), fullPage: true })
    await expect(page.locator('input[type=checkbox]')).toHaveCount(0)
    const restore = record.getByRole('button', { name: 'Restore now', exact: true })
    if (status === 'blacklisted' || status === 'cooldown') {
      await expect(restore).toBeVisible()
      await restore.focus()
      await page.keyboard.press('Enter')
      await expect.poll(() => requests).toContain('POST /api/groups/7/credential/restore')
      if (width === 1280 && status === 'blacklisted') {
        await expect(page.getByText('Unable to restore the channel credential now')).toBeVisible()
        await expect(restore).toBeEnabled()
        await restore.click()
      }
      await expect(record.getByText('Available', { exact: true })).toBeVisible()
    }
    await expect(restore).toHaveCount(0)
    await expect(metrics.getByText('0 success · 1 failure', { exact: true })).toBeVisible()
    await page.getByRole('button', { name: 'Update key', exact: true }).click()
    await page.getByPlaceholder('Paste a new API key').fill('fixture-new-value')
    await page.getByRole('button', { name: 'Save key', exact: true }).click()
    await expect(page.getByRole('button', { name: 'Copy key' })).toContainText('updated-mask')
    await expect(metrics.getByText('42 success', { exact: true })).toBeVisible()
    await page.screenshot({
      path: test.info().outputPath(`credential-${width}.png`),
      fullPage: true,
    })
    expect(requests).not.toContain('DELETE /api/groups/7/credential')
    expect(requests).not.toContain('POST /api/groups/7/credential/test')
  })
}
