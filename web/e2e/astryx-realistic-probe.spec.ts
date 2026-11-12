import { expect, test } from '@playwright/test'
import { mockGroupModels } from './fixtures/group-models-mock'

test('saved model exposes an actual request workspace', async ({ page }) => {
  await mockGroupModels(page, { models: [{ id: 'model-a' }] })
  await page.goto('/groups/1?tab=models')
  await expect(page.getByRole('checkbox', { name: 'Enable model model-a' })).toBeVisible()
  await page.getByRole('button', { name: 'Actual request for model-a' }).click()
  await expect(page.getByRole('dialog', { name: 'Actual request' })).toBeVisible()
})

test('log navigation stays disabled until an active request stops', async ({ page }) => {
  await mockGroupModels(page, { models: [{ id: 'model-a' }] })
  await page.addInitScript(() => {
    const nativeFetch = window.fetch.bind(window)
    window.fetch = (input, init) => {
      if (input !== '/v1/chat/completions') return nativeFetch(input, init)
      const source = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(
            new TextEncoder().encode(
              'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n',
            ),
          )
        },
      })
      return Promise.resolve(
        new Response(source, {
          headers: {
            'content-type': 'text/event-stream',
            'x-gptload-request-id': 'c9fd0246-b27a-4ef1-b6a6-d52442efcb6d',
          },
        }),
      )
    }
  })
  await page.goto('/groups/1?tab=models')
  await page.getByRole('button', { name: 'Actual request for model-a' }).click()
  const dialog = page.getByRole('dialog', { name: 'Actual request' })
  await dialog.getByLabel('Access key').fill('transient-key')
  await dialog.getByRole('checkbox', { name: 'Stream' }).check()
  await dialog.getByRole('button', { name: 'Send request' }).click()
  const log = dialog.getByRole('button', { name: 'View routing and logs' })
  await expect(dialog.getByText('Running', { exact: true })).toBeVisible()
  await expect(log).toBeDisabled()
  await dialog.getByRole('button', { name: 'Stop', exact: true }).click()
  await expect(dialog.getByText('Canceled', { exact: true })).toBeVisible()
  await expect(log).toBeEnabled()
})

test('closing clears the key and a late request cannot replace a new result', async ({ page }) => {
  await mockGroupModels(page, {
    models: [{ id: 'model-a', alias: 'client-a', alias_enabled: true }],
  })
  let releaseOld: (() => void) | undefined
  let oldStarted: (() => void) | undefined
  const started = new Promise<void>((resolve) => {
    oldStarted = resolve
  })
  const hold = new Promise<void>((resolve) => {
    releaseOld = resolve
  })
  let count = 0
  await page.route('**/v1/chat/completions', async (route) => {
    const request = route.request()
    expect(request.headers().authorization).toBe('Bearer transient-key')
    expect(request.postDataJSON().model).toBe('client-a')
    const current = ++count
    if (current === 1) {
      oldStarted?.()
      await hold
    }
    await route
      .fulfill({
        status: 200,
        headers: {
          'content-type': 'application/json',
          'x-gptload-request-id': 'c9fd0246-b27a-4ef1-b6a6-d52442efcb6d',
        },
        body: JSON.stringify({
          object: 'chat.completion',
          choices: [
            {
              finish_reason: 'stop',
              message: {
                role: 'assistant',
                content: current === 1 ? 'old output' : 'new output',
              },
            },
          ],
        }),
      })
      .catch(() => {})
  })
  await page.goto('/groups/1?tab=models')
  const entry = page.getByRole('button', { name: 'Actual request for model-a' })
  await entry.click()
  const dialog = page.getByRole('dialog', { name: 'Actual request' })
  await dialog.getByLabel('Access key').fill('transient-key')
  await dialog.getByRole('button', { name: 'Send request' }).click()
  await started
  await dialog.getByRole('button', { name: 'Close', exact: true }).last().click()
  await entry.click()
  await expect(dialog.getByLabel('Access key')).toHaveValue('')
  await dialog.getByLabel('Access key').fill('transient-key')
  await dialog.getByRole('button', { name: 'Send request' }).click()
  await expect(dialog.getByText('Complete', { exact: true })).toBeVisible()
  await expect(dialog.getByText('Terminal reason: stop')).toBeVisible()
  await expect(dialog.getByText('Response preview')).toBeVisible()
  await expect(dialog.getByText(/new output/)).toBeVisible()
  releaseOld?.()
  await expect(dialog.getByText(/new output/)).toBeVisible()
  await expect(dialog.getByText(/old output/)).toHaveCount(0)
  await dialog.getByRole('button', { name: 'View routing and logs' }).click()
  await expect(page).toHaveURL(/selected_request_id=c9fd0246-b27a-4ef1-b6a6-d52442efcb6d/)
})

test('editing clears results, protocol replacement confirms, and keyboard and layouts work', async ({
  page,
}, testInfo) => {
  const longModel = `model-${'long'.repeat(40)}`
  await mockGroupModels(page, { models: [{ id: longModel }] })
  await page.route('**/v1/chat/completions', (route) =>
    route.fulfill({
      contentType: 'text/event-stream',
      headers: { 'x-gptload-request-id': 'c9fd0246-b27a-4ef1-b6a6-d52442efcb6d' },
      body: 'data: {"choices":[{"delta":{"content":"done"},"finish_reason":null}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
    }),
  )
  await page.goto('/groups/1?tab=models')
  await page.getByRole('button', { name: `Actual request for ${longModel}` }).click()
  const dialog = page.getByRole('dialog', { name: 'Actual request' })
  await dialog.getByLabel('Access key').fill('key')
  await dialog.getByRole('checkbox', { name: 'Stream' }).check()
  await dialog.getByRole('button', { name: 'Send request' }).click()
  await expect(dialog.getByText('Complete', { exact: true })).toBeVisible()
  await expect(dialog.getByText('Terminal reason: stop')).toBeVisible()
  for (const width of [390, 1280]) {
    await page.setViewportSize({ width, height: 900 })
    await dialog.getByText('Terminal reason: stop').scrollIntoViewIfNeeded()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
    expect(await dialog.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true)
    await page.screenshot({ path: testInfo.outputPath(`terminal-${width}.png`) })
  }
  await dialog.getByLabel('Access key').fill('new-key')
  await expect(dialog.getByText('Complete', { exact: true })).toHaveCount(0)
  const json = dialog.getByLabel('Request JSON')
  await json.fill('{"model":"edited","messages":[]}')
  await dialog.getByLabel('Protocol').click()
  await page.getByRole('option', { name: 'openai-responses' }).click()
  await page.getByRole('alertdialog').getByRole('button', { name: 'Cancel', exact: true }).click()
  await expect(json).toHaveValue('{"model":"edited","messages":[]}')
  await dialog.getByLabel('Protocol').click()
  await page.getByRole('option', { name: 'openai-responses' }).click()
  await page.getByRole('alertdialog').getByRole('button', { name: 'Replace', exact: true }).click()
  await expect(json).toContainText('"input"')
  await json.focus()
  for (let i = 0; i < 16; i++) {
    await page.keyboard.press('Tab')
    const focus = await dialog.evaluate(
      (el) => el.contains(document.activeElement) || document.activeElement === document.body,
    )
    expect(focus).toBe(true)
  }
  await json.focus()
  // Tabbing over the header close button can open its tooltip. Escape dismisses
  // the top overlay, so wait for that tooltip's blur dismissal before the dialog.
  await expect(page.locator('[popover]:popover-open')).toHaveCount(0)
  await expect(page.getByRole('alertdialog')).toHaveCount(0)
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
})

test('group template follows its supported generation protocol', async ({ page }) => {
  await mockGroupModels(page, { models: [{ id: 'model-a' }], generationProtocol: 'anthropic' })
  await page.goto('/groups/1?tab=models')
  await page.getByRole('button', { name: 'Actual request for model-a' }).click()
  const dialog = page.getByRole('dialog', { name: 'Actual request' })
  await expect(dialog.getByText('/v1/messages', { exact: true })).toBeVisible()
  await expect(dialog.getByLabel('Request JSON')).toContainText('"max_tokens"')
})

test('schedule opens the workspace with its external model', async ({ page }) => {
  await mockGroupModels(page)
  await page.route('**/api/model-route/schedule*', async (route) => {
    const path = new URL(route.request().url()).pathname
    const data = path.endsWith('/detail')
      ? {
          observed_at_ms: 1,
          snapshot_revision: 1,
          external_model: 'worker',
          protocol: 'openai-responses',
          operation: 'responses_create',
          route_requirement: 'any',
          access_key: null,
          routable: false,
          reason_code: 'no_available_group',
          groups: [],
        }
      : {
          items: [
            {
              external_model: 'worker',
              protocol: 'openai-responses',
              operation: 'responses_create',
              candidate_count: 0,
              group_count: 0,
              cooled_candidates: 0,
              blacklisted_candidates: 0,
            },
          ],
        }
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ code: 0, message: 'ok', data }),
    })
  })
  await page.goto('/schedule?schedule_model=worker')
  await page.getByRole('button', { name: 'Actual request', exact: true }).click()
  const dialog = page.getByRole('dialog', { name: 'Actual request' })
  await expect(dialog.getByLabel('Request JSON')).toContainText('worker')
  await expect(dialog.getByLabel('Request JSON')).toContainText('"input"')
  await expect(dialog.getByText('/v1/responses', { exact: true })).toBeVisible()
  await expect(dialog.getByLabel('Access key')).toHaveValue('')
})

test('stream partial delivery is incomplete and an in-flight request can be stopped', async ({
  page,
}) => {
  await mockGroupModels(page, { models: [{ id: 'model-a' }] })
  let release: (() => void) | undefined
  let started: (() => void) | undefined
  const hold = new Promise<void>((resolve) => {
    release = resolve
  })
  const waiting = new Promise<void>((resolve) => {
    started = resolve
  })
  let requests = 0
  await page.route('**/v1/chat/completions', async (route) => {
    expect(route.request().postDataJSON().stream).toBe(true)
    if (++requests === 2) {
      started?.()
      await hold
    }
    await route
      .fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: 'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n',
      })
      .catch(() => {})
  })
  await page.goto('/groups/1?tab=models')
  await page.getByRole('button', { name: 'Actual request for model-a' }).click()
  const dialog = page.getByRole('dialog', { name: 'Actual request' })
  await dialog.getByLabel('Access key').fill('transient-key')
  await dialog.getByRole('checkbox', { name: 'Stream' }).check()
  await dialog.getByRole('button', { name: 'Send request' }).click()
  await expect(dialog.getByText('Incomplete', { exact: true })).toBeVisible()
  await expect(dialog.locator('pre')).toContainText('partial')
  await dialog.getByRole('button', { name: 'Send request' }).click()
  await waiting
  await dialog.getByRole('button', { name: 'Stop', exact: true }).click()
  await expect(dialog.getByText('Canceled', { exact: true })).toBeVisible()
  release?.()
})
