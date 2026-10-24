import { expect, test, type Locator, type Page } from '@playwright/test'

test.setTimeout(90_000)

async function openImport(page: Page): Promise<void> {
  await page.addInitScript(() => localStorage.setItem('gpt-load.auth-key', 'fixture-auth'))
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    const data =
      path === '/api/auth/session'
        ? { authenticated: true, principal_type: 'admin' }
        : path === '/api/channels'
          ? {
              items: [
                {
                  channel_id: 'openai',
                  name: 'OpenAI',
                  mark: 'OA',
                  icon: 'openai',
                  description: 'OpenAI official API',
                  search_terms: ['gpt'],
                  default_base_url: 'https://api.openai.com',
                  param_fields: [
                    {
                      key: 'base_url',
                      label: 'Base URL',
                      input_kind: 'url',
                      required: false,
                      sensitive: false,
                      default_value: null,
                    },
                  ],
                  credential_fields: [
                    {
                      key: 'api_key',
                      label: 'API Key',
                      input_kind: 'secret',
                      required: true,
                      sensitive: true,
                      default_value: null,
                    },
                  ],
                  connection: { type: 'api_key', credential_input: 'batch_text' },
                  capabilities: { model_discovery: false, outbound_proxy: false },
                  routes: [],
                  client_protocols: [],
                },
              ],
              total: 1,
            }
          : {}
    await route.fulfill({ json: { code: 0, message: 'ok', data } })
  })
  await page.goto('/import?mode=new', { waitUntil: 'commit' })
  await expect(page.locator('#channel-credentials')).toBeVisible({ timeout: 60_000 })
}

async function descriptions(input: Locator): Promise<string[]> {
  return input.evaluate((element) =>
    (element.getAttribute('aria-describedby') ?? '')
      .split(/\s+/u)
      .filter(Boolean)
      .map((id) => document.getElementById(id)?.textContent ?? ''),
  )
}

test('credential field exposes required, concise help and current error semantics without redundant notices', async ({
  page,
}) => {
  await openImport(page)
  const input = page.locator('#channel-credentials')
  await expect(input).toHaveAttribute('aria-required', 'true')
  expect((await descriptions(input)).join(' ')).toContain('Paste one API key on a single line')
  await expect(page.getByText(/Plaintext is never put in URLs/)).toHaveCount(0)
  await expect(page.getByText(/These are channel API keys/)).toHaveCount(0)
  await input.fill('{"api_key":')
  await expect(input).toHaveAttribute('aria-invalid', 'true')
  expect((await descriptions(input)).join(' ')).toContain(
    'Enter one valid JSON object, not an array.',
  )
  await input.fill('sk-fixture')
  await expect(input).not.toHaveAttribute('aria-invalid', 'true')
  expect((await descriptions(input)).join(' ')).not.toContain('Enter one valid JSON object')
  await expect(input).toHaveAccessibleName(/Key content/)
})

test('outbound proxy is collapsed inside More optional settings by default and desktop layout fits compactly', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await openImport(page)
  const proxySelect = page.locator('#import-group-proxy-mode')
  await expect(proxySelect).toBeHidden()
  const moreTrigger = page.getByText('More optional settings', { exact: true })
  await expect(moreTrigger).toBeVisible()
  await moreTrigger.click()
  await expect(proxySelect).toBeVisible()

  const nameBox = await page.getByRole('textbox', { name: /^Group name/ }).boundingBox()
  const providerBox = await page.getByRole('textbox', { name: /^Provider website/ }).boundingBox()
  expect(nameBox).not.toBeNull()
  expect(providerBox).not.toBeNull()
  expect(Math.abs(nameBox!.y - providerBox!.y)).toBeLessThanOrEqual(8)

  const saveBarBox = await page.locator('main footer').boundingBox()
  expect(saveBarBox).not.toBeNull()
  expect(saveBarBox!.y + saveBarBox!.height).toBeLessThanOrEqual(900)
})

test('credential field, section summary and footer agree on invalid input', async ({ page }) => {
  await openImport(page)
  const override = page.getByRole('switch', { name: 'Custom upstream URL' })
  await expect(override).toBeChecked()
  await page
    .getByRole('textbox', { name: 'Custom upstream URL' })
    .fill('https://api.example.test/v1')
  const input = page.locator('#channel-credentials')
  const section = page.getByRole('region', { name: 'Enter one API key required', exact: true })
  const footer = page.locator('main footer')
  for (const [value, message] of [
    ['sk-first\nsk-first', 'Keep only one credential, even if the lines are identical.'],
    ['sk-first\nsk-second', 'Keep only one credential, even if the lines are identical.'],
    ['{"api_key":', 'Enter one valid JSON object, not an array.'],
  ] as const) {
    await input.fill(value)
    await expect(footer).toContainText(message)
    await expect(section).toContainText(message)
    await expect(footer).not.toContainText('Enter one credential first')
    await expect(page.getByRole('button', { name: 'Create Group', exact: true })).toBeDisabled()
  }
  await input.fill('{\n  "api_key": "sk-fixture"\n}')
  await expect(section).toContainText('Paste one API key on a single line')
  await expect(page.getByRole('button', { name: 'Create Group', exact: true })).toBeEnabled()
  await expect(footer).toContainText('1 credentials')
})
