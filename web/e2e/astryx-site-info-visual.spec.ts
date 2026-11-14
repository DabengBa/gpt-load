import { expect, test, type Page } from '@playwright/test'

const site = {
  status: 'ok',
  site_root_url: 'https://site.example',
  site_name: 'Example Site',
  remaining_amount: 123.45,
  currency_symbol: '$',
  effective_group: 'long-group-name-without-spaces-abcdefghijklmnopqrstuvwxyz-0123456789',
  group_ratio: 1.25,
  fetched_at_ms: 1_700_000_000_000,
  last_error: '',
}
const base = {
  name: 'Visual acceptance group',
  channel_id: 'openai',
  params: {},
  provider_url: 'https://site.example',
}
const runtime = {
  first_byte_timeout: 30,
  request_timeout: 60,
  stream_idle_timeout: 30,
  blacklist_threshold: 3,
  header_rules: { set: {}, remove: [] },
  affinity_enabled: false,
  responses_websocket_enabled: false,
  responses_reasoning_status_filter_enabled: false,
}
const channel = {
  channel_id: 'openai',
  name: 'OpenAI',
  mark: 'OA',
  icon: 'openai',
  search_terms: ['openai'],
  description: '',
  default_base_url: 'https://api.openai.com',
  param_fields: [],
  credential_fields: [],
  connection: { credential_input: 'batch_text' },
  capabilities: { model_discovery: false, outbound_proxy: false },
  routes: [
    {
      client_protocol: 'openai-completions',
      operation: 'chat_completion',
      route_mode: 'native',
      model_dependent: false,
      possible_modes: [],
    },
  ],
  client_protocols: ['openai-completions'],
}

async function mock(
  page: Page,
  locale = 'en-US',
  longNames = false,
  providerUrl = base.provider_url,
  tokenConfigured = true,
) {
  await page.addInitScript((locale) => {
    localStorage.setItem('gpt-load.auth-key', 'visual-test')
    localStorage.setItem('gpt-load.locale', locale)
  }, locale)
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    let data: unknown = {}
    if (path === '/api/auth/session') data = { authenticated: true, principal_type: 'admin' }
    if (path === '/api/channels') data = { items: [channel], total: 1 }
    if (path === '/api/groups/options') data = []
    if (path === '/api/groups')
      data = {
        observed_at_ms: 1_700_000_000_000,
        summary: { total: 2, available: 2, unavailable: 0, disabled: 0 },
        items: [
          {
            ...base,
            id: 1,
            name: longNames ? 'production-us-east-primary' : base.name,
            status: 'available',
            model_count: 1,
            client_model_count: 1,
            credential_configured: true,
            credential_status: 'available',
            site_info: site,
          },
          {
            ...base,
            id: 2,
            name: longNames ? 'production-us-east-backup' : 'Second group',
            status: 'available',
            model_count: 1,
            client_model_count: 1,
            credential_configured: true,
            credential_status: 'available',
            site_info: { ...site, remaining_amount: null, group_ratio: 2, status: 'error' },
          },
        ],
        pagination: { page: 1, page_size: 100, total_items: 2, total_pages: 1 },
      }
    if (path === '/api/groups/1')
      data = {
        ...base,
        id: 1,
        service_status: 'available',
        service_status_reason: null,
        credential_configured: true,
        credential_status: 'available',
        model_count: 1,
      }
    if (path === '/api/groups/1/settings')
      data = {
        ...base,
        provider_url: providerUrl,
        enabled: true,
        overrides: {},
        effective: runtime,
        proxy: {
          configured_mode: 'inherit',
          effective_mode: 'direct',
          effective_source: 'default',
          has_auth: false,
        },
        site_access: {
          user_id: 'operator-123',
          token_configured: tokenConfigured,
          site_root_url: providerUrl === base.provider_url ? site.site_root_url : providerUrl,
          snapshot: tokenConfigured ? { ...site, announcement_count: 3 } : null,
        },
      }
    if (path === '/api/groups/1/models') data = { items: [], total: 0, pending: 0 }
    if (path === '/api/groups/1/credential') data = { credential: null }
    await route.fulfill({ json: { code: 0, message: 'ok', data } })
  })
}

async function noPageOverflow(page: Page) {
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    ),
  ).toBeLessThanOrEqual(1)
}

test('provider website validation is recoverable in creation and settings', async ({ page }) => {
  await mock(page)
  for (const path of ['/import?mode=new', '/groups/1']) {
    await page.goto(path)
    const field = page.getByRole('textbox', { name: /Provider website/ })
    if (path.startsWith('/import'))
      await page.getByRole('textbox', { name: /Credential content/ }).fill('test-key')
    const button = page.getByRole('button', {
      name: path.startsWith('/import') ? 'Create Group' : 'Save settings',
    })
    const message = page.locator('.astryx-field-status').filter({
      hasText:
        'Enter a valid HTTP or HTTPS provider website URL without credentials or a fragment.',
    })
    for (const invalid of [
      'https://site.example/#pricing',
      'https://user:pass@site.example',
      'ftp://site.example',
      'https://site.example/%zz',
      'https://site.example:65536',
      'https://site.example:99999',
    ]) {
      await field.fill(invalid)
      await expect(message).toBeVisible()
      await expect(button).toBeDisabled()
      await expect(field).toHaveValue(invalid)
      await expect(page.getByText('Site root: https://site.example', { exact: false })).toHaveCount(
        0,
      )
    }
    await field.fill('https://site.example/pricing?plan=pro')
    await expect(message).toHaveCount(0)
    await expect(button).toBeEnabled()
    await expect(page.getByText('https://site.example', { exact: false }).first()).toBeVisible()
  }
})

test('legacy provider port stays readable and does not block unrelated settings edits', async ({
  page,
}) => {
  await mock(page, 'en-US', false, 'https://site.example:99999')
  await page.goto('/groups/1')
  const field = page.getByRole('textbox', { name: /Provider website/ })
  await expect(field).toHaveValue('https://site.example:99999')
  await page.getByRole('textbox', { name: 'Name', exact: true }).fill('Updated group')
  const save = page.getByRole('button', { name: 'Save settings' })
  await expect(save).toBeEnabled()
  await field.fill('https://site.example:65536')
  await expect(save).toBeDisabled()
  await expect(field).toHaveAttribute('aria-invalid', 'true')
  await field.fill('https://site.example:99999')
  await expect(save).toBeEnabled()
})

for (const configured of [false, true]) {
  test(`legacy provider port blocks ${configured ? 'replacing' : 'adding'} a site token`, async ({
    page,
  }) => {
    await mock(page, 'en-US', false, 'https://site.example:99999', configured)
    await page.goto('/groups/1')
    const website = page.getByRole('textbox', { name: /Provider website/ })
    const token = page.getByLabel('Access token')
    const save = page.getByRole('button', { name: 'Save settings' })
    const error = page
      .locator('.astryx-field-status')
      .filter({ hasText: 'Enter a valid HTTP or HTTPS provider website URL' })
    await page.getByRole('textbox', { name: 'Name', exact: true }).fill('Updated group')
    await expect(save).toBeEnabled()
    await token.fill('new-secret')
    await expect(error).toBeVisible()
    await expect(website).toHaveAttribute('aria-invalid', 'true')
    await expect(save).toBeDisabled()
    await token.fill('')
    await expect(error).toHaveCount(0)
    await expect(save).toBeEnabled()
    if (configured) {
      await page.getByRole('button', { name: 'Clear token' }).click()
      await expect(error).toHaveCount(0)
      await expect(save).toBeEnabled()
      await page.getByRole('button', { name: /Undo/ }).click()
    }
    await token.fill('new-secret')
    await website.fill('https://site.example:8443')
    await expect(error).toHaveCount(0)
    await expect(save).toBeEnabled()
  })
}

test('legacy provider port validates changed site user ID only while a token remains configured', async ({
  page,
}) => {
  await mock(page, 'en-US', false, 'https://site.example:99999')
  await page.goto('/groups/1')
  const website = page.getByRole('textbox', { name: /Provider website/ })
  const userId = page.getByRole('textbox', { name: 'User ID' })
  const save = page.getByRole('button', { name: 'Save settings' })
  const error = page
    .locator('.astryx-field-status')
    .filter({ hasText: 'Enter a valid HTTP or HTTPS provider website URL' })
  await page.getByRole('textbox', { name: 'Name', exact: true }).fill('Updated group')
  await userId.fill('  operator-123  ')
  await expect(save).toBeEnabled()
  await expect(error).toHaveCount(0)
  await userId.fill('new-user')
  await expect(error).toBeVisible()
  await expect(website).toHaveAttribute('aria-invalid', 'true')
  await expect(save).toBeDisabled()
  await userId.fill('operator-123')
  await expect(error).toHaveCount(0)
  await expect(save).toBeEnabled()
  await userId.fill('new-user')
  await page.getByRole('button', { name: 'Clear token' }).click()
  await expect(error).toHaveCount(0)
  await expect(save).toBeEnabled()
  await page.getByRole('button', { name: /Undo/ }).click()
  await expect(error).toBeVisible()
  await website.fill('https://site.example:8443')
  await page.getByRole('switch', { name: /Confirm using the existing token/ }).click()
  await expect(error).toHaveCount(0)
  await expect(save).toBeEnabled()
})

test('legacy provider port allows changing site user ID without a configured token', async ({
  page,
}) => {
  await mock(page, 'en-US', false, 'https://site.example:99999', false)
  await page.goto('/groups/1')
  await page.getByRole('textbox', { name: 'User ID' }).fill('new-user')
  await expect(page.getByRole('button', { name: 'Save settings' })).toBeEnabled()
  await expect(page.getByRole('textbox', { name: /Provider website/ })).not.toHaveAttribute(
    'aria-invalid',
    'true',
  )
})

test('long common-prefix group names remain fully readable beside mobile metrics', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 900 })
  await mock(page, 'en-US', true)
  await page.goto('/groups')
  for (const name of ['production-us-east-primary', 'production-us-east-backup']) {
    const row = page.getByRole('row').filter({ hasText: name })
    const link = row.getByRole('link', { name: `View details for ${name}` }).first()
    for (const index of [3, 4]) {
      await row.getByRole('cell').nth(index).scrollIntoViewIfNeeded()
      const geometry = await link.evaluate((element) => {
        const range = document.createRange()
        range.selectNodeContents(element)
        const textRects = [...range.getClientRects()].map(({ left, right, top, bottom }) => ({
          left,
          right,
          top,
          bottom,
        }))
        const cell = element.closest('td')!.getBoundingClientRect()
        return {
          textRects,
          cell: { left: cell.left, right: cell.right, top: cell.top, bottom: cell.bottom },
          scrollWidth: element.scrollWidth,
          clientWidth: element.clientWidth,
        }
      })
      expect(geometry.scrollWidth).toBeLessThanOrEqual(geometry.clientWidth)
      expect(geometry.textRects.length).toBeGreaterThan(1)
      for (const rect of geometry.textRects) {
        expect(rect.left).toBeGreaterThanOrEqual(geometry.cell.left)
        expect(rect.right).toBeLessThanOrEqual(geometry.cell.right)
        expect(rect.top).toBeGreaterThanOrEqual(geometry.cell.top)
        expect(rect.bottom).toBeLessThanOrEqual(geometry.cell.bottom)
      }
    }
  }
  await page.screenshot({ path: '../.tmp/site-info/long-names-390.png', fullPage: true })
  await page.getByText('$123.45').evaluate((metric) => {
    const cell = metric.closest('td')!
    let scroller = cell.parentElement
    while (
      scroller &&
      (!['auto', 'scroll'].includes(getComputedStyle(scroller).overflowX) ||
        scroller.scrollWidth <= scroller.clientWidth)
    )
      scroller = scroller.parentElement
    if (!scroller) throw new Error('Table scroller not found')
    scroller.scrollLeft +=
      cell.getBoundingClientRect().left - scroller.getBoundingClientRect().left - 120
  })
  const balanceBox = await page.getByText('$123.45').boundingBox()
  const nameBox = await page
    .getByRole('link', { name: 'View details for production-us-east-primary' })
    .first()
    .boundingBox()
  expect(balanceBox!.x).toBeGreaterThan(nameBox!.x + nameBox!.width)
  await page.screenshot({ path: '../.tmp/site-info/long-names-balance-390.png', fullPage: true })
  const link = page
    .getByRole('link', { name: 'View details for production-us-east-primary' })
    .first()
  await link.focus()
  await expect(link).toBeFocused()
  await page.keyboard.press('Enter')
  await expect(page).toHaveURL(/\/groups\/1/)
  await page.goto('/groups')
  const manage = page.getByRole('button', {
    name: 'Manage credential for production-us-east-backup',
  })
  await manage.scrollIntoViewIfNeeded()
  await manage.click()
  await expect(page).toHaveURL(/\/groups\/2\?tab=credentials/)
})

test('mobile metrics retain group identity and expose ratio sync failure', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 900 })
  await mock(page)
  await page.goto('/groups')
  const row = page.getByRole('row').filter({ hasText: 'Second group' })
  await row.getByText('2.0x', { exact: true }).scrollIntoViewIfNeeded()
  await expect(
    row.getByRole('link', { name: 'View details for Second group' }).first(),
  ).toBeInViewport()
  await expect(row.getByText('Sync failed').last()).toBeInViewport()
  for (const name of ['Visual acceptance group', 'Second group']) {
    const groupRow = page.getByRole('row').filter({ hasText: name })
    const groupLink = groupRow.getByRole('link', { name: `View details for ${name}` }).first()
    for (const column of ['Account balance', 'API Key group ratio']) {
      const cell = groupRow.getByRole('cell').nth(column === 'Account balance' ? 3 : 4)
      await cell.scrollIntoViewIfNeeded()
      const boxes = await Promise.all([groupLink.boundingBox(), cell.boundingBox()])
      expect(boxes[0]).not.toBeNull()
      expect(boxes[1]).not.toBeNull()
      expect(boxes[0]!.x).toBeGreaterThanOrEqual(0)
      expect(boxes[0]!.x + boxes[0]!.width).toBeLessThanOrEqual(390)
      expect(boxes[1]!.x).toBeLessThan(390)
    }
  }
})

for (const [locale, status] of [
  ['en-US', 'Sync successful'],
  ['zh-CN', '同步成功'],
]) {
  test(`site status and date use ${locale}; modified links keep browser behavior`, async ({
    page,
  }) => {
    await mock(page, locale)
    await page.goto('/groups/1')
    const section = page.locator('#settings-general section[aria-label]')
    await expect(section).toContainText(status)

    // Browser context uses the host timezone; derive the expected date there as well.
    const browserDate = await page.evaluate(
      ({ locale, timestamp }) =>
        new Intl.DateTimeFormat(locale, { dateStyle: 'short', timeStyle: 'short' }).format(
          timestamp,
        ),
      { locale, timestamp: site.fetched_at_ms },
    )

    await expect(section).toContainText(browserDate)
    for (const modifier of ['ctrlKey', 'metaKey', 'shiftKey', 'altKey']) {
      expect(
        await section.locator('a').evaluate((el, modifier) => {
          let intercepted = false
          const stopNavigation = (event: Event) => {
            intercepted = event.defaultPrevented
            event.preventDefault()
          }
          document.addEventListener('click', stopNavigation, { once: true })
          const event = new MouseEvent('click', {
            bubbles: true,
            cancelable: true,
            [modifier]: true,
          })
          el.dispatchEvent(event)
          return intercepted
        }, modifier),
      ).toBe(false)
    }
  })
}

for (const width of [390, 1440]) {
  test(`site info visual paths ${width}`, async ({ page }) => {
    test.setTimeout(90_000)
    await page.setViewportSize({ width, height: 900 })
    await mock(page)
    await page.goto('/groups')
    await expect(
      page.getByRole('link', { name: 'View details for Visual acceptance group' }).first(),
    ).toBeVisible()
    await expect(page.getByText('$123.45')).toBeVisible()
    await expect(page.getByText('Groups are temporarily unavailable')).toHaveCount(0)
    await noPageOverflow(page)
    await page.screenshot({ path: `../.tmp/site-info/list-${width}.png`, fullPage: true })
    if (width === 390) {
      await page.getByText('$123.45').scrollIntoViewIfNeeded()
      await expect(page.getByText('$123.45')).toBeInViewport()
      await page.screenshot({ path: '../.tmp/site-info/list-390-scrolled.png', fullPage: true })
    }
    await page.goto('/groups/1')
    await expect(page.getByRole('heading', { name: 'Site access' })).toHaveCount(0)
    await expect(page.getByRole('region', { name: 'Site access' })).toBeVisible()
    await expect(page.getByText(/123\.45/)).toBeVisible()
    await expect(page.getByText(/long-group-name-without-spaces/)).toBeVisible()
    await expect(page.getByRole('link', { name: /Announcements: 3/ })).toBeVisible()
    await noPageOverflow(page)
    await page.getByRole('button', { name: 'Clear token' }).focus()
    await expect(page.getByRole('button', { name: 'Clear token' })).toBeFocused()
    await page.keyboard.press('Enter')
    await expect(page.getByRole('button', { name: 'Undo' })).toBeVisible()
    await page.keyboard.press('Enter')
    await expect(page.getByRole('button', { name: 'Clear token' })).toBeVisible()
    await page.getByRole('region', { name: 'Site access' }).scrollIntoViewIfNeeded()
    await page.screenshot({ path: `../.tmp/site-info/detail-${width}.png`, fullPage: true })
    await page.locator('#settings-general').screenshot({
      path: `../.tmp/site-info/site-access-${width}.png`,
    })
    await page.goto('/import?mode=new')
    await expect(page.getByText('Group name', { exact: false }).first()).toBeVisible()
    await page.getByRole('textbox', { name: /Provider website/ }).fill('https://site.example')
    await page.getByRole('textbox', { name: 'User ID' }).fill('operator-123')
    await page.getByLabel('Access token').fill('example-secret')
    await expect(page.getByLabel('Access token')).toHaveAttribute('type', 'password')
    await noPageOverflow(page)
    await page.screenshot({ path: `../.tmp/site-info/create-${width}.png`, fullPage: true })
  })
}
