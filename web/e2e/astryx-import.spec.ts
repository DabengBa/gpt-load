import { expect, test, type Page, type Route } from '@playwright/test'

// Phase 4: the React /import surface — mode switching, group_id deep links,
// stable idempotent submission and the
// sessionStorage re-auth recovery chain (401 -> capture -> login -> restore).

// The first navigation in a cold run pays vite's on-demand compile of the
// astryx import graph; keep generous headroom over the usual 90s budget.
test.setTimeout(120_000)
const FIRST_PAINT = 90_000

const AUTH_SESSION = { authenticated: true, principal_type: 'admin' }

const CHANNELS = [
  {
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
    capabilities: {
      model_discovery: false,

      outbound_proxy: false,
    },
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
  },
  {
    channel_id: 'cliproxyapi',
    name: 'CLIProxyAPI',
    mark: 'CPA',
    icon: 'cpa',
    search_terms: ['cpa'],
    description: '',
    default_base_url: '',

    param_fields: [],
    credential_fields: [],
    connection: { credential_input: 'batch_text' },
    capabilities: {
      model_discovery: false,

      outbound_proxy: false,
    },
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
  },
]

const PICKER_CHANNELS = [
  ['openai', 'OpenAI'],
  ['anthropic', 'Anthropic'],
  ['gemini', 'Gemini'],
  ['openai_compatible', 'OpenAI Compatible'],
  ['cliproxyapi', 'CLIProxyAPI'],
  ['deepseek', 'DeepSeek'],
  ['groq', 'Groq'],
  ['mistral', 'Mistral'],
  ['extra', 'Extra Channel'],
].map(([channel_id, name]) => ({
  ...CHANNELS[0],
  channel_id,
  name,
  icon: channel_id,
  param_fields: [
    {
      key: 'base_url',
      label: 'Base URL',
      input_kind: 'url',
      required: channel_id === 'openai_compatible',
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
}))

const SIMPLIFIED_CHANNELS = [
  ...PICKER_CHANNELS,
  ...[
    ['multi_protocol', 'Multi-Protocol'],
    ['azure_openai', 'Azure OpenAI'],
    ['aws_bedrock', 'AWS Bedrock'],
    ['google_vertex', 'Google Vertex'],
  ].map(([channel_id, name]) => ({
    ...PICKER_CHANNELS[0]!,
    channel_id,
    name,
    ...(channel_id === 'multi_protocol'
      ? {
          default_base_url: '',
          param_fields: [{ ...PICKER_CHANNELS[0]!.param_fields[0]!, required: true }],
        }
      : {}),
  })),
]

const GROUP_OPTIONS = [
  {
    id: 7,
    name: 'Alpha Group',
    channel_id: 'openai',
    params: {},
    provider_url: null,
    enabled: true,
    models: ['gpt-4o'],
  },
  {
    id: 9,
    name: 'Beta Gateway',
    channel_id: 'cliproxyapi',
    params: {},
    provider_url: null,
    enabled: true,
    models: [],
  },
]

const RUNTIME = {
  first_byte_timeout: 30,
  request_timeout: 60,
  stream_idle_timeout: 30,
  blacklist_threshold: 3,
  header_rules: { set: {}, remove: [] },
  affinity_enabled: false,
  responses_websocket_enabled: false,
  responses_reasoning_status_filter_enabled: false,
}

const GROUP_42 = {
  summary: {
    id: 42,
    name: 'Imported Group',

    channel_id: 'openai',
    params: {},
    provider_url: null,
    service_status: 'available',
    service_status_reason: null,
    credential_configured: true,
    credential_status: 'available',
    model_count: 0,
  },
  settings: {
    name: 'Imported Group',

    channel_id: 'openai',
    params: {},
    provider_url: null,
    enabled: true,
    site_access: {
      user_id: '',
      token_configured: false,
      site_root_url: '',
      snapshot: null,
    },
    overrides: { ...RUNTIME, parameter_overrides: [] },
    effective: RUNTIME,
    proxy: {
      configured_mode: 'inherit',
      effective_mode: 'direct',
      effective_source: 'default',
      has_auth: false,
    },
  },
  models: { items: [], total: 0, pending: 0 },
}

interface ImportRequests {
  createBodies: Record<string, unknown>[]
  createIdempotencyKeys: string[]
  credentialImports: { groupID: string; body: Record<string, unknown>; key: string }[]
}

async function mockImportApi(
  page: Page,
  options: {
    createStatus?: number
    credentialCount?: number
  } = {},
): Promise<ImportRequests> {
  const requests: ImportRequests = {
    createBodies: [],
    createIdempotencyKeys: [],
    credentialImports: [],
  }
  const ok = (data: unknown) => ({ code: 0, message: 'ok', data })
  await page.addInitScript((key) => {
    window.localStorage.setItem('gpt-load.auth-key', key)
  }, 'e2e-auth-key')
  await page.route('**/api/**', async (route: Route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    const fulfill = (data: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify(ok(data)),
      })
    if (path === '/api/auth/session') return fulfill(AUTH_SESSION)
    if (path === '/api/channels') return fulfill({ items: CHANNELS, total: CHANNELS.length })
    if (path === '/api/groups/options') return fulfill(GROUP_OPTIONS)
    if (path === '/api/groups/7')
      return fulfill({
        ...GROUP_42.summary,
        id: 7,
        name: 'Alpha Group',
        credential_configured: (options.credentialCount ?? 0) > 0,
        credential_status: (options.credentialCount ?? 0) > 0 ? 'available' : null,
      })
    if (path === '/api/groups' && request.method() === 'POST') {
      requests.createBodies.push((request.postDataJSON() ?? {}) as Record<string, unknown>)
      requests.createIdempotencyKeys.push(request.headers()['idempotency-key'] ?? '')
      if (options.createStatus === 401) {
        return route.fulfill({
          status: 401,
          contentType: 'application/json',
          body: JSON.stringify({ code: 401, message: 'unauthorized', data: null }),
        })
      }
      return fulfill({
        group_id: 42,
        group_name: 'Imported Group',
        credential_id: 9,
      })
    }
    const credentialImport = path.match(/^\/api\/groups\/(\d+)\/credential$/)
    if (credentialImport && request.method() === 'POST') {
      requests.credentialImports.push({
        groupID: credentialImport[1]!,
        body: (request.postDataJSON() ?? {}) as Record<string, unknown>,
        key: request.headers()['idempotency-key'] ?? '',
      })
      return fulfill({
        group_id: Number(credentialImport[1]),
        credential_id: 9,
      })
    }
    if (path === '/api/groups/42') return fulfill(GROUP_42.summary)
    if (path === '/api/groups/42/settings') return fulfill(GROUP_42.settings)
    if (path === '/api/groups/42/models') return fulfill(GROUP_42.models)
    return fulfill({})
  })
  return requests
}

for (const {
  locale,
  width,
  label,
  explanation,
  urlHelp,
  createLabel,
  moreLabel,
  searchLabel,
  autoName,
  warning,
  explicitName,
} of [
  {
    locale: 'en-US',
    width: 1440,
    label: 'Follow client protocol',
    explanation:
      'For declared operations, the next hop follows the client protocol. The final provider protocol depends on the gateway. The default health test uses Chat; a single result only represents the operation performed.',
    urlHelp:
      'Enter the gateway root or deployment prefix, for example https://gateway.example.com. Protocol paths are appended automatically.',
    createLabel: 'Create Group',
    moreLabel: 'Other channels',
    searchLabel: 'Search channels',
    autoName:
      'The name is generated from the upstream hostname or channel name; a suffix may be added if needed.',
    warning:
      'Protocol paths are appended to this base URL. If this is an API version path rather than a deployment prefix, requests may contain /v1/v1 or /v1beta/v1beta. Verify the gateway configuration.',
    explicitName: 'Creates “My Gateway”; the name can be changed later',
  },
  {
    locale: 'zh-CN',
    width: 390,
    label: '随客户端协议',
    explanation:
      '对已声明的操作，下一跳随客户端协议；最终供应商协议取决于网关。默认测活使用 Chat，单次结果仅代表实际执行的操作。',
    urlHelp: '填写网关根地址或部署前缀，例如 https://gateway.example.com；系统会自动追加协议路径',
    createLabel: '创建分组',
    moreLabel: '其他渠道',
    searchLabel: '搜索渠道',
    autoName: '名称由上游主机名或渠道名称自动生成；必要时会添加后缀。',
    warning:
      '系统会在此基础地址后追加协议路径。如果这不是部署前缀而是 API 版本路径，请求可能出现 /v1/v1 或 /v1beta/v1beta，请确认网关配置。',
    explicitName: '将创建分组“My Gateway”，之后可随时改名',
  },
]) {
  test(`simplified new channels and gateway boundary in ${locale}`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 })
    await page.addInitScript((value) => localStorage.setItem('gpt-load.locale', value), locale)
    const requests = await mockImportApi(page)
    await page.route('**/api/channels', (route) =>
      route.fulfill({
        json: {
          code: 0,
          message: 'ok',
          data: { items: SIMPLIFIED_CHANNELS, total: SIMPLIFIED_CHANNELS.length },
        },
      }),
    )
    await page.goto('/import?mode=new', { waitUntil: 'commit' })
    await expect(
      page.getByRole('button', { name: 'OpenAI Compatible', exact: true }),
    ).toHaveAttribute('aria-pressed', 'true', { timeout: FIRST_PAINT })
    for (const name of ['OpenAI', 'Anthropic', 'Gemini']) {
      await expect(page.getByRole('button', { name, exact: true })).toBeVisible()
    }
    await page.getByRole('button', { name: label, exact: true }).click()

    await expect(page.getByText(explanation, { exact: true })).toBeVisible()
    await expect(page.getByText(urlHelp, { exact: true })).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    )
    await page.locator('#channel-credentials').fill('fixture-key')
    const address = page.locator('input[inputmode="url"]').last()
    const create = page.getByRole('button', { name: createLabel, exact: true })
    await expect(address).toHaveValue('')
    await expect(address).toHaveAttribute('aria-required', 'true')
    await expect(create).toBeDisabled()
    expect(requests.createBodies).toHaveLength(0)
    await address.fill('https://gateway.example.test/v1')
    await expect(page.getByTestId('astryx-shell').getByText(warning, { exact: true })).toBeVisible()
    await address.fill('https://gateway.example.test/team/v1/')
    await expect(page.getByTestId('astryx-shell').getByText(warning, { exact: true })).toBeVisible()
    await expect(create).toBeEnabled()
    await address.fill('https://gateway.example.test/team/v1beta')
    await expect(page.getByTestId('astryx-shell').getByText(warning, { exact: true })).toBeVisible()
    await address.fill('https://gateway.example.test/team')
    await expect(page.getByTestId('astryx-shell').getByText(warning, { exact: true })).toHaveCount(
      0,
    )
    await address.fill('https://gateway.example.test/v1/team')
    await expect(page.getByTestId('astryx-shell').getByText(warning, { exact: true })).toHaveCount(
      0,
    )
    await address.fill('https://gateway.example.test')
    await expect(page.getByText(autoName, { exact: true })).toBeVisible()
    await expect(page.getByText(`${label} (Multi-Protocol)`, { exact: true })).toBeVisible()
    await page.getByLabel(locale === 'zh-CN' ? '分组名称' : 'Group name').fill('My Gateway')
    await expect(page.getByText(explicitName, { exact: true })).toBeVisible()
    await page.getByLabel(locale === 'zh-CN' ? '分组名称' : 'Group name').clear()
    await address.fill('https://gateway.example.test/team/v1/')

    await create.click()
    await expect.poll(() => requests.createBodies.length).toBe(1)
    expect(requests.createBodies[0]).toMatchObject({
      channel_id: 'multi_protocol',
      params: { base_url: 'https://gateway.example.test/team/v1/' },
      credential: 'fixture-key',
    })
    expect(requests.createBodies[0]).not.toHaveProperty('name')
  })

  test(`gateway original name search in ${locale}`, async ({ page }) => {
    await page.addInitScript((value) => localStorage.setItem('gpt-load.locale', value), locale)
    await mockImportApi(page)
    await page.route('**/api/channels', (route) =>
      route.fulfill({
        json: {
          code: 0,
          message: 'ok',
          data: { items: SIMPLIFIED_CHANNELS, total: SIMPLIFIED_CHANNELS.length },
        },
      }),
    )
    await page.goto('/import?mode=new', { waitUntil: 'commit' })
    await page.getByRole('button', { name: moreLabel, exact: true }).click({ timeout: FIRST_PAINT })
    const search = page.getByRole('combobox', { name: new RegExp(searchLabel) })
    await search.fill('Multi-Protocol')
    await expect(page.getByRole('option').filter({ hasText: label })).toBeVisible()
    await search.press('Enter')
    await expect(page.getByRole('button', { name: label, exact: true })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
  })
}

test('special authentication and legacy presets remain reachable with keyboard search', async ({
  page,
}) => {
  await mockImportApi(page)
  await page.route('**/api/channels', (route) =>
    route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data: { items: SIMPLIFIED_CHANNELS, total: SIMPLIFIED_CHANNELS.length },
      },
    }),
  )
  await page.goto('/import?mode=new', { waitUntil: 'commit' })
  await page
    .getByRole('button', { name: 'Other channels', exact: true })
    .click({ timeout: FIRST_PAINT })
  await expect(
    page
      .getByRole('option')
      .locator('span')
      .filter({ hasText: /^(Azure OpenAI|AWS Bedrock|Google Vertex)$/ }),
  ).toHaveText(['Azure OpenAI', 'AWS Bedrock', 'Google Vertex'])
  await page.keyboard.press('Escape')
  for (const name of ['Azure OpenAI', 'AWS Bedrock', 'Google Vertex', 'DeepSeek']) {
    const trigger = page
      .getByRole('button', {
        name: /Other channels|Azure OpenAI|AWS Bedrock|Google Vertex|DeepSeek/,
      })
      .last()
    await trigger.click({ timeout: FIRST_PAINT })
    const search = page.getByRole('combobox', { name: /Search channels/ })
    await search.fill(name)
    await search.press('Enter')
    await expect(page.getByRole('button', { name, exact: true })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
  }
})

test('fresh compatible default submits its own address and key', async ({ page }) => {
  const requests = await mockImportApi(page)
  await page.route('**/api/channels', (route) =>
    route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data: { items: SIMPLIFIED_CHANNELS, total: SIMPLIFIED_CHANNELS.length },
      },
    }),
  )
  await page.goto('/import?mode=new', { waitUntil: 'commit' })
  await expect(
    page.getByRole('button', { name: 'OpenAI Compatible', exact: true }),
  ).toHaveAttribute('aria-pressed', 'true', { timeout: FIRST_PAINT })
  await page
    .getByRole('textbox', { name: /^Upstream base URL/ })
    .fill('https://relay.example.test/v1')
  await page.locator('#channel-credentials').fill('fixture-key')
  await page.getByRole('button', { name: 'Create Group', exact: true }).click()
  await expect.poll(() => requests.createBodies.length).toBe(1)
  expect(requests.createBodies[0]).toMatchObject({
    channel_id: 'openai_compatible',
    params: { base_url: 'https://relay.example.test/v1' },
    credential: 'fixture-key',
  })
})

test('failed channel loading blocks creation and retry recovers the default', async ({ page }) => {
  const requests = await mockImportApi(page)
  let attempts = 0
  let releaseFirstRequest: () => void = () => {}
  const waitForRelease = new Promise<void>((resolve) => {
    releaseFirstRequest = resolve
  })
  await page.route('**/api/channels', async (route) => {
    attempts += 1
    if (attempts === 1) await waitForRelease
    return attempts === 1
      ? route.fulfill({ status: 500, json: { code: 500, message: 'failure', data: null } })
      : route.fulfill({
          json: {
            code: 0,
            message: 'ok',
            data: { items: SIMPLIFIED_CHANNELS, total: SIMPLIFIED_CHANNELS.length },
          },
        })
  })
  await page.goto('/import?mode=new', { waitUntil: 'commit' })
  await expect(page.getByText('Loading channels…')).toBeVisible({ timeout: FIRST_PAINT })
  await expect(page.getByRole('button', { name: 'Create Group', exact: true })).toBeDisabled()
  releaseFirstRequest()
  await expect(page.getByText('Unable to load channels')).toBeVisible({ timeout: FIRST_PAINT })
  await expect(page.getByRole('button', { name: 'Create Group', exact: true })).toBeDisabled()
  expect(requests.createBodies).toHaveLength(0)
  await page.getByRole('button', { name: 'Retry', exact: true }).click()
  await expect(
    page.getByRole('button', { name: 'OpenAI Compatible', exact: true }),
  ).toHaveAttribute('aria-pressed', 'true')
})

for (const width of [390, 1440]) {
  test(`import tasks have distinct responsive surfaces at ${width}px`, async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width, height: 900 })
    await mockImportApi(page)
    await page.goto('/import?mode=new', { waitUntil: 'load' })
    const steps = page.locator(
      '#import-channel-step-heading, #import-credentials-step-heading, #import-models-heading',
    )
    await expect(steps).toHaveCount(3)
    const panelCount = await steps.evaluateAll(
      (headings) =>
        new Set(headings.map((heading) => heading.closest('[data-content-panel]')).filter(Boolean))
          .size,
    )
    expect(panelCount).toBe(3)
    await expect(page.locator('[data-content-panel] [data-content-panel]')).toHaveCount(0)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    )
    await page.screenshot({ path: testInfo.outputPath(`import-new-${width}.png`), fullPage: true })
    await page.goto('/import?mode=existing&group_id=7', { waitUntil: 'load' })
    await expect(page.locator('#existing-target-heading')).toBeVisible()
    await expect(
      page
        .locator('[data-content-panel]')
        .filter({ has: page.locator('#existing-target-heading') }),
    ).toHaveCount(1)
    await expect(page.locator('[data-content-panel] [data-content-panel]')).toHaveCount(0)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    )
    await page.screenshot({
      path: testInfo.outputPath(`import-existing-${width}.png`),
      fullPage: true,
    })
  })
}

test('API-only picker retains external CPA without subscription authorization controls', async ({
  page,
}) => {
  await mockImportApi(page)
  await page.goto('/import', { waitUntil: 'commit' })
  const externalGateway = page.getByRole('button', { name: 'CLIProxyAPI', exact: true })
  await expect(externalGateway).toBeVisible({ timeout: FIRST_PAINT })
  await externalGateway.click()
  await expect(externalGateway).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByRole('radio', { name: 'Subscription account' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: /Sign in with/ })).toHaveCount(0)
})

test('visible group name label focuses its input', async ({ page }) => {
  await mockImportApi(page)
  await page.goto('/import?mode=new', { waitUntil: 'commit' })
  const label = page
    .locator('label')
    .filter({ hasText: /^Group name/ })
    .filter({ visible: true })
  await expect(label).toBeVisible({ timeout: FIRST_PAINT })
  await label.click()
  await expect(page.locator(`input[id="${await label.getAttribute('for')}"]`)).toBeFocused()
})

test('optional OpenAI address can be omitted and required compatible address blocks creation', async ({
  page,
}) => {
  const requests = await mockImportApi(page)
  await page.route('**/api/channels', (route) =>
    route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data: { items: PICKER_CHANNELS, total: PICKER_CHANNELS.length },
      },
    }),
  )
  await page.goto('/import?mode=new', { waitUntil: 'commit' })
  const address = page.getByRole('textbox', { name: /^Upstream base URL/ })
  await expect(address).toBeVisible({ timeout: FIRST_PAINT })
  await expect(page.getByRole('switch', { name: 'Custom upstream URL' })).toHaveCount(0)
  await page.locator('#channel-credentials').fill('fixture-key')
  await page.getByRole('button', { name: 'Other channels', exact: true }).click()
  const search = page.getByRole('combobox', { name: /Search channels/ })
  await search.fill('openai_compatible')
  await search.press('Enter')
  await expect(address).toHaveAttribute('aria-required', 'true')
  await expect(page.getByRole('button', { name: 'Create Group', exact: true })).toBeDisabled()
  await address.fill('not a URL')
  await expect(page.getByRole('button', { name: 'Create Group', exact: true })).toBeDisabled()
  await address.fill('https://relay.example.test/v1')
  await expect(page.getByRole('button', { name: 'Create Group', exact: true })).toBeEnabled()
  await page.getByRole('button', { name: 'OpenAI', exact: true }).click()
  await address.clear()
  await page.getByRole('button', { name: 'Create Group', exact: true }).click()
  await expect.poll(() => requests.createBodies.length).toBe(1)
  expect(
    (requests.createBodies[0]?.params as Record<string, unknown> | undefined)?.base_url,
  ).toBeUndefined()
})
test('channel search exposes its list and selects an option with the keyboard', async ({
  page,
}) => {
  await mockImportApi(page)
  await page.route('**/api/channels', (route) =>
    route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data: {
          items: [
            ...CHANNELS,
            ...['anthropic', 'gemini', 'openai_compatible', 'extra'].map((channel_id) => ({
              ...CHANNELS[0],
              channel_id,
              name: channel_id,
            })),
          ],
          total: 6,
        },
      },
    }),
  )
  await page.goto('/import?mode=new', { waitUntil: 'commit' })
  await page
    .getByRole('button', { name: 'Other channels', exact: true })
    .click({ timeout: FIRST_PAINT })
  const search = page.getByRole('combobox', { name: /Search channels/ })
  await expect(search).toHaveAttribute('aria-expanded', 'true')
  const list = page.getByRole('listbox', { name: 'Other channels' })
  await expect(list).toBeVisible()
  await expect(search).toHaveAttribute('aria-controls', (await list.getAttribute('id')) as string)
  await search.fill('extra')
  await expect(search).toHaveAttribute('aria-activedescendant', /channel-extra$/)
  await search.press('Enter')
  await expect(page.getByRole('button', { name: 'extra', exact: true })).toBeVisible()
  await expect(list).not.toBeVisible()
})

for (const width of [390, 320]) {
  test(`channel picker contents fit the popup and viewport at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 844 })
    await mockImportApi(page)
    await page.route('**/api/channels', (route) =>
      route.fulfill({
        json: {
          code: 0,
          message: 'ok',
          data: { items: PICKER_CHANNELS, total: PICKER_CHANNELS.length },
        },
      }),
    )
    await page.goto('/import?mode=new', { waitUntil: 'commit' })
    await page.getByRole('button', { name: 'Other channels', exact: true }).click({
      timeout: FIRST_PAINT,
    })
    const popup = page.getByRole('dialog', { name: 'Other channels', exact: true })
    const search = popup.getByRole('combobox', { name: /Search channels/ })
    await expect(search).toBeFocused()
    await expect(popup.getByRole('option')).toHaveCount(5)
    // Check inner controls and icons, not just the outer surface: a fitting
    // popover can still clip an oversized inner panel or scroll it sideways.
    await expect
      .poll(() =>
        popup.evaluate((surface) => {
          const layer = surface.closest('[popover]')!
          const outer = layer.getBoundingClientRect()
          const left = Math.max(0, outer.left)
          const right = Math.min(window.innerWidth, outer.right)
          const top = Math.max(0, outer.top)
          const bottom = Math.min(window.innerHeight, outer.bottom)
          return Array.from(
            surface.querySelectorAll(
              '[role="combobox"], [role="listbox"], [role="option"], .channel-icon, svg',
            ),
          )
            .filter((element) => {
              const rect = element.getBoundingClientRect()
              return (
                rect.left < left - 1 ||
                rect.right > right + 1 ||
                rect.top < top - 1 ||
                rect.bottom > bottom + 1
              )
            })
            .map((element) => element.getAttribute('role') ?? element.className.toString())
        }),
      )
      .toEqual([])
    await search.fill('extra')
    await search.press('Enter')
    await expect(page.getByRole('button', { name: 'Extra Channel', exact: true })).toBeVisible()
    await expect(popup).not.toBeVisible()
  })
}

test('unmatched channel search explains the compatible alternative and selects it by keyboard', async ({
  page,
}) => {
  await mockImportApi(page)
  await page.route('**/api/channels', (route) =>
    route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data: { items: PICKER_CHANNELS, total: PICKER_CHANNELS.length },
      },
    }),
  )
  await page.goto('/import?mode=new', { waitUntil: 'commit' })
  await page.getByRole('button', { name: 'Other channels', exact: true }).click({
    timeout: FIRST_PAINT,
  })
  const popup = page.getByRole('dialog', { name: 'Other channels', exact: true })
  const search = popup.getByRole('combobox', { name: /Search channels/ })
  await search.fill('zzzznonexistent')
  await expect(popup.getByText('No channels match', { exact: true })).toBeVisible()
  await expect(
    popup.getByText('Use this option only if the provider supports the OpenAI API.', {
      exact: true,
    }),
  ).toBeVisible()
  const alternative = popup.getByRole('option', {
    name: 'Use OpenAI-compatible channel',
    exact: true,
  })
  await expect(alternative).toBeVisible()
  await expect(search).toHaveAttribute(
    'aria-activedescendant',
    (await alternative.getAttribute('id'))!,
  )
  await search.press('Enter')
  await expect(popup).not.toBeVisible()
  await expect(
    page.getByRole('button', { name: 'OpenAI Compatible', exact: true }),
  ).toHaveAttribute('aria-pressed', 'true')
})

test('adds the first manual model from the empty import form', async ({ page }) => {
  const requests = await mockImportApi(page)
  await page.goto('/import', { waitUntil: 'commit' })
  await expect(page.getByRole('heading', { name: 'Import channel credentials' })).toBeVisible({
    timeout: FIRST_PAINT,
  })
  await page.getByRole('button', { name: 'Add model', exact: true }).click()
  await expect(page.locator('[data-model-id-index="0"]')).toBeVisible()
  await page.locator('[data-model-id-index="0"]').fill('ux-first-model')
  await page.getByRole('button', { name: 'Add model', exact: true }).click()
  await expect(page.locator('[data-model-id-index="1"]')).toBeVisible()
  await page.locator('[data-model-id-index="1"]').fill('ux-second-model')
  const firstAlias = page.getByLabel('Alias for ux-first-model')
  const secondAlias = page.getByLabel('Alias for ux-second-model')
  await expect(firstAlias).toBeVisible()
  await expect(page.getByLabel('Enable an alias for ux-first-model')).toHaveCount(0)
  await firstAlias.fill('  public-first  ')
  await secondAlias.fill('   ')
  const weight = page.locator('[data-model-weight-index="0"]')
  const priority = page.locator('[data-model-priority-index="0"]')
  await expect(weight).toHaveValue('1')
  await expect(priority).toHaveValue('1')
  await expect(weight).toBeEditable()
  await expect(priority).toBeEditable()
  const rows = page.locator('[data-testid="model-alias-editor__record"]')
  await expect(rows).toHaveCount(2)
  await expect(rows.nth(0)).not.toContainText('%')
  await expect(rows.nth(1)).not.toContainText('%')
  await expect(page.locator('[role="table"] i')).toHaveCount(0)
  await weight.fill('9')
  await priority.fill('2')
  await page.locator('#channel-credentials').fill('sk-route-fields')
  await page.getByRole('button', { name: 'Create Group', exact: true }).click()
  await expect.poll(() => requests.createBodies.length).toBe(1)
  expect(requests.createBodies[0]).toMatchObject({
    models: [
      { id: 'ux-first-model', alias: 'public-first', alias_enabled: true, weight: 9, priority: 2 },
      { id: 'ux-second-model', alias: '', alias_enabled: false, weight: null, priority: null },
    ],
  })
})

test('single credential rejects multiple lines including duplicate keys', async ({ page }) => {
  const requests = await mockImportApi(page)
  await page.goto('/import', { waitUntil: 'commit' })
  const input = page.locator('#channel-credentials')
  await expect(input).toBeVisible({ timeout: FIRST_PAINT })
  await input.fill('sk-first\nsk-second')
  await expect(input).toHaveAttribute('aria-invalid', 'true')
  await expect(page.getByRole('button', { name: 'Create Group' })).toBeDisabled()
  expect(requests.createBodies).toHaveLength(0)
  await input.fill('sk-first\nsk-first')
  await expect(page.getByRole('button', { name: 'Create Group' })).toBeDisabled()
  await input.fill('sk-first')
  await expect(page.getByRole('button', { name: 'Create Group' })).toBeEnabled()
})

test('one formatted OpenAI JSON credential can be submitted unchanged', async ({ page }) => {
  const requests = await mockImportApi(page)
  await page.goto('/import', { waitUntil: 'commit' })
  const input = page.locator('#channel-credentials')
  await expect(input).toBeVisible({ timeout: FIRST_PAINT })
  const credential = '{\n  "api_key": "test-json-placeholder"\n}'
  await input.fill(credential)
  await expect(input).not.toHaveAttribute('aria-invalid', 'true')
  await page.getByRole('button', { name: 'Create Group', exact: true }).click()
  await expect.poll(() => requests.createBodies.length).toBe(1)
  expect(requests.createBodies[0]!.credential).toBe(credential)
})

test('existing Group summary failure is visible and retry recovers the first import', async ({
  page,
}) => {
  await mockImportApi(page)
  let fail = true
  await page.route('**/api/groups/7', async (route) => {
    await route.fulfill({
      status: fail ? 500 : 200,
      contentType: 'application/json',
      body: JSON.stringify(
        fail
          ? { code: 'INTERNAL_SERVER_ERROR', message: 'failed', data: null }
          : {
              code: 0,
              message: 'ok',
              data: {
                ...GROUP_42.summary,
                id: 7,
                name: 'Alpha Group',
                credential_configured: false,
                credential_status: null,
              },
            },
      ),
    })
  })
  await page.goto('/import?group_id=7', { waitUntil: 'commit' })
  await expect(page.getByText('Unable to load Group details.', { exact: true })).toBeVisible({
    timeout: FIRST_PAINT,
  })
  await expect(page.locator('#channel-credentials')).toBeDisabled()
  fail = false
  await page.getByRole('button', { name: 'Retry', exact: true }).click()
  await expect(page.locator('#channel-credentials')).toBeEnabled()
  await page.locator('#channel-credentials').fill('sk-first\nsk-second')
  await expect(page.getByRole('button', { name: 'Import credential', exact: true })).toBeDisabled()
  await page.locator('#channel-credentials').fill('sk-first')
  await expect(page.getByRole('button', { name: 'Import credential', exact: true })).toBeEnabled()
})

test('same-target conflict offers management, never append', async ({ page }) => {
  const requests = await mockImportApi(page)
  await page.route('**/api/groups', async (route) => {
    await route.fulfill({
      status: 409,
      contentType: 'application/json',
      body: JSON.stringify({
        code: 'CHANNEL_TARGET_CONFLICT',
        message: 'conflict',
        data: { groups: [{ id: 7, name: 'Alpha Group' }] },
      }),
    })
  })
  await page.goto('/import', { waitUntil: 'commit' })
  await expect(page.locator('#channel-credentials')).toBeVisible({ timeout: FIRST_PAINT })
  await page.locator('#channel-credentials').fill('sk-first')
  await page.getByRole('button', { name: 'Create Group', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Manage Group', exact: true })).toBeVisible()
  await expect(
    page.getByRole('button', { name: 'Import credentials here', exact: true }),
  ).toHaveCount(0)
  await page.getByRole('button', { name: 'Manage Group', exact: true }).click()
  await page.getByRole('button', { name: 'Discard changes', exact: true }).click()
  await expect(page).toHaveURL(/\/groups\/7$/)
  expect(requests.credentialImports).toHaveLength(0)
})

test('single-credential server rejection explains how to correct the draft', async ({ page }) => {
  await mockImportApi(page)
  await page.route('**/api/groups', async (route) => {
    await route.fulfill({
      status: 409,
      contentType: 'application/json',
      body: JSON.stringify({ code: 'SINGLE_CREDENTIAL_REQUIRED', message: 'only one', data: null }),
    })
  })
  await page.goto('/import', { waitUntil: 'commit' })
  await expect(page.locator('#channel-credentials')).toBeVisible({ timeout: FIRST_PAINT })
  await page.locator('#channel-credentials').fill('sk-first')
  await page.getByRole('button', { name: 'Create Group', exact: true }).click()
  await expect(
    page.getByText('Keep only one credential, even if the lines are identical.', { exact: true }),
  ).toBeVisible()
  await expect(page.getByText('Unable to create the Group', { exact: true })).toHaveCount(0)
})

test('mode switch swaps views and canonicalizes the URL', async ({ page }) => {
  await mockImportApi(page)
  await page.goto('/import', { waitUntil: 'commit' })

  await expect(page.getByRole('heading', { name: 'Import channel credentials' })).toBeVisible({
    timeout: FIRST_PAINT,
  })
  await expect(page.getByRole('radio', { name: 'New Group' })).toHaveAttribute(
    'aria-checked',
    'true',
  )

  await page.getByRole('radio', { name: 'Existing Group' }).click()
  await expect(page).toHaveURL(/\/import\?.*mode=existing/)
  await expect(page.getByRole('combobox', { name: 'Group' })).toBeVisible()
})

test('group_id deep link lands in existing mode with the group preselected', async ({ page }) => {
  await mockImportApi(page)
  await page.goto('/import?group_id=7', { waitUntil: 'commit' })

  await expect(page.getByRole('radio', { name: 'Existing Group' })).toHaveAttribute(
    'aria-checked',
    'true',
    { timeout: FIRST_PAINT },
  )
  await expect(page.getByRole('combobox', { name: 'Group' })).toContainText('Alpha Group')
})

test('empty existing-group import posts one credential with a stable idempotency key', async ({
  page,
}) => {
  const requests = await mockImportApi(page)
  await page.goto('/import?group_id=7', { waitUntil: 'commit' })

  const submit = page.getByRole('button', { name: 'Import credential', exact: true })
  await expect(submit).toBeVisible({ timeout: FIRST_PAINT })
  await page.locator('#channel-credentials').fill('sk-alpha-1')
  await expect(submit).toBeEnabled()
  await submit.click()

  await expect.poll(() => requests.credentialImports.length).toBe(1)
  expect(requests.credentialImports[0]!.groupID).toBe('7')
  expect(requests.credentialImports[0]!.key.length).toBeGreaterThan(0)
  expect(String(requests.credentialImports[0]!.body.credential)).toContain('sk-alpha-1')
})

test('populated existing Group guides management instead of importing or replacing', async ({
  page,
}) => {
  const requests = await mockImportApi(page, { credentialCount: 1 })
  await page.goto('/import?group_id=7', { waitUntil: 'commit' })
  const manage = page.getByRole('link', { name: 'Manage credential', exact: true })
  await expect(manage).toBeVisible({ timeout: FIRST_PAINT })
  await expect(manage).toHaveAttribute('href', '/groups/7')
  await expect(page.locator('#channel-credentials')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Import credential', exact: true })).toHaveCount(0)
  await manage.click()
  await expect(page).toHaveURL(/\/groups\/7$/)
  expect(requests.credentialImports).toHaveLength(0)
})

test('new-group create posts one credential with an idempotency key and navigates to the group', async ({
  page,
}) => {
  const requests = await mockImportApi(page)
  await page.goto('/import', { waitUntil: 'commit' })

  await expect(page.getByRole('heading', { name: 'Import channel credentials' })).toBeVisible({
    timeout: FIRST_PAINT,
  })
  // First channel is auto-adopted; credentials make the form submittable.
  await page.locator('#channel-credentials').fill('sk-live-1')
  await page.getByRole('button', { name: 'Create Group' }).click()

  await expect.poll(() => requests.createBodies.length).toBe(1)
  const body = requests.createBodies[0]!
  expect(body.channel_id).toBe('openai')
  expect(body.credential).toBe('sk-live-1')
  expect(body).not.toHaveProperty('credentials')
  expect(requests.createIdempotencyKeys[0]!.length).toBeGreaterThan(0)

  await page.waitForURL(/\/groups\/42/, { timeout: 15_000 })
})

test('a 401 captures the draft to sessionStorage and re-login restores it', async ({
  page,
  context,
}) => {
  await mockImportApi(page, { createStatus: 401 })
  await page.route('**/api/channels', (route) =>
    route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data: { items: SIMPLIFIED_CHANNELS, total: SIMPLIFIED_CHANNELS.length },
      },
    }),
  )
  await page.goto('/import', { waitUntil: 'commit' })

  await expect(page.getByRole('heading', { name: 'Import channel credentials' })).toBeVisible({
    timeout: FIRST_PAINT,
  })
  await page.getByRole('button', { name: 'OpenAI', exact: true }).click()
  await page
    .getByRole('textbox', { name: /^Upstream base URL/ })
    .fill('https://legacy.example.test/v1')
  await page.locator('#channel-credentials').fill('sk-rescue-1')
  await page.getByRole('button', { name: 'Create Group' }).click()

  // The unauthorized handler captures the draft, clears the session, and
  // bounces to /login?redirect=...
  await page.waitForURL(/\/login\?.*redirect=/, { timeout: 15_000 })
  const stored = await page.evaluate(() =>
    window.sessionStorage.getItem('gpt-load.import-reauth-draft'),
  )
  expect(stored).not.toBeNull()
  const draft = JSON.parse(stored!) as { draft: { mode: string; credentials: string } }
  expect(draft.draft.mode).toBe('new')
  expect(draft.draft.credentials).toContain('sk-rescue-1')

  // Re-auth: the session endpoint still answers admin, so signing in lands
  // back on /import and the consumed draft repopulates the form.
  await page.evaluate(() => window.localStorage.setItem('gpt-load.auth-key', 'e2e-auth-key'))
  await page.getByRole('textbox', { name: 'Sign-in key', exact: true }).fill('e2e-auth-key')
  await page.getByRole('button', { name: 'Sign in' }).click()

  await page.waitForURL(/\/import/, { timeout: 15_000 })
  await expect(page.locator('#channel-credentials')).toHaveValue(/sk-rescue-1/, {
    timeout: 15_000,
  })
  await expect(page.getByRole('button', { name: 'OpenAI', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  )
  await expect(page.getByRole('textbox', { name: /^Upstream base URL/ })).toHaveValue(
    'https://legacy.example.test/v1',
  )
  // The draft is single-consumption: sessionStorage no longer holds it.
  await expect
    .poll(() => page.evaluate(() => window.sessionStorage.getItem('gpt-load.import-reauth-draft')))
    .toBeNull()
  void context
})

test('import discovery runs against the entered connection and appends candidates', async ({
  page,
}) => {
  const requests = await mockImportApi(page)
  const discoveries: unknown[] = []
  // The default OpenAI fixture advertises no discovery capability; enable it
  // so the publish step exposes the discovery entry point.
  await page.route('**/api/channels', (route) =>
    route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data: {
          items: [
            {
              ...CHANNELS[0]!,
              capabilities: { ...CHANNELS[0]!.capabilities, model_discovery: true },
            },
            ...CHANNELS.slice(1),
          ],
          total: CHANNELS.length,
        },
      },
    }),
  )
  await page.route('**/api/models/discover', async (route) => {
    discoveries.push(route.request().postDataJSON())
    return route.fulfill({
      json: {
        code: 0,
        message: 'ok',
        data: {
          models: [
            {
              id: 'gpt-discovered',
              name: 'GPT Discovered',
              sources: ['live'],
              pricing_status: 'pending',
              pricing_source: null,
            },
            {
              id: 'gpt-priced',
              name: 'GPT Priced',
              sources: ['catalog'],
              pricing_status: 'configured',
              pricing_source: 'OpenAI',
            },
          ],
        },
      },
    })
  })
  await page.goto('/import?mode=new', { waitUntil: 'commit' })
  await expect(page.getByRole('heading', { name: 'Import channel credentials' })).toBeVisible({
    timeout: FIRST_PAINT,
  })
  await page.locator('#channel-credentials').fill('sk-discover')

  const discover = page.getByRole('button', { name: 'Get upstream models' })
  await expect(discover).toBeEnabled()
  await discover.click()

  const dialog = page.getByRole('dialog', { name: 'Get upstream models' })
  await expect(dialog).toBeVisible()

  // Runtime evidence: the discovery request carries the entered credential.
  await expect.poll(() => discoveries.length).toBe(1)
  expect(discoveries[0]).toMatchObject({
    channel_id: 'openai',
    credentials: 'sk-discover',
  })

  // Evidence legend stays readable without tooltips.
  const legend = dialog.getByLabel('Evidence legend')
  await expect(legend).toBeVisible()
  await expect(legend.getByText('Live discovery', { exact: true })).toBeVisible()

  await expect(dialog.getByText('0 of 2 available models selected')).toBeVisible()
  await dialog
    .getByRole('checkbox', { name: /^GPT Discovered GPT Discovered gpt-discovered/ })
    .click()
  await dialog.getByRole('checkbox', { name: /^GPT Priced GPT Priced gpt-priced/ }).click()
  await expect(dialog.getByText('2 of 2 available models selected')).toBeVisible()

  await dialog.getByRole('button', { name: 'Add to current list' }).click()
  await expect(dialog).not.toBeVisible()
  const modelTable = page.getByRole('table', { name: 'Models to publish' })
  await expect(modelTable.getByRole('code')).toHaveText(['gpt-discovered', 'gpt-priced'])
  expect(requests.createBodies).toHaveLength(0)
})
