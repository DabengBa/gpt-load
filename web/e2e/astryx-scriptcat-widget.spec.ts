import { readFile } from 'node:fs/promises'

import { expect, test, type Page } from '@playwright/test'

const userscript = await readFile(
  new URL('../../tools/scriptcat/gpt-load-channel-import.user.js', import.meta.url),
  'utf8',
)

async function installGMBridge(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const values = new Map<string, unknown>()
    const bridge = { values, openedURL: '' }
    Object.defineProperty(window, '__scriptcatTest', { value: bridge })
    Object.defineProperty(window, 'GM', {
      value: {
        getValue: async (key: string, fallback?: unknown) => values.get(key) ?? fallback,
        setValue: async (key: string, value: unknown) => values.set(key, value),
        deleteValue: async (key: string) => values.delete(key),
        listValues: async () => [...values.keys()],
      },
    })
    Object.defineProperty(window, 'GM_openInTab', {
      value: (url: string) => {
        bridge.openedURL = url
      },
    })
  })
}

async function serveKeyPage(page: Page, path = '/keys'): Promise<void> {
  await page.route('https://totokens.cc/**', async (route) =>
    route.fulfill({
      status: 200,
      contentType: 'text/html',
      body:
        path === '/keys'
          ? '<main><h1>API Keys</h1><table><tr><td>sk-widget-fixture-12345678</td></tr></table></main>'
          : '<main><h1>Account</h1></main>',
    }),
  )
  await page.goto(`https://totokens.cc${path}`)
  await page.addScriptTag({ content: userscript })
}


function receiverGMBridge(entries: Array<[string, unknown]>): void {
  const values = new Map(entries)
  Object.defineProperty(window, 'GM', {
    value: {
      getValue: async (key: string, fallback?: unknown) => values.get(key) ?? fallback,
      setValue: async (key: string, value: unknown) => values.set(key, value),
      deleteValue: async (key: string) => values.delete(key),
      listValues: async () => [...values.keys()],
    },
  })
  Object.defineProperty(window, 'GM_openInTab', { value: () => {} })
  localStorage.setItem('gpt-load.auth-key', 'fixture-auth')
}

async function mockImportAPI(page: Page): Promise<void> {
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
                  connection: { credential_input: 'batch_text' },
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
}

test('automatically shows the import widget on To-Tokens keys and transfers only after confirmation', async ({
  page,
}) => {
  await installGMBridge(page)
  await serveKeyPage(page)

  const launcher = page.getByRole('button', { name: '导入到 GPT-Load' })
  await expect(launcher).toBeVisible()
  await launcher.click()
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as Window & { __scriptcatTest: { openedURL: string } }).__scriptcatTest.openedURL,
      ),
    )
    .toBe('')

  const apiKey = page.locator('#sc-api-key')
  await expect(apiKey).toHaveValue('sk-widget-fixture-12345678')
  await expect(page.getByLabel('Base URL')).toHaveValue('https://totokens.cc')
  await expect(page.getByLabel('GPT-Load 地址')).toHaveValue('')
  await page.getByLabel('GPT-Load 地址').fill('https://gpt-load.example.test')
  await page.getByLabel('Base URL').fill('https://api.custom.example.test/v1')
  await page.getByRole('button', { name: '关闭导入面板' }).click()
  await launcher.click()
  await expect(page.locator('#sc-api-key')).toHaveValue('sk-widget-fixture-12345678')
  await expect(page.getByLabel('Base URL')).toHaveValue('https://api.custom.example.test/v1')
  await expect(page.getByLabel('GPT-Load 地址')).toHaveValue('https://gpt-load.example.test')
  await page.getByRole('button', { name: '打开导入页' }).click()

  const openedURL = await page.evaluate(
    () => (window as Window & { __scriptcatTest: { openedURL: string } }).__scriptcatTest.openedURL,
  )
  const target = new URL(openedURL)
  expect(target.origin).toBe('https://gpt-load.example.test')
  expect(target.pathname).toBe('/import')
  expect(target.searchParams.get('mode')).toBe('new')
  expect(target.searchParams.get('scriptcat_handoff')).toBeTruthy()
  expect(openedURL).not.toContain('sk-widget-fixture-12345678')
  const transfer = await page.evaluate(() => {
    const bridge = (window as Window & { __scriptcatTest: { values: Map<string, unknown> } })
      .__scriptcatTest
    return [...bridge.values.entries()].find(([key]) => key.startsWith('pending-import:'))?.[1]
  })
  expect(transfer).toMatchObject({ baseURL: 'https://api.custom.example.test/v1' })

  await page.evaluate(() => {
    history.pushState({}, '', '/login?redirect=/keys')
    document.body.innerHTML = '<form><input type="password"><button>登录</button></form>'
  })
  await expect(launcher).toBeHidden()
})

test('shows the import widget on generic keys routes and uses the current origin as Base URL', async ({
  page,
}) => {
  await installGMBridge(page)
  await page.route('https://provider.example.test/**', (route) =>
    route.fulfill({
      contentType: 'text/html',
      body: '<main><h1>API Keys</h1><table><tr><td>sk-fx-widget-fixture-12345678</td></tr></table></main>',
    }),
  )
  await page.goto('https://provider.example.test/keys')
  await page.addScriptTag({ content: userscript })

  const widget = page.locator('#gpt-load-scriptcat-widget')
  await expect(widget).toHaveCount(1)
  await page.getByRole('button', { name: '导入到 GPT-Load' }).click()
  await expect(page.locator('#sc-api-key')).toHaveValue('sk-fx-widget-fixture-12345678')
  await expect(page.locator('#sc-base-url')).toHaveValue('https://provider.example.test')
})

test('ScriptCat handoff fills group name, provider website, API key, and Base URL in GPT-Load', async ({
  page,
  baseURL,
}) => {
  await installGMBridge(page)
  await serveKeyPage(page)

  await page.getByRole('button', { name: '导入到 GPT-Load' }).click()
  await page.locator('#sc-api-key').fill('sk-full-flow-fixture')
  await page.getByLabel('分组名').fill('To-Tokens production')
  await page.getByLabel('Base URL').fill('https://api.totokens.example/v1')
  await page.getByLabel('GPT-Load 地址').fill(baseURL!)
  await page.getByRole('button', { name: '打开导入页' }).click()

  const handoff = await page.evaluate(() => {
    const bridge = (
      window as Window & {
        __scriptcatTest: { values: Map<string, unknown>; openedURL: string }
      }
    ).__scriptcatTest
    const target = new URL(bridge.openedURL)
    const transferKey = `pending-import:${target.searchParams.get('scriptcat_handoff')}`
    return {
      targetURL: bridge.openedURL,
      transferKey,
      transfer: bridge.values.get(transferKey),
    }
  })
  expect(handoff.targetURL).not.toContain('sk-full-flow-fixture')
  expect(handoff.transfer).toMatchObject({
    apiKey: 'sk-full-flow-fixture',
    baseURL: 'https://api.totokens.example/v1',
    groupName: 'To-Tokens production',
    providerUrl: 'https://totokens.cc',
  })

  const receiver = await page.context().newPage()
  await mockImportAPI(receiver)
  await receiver.addInitScript({
    content: `(${receiverGMBridge.toString()})(${JSON.stringify([
      [handoff.transferKey, handoff.transfer],
    ])});\n${userscript}`,
  })
  await receiver.goto(handoff.targetURL)

  await expect(receiver.locator('#channel-credentials')).toHaveValue('sk-full-flow-fixture')
  await expect(receiver.getByRole('textbox', { name: /^Group name/ })).toHaveValue(
    'To-Tokens production',
  )
  await expect(receiver.getByRole('textbox', { name: /^Provider website/ })).toHaveValue(
    'https://totokens.cc',
  )
  await receiver.getByRole('button', { name: 'OpenAI', exact: true }).click()
  await expect(receiver.getByRole('textbox', { name: /^Custom upstream URL/ })).toHaveValue(
    'https://api.totokens.example/v1',
  )
})

test('a concurrent ScriptCat handoff delivers credentials to only one tab', async ({ context }) => {
  const origin = 'https://gpt-load.example.test'
  const transferKey = 'pending-import:concurrent-fixture'
  const sessionKey = 'gpt-load.scriptcat-import-payload'
  const values = new Map([
    [
      transferKey,
      {
        createdAt: Date.now(),
        targetOrigin: origin,
        apiKey: 'sk-concurrent-fixture',
        baseURL: 'https://relay.example.test/v1',
        groupName: 'Concurrent fixture',
        providerUrl: 'https://provider.example.test',
      },
    ],
  ])
  const readers = new Set<Page>()
  const firstRead = Promise.withResolvers<void>()
  const releaseRead = Promise.withResolvers<void>()
  let deletions = 0
  await context.exposeBinding('readScriptcatTransfer', async ({ page }, key: string) => {
    const snapshot = values.get(key)
    if (snapshot) {
      readers.add(page)
      firstRead.resolve()
      await releaseRead.promise
    }
    return snapshot
  })
  await context.exposeBinding('deleteScriptcatTransfer', (_source, key: string) => {
    deletions += 1
    values.delete(key)
  })
  await context.addInitScript(() => {
    const bridge = window as Window & {
      readScriptcatTransfer(key: string): Promise<unknown>
      deleteScriptcatTransfer(key: string): Promise<void>
    }
    Object.defineProperty(window, 'GM', {
      value: {
        getValue: bridge.readScriptcatTransfer,
        deleteValue: bridge.deleteScriptcatTransfer,
      },
    })
  })
  await context.route(`${origin}/**`, (route) =>
    route.fulfill({ contentType: 'text/html', body: '<title>ScriptCat receiver fixture</title>' }),
  )
  const pages = await Promise.all([context.newPage(), context.newPage()])
  await Promise.all(
    pages.map((page) => page.goto(`${origin}/import?scriptcat_handoff=concurrent-fixture`)),
  )
  const receiving = Promise.all(pages.map((page) => page.evaluate(userscript)))
  try {
    await firstRead.promise
    // Hold the first GM response until the second tab either reads the same
    // transfer (the race) or queues behind the browser's real exclusive lock.
    await expect
      .poll(async () => {
        const { pending } = await pages[0].evaluate(() => navigator.locks.query())
        return readers.size === 2 || pending?.length === 1
      })
      .toBe(true)
  } finally {
    releaseRead.resolve()
    await receiving
  }
  const payloads = await Promise.all(
    pages.map((page) => page.evaluate((key) => sessionStorage.getItem(key), sessionKey)),
  )
  expect(payloads.filter((payload) => payload !== null)).toHaveLength(1)
  expect(readers.size).toBe(1)
  expect(deletions).toBe(1)
  expect(values.has(transferKey)).toBe(false)
})

test('does not show the import widget on unrelated pages', async ({ page }) => {
  await installGMBridge(page)
  await serveKeyPage(page, '/account')
  await expect(page.locator('#gpt-load-scriptcat-widget')).toHaveCount(0)
})

test('generic SPA navigation shows the widget only on the token route without domain-specific matches', async ({ page }) => {
  const startURL = 'https://provider.example.test/console'
  const matches = [...userscript.matchAll(/^\/\/\s*@match\s+(\S+)/gm)].map((match) => match[1])
  expect(matches).toContain('*://*/*')
  expect(matches).not.toContain('*://*/console/tokens*')
  expect(matches.every((match) => !/https?:\/\//u.test(match))).toBe(true)
  await installGMBridge(page)
  await page.route('https://provider.example.test/**', (route) =>
    route.fulfill({ contentType: 'text/html', body: '<main>PackyAPI console</main>' }),
  )
  await page.goto(startURL)
  await page.addScriptTag({ content: userscript })
  const widget = page.locator('#gpt-load-scriptcat-widget')
  await expect(widget).toHaveCount(0)
  await page.evaluate(() => history.pushState({}, '', '/console/token'))
  await expect(page.getByRole('button', { name: '导入到 GPT-Load' })).toBeVisible()
  await page.evaluate(() => history.pushState({}, '', '/console'))
  await expect(widget).toBeHidden()
  await page.evaluate(() => history.pushState({}, '', '/console/token'))
  await expect(page.getByRole('button', { name: '导入到 GPT-Load' })).toBeVisible()
})

test('keeps the panel within a narrow mobile viewport', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 640 })
  await installGMBridge(page)
  await serveKeyPage(page)
  await page.getByRole('button', { name: '导入到 GPT-Load' }).click()

  const panel = page.getByRole('region', { name: '导入 API Key' })
  const bounds = await panel.boundingBox()
  expect(bounds).not.toBeNull()
  expect(bounds!.x).toBeGreaterThanOrEqual(0)
  expect(bounds!.y).toBeGreaterThanOrEqual(0)
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(320)
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(320)
})
