import { expect, test, type BrowserContext, type Page } from '@playwright/test'

// Post-cutover selection contract: the dev server and the Go binary serve the
// single Astryx document for every page route and every unknown browser path,
// regardless of any leftover `gpt-load.frontend` cookie.

const PAGE_PATH = '/settings'
const DETAIL_PATH = '/groups/7'
const IMPORT_PATH = '/import'
const UNKNOWN_PATH = '/definitely-not-a-route'
const COOKIE = 'gpt-load.frontend'

const ASTRYX_MARKER = '/src/frontends/astryx/main.tsx'

async function setFrontendCookie(context: BrowserContext, value: string) {
  await context.clearCookies()
  await context.addCookies([{ name: COOKIE, value, domain: '127.0.0.1', path: '/' }])
}

// Request-level check: page.request sends the context cookies, and the
// document is discriminated on the served HTML marker — no client-side router
// involved.
async function expectDocument(
  page: Page,
  path: string,
  marker: string,
  status = 200,
) {
  const response = await page.request.get(path, {
    headers: { accept: 'text/html' },
  })
  expect(response.status()).toBe(status)
  expect(await response.text()).toContain(marker)
}

test('page route serves the Astryx document without any cookie', async ({
  page,
  context,
}) => {
  await context.clearCookies()
  await expectDocument(page, PAGE_PATH, ASTRYX_MARKER)
})

test('param and import routes serve the Astryx document', async ({ page }) => {
  await expectDocument(page, DETAIL_PATH, ASTRYX_MARKER)
  await expectDocument(page, IMPORT_PATH, ASTRYX_MARKER)
})

test('a leftover classic cookie is ignored', async ({ page, context }) => {
  await setFrontendCookie(context, 'classic')
  await expectDocument(page, PAGE_PATH, ASTRYX_MARKER)
})

test('an unrecognized cookie value is ignored', async ({ page, context }) => {
  await setFrontendCookie(context, 'bogus')
  await expectDocument(page, PAGE_PATH, ASTRYX_MARKER)
})

test('unknown browser path serves the Astryx document', async ({
  page,
  context,
}) => {
  // Dev-server SPA fallback returns 200; the Go binary's 404 contract for
  // unknown paths is covered by go-csp.spec.ts.
  await context.clearCookies()
  await expectDocument(page, UNKNOWN_PATH, ASTRYX_MARKER)
})
