import { expect, test, type BrowserContext, type Page } from '@playwright/test'

// B5: dev-server document selection mirrors internal/webui/server.go.
//   flagged manifest route + cookie "astryx" -> astryx.html
//   unknown browser path + cookie "astryx"   -> astryx.html (404-fallback parity)
//   otherwise                                 -> index.html
// Phase 4 note: every manifest route is flagged now, so the cookie-off side
// of the matrix is exercised on the flagged paths themselves.
// This spec runs in the `astryx` project, whose storageState seeds the
// opt-in cookie; individual tests adjust the context cookies to cover the
// rest of the matrix.

const FLAGGED_PATH = '/settings' // page_routes.json: astryx: true
const DETAIL_PATH = '/groups/7' // page_routes.json: /groups/:id astryx: true
const IMPORT_PATH = '/import' // page_routes.json: astryx: true (Phase 4)
const UNKNOWN_PATH = '/definitely-not-a-route'
const COOKIE = 'gpt-load.frontend'

const CLASSIC_MARKER = '/src/main.ts'
const ASTRYX_MARKER = '/src/frontends/astryx/main.tsx'

async function setFrontendCookie(context: BrowserContext, value: string) {
  await context.clearCookies()
  await context.addCookies([{ name: COOKIE, value, domain: '127.0.0.1', path: '/' }])
}

// Request-level check: page.request sends the context cookies, and document
// discrimination happens on the served HTML marker — no client-side router
// redirects involved.
async function expectDocument(page: Page, path: string, marker: string) {
  const response = await page.request.get(path, {
    headers: { accept: 'text/html' },
  })
  expect(response.ok()).toBe(true)
  expect(await response.text()).toContain(marker)
}

test('flagged route serves the Astryx document when opted in', async ({ page }) => {
  await expectDocument(page, FLAGGED_PATH, ASTRYX_MARKER)
})

test('flagged route serves the classic document without the cookie', async ({
  page,
  context,
}) => {
  await context.clearCookies()
  await expectDocument(page, FLAGGED_PATH, CLASSIC_MARKER)
})

test('flagged route serves the classic document for an explicit classic cookie', async ({
  page,
  context,
}) => {
  await setFrontendCookie(context, 'classic')
  await expectDocument(page, FLAGGED_PATH, CLASSIC_MARKER)
})

test('flagged route ignores unrecognized cookie values', async ({
  page,
  context,
}) => {
  await setFrontendCookie(context, 'bogus')
  await expectDocument(page, FLAGGED_PATH, CLASSIC_MARKER)
})

// Phase 4: every manifest route is now Astryx-flagged — the former
// "/import unflagged" fixture covers flag + cookie selection on the newly
// migrated routes instead (param route + import).
test('flagged param route serves the Astryx document when opted in', async ({
  page,
}) => {
  await expectDocument(page, DETAIL_PATH, ASTRYX_MARKER)
})

test('flagged /import serves the Astryx document when opted in', async ({
  page,
}) => {
  await expectDocument(page, IMPORT_PATH, ASTRYX_MARKER)
})

test('unknown browser path falls back to the Astryx document when opted in', async ({
  page,
}) => {
  await expectDocument(page, UNKNOWN_PATH, ASTRYX_MARKER)
})

test('unknown browser path falls back to the classic document without the cookie', async ({
  page,
  context,
}) => {
  await context.clearCookies()
  await expectDocument(page, UNKNOWN_PATH, CLASSIC_MARKER)
})

// B6: the classic preferences control writes the shared cookie and reloads;
// the next flagged-route request must then serve the Astryx document.
test('preferences switch opts in and flagged routes serve Astryx', async ({
  page,
  context,
}) => {
  await context.clearCookies()
  await page.goto('/login')

  await page.getByRole('button', { name: 'Preferences' }).click()
  const frontendGroup = page.getByRole('group', { name: 'Interface' })
  await expect(frontendGroup).toBeVisible()
  // dispatchEvent avoids the click retry loop racing the control's reload.
  await frontendGroup.locator('label', { hasText: 'Preview' }).dispatchEvent('click')

  // The control reloads the page after writing the cookie.
  await page.waitForLoadState('load')
  const cookies = await context.cookies()
  expect(cookies.find((cookie) => cookie.name === COOKIE)?.value).toBe('astryx')

  // The control reloads onto /login — itself a flagged route, so the
  // reloaded document is already Astryx; a flagged route keeps serving the
  // Astryx document under the production contract.
  await expectDocument(page, FLAGGED_PATH, ASTRYX_MARKER)
})
