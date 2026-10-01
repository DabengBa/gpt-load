import { spawn, type ChildProcess } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { expect, test } from '@playwright/test'

import { resolveChromium125Executable } from './browser-executables'

// B5: Content-Security-Policy smoke against the real Go binary — the dev
// server cannot reproduce the production header, so this spec only runs
// where a compiled artifact exists (`make build`, or GPT_LOAD_BINARY).
// The binary does not build on this project's Windows dev box, so the
// spec skips there and runs on Unix CI runners.
//
// B13: the same sweep also runs under the `chromium-125` and
// `chrome-latest` projects for the Phase 1 browser-floor gate.
// `chromium-125` skips when no Chrome/Chromium 125 executable is
// provisioned (see browser-executables.ts).

const repoRoot = fileURLToPath(new URL('../..', import.meta.url))
// GPT_LOAD_ORIGIN connects to an already-running server instead of spawning
// one — used when the binary was built for another platform (e.g. the real
// linux binary served from WSL2 while Playwright runs on Windows).
const externalOrigin = process.env.GPT_LOAD_ORIGIN
const binaryCandidates = [
  process.env.GPT_LOAD_BINARY,
  resolve(repoRoot, 'gpt-load.exe'),
  resolve(repoRoot, 'gpt-load'),
].filter((path): path is string => typeof path === 'string')
const binary = binaryCandidates.find((path) => existsSync(path))

const port = 40_000 + (process.pid % 20_000)
const origin = externalOrigin ?? `http://127.0.0.1:${port}`

// Playwright's loader cannot take plain JSON imports; read the manifest
// directly. The sweep below covers every document route.
const pagePaths = (
  JSON.parse(readFileSync(resolve(repoRoot, 'internal/webui/page_routes.json'), 'utf8')) as {
    routes: Array<{ path: string }>
  }
).routes.map((entry) => entry.path)

let server: ChildProcess | undefined
let dataDir: string | undefined

test.beforeAll(async ({}, testInfo) => {
  test.skip(
    binary === undefined && externalOrigin === undefined,
    'no gpt-load binary: run `make build`, set GPT_LOAD_BINARY, ' +
      'or point GPT_LOAD_ORIGIN at a running server',
  )
  test.skip(
    testInfo.project.name === 'chromium-125' && resolveChromium125Executable() === undefined,
    'no Chromium 125 executable: set GPT_LOAD_CHROME_125_EXE or run ' +
      '`npx @puppeteer/browsers install chrome@125`',
  )
  if (externalOrigin === undefined) {
    dataDir = mkdtempSync(resolve(tmpdir(), 'gpt-load-csp-'))
    server = spawn(binary as string, [], {
      cwd: dataDir,
      env: {
        ...process.env,
        HOST: '127.0.0.1',
        PORT: String(port),
        AUTH_KEY: 'csp-smoke-key',
      },
      stdio: 'ignore',
    })
  }
  const deadline = Date.now() + 15_000
  for (;;) {
    try {
      const response = await fetch(`${origin}/login`, {
        headers: { accept: 'text/html' },
      })
      if (response.status > 0) return
    } catch {
      // not up yet
    }
    if (Date.now() > deadline) {
      throw new Error('gpt-load binary did not start listening within 15s')
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 250))
  }
})

test.afterAll(async () => {
  if (server) {
    server.kill()
    // Windows releases the terminated child's handles asynchronously; wait
    // for exit before removing the data dir or rmdir races EBUSY.
    await Promise.race([
      new Promise((resolvePromise) => server?.once('exit', resolvePromise)),
      new Promise((resolvePromise) => setTimeout(resolvePromise, 5_000)),
    ])
  }
  if (dataDir) {
    rmSync(dataDir, {
      recursive: true,
      force: true,
      maxRetries: 10,
      retryDelay: 200,
    })
  }
})

async function expectCleanDocument(
  page: import('@playwright/test').Page,
  path: string,
  marker: string,
) {
  const consoleErrors: string[] = []
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text())
  })
  await page.addInitScript(() => {
    ;(window as unknown as { __cspViolations?: string[] }).__cspViolations = []
    document.addEventListener('securitypolicyviolation', (event) => {
      ;(window as unknown as { __cspViolations?: string[] }).__cspViolations?.push(
        `${event.violatedDirective}: ${event.blockedURI}`,
      )
    })
  })

  const response = await page.goto(`${origin}${path}`)
  expect(response).not.toBeNull()
  const csp = response?.headers()['content-security-policy']
  expect(csp).toContain("default-src 'self'")
  expect(csp).toContain("object-src 'none'")

  await expect.poll(async () => (await page.content()).includes(marker)).toBe(true)
  await expect
    .poll(async () =>
      page.evaluate(
        () => (window as unknown as { __cspViolations?: string[] }).__cspViolations?.length ?? 0,
      ),
    )
    .toBe(0)
  expect(consoleErrors).toEqual([])
}

// The gate requires zero CSP violations on the single embedded document,
// so the sweep is driven by the manifest rather than a hardcoded route.
for (const routePath of pagePaths) {
  test(`Go server CSP keeps the document clean: ${routePath}`, async ({ page }) => {
    await expectCleanDocument(page, routePath, '/assets/index')
  })
}

// A leftover gpt-load.frontend cookie must not change the served document.
test('Go server ignores the retired frontend cookie', async ({ page, context }) => {
  await context.addCookies([{ name: 'gpt-load.frontend', value: 'classic', url: origin }])
  await expectCleanDocument(page, '/settings', '/assets/index')
})
