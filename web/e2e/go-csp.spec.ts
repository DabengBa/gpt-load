import { spawn, type ChildProcess } from 'node:child_process'
import { existsSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { expect, test } from '@playwright/test'

// B5: Content-Security-Policy smoke against the real Go binary — the dev
// server cannot reproduce the production header, so this spec only runs
// where a compiled artifact exists (`make build`, or GPT_LOAD_BINARY).
// The binary does not build on this project's Windows dev box, so the
// spec skips there and runs on Unix CI runners.

const repoRoot = fileURLToPath(new URL('../..', import.meta.url))
const binaryCandidates = [
  process.env.GPT_LOAD_BINARY,
  resolve(repoRoot, 'gpt-load.exe'),
  resolve(repoRoot, 'gpt-load'),
].filter((path): path is string => typeof path === 'string')
const binary = binaryCandidates.find((path) => existsSync(path))

const port = 40_000 + (process.pid % 20_000)
const origin = `http://127.0.0.1:${port}`

let server: ChildProcess | undefined
let dataDir: string | undefined

test.beforeAll(async () => {
  test.skip(
    binary === undefined,
    'no gpt-load binary: run `make build` or set GPT_LOAD_BINARY',
  )
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

test.afterAll(() => {
  server?.kill()
  if (dataDir) rmSync(dataDir, { recursive: true, force: true })
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
        () =>
          (window as unknown as { __cspViolations?: string[] }).__cspViolations
            ?.length ?? 0,
      ),
    )
    .toBe(0)
  expect(consoleErrors).toEqual([])
}

test('Go server CSP keeps the classic document clean', async ({ page }) => {
  await expectCleanDocument(page, '/', '/assets/index')
})

test('Go server CSP keeps the Astryx document clean when opted in', async ({
  page,
  context,
}) => {
  await context.addCookies([
    { name: 'gpt-load.frontend', value: 'astryx', url: origin },
  ])
  await expectCleanDocument(page, '/settings', '/assets/astryx')
})
