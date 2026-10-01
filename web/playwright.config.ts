import { fileURLToPath } from 'node:url'

import { defineConfig } from '@playwright/test'

import { resolveChromium125Executable } from './e2e/browser-executables'

const port = Number(process.env.PLAYWRIGHT_PORT ?? 41737)
if (!Number.isInteger(port) || port < 1024 || port > 65_535) {
  throw new Error('PLAYWRIGHT_PORT must be a valid loopback port')
}
const webRoot = fileURLToPath(new URL('.', import.meta.url))
  .replace(/\\/g, '/')
  .replace(/\/$/, '')
const baseURL = `http://127.0.0.1:${port}`

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  reporter: 'line',
  projects: [
    // Every UI spec exercises the single Astryx document.
    {
      name: 'astryx',
      testMatch: /astryx-.*\.spec\.ts/,
    },
    // go-csp spawns the compiled Go binary itself; it skips when no binary
    // artifact exists for this platform.
    {
      name: 'go-csp',
      testMatch: /go-csp\.spec\.ts/,
    },
    // B13 browser-floor gate: the same Go-CSP sweep under a real
    // Chrome/Chromium 125 binary. Resolution lives in
    // e2e/browser-executables.ts; the spec skips when no 125 binary is
    // provisioned rather than silently running the bundled Chromium.
    {
      name: 'chromium-125',
      testMatch: /go-csp\.spec\.ts/,
      use: {
        launchOptions: {
          executablePath: resolveChromium125Executable(),
        },
      },
    },
    // The other half of the gate matrix: whatever stable Chrome is installed
    // on the host. `channel: 'chrome'` picks up system Chrome.
    {
      name: 'chrome-latest',
      testMatch: /go-csp\.spec\.ts/,
      use: { channel: 'chrome' },
    },
  ],
  use: {
    baseURL,
    locale: 'en-US',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: `node "${webRoot}/node_modules/vite/bin/vite.js" "${webRoot}" --config "${webRoot}/vite.config.ts" --host 127.0.0.1 --port ${port}`,
    cwd: webRoot,
    env: { NODE_OPTIONS: '' },
    port,
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
})
