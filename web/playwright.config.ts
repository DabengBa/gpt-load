import { fileURLToPath } from 'node:url'

import { defineConfig } from '@playwright/test'

const port = Number(process.env.PLAYWRIGHT_PORT ?? 41737)
if (!Number.isInteger(port) || port < 1024 || port > 65_535) {
  throw new Error('PLAYWRIGHT_PORT must be a valid loopback port')
}
const webRoot = fileURLToPath(new URL('.', import.meta.url)).replace(/\\/g, '/').replace(/\/$/, '')
const baseURL = `http://127.0.0.1:${port}`

// StorageState that pre-seeds the frontend preference cookie, mirroring
// internal/webui/server.go frontendCookieName.
const astryxFrontend = {
  cookies: [
    {
      name: 'gpt-load.frontend',
      value: 'astryx',
      url: baseURL,
      sameSite: 'Strict' as const,
    },
  ],
  origins: [],
}

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  reporter: 'line',
  projects: [
    // Classic frontend specs run without the preference cookie — the default
    // production behavior — and must never see Astryx-owned specs.
    {
      name: 'classic',
      testIgnore: [/astryx-.*\.spec\.ts/, /go-csp\.spec\.ts/],
    },
    // astryx-*.spec.ts exercise the Astryx entry; the project seeds the
    // selection cookie so document selection matches production opt-in.
    {
      name: 'astryx',
      testMatch: /astryx-.*\.spec\.ts/,
      use: { storageState: astryxFrontend },
    },
    // go-csp spawns the compiled Go binary itself; it skips when no binary
    // artifact exists for this platform.
    {
      name: 'go-csp',
      testMatch: /go-csp\.spec\.ts/,
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
