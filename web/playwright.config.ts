import { fileURLToPath } from 'node:url'

import { defineConfig } from '@playwright/test'

const port = Number(process.env.PLAYWRIGHT_PORT ?? 41737)
if (!Number.isInteger(port) || port < 1024 || port > 65_535) {
  throw new Error('PLAYWRIGHT_PORT must be a valid loopback port')
}
const webRoot = fileURLToPath(new URL('.', import.meta.url))

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  reporter: 'line',
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: `env -u NODE_OPTIONS sh -c 'cd ${webRoot} && exec node ${webRoot}node_modules/vite/bin/vite.js ${webRoot} --config ${webRoot}vite.config.ts --host 127.0.0.1 --port ${port}'`,
    cwd: webRoot,
    port,
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
})
