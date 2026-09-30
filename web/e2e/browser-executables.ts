import { existsSync, readdirSync } from 'node:fs'
import { homedir, platform } from 'node:os'
import { join } from 'node:path'

// B13: the browser-floor gate needs a real Chrome/Chromium 125 binary — the
// bundled Chromium is always latest and cannot prove the support matrix.
// Resolution order:
//   1. GPT_LOAD_CHROME_125_EXE — explicit override (CI caches, local installs)
//   2. The @puppeteer/browsers cache — `npx @puppeteer/browsers install
//      chrome@125` is the provisioning path named by the migration plan
// The result is shared between playwright.config.ts (launchOptions) and
// go-csp.spec.ts (skip guard) so an unprovisioned machine reports a skip
// instead of silently running the bundled Chromium.

const EXECUTABLE_CANDIDATES: Record<string, string[]> = {
  win32: [join('chrome-win64', 'chrome.exe'), join('chrome-win', 'chrome.exe')],
  linux: [join('chrome-linux64', 'chrome'), join('chrome-linux', 'chrome')],
  darwin: [
    join(
      'chrome-mac-arm64',
      'Google Chrome for Testing.app',
      'Contents',
      'MacOS',
      'Google Chrome for Testing',
    ),
    join(
      'chrome-mac-x64',
      'Google Chrome for Testing.app',
      'Contents',
      'MacOS',
      'Google Chrome for Testing',
    ),
  ],
}

export function resolveChromium125Executable(): string | undefined {
  const override = process.env.GPT_LOAD_CHROME_125_EXE
  if (override && existsSync(override)) return override

  const cacheDir = join(homedir(), '.cache', 'puppeteer', 'chrome')
  const subpaths = EXECUTABLE_CANDIDATES[platform()] ?? []
  if (subpaths.length === 0 || !existsSync(cacheDir)) return undefined

  // Puppeteer cache dirs are named `<platform>-<version>`; accept any 125.x.
  for (const dirName of readdirSync(cacheDir)) {
    if (!/-125\.\d+\.\d+\.\d+$/.test(dirName)) continue
    for (const subpath of subpaths) {
      const candidate = join(cacheDir, dirName, subpath)
      if (existsSync(candidate)) return candidate
    }
  }
  return undefined
}
