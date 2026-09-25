#!/usr/bin/env node
// Cross-language contract: the frontend-selection cookie name is defined once
// in Go (internal/webui/server.go) and once in shared TS
// (src/shared/controllers/frontend-preference.ts). Both must agree, and the
// dev selector plus the classic switch control must consume the shared
// module rather than re-declaring the literal.

import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const webRoot = fileURLToPath(new URL('..', import.meta.url))
const repoRoot = fileURLToPath(new URL('../..', import.meta.url))

const goSource = readFileSync(`${repoRoot}internal/webui/server.go`, 'utf8')
const goMatch = goSource.match(/frontendCookieName\s*=\s*"([^"]+)"/)
if (goMatch === null) {
  console.error('FAIL: frontendCookieName not found in internal/webui/server.go')
  process.exit(1)
}

const sharedPath = `${webRoot}/src/shared/controllers/frontend-preference.ts`
const sharedSource = readFileSync(sharedPath, 'utf8')
const tsMatch = sharedSource.match(/frontendCookieName\s*=\s*'([^']+)'/)
if (tsMatch === null) {
  console.error(`FAIL: frontendCookieName not found in ${sharedPath}`)
  process.exit(1)
}

if (goMatch[1] !== tsMatch[1]) {
  console.error(
    `FAIL: cookie name drift — Go uses "${goMatch[1]}", shared TS uses "${tsMatch[1]}"`,
  )
  process.exit(1)
}

const consumers = [
  [`${webRoot}/vite.config.ts`, 'shared/controllers/frontend-preference'],
  [
    `${webRoot}/src/frontends/classic/features/preferences/PreferencesControl.vue`,
    'shared/controllers/frontend-preference',
  ],
]
for (const [path, needle] of consumers) {
  const source = readFileSync(path, 'utf8')
  if (!source.includes(needle)) {
    console.error(`FAIL: ${path} does not consume the shared frontend-preference module`)
    process.exit(1)
  }
}

console.log(`PASS: frontend cookie contract holds at "${goMatch[1]}"`)
