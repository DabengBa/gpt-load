import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'

// web/ is the parent of scripts/, used as vite root so the '@' alias resolves.
const WEB_ROOT = fileURLToPath(new URL('..', import.meta.url))
const FIXTURES_DIR = fileURLToPath(new URL('./fixtures/', import.meta.url))

// Each fixture carries its expected verdict. The real projector in
// src/app/resources/health.ts is the single source of truth; this script only
// loads it and asserts the verdict per fixture.
const CASES = [
  { name: 'default', file: 'default.json', expect: 'success' },
  {
    name: 'counts-unavailable',
    file: 'counts-unavailable.json',
    expect: 'success',
    expectError: 'counts_unavailable',
  },
  { name: 'full', file: 'full.json', expect: 'success' },
  { name: 'unknown-nested-key', file: 'unknown-nested-key.json', expect: 'invalid' },
  { name: 'unknown-top-level-key', file: 'unknown-top-level-key.json', expect: 'invalid' },
]

function readJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'))
}

async function main() {
  const server = await createServer({
    root: WEB_ROOT,
    server: { middlewareMode: true },
    appType: 'custom',
    logLevel: 'error',
  })

  let failed = 0
  try {
    const health = await server.ssrLoadModule('/src/app/resources/health.ts')
    // Same module instance health.ts imports (the '@/api/errors' alias).
    const errors = await server.ssrLoadModule('/src/api/errors.ts')
    const { InvalidResponseError } = errors
    const projectRuntimeHealth = health.projectRuntimeHealth

    for (const c of CASES) {
      const payload = readJson(FIXTURES_DIR + c.file)
      const expected = c.expect === 'success' ? 'success' : 'InvalidResponseError'
      let actual = 'success'
      let detail = ''

      try {
        const result = projectRuntimeHealth(payload)
        if (c.expect !== 'success') {
          actual = 'success'
        } else if (c.expectError !== undefined && result.debug_capture.error !== c.expectError) {
          actual = 'success'
          detail = `error mismatch: expected "${c.expectError}" got "${result.debug_capture.error}"`
        }
      } catch (err) {
        actual =
          err instanceof InvalidResponseError
            ? 'InvalidResponseError'
            : `threw:${err?.constructor?.name ?? err}`
        if (c.expect !== 'invalid') {
          detail = err instanceof Error ? err.message : String(err)
        }
      }

      const ok = actual === expected && detail === ''
      if (ok) {
        console.log(`PASS  ${c.name}: expected=${expected} actual=${actual}`)
      } else {
        failed++
        console.log(
          `FAIL  ${c.name}: expected=${expected} actual=${actual}${detail ? ` detail=${detail}` : ''}`,
        )
      }
    }

    const payloadIndex = process.argv.indexOf('--payload')
    if (payloadIndex !== -1 && payloadIndex + 1 < process.argv.length) {
      const payloadPath = process.argv[payloadIndex + 1]
      const payload = readJson(payloadPath)
      try {
        projectRuntimeHealth(payload)
        console.log(`PASS  --payload ${payloadPath}: success`)
      } catch (err) {
        failed++
        const actual =
          err instanceof InvalidResponseError
            ? 'InvalidResponseError'
            : `threw:${err?.constructor?.name ?? err}`
        console.log(`FAIL  --payload ${payloadPath}: expected=success actual=${actual}`)
      }
    }
  } finally {
    await server.close()
  }

  return failed
}

const failed = await main()
if (failed > 0) {
  process.exitCode = 1
}
