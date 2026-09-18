import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'

// web/ is the parent of scripts/, used as vite root so the '@' alias resolves.
const WEB_ROOT = fileURLToPath(new URL('..', import.meta.url))
const FIXTURES_DIR = fileURLToPath(new URL('./fixtures/', import.meta.url))

// Each fixture carries its expected verdict. The real projector in
// src/frontends/classic/app/resources/health.ts is the single source of truth; this script only
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
  {
    name: 'recovery-cooldown-expiry',
    file: 'recovery-cooldown-expiry.json',
    expect: 'success',
    expectRecovery: { automatic: true, mode: 'cooldown_expiry', at_ms: 1700000005000 },
  },
  {
    name: 'recovery-scheduled-release',
    file: 'recovery-scheduled-release.json',
    expect: 'success',
    expectRecovery: { automatic: true, mode: 'scheduled_release', at_ms: 1700000005000 },
  },
  {
    name: 'recovery-scheduled-release-null-at-ms',
    file: 'recovery-scheduled-release-null-at-ms.json',
    expect: 'success',
    expectRecovery: { automatic: true, mode: 'scheduled_release', at_ms: null },
  },
  {
    name: 'recovery-scheduled-release-non-automatic',
    file: 'recovery-scheduled-release-non-automatic.json',
    expect: 'invalid',
  },
  {
    name: 'recovery-scheduled-release-with-at-ms',
    file: 'recovery-scheduled-release-with-at-ms.json',
    expect: 'invalid',
  },
  {
    name: 'recovery-cooldown-expiry-missing-at-ms',
    file: 'recovery-cooldown-expiry-missing-at-ms.json',
    expect: 'invalid',
  },
  { name: 'recovery-invalid-mode', file: 'recovery-invalid-mode.json', expect: 'invalid' },
  { name: 'recovery-invalid-at-ms', file: 'recovery-invalid-at-ms.json', expect: 'invalid' },
]

function readJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'))
}

function findRecovery(payload) {
  for (const collection of ['cooldown_credentials', 'blacklisted_credentials']) {
    const record = payload[collection]?.[0]
    if (record?.recovery !== undefined) return record.recovery
  }
  return null
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
    const health = await server.ssrLoadModule('/src/frontends/classic/app/resources/health.ts')
    // Same module instance health.ts imports (the '@/api/errors' alias).
    const errors = await server.ssrLoadModule('/src/shared/http/errors.ts')
    const { InvalidResponseError } = errors
    const projectRuntimeHealth = health.projectRuntimeHealth

    for (const c of CASES) {
      const payload = readJson(FIXTURES_DIR + c.file)
      const inputRecovery = findRecovery(payload)
      const expected = c.expect === 'success' ? 'success' : 'InvalidResponseError'
      let actual = 'success'
      let outputRecovery = null
      let detail = ''

      try {
        const result = projectRuntimeHealth(payload)
        outputRecovery = findRecovery(result)
        if (c.expect !== 'success') {
          actual = 'success'
        } else if (c.expectError !== undefined && result.debug_capture.error !== c.expectError) {
          actual = 'success'
          detail = `error mismatch: expected "${c.expectError}" got "${result.debug_capture.error}"`
        } else if (
          c.expectRecovery !== undefined &&
          JSON.stringify(outputRecovery) !== JSON.stringify(c.expectRecovery)
        ) {
          actual = 'success'
          detail = `recovery mismatch: expected ${JSON.stringify(c.expectRecovery)} got ${JSON.stringify(outputRecovery)}`
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
      const output = actual === 'success' ? { recovery: outputRecovery } : { error: actual }
      if (ok) {
        console.log(
          `PASS  ${c.name}: input.recovery=${JSON.stringify(inputRecovery)} output=${JSON.stringify(output)}`,
        )
      } else {
        failed++
        console.log(
          `FAIL  ${c.name}: input.recovery=${JSON.stringify(inputRecovery)} output=${JSON.stringify(output)} expected=${expected}${detail ? ` detail=${detail}` : ''}`,
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
