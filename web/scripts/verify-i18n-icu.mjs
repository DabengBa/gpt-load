import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'
import { parse } from '@formatjs/icu-messageformat-parser'

// Catalogs live in src/shared/i18n/locales and are consumed verbatim by
// react-intl (ICU) in the Astryx frontend. vue-i18n accepts literal and
// HTML-looking syntax that ICU rejects, so every message must parse cleanly
// and every locale must expose the same keys with the same placeholders.
const WEB_ROOT = fileURLToPath(new URL('..', import.meta.url))
const locales = ['zh-CN', 'en-US', 'ja-JP']
const namespaces = [
  'core',
  'import',
  'group',
  'access-keys',
  'monitor',
  'models',
  'model-prices',
  'settings',
]

function flatten(value, prefix = '') {
  const entries = []
  for (const [key, item] of Object.entries(value)) {
    const path = prefix === '' ? key : `${prefix}.${key}`
    if (typeof item === 'string') {
      entries.push([path, item])
    } else if (item !== null && typeof item === 'object' && !Array.isArray(item)) {
      entries.push(...flatten(item, path))
    }
  }
  return entries
}

function placeholderSet(message) {
  const names = new Set()
  const walk = (elements) => {
    for (const element of elements) {
      if (element.type === 1 || element.type === 8) {
        names.add(String(element.value))
      }
      for (const option of Object.values(element.options ?? {})) {
        walk(option.value)
      }
    }
  }
  walk(parse(message, { captureLocation: false }))
  return names
}

const server = await createServer({
  root: WEB_ROOT,
  // Catalogs are self-contained TS; skipping vite.config keeps plugin
  // handles (dev middleware, watchers) from holding the event loop open.
  configFile: false,
  logLevel: 'silent',
  server: { middlewareMode: true },
})
try {
  const failures = []
  const catalogs = {}
  for (const locale of locales) {
    catalogs[locale] = {}
    for (const namespace of namespaces) {
      const module = await server.ssrLoadModule(
        `/src/shared/i18n/locales/${locale}/${namespace}.ts`,
      )
      catalogs[locale][namespace] = flatten(module.default)
    }
  }

  // 1. Every message must parse as ICU. A few catalog entries are interpolation
  // fragments (passed as values into other messages), never parsed by ICU —
  // keep them out of the parse pass but still covered by the parity checks.
  const interpolationFragments = new Set(['import.subscription.callbackPortToken'])
  for (const locale of locales) {
    for (const namespace of namespaces) {
      for (const [path, message] of catalogs[locale][namespace]) {
        if (interpolationFragments.has(path)) continue
        try {
          parse(message, { captureLocation: false })
        } catch (error) {
          failures.push(`icu-parse ${locale}/${namespace}:${path} -> ${error.message}`)
        }
      }
    }
  }

  // 2. Key parity against en-US (the fallback catalog).
  // ja-JP predates full monitor coverage: these keys are absent today and the
  // UI falls back to en-US. Keep the allowlist so only new drift fails.
  const knownMissing = new Set([
    'ja-JP/monitor:monitor.logs.copySuccess',
    'ja-JP/monitor:monitor.logs.copyFailure',
    'ja-JP/monitor:monitor.logs.resultSummary',
    'ja-JP/monitor:monitor.logs.columns.timeNewestFirst',
    'ja-JP/monitor:monitor.logs.columns.tokensDetail',
    'ja-JP/monitor:monitor.logs.response.httpStatus',
  ])
  const seenMissing = new Set()
  for (const locale of locales.filter((item) => item !== 'en-US')) {
    for (const namespace of namespaces) {
      const expected = new Set(catalogs['en-US'][namespace].map(([path]) => path))
      const actual = new Set(catalogs[locale][namespace].map(([path]) => path))
      for (const key of expected) {
        const id = `${locale}/${namespace}:${key}`
        if (!actual.has(key)) {
          if (knownMissing.has(id)) {
            seenMissing.add(id)
          } else {
            failures.push(`missing-key ${id}`)
          }
        }
      }
      for (const key of actual) {
        if (!expected.has(key)) failures.push(`extra-key ${locale}/${namespace}:${key}`)
      }
    }
  }
  for (const id of knownMissing) {
    if (!seenMissing.has(id)) failures.push(`stale-allowlist ${id} (key now exists or moved)`)
  }

  // 3. Placeholder parity: every shared key must interpolate the same names.
  for (const namespace of namespaces) {
    const enMap = new Map(catalogs['en-US'][namespace])
    for (const locale of locales.filter((item) => item !== 'en-US')) {
      const map = new Map(catalogs[locale][namespace])
      for (const [path, message] of enMap) {
        const localized = map.get(path)
        if (localized === undefined) continue
        let enNames
        let localNames
        try {
          enNames = placeholderSet(message)
          localNames = placeholderSet(localized)
        } catch {
          continue // parse failures already recorded above
        }
        const missing = [...enNames].filter((name) => !localNames.has(name))
        const extra = [...localNames].filter((name) => !enNames.has(name))
        if (missing.length || extra.length) {
          failures.push(
            `placeholder-mismatch ${locale}/${namespace}:${path} ` +
              `missing=[${missing}] extra=[${extra}]`,
          )
        }
      }
    }
  }

  if (failures.length) {
    console.log('i18n ICU contract: FAIL')
    for (const failure of failures) console.log(`  ${failure}`)
    process.exitCode = 1
  } else {
    let total = 0
    for (const locale of locales) {
      for (const namespace of namespaces) total += catalogs[locale][namespace].length
    }
    console.log(`i18n ICU contract: PASS (${total} messages checked)`)
  }
} finally {
  await server.close()
}

// The stylex unplugin keeps worker handles alive after server.close();
// exit explicitly so the contract verdict does not hang the gate.
process.exit(process.exitCode ?? 0)
