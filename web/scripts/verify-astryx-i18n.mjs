// Verifies the Astryx component catalogs we wire into
// InternationalizationProvider stay key-aligned with the shipped `en`
// catalog, and that the provider wiring covers every AppLocale.
// Run from the repo root or web/: `node scripts/verify-astryx-i18n.mjs`.
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

const webRoot = fileURLToPath(new URL('..', import.meta.url))
const coreLocales = (locale) =>
  `${webRoot}node_modules/@astryxdesign/core/locales/${locale}.json`

const astryxLocales = ['en', 'zh-CN', 'ja-JP']

const catalogs = {}
for (const locale of astryxLocales) {
  try {
    catalogs[locale] = JSON.parse(await readFile(coreLocales(locale), 'utf8'))
  } catch (error) {
    console.error(`FAIL: cannot read @astryxdesign/core locale "${locale}": ${error.message}`)
    process.exit(1)
  }
}

let failures = 0
const enKeys = new Set(Object.keys(catalogs.en))
for (const locale of astryxLocales) {
  if (locale === 'en') continue
  const keys = new Set(Object.keys(catalogs[locale]))
  // Missing keys fall back to the bundled `en` catalog per-message — that is
  // upstream translation lag, reported but non-fatal. Extra keys mean a
  // renamed/removed upstream key the fallback can no longer cover — fatal.
  const missing = [...enKeys].filter((key) => !keys.has(key))
  const extra = [...keys].filter((key) => !enKeys.has(key))
  if (missing.length > 0) {
    console.warn(
      `WARN: astryx locale "${locale}" is missing ${missing.length}/${enKeys.size} ` +
        `en.json keys (upstream lag; those strings fall back to English)`,
    )
  }
  if (extra.length > 0) {
    failures += extra.length
    console.error(
      `FAIL: astryx locale "${locale}" has ${extra.length} keys absent from en.json ` +
        `(renamed upstream keys?): ${extra.slice(0, 5).join(', ')}`,
    )
  }
}

// The provider wiring must pass a catalog for every AppLocale except en-US
// (whose strings ship inside the components themselves).
const i18nSource = await readFile(
  `${webRoot}src/frontends/astryx/app/i18n.tsx`,
  'utf8',
)
for (const tag of ['zh-CN', 'ja-JP']) {
  if (!i18nSource.includes(`'${tag}'`)) {
    failures += 1
    console.error(`FAIL: app/i18n.tsx does not wire an astryx catalog for "${tag}"`)
  }
}

if (failures > 0) process.exit(1)
console.log(
  `PASS: astryx locale catalogs aligned across ${astryxLocales.join('/')} ` +
    `(${enKeys.size} keys) and provider wiring covers all app locales`,
)
