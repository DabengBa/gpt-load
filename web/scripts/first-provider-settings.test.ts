import assert from 'node:assert/strict'
import test from 'node:test'

import { isValidPositiveCount } from '../src/shared/domain/settings/settings-patch.ts'
import { runtimeSettingKeys } from '../src/shared/control/resources/settings.ts'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

test('first_provider_attempts is a runtime setting key', () => {
  assert.equal((runtimeSettingKeys as readonly string[]).includes('first_provider_attempts'), true)
})

test('first_provider_attempts validation requires at least one attempt', () => {
  assert.equal(isValidPositiveCount(1), true)
  assert.equal(isValidPositiveCount(3), true)
  assert.equal(isValidPositiveCount(0), false)
  assert.equal(isValidPositiveCount(-1), false)
  assert.equal(isValidPositiveCount(1.5), false)
  assert.equal(isValidPositiveCount(Number.NaN), false)
})

test('all three locales document first_provider_attempts and its error message', () => {
  for (const locale of ['en-US', 'zh-CN', 'ja-JP']) {
    const path = fileURLToPath(
      new URL(`../src/shared/i18n/locales/${locale}/settings.ts`, import.meta.url),
    )
    const source = readFileSync(path, 'utf8')
    assert.ok(source.includes('first_provider_attempts'), `${locale} missing field label`)
    assert.ok(
      source.includes('firstProviderAttemptsHelp'),
      `${locale} missing field help`,
    )
    assert.ok(source.includes('positiveCountError'), `${locale} missing validation message`)
  }
})
