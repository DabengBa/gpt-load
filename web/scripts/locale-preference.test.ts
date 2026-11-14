import assert from 'node:assert/strict'
import test from 'node:test'

import { getBrowserLocale, supportedLocales } from '../src/shared/preferences/locale.ts'

function resolveLocale(stored: string | null, languages: string[], language = languages[0] ?? '') {
  const windowDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'window')
  const navigatorDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator')
  try {
    Object.defineProperty(globalThis, 'window', {
      configurable: true,
      value: { localStorage: { getItem: () => stored } },
    })
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: { languages, language },
    })
    return getBrowserLocale()
  } finally {
    if (windowDescriptor) Object.defineProperty(globalThis, 'window', windowDescriptor)
    else Reflect.deleteProperty(globalThis, 'window')
    if (navigatorDescriptor) Object.defineProperty(globalThis, 'navigator', navigatorDescriptor)
    else Reflect.deleteProperty(globalThis, 'navigator')
  }
}

test('only Chinese and English are offered', () => {
  assert.deepEqual(supportedLocales, ['zh-CN', 'en-US'])
})

test('obsolete Japanese preference falls through to browser languages', () => {
  assert.equal(resolveLocale('ja-JP', ['ja-JP', 'zh-TW', 'en-US']), 'zh-CN')
  assert.equal(resolveLocale('ja-JP', ['ja-JP']), 'en-US')
})

test('browser selection skips Japanese and preserves Chinese and English priority', () => {
  assert.equal(resolveLocale(null, ['ja-JP', 'en-GB', 'zh-CN']), 'en-US')
  assert.equal(resolveLocale(null, ['ja-JP', 'zh-TW', 'en-US']), 'zh-CN')
  assert.equal(resolveLocale(null, ['ja-JP']), 'en-US')
  assert.equal(resolveLocale(null, ['fr'], 'zh-CN'), 'zh-CN')
})

test('supported stored preferences still take precedence over browser languages', () => {
  assert.equal(resolveLocale('en-US', ['zh-CN']), 'en-US')
  assert.equal(resolveLocale('zh-CN', ['en-US']), 'zh-CN')
})
