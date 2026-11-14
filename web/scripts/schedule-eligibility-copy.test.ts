import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import en from '../src/shared/i18n/locales/en-US/monitor.ts'

import zh from '../src/shared/i18n/locales/zh-CN/monitor.ts'

test('tier-demoted rows only describe schedulability in every locale', () => {
  assert.equal(zh.monitor.schedule.reasons.tier_demoted, '可调度')
  assert.equal(en.monitor.schedule.reasons.tier_demoted, 'Schedulable')
})

test('schedule target demo does not replace share with a priority label', () => {
  const demo = readFileSync(
    new URL('../../docs/design/schedule-center-target-demo.html', import.meta.url),
    'utf8',
  )
  assert.match(demo, /value="2"><\/div><div class="share">100\.0%<\/div>/)
})
