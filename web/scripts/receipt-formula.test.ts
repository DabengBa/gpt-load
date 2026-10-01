import assert from 'node:assert/strict'
import test from 'node:test'
import { formatReceiptFormulaLine } from '../src/shared/domain/monitor/receipt-formula.ts'

test('frozen component formula preserves cache coefficient and does not recalculate amount', () => {
  assert.equal(
    formatReceiptFormulaLine(
      {
        code: 'cache_write_1h',
        quantity: '10',
        rate_nano_usd_per_million: '1000000000',
        multiplier: { numerator: '8', denominator: '5' },
        state: 'priced',
        amount_nano_usd: '16000',
      },
      'en-US',
    ),
    '10 × $1.00/1M × 8/5',
  )
})
test('unpriced component is explicit', () => {
  assert.equal(
    formatReceiptFormulaLine(
      {
        code: 'input',
        quantity: '10',
        rate_nano_usd_per_million: null,
        multiplier: { numerator: '1', denominator: '1' },
        state: 'unpriced',
        amount_nano_usd: null,
      },
      'en-US',
    ),
    '10 × —',
  )
})
