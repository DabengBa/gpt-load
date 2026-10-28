import assert from 'node:assert/strict'
import test from 'node:test'

import {
  normalizeUsageBreakdownSort,
  projectUsageBreakdown,
} from '../src/shared/control/resources/usage.ts'

function aggregate(overrides: Record<string, number | string> = {}) {
  return {
    request_count: 1,
    success_count: 1,
    failure_count: 0,
    uncached_input_tokens: 0,
    cache_read_tokens: 0,
    cache_write_5m_tokens: 0,
    cache_write_1h_tokens: 0,
    cache_write_unknown_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
    estimated_cost_nano_usd: '0',
    duration_ms_total: 0,
    duration_sample_count: 0,
    first_response_ms_total: 0,
    first_response_sample_count: 0,
    usage_missing_count: 0,
    partial_count: 0,
    unpriced_request_count: 0,
    pricing_partial_count: 0,
    success_output_tokens: 42,
    success_duration_ms_total: 900,
    success_efficiency_sample_count: 1,
    ...overrides,
  }
}

function breakdown(row: ReturnType<typeof aggregate>) {
  return {
    scope: 'admin',
    rows: [
      {
        ...row,
        model: 'model',
        group_id: 1,
        channel_id: 'openai',
        attempt_count: 1,
        attempt_failure_count: 0,
        normal_attempt_count: 1,
        slow_attempt_count: 0,
        faulty_attempt_count: 0,
      },
    ],
    total: aggregate(),
    attempt_total: {
      attempt_count: 1,
      attempt_failure_count: 0,
      normal_attempt_count: 1,
      slow_attempt_count: 0,
      faulty_attempt_count: 0,
    },
    pagination: { page: 1, page_size: 20, total_items: 1, total_pages: 1 },
  }
}

test('projects success-only usage fields from breakdown rows and totals', () => {
  const result = projectUsageBreakdown(breakdown(aggregate()))
  assert.equal(result.rows[0]?.success_output_tokens, 42)
  assert.equal(result.rows[0]?.success_duration_ms_total, 900)
  assert.equal(result.rows[0]?.success_efficiency_sample_count, 1)
  assert.equal(result.total.success_output_tokens, 42)
})

test('rejects success efficiency sample counts above successful requests', () => {
  assert.throws(() =>
    projectUsageBreakdown(
      breakdown(
        aggregate({
          success_count: 0,
          success_efficiency_sample_count: 1,
        }),
      ),
    ),
  )
})

test('accepts cache rate and token efficiency as breakdown sort keys', () => {
  assert.equal(normalizeUsageBreakdownSort('cache_rate'), 'cache_rate')
  assert.equal(normalizeUsageBreakdownSort('token_efficiency'), 'token_efficiency')
})
