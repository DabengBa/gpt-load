import assert from 'node:assert/strict'
import test from 'node:test'

import { projectRequestLogDetail } from '../src/shared/control/resources/request-logs.ts'
import { projectGroupSummary } from '../src/shared/control/resources/groups.ts'

test('group DTO rejects retired price_multiplier', () => {
  const group = {
    id: 1,
    name: 'group',
    channel_id: 'openai',
    connection_type: 'api_key',
    params: {},
    provider_url: null,
    service_status: 'available',
    service_status_reason: null,
    credential_count: 1,
    model_count: 1,
  }
  assert.equal(projectGroupSummary(group).name, 'group')
  assert.throws(() => projectGroupSummary({ ...group, price_multiplier: '1' }))
})

const line = {
  code: 'input',
  quantity: '10',
  rate_nano_usd_per_million: '1000000',
  multiplier: { numerator: '1', denominator: '1' },
  state: 'priced',
  amount_nano_usd: '10',
}

function detail(receipt: Record<string, unknown>) {
  return {
    request_id: 'aaaaaaaa-1111-4111-8111-111111111111',
    completed_at_ms: 1,
    access_key: { id: 1, name: 'key', deleted: false },
    protocol: 'openai-completions',
    operation: null,
    upstream_protocol: null,
    client_model: 'model',
    upstream_model: 'model',
    upstream_reported_model: 'model',
    feedback_status: null,
    feedback_reason: null,
    provider_first_response_ms: null,
    provider_tokens_per_second: null,
    model_consistency: 'match',
    reasoning: null,
    status: 'success',
    status_code: 200,
    stream: false,
    first_response_ms: null,
    duration_ms: 1,
    attempt_count: 1,
    error_code: '',
    error_summary: '',
    affinity_hit: false,
    continuity_hit: false,
    affinity_source: 'none',
    affinity_state: 'no_signal',
    affinity_key: null,
    group_id: 1,
    channel_id: 'openai',
    credential_id: 1,
    credential_name: 'credential',
    route_mode: 'native',
    usage_state: 'complete',
    cost_state: 'priced',
    pricing_completeness: 'complete',
    pricing_mode: 'standard',
    context_threshold_tokens: null,
    input_tokens: '10',
    cache_read_tokens: '0',
    cache_write_5m_tokens: '0',
    cache_write_1h_tokens: '0',
    cache_write_unknown_tokens: '0',
    output_tokens: '0',
    estimated_cost_nano_usd: '10',
    attempts: [
      {
        sequence: 1,
        group_id: 1,
        group_name: 'group',
        channel_id: 'openai',
        credential_id: 1,
        credential_name: 'credential',
        operation: null,
        route_mode: 'native',
        upstream_model: 'model',
        upstream_request_id: null,
        dispatch_state: 'not_sent',
        feedback_status: null,
        feedback_reason: null,
        provider_first_response_ms: null,
        provider_tokens_per_second: null,
        response_started: true,
        upstream_protocol: null,
        reasoning: null,
        status_code: 200,
        duration_ms: 1,
        failure_category: 'ok',
        failure_origin: null,
        failure_scope: null,
        retry_directive: null,
        effect: null,
        rule_id: null,
        action: 'terminate',
        will_retry: false,
        error_code: '',
        error_summary: '',
        committed: true,
        pricing_receipt: receipt,
      },
    ],
  }
}

test('accepts only schema 7 receipts without multiplier totals', () => {
  const receipt = {
    schema_version: 7,
    method: 'unit_rate_sum',
    method_version: 1,
    currency: 'USD',
    pricing_mode: 'standard',
    rule: { channel_id: 'openai', model_id: 'model' },
    context_threshold_tokens: null,
    line_items: [line],
    total_nano_usd: '10',
  }
  const projected = projectRequestLogDetail(detail(receipt))
  assert.equal(projected.attempts[0]?.pricing_receipt?.schema_version, 7)
})

test('rejects legacy receipt schemas and multiplier fields', () => {
  const base = {
    schema_version: 7,
    method: 'unit_rate_sum',
    method_version: 1,
    currency: 'USD',
    pricing_mode: 'standard',
    rule: { channel_id: 'openai', model_id: 'model' },
    context_threshold_tokens: null,
    line_items: [line],
    total_nano_usd: '10',
  }
  assert.throws(() => projectRequestLogDetail(detail({ ...base, schema_version: 6 })))
  assert.throws(() =>
    projectRequestLogDetail(
      detail({ ...base, price_multipliers: { group: '1', access_key: '1' } }),
    ),
  )
  assert.throws(() => projectRequestLogDetail(detail({ ...base, base_total_nano_usd: '10' })))
  assert.throws(() => projectRequestLogDetail(detail({ ...base, rule: { model_id: 'model' } })))
  assert.throws(() =>
    projectRequestLogDetail(
      detail({ ...base, rule: { ...base.rule, scope_key: 'provider:openai' } }),
    ),
  )
})
