import assert from 'node:assert/strict'
import test from 'node:test'

import { projectChannelList } from '../src/shared/control/resources/channels.ts'
import { channelOperations } from '../src/shared/control/resources/channel-contract.ts'
import { projectRequestLogDetail } from '../src/shared/control/resources/request-logs.ts'

test('Codex standalone search is no longer a channel operation', () => {
  assert.equal((channelOperations as readonly string[]).includes('web_search'), false)
})

const channel = {
  channel_id: 'codex_downstream',
  name: 'Codex downstream',
  mark: 'CX',
  icon: 'codex',
  search_terms: ['codex'],
  description: '',
  default_base_url: 'https://api.openai.com',
  param_fields: [],
  credential_fields: [
    {
      key: 'api_key',
      label: 'API key',
      input_kind: 'secret',
      required: true,
      sensitive: true,
      default_value: null,
    },
  ],
  connection: { type: 'api_key', credential_input: 'batch_text' },
  capabilities: { model_discovery: true, outbound_proxy: true },
  routes: [
    {
      client_protocol: 'openai-responses',
      operation: 'responses_create',
      route_mode: 'native',
      model_dependent: false,
      possible_modes: [],
    },
    {
      client_protocol: 'openai-responses',
      operation: 'web_search',
      route_mode: 'native',
      model_dependent: false,
      possible_modes: [],
    },
  ],
  client_protocols: ['openai-responses'],
}

test('channel projection rejects a retired standalone search route', () => {
  assert.throws(() => projectChannelList({ items: [channel], total: 1 }))
  assert.deepEqual(
    projectChannelList({ items: [{ ...channel, routes: [channel.routes[0]] }], total: 1 }).items[0]
      ?.routes[0]?.operation,
    'responses_create',
  )
})

const attempt = {
  sequence: 1,
  group_id: 1,
  group_name: 'group',
  channel_id: 'openai',
  credential_id: 1,
  credential_name: 'credential',
  operation: 'chat_completion',
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
  failure_category: 'invalid_key',
  failure_origin: 'upstream',
  failure_scope: 'credential',
  retry_directive: 'next_candidate',
  effect: 'none',
  rule_id: null,
  action: 'terminate',
  will_retry: false,
  error_code: '',
  error_summary: '',
  committed: true,
  pricing_receipt: null,
}

function logDetail(overrides: Record<string, unknown>) {
  return {
    request_id: 'aaaaaaaa-1111-4111-8111-111111111111',
    completed_at_ms: 1,
    access_key: { id: 1, name: 'key', deleted: false },
    protocol: 'openai-completions',
    operation: 'chat_completion',
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
    final_attempt_duration_ms: 1,
    attempt_count: 1,
    error_code: '',
    error_summary: '',
    affinity_hit: false,
    resolved_affinity_credential_id: 0,
    first_attempt_credential_id: 0,
    affinity_hit_assignment_reached: false,
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
    attempts: [{ ...attempt, ...overrides }],
  }
}

test('request log projection rejects retired retry and operation values', () => {
  assert.equal(
    projectRequestLogDetail(logDetail({})).attempts[0]?.retry_directive,
    'next_candidate',
  )
  assert.throws(
    () => projectRequestLogDetail(logDetail({ retry_directive: 'refresh_credential' })),
    { name: 'InvalidResponseError' },
  )
  assert.throws(() => projectRequestLogDetail(logDetail({ operation: 'web_search' })), {
    name: 'InvalidResponseError',
  })
  assert.throws(() => projectRequestLogDetail({ ...logDetail({}), operation: 'web_search' }), {
    name: 'InvalidResponseError',
  })
  assert.equal(projectRequestLogDetail(logDetail({})).attempts[0]?.operation, 'chat_completion')
})
