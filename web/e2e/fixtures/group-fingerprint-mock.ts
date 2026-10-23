import type { Page, Route } from '@playwright/test'

const RUNTIME = {
  first_byte_timeout: 30,
  request_timeout: 60,
  stream_idle_timeout: 30,
  blacklist_threshold: 3,
  header_rules: { set: {}, remove: [] },
  affinity_enabled: false,
  responses_websocket_enabled: false,
  responses_reasoning_status_filter_enabled: false,
  parameter_overrides: [],
}

export const fingerprintLogID = '9f1c2b3a-4d5e-4f60-8a9b-0c1d2e3f4a5b'

const MODEL_ROW = {
  id: 'gpt-5.4',
  concurrency_limit: 3,
  alias: '',
  alias_enabled: false,
  client_model: 'gpt-5.4',
  test_alias: 'a4g233',
  entry_id: 'entry-1',
  weight: null,
  priority: null,
  pricing_status: 'configured',
}

// Mirrors control.ModelTraceProbeResponse with the analysis/failure either-or.
export function fingerprintSuccess() {
  return {
    group_id: 1,
    group_name: 'Fingerprint Group',
    model: 'gpt-5.4',
    service: { base_url: 'http://modeltrace:8080', enabled: true },
    challenges: [
      {
        id: 'query-01',
        expected_count: 218,
        outcome: 'succeeded',
        reason: null,
        log_id: fingerprintLogID,
        latency_ms: 1_240,
        status_code: 200,
      },
      {
        id: 'query-02',
        expected_count: 233,
        outcome: 'succeeded',
        reason: null,
        log_id: fingerprintLogID,
        latency_ms: 1_310,
        status_code: 200,
      },
    ],
    analysis: {
      prediction: 'gpt-5.4',
      prediction_name: 'GPT-5.4',
      probability: 0.912,
      family_prediction: 'gpt',
      family_probability: 0.93,
      used_outputs: 2,
      method: 'global_hellinger',
      candidates: [
        { model: 'gpt-5.4', family: 'gpt', probability: 0.912, conditional_probability: 0.98 },
        { model: 'gpt-5.6-sol', family: 'gpt', probability: 0.081, conditional_probability: 0.02 },
      ],
      diagnostics: [
        { prompt_id: 'query-01', accepted: true, parsed_numbers: 218, minimum_numbers: 200 },
      ],
      bank: { models: 18, built_at: '2026-03-01T00:00:00Z', bank_sha256: 'abc123' },
      upstream: { repository: 'xqy2006/ModelTrace', commit: 'deadbeef' },
    },
    failure: null,
    tested_at_ms: 1_792_714_550_837,
  }
}

export interface FingerprintMock {
  calls: unknown[]
  failNext(code: string): void
}

export async function mockGroupFingerprint(page: Page): Promise<FingerprintMock> {
  const calls: unknown[] = []
  const failures: string[] = []
  await page.addInitScript((key) => {
    window.localStorage.setItem('gpt-load.auth-key', key)
    window.localStorage.setItem('gpt-load.locale', 'en-US')
  }, 'e2e-auth-key')
  await page.route('**/api/**', async (route: Route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    const fulfill = (data: unknown, status = 200, code = 0, message = 'ok') =>
      route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify(status === 200 ? { code, message, data } : { code, message }),
      })
    if (path === '/api/auth/session')
      return fulfill({ authenticated: true, principal_type: 'admin' })
    if (path === '/api/channels')
      return fulfill({
        items: [
          {
            channel_id: 'openai',
            name: 'OpenAI',
            mark: 'OA',
            icon: 'openai',
            search_terms: ['openai'],
            description: '',
            default_base_url: 'https://api.openai.com',
            param_fields: [],
            credential_fields: [],
            connection: { type: 'api_key', credential_input: 'batch_text' },
            capabilities: { model_discovery: false, outbound_proxy: false },
            routes: [
              {
                client_protocol: 'openai-completions',
                operation: 'chat_completion',
                route_mode: 'native',
                model_dependent: false,
                possible_modes: [],
              },
              {
                client_protocol: 'openai-completions',
                operation: 'probe',
                route_mode: 'native',
                model_dependent: false,
                possible_modes: [],
              },
            ],
            client_protocols: ['openai-completions'],
          },
        ],
        total: 1,
      })
    const groupBase = {
      id: 1,
      name: 'Fingerprint Group',
      channel_id: 'openai',
      connection_type: 'api_key',
      params: {},
      provider_url: null,
    }
    if (path === '/api/groups/1')
      return fulfill({
        ...groupBase,
        service_status: 'available',
        service_status_reason: null,
        credential_configured: true,
        credential_status: 'available',
        model_count: 1,
      })
    if (path === '/api/groups/1/settings')
      return fulfill({
        name: groupBase.name,
        channel_id: groupBase.channel_id,
        connection_type: groupBase.connection_type,
        params: groupBase.params,
        provider_url: groupBase.provider_url,
        enabled: true,
        overrides: RUNTIME,
        effective: {
          first_byte_timeout: RUNTIME.first_byte_timeout,
          request_timeout: RUNTIME.request_timeout,
          stream_idle_timeout: RUNTIME.stream_idle_timeout,
          blacklist_threshold: RUNTIME.blacklist_threshold,
          header_rules: RUNTIME.header_rules,
          affinity_enabled: RUNTIME.affinity_enabled,
          responses_websocket_enabled: RUNTIME.responses_websocket_enabled,
          responses_reasoning_status_filter_enabled:
            RUNTIME.responses_reasoning_status_filter_enabled,
        },
        proxy: {
          configured_mode: 'inherit',
          effective_mode: 'direct',
          effective_source: 'default',
          has_auth: false,
        },
      })
    if (path === '/api/groups/1/models')
      return fulfill({ items: [MODEL_ROW], total: 1, pending: 0 })
    if (path === '/api/groups/1/credential') return fulfill({ credential: null })
    if (path === '/api/modeltrace-probe') {
      calls.push(request.postDataJSON())
      const failure = failures.shift()
      if (failure !== undefined) {
        return fulfill(
          undefined,
          503,
          failure,
          'ModelTrace fingerprint detection is not configured',
        )
      }
      return fulfill(fingerprintSuccess())
    }
    if (path === '/api/logs') return fulfill({ items: [], next_cursor: null })
    return fulfill({})
  })
  return {
    calls,
    failNext: (code: string) => failures.push(code),
  }
}
