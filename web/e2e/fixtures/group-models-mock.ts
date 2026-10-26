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

const CANDIDATES = [
  {
    id: 'gpt-5.2',
    name: 'GPT-5.2',
    sources: ['live'],
    pricing_status: 'pending',
    pricing_source: null,
  },
  {
    id: 'gpt-5.2-mini',
    name: 'GPT-5.2 Mini',
    sources: ['catalog'],
    pricing_status: 'configured',
    pricing_source: 'OpenAI',
  },
]

export async function mockGroupModels(page: Page): Promise<{ puts: unknown[] }> {
  const puts: unknown[] = []
  await page.addInitScript((key) => {
    window.localStorage.setItem('gpt-load.auth-key', key)
    window.localStorage.setItem('gpt-load.locale', 'en-US')
  }, 'e2e-auth-key')
  await page.route('**/api/**', async (route: Route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    const fulfill = (data: unknown) =>
      route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ code: 0, message: 'ok', data }),
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
            connection: { credential_input: 'batch_text' },
            capabilities: {
              model_discovery: true,

              outbound_proxy: false,
            },
            routes: [
              {
                client_protocol: 'openai-completions',
                operation: 'chat_completion',
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
      name: 'Discovery Group',
      channel_id: 'openai',
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
        model_count: 0,
      })
    if (path === '/api/groups/1/settings')
      return fulfill({
        name: groupBase.name,
        channel_id: groupBase.channel_id,
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
    if (path === '/api/groups/1/models' && request.method() === 'GET')
      return fulfill({ items: [], total: 0, pending: 0 })
    if (path === '/api/groups/1/models' && request.method() === 'PUT') {
      puts.push(request.postDataJSON())
      return fulfill({ items: [], total: 0, pending: 0 })
    }
    if (path === '/api/groups/1/models/discover') return fulfill({ models: CANDIDATES })
    if (path === '/api/groups/1/credential') return fulfill({ credential: null })
    return fulfill({})
  })
  return { puts }
}
