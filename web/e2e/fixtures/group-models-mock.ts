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

const SITE_ACCESS = {
  user_id: '',
  token_configured: false,
  site_root_url: '',
  snapshot: null,
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

export interface GroupModelSeed {
  id: string
  alias?: string
  alias_enabled?: boolean
  entry_id?: string
  weight?: number | null
  priority?: number | null
}

function modelItem(seed: GroupModelSeed, index: number, testAlias?: string, entryID?: string) {
  const alias = seed.alias ?? ''
  const aliasEnabled = seed.alias_enabled ?? false
  return {
    id: seed.id,
    alias,
    alias_enabled: aliasEnabled,
    client_model: aliasEnabled ? alias : seed.id,
    concurrency_limit: 0,
    test_alias: testAlias ?? `tst${String(index).padStart(3, '0')}`,
    entry_id: entryID ?? `e${index.toString(16).padStart(12, '0')}`,
    weight: seed.weight ?? null,
    priority: seed.priority ?? null,
    pricing_status: 'configured',
  }
}

export async function mockGroupModels(
  page: Page,
  options: {
    models?: GroupModelSeed[]
    extraChannel?: boolean
    generationProtocol?: 'openai-completions' | 'anthropic'
    enabled?: boolean
  } = {},
): Promise<{ puts: unknown[] }> {
  const puts: unknown[] = []
  let items = (options.models ?? []).map((seed, index) => modelItem(seed, index))
  let nextIdentity = items.length
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
                client_protocol: options.generationProtocol ?? 'openai-completions',
                operation: 'chat_completion',
                route_mode: 'native',
                model_dependent: false,
                possible_modes: [],
              },
              ...(options.extraChannel
                ? [
                    {
                      client_protocol: 'openai-embeddings',
                      operation: 'probe',
                      route_mode: 'native',
                      model_dependent: false,
                      possible_modes: [],
                    },
                  ]
                : []),
            ],
            client_protocols: options.extraChannel
              ? ['openai-completions', 'openai-embeddings']
              : [options.generationProtocol ?? 'openai-completions'],
          },
          ...(options.extraChannel
            ? [
                {
                  channel_id: 'gemini',
                  name: 'Gemini',
                  mark: 'G',
                  icon: 'gemini',
                  search_terms: ['gemini'],
                  description: '',
                  default_base_url: 'https://example.com',
                  param_fields: [],
                  credential_fields: [],
                  connection: { credential_input: 'batch_text' },
                  capabilities: { model_discovery: false, outbound_proxy: false },
                  routes: [],
                  client_protocols: [],
                },
              ]
            : []),
        ],
        total: options.extraChannel ? 2 : 1,
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
        enabled: options.enabled ?? true,
        site_access: SITE_ACCESS,
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
      return fulfill({ items, total: items.length, pending: 0 })
    if (path === '/api/groups/1/models' && request.method() === 'PUT') {
      const body = request.postDataJSON() as { models?: GroupModelSeed[] }
      puts.push(body)
      const assigned = new Set<string>()
      items = (body.models ?? []).map((model) => {
        let previous = model.entry_id
          ? items.find((item) => item.entry_id === model.entry_id)
          : undefined
        if (!previous) {
          let matches = items.filter(
            (item) => item.id === model.id && item.alias === (model.alias ?? ''),
          )
          if (matches.length === 0)
            matches = items.filter((item) => item.id === model.id && item.alias === '')
          if (matches.length === 1 && !assigned.has(matches[0].entry_id)) previous = matches[0]
        }
        if (previous) assigned.add(previous.entry_id)
        return modelItem(
          {
            ...model,
            weight: model.weight === undefined ? previous?.weight : model.weight,
            priority: model.priority === undefined ? previous?.priority : model.priority,
          },
          previous ? 0 : nextIdentity++,
          previous?.test_alias,
          model.entry_id || previous?.entry_id,
        )
      })
      return fulfill({ items, total: items.length, pending: 0 })
    }
    if (path === '/api/groups/1/models/discover') return fulfill({ models: CANDIDATES })
    if (path === '/api/groups/1/credential') return fulfill({ credential: null })
    return fulfill({})
  })
  return { puts }
}
