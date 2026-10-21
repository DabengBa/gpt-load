import assert from 'node:assert/strict'
import test from 'node:test'
import { projectChannelList } from '../src/shared/control/resources/channels.ts'

const channel = {
  channel_id: 'cliproxyapi',
  name: 'CLIProxyAPI',
  mark: 'CPA',
  icon: 'cpa',
  search_terms: ['cpa'],
  description: 'External gateway',
  default_base_url: '',
  param_fields: [
    {
      key: 'base_url',
      label: 'Base URL',
      input_kind: 'url',
      required: true,
      sensitive: false,
      default_value: null,
    },
  ],
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
      client_protocol: 'openai-completions',
      operation: 'chat_completion',
      route_mode: 'native',
      model_dependent: false,
      possible_modes: [],
    },
  ],
  client_protocols: ['openai-completions'],
}

test('external CPA and ordinary/cloud channel schemas project without subscription metadata', () => {
  for (const channel_id of [
    'cliproxyapi',
    'openai',
    'anthropic',
    'gemini',
    'vertex',
    'bedrock',
    'azure',
  ]) {
    const value = { ...channel, channel_id }
    assert.deepEqual(projectChannelList({ items: [value], total: 1 }).items[0], value)
  }
})

test('subscription authorization cannot project as an API channel', () => {
  assert.throws(() =>
    projectChannelList({
      items: [
        {
          ...channel,
          connection: {
            type: 'subscription',
            credential_input: 'authorization',
            authorization_methods: ['browser_oauth'],
          },
        },
      ],
      total: 1,
    }),
  )
})
