import assert from 'node:assert/strict'
import test from 'node:test'

import {
  createProbeTemplate,
  runRealisticProbe,
  type RealisticProbeProtocol,
} from '../src/shared/probe/realistic-probe.ts'

const key = 'fake-key-for-test'
const id = 'aaaaaaaa-1111-4111-8111-111111111111'
const protocols: RealisticProbeProtocol[] = [
  'openai-responses',
  'openai-completions',
  'anthropic',
  'gemini',
]
const results = {
  'openai-responses': {
    object: 'response',
    status: 'completed',
    output: [{ type: 'function_call', call_id: 'c', name: 'f', arguments: '{}' }],
  },
  'openai-completions': {
    object: 'chat.completion',
    choices: [
      {
        message: {
          role: 'assistant',
          tool_calls: [{ id: 'c', type: 'function', function: { name: 'f', arguments: '{}' } }],
        },
        finish_reason: 'tool_calls',
      },
    ],
  },
  anthropic: {
    type: 'message',
    role: 'assistant',
    content: [{ type: 'text', text: 'ok' }],
    stop_reason: 'end_turn',
  },
  gemini: { candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }] },
}

function opts(protocol: RealisticProbeProtocol, stream = false) {
  return {
    protocol,
    model: 'client-model',
    stream,
    body: createProbeTemplate(protocol, 'client-model', stream),
    accessKey: key,
    signal: new AbortController().signal,
  }
}
function response(body: unknown, status = 200, requestId = id) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'X-GPTLoad-Request-ID': requestId, 'Content-Type': 'application/json' },
  })
}
function sse(chunks: Uint8Array[]) {
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(chunk)
        controller.close()
      },
    }),
    { headers: { 'Content-Type': 'text/event-stream' } },
  )
}
const bytes = (s: string) => new TextEncoder().encode(s)

for (const protocol of protocols)
  test(`${protocol}: native request and tool/text completion`, async () => {
    const options = opts(protocol)
    const body = JSON.parse(options.body)
    if (protocol !== 'gemini') {
      body.model = 'actual-client'
      body.temperature = 0.5
    }
    options.body = JSON.stringify(body)
    let called = false
    const result = await runRealisticProbe(options, async (path, init) => {
      called = true
      assert.equal(
        path,
        {
          'openai-responses': '/v1/responses',
          'openai-completions': '/v1/chat/completions',
          anthropic: '/v1/messages',
          gemini: '/v1beta/models/client-model:generateContent',
        }[protocol],
      )
      assert.equal(init?.body, options.body)
      assert.equal(new Headers(init?.headers).get('Authorization'), `Bearer ${key}`)
      assert.equal(init?.credentials, 'omit')
      assert.equal(init?.redirect, 'error')
      assert.equal(init?.signal, options.signal)
      return response(results[protocol])
    })
    assert.equal(called, true)
    assert.equal(result.outcome, 'completed')
    assert.equal(result.requestId, id)
    assert.ok(!result.output.includes(key))
  })

test('reject invalid body, path and key before fetch', async () => {
  for (const patch of [
    { body: '[]' },
    { body: '{' },
    { body: '{"model":"","stream":false}' },
    { body: '{"model":"m","stream":"true"}' },
    { protocol: 'gemini' as const, model: '../x' },
    { accessKey: 'bad\nkey' },
  ]) {
    let called = false
    const result = await runRealisticProbe(
      { ...opts('openai-completions'), ...patch },
      async () => {
        called = true
        return response({})
      },
    )
    assert.equal(called, false)
    assert.equal(result.outcome, 'failed')
  }
})

test('HTTP errors retain only canonical request ID and redact reflected key', async () => {
  for (const [requestId, expected] of [
    [id, id],
    [id.toUpperCase(), null],
    ['abc', null],
  ]) {
    const result = await runRealisticProbe(opts('openai-responses'), async () =>
      response({ error: { message: key } }, 401, requestId),
    )
    assert.equal(result.outcome, 'failed')
    assert.equal(result.requestId, expected)
    assert.ok(!JSON.stringify(result).includes(key))
  }
})

test('split UTF-8, CRLF, comments and multiline SSE response complete', async () => {
  const text =
    ': comment\r\nevent: response.output_text.delta\r\ndata: {"type":"response.output_text.delta",\r\ndata: "delta":"你好"}\r\n\r\nevent: response.completed\r\ndata: {"type":"response.completed","response":{"object":"response","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"你好"}]}]}}\r\n\r\n'
  const data = bytes(text)
  const chunks = Array.from(data, (_, i) => data.slice(i, i + 1))
  const updates: string[] = []
  const result = await runRealisticProbe(
    { ...opts('openai-responses', true), onUpdate: (value) => updates.push(value.outcome) },
    async () => sse(chunks),
  )
  assert.equal(result.outcome, 'completed')
  assert.ok(result.output.includes('你好'))
  assert.notEqual(result.firstOutputMs, null)
  assert.ok(updates.includes('running'))
})

test('partial, malformed, arbitrary and empty responses cannot complete', async () => {
  for (const wire of [
    'data: {"type":"response.output_text.delta","delta":"partial"}\n\n',
    'data: {bad}\n\n',
    'data: {"hello":"world"}\n\n',
    'data: [DONE]\n\n',
  ]) {
    const result = await runRealisticProbe(opts('openai-responses', true), async () =>
      sse([bytes(wire)]),
    )
    assert.equal(result.outcome, 'incomplete')
  }
  for (const protocol of protocols) {
    const result = await runRealisticProbe(opts(protocol), async () => response({ hello: 'world' }))
    assert.equal(result.outcome, 'incomplete')
  }
})

test('explicit failure and budget/filter finishes are distinct from disconnect', async () => {
  const cases: [RealisticProbeProtocol, unknown, string][] = [
    ['openai-responses', { object: 'response', status: 'incomplete', output: [] }, 'incomplete'],
    [
      'openai-responses',
      { object: 'response', status: 'failed', error: { message: 'oops' } },
      'failed',
    ],
    [
      'openai-completions',
      {
        object: 'chat.completion',
        choices: [{ message: { role: 'assistant', content: 'partial' }, finish_reason: 'length' }],
      },
      'incomplete',
    ],
    [
      'openai-completions',
      {
        object: 'chat.completion',
        choices: [
          { message: { role: 'assistant', content: null }, finish_reason: 'content_filter' },
        ],
      },
      'incomplete',
    ],
    [
      'anthropic',
      {
        type: 'message',
        role: 'assistant',
        content: [{ type: 'text', text: 'x' }],
        stop_reason: 'max_tokens',
      },
      'incomplete',
    ],
    [
      'gemini',
      { candidates: [{ content: { parts: [{ text: 'x' }] }, finishReason: 'MAX_TOKENS' }] },
      'incomplete',
    ],
  ]
  for (const [protocol, body, outcome] of cases) {
    const result = await runRealisticProbe(opts(protocol), async () => response(body))
    assert.equal(result.outcome, outcome)
    assert.ok(result.terminal)
  }
})

test('cancellation preserves partial output', async () => {
  const controller = new AbortController()
  let streamController: ReadableStreamDefaultController<Uint8Array>
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      streamController = c
      c.enqueue(bytes('data: {"type":"response.output_text.delta","delta":"partial"}\n\n'))
    },
  })
  const result = await runRealisticProbe(
    {
      ...opts('openai-responses', true),
      signal: controller.signal,
      onUpdate: (value) => {
        if (value.output.includes('partial')) {
          controller.abort()
          streamController.error(new DOMException('aborted', 'AbortError'))
        }
      },
    },
    async () => new Response(stream),
  )
  assert.equal(result.outcome, 'canceled')
  assert.ok(result.output.includes('partial'))
})

test('native streaming tool calls and non-success finish reasons', async () => {
  const cases: [RealisticProbeProtocol, unknown[], string, string][] = [
    [
      'openai-completions',
      [
        {
          choices: [
            {
              delta: { tool_calls: [{ index: 0, function: { name: 'f', arguments: '{}' } }] },
              finish_reason: null,
            },
          ],
        },
        { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
        '[DONE]',
      ],
      'completed',
      'tool_calls',
    ],
    [
      'openai-completions',
      [{ choices: [{ delta: {}, finish_reason: 'content_filter' }] }, '[DONE]'],
      'incomplete',
      'content_filter',
    ],
    [
      'anthropic',
      [
        {
          type: 'content_block_start',
          content_block: { type: 'tool_use', id: 't', name: 'f', input: {} },
        },
        { type: 'message_delta', delta: { stop_reason: 'tool_use' } },
        { type: 'message_stop' },
      ],
      'completed',
      'tool_use',
    ],
    [
      'anthropic',
      [{ type: 'message_delta', delta: { stop_reason: 'max_tokens' } }, { type: 'message_stop' }],
      'incomplete',
      'max_tokens',
    ],
    [
      'gemini',
      [
        { candidates: [{ content: { parts: [{ functionCall: { name: 'f', args: {} } }] } }] },
        { candidates: [{ finishReason: 'STOP' }] },
      ],
      'completed',
      'STOP',
    ],
    ['gemini', [{ candidates: [{ finishReason: 'SAFETY' }] }], 'incomplete', 'SAFETY'],
  ]
  for (const [protocol, events, outcome, terminal] of cases) {
    const wire = events
      .map((value) => `data: ${typeof value === 'string' ? value : JSON.stringify(value)}\n\n`)
      .join('')
    const result = await runRealisticProbe(opts(protocol, true), async (path) => {
      if (protocol === 'gemini')
        assert.equal(path, '/v1beta/models/client-model:streamGenerateContent?alt=sse')
      return sse([bytes(wire)])
    })
    assert.equal(result.outcome, outcome, protocol)
    assert.equal(result.terminal, terminal, protocol)
  }
})

test('tool terminals require complete streamed tool structure', async () => {
  const cases: [RealisticProbeProtocol, unknown[]][] = [
    [
      'openai-completions',
      [
        { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '{}' } }] } }] },
        { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
        '[DONE]',
      ],
    ],
    [
      'openai-completions',
      [
        { choices: [{ delta: { tool_calls: [{ index: 0, function: { name: 'f' } }] } }] },
        { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
        '[DONE]',
      ],
    ],
    [
      'anthropic',
      [
        {
          type: 'content_block_delta',
          index: 0,
          delta: { type: 'input_json_delta', partial_json: '{}' },
        },
        { type: 'message_delta', delta: { stop_reason: 'tool_use' } },
        { type: 'message_stop' },
      ],
    ],
  ]
  for (const [protocol, events] of cases) {
    const wire = events
      .map((event) => `data: ${typeof event === 'string' ? event : JSON.stringify(event)}\n\n`)
      .join('')
    const result = await runRealisticProbe(opts(protocol, true), async () => sse([bytes(wire)]))
    assert.equal(result.outcome, 'incomplete', protocol)
  }
})

test('split tool calls require each function name and arguments', async () => {
  const events = [
    { choices: [{ delta: { tool_calls: [{ index: 0, function: { name: 'first' } }] } }] },
    {
      choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '{}' } }] } }],
    },
    {
      choices: [
        { delta: { tool_calls: [{ index: 1, function: { name: 'second', arguments: '{}' } }] } },
      ],
    },
    { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
    '[DONE]',
  ]
  for (const missingArgument of [false, true]) {
    const selected = missingArgument ? events.filter((_, index) => index !== 1) : events
    const wire = selected
      .map((event) => `data: ${typeof event === 'string' ? event : JSON.stringify(event)}\n\n`)
      .join('')
    const result = await runRealisticProbe(opts('openai-completions', true), async () =>
      sse([bytes(wire)]),
    )
    assert.equal(result.outcome, missingArgument ? 'incomplete' : 'completed')
  }
  const invalidMessage = {
    object: 'chat.completion',
    choices: [
      {
        message: {
          role: 'assistant',
          content: 'text',
          tool_calls: [{ function: { name: 'first', arguments: '' } }],
        },
        finish_reason: 'tool_calls',
      },
    ],
  }
  const invalid = await runRealisticProbe(opts('openai-completions'), async () =>
    response(invalidMessage),
  )
  assert.equal(invalid.outcome, 'incomplete')
})

test('Anthropic tool input deltas must belong to a valid tool block', async () => {
  const start = {
    type: 'content_block_start',
    index: 1,
    content_block: { type: 'tool_use', id: 't', name: 'f', input: {} },
  }
  const delta = {
    type: 'content_block_delta',
    index: 1,
    delta: { type: 'input_json_delta', partial_json: '{}' },
  }
  const end = [
    { type: 'message_delta', delta: { stop_reason: 'tool_use' } },
    { type: 'message_stop' },
  ]
  for (const [events, outcome] of [
    [[start, delta, ...end], 'completed'],
    [
      [{ ...start, content_block: { ...start.content_block, id: '' } }, delta, ...end],
      'incomplete',
    ],
    [[start, { ...delta, index: 2 }, ...end], 'incomplete'],
  ] as const) {
    const wire = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('')
    const result = await runRealisticProbe(opts('anthropic', true), async () => sse([bytes(wire)]))
    assert.equal(result.outcome, outcome)
  }
})

test('Anthropic streamed tool input must form a complete JSON object', async () => {
  const start = {
    type: 'content_block_start',
    index: 0,
    content_block: { type: 'tool_use', id: 't', name: 'f', input: {} },
  }
  const end = [
    { type: 'message_delta', delta: { stop_reason: 'tool_use' } },
    { type: 'message_stop' },
  ]
  for (const [fragments, outcome] of [
    [['{"city":', '"Paris"}'], 'completed'],
    [['{"city":'], 'incomplete'],
    [['not-json'], 'incomplete'],
    [['[]'], 'incomplete'],
  ] as const) {
    const deltas = fragments.map((partial_json) => ({
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'input_json_delta', partial_json },
    }))
    const wire = [start, ...deltas, ...end]
      .map((event) => `data: ${JSON.stringify(event)}\n\n`)
      .join('')
    const result = await runRealisticProbe(opts('anthropic', true), async () => sse([bytes(wire)]))
    assert.equal(result.outcome, outcome, fragments.join(''))
  }
})

test('authoritative body stream, malformed terminal and structured stream failure', async () => {
  const options = { ...opts('openai-responses', true), stream: false }
  const invalid = `data: ${JSON.stringify({ type: 'response.completed', response: { object: 'response', status: 'completed', output: [] } })}\n\n`
  const invalidResult = await runRealisticProbe(options, async () => sse([bytes(invalid)]))
  assert.equal(invalidResult.outcome, 'incomplete')
  const failed = await runRealisticProbe(options, async () =>
    sse([bytes('event: error\ndata: {"type":"error","message":"oops"}\n\n')]),
  )
  assert.equal(failed.outcome, 'failed')
})

test('reflected key is masked in incremental and final output, including truncated preview', async () => {
  const updates: string[] = []
  const text =
    `data: ${JSON.stringify({ type: 'response.output_text.delta', delta: key.repeat(10000) })}\n\n` +
    `data: ${JSON.stringify({ type: 'response.completed', response: results['openai-responses'] })}\n\n`
  const result = await runRealisticProbe(
    { ...opts('openai-responses', true), onUpdate: (value) => updates.push(JSON.stringify(value)) },
    async () => sse([bytes(text)]),
  )
  assert.equal(result.outcome, 'completed')
  assert.ok(result.output.length <= 131072)
  assert.ok(!JSON.stringify(result).includes(key))
  assert.ok(updates.every((update) => !update.includes(key)))
})

test('preview limit never exposes a reflected key split across reads', async () => {
  const updates: string[] = []
  const prefix = 'x'.repeat(131072 - key.length + 3) + key.slice(0, -3)
  const source = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes(prefix))
      controller.enqueue(bytes(key.slice(-3)))
      controller.close()
    },
  })

  test('partial reflected key remains masked when the request is canceled', async () => {
    const controller = new AbortController()
    const updates: string[] = []
    const source = new ReadableStream<Uint8Array>({
      start(stream) {
        stream.enqueue(bytes('safe ' + key.slice(0, -3)))
      },
    })
    const result = await runRealisticProbe(
      {
        ...opts('openai-responses'),
        signal: controller.signal,
        onUpdate: (value) => {
          updates.push(value.output)
          if (value.output.includes('safe ')) controller.abort()
        },
      },
      async () => new Response(source, { status: 400 }),
    )
    assert.equal(result.outcome, 'canceled')
    assert.ok(result.output.includes('safe '))
    assert.ok(result.output.includes('[REDACTED]'))
    assert.ok(updates.every((update) => !update.includes(key.slice(0, 4))))
    assert.ok(!result.output.includes(key.slice(0, 4)))
  })

  test('JSON-escaped reflected access keys are masked', async () => {
    const specialKey = 'fake-"quote\\tail'
    const encoded = JSON.stringify(specialKey).slice(1, -1)
    const result = await runRealisticProbe(
      { ...opts('openai-responses'), accessKey: specialKey },
      async () => response({ error: { message: specialKey } }, 400),
    )
    assert.equal(result.outcome, 'failed')
    assert.ok(result.output.includes('[REDACTED]'))
    assert.ok(!result.output.includes(encoded))
    assert.ok(!result.output.includes(specialKey))
  })
  const result = await runRealisticProbe(
    { ...opts('openai-responses'), onUpdate: (value) => updates.push(value.output) },
    async () => new Response(source, { status: 400 }),
  )
  assert.equal(result.outcome, 'failed')
  assert.ok(!result.output.includes(key.slice(0, 4)))
  assert.ok(updates.every((update) => !update.includes(key.slice(0, 4))))
})

test('oversized individual stream event and non-stream body remain incomplete', async () => {
  const stream = await runRealisticProbe(opts('openai-responses', true), async () =>
    sse([bytes('data: ' + 'x'.repeat(300000))]),
  )
  assert.equal(stream.outcome, 'incomplete')
  const body = await runRealisticProbe(
    opts('openai-responses'),
    async () => new Response('x'.repeat(1100000)),
  )
  assert.equal(body.outcome, 'incomplete')
})

test('abort settles pending response reader and preserves observed output', async () => {
  const controller = new AbortController()
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(bytes('data: {"type":"response.output_text.delta","delta":"part"}\n\n'))
    },
  })
  const result = await runRealisticProbe(
    {
      ...opts('openai-responses', true),
      signal: controller.signal,
      onUpdate: (value) => {
        if (value.output.includes('part')) controller.abort()
      },
    },
    async () => new Response(stream),
  )
  assert.equal(result.outcome, 'canceled')
  assert.match(result.output, /part/)
})

test('tool payloads must be structured and legacy function calls remain valid', async () => {
  const legacy = {
    object: 'chat.completion',
    choices: [
      {
        message: {
          role: 'assistant',
          content: null,
          function_call: { name: 'f', arguments: '{}' },
        },
        finish_reason: 'function_call',
      },
    ],
  }
  const valid = await runRealisticProbe(opts('openai-completions'), async () => response(legacy))
  assert.equal(valid.outcome, 'completed')
  const events = [
    { choices: [{ delta: { tool_calls: [{}] }, finish_reason: null }] },
    { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
    '[DONE]',
  ]
  const wire = events
    .map((value) => `data: ${typeof value === 'string' ? value : JSON.stringify(value)}\n\n`)
    .join('')
  const invalid = await runRealisticProbe(opts('openai-completions', true), async () =>
    sse([bytes(wire)]),
  )
  assert.equal(invalid.outcome, 'incomplete')
})

test('oversized stream cancels its response and thrown callbacks never escape', async () => {
  let canceled = false
  const source = new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(bytes('data: ' + 'x'.repeat(300000)))
    },
    cancel() {
      canceled = true
    },
  })
  const result = await runRealisticProbe(
    {
      ...opts('openai-responses', true),
      onUpdate: () => {
        throw new Error('observer failed')
      },
    },
    async () => new Response(source),
  )
  assert.notEqual(result.outcome, 'completed')
  assert.equal(canceled, true)
})

test('Responses can contain reasoning alongside a valid output', async () => {
  const value = {
    object: 'response',
    status: 'completed',
    output: [
      { type: 'reasoning', id: 'rs_1', summary: [] },
      { type: 'message', content: [{ type: 'output_text', text: 'ok' }] },
    ],
  }
  const result = await runRealisticProbe(opts('openai-responses'), async () => response(value))
  assert.equal(result.outcome, 'completed')
})

test('preview is bounded but subsequent terminal is detected', async () => {
  const chunk = `data: ${JSON.stringify({ type: 'response.output_text.delta', delta: 'x'.repeat(1000) })}\n\n`
  const terminal = `data: ${JSON.stringify({ type: 'response.completed', response: results['openai-responses'] })}\n\n`
  const result = await runRealisticProbe(opts('openai-responses', true), async () =>
    sse([bytes(chunk.repeat(200) + terminal)]),
  )
  assert.equal(result.outcome, 'completed')
  assert.equal(result.truncated, true)
  assert.ok(result.output.length <= 131072)
})

test('server-owned native content is forwarded unchanged and Gemini allows safe single-segment models', async () => {
  const inputs = [
    { ...opts('openai-responses'), body: '{"model":"m","prompt":{"id":"p"},"conversation":"c"}' },
    { ...opts('openai-completions'), body: '{"model":"m","future_native_field":true}' },
    { ...opts('anthropic'), body: '{"model":"m"}' },
    { ...opts('gemini'), model: 'gemini+v1:preview', body: '{"cachedContent":"cached"}' },
  ]
  for (const options of inputs) {
    let called = false
    const result = await runRealisticProbe(options, async (path, init) => {
      called = true
      assert.equal(init?.body, options.body)
      if (options.protocol === 'gemini')
        assert.equal(path, '/v1beta/models/gemini%2Bv1%3Apreview:generateContent')
      return response({ error: { message: 'Input must be a list' } }, 400)
    })
    assert.equal(called, true)
    assert.equal(result.outcome, 'failed')
    assert.match(result.output, /Input must be a list/)
  }
})

test('HTTP error preview is bounded, redacted and response reader canceled', async () => {
  let canceled = false
  const wire = `Input must be a list: ${key} ` + 'x'.repeat(200000)
  const source = new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(bytes(wire))
    },
    cancel() {
      canceled = true
    },
  })
  const result = await runRealisticProbe(
    opts('openai-responses'),
    async () => new Response(source, { status: 400 }),
  )
  assert.equal(result.outcome, 'failed')
  assert.match(result.output, /Input must be a list/)
  assert.doesNotMatch(result.output, new RegExp(key))
  assert.ok(result.output.length <= 131072)
  assert.equal(result.truncated, true)
  assert.equal(canceled, true)
})

test('Responses completed permits typed extension with valid output and reasoning-only delivery', async () => {
  const responseWith = (output: unknown[]) => ({ object: 'response', status: 'completed', output })
  for (const output of [
    [
      { type: 'web_search_call', id: 'w', status: 'completed' },
      { type: 'message', content: [{ type: 'output_text', text: 'ok' }] },
    ],
    [{ type: 'reasoning', id: 'r', summary: [] }],
  ]) {
    const result = await runRealisticProbe(opts('openai-responses'), async () =>
      response(responseWith(output)),
    )
    assert.equal(result.outcome, 'completed')
    if (output.length === 1) assert.equal(result.firstOutputMs, null)
  }
  for (const output of [[], [{}], [{ type: 'unrecognized' }]]) {
    const result = await runRealisticProbe(opts('openai-responses'), async () =>
      response(responseWith(output)),
    )
    assert.equal(result.outcome, 'incomplete')
  }
})

test('Responses preserve incomplete reason and terminal-only tool output timing', async () => {
  const incomplete = {
    object: 'response',
    status: 'incomplete',
    incomplete_details: { reason: 'max_output_tokens' },
    output: [],
  }
  const normal = await runRealisticProbe(opts('openai-responses'), async () => response(incomplete))
  assert.equal(normal.terminal, 'max_output_tokens')
  const stream = await runRealisticProbe(opts('openai-responses', true), async () =>
    sse([
      bytes(`data: ${JSON.stringify({ type: 'response.incomplete', response: incomplete })}\n\n`),
    ]),
  )
  assert.equal(stream.terminal, 'max_output_tokens')
  const tool = await runRealisticProbe(opts('openai-responses', true), async () =>
    sse([
      bytes(
        `data: ${JSON.stringify({ type: 'response.completed', response: results['openai-responses'] })}\n\n`,
      ),
    ]),
  )
  assert.equal(tool.outcome, 'completed')
  assert.notEqual(tool.firstOutputMs, null)
})

test('Responses incomplete event cannot complete with a contradictory completed payload', async () => {
  const wire = `data: ${JSON.stringify({ type: 'response.incomplete', response: results['openai-responses'] })}\n\n`
  const result = await runRealisticProbe(opts('openai-responses', true), async () =>
    sse([bytes(wire)]),
  )
  assert.equal(result.outcome, 'incomplete')
})

test('multiple nonstream choices and candidates cannot hide incomplete or missing finishes', async () => {
  const chat = {
    object: 'chat.completion',
    choices: [
      { index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' },
      { index: 1, message: { role: 'assistant', content: 'partial' }, finish_reason: 'length' },
    ],
  }
  const chatOpts = { ...opts('openai-completions'), body: JSON.stringify({ model: 'm', n: 2 }) }
  const result = await runRealisticProbe(chatOpts, async () => response(chat))
  assert.equal(result.outcome, 'incomplete')
  assert.equal(result.terminal, 'length')
  const missing = await runRealisticProbe(chatOpts, async () =>
    response({ ...chat, choices: chat.choices.slice(0, 1) }),
  )
  assert.equal(missing.outcome, 'incomplete')
  const gemini = {
    candidates: [
      { index: 0, content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' },
      { index: 1, content: { parts: [{ text: 'partial' }] }, finishReason: 'SAFETY' },
    ],
  }
  const geminiOpts = {
    ...opts('gemini'),
    body: JSON.stringify({ generationConfig: { candidateCount: 2 } }),
  }
  const geminiResult = await runRealisticProbe(geminiOpts, async () => response(gemini))
  assert.equal(geminiResult.outcome, 'incomplete')
  assert.equal(geminiResult.terminal, 'SAFETY')
  const geminiMissing = await runRealisticProbe(geminiOpts, async () =>
    response({ candidates: gemini.candidates.slice(0, 1) }),
  )
  assert.equal(geminiMissing.outcome, 'incomplete')
})

test('stream aggregates requested choices and candidates across separate events', async () => {
  const cases = [
    {
      protocol: 'openai-completions' as const,
      body: { model: 'm', stream: true, n: 2 },
      events: [
        { choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: 'stop' }] },
        { choices: [{ index: 1, delta: { content: 'partial' }, finish_reason: 'length' }] },
        '[DONE]',
      ],
      terminal: 'length',
    },
    {
      protocol: 'gemini' as const,
      body: { generationConfig: { candidateCount: 2 } },
      events: [
        { candidates: [{ index: 0, content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }] },
        {
          candidates: [
            { index: 1, content: { parts: [{ text: 'partial' }] }, finishReason: 'SAFETY' },
          ],
        },
      ],
      terminal: 'SAFETY',
    },
  ]
  for (const item of cases) {
    const wire = item.events
      .map((event) => `data: ${typeof event === 'string' ? event : JSON.stringify(event)}\n\n`)
      .join('')
    const result = await runRealisticProbe(
      { ...opts(item.protocol, true), body: JSON.stringify(item.body) },
      async () => sse([bytes(wire)]),
    )
    assert.equal(result.outcome, 'incomplete', item.protocol)
    assert.equal(result.terminal, item.terminal, item.protocol)
  }
})

test('Gemini EOF with an unterminated trailing SSE event stays incomplete', async () => {
  const complete = `data: ${JSON.stringify(results.gemini)}\n\n`
  const tails = ['data: {"error":', `data: ${JSON.stringify(results.gemini)}\n`]
  for (const tail of tails) {
    const result = await runRealisticProbe(opts('gemini', true), async () =>
      sse([bytes(complete + tail)]),
    )
    assert.equal(result.outcome, 'incomplete')
  }
})

test('Gemini accepts completed events followed by SSE comments and CR line endings', async () => {
  for (const end of ['\n\n:keepalive', '\r\r:keepalive', '\r\n\r\n:keepalive\r']) {
    const wire = `data: ${JSON.stringify(results.gemini)}${end}`
    const result = await runRealisticProbe(opts('gemini', true), async () => sse([bytes(wire)]))
    assert.equal(result.outcome, 'completed', JSON.stringify(end))
  }
})

test('multiple completed candidates succeed and unfinished ones do not', async () => {
  for (const protocol of ['openai-completions', 'gemini'] as const) {
    const isChat = protocol === 'openai-completions'
    const body = isChat
      ? { model: 'm', stream: true, n: 2 }
      : { generationConfig: { candidateCount: 2 } }
    const entry = (index: number, finish: boolean) =>
      isChat
        ? { choices: [{ index, delta: { content: 'ok' }, finish_reason: finish ? 'stop' : null }] }
        : {
            candidates: [
              {
                index,
                content: { parts: [{ text: 'ok' }] },
                ...(finish ? { finishReason: 'STOP' } : {}),
              },
            ],
          }
    for (const finished of [true, false]) {
      const events = [entry(0, true), entry(1, finished)]
      const wire =
        events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('') +
        (isChat ? 'data: [DONE]\n\n' : '')
      const result = await runRealisticProbe(
        { ...opts(protocol, true), body: JSON.stringify(body) },
        async () => sse([bytes(wire)]),
      )
      assert.equal(result.outcome, finished ? 'completed' : 'incomplete', protocol)
    }
  }
})

test('candidate and event limits report bounded incomplete responses', async () => {
  const count = await runRealisticProbe(
    { ...opts('openai-completions'), body: JSON.stringify({ model: 'm', n: 1025 }) },
    async () => response(results['openai-completions']),
  )
  assert.equal(count.outcome, 'incomplete')
  assert.match(count.error ?? '', /limit/)
  const event = await runRealisticProbe(opts('openai-responses', true), async () =>
    sse([bytes(`data: ${'x'.repeat(300000)}`)]),
  )
  assert.equal(event.outcome, 'incomplete')
  assert.equal(event.error, 'Stream event too large')
})

test('Gemini rejects unsafe path segments without sending a request', async () => {
  for (const model of ['', 'a/b', 'a b', 'a\n', 'a\u0080']) {
    let called = false
    const result = await runRealisticProbe({ ...opts('gemini'), model }, async () => {
      called = true
      return response({})
    })
    assert.equal(result.outcome, 'failed')
    assert.equal(called, false)
  }
})
