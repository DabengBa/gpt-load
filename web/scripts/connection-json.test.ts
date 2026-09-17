import assert from 'node:assert/strict'
import test from 'node:test'

import {
  parseConnectionJSON,
  mapConnectionToChannel,
  type ConnectionJSON,
  type ChannelSchema,
} from '../src/features/import/connection-json.ts'

// ---------------------------------------------------------------------------
// Parser — 严格识别 newapi_channel_conn 连接 JSON
// ---------------------------------------------------------------------------

// R1 — 紧凑 JSON 与格式化多行 JSON 返回 trim 后的 baseURL/apiKey。
test('R1: parses a compact New API connection object', () => {
  assert.deepEqual(
    parseConnectionJSON(
      '{"_type":"newapi_channel_conn","key":"sk-placeholder-compact","url":"https://gateway.example.invalid"}',
    ),
    { baseURL: 'https://gateway.example.invalid', apiKey: 'sk-placeholder-compact' },
  )
})

test('R1: parses a formatted multi-line New API connection object and trims values', () => {
  const formatted = [
    '{',
    '  "_type": "newapi_channel_conn",',
    '  "key": "  sk-placeholder-formatted  ",',
    '  "url": "  https://gateway.example.invalid  "',
    '}',
  ].join('\n')

  assert.deepEqual(parseConnectionJSON(formatted), {
    baseURL: 'https://gateway.example.invalid',
    apiKey: 'sk-placeholder-formatted',
  })
})

test('R1: keeps a structurally valid result when the URL is not a valid URL', () => {
  assert.deepEqual(
    parseConnectionJSON(
      '{"_type":"newapi_channel_conn","key":"sk-placeholder-badurl","url":"not-a-url"}',
    ),
    { baseURL: 'not-a-url', apiKey: 'sk-placeholder-badurl' },
  )
})

// R3 — 普通 Key、其他 JSON 与控制字符保持不转换。
test('R3: rejects a plain API key', () => {
  assert.equal(parseConnectionJSON('sk-placeholder-plain'), null)
  assert.equal(parseConnectionJSON('  sk-placeholder-plain  '), null)
})

test('R3: rejects other JSON payloads', () => {
  assert.equal(parseConnectionJSON('{"url":"https://gateway.example.invalid","key":"sk-x"}'), null)
  assert.equal(
    parseConnectionJSON(
      '{"_type":"other_channel_conn","key":"sk-x","url":"https://gateway.example.invalid"}',
    ),
    null,
  )
})

test('R3: rejects control characters in credentials', () => {
  assert.equal(
    parseConnectionJSON(
      '{"_type":"newapi_channel_conn","key":"sk-placeholder\\u0000","url":"https://gateway.example.invalid"}',
    ),
    null,
  )
  assert.equal(
    parseConnectionJSON(
      '{"_type":"newapi_channel_conn","key":"sk-placeholder","url":"https://gateway.example.invalid\\n"}',
    ),
    null,
  )
})

test('R3: rejects Unicode C1 control characters in key and URL', () => {
  for (const control of ['\u0080', '\u0085', '\u009f']) {
    const codePoint = `U+${control.charCodeAt(0).toString(16).toUpperCase()}`
    assert.equal(
      parseConnectionJSON(
        JSON.stringify({
          _type: 'newapi_channel_conn',
          key: `sk-placeholder${control}`,
          url: 'https://gateway.example.invalid',
        }),
      ),
      null,
      `key ${codePoint}`,
    )
    assert.equal(
      parseConnectionJSON(
        JSON.stringify({
          _type: 'newapi_channel_conn',
          key: 'sk-placeholder',
          url: `https://gateway.example.invalid${control}`,
        }),
      ),
      null,
      `url ${codePoint}`,
    )
  }

  // 非控制字符的 Unicode 不应被误杀，确保 Cc 判定不过宽。
  assert.deepEqual(
    parseConnectionJSON(
      JSON.stringify({
        _type: 'newapi_channel_conn',
        key: 'sk-placeholder',
        url: 'https://gateway.example.invalid/caf\u00e9',
      }),
    ),
    { baseURL: 'https://gateway.example.invalid/caf\u00e9', apiKey: 'sk-placeholder' },
  )
})

test('rejects non-object JSON shapes', () => {
  for (const raw of [
    '[]',
    '["newapi_channel_conn"]',
    'null',
    'true',
    '42',
    '3.14',
    '"newapi_channel_conn"',
    '',
    '   ',
    '{',
    '{"_type":"newapi_channel_conn",}',
    'undefined',
  ]) {
    assert.equal(parseConnectionJSON(raw), null, raw)
  }
})

test('rejects objects with the wrong key shape', () => {
  const cases: Record<string, string> = {
    'missing key': '{"_type":"newapi_channel_conn","url":"https://gateway.example.invalid"}',
    'missing url': '{"_type":"newapi_channel_conn","key":"sk-placeholder"}',
    'missing _type': '{"key":"sk-placeholder","url":"https://gateway.example.invalid"}',
    'extra field': JSON.stringify({
      _type: 'newapi_channel_conn',
      key: 'sk-placeholder',
      url: 'https://gateway.example.invalid',
      name: 'extra',
    }),
    'null key':
      '{"_type":"newapi_channel_conn","key":null,"url":"https://gateway.example.invalid"}',
    'number key': '{"_type":"newapi_channel_conn","key":7,"url":"https://gateway.example.invalid"}',
    'object key':
      '{"_type":"newapi_channel_conn","key":{},"url":"https://gateway.example.invalid"}',
    'array url': '{"_type":"newapi_channel_conn","key":"sk-placeholder","url":[]}',
    'empty key': '{"_type":"newapi_channel_conn","key":"","url":"https://gateway.example.invalid"}',
    'whitespace key':
      '{"_type":"newapi_channel_conn","key":"   ","url":"https://gateway.example.invalid"}',
    'empty url': '{"_type":"newapi_channel_conn","key":"sk-placeholder","url":""}',
    'whitespace url': '{"_type":"newapi_channel_conn","key":"sk-placeholder","url":"   "}',
    'wrong _type':
      '{"_type":"NEWAPI_CHANNEL_CONN","key":"sk-placeholder","url":"https://x.invalid"}',
  }

  for (const [label, raw] of Object.entries(cases)) {
    assert.equal(parseConnectionJSON(raw), null, label)
  }
})

// ---------------------------------------------------------------------------
// mapConnectionToChannel — 按渠道 schema 将解析结果映射到既有字段
// ---------------------------------------------------------------------------

const parsed: ConnectionJSON = {
  baseURL: 'https://gateway.example.invalid',
  apiKey: 'sk-placeholder',
}

function schema(
  paramFields: Array<{ key: string; input_kind: string }>,
  credentialFields: Array<{ key: string }>,
): ChannelSchema {
  return { param_fields: paramFields, credential_fields: credentialFields }
}

// R1 — 标准单字段 API Key 渠道：URL 写 base_url，凭据写纯 API Key。
test('R1: standard single api_key channel maps URL to base_url and credentials to plain key', () => {
  assert.deepEqual(
    mapConnectionToChannel(
      parsed,
      schema([{ key: 'base_url', input_kind: 'url' }], [{ key: 'api_key' }]),
    ),
    { urlParamKey: 'base_url', credentials: 'sk-placeholder' },
  )
})

// R1 — Azure 类多字段 API Key 渠道：URL 写 endpoint，凭据写结构化 JSON。
test('R1: azure-like multi-field channel maps URL to endpoint and credentials to structured JSON', () => {
  const result = mapConnectionToChannel(
    parsed,
    schema([{ key: 'endpoint', input_kind: 'url' }], [{ key: 'api_key' }, { key: 'entra_token' }]),
  )
  assert.equal(result.urlParamKey, 'endpoint')
  assert.equal(result.credentials, JSON.stringify({ api_key: 'sk-placeholder' }))
})

// R1 — AWS Bedrock 类：无 URL 参数，只收敛 api_key 凭据。
test('R1: channel without URL param maps URL to null and still writes credentials', () => {
  const result = mapConnectionToChannel(
    parsed,
    schema([{ key: 'region', input_kind: 'text' }], [{ key: 'api_key' }, { key: 'access_key' }]),
  )
  assert.equal(result.urlParamKey, null)
  assert.equal(result.credentials, JSON.stringify({ api_key: 'sk-placeholder' }))
})

// R1 — 无 api_key 目标字段（例如 service-account 专用结构）保留原输入。
test('R1: channel without api_key credential field returns null credentials to preserve raw input', () => {
  const result = mapConnectionToChannel(
    parsed,
    schema([{ key: 'location', input_kind: 'text' }], [{ key: 'service_account_json' }]),
  )
  assert.equal(result.urlParamKey, null)
  assert.equal(result.credentials, null)
})

// R1 — URL 参数选择第一个 input_kind === 'url' 的字段。
test('R1: selects the first url-kind param field when multiple exist', () => {
  const result = mapConnectionToChannel(
    parsed,
    schema(
      [
        { key: 'region', input_kind: 'text' },
        { key: 'base_url', input_kind: 'url' },
      ],
      [{ key: 'api_key' }],
    ),
  )
  assert.equal(result.urlParamKey, 'base_url')
  assert.equal(result.credentials, 'sk-placeholder')
})

// R1 — endpoint 优先于 base_url 时仍取第一个 url 字段。
test('R1: endpoint channel selects endpoint as the url param', () => {
  const result = mapConnectionToChannel(
    parsed,
    schema([{ key: 'endpoint', input_kind: 'url' }], [{ key: 'api_key' }, { key: 'entra_token' }]),
  )
  assert.equal(result.urlParamKey, 'endpoint')
})
