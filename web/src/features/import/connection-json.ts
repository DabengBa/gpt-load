export interface ConnectionJSON {
  baseURL: string
  apiKey: string
}

/**
 * 当前渠道 schema 的最小投影，用于将连接 JSON 映射到既有字段。
 * 字段形状与 `ChannelDto` 的相关子集一致，便于直接传入。
 */
export interface ChannelSchema {
  param_fields: ReadonlyArray<{ key: string; input_kind: string }>
  credential_fields: ReadonlyArray<{ key: string }>
}

export interface ConnectionMapping {
  /** URL 目标参数 key；无 URL 参数渠道为 null */
  urlParamKey: string | null
  /** 写入凭据框的值；无 api_key 目标时为 null（保留原输入） */
  credentials: string | null
}

const connectionType = 'newapi_channel_conn'
const connectionKeys = new Set(['_type', 'key', 'url'])
const controlCharacterPattern = /\p{Cc}/u

/**
 * 解析 New API 客户端生成的连接 JSON。
 *
 * 只接受键恰好为 `_type`、`key`、`url` 的对象，且 `_type` 必须为
 * `newapi_channel_conn`，`key`、`url` 为非空字符串。任何其他输入返回 `null`；
 * URL 合法性由既有表单校验处理，这里不做判定。
 */
export function parseConnectionJSON(raw: string): ConnectionJSON | null {
  let value: unknown
  try {
    value = JSON.parse(raw)
  } catch {
    return null
  }

  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null

  const record = value as Record<string, unknown>
  const keys = Object.keys(record)
  if (keys.length !== connectionKeys.size || !keys.every((key) => connectionKeys.has(key))) {
    return null
  }
  if (record._type !== connectionType) return null

  const rawKey = typeof record.key === 'string' ? record.key : null
  const rawURL = typeof record.url === 'string' ? record.url : null
  if (rawKey === null || rawURL === null) return null
  if (controlCharacterPattern.test(rawKey) || controlCharacterPattern.test(rawURL)) return null

  const apiKey = rawKey.trim()
  const baseURL = rawURL.trim()
  if (!apiKey || !baseURL) return null

  return { baseURL, apiKey }
}

/**
 * 将连接 JSON 解析结果按当前渠道 schema 映射到既有字段。
 *
 * - URL 目标取第一个 `input_kind === 'url'` 的参数（标准渠道 `base_url`、
 *   Azure 类 `endpoint`）；无 URL 参数返回 `null`，不新增未知参数。
 * - 凭据目标按 `api_key` 字段是否存在决定：单 `api_key` 写纯文本；
 *   多字段含 `api_key` 写最小结构化 JSON `{"api_key":"..."}`；
 *   无 `api_key` 返回 `null`，由调用方保留原始输入。
 */
export function mapConnectionToChannel(
  parsed: ConnectionJSON,
  channel: ChannelSchema,
): ConnectionMapping {
  const urlField = channel.param_fields.find((field) => field.input_kind === 'url') ?? null
  const urlParamKey = urlField?.key ?? null

  const hasAPIKey = channel.credential_fields.some((field) => field.key === 'api_key')
  if (!hasAPIKey) {
    return { urlParamKey, credentials: null }
  }

  const isSingleAPIKey =
    channel.credential_fields.length === 1 && channel.credential_fields[0]!.key === 'api_key'
  const credentials = isSingleAPIKey ? parsed.apiKey : JSON.stringify({ api_key: parsed.apiKey })
  return { urlParamKey, credentials }
}
