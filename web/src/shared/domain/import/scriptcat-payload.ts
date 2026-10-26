export const scriptcatImportPayloadStorageKey = 'gpt-load.scriptcat-import-payload'
export const scriptcatImportPayloadTtlMs = 15 * 60_000

export interface ScriptcatImportPayload {
  apiKey: string
  baseURL: string
  groupName: string
  providerUrl: string
}

interface StoredScriptcatImportPayload extends ScriptcatImportPayload {
  version: 2
  createdAt: number
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isHTTPURL(value: string): boolean {
  try {
    return ['http:', 'https:'].includes(new URL(value).protocol)
  } catch {
    return false
  }
}

function isPayload(value: unknown, now: number): value is StoredScriptcatImportPayload {
  if (!isRecord(value)) return false
  const fields = ['version', 'createdAt', 'apiKey', 'baseURL', 'groupName', 'providerUrl']
  return (
    Object.keys(value).length === fields.length &&
    fields.every((field) => Object.hasOwn(value, field)) &&
    value.version === 2 &&
    typeof value.createdAt === 'number' &&
    Number.isFinite(value.createdAt) &&
    value.createdAt <= now &&
    now - value.createdAt <= scriptcatImportPayloadTtlMs &&
    typeof value.apiKey === 'string' &&
    value.apiKey.trim().length > 0 &&
    value.apiKey.length <= 16_384 &&
    typeof value.baseURL === 'string' &&
    value.baseURL.trim().length > 0 &&
    value.baseURL.length <= 2_048 &&
    typeof value.groupName === 'string' &&
    value.groupName.trim().length > 0 &&
    value.groupName.length <= 256 &&
    typeof value.providerUrl === 'string' &&
    value.providerUrl.length <= 2_048 &&
    isHTTPURL(value.providerUrl)
  )
}

export function consumeScriptcatImportPayload(
  storage: Storage,
  now = Date.now(),
): ScriptcatImportPayload | null {
  let raw: string | null
  try {
    raw = storage.getItem(scriptcatImportPayloadStorageKey)
    if (raw === null) return null
    storage.removeItem(scriptcatImportPayloadStorageKey)
  } catch {
    return null
  }

  try {
    const value: unknown = JSON.parse(raw)
    if (!isPayload(value, now)) return null
    return {
      apiKey: value.apiKey,
      baseURL: value.baseURL,
      groupName: value.groupName,
      providerUrl: value.providerUrl,
    }
  } catch {
    return null
  }
}
