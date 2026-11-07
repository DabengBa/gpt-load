import type {
  ChannelParamsDto,
  GroupSettingsDto,
  ParameterJSONValue,
  ParameterOverrideRuleDto,
} from '@shared/control/types'
import type { ChannelFieldDto } from '@shared/control/resources/channels'
import type {
  GroupRuntimeConfigDto,
  GroupSettingsUpdateRequest,
  HeaderRulesDto,
} from '@shared/control/resources/groups'

export type GroupTimeoutKey = 'first_byte_timeout' | 'request_timeout' | 'stream_idle_timeout'
export type GroupPolicyCountKey = 'blacklist_threshold'

export interface GroupSettingsDraft {
  siteUserId: string
  siteAccessToken: string
  clearSiteAccessToken: boolean
  confirmSiteRootChange: boolean
  channel_id: string
  params: ChannelParamsDto
  name: string
  provider_url: string | null
  enabled: boolean

  overrides: GroupRuntimeConfigDto
}

export const groupTimeoutKeys: readonly GroupTimeoutKey[] = [
  'first_byte_timeout',
  'request_timeout',
  'stream_idle_timeout',
]
export const groupPolicyCountKeys: readonly GroupPolicyCountKey[] = ['blacklist_threshold']

function cloneHeaders(value: HeaderRulesDto): HeaderRulesDto {
  return { set: { ...value.set }, remove: [...value.remove] }
}

function cloneParameterValue(value: unknown): ParameterJSONValue {
  if (Array.isArray(value)) return value.map(cloneParameterValue)
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, nested]) => [key, cloneParameterValue(nested)]),
    )
  }
  if (
    value === null ||
    typeof value === 'boolean' ||
    typeof value === 'number' ||
    typeof value === 'string'
  )
    return value
  throw new TypeError('invalid JSON value')
}

function cloneParameterOverrides(value: ParameterOverrideRuleDto[]): ParameterOverrideRuleDto[] {
  return value.map((rule) => ({
    match: { ...rule.match },
    ...(rule.set
      ? {
          set: Object.fromEntries(
            Object.entries(rule.set).map(([key, nested]) => [key, cloneParameterValue(nested)]),
          ),
        }
      : {}),
    ...(rule.remove ? { remove: [...rule.remove] } : {}),
  }))
}

function cloneOverrides(value: GroupRuntimeConfigDto): GroupRuntimeConfigDto {
  const next: GroupRuntimeConfigDto = {}
  for (const key of groupTimeoutKeys) if (value[key] !== undefined) next[key] = value[key]
  for (const key of groupPolicyCountKeys) if (value[key] !== undefined) next[key] = value[key]
  if (value.header_rules) next.header_rules = cloneHeaders(value.header_rules)
  if (value.affinity_enabled !== undefined) next.affinity_enabled = value.affinity_enabled
  if (value.responses_websocket_enabled !== undefined)
    next.responses_websocket_enabled = value.responses_websocket_enabled
  if (value.responses_reasoning_status_filter_enabled !== undefined)
    next.responses_reasoning_status_filter_enabled = value.responses_reasoning_status_filter_enabled
  if (value.parameter_overrides?.length)
    next.parameter_overrides = cloneParameterOverrides(value.parameter_overrides)
  return next
}

function normalizeHeaders(value: HeaderRulesDto): HeaderRulesDto {
  return {
    set: Object.fromEntries(
      Object.entries(value.set).sort(([left], [right]) => left.localeCompare(right)),
    ),
    remove: [...value.remove].sort(),
  }
}

function normalizeOverrides(value: GroupRuntimeConfigDto): GroupRuntimeConfigDto {
  const next = cloneOverrides(value)
  if (next.header_rules) next.header_rules = normalizeHeaders(next.header_rules)
  if (next.parameter_overrides) {
    const rules = next.parameter_overrides.map((rule) => {
      const model = rule.match.model?.trim()
      const match = {
        ...(rule.match.protocol ? { protocol: rule.match.protocol } : {}),
        ...(model ? { model } : {}),
      }
      const set = rule.set ? normalizeParameterObject(rule.set) : undefined
      return {
        match,
        ...(set && Object.keys(set).length > 0 ? { set } : {}),
        ...(rule.remove && rule.remove.length > 0 ? { remove: [...rule.remove] } : {}),
      }
    })
    if (rules.length > 0) next.parameter_overrides = rules
    else delete next.parameter_overrides
  }
  return next
}

function normalizeParameterObject(
  value: Record<string, unknown>,
): Record<string, ParameterJSONValue> {
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, nested]) => [key, normalizeParameterValue(nested)]),
  )
}

function normalizeParameterValue(value: unknown): ParameterJSONValue {
  if (Array.isArray(value)) return value.map(normalizeParameterValue)
  if (value !== null && typeof value === 'object')
    return normalizeParameterObject(value as Record<string, unknown>)
  if (
    value === null ||
    typeof value === 'boolean' ||
    typeof value === 'number' ||
    typeof value === 'string'
  )
    return value
  throw new TypeError('invalid JSON value')
}

export function createGroupSettingsDraft(group: GroupSettingsDto): GroupSettingsDraft {
  return {
    ...group,
    siteUserId: group.site_access.user_id,
    siteAccessToken: '',
    clearSiteAccessToken: false,
    confirmSiteRootChange: false,
    params: { ...group.params },
    overrides: cloneOverrides(group.overrides),
  }
}

export function extractSiteRootUrl(providerUrl: string): string {
  try {
    const value = providerUrl.trim()
    if (
      !/^https?:\/\//i.test(value) ||
      /[\\\x00-\x1f\x7f]/.test(value) ||
      /%(?![\da-f]{2})/i.test(value.split('?')[0])
    )
      return ''
    const authority = value.match(/^https?:\/\/([^/?#]*)/i)?.[1] ?? ''
    // net/url rejects escaped ASCII in hostnames but permits escaped UTF-8.
    if (/%[0-7][\da-f]/i.test(authority)) return ''
    const port = authority.match(/:(\d+)$/)?.[1]
    // net/url historically accepts numeric ports above 65535. Preserve those
    // saved URLs even though WHATWG URL refuses to parse them.
    const oversizedPort = port && Number(port) > 65535 ? port : ''
    const parseValue = oversizedPort
      ? value.replace(/^(https?:\/\/[^/?#]*):\d+(?=[/?#]|$)/i, '$1:1')
      : value
    const url = new URL(parseValue)
    if (
      (url.protocol !== 'http:' && url.protocol !== 'https:') ||
      url.username ||
      url.password ||
      url.hash ||
      /^https?:\/\/[^/?#]*@/i.test(value)
    )
      return ''
    return `${url.protocol}//${oversizedPort ? `${url.hostname}:${oversizedPort}` : url.host}`
  } catch {
    return ''
  }
}

export function isValidProviderUrlForWrite(providerUrl: string): boolean {
  const root = extractSiteRootUrl(providerUrl)
  if (!root) return false
  const authority = providerUrl.trim().match(/^https?:\/\/([^/?#]*)/i)?.[1] ?? ''
  const port = authority.match(/:(\d+)$/)?.[1]
  return !port || Number(port) <= 65535
}

/**
 * Format a group ratio with its full precision, keeping at least 1 decimal place.
 * Rounding is deliberately avoided so small ratios such as 0.04 never collapse to 0.
 */
export function formatGroupRatio(r: number): string {
  return Number.isInteger(r) ? r.toFixed(1) : String(r)
}

export function requiresSiteRootConfirmation(
  base: GroupSettingsDto,
  draft: GroupSettingsDraft,
): boolean {
  return (
    base.site_access.token_configured &&
    !draft.clearSiteAccessToken &&
    !draft.siteAccessToken.trim() &&
    extractSiteRootUrl(draft.provider_url ?? '') !==
      extractSiteRootUrl(base.site_access.site_root_url)
  )
}

export function preserveChannelParams(
  current: ChannelParamsDto,
  fields: readonly ChannelFieldDto[],
): ChannelParamsDto {
  return Object.fromEntries(
    fields
      .filter(
        ({ key, required, default_value: defaultValue }) =>
          Object.prototype.hasOwnProperty.call(current, key) || required || defaultValue !== null,
      )
      .map(({ key, default_value: defaultValue }) => [key, current[key] ?? defaultValue ?? '']),
  )
}

export function setGroupConfigOverride(
  draft: GroupSettingsDraft,
  key: GroupTimeoutKey,
  enabled: boolean,
  effective: number,
): GroupSettingsDraft {
  const overrides = cloneOverrides(draft.overrides)
  if (enabled) overrides[key] = effective
  else delete overrides[key]
  return { ...draft, overrides }
}

export function setGroupPolicyCountOverride(
  draft: GroupSettingsDraft,
  key: GroupPolicyCountKey,
  enabled: boolean,
  effective: number,
): GroupSettingsDraft {
  const overrides = cloneOverrides(draft.overrides)
  if (enabled) overrides[key] = effective
  else delete overrides[key]
  return { ...draft, overrides }
}

export function buildGroupSettingsPatch(
  base: GroupSettingsDto,
  draft: GroupSettingsDraft,
  forceSiteRootConfirmation = false,
): GroupSettingsUpdateRequest {
  const patch: GroupSettingsUpdateRequest = {}
  const overrides = normalizeOverrides(draft.overrides)
  if (draft.channel_id !== base.channel_id) patch.channel_id = draft.channel_id
  if (draft.name.trim() !== base.name) patch.name = draft.name.trim()
  const params = Object.fromEntries(
    Object.entries(draft.params)
      .map(([key, value]) => [key, value.trim()] as const)
      .sort(([left], [right]) => left.localeCompare(right)),
  )
  const baseParams = Object.fromEntries(
    Object.entries(base.params).sort(([left], [right]) => left.localeCompare(right)),
  )
  if (JSON.stringify(params) !== JSON.stringify(baseParams)) patch.params = params
  const providerURL = draft.provider_url?.trim() || null
  if (providerURL !== base.provider_url) patch.provider_url = providerURL
  if (draft.enabled !== base.enabled) patch.enabled = draft.enabled
  const siteAccess: NonNullable<GroupSettingsUpdateRequest['site_access']> = {}
  if (draft.siteUserId.trim() !== base.site_access.user_id)
    siteAccess.user_id = draft.siteUserId.trim()
  if (draft.siteAccessToken.trim()) siteAccess.access_token = draft.siteAccessToken
  if (draft.clearSiteAccessToken) siteAccess.clear_access_token = true
  if (
    (forceSiteRootConfirmation || requiresSiteRootConfirmation(base, draft)) &&
    draft.confirmSiteRootChange
  )
    siteAccess.confirm_site_root_change = true
  if (Object.keys(siteAccess).length) patch.site_access = siteAccess

  if (JSON.stringify(overrides) !== JSON.stringify(normalizeOverrides(base.overrides))) {
    patch.overrides = overrides
  }
  return patch
}
