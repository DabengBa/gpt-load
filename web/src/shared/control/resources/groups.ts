import { keepPreviousData, type QueryClient, type QueryFunctionContext } from '@tanstack/query-core'

import type { ApiClient } from '@shared/http/client'
import type {
  ChannelParamsDto,
  GroupCollectionFilters,
  GroupCollectionItemDto,
  GroupCollectionSiteInfoDto,
  GroupCollectionPaginationDto,
  GroupCollectionResponseDto,
  GroupCollectionStatus,
  GroupCollectionSummaryDto,
  GroupUnavailableReason,
  GroupModelItemDto,
  GroupModelsDto,
  GroupOptionDto,
  EntryCircuitBreakerDto,
  GroupSiteAccessDto,
  GroupSiteAccessPatchInput,
  GroupSiteInfoSnapshotDto,
  GroupSettingsDto,
  GroupSummaryDto,
  HeaderRulesDto,
  GroupRuntimeConfigDto,
  GroupEffectiveConfigDto,
  ParameterJSONValue,
  ParameterOverrideMatchDto,
  ParameterOverrideRuleDto,
  ProxyConfigInput,
  ProxyMutation,
} from '@shared/control/types'
import { enabledDataProtocols } from '@shared/control/protocols'
import { InvalidResponseError } from '@shared/http/errors'
import { controlQueryKeys, normalizeGroupCollectionFilters } from '@shared/control/query-keys'
import { projectChannelID } from '@shared/control/resources/channels'
import { projectModelCandidate, type ModelCandidate } from '@shared/control/resources/providers'
import { isJSONSafeNumber } from '@shared/lib/json-number'
import { extractSiteRootUrl } from '@shared/domain/groups/settings/group-settings-patch'

import {
  assertNoSecretLikeFields,
  projectArray,
  projectBoolean,
  projectEpochMilliseconds,
  projectEnum,
  projectFiniteNumber,
  projectHTTPURL,
  projectNullableEpochMilliseconds,
  projectRecord,
  projectSafeInteger,
  projectString,
} from './projector'
import { projectProxyView } from './proxy'

const groupSummaryFields = [
  'id',
  'name',

  'channel_id',
  'params',
  'provider_url',
  'service_status',
  'service_status_reason',
  'credential_configured',
  'credential_status',
  'model_count',
] as const
const groupSettingsFields = [
  'name',

  'channel_id',
  'params',
  'provider_url',
  'enabled',
  'overrides',
  'effective',
  'proxy',
  'site_access',
] as const
const groupModelsFields = ['items', 'total', 'pending'] as const
const groupModelItemFields = [
  'id',
  'concurrency_limit',
  'alias',
  'alias_enabled',
  'client_model',
  'test_alias',
  'entry_id',
  'weight',
  'priority',
  'circuit_breaker',
  'pricing_status',
  'price_id',
  'reasoning_effort',
  'enabled',
] as const
const groupCollectionFields = ['observed_at_ms', 'summary', 'items', 'pagination'] as const
const groupCollectionSummaryFields = ['total', 'available', 'unavailable', 'disabled'] as const
const groupCollectionItemFields = [
  'id',
  'name',

  'channel_id',
  'params',
  'provider_url',
  'status',
  'model_count',
  'client_model_count',
  'credential_configured',
  'credential_status',
  'site_info',
] as const
const groupCollectionPaginationFields = ['page', 'page_size', 'total_items', 'total_pages'] as const
const groupOptionFields = [
  'id',
  'name',
  'channel_id',
  'params',
  'provider_url',
  'enabled',
  'models',
] as const

const groupCollectionStatuses = ['available', 'unavailable', 'disabled'] as const
const groupUnavailableReasons = ['no_available_credentials', 'no_models'] as const
const groupSiteInfoStatuses = ['unconfigured', 'pending', 'ok', 'error'] as const
const groupSiteInfoFields = [
  'status',
  'site_root_url',
  'site_name',
  'remaining_amount',
  'currency_symbol',
  'effective_group',
  'group_ratio',
  'fetched_at_ms',
  'last_error',
] as const
const groupSiteInfoSnapshotFields = [...groupSiteInfoFields, 'announcement_count'] as const
const groupSiteAccessFields = ['user_id', 'token_configured', 'site_root_url', 'snapshot'] as const
/**
 * The backend re-reads each site on its own 60 minute cadence and only exposes the
 * result after that tick, so a client interval matching 60 minutes can land just
 * before the tick and stay stale for almost an hour. Visible pages therefore poll
 * once a minute — the backend resolver ticks once a minute — and `pending` waits for
 * the first fetch.
 */
const siteInfoPendingRefetchIntervalMs = 2_000
export const siteInfoVisibleRefetchIntervalMs = 60_000
const runtimeSettingFields = [
  'first_byte_timeout',
  'request_timeout',
  'stream_idle_timeout',
  'blacklist_threshold',
  'header_rules',
  'affinity_enabled',
  'responses_websocket_enabled',
  'responses_reasoning_status_filter_enabled',
] as const
const groupRuntimeSettingFields = [...runtimeSettingFields, 'parameter_overrides'] as const

export type {
  HeaderRulesDto,
  GroupRuntimeConfigDto,
  GroupEffectiveConfigDto,
  GroupModelItemDto,
  GroupModelsDto,
  GroupSettingsDto,
  GroupSummaryDto,
} from '@shared/control/types'

export type GroupSettingsUpdateRequest = Partial<{
  name: string
  channel_id: string
  params: ChannelParamsDto
  provider_url: string | null
  enabled: boolean
  overrides: GroupRuntimeConfigDto
  proxy: ProxyMutation
  site_access: GroupSiteAccessPatchInput
}>

export interface AccessKeyReferenceDto {
  id: number
  name: string
}

export interface GroupInUseData {
  access_keys: AccessKeyReferenceDto[]
}

export interface ModelDiscoveryRequest {
  channel_id: string
  params: ChannelParamsDto
  credentials?: string
  proxy?: ProxyConfigInput
}

export interface ModelDiscoveryResult {
  models: ModelCandidate[]
}

export interface GroupModelUpdateDto {
  id: string
  concurrency_limit?: number
  alias: string
  alias_enabled: boolean
  entry_id?: string
  weight?: number | null
  priority?: number | null
  circuit_breaker?: EntryCircuitBreakerDto | null
  reasoning_effort?: string | null
  enabled?: boolean
}

export interface GroupModelsReplaceRequest {
  models: GroupModelUpdateDto[]
}

export interface GroupCreateRequest {
  name?: string
  channel_id: string
  params: ChannelParamsDto
  provider_url?: string | null
  models: GroupModelUpdateDto[]
  credential?: string
  proxy?: ProxyConfigInput
  site_access?: { user_id: string; access_token: string }
  confirm_same_target: boolean
}

export interface GroupCreateResult {
  group_id: number
  group_name: string
  credential_id: number
}

export interface GroupCopyResult {
  group_id: number
  group_name: string
  credential_id: number | null
}

export interface CredentialImportRequest {
  credential: string
}

export interface CredentialImportResult {
  group_id: number
  credential_id: number
}

const credentialValidationReasonCodes = [
  'unknown_field',
  'required',
  'invalid_type',
  'invalid_json',
  'missing_required_field',
  'conflicting_auth_methods',
  'incomplete_auth_method',
  'invalid_value',
] as const

export interface CredentialValidationData {
  entry: number
  field: string
  reason_code: (typeof credentialValidationReasonCodes)[number]
}

export interface SameTargetConflictData {
  groups: Array<{ id: number; name: string }>
}

export function readCredentialValidationData(value: unknown): CredentialValidationData | null {
  try {
    const record = projectRecord(value)
    assertNoSecretLikeFields(record, ['entry', 'field', 'reason_code'])
    return {
      entry: projectSafeInteger(record.entry, { minimum: 1, maximum: 1_000 }),
      field: projectNonBlankString(record.field),
      reason_code: projectEnum(record.reason_code, credentialValidationReasonCodes),
    }
  } catch {
    return null
  }
}

function projectNonBlankString(value: unknown): string {
  const result = projectString(value)
  if (result.trim().length === 0) throw new InvalidResponseError()
  return result
}

function projectChannelParams(value: unknown): ChannelParamsDto {
  const record = projectRecord(value)
  const params: ChannelParamsDto = {}
  for (const [key, paramValue] of Object.entries(record)) {
    if (key !== key.trim() || !/^[a-z][a-z0-9_]*$/u.test(key)) throw new InvalidResponseError()
    params[key] = projectString(paramValue)
  }
  return params
}

function projectHeaderRules(value: unknown): HeaderRulesDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, ['set', 'remove'])
  const setRecord = projectRecord(record.set)
  const set: Record<string, string> = {}
  for (const [name, headerValue] of Object.entries(setRecord)) {
    if (name.trim().length === 0) throw new InvalidResponseError()
    set[name] = projectString(headerValue, { allowEmpty: true })
  }
  return {
    set,
    remove: projectArray(record.remove, projectNonBlankString),
  }
}

function projectParameterJSONValue(value: unknown, depth = 0): ParameterJSONValue {
  if (depth > 64) throw new InvalidResponseError()
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return value
  if (typeof value === 'number') {
    const number = projectFiniteNumber(value)
    if (!isJSONSafeNumber(number)) throw new InvalidResponseError()
    return number
  }
  if (Array.isArray(value)) return value.map((item) => projectParameterJSONValue(item, depth + 1))
  const record = projectRecord(value)
  return Object.fromEntries(
    Object.entries(record).map(([key, nested]) => [
      key,
      projectParameterJSONValue(nested, depth + 1),
    ]),
  )
}

function projectParameterOverrideMatch(value: unknown): ParameterOverrideMatchDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, ['protocol', 'model'])
  const result: ParameterOverrideMatchDto = {}
  if (Object.prototype.hasOwnProperty.call(record, 'protocol'))
    result.protocol = projectEnum(record.protocol, enabledDataProtocols)
  if (Object.prototype.hasOwnProperty.call(record, 'model'))
    result.model = projectString(record.model, { allowEmpty: true })
  return result
}

function projectParameterOverrideRule(value: unknown): ParameterOverrideRuleDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, ['match', 'set', 'remove'])
  const result: ParameterOverrideRuleDto = {
    match: Object.prototype.hasOwnProperty.call(record, 'match')
      ? projectParameterOverrideMatch(record.match)
      : {},
  }
  if (Object.prototype.hasOwnProperty.call(record, 'set')) {
    const set = projectRecord(record.set)
    result.set = Object.fromEntries(
      Object.entries(set).map(([key, nested]) => [key, projectParameterJSONValue(nested, 1)]),
    )
  }
  if (Object.prototype.hasOwnProperty.call(record, 'remove'))
    result.remove = projectArray(record.remove, (path) => projectString(path, { allowEmpty: true }))
  return result
}

function projectParameterOverrides(value: unknown): ParameterOverrideRuleDto[] {
  return projectArray(value, projectParameterOverrideRule)
}

function projectRuntimeConfig(value: unknown, complete: false): GroupRuntimeConfigDto
function projectRuntimeConfig(value: unknown, complete: true): GroupEffectiveConfigDto
function projectRuntimeConfig(
  value: unknown,
  complete: boolean,
): GroupRuntimeConfigDto | GroupEffectiveConfigDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, complete ? runtimeSettingFields : groupRuntimeSettingFields)
  const result: GroupRuntimeConfigDto = {}

  for (const field of ['first_byte_timeout', 'request_timeout', 'stream_idle_timeout'] as const) {
    if (complete || Object.prototype.hasOwnProperty.call(record, field)) {
      result[field] = projectSafeInteger(record[field], { minimum: 1 })
    }
  }
  if (complete || Object.prototype.hasOwnProperty.call(record, 'blacklist_threshold')) {
    result.blacklist_threshold = projectSafeInteger(record.blacklist_threshold, { minimum: 0 })
  }
  if (complete || Object.prototype.hasOwnProperty.call(record, 'header_rules')) {
    result.header_rules = projectHeaderRules(record.header_rules)
  }
  if (complete || Object.prototype.hasOwnProperty.call(record, 'affinity_enabled')) {
    result.affinity_enabled = projectBoolean(record.affinity_enabled)
  }
  if (complete || Object.prototype.hasOwnProperty.call(record, 'responses_websocket_enabled')) {
    result.responses_websocket_enabled = projectBoolean(record.responses_websocket_enabled)
  }
  if (
    complete ||
    Object.prototype.hasOwnProperty.call(record, 'responses_reasoning_status_filter_enabled')
  ) {
    result.responses_reasoning_status_filter_enabled = projectBoolean(
      record.responses_reasoning_status_filter_enabled,
    )
  }
  if (!complete && Object.prototype.hasOwnProperty.call(record, 'parameter_overrides')) {
    result.parameter_overrides = projectParameterOverrides(record.parameter_overrides)
  }
  return result as GroupRuntimeConfigDto | GroupEffectiveConfigDto
}

function projectNullableHTTPURL(value: unknown): string | null {
  if (value === null) return null
  if (typeof value === 'string' && value === value.trim() && extractSiteRootUrl(value)) return value
  return projectHTTPURL(value)
}

/** Balances can go negative on the upstream site, so only nullability and finiteness are enforced. */
function projectNullableFiniteNumber(value: unknown): number | null {
  return value === null ? null : projectFiniteNumber(value)
}

/** `site_root_url` is derived from the provider URL, so it is either empty or an http(s) root. */
function projectSiteRootURL(value: unknown): string {
  if (value === '') return ''
  if (typeof value === 'string' && value === value.trim() && extractSiteRootUrl(value)) return value
  return projectHTTPURL(value)
}

/** Mirrors `GroupCollectionSiteInfo`: absent credentials surface as `null`, never as a blank object. */
export function projectGroupSiteInfo(value: unknown): GroupCollectionSiteInfoDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, groupSiteInfoFields)
  return {
    status: projectEnum(record.status, groupSiteInfoStatuses),
    site_root_url: projectSiteRootURL(record.site_root_url),
    site_name: projectString(record.site_name, { allowEmpty: true }),
    remaining_amount: projectNullableFiniteNumber(record.remaining_amount),
    currency_symbol: projectString(record.currency_symbol, { allowEmpty: true }),
    effective_group: projectString(record.effective_group, { allowEmpty: true }),
    group_ratio: projectNullableFiniteNumber(record.group_ratio),
    fetched_at_ms: projectNullableEpochMilliseconds(record.fetched_at_ms),
    last_error: projectString(record.last_error, { allowEmpty: true }),
  }
}

function projectGroupSiteInfoSnapshot(value: unknown): GroupSiteInfoSnapshotDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, groupSiteInfoSnapshotFields)
  return {
    ...projectGroupSiteInfo(
      Object.fromEntries(groupSiteInfoFields.map((field) => [field, record[field]] as const)),
    ),
    announcement_count: projectSafeInteger(record.announcement_count, { minimum: 0 }),
  }
}

export function projectGroupSiteAccess(value: unknown): GroupSiteAccessDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, groupSiteAccessFields)
  const tokenConfigured = projectBoolean(record.token_configured)
  const snapshot = record.snapshot === null ? null : projectGroupSiteInfoSnapshot(record.snapshot)
  // The backend only exposes a snapshot once a token is stored.
  if (tokenConfigured !== (snapshot !== null)) throw new InvalidResponseError()
  return {
    user_id: projectString(record.user_id, { allowEmpty: true }),
    token_configured: tokenConfigured,
    site_root_url: projectSiteRootURL(record.site_root_url),
    snapshot,
  }
}

export function projectGroupSummary(value: unknown): GroupSummaryDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, groupSummaryFields)
  const serviceStatus = projectEnum(record.service_status, groupCollectionStatuses)
  const serviceStatusReason =
    record.service_status_reason === null
      ? null
      : (projectEnum(
          record.service_status_reason,
          groupUnavailableReasons,
        ) as GroupUnavailableReason)
  if ((serviceStatus === 'unavailable') !== (serviceStatusReason !== null)) {
    throw new InvalidResponseError()
  }
  if (projectBoolean(record.credential_configured) !== (record.credential_status !== null)) {
    throw new InvalidResponseError()
  }
  return {
    id: projectSafeInteger(record.id, { minimum: 1 }),
    name: projectNonBlankString(record.name),
    channel_id: projectChannelID(record.channel_id),
    params: projectChannelParams(record.params),
    provider_url: projectNullableHTTPURL(record.provider_url),

    service_status: serviceStatus,
    service_status_reason: serviceStatusReason,
    credential_configured: projectBoolean(record.credential_configured),
    credential_status:
      record.credential_status === null
        ? null
        : projectEnum(record.credential_status, [
            'available',
            'cooldown',
            'blacklisted',
            'disabled',
          ] as const),
    model_count: projectSafeInteger(record.model_count, { minimum: 0 }),
  }
}

export function projectGroupSettings(value: unknown): GroupSettingsDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, groupSettingsFields)
  return {
    name: projectNonBlankString(record.name),
    channel_id: projectChannelID(record.channel_id),
    params: projectChannelParams(record.params),
    provider_url: projectNullableHTTPURL(record.provider_url),

    enabled: projectBoolean(record.enabled),
    overrides: projectRuntimeConfig(record.overrides, false),
    effective: projectRuntimeConfig(record.effective, true),
    proxy: projectProxyView(record.proxy),
    site_access: projectGroupSiteAccess(record.site_access),
  }
}

function projectNullableSafeInteger(value: unknown): number | null {
  return value === null || value === undefined ? null : projectSafeInteger(value)
}

function projectGroupModelCircuitBreaker(value: unknown): EntryCircuitBreakerDto | null {
  if (value === null) return null
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, ['blacklist_threshold', 'cooldown_seconds'])
  const result: EntryCircuitBreakerDto = {}
  if (Object.prototype.hasOwnProperty.call(record, 'blacklist_threshold')) {
    result.blacklist_threshold =
      record.blacklist_threshold === null
        ? null
        : projectSafeInteger(record.blacklist_threshold, { minimum: 1 })
  }
  if (Object.prototype.hasOwnProperty.call(record, 'cooldown_seconds')) {
    result.cooldown_seconds =
      record.cooldown_seconds === null
        ? null
        : projectSafeInteger(record.cooldown_seconds, { minimum: 0 })
  }
  return result
}

function projectGroupModelItem(value: unknown): GroupModelItemDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, groupModelItemFields)
  const alias = projectString(record.alias, { allowEmpty: true })
  const aliasEnabled = projectBoolean(record.alias_enabled)
  const id = projectNonBlankString(record.id)
  const clientModel = projectNonBlankString(record.client_model)
  const testAlias = projectString(record.test_alias)
  if (!/^[a-z0-9]{6}$/u.test(testAlias)) throw new InvalidResponseError()
  if ((alias !== '') !== aliasEnabled || clientModel !== (aliasEnabled ? alias : id)) {
    throw new InvalidResponseError()
  }
  const result: GroupModelItemDto = {
    id,
    concurrency_limit: projectSafeInteger(record.concurrency_limit, {
      minimum: 0,
      maximum: 2147483647,
    }),
    alias,
    alias_enabled: aliasEnabled,
    client_model: clientModel,
    test_alias: testAlias,
    weight: projectNullableSafeInteger(record.weight),
    priority: projectNullableSafeInteger(record.priority),
    pricing_status: projectEnum(record.pricing_status, ['pending', 'configured'] as const),
  }
  if (Object.prototype.hasOwnProperty.call(record, 'entry_id')) {
    result.entry_id = projectNonBlankString(record.entry_id)
  }
  if (Object.prototype.hasOwnProperty.call(record, 'circuit_breaker')) {
    result.circuit_breaker = projectGroupModelCircuitBreaker(record.circuit_breaker)
  }
  if (Object.prototype.hasOwnProperty.call(record, 'price_id')) {
    result.price_id = projectSafeInteger(record.price_id, { minimum: 1 })
  }
  if (Object.prototype.hasOwnProperty.call(record, 'reasoning_effort')) {
    result.reasoning_effort = projectString(record.reasoning_effort, { allowEmpty: true })
  }
  if (Object.prototype.hasOwnProperty.call(record, 'enabled')) {
    result.enabled = projectBoolean(record.enabled)
  }
  return result
}

export function projectGroupModels(value: unknown): GroupModelsDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, groupModelsFields)
  const items = projectArray(record.items, projectGroupModelItem)
  const total = projectSafeInteger(record.total, { minimum: 0 })
  const pending = projectSafeInteger(record.pending, { minimum: 0 })
  // 同一对外名允许映射到多个上游模型（设计 §3）；client_model 不再要求唯一。
  if (
    items.length !== total ||
    pending > total ||
    items.filter(({ pricing_status }) => pricing_status === 'pending').length !== pending
  ) {
    throw new InvalidResponseError()
  }
  return { items, total, pending }
}

function projectGroupCollectionSummary(value: unknown): GroupCollectionSummaryDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, groupCollectionSummaryFields)
  const result = {
    total: projectSafeInteger(record.total, { minimum: 0 }),
    available: projectSafeInteger(record.available, { minimum: 0 }),
    unavailable: projectSafeInteger(record.unavailable, { minimum: 0 }),
    disabled: projectSafeInteger(record.disabled, { minimum: 0 }),
  }
  if (result.total !== result.available + result.unavailable + result.disabled) {
    throw new InvalidResponseError()
  }
  return result
}

function projectGroupCollectionItem(value: unknown): GroupCollectionItemDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, groupCollectionItemFields)
  const status = projectEnum(record.status, groupCollectionStatuses) as GroupCollectionStatus
  const credentialConfigured = projectBoolean(record.credential_configured)
  const credentialStatus =
    record.credential_status === null
      ? null
      : projectEnum(record.credential_status, [
          'available',
          'cooldown',
          'blacklisted',
          'disabled',
        ] as const)
  if (credentialConfigured !== (credentialStatus !== null)) throw new InvalidResponseError()
  const modelCount = projectSafeInteger(record.model_count, { minimum: 0 })
  const siteInfo = record.site_info === null ? null : projectGroupSiteInfo(record.site_info)

  return {
    id: projectSafeInteger(record.id, { minimum: 1 }),
    name: projectNonBlankString(record.name),
    channel_id: projectChannelID(record.channel_id),
    params: projectChannelParams(record.params),
    provider_url: projectNullableHTTPURL(record.provider_url),
    status,

    model_count: modelCount,
    client_model_count: projectSafeInteger(record.client_model_count, { minimum: 0 }),
    credential_configured: credentialConfigured,
    credential_status: credentialStatus,
    site_info: siteInfo,
  }
}

function projectGroupCollectionPagination(value: unknown): GroupCollectionPaginationDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, groupCollectionPaginationFields)
  return {
    page: projectSafeInteger(record.page, { minimum: 1 }),
    page_size: projectSafeInteger(record.page_size, { minimum: 100, maximum: 100 }) as 100,
    total_items: projectSafeInteger(record.total_items, { minimum: 0 }),
    total_pages: projectSafeInteger(record.total_pages, { minimum: 0 }),
  }
}

function expectedGroupCollectionTotalPages(totalItems: number, pageSize: number): number {
  if (totalItems === 0) return 0
  const completePages = Math.floor(totalItems / pageSize)
  return completePages + (totalItems % pageSize === 0 ? 0 : 1)
}

function expectedGroupCollectionPageItems(pagination: GroupCollectionPaginationDto): number {
  if (pagination.total_items === 0 || pagination.page > pagination.total_pages) return 0
  if (pagination.page < pagination.total_pages) return pagination.page_size
  const finalPageItems = pagination.total_items % pagination.page_size
  return finalPageItems === 0 ? pagination.page_size : finalPageItems
}

export function projectGroupCollection(value: unknown): GroupCollectionResponseDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, groupCollectionFields)
  const items = projectArray(record.items, projectGroupCollectionItem)
  const summary = projectGroupCollectionSummary(record.summary)
  const pagination = projectGroupCollectionPagination(record.pagination)
  if (
    pagination.total_items > summary.total ||
    pagination.total_pages !==
      expectedGroupCollectionTotalPages(pagination.total_items, pagination.page_size) ||
    items.length !== expectedGroupCollectionPageItems(pagination) ||
    new Set(items.map(({ id }) => id)).size !== items.length
  ) {
    throw new InvalidResponseError()
  }
  return {
    observed_at_ms: projectEpochMilliseconds(record.observed_at_ms),
    summary,
    items,
    pagination,
  }
}

function projectGroupOption(value: unknown): GroupOptionDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, groupOptionFields)
  const models = projectArray(record.models, projectNonBlankString)
  if (new Set(models).size !== models.length) throw new InvalidResponseError()
  return {
    id: projectSafeInteger(record.id, { minimum: 1 }),
    name: projectNonBlankString(record.name),
    channel_id: projectChannelID(record.channel_id),
    params: projectChannelParams(record.params),
    provider_url: projectNullableHTTPURL(record.provider_url),
    enabled: projectBoolean(record.enabled),
    models,
  }
}

export function projectGroupOptions(value: unknown): GroupOptionDto[] {
  const options = projectArray(value, projectGroupOption)
  if (new Set(options.map(({ id }) => id)).size !== options.length) {
    throw new InvalidResponseError()
  }
  return options
}

function projectDiscoveryResult(value: unknown): ModelDiscoveryResult {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, ['models'])
  const models = projectArray(record.models, projectModelCandidate)
  if (new Set(models.map(({ id }) => id)).size !== models.length) throw new InvalidResponseError()
  return { models }
}

function projectGroupCreateResult(value: unknown): GroupCreateResult {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, ['group_id', 'group_name', 'credential_id'])
  return {
    group_id: projectSafeInteger(record.group_id, { minimum: 1 }),
    group_name: projectNonBlankString(record.group_name),
    credential_id: projectSafeInteger(record.credential_id, { minimum: 1 }),
  }
}

function projectGroupCopyResult(value: unknown): GroupCopyResult {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, ['group_id', 'group_name', 'credential_id'])
  return {
    group_id: projectSafeInteger(record.group_id, { minimum: 1 }),
    group_name: projectNonBlankString(record.group_name),
    credential_id:
      record.credential_id === null
        ? null
        : projectSafeInteger(record.credential_id, { minimum: 1 }),
  }
}

function projectCredentialImportResult(value: unknown): CredentialImportResult {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, ['group_id', 'credential_id'])
  return {
    group_id: projectSafeInteger(record.group_id, { minimum: 1 }),
    credential_id: projectSafeInteger(record.credential_id, { minimum: 1 }),
  }
}

function projectIDName(value: unknown): { id: number; name: string } {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, ['id', 'name'])
  return {
    id: projectSafeInteger(record.id, { minimum: 1 }),
    name: projectNonBlankString(record.name),
  }
}

export function isSameTargetConflictData(value: unknown): value is SameTargetConflictData {
  try {
    const record = projectRecord(value)
    const groups = projectArray(record.groups, projectIDName)
    return groups.length > 0
  } catch {
    return false
  }
}

export function isGroupInUseData(value: unknown): value is GroupInUseData {
  try {
    const record = projectRecord(value)
    const accessKeys = projectArray(record.access_keys, projectIDName)
    return accessKeys.length > 0
  } catch {
    return false
  }
}

export async function listGroupCollection(
  client: ApiClient,
  filters: GroupCollectionFilters,
  signal?: AbortSignal,
): Promise<GroupCollectionResponseDto> {
  const normalized = normalizeGroupCollectionFilters(filters)
  const params = new URLSearchParams()
  params.set('include', 'site_info')
  params.set('sort', normalized.sort)
  params.set('page', String(normalized.page))
  params.set('page_size', String(normalized.page_size))
  if (normalized.q !== undefined) params.set('q', normalized.q)
  if (normalized.status !== undefined) params.set('status', normalized.status)
  const result = projectGroupCollection(
    await client.request(`/api/groups?${params.toString()}`, { method: 'GET', signal }),
  )
  if (result.pagination.page !== normalized.page) throw new InvalidResponseError()
  return result
}

export async function listGroupOptions(
  client: ApiClient,
  signal?: AbortSignal,
): Promise<GroupOptionDto[]> {
  return projectGroupOptions(await client.request('/api/groups/options', { method: 'GET', signal }))
}

export async function getGroupSummary(
  client: ApiClient,
  groupID: number,
  signal?: AbortSignal,
): Promise<GroupSummaryDto> {
  return projectGroupSummary(
    await client.request(`/api/groups/${groupID}`, { method: 'GET', signal }),
  )
}

export async function getGroupSettings(
  client: ApiClient,
  groupID: number,
  signal?: AbortSignal,
): Promise<GroupSettingsDto> {
  return projectGroupSettings(
    await client.request(`/api/groups/${groupID}/settings?include=site_info`, {
      method: 'GET',
      signal,
    }),
  )
}

export async function getGroupModels(
  client: ApiClient,
  groupID: number,
  signal?: AbortSignal,
): Promise<GroupModelsDto> {
  return projectGroupModels(
    await client.request(`/api/groups/${groupID}/models?include_schedule=true`, {
      method: 'GET',
      signal,
    }),
  )
}

export const manualGroupQueryOptions = {
  staleTime: Number.POSITIVE_INFINITY,
  refetchOnWindowFocus: false,
  refetchOnReconnect: false,
} as const

/** Read-only group projections may lag briefly, but must not become a manual snapshot. */
export const readOnlyGroupQueryOptions = {
  staleTime: 30_000,
  refetchOnWindowFocus: true,
  refetchOnReconnect: true,
} as const

export function groupCollectionQueryOptions(client: ApiClient, filters: GroupCollectionFilters) {
  const key = controlQueryKeys.groups.collection(filters)
  return {
    ...readOnlyGroupQueryOptions,
    queryKey: key,
    queryFn: ({ queryKey, signal }: QueryFunctionContext<typeof key>) =>
      listGroupCollection(client, queryKey[3], signal),
    placeholderData: keepPreviousData,
    // A pending site info resolves on the next sync tick; a resolved snapshot changes at
    // the next 60 minute backend sync, which a visible page picks up within a minute.
    // Polling pauses while the tab is hidden and resumes through the visibility refetch.
    refetchIntervalInBackground: false,
    refetchInterval: (query: { state: { data: GroupCollectionResponseDto | undefined } }) =>
      query.state.data?.items.some((item) => item.site_info?.status === 'pending')
        ? siteInfoPendingRefetchIntervalMs
        : siteInfoVisibleRefetchIntervalMs,
  }
}

export function groupOptionsQueryOptions(client: ApiClient, enabled: boolean = true) {
  return {
    ...readOnlyGroupQueryOptions,
    queryKey: controlQueryKeys.groups.options(),
    queryFn: ({ signal }: QueryFunctionContext) => listGroupOptions(client, signal),
    enabled: enabled,
    // Group 选项目录可能由其他页面或标签页修改，进入消费者页面时必须重新校验。
    refetchOnMount: 'always' as const,
  }
}

export function groupSummaryQueryOptions(client: ApiClient, groupID: number | undefined) {
  return {
    ...readOnlyGroupQueryOptions,
    queryKey:
      groupID === undefined
        ? controlQueryKeys.groups.summaries()
        : controlQueryKeys.groups.summary(groupID),
    queryFn: ({ signal }: QueryFunctionContext) => {
      const id = groupID
      if (id === undefined) throw new InvalidResponseError()
      return getGroupSummary(client, id, signal)
    },
    enabled: groupID !== undefined,
  }
}

export function groupSettingsQueryOptions(client: ApiClient, groupID: number | undefined) {
  return {
    ...manualGroupQueryOptions,
    queryKey:
      groupID === undefined
        ? controlQueryKeys.groups.settingsAll()
        : controlQueryKeys.groups.settings(groupID),
    queryFn: ({ signal }: QueryFunctionContext) => {
      const id = groupID
      if (id === undefined) throw new InvalidResponseError()
      return getGroupSettings(client, id, signal)
    },
    enabled: groupID !== undefined,
    // Same cadence as the collection: follow a pending fetch closely, then poll once a
    // minute so a snapshot written by the 60 minute backend sync appears without a
    // reload. Unsaved drafts are protected by the hydration guards in the settings tab.
    refetchIntervalInBackground: false,
    refetchInterval: (query: { state: { data: GroupSettingsDto | undefined } }) =>
      query.state.data?.site_access?.snapshot?.status === 'pending'
        ? siteInfoPendingRefetchIntervalMs
        : siteInfoVisibleRefetchIntervalMs,
  }
}

export function groupModelsQueryOptions(client: ApiClient, groupID: number | undefined) {
  return {
    ...manualGroupQueryOptions,
    queryKey:
      groupID === undefined
        ? controlQueryKeys.groups.modelsAll()
        : controlQueryKeys.groups.models(groupID),
    queryFn: ({ signal }: QueryFunctionContext) => {
      const id = groupID
      if (id === undefined) throw new InvalidResponseError()
      return getGroupModels(client, id, signal)
    },
    enabled: groupID !== undefined,
  }
}

export async function updateGroupSettings(
  client: ApiClient,
  groupID: number,
  body: GroupSettingsUpdateRequest,
  signal?: AbortSignal,
): Promise<GroupSettingsDto> {
  return projectGroupSettings(
    await client.request(`/api/groups/${groupID}/settings?include=site_info`, {
      method: 'PUT',
      json: body,
      signal,
    }),
  )
}

export async function updateGroupConfiguration(
  client: ApiClient,
  groupID: number,
  settings: GroupSettingsUpdateRequest,
  models: GroupModelUpdateDto[],
  signal?: AbortSignal,
): Promise<GroupSettingsDto> {
  return projectGroupSettings(
    await client.request(`/api/groups/${groupID}/configuration`, {
      method: 'PUT',
      json: { settings, models },
      signal,
    }),
  )
}

export async function deleteGroup(
  client: ApiClient,
  groupID: number,
  signal?: AbortSignal,
): Promise<void> {
  await client.request(`/api/groups/${groupID}`, { method: 'DELETE', signal })
}

export async function retryGroupSiteInfo(client: ApiClient, groupID: number): Promise<void> {
  await client.request(`/api/groups/${groupID}/site-info/retry`, { method: 'POST' })
}

export async function discoverModels(
  client: ApiClient,
  body: ModelDiscoveryRequest,
  signal?: AbortSignal,
): Promise<ModelDiscoveryResult> {
  return projectDiscoveryResult(
    await client.request('/api/models/discover', {
      method: 'POST',
      json: body,
      signal,
    }),
  )
}

export async function discoverGroupModels(
  client: ApiClient,
  groupID: number,
  signal?: AbortSignal,
): Promise<ModelDiscoveryResult> {
  return projectDiscoveryResult(
    await client.request(`/api/groups/${groupID}/models/discover`, {
      method: 'POST',
      signal,
    }),
  )
}

export async function replaceGroupModelsResource(
  client: ApiClient,
  groupID: number,
  body: GroupModelsReplaceRequest,
  signal?: AbortSignal,
): Promise<GroupModelsDto> {
  return projectGroupModels(
    await client.request(`/api/groups/${groupID}/models?include_schedule=true`, {
      method: 'PUT',
      json: body,
      signal,
    }),
  )
}

/** Cache writes are exact and never cause a background refetch. */
export function cacheGroupSettings(
  queryClient: QueryClient,
  groupID: number,
  settings: GroupSettingsDto,
): void {
  queryClient.setQueryData(controlQueryKeys.groups.settings(groupID), settings)
}

/** Prevent a pre-recovery poll from replacing a newer authoritative read. */
export async function cacheRecoveredGroupSettings(
  queryClient: QueryClient,
  groupID: number,
  settings: GroupSettingsDto,
): Promise<void> {
  await queryClient.cancelQueries({
    queryKey: controlQueryKeys.groups.settings(groupID),
    exact: true,
  })
  cacheGroupSettings(queryClient, groupID, settings)
}

/** Invalidate cached resources whose representations derive from Group models. */
export async function invalidateGroupModelDependents(
  queryClient: QueryClient,
  groupID: number,
): Promise<void> {
  await invalidateGroupDependents(queryClient, [
    {
      queryKey: controlQueryKeys.groups.summary(groupID),
      exact: true,
      refetchType: 'active',
    },
    {
      queryKey: controlQueryKeys.groups.collectionAll,
      refetchType: 'active',
    },
    {
      queryKey: controlQueryKeys.groups.options(),
      exact: true,
      refetchType: 'active',
    },
    {
      queryKey: controlQueryKeys.home.base(),
      exact: true,
      refetchType: 'active',
    },
    {
      queryKey: controlQueryKeys.modelPrices(),
      refetchType: 'none',
    },
    {
      queryKey: controlQueryKeys.models.all,
      refetchType: 'none',
    },
    {
      queryKey: controlQueryKeys.modelRouteSchedule.all,
      refetchType: 'active',
    },
  ])
}

/** Settings can change every Group representation; refresh active consumers and stale the rest. */
export async function invalidateGroupSettingsDependents(
  queryClient: QueryClient,
  groupID: number,
): Promise<void> {
  await invalidateGroupDependents(queryClient, [
    // Credentials and provider URL drive site announcements, so an active announcements
    // view must reload together with the settings that changed them.
    {
      queryKey: ['control', 'announcements'],
      refetchType: 'active',
    },
    {
      queryKey: controlQueryKeys.groups.summary(groupID),
      exact: true,
      refetchType: 'active',
    },
    {
      queryKey: controlQueryKeys.groups.models(groupID),
      exact: true,
      refetchType: 'active',
    },
    {
      queryKey: controlQueryKeys.groups.credentialsAll(groupID),
      refetchType: 'active',
    },
    {
      queryKey: controlQueryKeys.groups.collectionAll,
      refetchType: 'active',
    },
    {
      queryKey: controlQueryKeys.groups.options(),
      exact: true,
      refetchType: 'active',
    },
    {
      queryKey: controlQueryKeys.home.base(),
      exact: true,
      refetchType: 'active',
    },
    {
      queryKey: controlQueryKeys.modelPrices(),
      refetchType: 'active',
    },
    {
      queryKey: controlQueryKeys.models.all,
      refetchType: 'active',
    },
    {
      queryKey: controlQueryKeys.modelRouteSchedule.all,
      refetchType: 'active',
    },
  ])
}

async function invalidateGroupDependents(
  queryClient: QueryClient,
  queries: readonly Parameters<QueryClient['invalidateQueries']>[0][],
): Promise<void> {
  await Promise.all(queries.map((filters) => queryClient.invalidateQueries(filters)))
}

export async function cacheGroupModels(
  queryClient: QueryClient,
  groupID: number,
  models: GroupModelsDto,
): Promise<void> {
  await queryClient.cancelQueries({
    queryKey: controlQueryKeys.groups.models(groupID),
    exact: true,
  })
  queryClient.setQueryData(controlQueryKeys.groups.models(groupID), models)
  queryClient.setQueryData<GroupSummaryDto>(controlQueryKeys.groups.summary(groupID), (summary) =>
    summary === undefined ? summary : { ...summary, model_count: models.total },
  )
  const clientModels = models.items.map(({ client_model: clientModel }) => clientModel)
  queryClient.setQueryData<GroupOptionDto[]>(controlQueryKeys.groups.options(), (options) =>
    options?.map((option) =>
      option.id === groupID ? { ...option, models: clientModels } : option,
    ),
  )
}

/** Remove every cache entry whose identity is the deleted Group, including filtered key pages. */
export function clearGroupResourceCaches(queryClient: QueryClient, groupID: number): void {
  queryClient.removeQueries({ queryKey: controlQueryKeys.groups.summary(groupID), exact: true })
  queryClient.removeQueries({ queryKey: controlQueryKeys.groups.models(groupID), exact: true })
  queryClient.removeQueries({ queryKey: controlQueryKeys.groups.settings(groupID), exact: true })
  queryClient.removeQueries({
    queryKey: controlQueryKeys.groups.credentialsAll(groupID),
    exact: false,
  })
}

export async function createGroup(
  client: ApiClient,
  body: GroupCreateRequest,
  idempotencyKey: string,
  signal?: AbortSignal,
): Promise<GroupCreateResult> {
  return projectGroupCreateResult(
    await client.request('/api/groups', {
      method: 'POST',
      headers: { 'Idempotency-Key': idempotencyKey },
      json: body,
      signal,
    }),
  )
}

export async function copyGroup(
  client: ApiClient,
  groupID: number,
  idempotencyKey: string,
  signal?: AbortSignal,
): Promise<GroupCopyResult> {
  return projectGroupCopyResult(
    await client.request(`/api/groups/${groupID}/copy`, {
      method: 'POST',
      headers: { 'Idempotency-Key': idempotencyKey },
      json: {},
      signal,
    }),
  )
}

export async function importGroupCredentials(
  client: ApiClient,
  groupID: number,
  body: CredentialImportRequest,
  idempotencyKey: string,
  signal?: AbortSignal,
): Promise<CredentialImportResult> {
  return projectCredentialImportResult(
    await client.request(`/api/groups/${groupID}/credential`, {
      method: 'POST',
      headers: { 'Idempotency-Key': idempotencyKey },
      json: body,
      signal,
    }),
  )
}
