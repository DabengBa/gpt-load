import { queryOptions } from '@tanstack/vue-query'
import { computed, toValue, type MaybeRefOrGetter } from 'vue'

import type { ApiClient } from '@/api/client'
import { enabledDataProtocols } from '@/api/control/protocols'
import { type AccessKeyDto, type AccessProtocol } from '@/api/control/types'
import { ApiError, InvalidResponseError } from '@/api/errors'
import { controlQueryKeys } from '@/app/query-keys'
import {
  routeInspectOperations,
  routeInspectReasonCodes,
  routeInspectRequirements,
  type RouteInspectCredentialDto,
  type RouteInspectOperation,
  type RouteInspectReasonCode,
  type RouteInspectRequirement,
} from '@/app/resources/route-inspection'

import {
  assertNoSecretLikeFields,
  projectArray,
  projectBoolean,
  projectEpochMilliseconds,
  projectEnum,
  projectFiniteNumber,
  projectNullableEpochMilliseconds,
  projectRecord,
  projectSafeInteger,
  projectString,
} from './projector'

export interface ModelRouteScheduleIndexItemDto {
  external_model: string
  protocol: AccessProtocol
  operation: RouteInspectOperation
  candidate_count: number
  group_count: number
  has_fallback: boolean
  cooled_candidates: number
  blacklisted_candidates: number
}

export interface ModelRouteScheduleIndexDto {
  items: ModelRouteScheduleIndexItemDto[]
}

export type ModelRouteScheduleRuntimeState = 'available' | 'blacklisted' | 'cooldown'
export type ModelRouteScheduleBreakerSource = 'default' | 'entry'

export interface ModelRouteScheduleBreakerParametersDto {
  blacklist_threshold: number | null
  cooldown_seconds: number | null
}

export interface ModelRouteScheduleBreakerSourcesDto {
  blacklist_threshold: ModelRouteScheduleBreakerSource
  cooldown_seconds: ModelRouteScheduleBreakerSource
}

export interface ModelRouteScheduleBreakerDto {
  configured: ModelRouteScheduleBreakerParametersDto
  effective: ModelRouteScheduleBreakerParametersDto
  sources: ModelRouteScheduleBreakerSourcesDto
}

export interface ModelRouteScheduleRuntimeDto {
  state: ModelRouteScheduleRuntimeState
  cooldown_until_ms: number | null
  failure_count: number
}

export interface ModelRouteScheduleEntryDto {
  entry_id: string
  model_id: string
  alias: string
  weight: number
  priority: number
  fallback: boolean
  circuit_breaker: ModelRouteScheduleBreakerDto
  runtime: ModelRouteScheduleRuntimeDto
  included: boolean
  routable: boolean
  reason_code: RouteInspectReasonCode | null
  effective_share: number
  credentials: RouteInspectCredentialDto[]
}

export interface ModelRouteScheduleGroupDto {
  group_id: number
  group_name: string
  channel_id: string
  entries: ModelRouteScheduleEntryDto[]
}

export interface ModelRouteScheduleDetailRequest {
  protocol: AccessProtocol
  external_model: string
  access_key_id: number
  operation?: RouteInspectOperation
}

export interface ModelRouteScheduleDetailDto {
  observed_at_ms: number
  snapshot_revision: number
  external_model: string | null
  protocol: AccessProtocol
  operation: RouteInspectOperation
  route_requirement: RouteInspectRequirement
  access_key: {
    id: number
    name: string
    status: AccessKeyDto['status']
  }
  routable: boolean
  reason_code: RouteInspectReasonCode | null
  groups: ModelRouteScheduleGroupDto[]
}

export interface ModelRouteScheduleBreakerPatch {
  blacklist_threshold?: number | null
  cooldown_seconds?: number | null
}

export interface ModelRouteSchedulePatchUpdate {
  group_id: number
  entry_id: string
  weight?: number | null
  priority?: number | null
  circuit_breaker?: ModelRouteScheduleBreakerPatch | null
}

export interface ModelRouteSchedulePatchRequest {
  snapshot_revision: number
  protocol?: AccessProtocol
  external_model?: string
  access_key_id?: number
  operation?: RouteInspectOperation
  updates: ModelRouteSchedulePatchUpdate[]
}

export interface ModelRouteSchedulePatchResponse {
  snapshot_revision_new: number
  detail: ModelRouteScheduleDetailDto | null
}

export interface ModelRouteScheduleRecoverRequest {
  group_id: number
  entry_id: string
}

export interface ModelRouteScheduleRecoverResponse {
  group_id: number
  entry_id: string
  runtime: ModelRouteScheduleRuntimeDto
}

export const modelRouteScheduleRevisionConflictCode = 'MODEL_ROUTE_SCHEDULE_REVISION_CONFLICT'

const indexFields = [
  'external_model',
  'protocol',
  'operation',
  'candidate_count',
  'group_count',
  'has_fallback',
  'cooled_candidates',
  'blacklisted_candidates',
] as const
const detailFields = [
  'observed_at_ms',
  'snapshot_revision',
  'external_model',
  'protocol',
  'operation',
  'route_requirement',
  'access_key',
  'routable',
  'reason_code',
  'groups',
] as const
const groupFields = ['group_id', 'group_name', 'channel_id', 'entries'] as const
const entryFields = [
  'entry_id',
  'model_id',
  'alias',
  'weight',
  'priority',
  'fallback',
  'circuit_breaker',
  'runtime',
  'included',
  'routable',
  'reason_code',
  'effective_share',
  'credentials',
] as const
const credentialFields = ['credential_id', 'available', 'reason_code', 'cooldown_until_ms'] as const
const breakerFields = ['configured', 'effective', 'sources'] as const
const breakerParameterFields = ['blacklist_threshold', 'cooldown_seconds'] as const
const breakerSourceFields = ['blacklist_threshold', 'cooldown_seconds'] as const
const runtimeFields = ['state', 'cooldown_until_ms', 'failure_count'] as const
const accessKeyFields = ['id', 'name', 'status'] as const
const accessKeyStatuses = ['active', 'disabled'] as const
const runtimeStates = ['available', 'blacklisted', 'cooldown'] as const
const breakerSources = ['default', 'entry'] as const

function invalidResponse(): never {
  throw new InvalidResponseError()
}

function projectNonBlankString(value: unknown): string {
  const result = projectString(value)
  if (result !== result.trim()) invalidResponse()
  return result
}

function projectNullableNonBlankString(value: unknown): string | null {
  return value === null ? null : projectNonBlankString(value)
}

function projectReason(value: unknown): RouteInspectReasonCode | null {
  return value === null ? null : projectEnum(value, routeInspectReasonCodes)
}

function projectCredential(value: unknown): RouteInspectCredentialDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, credentialFields)
  return {
    credential_id: projectSafeInteger(record.credential_id, { minimum: 1 }),
    available: projectBoolean(record.available),
    reason_code: projectReason(record.reason_code),
    cooldown_until_ms: projectNullableEpochMilliseconds(record.cooldown_until_ms),
  }
}

function projectBreakerParameters(value: unknown): ModelRouteScheduleBreakerParametersDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, breakerParameterFields)
  return {
    blacklist_threshold:
      record.blacklist_threshold === null
        ? null
        : projectSafeInteger(record.blacklist_threshold, { minimum: 1 }),
    cooldown_seconds:
      record.cooldown_seconds === null
        ? null
        : projectSafeInteger(record.cooldown_seconds, { minimum: 0 }),
  }
}

function projectBreaker(value: unknown): ModelRouteScheduleBreakerDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, breakerFields)
  const configured = projectBreakerParameters(record.configured)
  const effective = projectBreakerParameters(record.effective)
  const sourcesRecord = projectRecord(record.sources)
  assertNoSecretLikeFields(sourcesRecord, breakerSourceFields)
  const sources = {
    blacklist_threshold: projectEnum(sourcesRecord.blacklist_threshold, breakerSources),
    cooldown_seconds: projectEnum(sourcesRecord.cooldown_seconds, breakerSources),
  }

  for (const parameter of ['blacklist_threshold', 'cooldown_seconds'] as const) {
    if (
      (sources[parameter] === 'entry' &&
        (configured[parameter] === null || configured[parameter] !== effective[parameter])) ||
      (sources[parameter] === 'default' && configured[parameter] !== null)
    ) {
      invalidResponse()
    }
  }
  return { configured, effective, sources }
}

function projectRuntime(value: unknown, observedAtMS?: number): ModelRouteScheduleRuntimeDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, runtimeFields)
  const cooldownUntilMS = projectNullableEpochMilliseconds(record.cooldown_until_ms)
  if (observedAtMS !== undefined && cooldownUntilMS !== null && cooldownUntilMS <= observedAtMS) {
    invalidResponse()
  }
  return {
    state: projectEnum(record.state, runtimeStates),
    cooldown_until_ms: cooldownUntilMS,
    failure_count: projectSafeInteger(record.failure_count, { minimum: 0 }),
  }
}

function projectEntry(value: unknown, observedAtMS: number): ModelRouteScheduleEntryDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, entryFields)
  const priority = projectSafeInteger(record.priority, { minimum: 1 })
  const fallback = projectBoolean(record.fallback)
  if (fallback !== priority > 1) invalidResponse()
  return {
    entry_id: projectNonBlankString(record.entry_id),
    model_id: projectNonBlankString(record.model_id),
    alias: projectString(record.alias, { allowEmpty: true }),
    weight: projectSafeInteger(record.weight, { minimum: 0, maximum: 100 }),
    priority,
    fallback,
    circuit_breaker: projectBreaker(record.circuit_breaker),
    runtime: projectRuntime(record.runtime, observedAtMS),
    included: projectBoolean(record.included),
    routable: projectBoolean(record.routable),
    reason_code: projectReason(record.reason_code),
    effective_share: projectFiniteNumber(record.effective_share, { minimum: 0, maximum: 1 }),
    credentials: projectArray(record.credentials, projectCredential),
  }
}

function projectGroup(value: unknown, observedAtMS: number): ModelRouteScheduleGroupDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, groupFields)
  return {
    group_id: projectSafeInteger(record.group_id, { minimum: 1 }),
    group_name: projectNonBlankString(record.group_name),
    channel_id: projectNonBlankString(record.channel_id),
    entries: projectArray(record.entries, (entry) => projectEntry(entry, observedAtMS)),
  }
}

function projectAccessKey(value: unknown): ModelRouteScheduleDetailDto['access_key'] {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, accessKeyFields)
  return {
    id: projectSafeInteger(record.id, { minimum: 1 }),
    name: projectNonBlankString(record.name),
    status: projectEnum(record.status, accessKeyStatuses),
  }
}

export function projectModelRouteScheduleIndex(value: unknown): ModelRouteScheduleIndexDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, ['items'])
  const items = projectArray(record.items, (item) => {
    const itemRecord = projectRecord(item)
    assertNoSecretLikeFields(itemRecord, indexFields)
    const result = {
      external_model: projectNonBlankString(itemRecord.external_model),
      protocol: projectEnum(itemRecord.protocol, enabledDataProtocols),
      operation: projectEnum(itemRecord.operation, routeInspectOperations),
      candidate_count: projectSafeInteger(itemRecord.candidate_count, { minimum: 0 }),
      group_count: projectSafeInteger(itemRecord.group_count, { minimum: 0 }),
      has_fallback: projectBoolean(itemRecord.has_fallback),
      cooled_candidates: projectSafeInteger(itemRecord.cooled_candidates, { minimum: 0 }),
      blacklisted_candidates: projectSafeInteger(itemRecord.blacklisted_candidates, { minimum: 0 }),
    }
    if (
      result.group_count > result.candidate_count ||
      result.cooled_candidates + result.blacklisted_candidates > result.candidate_count
    ) {
      invalidResponse()
    }
    return result
  })
  if (
    new Set(items.map(({ external_model: externalModel }) => externalModel)).size !== items.length
  ) {
    invalidResponse()
  }
  return { items }
}

export function projectModelRouteScheduleDetail(value: unknown): ModelRouteScheduleDetailDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, detailFields)
  const observedAtMS = projectEpochMilliseconds(record.observed_at_ms)
  const groups = projectArray(record.groups, (group) => projectGroup(group, observedAtMS))
  const entryKeys = groups.flatMap(({ group_id: groupID, entries }) =>
    entries.map(({ entry_id: entryID }) => `${groupID}\u0000${entryID}`),
  )
  if (new Set(entryKeys).size !== entryKeys.length) invalidResponse()
  return {
    observed_at_ms: observedAtMS,
    snapshot_revision: projectSafeInteger(record.snapshot_revision, { minimum: 1 }),
    external_model: projectNullableNonBlankString(record.external_model),
    protocol: projectEnum(record.protocol, enabledDataProtocols),
    operation: projectEnum(record.operation, routeInspectOperations),
    route_requirement: projectEnum(record.route_requirement, routeInspectRequirements),
    access_key: projectAccessKey(record.access_key),
    routable: projectBoolean(record.routable),
    reason_code: projectReason(record.reason_code),
    groups,
  }
}

function projectPatchResponse(value: unknown): ModelRouteSchedulePatchResponse {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, ['snapshot_revision_new', 'detail'])
  const snapshotRevisionNew = projectSafeInteger(record.snapshot_revision_new, { minimum: 1 })
  const detail = record.detail === null ? null : projectModelRouteScheduleDetail(record.detail)
  if (detail !== null && detail.snapshot_revision !== snapshotRevisionNew) invalidResponse()
  return { snapshot_revision_new: snapshotRevisionNew, detail }
}

function projectRecoverResponse(value: unknown): ModelRouteScheduleRecoverResponse {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, ['group_id', 'entry_id', 'runtime'])
  const runtime = projectRuntime(record.runtime)
  if (
    runtime.state !== 'available' ||
    runtime.cooldown_until_ms !== null ||
    runtime.failure_count !== 0
  ) {
    invalidResponse()
  }
  return {
    group_id: projectSafeInteger(record.group_id, { minimum: 1 }),
    entry_id: projectNonBlankString(record.entry_id),
    runtime,
  }
}

function detailPath(request: ModelRouteScheduleDetailRequest): `/api/${string}` {
  const params = new URLSearchParams({
    protocol: request.protocol,
    external_model: request.external_model,
    access_key_id: String(request.access_key_id),
  })
  if (request.operation !== undefined) params.set('operation', request.operation)
  return `/api/model-route/schedule/detail?${params.toString()}`
}

export async function getModelRouteScheduleIndex(
  client: ApiClient,
  signal?: AbortSignal,
): Promise<ModelRouteScheduleIndexDto> {
  return projectModelRouteScheduleIndex(
    await client.request('/api/model-route/schedule', { method: 'GET', signal }),
  )
}

export async function getModelRouteScheduleDetail(
  client: ApiClient,
  request: ModelRouteScheduleDetailRequest,
  signal?: AbortSignal,
): Promise<ModelRouteScheduleDetailDto> {
  return projectModelRouteScheduleDetail(
    await client.request(detailPath(request), { method: 'GET', signal }),
  )
}

export async function updateModelRouteSchedule(
  client: ApiClient,
  body: ModelRouteSchedulePatchRequest,
  signal?: AbortSignal,
): Promise<ModelRouteSchedulePatchResponse> {
  return projectPatchResponse(
    await client.request('/api/model-route/schedule', { method: 'PATCH', json: body, signal }),
  )
}

export async function recoverModelRouteScheduleEntry(
  client: ApiClient,
  body: ModelRouteScheduleRecoverRequest,
  signal?: AbortSignal,
): Promise<ModelRouteScheduleRecoverResponse> {
  return projectRecoverResponse(
    await client.request('/api/model-route/schedule/recover', {
      method: 'POST',
      json: body,
      signal,
    }),
  )
}

export function modelRouteScheduleIndexQueryOptions(client: ApiClient) {
  return queryOptions({
    queryKey: controlQueryKeys.modelRouteSchedule.index(),
    queryFn: ({ signal }) => getModelRouteScheduleIndex(client, signal),
  })
}

export function modelRouteScheduleDetailQueryOptions(
  client: ApiClient,
  request: MaybeRefOrGetter<ModelRouteScheduleDetailRequest | undefined>,
) {
  return queryOptions({
    queryKey: computed(() => {
      const context = toValue(request)
      return context === undefined
        ? controlQueryKeys.modelRouteSchedule.details()
        : controlQueryKeys.modelRouteSchedule.detail(context)
    }),
    queryFn: ({ signal }) => {
      const context = toValue(request)
      if (context === undefined) throw new InvalidResponseError()
      return getModelRouteScheduleDetail(client, context, signal)
    },
    enabled: computed(() => toValue(request) !== undefined),
  })
}

export function isModelRouteScheduleRevisionConflict(error: unknown): error is ApiError {
  return (
    error instanceof ApiError &&
    error.status === 409 &&
    error.code === modelRouteScheduleRevisionConflictCode
  )
}
