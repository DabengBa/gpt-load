import type { ApiClient } from '@/api/client'
import { enabledDataProtocols } from '@/api/control/protocols'
import {
  routeStrategies,
  type AccessKeyDto,
  type AccessProtocol,
  type RouteStrategy,
} from '@/api/control/types'
import { InvalidResponseError } from '@/api/errors'

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

export type RouteInspectReasonCode =
  | 'access_key_disabled'
  | 'access_key_expired'
  | 'protocol_filtered'
  | 'model_filtered'
  | 'model_required_by_filter'
  | 'operation_unsupported'
  | 'native_route_required'
  | 'no_route_target'
  | 'group_disabled'
  | 'group_filtered'
  | 'no_available_group'
  | 'no_credentials'
  | 'credential_blacklisted'
  | 'credential_cooldown'
  | 'credential_auth_unavailable'
  | 'credential_not_allowed'
  | 'no_available_credential'
  | 'entry_blacklisted'
  | 'entry_cooldown'
  | 'entry_weight_zero'
  | 'tier_demoted'

export interface RouteInspectRequest {
  protocol: AccessProtocol
  external_model: string
  access_key_id: number
}

export type RouteInspectOperation =
  | 'chat_completion'
  | 'responses_create'
  | 'responses_retrieve'
  | 'responses_delete'
  | 'responses_cancel'
  | 'responses_input_items'
  | 'responses_compact'
  | 'responses_input_tokens'
  | 'count_tokens'
  | 'responses_passthrough'
  | 'images_generate'
  | 'images_edit'
  | 'embeddings_create'
  | 'rerank'
export type RouteInspectRequirement = 'any' | 'native'
export type RouteInspectMode = 'native' | 'converted'

export interface RouteInspectCredentialDto {
  credential_id: number
  available: boolean
  reason_code: RouteInspectReasonCode | null
  cooldown_until_ms: number | null
}

export interface RouteInspectGroupDto {
  group_id: number
  group_name: string
  channel_id: string
  route_mode: RouteInspectMode
  route_requirement_satisfied: boolean
  entry_id: string
  upstream_model: string | null
  entry_weight: number
  priority: number
  fallback: boolean
  configured_share: number
  effective_share: number
  entry_cooldown_until_ms: number | null
  included: boolean
  routable: boolean
  reason_code: RouteInspectReasonCode | null
  credentials: RouteInspectCredentialDto[]
}

export interface RouteInspectResponseDto {
  observed_at_ms: number
  snapshot_revision: number
  route_strategy: RouteStrategy
  protocol: AccessProtocol
  operation: RouteInspectOperation
  route_requirement: RouteInspectRequirement
  external_model: string | null
  access_key: {
    id: number
    name: string
    status: AccessKeyDto['status']
  }
  routable: boolean
  reason_code: RouteInspectReasonCode | null
  groups: RouteInspectGroupDto[]
}

const accessKeyStatuses = ['active', 'disabled'] as const
export const routeInspectOperations = [
  'chat_completion',
  'responses_create',
  'responses_retrieve',
  'responses_delete',
  'responses_cancel',
  'responses_input_items',
  'responses_compact',
  'responses_input_tokens',
  'count_tokens',
  'responses_passthrough',
  'images_generate',
  'images_edit',
  'embeddings_create',
  'rerank',
] as const
export const routeInspectRequirements = ['any', 'native'] as const
const routeModes = ['native', 'converted'] as const
export const routeInspectReasonCodes = [
  'access_key_disabled',
  'access_key_expired',
  'protocol_filtered',
  'model_filtered',
  'model_required_by_filter',
  'operation_unsupported',
  'native_route_required',
  'no_route_target',
  'group_disabled',
  'group_filtered',
  'no_available_group',
  'no_credentials',
  'credential_blacklisted',

  'credential_cooldown',
  'credential_auth_unavailable',
  'credential_not_allowed',
  'no_available_credential',
  'entry_blacklisted',
  'entry_cooldown',
  'entry_weight_zero',
  'tier_demoted',
] as const

function invalidResponse(): never {
  throw new InvalidResponseError()
}

function projectNonBlankString(value: unknown): string {
  const result = projectString(value)
  if (result.trim().length === 0 || result !== result.trim()) invalidResponse()
  return result
}

function projectNullableNonBlankString(value: unknown): string | null {
  return value === null ? null : projectNonBlankString(value)
}

function projectReason(value: unknown): RouteInspectReasonCode | null {
  return value === null ? null : projectEnum(value, routeInspectReasonCodes)
}

function projectRouteCredential(value: unknown): RouteInspectCredentialDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, [
    'credential_id',
    'available',
    'reason_code',
    'cooldown_until_ms',
  ])
  return {
    credential_id: projectSafeInteger(record.credential_id, { minimum: 1 }),
    available: projectBoolean(record.available),
    reason_code: projectReason(record.reason_code),
    cooldown_until_ms: projectNullableEpochMilliseconds(record.cooldown_until_ms),
  }
}

function projectRouteGroup(value: unknown): RouteInspectGroupDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, [
    'group_id',
    'group_name',
    'channel_id',
    'route_mode',
    'route_requirement_satisfied',
    'entry_id',
    'upstream_model',
    'entry_weight',
    'priority',
    'fallback',
    'configured_share',
    'effective_share',
    'entry_cooldown_until_ms',
    'included',
    'routable',
    'reason_code',
    'credentials',
  ])
  const result = {
    group_id: projectSafeInteger(record.group_id, { minimum: 1 }),
    group_name: projectNonBlankString(record.group_name),
    channel_id: projectNonBlankString(record.channel_id),
    route_mode: projectEnum(record.route_mode, routeModes),
    route_requirement_satisfied: projectBoolean(record.route_requirement_satisfied),
    entry_id: projectNonBlankString(record.entry_id),
    upstream_model: projectNullableNonBlankString(record.upstream_model),
    entry_weight: projectSafeInteger(record.entry_weight, { minimum: 0 }),
    priority: projectSafeInteger(record.priority, { minimum: 1 }),
    fallback: projectBoolean(record.fallback),
    configured_share: projectFiniteNumber(record.configured_share, { minimum: 0, maximum: 1 }),
    effective_share: projectFiniteNumber(record.effective_share, { minimum: 0, maximum: 1 }),
    entry_cooldown_until_ms: projectNullableEpochMilliseconds(record.entry_cooldown_until_ms),
    included: projectBoolean(record.included),
    routable: projectBoolean(record.routable),
    reason_code: projectReason(record.reason_code),
    credentials: projectArray(record.credentials, projectRouteCredential),
  }
  if (result.fallback !== result.priority > 1) invalidResponse()
  return result
}

function projectAccessKey(value: unknown): RouteInspectResponseDto['access_key'] {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, ['id', 'name', 'status'])
  return {
    id: projectSafeInteger(record.id, { minimum: 1 }),
    name: projectNonBlankString(record.name),
    status: projectEnum(record.status, accessKeyStatuses),
  }
}

export function projectRouteInspection(value: unknown): RouteInspectResponseDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, [
    'observed_at_ms',
    'snapshot_revision',
    'route_strategy',
    'protocol',
    'operation',
    'route_requirement',
    'external_model',
    'access_key',
    'routable',
    'reason_code',
    'groups',
  ])
  const observedAtMS = projectEpochMilliseconds(record.observed_at_ms)
  const groups = projectArray(record.groups, projectRouteGroup)
  const configuredSharesByPriority = new Map<number, number>()
  for (const group of groups) {
    configuredSharesByPriority.set(
      group.priority,
      (configuredSharesByPriority.get(group.priority) ?? 0) + group.configured_share,
    )
  }
  for (const total of configuredSharesByPriority.values()) {
    if (total > 0 && Math.abs(total - 1) > 1e-9) invalidResponse()
  }
  if (
    groups.some(
      ({ entry_cooldown_until_ms: cooldownUntilMS }) =>
        cooldownUntilMS !== null && cooldownUntilMS <= observedAtMS,
    )
  ) {
    invalidResponse()
  }
  return {
    observed_at_ms: observedAtMS,
    snapshot_revision: projectSafeInteger(record.snapshot_revision, { minimum: 1 }),
    route_strategy: projectEnum(record.route_strategy, routeStrategies),
    protocol: projectEnum(record.protocol, enabledDataProtocols),
    operation: projectEnum(record.operation, routeInspectOperations),
    route_requirement: projectEnum(record.route_requirement, routeInspectRequirements),
    external_model: projectNullableNonBlankString(record.external_model),
    access_key: projectAccessKey(record.access_key),
    routable: projectBoolean(record.routable),
    reason_code: projectReason(record.reason_code),
    groups,
  }
}

export async function inspectRoute(
  client: ApiClient,
  body: RouteInspectRequest,
  signal?: AbortSignal,
): Promise<RouteInspectResponseDto> {
  return projectRouteInspection(
    await client.request('/api/route/inspect', {
      method: 'POST',
      json: body,
      signal,
    }),
  )
}
