import { type QueryClient, type QueryFunctionContext } from '@tanstack/query-core'

import type { ApiClient } from '@shared/http/client'
import { enabledDataProtocols } from '@shared/control/protocols'
import type {
  CredentialDetailDto,
  CredentialItemDto,
  CredentialRecoveryDto,
  CredentialRevealDto,
  CredentialTestResultDto,
} from '@shared/control/types'
import { InvalidResponseError } from '@shared/http/errors'
import { controlQueryKeys } from '@shared/control/query-keys'

import {
  assertNoSecretLikeFields,
  projectBoolean,
  projectEnum,
  projectEpochMilliseconds,
  projectNullableEpochMilliseconds,
  projectNullableRequestID,
  projectRecord,
  projectSafeInteger,
  projectString,
} from './projector'

export type {
  CredentialDetailDto,
  CredentialItemDto,
  CredentialRecoveryDto,
  CredentialRevealDto,
  CredentialStatus,
  CredentialTestOutcome,
  CredentialTestReason,
  CredentialTestResultDto,
} from '@shared/control/types'

export interface CredentialPatch {
  credential?: string
}
const credentialItemFields = [
  'credential_id',
  'connection_type',
  'secret_version',
  'mask',
  'effective_status',
  'recent_success_count',
  'recent_failure_count',
  'consecutive_failure_count',
  'last_failure_category',
  'last_failure_summary',
  'last_status_code',
  'cooldown_until_ms',
  'recovery',
] as const
const credentialDetailFields = ['credential'] as const
const credentialRecoveryFields = ['mode', 'automatic', 'at_ms'] as const
const credentialTestResultFields = [
  'outcome',
  'model',
  'protocol',
  'latency_ms',
  'reason',
  'recovered',
  'log_id',
  'tested_at_ms',
] as const
const credentialTestOutcomes = ['passed', 'failed', 'inconclusive'] as const
const failedCredentialTestReasons = [
  'invalid_credential',
  'insufficient_balance',
  'model_unavailable',
  'no_answer',
  'invalid_response',
] as const
const inconclusiveCredentialTestReasons = [
  'rate_limited',
  'timeout',
  'upstream_error',
  'probe_incompatible',
  'unknown',
] as const
const effectiveStatuses = ['available', 'cooldown', 'blacklisted', 'disabled'] as const
const recoveryModes = ['none', 'cooldown', 'scheduled_release', 'manual'] as const
const failureCategories = [
  'ok',
  'rate_limited',
  'model_unavailable',
  'invalid_key',
  'billing',
  'upstream_host_error',
  'client_error',
  'downstream_cancel',
  'authentication_required',
  'ambiguous',
] as const
const connectionTypes = ['api_key'] as const

function invalidResponse(): never {
  throw new InvalidResponseError()
}

function projectMask(value: unknown): string {
  return projectString(value)
}

function projectRecovery(value: unknown): CredentialRecoveryDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, credentialRecoveryFields)
  const result = {
    mode: projectEnum(record.mode, recoveryModes),
    automatic: projectBoolean(record.automatic),
    at_ms: projectNullableEpochMilliseconds(record.at_ms),
  }
  if (
    (result.mode === 'cooldown' && (!result.automatic || result.at_ms === null)) ||
    (result.mode === 'scheduled_release' && !result.automatic) ||
    (result.mode === 'manual' && result.automatic) ||
    (result.mode === 'none' && (result.automatic || result.at_ms !== null))
  ) {
    invalidResponse()
  }
  return result
}

export function projectCredentialItem(value: unknown, expectedId?: number): CredentialItemDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, credentialItemFields)
  const credentialId = projectSafeInteger(record.credential_id, { minimum: 1 })
  if (expectedId !== undefined && credentialId !== expectedId) invalidResponse()
  const connectionType = projectEnum(record.connection_type, connectionTypes)
  const effectiveStatus = projectEnum(record.effective_status, effectiveStatuses)
  const cooldownUntil = projectNullableEpochMilliseconds(record.cooldown_until_ms)
  const recovery = projectRecovery(record.recovery)
  if (
    (effectiveStatus === 'cooldown') !== (cooldownUntil !== null) ||
    (recovery.mode === 'cooldown') !== (effectiveStatus === 'cooldown') ||
    (recovery.mode === 'scheduled_release') !== (effectiveStatus === 'blacklisted')
  ) {
    invalidResponse()
  }
  return {
    credential_id: credentialId,
    connection_type: connectionType,
    secret_version: projectSafeInteger(record.secret_version, { minimum: 1 }),
    mask: projectMask(record.mask),
    effective_status: effectiveStatus,
    recent_success_count: projectSafeInteger(record.recent_success_count, { minimum: 0 }),
    recent_failure_count: projectSafeInteger(record.recent_failure_count, { minimum: 0 }),
    consecutive_failure_count: projectSafeInteger(record.consecutive_failure_count, { minimum: 0 }),
    last_failure_category: projectEnum(record.last_failure_category, failureCategories),
    ...(record.last_failure_summary === undefined
      ? {}
      : { last_failure_summary: projectString(record.last_failure_summary) }),
    last_status_code:
      record.last_status_code === null
        ? null
        : projectSafeInteger(record.last_status_code, { minimum: 100, maximum: 999 }),
    cooldown_until_ms: cooldownUntil,
    recovery,
  }
}

export function projectCredentialDetail(value: unknown): CredentialDetailDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, credentialDetailFields)
  return {
    credential: record.credential === null ? null : projectCredentialItem(record.credential),
  }
}

function normalizePatch(patch: CredentialPatch): CredentialPatch {
  if (
    Object.keys(patch).length !== 1 ||
    typeof patch.credential !== 'string' ||
    patch.credential.trim() === ''
  ) {
    throw new Error('INVALID_CREDENTIAL_PATCH')
  }
  return { credential: patch.credential }
}

export async function getCredentialDetail(
  client: ApiClient,
  groupId: number,
  signal?: AbortSignal,
): Promise<CredentialDetailDto> {
  const result = projectCredentialDetail(
    await client.request(`/api/groups/${groupId}/credential`, {
      method: 'GET',
      signal,
    }),
  )
  return result
}

export const manualGroupQueryOptions = {
  refetchOnWindowFocus: false,
  refetchOnReconnect: false,
} as const

export function credentialQueryOptions(client: ApiClient, groupID: number) {
  const key = controlQueryKeys.groups.credentialsAll(groupID)
  return {
    ...manualGroupQueryOptions,
    queryKey: key,
    queryFn: ({ queryKey, signal }: QueryFunctionContext<typeof key>) =>
      getCredentialDetail(client, queryKey[3], signal),
  }
}

export async function updateCredential(
  client: ApiClient,
  groupId: number,
  credentialId: number,
  patch: CredentialPatch,
  signal?: AbortSignal,
): Promise<CredentialItemDto> {
  return projectCredentialItem(
    await client.request(`/api/groups/${groupId}/credential`, {
      method: 'PUT',
      headers: { 'X-Credential-ID': String(credentialId) },
      json: normalizePatch(patch),
      signal,
    }),
    credentialId,
  )
}

export async function cacheCredentialItem(
  queryClient: QueryClient,
  groupId: number,
  item: CredentialItemDto,
): Promise<void> {
  queryClient.setQueryData<CredentialDetailDto>(controlQueryKeys.groups.credentialsAll(groupId), {
    credential: item,
  })
}

export async function deleteCredential(
  client: ApiClient,
  groupId: number,
  credentialId: number,
  signal?: AbortSignal,
): Promise<void> {
  await client.request(`/api/groups/${groupId}/credential`, {
    method: 'DELETE',
    headers: { 'X-Credential-ID': String(credentialId) },
    signal,
  })
}

export async function revealCredential(
  client: ApiClient,
  groupId: number,
  credentialId: number,
  signal?: AbortSignal,
): Promise<CredentialRevealDto> {
  const record = projectRecord(
    await client.request(`/api/groups/${groupId}/credential/reveal`, {
      method: 'POST',
      headers: { 'X-Credential-ID': String(credentialId) },
      signal,
    }),
  )
  assertNoSecretLikeFields(record, ['credential_id', 'credential', 'revealed_at_ms'])
  if (projectSafeInteger(record.credential_id, { minimum: 1 }) !== credentialId) invalidResponse()
  const credentialRecord = projectRecord(record.credential)
  const credential: Record<string, string> = {}
  for (const [key, value] of Object.entries(credentialRecord)) {
    if (key !== key.trim() || !/^[a-z][a-z0-9_]*$/u.test(key)) invalidResponse()
    credential[key] = projectString(value)
  }
  if (Object.keys(credential).length === 0) invalidResponse()
  return {
    credential_id: credentialId,
    credential,
    revealed_at_ms: projectEpochMilliseconds(record.revealed_at_ms),
  }
}

export async function restoreCredential(
  client: ApiClient,
  groupId: number,
  credentialId: number,
  signal?: AbortSignal,
): Promise<CredentialItemDto> {
  return projectCredentialItem(
    await client.request(`/api/groups/${groupId}/credential/restore`, {
      method: 'POST',
      headers: { 'X-Credential-ID': String(credentialId) },
      json: {},
      signal,
    }),
    credentialId,
  )
}

export function projectCredentialTestResult(value: unknown): CredentialTestResultDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, credentialTestResultFields)
  const outcome = projectEnum(record.outcome, credentialTestOutcomes)
  const reason =
    record.reason === null
      ? null
      : outcome === 'failed'
        ? projectEnum(record.reason, failedCredentialTestReasons)
        : outcome === 'inconclusive'
          ? projectEnum(record.reason, inconclusiveCredentialTestReasons)
          : invalidResponse()
  const recovered = projectBoolean(record.recovered)
  if ((outcome === 'passed') !== (reason === null) || (recovered && outcome !== 'passed')) {
    invalidResponse()
  }
  return {
    outcome,
    model: projectString(record.model),
    protocol: projectEnum(record.protocol, enabledDataProtocols),
    latency_ms: projectSafeInteger(record.latency_ms, { minimum: 0 }),
    reason,
    recovered,
    log_id: projectNullableRequestID(record.log_id),
    tested_at_ms: projectEpochMilliseconds(record.tested_at_ms),
  }
}

export async function testCredentialConnection(
  client: ApiClient,
  groupId: number,
  credentialId: number,
  signal?: AbortSignal,
): Promise<CredentialTestResultDto> {
  return projectCredentialTestResult(
    await client.request(`/api/groups/${groupId}/credential/test`, {
      method: 'POST',
      headers: { 'X-Credential-ID': String(credentialId) },
      json: {},
      signal,
    }),
  )
}
