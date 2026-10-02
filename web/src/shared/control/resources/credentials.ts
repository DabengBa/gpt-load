import { type QueryClient, type QueryFunctionContext } from '@tanstack/query-core'

import type { ApiClient } from '@shared/http/client'
import { enabledDataProtocols } from '@shared/control/protocols'
import type {
  CredentialDailyUsageDto,
  CredentialDetailDto,
  CredentialDownloadDto,
  CredentialItemDto,
  CredentialObservationDto,
  CredentialObservationSnapshotDto,
  CredentialObservedWindowUsageDto,
  CredentialQuotaWindowDto,
  CredentialResetCreditConsumeDto,
  CredentialRecoveryDto,
  CredentialRevealDto,
  CredentialTestResultDto,
} from '@shared/control/types'
import { InvalidResponseError } from '@shared/http/errors'
import { controlQueryKeys } from '@shared/control/query-keys'

import {
  assertNoSecretLikeFields,
  projectArray,
  projectBoolean,
  projectEnum,
  projectEpochMilliseconds,
  projectFiniteNumber,
  projectNullableEpochMilliseconds,
  projectNullableRequestID,
  projectNonNegativeInt64String,
  projectRecord,
  projectSafeInteger,
  projectString,
} from './projector'

export type {
  CredentialDailyUsageDto,
  CredentialDetailDto,
  CredentialDownloadDto,
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
  'account',
  'effective_status',
  'auth_state',
  'auth_error_code',
  'observation',
  'recent_success_count',
  'recent_failure_count',
  'consecutive_failure_count',
  'last_failure_category',
  'last_status_code',
  'cooldown_until_ms',
  'last_used_at_ms',
  'daily_usage',
  'recovery',
] as const
const credentialDetailFields = ['credential', 'observation'] as const
const credentialDownloadFields = ['filename', 'credential'] as const
const credentialDailyUsageFields = [
  'window_seconds',
  'success_count',
  'failure_count',
  'data_complete',
] as const
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
const connectionTypes = ['api_key', 'subscription'] as const
const authStates = ['ready', 'refreshing', 'reauthorization_required', 'outcome_unknown'] as const
const observationStates = ['fresh', 'stale', 'refreshing', 'error', 'unavailable'] as const
const quotaStates = ['available', 'exhausted', 'unknown'] as const
const planLevels = ['free', 'standard', 'premium', 'elite'] as const
const accountFields = ['email', 'email_mask', 'expires_at_ms', 'last_refresh_at_ms'] as const
const observationFields = [
  'state',
  'snapshot',
  'observation_version',
  'observed_at_ms',
  'last_attempt_at_ms',
  'last_error_code',
] as const
const observationSnapshotFields = [
  'plan_summary',
  'account_summary',
  'quota_windows',
  'reset_credits_available',
  'reset_credits',
] as const
const planFields = ['name', 'level'] as const
const observationAccountFields = [
  'display_name',
  'email',
  'organization_name',
  'organization_type',
  'organization_role',
  'workspace_role',
  'organization_rate_limit_tier',
  'user_rate_limit_tier',
  'seat_tier',
  'billing_type',
  'extra_usage_enabled',
  'extra_usage_disabled_reason',
  'account_created_at_ms',
  'subscription_created_at_ms',
] as const
const resetCreditFields = ['expires_at_ms'] as const
const resetCreditConsumeFields = [
  'status',
  'windows_reset',
  'redeemed_at_ms',
  'observation',
  'observation_pending',
  'replayed',
] as const
const quotaWindowFields = [
  'source_id',
  'id',
  'label',
  'label_key',
  'scope',
  'unit',
  'used',
  'limit',
  'remaining',
  'utilization',
  'reset_at_ms',
  'window_seconds',
  'model_ids',
  'state',
  'is_primary',
  'observed_usage',
] as const
const quotaLabelKeys = [
  'session',
  'weekly',
  'extra_usage',
  'included_usage',
  'pay_as_you_go',
  'oauth_apps',
] as const
const observedWindowUsageFields = [
  'window_start_ms',
  'window_end_ms',
  'source',
  'data_complete',
  'usage_complete',
  'pricing_complete',
  'request_count',
  'input_tokens',
  'output_tokens',
  'total_tokens',
  'estimated_reference_cost_nano_usd',
  'last_used_at_ms',
] as const

function projectObservedWindowUsage(value: unknown): CredentialObservedWindowUsageDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, observedWindowUsageFields)
  const windowStart = projectEpochMilliseconds(record.window_start_ms)
  const windowEnd = projectEpochMilliseconds(record.window_end_ms)
  if (windowEnd <= windowStart) invalidResponse()
  return {
    window_start_ms: windowStart,
    window_end_ms: windowEnd,
    source: projectEnum(record.source, ['request_logs', 'usage_stats'] as const),
    data_complete: projectBoolean(record.data_complete),
    usage_complete: projectBoolean(record.usage_complete),
    pricing_complete: projectBoolean(record.pricing_complete),
    request_count: projectSafeInteger(record.request_count, { minimum: 0 }),
    input_tokens: projectSafeInteger(record.input_tokens, { minimum: 0 }),
    output_tokens: projectSafeInteger(record.output_tokens, { minimum: 0 }),
    total_tokens: projectSafeInteger(record.total_tokens, { minimum: 0 }),
    estimated_reference_cost_nano_usd: projectNonNegativeInt64String(
      record.estimated_reference_cost_nano_usd,
    ),
    ...(record.last_used_at_ms === undefined
      ? {}
      : { last_used_at_ms: projectEpochMilliseconds(record.last_used_at_ms) }),
  }
}

function invalidResponse(): never {
  throw new InvalidResponseError()
}

function projectCredentialJSONValue(value: unknown): unknown {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) invalidResponse()
    return value
  }
  if (Array.isArray(value)) return value.map(projectCredentialJSONValue)
  const record = projectRecord(value)
  const result: Record<string, unknown> = {}
  for (const [key, nested] of Object.entries(record)) {
    if (!/^[a-z][a-z0-9_]*$/u.test(key)) invalidResponse()
    result[key] = projectCredentialJSONValue(nested)
  }
  return result
}

function projectCredentialJSON(value: unknown): Record<string, unknown> {
  const projected = projectCredentialJSONValue(value)
  if (typeof projected !== 'object' || projected === null || Array.isArray(projected)) {
    invalidResponse()
  }
  const result = projected as Record<string, unknown>
  if (Object.keys(result).length === 0) invalidResponse()
  return result
}

function projectMask(value: unknown): string {
  return projectString(value)
}

function projectAccount(
  value: unknown,
  connectionType: 'api_key' | 'subscription',
): CredentialItemDto['account'] {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, accountFields)
  const email = record.email === undefined ? undefined : projectString(record.email)
  const emailMask = record.email_mask === undefined ? undefined : projectString(record.email_mask)
  if (connectionType === 'api_key' && (email !== undefined || emailMask !== undefined)) {
    invalidResponse()
  }
  return {
    ...(email === undefined ? {} : { email }),
    ...(emailMask === undefined ? {} : { email_mask: emailMask }),
    ...(record.expires_at_ms === undefined
      ? {}
      : { expires_at_ms: projectEpochMilliseconds(record.expires_at_ms) }),
    ...(record.last_refresh_at_ms === undefined
      ? {}
      : { last_refresh_at_ms: projectEpochMilliseconds(record.last_refresh_at_ms) }),
  }
}

function projectInternalErrorCode(value: unknown): string {
  const code = projectString(value)
  if (!/^[a-z0-9_]{1,64}$/u.test(code)) invalidResponse()
  return code
}

function projectOptionalNumber(
  record: Record<string, unknown>,
  key: string,
  bounds: { minimum?: number; maximum?: number } = {},
): number | undefined {
  return record[key] === undefined ? undefined : projectFiniteNumber(record[key], bounds)
}

function projectQuotaWindow(value: unknown): CredentialQuotaWindowDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, quotaWindowFields)
  const id = projectString(record.id)
  const label = projectString(record.label)
  const scope = projectString(record.scope)
  const unit = projectString(record.unit)
  if (
    id !== id.trim() ||
    label.trim().length === 0 ||
    scope.trim().length === 0 ||
    unit.trim().length === 0
  ) {
    invalidResponse()
  }
  const modelIDs =
    record.model_ids === undefined
      ? undefined
      : projectArray(record.model_ids, (modelID) => projectString(modelID))
  if (modelIDs && new Set(modelIDs).size !== modelIDs.length) invalidResponse()
  const used = projectOptionalNumber(record, 'used', { minimum: 0 })
  const limit = projectOptionalNumber(record, 'limit', { minimum: 0 })
  const remaining = projectOptionalNumber(record, 'remaining', { minimum: 0 })
  const utilization = projectOptionalNumber(record, 'utilization', { minimum: 0, maximum: 1 })
  return {
    ...(record.source_id === undefined ? {} : { source_id: projectString(record.source_id) }),
    id,
    label,
    ...(record.label_key === undefined
      ? {}
      : { label_key: projectEnum(record.label_key, quotaLabelKeys) }),
    scope,
    unit,
    ...(used === undefined ? {} : { used }),
    ...(limit === undefined ? {} : { limit }),
    ...(remaining === undefined ? {} : { remaining }),
    ...(utilization === undefined ? {} : { utilization }),
    ...(record.reset_at_ms === undefined
      ? {}
      : { reset_at_ms: projectEpochMilliseconds(record.reset_at_ms) }),
    ...(record.window_seconds === undefined
      ? {}
      : {
          window_seconds: projectSafeInteger(record.window_seconds, {
            minimum: 1,
          }),
        }),
    ...(modelIDs === undefined ? {} : { model_ids: modelIDs }),
    state: projectEnum(record.state, quotaStates),
    ...(record.is_primary === undefined ? {} : { is_primary: projectBoolean(record.is_primary) }),
    ...(record.observed_usage === undefined
      ? {}
      : { observed_usage: projectObservedWindowUsage(record.observed_usage) }),
  }
}

function projectObservationSnapshot(value: unknown): CredentialObservationSnapshotDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, observationSnapshotFields)
  const planRecord = projectRecord(record.plan_summary)
  assertNoSecretLikeFields(planRecord, planFields)
  const planName =
    planRecord.name === undefined
      ? undefined
      : projectString(planRecord.name, { allowEmpty: false })
  const planLevel =
    planRecord.level === undefined ? undefined : projectEnum(planRecord.level, planLevels)
  const quotaWindows = projectArray(record.quota_windows, projectQuotaWindow)
  if (
    new Set(quotaWindows.map(({ id }) => id)).size !== quotaWindows.length ||
    quotaWindows.filter(({ is_primary }) => is_primary).length > 1
  ) {
    invalidResponse()
  }
  let accountSummary: CredentialObservationSnapshotDto['account_summary']
  if (record.account_summary !== undefined) {
    const accountRecord = projectRecord(record.account_summary)
    assertNoSecretLikeFields(accountRecord, observationAccountFields)
    accountSummary = {}
    for (const field of [
      'display_name',
      'email',
      'organization_name',
      'organization_type',
      'organization_role',
      'workspace_role',
      'organization_rate_limit_tier',
      'user_rate_limit_tier',
      'seat_tier',
      'billing_type',
      'extra_usage_disabled_reason',
    ] as const) {
      if (accountRecord[field] !== undefined) {
        accountSummary[field] = projectString(accountRecord[field], { allowEmpty: false })
      }
    }
    if (accountRecord.extra_usage_enabled !== undefined) {
      accountSummary.extra_usage_enabled = projectBoolean(accountRecord.extra_usage_enabled)
    }
    if (accountRecord.account_created_at_ms !== undefined) {
      accountSummary.account_created_at_ms = projectEpochMilliseconds(
        accountRecord.account_created_at_ms,
      )
    }
    if (accountRecord.subscription_created_at_ms !== undefined) {
      accountSummary.subscription_created_at_ms = projectEpochMilliseconds(
        accountRecord.subscription_created_at_ms,
      )
    }
  }
  return {
    plan_summary: {
      ...(planName === undefined ? {} : { name: planName }),
      ...(planLevel === undefined ? {} : { level: planLevel }),
    },
    ...(accountSummary === undefined ? {} : { account_summary: accountSummary }),
    quota_windows: quotaWindows,
    ...(record.reset_credits_available === undefined
      ? {}
      : {
          reset_credits_available: projectSafeInteger(record.reset_credits_available, {
            minimum: 0,
          }),
        }),
    ...(record.reset_credits === undefined
      ? {}
      : {
          reset_credits: projectArray(record.reset_credits, (value) => {
            const credit = projectRecord(value)
            assertNoSecretLikeFields(credit, resetCreditFields)
            return credit.expires_at_ms === undefined || credit.expires_at_ms === null
              ? {}
              : { expires_at_ms: projectEpochMilliseconds(credit.expires_at_ms) }
          }),
        }),
  }
}

function projectObservation(value: unknown): CredentialObservationDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, observationFields)
  return {
    state: projectEnum(record.state, observationStates),
    snapshot: record.snapshot === null ? null : projectObservationSnapshot(record.snapshot),
    observation_version: projectSafeInteger(record.observation_version, { minimum: 0 }),
    observed_at_ms: projectNullableEpochMilliseconds(record.observed_at_ms),
    last_attempt_at_ms: projectNullableEpochMilliseconds(record.last_attempt_at_ms),
    ...(record.last_error_code === undefined
      ? {}
      : { last_error_code: projectString(record.last_error_code) }),
  }
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
    account: projectAccount(record.account, connectionType),
    auth_state: projectEnum(record.auth_state, authStates),
    ...(record.auth_error_code === undefined
      ? {}
      : { auth_error_code: projectInternalErrorCode(record.auth_error_code) }),
    ...(record.observation === undefined
      ? {}
      : { observation: projectObservation(record.observation) }),
    effective_status: effectiveStatus,
    recent_success_count: projectSafeInteger(record.recent_success_count, { minimum: 0 }),
    recent_failure_count: projectSafeInteger(record.recent_failure_count, { minimum: 0 }),
    consecutive_failure_count: projectSafeInteger(record.consecutive_failure_count, { minimum: 0 }),
    last_failure_category: projectEnum(record.last_failure_category, failureCategories),
    last_status_code:
      record.last_status_code === null
        ? null
        : projectSafeInteger(record.last_status_code, { minimum: 100, maximum: 999 }),
    cooldown_until_ms: cooldownUntil,
    ...(record.last_used_at_ms === undefined
      ? {}
      : { last_used_at_ms: projectEpochMilliseconds(record.last_used_at_ms) }),
    ...(record.daily_usage === undefined
      ? {}
      : { daily_usage: projectDailyUsage(record.daily_usage) }),
    recovery,
  }
}

function projectDailyUsage(value: unknown): CredentialDailyUsageDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, credentialDailyUsageFields)
  return {
    window_seconds: projectSafeInteger(record.window_seconds, { minimum: 1 }),
    success_count: projectSafeInteger(record.success_count, { minimum: 0 }),
    failure_count: projectSafeInteger(record.failure_count, { minimum: 0 }),
    data_complete: projectBoolean(record.data_complete),
  }
}

export function projectCredentialDetail(value: unknown): CredentialDetailDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, credentialDetailFields)
  return {
    credential: record.credential === null ? null : projectCredentialItem(record.credential),
    observation: record.observation === null ? null : projectObservation(record.observation),
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
    observation: item.observation ?? null,
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

function projectCredentialDownload(value: unknown): CredentialDownloadDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, credentialDownloadFields)
  const filename = projectString(record.filename)
  if (!/^[a-z0-9][a-z0-9._-]{0,191}\.json$/u.test(filename)) invalidResponse()
  return {
    filename,
    credential: projectCredentialJSON(record.credential),
  }
}

export async function downloadCredential(
  client: ApiClient,
  groupId: number,
  credentialId: number,
  signal?: AbortSignal,
): Promise<CredentialDownloadDto> {
  return projectCredentialDownload(
    await client.request(`/api/groups/${groupId}/credential/download`, {
      method: 'POST',
      headers: { 'X-Credential-ID': String(credentialId) },
      json: {},
      signal,
    }),
  )
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

export async function refreshCredentialObservation(
  client: ApiClient,
  groupId: number,
  credentialId: number,
  signal?: AbortSignal,
): Promise<CredentialObservationDto> {
  return projectObservation(
    await client.request(`/api/groups/${groupId}/credential/observation-refresh`, {
      method: 'POST',
      headers: { 'X-Credential-ID': String(credentialId) },
      json: {},
      signal,
    }),
  )
}

export async function refreshCredential(
  client: ApiClient,
  groupId: number,
  credentialId: number,
  signal?: AbortSignal,
): Promise<CredentialItemDto> {
  return projectCredentialItem(
    await client.request(`/api/groups/${groupId}/credential/refresh`, {
      method: 'POST',
      headers: { 'X-Credential-ID': String(credentialId) },
      json: {},
      signal,
    }),
    credentialId,
  )
}

export async function consumeCredentialResetCredit(
  client: ApiClient,
  groupId: number,
  credentialId: number,
  idempotencyKey: string,
  signal?: AbortSignal,
): Promise<CredentialResetCreditConsumeDto> {
  const record = projectRecord(
    await client.request(`/api/groups/${groupId}/credential/reset-credits/consume`, {
      method: 'POST',
      headers: { 'Idempotency-Key': idempotencyKey, 'X-Credential-ID': String(credentialId) },
      json: {},
      signal,
    }),
  )
  assertNoSecretLikeFields(record, resetCreditConsumeFields)
  const status = projectEnum(record.status, ['succeeded'] as const)
  return {
    status,
    windows_reset: projectSafeInteger(record.windows_reset, { minimum: 0 }),
    ...(record.redeemed_at_ms === undefined
      ? {}
      : { redeemed_at_ms: projectEpochMilliseconds(record.redeemed_at_ms) }),
    ...(record.observation === undefined
      ? {}
      : { observation: projectObservation(record.observation) }),
    ...(record.observation_pending === undefined
      ? {}
      : { observation_pending: projectBoolean(record.observation_pending) }),
    replayed: projectBoolean(record.replayed),
  }
}
