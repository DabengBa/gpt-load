import type { QueryFunctionContext } from '@tanstack/query-core'

import type { ApiClient } from '@shared/http/client'
import type {
  AgentCredentialCreateResultDto,
  AgentCredentialDto,
  AgentCredentialListDto,
  AgentCredentialScope,
} from '@shared/control/types'
import { InvalidResponseError } from '@shared/http/errors'
import { controlQueryKeys } from '@shared/control/query-keys'

import {
  assertNoSecretLikeFields,
  projectArray,
  projectEnum,
  projectEpochMilliseconds,
  projectNullableEpochMilliseconds,
  projectRecord,
  projectSafeInteger,
  projectString,
} from './projector'

export type {
  AgentCredentialCreateResultDto,
  AgentCredentialDto,
  AgentCredentialListDto,
  AgentCredentialScope,
} from '@shared/control/types'

export const agentCredentialScopes = [
  'diagnostics:read',
  'changes:propose',
  'changes:apply',
] as const satisfies readonly AgentCredentialScope[]

export interface CreateAgentCredentialRequest {
  name: string
  scopes: AgentCredentialScope[]
  expires_at_ms?: number | null
}

const metadataFields = [
  'id',
  'name',
  'scopes',
  'status',
  'expires_at_ms',
  'disabled_at_ms',
  'created_at_ms',
  'updated_at_ms',
] as const
const listFields = ['items'] as const
const createFields = [...metadataFields, 'secret', 'replayed', 'operation_id'] as const

function invalidResponse(): never {
  throw new InvalidResponseError()
}

function projectNonBlankTrimmedString(value: unknown): string {
  const result = projectString(value)
  if (result.trim().length === 0 || result !== result.trim()) invalidResponse()
  return result
}

export function projectAgentCredentialMetadata(value: unknown): AgentCredentialDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, metadataFields)
  const scopes = projectArray(record.scopes, (scope) =>
    projectEnum(scope, agentCredentialScopes),
  )
  if (scopes.length === 0 || new Set(scopes).size !== scopes.length) invalidResponse()
  return {
    id: projectSafeInteger(record.id, { minimum: 1 }),
    name: projectNonBlankTrimmedString(record.name),
    scopes,
    status: projectEnum(record.status, ['active', 'disabled'] as const),
    expires_at_ms: projectNullableEpochMilliseconds(record.expires_at_ms),
    disabled_at_ms: projectNullableEpochMilliseconds(record.disabled_at_ms),
    created_at_ms: projectEpochMilliseconds(record.created_at_ms),
    updated_at_ms: projectEpochMilliseconds(record.updated_at_ms),
  }
}

export function projectAgentCredentialList(value: unknown): AgentCredentialListDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, listFields)
  const items = projectArray(record.items, projectAgentCredentialMetadata)
  if (new Set(items.map(({ id }) => id)).size !== items.length) invalidResponse()
  return { items }
}

export async function listAgentCredentials(
  client: ApiClient,
  signal?: AbortSignal,
): Promise<AgentCredentialListDto> {
  return projectAgentCredentialList(
    await client.request('/api/agent-credentials', { method: 'GET', signal }),
  )
}

export function agentCredentialListQueryOptions(client: ApiClient) {
  const key = controlQueryKeys.agentCredentials.list()
  return {
    queryKey: key,
    queryFn: ({ signal }: QueryFunctionContext) => listAgentCredentials(client, signal),
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  }
}

/**
 * The plaintext `secret` is present only on the first (non-replayed) response;
 * the backend stores only an HMAC fingerprint, so it can never be re-read.
 */
export async function createAgentCredential(
  client: ApiClient,
  body: CreateAgentCredentialRequest,
  idempotencyKey: string,
  signal?: AbortSignal,
): Promise<AgentCredentialCreateResultDto> {
  const record = projectRecord(
    await client.request('/api/agent-credentials', {
      method: 'POST',
      headers: { 'Idempotency-Key': idempotencyKey },
      json: body,
      signal,
    }),
  )
  assertNoSecretLikeFields(record, createFields)
  const secret = record.secret === undefined ? undefined : projectString(record.secret)
  if (
    typeof record.replayed !== 'boolean' ||
    record.replayed === (secret !== undefined)
  ) {
    invalidResponse()
  }
  const metadata = Object.fromEntries(metadataFields.map((field) => [field, record[field]]))
  return {
    ...projectAgentCredentialMetadata(metadata),
    ...(secret === undefined ? {} : { secret }),
    replayed: record.replayed,
    operation_id: projectString(record.operation_id),
  }
}

export async function disableAgentCredential(
  client: ApiClient,
  id: number,
  signal?: AbortSignal,
): Promise<AgentCredentialDto> {
  return projectAgentCredentialMetadata(
    await client.request(`/api/agent-credentials/${id}/disable`, {
      method: 'POST',
      signal,
    }),
  )
}

export const agentCredentialResources = {
  list: {
    queryKey: controlQueryKeys.agentCredentials.all,
    gcTime: 0,
    cleanup: 'authenticated-session',
    optimisticUpdates: false,
    allowedFields: metadataFields,
  },
} as const
