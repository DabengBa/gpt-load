import type { ApiClient } from '@/api/client'
import { enabledDataProtocols } from '@/api/control/protocols'

import {
  assertNoSecretLikeFields,
  projectArray,
  projectEnum,
  projectEpochMilliseconds,
  projectNullableRequestID,
  projectRecord,
  projectSafeInteger,
  projectString,
} from './projector'

export type ModelProbeOutcome = 'passed' | 'failed' | 'inconclusive'

export type ModelProbeReason =
  | 'invalid_credential'
  | 'model_unavailable'
  | 'rate_limited'
  | 'timeout'
  | 'upstream_error'
  | 'probe_incompatible'
  | 'unknown'
  | 'target_unavailable'
  | 'no_schedulable_credential'

export interface ModelProbeTargetDto {
  group_id: number
  model: string
}

export interface ModelProbeResultDto {
  group_id: number
  group_name: string
  model: string
  outcome: ModelProbeOutcome
  reason: ModelProbeReason | null
  protocol: string | null
  route_mode: 'native' | 'converted' | null
  status_code: number | null
  latency_ms: number | null
  credential_id: number | null
  credential_label: string | null
  log_id: string | null
  tested_at_ms: number
}

const probeOutcomes = ['passed', 'failed', 'inconclusive'] as const
const probeReasons = [
  'invalid_credential',
  'model_unavailable',
  'rate_limited',
  'timeout',
  'upstream_error',
  'probe_incompatible',
  'unknown',
  'target_unavailable',
  'no_schedulable_credential',
] as const
const routeModes = ['native', 'converted'] as const

// The response key set is asserted exactly, so the backend cannot add a field to
// this contract in one change and leave the projector behind: an unknown key
// fails projection at runtime, and `satisfies` below fails the build if a DTO
// field stops being projected.
export const modelProbeResultFields = [
  'group_id',
  'group_name',
  'model',
  'outcome',
  'reason',
  'protocol',
  'route_mode',
  'status_code',
  'latency_ms',
  'credential_id',
  'credential_label',
  'log_id',
  'tested_at_ms',
] as const satisfies readonly (keyof ModelProbeResultDto)[]

const modelProbeResponseFields = ['results'] as const

export function projectModelProbeResult(value: unknown): ModelProbeResultDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, modelProbeResultFields)
  return {
    group_id: projectSafeInteger(record.group_id, { minimum: 1 }),
    group_name: projectString(record.group_name, { allowEmpty: true }),
    model: projectString(record.model),
    outcome: projectEnum(record.outcome, probeOutcomes),
    reason: record.reason === null ? null : projectEnum(record.reason, probeReasons),
    protocol: record.protocol === null ? null : projectEnum(record.protocol, enabledDataProtocols),
    route_mode: record.route_mode === null ? null : projectEnum(record.route_mode, routeModes),
    status_code:
      record.status_code === null
        ? null
        : projectSafeInteger(record.status_code, { minimum: 0, maximum: 599 }),
    latency_ms:
      record.latency_ms === null ? null : projectSafeInteger(record.latency_ms, { minimum: 0 }),
    credential_id:
      record.credential_id === null
        ? null
        : projectSafeInteger(record.credential_id, { minimum: 1 }),
    credential_label:
      record.credential_label === null ? null : projectString(record.credential_label),
    log_id: projectNullableRequestID(record.log_id),
    tested_at_ms: projectEpochMilliseconds(record.tested_at_ms),
  } satisfies ModelProbeResultDto
}

export function projectModelProbeResponse(value: unknown): ModelProbeResultDto[] {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, modelProbeResponseFields)
  return projectArray(record.results, projectModelProbeResult)
}

export async function runModelProbe(
  client: ApiClient,
  targets: readonly ModelProbeTargetDto[],
): Promise<ModelProbeResultDto[]> {
  return projectModelProbeResponse(
    await client.request('/api/model-probe', {
      method: 'POST',
      json: { targets },
    }),
  )
}
