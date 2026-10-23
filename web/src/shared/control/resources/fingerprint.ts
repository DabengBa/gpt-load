import type { ApiClient } from '@shared/http/client'
import { ApiError, InvalidResponseError } from '@shared/http/errors'

import { modelProbeReasons, type ModelProbeReason } from './model-probe'
import {
  assertNoSecretLikeFields,
  projectArray,
  projectBoolean,
  projectEnum,
  projectEpochMilliseconds,
  projectFiniteNumber,
  projectHTTPURL,
  projectNullableRequestID,
  projectRecord,
  projectSafeInteger,
  projectString,
} from './projector'

// Model fingerprint detection. The backend executes the ModelTrace challenges
// through the target's real route and hands the answers to the separately
// deployed ModelTrace service. This module only speaks the control API:
// upstream credentials and the ModelTrace token never reach the browser.
//
// Wire contract (`control.modeltrace-probe.run`, POST /api/modeltrace-probe):
//   { group_id, model }
//     -> { group_id, group_name, model, service, challenges, analysis,
//          failure, tested_at_ms }
// `analysis` and `failure` are mutually exclusive: a run either attributes the
// outputs or explains why it could not. Errors use the standard error envelope
// (MODELTRACE_DISABLED / MODELTRACE_MISCONFIGURED / MODELTRACE_UNAVAILABLE /
// MODELTRACE_ANALYSIS_REJECTED / NO_ACTIVE_CREDENTIAL / BAD_REQUEST).

export const modelFingerprintPath = '/api/modeltrace-probe' as const

export interface ModelFingerprintTargetDto {
  group_id: number
  model: string
}

export interface ModelFingerprintServiceDto {
  base_url: string
  enabled: boolean
}

export type ModelFingerprintChallengeOutcome = 'succeeded' | 'failed'

export interface ModelFingerprintChallengeDto {
  id: string
  expected_count: number
  outcome: ModelFingerprintChallengeOutcome
  reason: ModelProbeReason | null
  log_id: string | null
  latency_ms: number | null
  status_code: number | null
}

export interface ModelFingerprintCandidateDto {
  model: string
  family: string
  probability: number
  conditional_probability: number
}

export interface ModelFingerprintDiagnosticDto {
  prompt_id: string
  accepted: boolean
  parsed_numbers: number
  minimum_numbers: number
}

export interface ModelFingerprintBankDto {
  models: number
  built_at: string
  bank_sha256: string
}

export interface ModelFingerprintUpstreamDto {
  repository: string
  commit: string
}

export interface ModelFingerprintAnalysisDto {
  prediction: string
  prediction_name: string
  probability: number
  family_prediction: string
  family_probability: number
  used_outputs: number
  method: string
  candidates: ModelFingerprintCandidateDto[]
  diagnostics: ModelFingerprintDiagnosticDto[]
  bank: ModelFingerprintBankDto
  upstream: ModelFingerprintUpstreamDto
}

export const fingerprintFailureKinds = ['no_usable_outputs', 'analysis_rejected', 'analysis_unavailable'] as const

export type FingerprintFailureKind = (typeof fingerprintFailureKinds)[number]

export interface ModelFingerprintResultDto {
  group_id: number
  group_name: string
  model: string
  service: ModelFingerprintServiceDto | null
  challenges: ModelFingerprintChallengeDto[]
  analysis: ModelFingerprintAnalysisDto | null
  failure: FingerprintFailureKind | null
  tested_at_ms: number
}

// Exact key sets: an unknown key fails projection at runtime, and `satisfies`
// fails the build if a DTO field stops being projected.
export const modelFingerprintResultFields = [
  'group_id',
  'group_name',
  'model',
  'service',
  'challenges',
  'analysis',
  'failure',
  'tested_at_ms',
] as const satisfies readonly (keyof ModelFingerprintResultDto)[]

const serviceFields = [
  'base_url',
  'enabled',
] as const satisfies readonly (keyof ModelFingerprintServiceDto)[]
const challengeFields = [
  'id',
  'expected_count',
  'outcome',
  'reason',
  'log_id',
  'latency_ms',
  'status_code',
] as const satisfies readonly (keyof ModelFingerprintChallengeDto)[]
const analysisFields = [
  'prediction',
  'prediction_name',
  'probability',
  'family_prediction',
  'family_probability',
  'used_outputs',
  'method',
  'candidates',
  'diagnostics',
  'bank',
  'upstream',
] as const satisfies readonly (keyof ModelFingerprintAnalysisDto)[]
const candidateFields = [
  'model',
  'family',
  'probability',
  'conditional_probability',
] as const satisfies readonly (keyof ModelFingerprintCandidateDto)[]
const diagnosticFields = [
  'prompt_id',
  'accepted',
  'parsed_numbers',
  'minimum_numbers',
] as const satisfies readonly (keyof ModelFingerprintDiagnosticDto)[]
const bankFields = [
  'models',
  'built_at',
  'bank_sha256',
] as const satisfies readonly (keyof ModelFingerprintBankDto)[]
const upstreamFields = [
  'repository',
  'commit',
] as const satisfies readonly (keyof ModelFingerprintUpstreamDto)[]
const challengeOutcomes = ['succeeded', 'failed'] as const

function invalidResponse(): never {
  throw new InvalidResponseError()
}

function projectNonBlankString(value: unknown): string {
  const result = projectString(value)
  if (result.trim().length === 0 || result !== result.trim()) invalidResponse()
  return result
}

function projectProbability(value: unknown): number {
  return projectFiniteNumber(value, { minimum: 0, maximum: 1 })
}

function projectService(value: unknown): ModelFingerprintServiceDto | null {
  if (value === null) return null
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, serviceFields)
  if (projectBoolean(record.enabled) !== true) invalidResponse()
  return { base_url: projectHTTPURL(record.base_url), enabled: true }
}

function projectChallenge(value: unknown): ModelFingerprintChallengeDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, challengeFields)
  return {
    id: projectNonBlankString(record.id),
    expected_count: projectSafeInteger(record.expected_count, { minimum: 1 }),
    outcome: projectEnum(record.outcome, challengeOutcomes),
    reason: record.reason === null ? null : projectEnum(record.reason, modelProbeReasons),
    log_id: projectNullableRequestID(record.log_id),
    latency_ms:
      record.latency_ms === null ? null : projectSafeInteger(record.latency_ms, { minimum: 0 }),
    status_code:
      record.status_code === null
        ? null
        : projectSafeInteger(record.status_code, { minimum: 0, maximum: 599 }),
  }
}

function projectCandidate(value: unknown): ModelFingerprintCandidateDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, candidateFields)
  return {
    model: projectNonBlankString(record.model),
    family: projectString(record.family, { allowEmpty: true }),
    probability: projectProbability(record.probability),
    conditional_probability: projectProbability(record.conditional_probability),
  }
}

function projectDiagnostic(value: unknown): ModelFingerprintDiagnosticDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, diagnosticFields)
  return {
    prompt_id: projectNonBlankString(record.prompt_id),
    accepted: projectBoolean(record.accepted),
    parsed_numbers: projectSafeInteger(record.parsed_numbers, { minimum: 0 }),
    minimum_numbers: projectSafeInteger(record.minimum_numbers, { minimum: 0 }),
  }
}

function projectAnalysis(value: unknown): ModelFingerprintAnalysisDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, analysisFields)
  const candidates = projectArray(record.candidates, projectCandidate)
  if (candidates.length === 0) invalidResponse()
  if (new Set(candidates.map((candidate) => candidate.model)).size !== candidates.length) {
    invalidResponse()
  }
  const bank = projectRecord(record.bank)
  assertNoSecretLikeFields(bank, bankFields)
  const upstream = projectRecord(record.upstream)
  assertNoSecretLikeFields(upstream, upstreamFields)
  return {
    prediction: projectNonBlankString(record.prediction),
    prediction_name: projectString(record.prediction_name, { allowEmpty: true }),
    probability: projectProbability(record.probability),
    family_prediction: projectString(record.family_prediction, { allowEmpty: true }),
    family_probability: projectProbability(record.family_probability),
    used_outputs: projectSafeInteger(record.used_outputs, { minimum: 0 }),
    method: projectString(record.method, { allowEmpty: true }),
    candidates,
    diagnostics: projectArray(record.diagnostics, projectDiagnostic),
    bank: {
      models: projectSafeInteger(bank.models, { minimum: 0 }),
      built_at: projectString(bank.built_at, { allowEmpty: true }),
      bank_sha256: projectString(bank.bank_sha256, { allowEmpty: true }),
    },
    upstream: {
      repository: projectString(upstream.repository, { allowEmpty: true }),
      commit: projectString(upstream.commit, { allowEmpty: true }),
    },
  }
}

export function projectModelFingerprintResult(value: unknown): ModelFingerprintResultDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, modelFingerprintResultFields)
  const challenges = projectArray(record.challenges, projectChallenge)
  if (challenges.length === 0) invalidResponse()
  const analysis = record.analysis === null ? null : projectAnalysis(record.analysis)
  const failure =
    record.failure === null ? null : projectEnum(record.failure, fingerprintFailureKinds)
  // A run either attributes the outputs or explains why it could not; both or
  // neither would make the UI state ambiguous.
  if ((analysis === null) === (failure === null)) invalidResponse()
  return {
    group_id: projectSafeInteger(record.group_id, { minimum: 1 }),
    group_name: projectString(record.group_name, { allowEmpty: true }),
    model: projectNonBlankString(record.model),
    service: projectService(record.service),
    challenges,
    analysis,
    failure,
    tested_at_ms: projectEpochMilliseconds(record.tested_at_ms),
  }
}

export async function runModelFingerprint(
  client: ApiClient,
  target: ModelFingerprintTargetDto,
): Promise<ModelFingerprintResultDto> {
  return projectModelFingerprintResult(
    await client.request(modelFingerprintPath, {
      method: 'POST',
      json: { group_id: target.group_id, model: target.model },
    }),
  )
}

// One stable UI failure per control error code; anything unexpected collapses
// into the generic failure instead of inventing an explanation.
export const fingerprintRunFailures = [
  'disabled',
  'misconfigured',
  'unavailable',
  'analysis_rejected',
  'no_active_credential',
  'unsupported_target',
  'failed',
] as const

export type FingerprintRunFailure = (typeof fingerprintRunFailures)[number]

export function fingerprintRunFailure(cause: unknown): FingerprintRunFailure {
  if (!(cause instanceof ApiError)) return 'failed'
  switch (cause.code) {
    case 'MODELTRACE_DISABLED':
      return 'disabled'
    case 'MODELTRACE_MISCONFIGURED':
      return 'misconfigured'
    case 'MODELTRACE_UNAVAILABLE':
      return 'unavailable'
    case 'MODELTRACE_ANALYSIS_REJECTED':
      return 'analysis_rejected'
    case 'NO_ACTIVE_CREDENTIAL':
      return 'no_active_credential'
    case 'BAD_REQUEST':
      return 'unsupported_target'
    default:
      return 'failed'
  }
}
