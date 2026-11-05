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
  projectNullableRequestID,
  projectRecord,
  projectSafeInteger,
  projectString,
} from './projector'

// Challenges use the target's real route; analysis runs locally in the backend.
//
// Wire contract (`control.modeltrace-probe.run`, POST /api/modeltrace-probe):
//   { group_id, model }
//     -> { group_id, group_name, model, challenges, analysis,
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
  display_name: string
  family: string
  family_name: string
  score: number
  probability: number | null
}

export interface ModelFingerprintDiagnosticDto {
  index: number
  accepted: boolean
  parsed_numbers: number
  minimum_numbers: number
  raw_out_of_range_count: number
}

export interface ModelFingerprintAnalysisDto {
  decision: 'scored' | 'partial' | 'unscorable' | 'not_model'
  prediction: string
  prediction_name: string
  probability: number | null
  family_prediction: string
  family_prediction_name: string
  ranking_score: number | null
  probability_status: string
  probability_scope: string | null
  calibration: { method: string; run: string; tau: number; sha256: string | null } | null
  used_outputs: number
  method: string
  results: ModelFingerprintCandidateDto[]
  diagnostics: ModelFingerprintDiagnosticDto[]
  sequence: { autocorrelation: number; small_step_share: number; model_like: boolean } | null
  model_version: { base_sha256: string }
  evidence: {
    insufficient: boolean
    label: string
    reason: string
    threshold: number | null
    method: string
  }
}

export const fingerprintFailureKinds = [
  'no_usable_outputs',
  'analysis_rejected',
  'analysis_unavailable',
] as const

export type FingerprintFailureKind = (typeof fingerprintFailureKinds)[number]

export interface ModelFingerprintResultDto {
  group_id: number
  group_name: string
  model: string

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

  'challenges',
  'analysis',
  'failure',
  'tested_at_ms',
] as const satisfies readonly (keyof ModelFingerprintResultDto)[]

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
  'decision',
  'prediction',
  'prediction_name',
  'probability',
  'family_prediction',
  'family_prediction_name',
  'ranking_score',
  'probability_status',
  'probability_scope',
  'calibration',
  'used_outputs',
  'method',
  'results',
  'diagnostics',
  'sequence',
  'model_version',
  'evidence',
] as const satisfies readonly (keyof ModelFingerprintAnalysisDto)[]
const candidateFields = [
  'model',
  'display_name',
  'family',
  'family_name',
  'score',
  'probability',
] as const satisfies readonly (keyof ModelFingerprintCandidateDto)[]
const diagnosticFields = [
  'index',
  'accepted',
  'parsed_numbers',
  'minimum_numbers',
  'raw_out_of_range_count',
] as const satisfies readonly (keyof ModelFingerprintDiagnosticDto)[]
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
    display_name: projectString(record.display_name, { allowEmpty: true }),
    family: projectString(record.family, { allowEmpty: true }),
    family_name: projectString(record.family_name, { allowEmpty: true }),
    score: projectFiniteNumber(record.score),
    probability: record.probability === null ? null : projectProbability(record.probability),
  }
}

function projectDiagnostic(value: unknown): ModelFingerprintDiagnosticDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, diagnosticFields)
  return {
    index: projectSafeInteger(record.index, { minimum: 0 }),
    accepted: projectBoolean(record.accepted),
    parsed_numbers: projectSafeInteger(record.parsed_numbers, { minimum: 0 }),
    minimum_numbers: projectSafeInteger(record.minimum_numbers, { minimum: 0 }),
    raw_out_of_range_count: projectSafeInteger(record.raw_out_of_range_count, { minimum: 0 }),
  }
}

function projectAnalysis(value: unknown): ModelFingerprintAnalysisDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, analysisFields)
  const results = projectArray(record.results, projectCandidate)
  if (new Set(results.map((candidate) => candidate.model)).size !== results.length) {
    invalidResponse()
  }
  const decision = projectEnum(record.decision, [
    'scored',
    'partial',
    'unscorable',
    'not_model',
  ] as const)
  const calibration = record.calibration === null ? null : projectRecord(record.calibration)
  const sequence = record.sequence === null ? null : projectRecord(record.sequence)
  const version = projectRecord(record.model_version)
  const evidence = projectRecord(record.evidence)
  if (calibration) assertNoSecretLikeFields(calibration, ['method', 'run', 'tau', 'sha256'])
  if (sequence)
    assertNoSecretLikeFields(sequence, ['autocorrelation', 'small_step_share', 'model_like'])
  assertNoSecretLikeFields(version, ['base_sha256'])
  assertNoSecretLikeFields(evidence, ['insufficient', 'label', 'reason', 'threshold', 'method'])
  return {
    decision,
    prediction: projectString(record.prediction, { allowEmpty: true }),
    prediction_name: projectString(record.prediction_name, { allowEmpty: true }),
    probability: record.probability === null ? null : projectProbability(record.probability),
    family_prediction: projectString(record.family_prediction, { allowEmpty: true }),
    family_prediction_name: projectString(record.family_prediction_name, { allowEmpty: true }),
    ranking_score: record.ranking_score === null ? null : projectFiniteNumber(record.ranking_score),
    probability_status: projectString(record.probability_status),
    probability_scope:
      record.probability_scope === null ? null : projectString(record.probability_scope),
    calibration:
      calibration === null
        ? null
        : {
            method: projectString(calibration.method),
            run: projectString(calibration.run, { allowEmpty: true }),
            tau: projectFiniteNumber(calibration.tau),
            sha256: calibration.sha256 === null ? null : projectString(calibration.sha256),
          },
    used_outputs: projectSafeInteger(record.used_outputs, { minimum: 0 }),
    method: projectString(record.method, { allowEmpty: true }),
    results,
    diagnostics: projectArray(record.diagnostics, projectDiagnostic),
    sequence:
      sequence === null
        ? null
        : {
            autocorrelation: projectFiniteNumber(sequence.autocorrelation),
            small_step_share: projectFiniteNumber(sequence.small_step_share),
            model_like: projectBoolean(sequence.model_like),
          },
    model_version: { base_sha256: projectString(version.base_sha256, { allowEmpty: true }) },
    evidence: {
      insufficient: projectBoolean(evidence.insufficient),
      label: projectString(evidence.label, { allowEmpty: true }),
      reason: projectString(evidence.reason, { allowEmpty: true }),
      threshold: evidence.threshold === null ? null : projectFiniteNumber(evidence.threshold),
      method: projectString(evidence.method, { allowEmpty: true }),
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
