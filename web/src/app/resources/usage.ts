import { keepPreviousData, queryOptions } from '@tanstack/vue-query'
import { computed, toValue, type MaybeRefOrGetter } from 'vue'

import type { ApiClient } from '@/api/client'
import { InvalidResponseError } from '@/api/errors'
import { controlQueryKeys } from '@/app/query-keys'
import { timeRanges, type TimeRange } from '@/lib/time'
import { projectChannelID } from './channels'

import {
  assertNoSecretLikeFields,
  projectArray,
  projectEpochMilliseconds,
  projectEnum,
  projectNonNegativeInt64String,
  projectNullableEpochMilliseconds,
  projectRecord,
  projectSafeInteger,
  projectString,
} from './projector'

export type UsageDistributionDimension = 'group' | 'model' | 'access_key'
export type UsageDistributionMetric = 'requests' | 'tokens' | 'cost'
export const usageRanges = timeRanges
export type UsageRange = TimeRange

export type UsageBreakdownPageSize = 20 | 50 | 100
export type UsageBreakdownSort =
  | 'model'
  | 'group'
  | 'channel'
  | 'request_count'
  | 'success_count'
  | 'failure_count'
  | 'success_rate'
  | 'average_latency'
  | 'uncached_input_tokens'
  | 'cache_read_tokens'
  | 'cache_write_5m_tokens'
  | 'cache_write_1h_tokens'
  | 'cache_write_unknown_tokens'
  | 'output_tokens'
  | 'total_tokens'
  | 'estimated_cost_nano_usd'
export type UsageBreakdownSortDirection = 'asc' | 'desc'

export const defaultUsageBreakdownSort: UsageBreakdownSort = 'estimated_cost_nano_usd'

export function normalizeUsageBreakdownSort(value: unknown): UsageBreakdownSort {
  switch (value) {
    case 'model':
    case 'group':
    case 'channel':
    case 'request_count':
    case 'success_count':
    case 'failure_count':
    case 'success_rate':
    case 'average_latency':
    case 'uncached_input_tokens':
    case 'cache_read_tokens':
    case 'cache_write_5m_tokens':
    case 'cache_write_1h_tokens':
    case 'cache_write_unknown_tokens':
    case 'output_tokens':
    case 'total_tokens':
    case 'estimated_cost_nano_usd':
      return value
    default:
      return defaultUsageBreakdownSort
  }
}

export function defaultUsageBreakdownSortDirectionFor(
  sort: UsageBreakdownSort,
): UsageBreakdownSortDirection {
  return sort === 'model' || sort === 'group' || sort === 'channel' ? 'asc' : 'desc'
}

export function normalizeUsageBreakdownSortDirection(
  value: unknown,
  sort: UsageBreakdownSort,
): UsageBreakdownSortDirection {
  if (value === 'asc' || value === 'desc') return value
  return defaultUsageBreakdownSortDirectionFor(sort)
}

export interface UsageFilters {
  range: UsageRange
  group_id?: number
  channel_id?: string
  credential_id?: number
  upstream_model?: string
  breakdown_page?: number
  breakdown_page_size?: UsageBreakdownPageSize
  breakdown_sort?: UsageBreakdownSort
  breakdown_sort_direction?: UsageBreakdownSortDirection
}

export interface UsageAggregateDto {
  request_count: number
  success_count: number
  failure_count: number
  uncached_input_tokens: number
  cache_read_tokens: number
  cache_write_5m_tokens: number
  cache_write_1h_tokens: number
  cache_write_unknown_tokens: number
  output_tokens: number
  total_tokens: number
  estimated_cost_nano_usd: string
  duration_ms_total: number
  duration_sample_count: number
  usage_missing_count: number
  partial_count: number
  unpriced_request_count: number
  pricing_partial_count: number
}

export interface UsageDistributionAggregateDto {
  request_count: number
  total_tokens: number
  estimated_cost_nano_usd: string
}

export interface UsageReportDto {
  range: UsageRange
  granularity: 'hour' | 'day'
  bucket_width_ms: number
  from_ms: number
  to_ms: number
  observed_at_ms: number
  summary: UsageAggregateDto
  series: Array<UsageAggregateDto & { bucket_start_ms: number; bucket_end_ms: number }>
  distributions: {
    group?: Record<UsageDistributionMetric, UsageDistributionDto>
    model: Record<UsageDistributionMetric, UsageDistributionDto>
    access_key?: Record<UsageDistributionMetric, UsageDistributionDto>
  }
  collection_health: {
    scope: 'current_process' | 'access_key'
    dropped_total: number
    write_failure_total: number
    last_write_failure_at_ms: number | null
  }
  breakdown: UsageBreakdownDto
}

export type UsageBreakdownScope = 'admin' | 'access_key'

export interface UsageBreakdownRowDto extends UsageAggregateDto {
  model: string
  group_id?: number
  channel_id?: string
}

export interface UsageBreakdownDto {
  scope: UsageBreakdownScope
  rows: UsageBreakdownRowDto[]
  total: UsageAggregateDto
  pagination: {
    page: number
    page_size: UsageBreakdownPageSize
    total_items: number
    total_pages: number
  }
}

export interface UsageDistributionDto {
  dimension: UsageDistributionDimension
  metric: UsageDistributionMetric
  items: Array<
    UsageDistributionAggregateDto & {
      group_id?: number
      model?: string
      access_key_id?: number
    }
  >
  other: UsageDistributionAggregateDto | null
}

const aggregateKeys = [
  'request_count',
  'success_count',
  'failure_count',
  'uncached_input_tokens',
  'cache_read_tokens',
  'cache_write_5m_tokens',
  'cache_write_1h_tokens',
  'cache_write_unknown_tokens',
  'output_tokens',
  'total_tokens',
  'duration_ms_total',
  'duration_sample_count',
  'usage_missing_count',
  'partial_count',
  'unpriced_request_count',
  'pricing_partial_count',
] as const
const aggregateFields = [...aggregateKeys, 'estimated_cost_nano_usd'] as const
const distributionAggregateFields = [
  'request_count',
  'total_tokens',
  'estimated_cost_nano_usd',
] as const
const reportFields = [
  'range',
  'granularity',
  'bucket_width_ms',
  'from_ms',
  'to_ms',
  'observed_at_ms',
  'summary',
  'series',
  'distributions',
  'collection_health',
  'breakdown',
] as const
const hourMs = 60 * 60 * 1000
const dayMs = 24 * hourMs
const usageRangeContract: Record<
  UsageRange,
  { granularity: 'hour' | 'day'; bucketWidthMs: number; buckets: number }
> = {
  '1h': { granularity: 'hour', bucketWidthMs: hourMs, buckets: 1 },
  '24h': { granularity: 'hour', bucketWidthMs: hourMs, buckets: 24 },
  '3d': { granularity: 'hour', bucketWidthMs: 3 * hourMs, buckets: 24 },
  '7d': { granularity: 'hour', bucketWidthMs: 6 * hourMs, buckets: 28 },
  '15d': { granularity: 'hour', bucketWidthMs: 12 * hourMs, buckets: 30 },
  '30d': { granularity: 'day', bucketWidthMs: dayMs, buckets: 30 },
}

function invalidResponse(): never {
  throw new InvalidResponseError()
}

function isUTCAligned(timestampValue: number, bucketWidthMs: number): boolean {
  return timestampValue % bucketWidthMs === 0
}

export function projectUsageAggregate(value: unknown): UsageAggregateDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, aggregateFields)
  const result: UsageAggregateDto = {
    request_count: projectSafeInteger(record.request_count, { minimum: 0 }),
    success_count: projectSafeInteger(record.success_count, { minimum: 0 }),
    failure_count: projectSafeInteger(record.failure_count, { minimum: 0 }),
    uncached_input_tokens: projectSafeInteger(record.uncached_input_tokens, { minimum: 0 }),
    cache_read_tokens: projectSafeInteger(record.cache_read_tokens, { minimum: 0 }),
    cache_write_5m_tokens: projectSafeInteger(record.cache_write_5m_tokens, { minimum: 0 }),
    cache_write_1h_tokens: projectSafeInteger(record.cache_write_1h_tokens, { minimum: 0 }),
    cache_write_unknown_tokens: projectSafeInteger(record.cache_write_unknown_tokens, {
      minimum: 0,
    }),
    output_tokens: projectSafeInteger(record.output_tokens, { minimum: 0 }),
    total_tokens: projectSafeInteger(record.total_tokens, { minimum: 0 }),
    estimated_cost_nano_usd: projectNonNegativeInt64String(record.estimated_cost_nano_usd),
    duration_ms_total: projectSafeInteger(record.duration_ms_total, { minimum: 0 }),
    duration_sample_count: projectSafeInteger(record.duration_sample_count, { minimum: 0 }),
    usage_missing_count: projectSafeInteger(record.usage_missing_count, { minimum: 0 }),
    partial_count: projectSafeInteger(record.partial_count, { minimum: 0 }),
    unpriced_request_count: projectSafeInteger(record.unpriced_request_count, { minimum: 0 }),
    pricing_partial_count: projectSafeInteger(record.pricing_partial_count, { minimum: 0 }),
  }
  if (
    result.duration_sample_count > result.request_count ||
    result.success_count + result.failure_count !== result.request_count ||
    result.total_tokens !==
      result.uncached_input_tokens +
        result.cache_read_tokens +
        result.cache_write_5m_tokens +
        result.cache_write_1h_tokens +
        result.cache_write_unknown_tokens +
        result.output_tokens ||
    result.usage_missing_count > result.request_count ||
    result.partial_count > result.request_count ||
    result.unpriced_request_count > result.request_count ||
    result.pricing_partial_count > result.request_count
  ) {
    invalidResponse()
  }
  return result
}

const breakdownAggregateFields = [...aggregateFields] as const

function sameUsageAggregate(left: UsageAggregateDto, right: UsageAggregateDto): boolean {
  return breakdownAggregateFields.every((field) => left[field] === right[field])
}

function projectUsageModel(value: unknown): string {
  const model = projectString(value)
  if (
    new TextEncoder().encode(model).length > 255 ||
    model !== model.trim() ||
    /[\p{Cc}]/u.test(model)
  ) {
    invalidResponse()
  }
  return model
}

function expectedUsagePageItems(pagination: UsageBreakdownDto['pagination']): number {
  if (pagination.total_items === 0 || pagination.page > pagination.total_pages) return 0
  if (pagination.page < pagination.total_pages) return pagination.page_size
  const remainder = pagination.total_items % pagination.page_size
  return remainder === 0 ? pagination.page_size : remainder
}

function projectUsagePagination(value: unknown): UsageBreakdownDto['pagination'] {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, ['page', 'page_size', 'total_items', 'total_pages'])
  const page = projectSafeInteger(record.page, { minimum: 1 })
  const pageSize = projectSafeInteger(record.page_size, { minimum: 20, maximum: 100 })
  const totalItems = projectSafeInteger(record.total_items, { minimum: 0 })
  const totalPages = projectSafeInteger(record.total_pages, { minimum: 0 })
  if (
    (pageSize !== 20 && pageSize !== 50 && pageSize !== 100) ||
    totalPages !== (totalItems === 0 ? 0 : Math.ceil(totalItems / pageSize))
  ) {
    invalidResponse()
  }
  return {
    page,
    page_size: pageSize as UsageBreakdownPageSize,
    total_items: totalItems,
    total_pages: totalPages,
  }
}

export function projectUsageBreakdown(value: unknown): UsageBreakdownDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, ['scope', 'rows', 'total', 'pagination'])
  const scope = projectEnum(record.scope, ['admin', 'access_key'] as const)
  const total = projectUsageAggregate(record.total)
  const pagination = projectUsagePagination(record.pagination)
  const rows = projectArray(record.rows, (value): UsageBreakdownRowDto => {
    const row = projectRecord(value)
    const identityFields = scope === 'admin' ? ['model', 'group_id', 'channel_id'] : ['model']
    assertNoSecretLikeFields(row, [...identityFields, ...breakdownAggregateFields])
    const model = projectUsageModel(row.model)
    const aggregate = projectUsageAggregate(
      Object.fromEntries(breakdownAggregateFields.map((field) => [field, row[field]])),
    )
    if (scope === 'admin') {
      return {
        ...aggregate,
        model,
        group_id: projectSafeInteger(row.group_id, { minimum: 1 }),
        channel_id: projectChannelID(row.channel_id),
      }
    }
    return { ...aggregate, model }
  })
  const identities = new Set<string>()
  for (const row of rows) {
    const identity = JSON.stringify([
      scope,
      row.model,
      row.group_id ?? null,
      row.channel_id ?? null,
    ])
    if (identities.has(identity)) invalidResponse()
    identities.add(identity)
  }
  if (rows.length !== expectedUsagePageItems(pagination)) invalidResponse()
  return { scope, rows, total, pagination }
}

function projectUsageDistributionAggregate(value: unknown): UsageDistributionAggregateDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, distributionAggregateFields)
  return {
    request_count: projectSafeInteger(record.request_count, { minimum: 0 }),
    total_tokens: projectSafeInteger(record.total_tokens, { minimum: 0 }),
    estimated_cost_nano_usd: projectNonNegativeInt64String(record.estimated_cost_nano_usd),
  }
}

function projectCollectionHealth(value: unknown): UsageReportDto['collection_health'] {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, [
    'scope',
    'dropped_total',
    'write_failure_total',
    'last_write_failure_at_ms',
  ])
  return {
    scope: projectEnum(record.scope, ['current_process', 'access_key'] as const),
    dropped_total: projectSafeInteger(record.dropped_total, { minimum: 0 }),
    write_failure_total: projectSafeInteger(record.write_failure_total, { minimum: 0 }),
    last_write_failure_at_ms: projectNullableEpochMilliseconds(record.last_write_failure_at_ms),
  }
}

export function projectUsageReport(value: unknown): UsageReportDto {
  const record = projectRecord(value)
  assertNoSecretLikeFields(record, reportFields)
  const range = projectEnum(record.range, usageRanges)
  const granularity = projectEnum(record.granularity, ['hour', 'day'] as const)
  const rangeContract = usageRangeContract[range]
  if (granularity !== rangeContract.granularity) invalidResponse()
  const bucketWidthMs = projectSafeInteger(record.bucket_width_ms, { minimum: hourMs })
  if (bucketWidthMs !== rangeContract.bucketWidthMs) invalidResponse()
  const observedAtMS = projectEpochMilliseconds(record.observed_at_ms)
  const rangeFromMS = projectEpochMilliseconds(record.from_ms)
  const rangeToMS = projectEpochMilliseconds(record.to_ms)
  const bucketCount = rangeContract.buckets
  if (
    !isUTCAligned(rangeFromMS, bucketWidthMs) ||
    !isUTCAligned(rangeToMS, bucketWidthMs) ||
    rangeToMS - rangeFromMS !== bucketWidthMs * bucketCount ||
    observedAtMS < rangeToMS - bucketWidthMs ||
    observedAtMS >= rangeToMS
  ) {
    invalidResponse()
  }

  let previousBucketEndMS = rangeFromMS
  const series = projectArray(record.series, (value) => {
    const item = projectRecord(value)
    assertNoSecretLikeFields(item, [
      ...aggregateKeys,
      'estimated_cost_nano_usd',
      'bucket_start_ms',
      'bucket_end_ms',
    ])
    const bucketStartMS = projectEpochMilliseconds(item.bucket_start_ms)
    const bucketEndMS = projectEpochMilliseconds(item.bucket_end_ms)
    if (
      !isUTCAligned(bucketStartMS, bucketWidthMs) ||
      !isUTCAligned(bucketEndMS, bucketWidthMs) ||
      bucketEndMS - bucketStartMS !== bucketWidthMs ||
      bucketStartMS < rangeFromMS ||
      bucketEndMS > rangeToMS ||
      bucketStartMS < previousBucketEndMS
    ) {
      invalidResponse()
    }
    previousBucketEndMS = bucketEndMS
    return {
      ...projectUsageAggregate(
        Object.fromEntries(aggregateFields.map((field) => [field, item[field]])),
      ),
      bucket_start_ms: bucketStartMS,
      bucket_end_ms: bucketEndMS,
    }
  })
  const distributionsRecord = projectRecord(record.distributions)
  assertNoSecretLikeFields(distributionsRecord, ['group', 'model', 'access_key'])

  function projectDistribution(
    value: unknown,
    expectedDimension: UsageDistributionDimension,
    expectedMetric: UsageDistributionMetric,
  ): UsageDistributionDto {
    const distributionRecord = projectRecord(value)
    assertNoSecretLikeFields(distributionRecord, ['dimension', 'metric', 'items', 'other'])
    const distributionDimension = projectEnum(distributionRecord.dimension, [
      'group',
      'model',
      'access_key',
    ] as const)
    const distributionMetric = projectEnum(distributionRecord.metric, [
      'requests',
      'tokens',
      'cost',
    ] as const)
    if (distributionDimension !== expectedDimension || distributionMetric !== expectedMetric) {
      invalidResponse()
    }
    const identities = new Set<string>()
    const distributionItems = projectArray(distributionRecord.items, (value) => {
      const item = projectRecord(value)
      const identityField =
        distributionDimension === 'group'
          ? 'group_id'
          : distributionDimension === 'access_key'
            ? 'access_key_id'
            : 'model'
      assertNoSecretLikeFields(item, [...distributionAggregateFields, identityField])
      const aggregate = projectUsageDistributionAggregate(
        Object.fromEntries(distributionAggregateFields.map((field) => [field, item[field]])),
      )
      if (distributionDimension === 'group') {
        const groupID = projectSafeInteger(item.group_id, { minimum: 1 })
        if (identities.has(String(groupID))) invalidResponse()
        identities.add(String(groupID))
        return { ...aggregate, group_id: groupID }
      }
      if (distributionDimension === 'access_key') {
        const accessKeyID = projectSafeInteger(item.access_key_id, { minimum: 1 })
        if (identities.has(String(accessKeyID))) invalidResponse()
        identities.add(String(accessKeyID))
        return { ...aggregate, access_key_id: accessKeyID }
      }
      const model = projectString(item.model, { allowEmpty: true })
      if (
        new TextEncoder().encode(model).length > 255 ||
        model !== model.trim() ||
        /[\p{Cc}]/u.test(model) ||
        identities.has(model)
      ) {
        invalidResponse()
      }
      identities.add(model)
      return { ...aggregate, model }
    })
    if (distributionItems.length > 5) invalidResponse()
    const distributionOther =
      distributionRecord.other === null
        ? null
        : projectUsageDistributionAggregate(distributionRecord.other)
    const summary = projectUsageAggregate(record.summary)
    const visibleAndOther = [
      ...distributionItems,
      ...(distributionOther === null ? [] : [distributionOther]),
    ]
    const distributedRequests = visibleAndOther.reduce(
      (total, item) => total + item.request_count,
      0,
    )
    const distributedCost = visibleAndOther.reduce(
      (total, item) => total + BigInt(item.estimated_cost_nano_usd),
      0n,
    )
    const distributedTokens = visibleAndOther.reduce((total, item) => total + item.total_tokens, 0)
    if (
      distributedRequests !== summary.request_count ||
      distributedTokens !== summary.total_tokens ||
      distributedCost !== BigInt(summary.estimated_cost_nano_usd)
    ) {
      invalidResponse()
    }
    return {
      dimension: distributionDimension,
      metric: distributionMetric,
      items: distributionItems,
      other: distributionOther,
    }
  }

  function projectDistributionMetricRecord(
    value: unknown,
    dimension: UsageDistributionDimension,
  ): Record<UsageDistributionMetric, UsageDistributionDto> {
    const metricRecord = projectRecord(value)
    assertNoSecretLikeFields(metricRecord, ['requests', 'tokens', 'cost'])
    return {
      requests: projectDistribution(metricRecord.requests, dimension, 'requests'),
      tokens: projectDistribution(metricRecord.tokens, dimension, 'tokens'),
      cost: projectDistribution(metricRecord.cost, dimension, 'cost'),
    }
  }

  const modelDistributions = projectDistributionMetricRecord(distributionsRecord.model, 'model')
  const groupDistributions = Object.prototype.hasOwnProperty.call(distributionsRecord, 'group')
    ? projectDistributionMetricRecord(distributionsRecord.group, 'group')
    : undefined
  const accessKeyDistributions = Object.prototype.hasOwnProperty.call(
    distributionsRecord,
    'access_key',
  )
    ? projectDistributionMetricRecord(distributionsRecord.access_key, 'access_key')
    : undefined
  const summary = projectUsageAggregate(record.summary)
  const breakdown = projectUsageBreakdown(record.breakdown)
  if (!sameUsageAggregate(breakdown.total, summary)) invalidResponse()

  return {
    range,
    granularity,
    bucket_width_ms: bucketWidthMs,
    from_ms: rangeFromMS,
    to_ms: rangeToMS,
    observed_at_ms: observedAtMS,
    summary,
    series,
    distributions: {
      ...(groupDistributions === undefined ? {} : { group: groupDistributions }),
      model: modelDistributions,
      ...(accessKeyDistributions === undefined ? {} : { access_key: accessKeyDistributions }),
    },
    collection_health: projectCollectionHealth(record.collection_health),
    breakdown,
  }
}

export function normalizeUsageFilters(filters: UsageFilters): UsageFilters {
  const breakdownSort = normalizeUsageBreakdownSort(filters.breakdown_sort)
  const result: UsageFilters = {
    range: filters.range,
    breakdown_page: normalizeUsagePage(filters.breakdown_page),
    breakdown_page_size: normalizeUsagePageSize(filters.breakdown_page_size),
    breakdown_sort: breakdownSort,
  }
  if (filters.group_id !== undefined) result.group_id = filters.group_id
  if (filters.channel_id !== undefined) result.channel_id = filters.channel_id
  if (filters.credential_id !== undefined) result.credential_id = filters.credential_id
  if (filters.upstream_model !== undefined) result.upstream_model = filters.upstream_model
  result.breakdown_sort_direction = normalizeUsageBreakdownSortDirection(
    filters.breakdown_sort_direction,
    breakdownSort,
  )
  return result
}

function normalizeUsagePage(value: unknown): number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : 1
}

function normalizeUsagePageSize(value: unknown): UsageBreakdownPageSize {
  return value === 50 || value === 100 ? value : 20
}

export function usageQueryIdentity(filters: UsageFilters) {
  return controlQueryKeys.usage.report(normalizeUsageFilters(filters))
}

export async function getUsageReport(
  client: ApiClient,
  filters: UsageFilters,
  signal?: AbortSignal,
): Promise<UsageReportDto> {
  const normalized = normalizeUsageFilters(filters)
  const breakdownSort = normalizeUsageBreakdownSort(normalized.breakdown_sort)
  const breakdownSortDirection = normalizeUsageBreakdownSortDirection(
    normalized.breakdown_sort_direction,
    breakdownSort,
  )
  const params = new URLSearchParams([['range', normalized.range]])
  if (normalized.group_id !== undefined) params.append('group_id', String(normalized.group_id))
  if (normalized.channel_id !== undefined) params.append('channel_id', normalized.channel_id)
  if (normalized.credential_id !== undefined) {
    params.append('credential_id', String(normalized.credential_id))
  }
  if (normalized.upstream_model !== undefined) {
    params.append('upstream_model', normalized.upstream_model)
  }
  params.append('breakdown_page', String(normalized.breakdown_page))
  params.append('breakdown_page_size', String(normalized.breakdown_page_size))
  params.append('breakdown_sort', breakdownSort)
  params.append('breakdown_sort_direction', breakdownSortDirection)

  const report = projectUsageReport(
    await client.request(`/api/usage?${params.toString()}`, { method: 'GET', signal }),
  )
  return report
}

export function usageQueryOptions(
  client: ApiClient,
  filters: MaybeRefOrGetter<UsageFilters>,
  intervalMs?: number,
) {
  return queryOptions({
    queryKey: computed(() => usageQueryIdentity(toValue(filters))),
    queryFn: ({ signal }) => getUsageReport(client, toValue(filters), signal),
    placeholderData: keepPreviousData,
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    ...(intervalMs !== undefined
      ? {
          refetchInterval: intervalMs,
          refetchIntervalInBackground: false,
          refetchOnWindowFocus: false,
        }
      : {}),
  })
}
