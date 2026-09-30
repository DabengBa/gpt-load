<script setup lang="ts">
import { useQuery } from '@tanstack/vue-query'
import { ArrowRight, CircleHelp, Info, Magnet, Search, TriangleAlert } from '@lucide/vue'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'

import { useApiClient } from '@shared/http/client-context'
import { useCollectionLoading } from '@/app/loading-state'
import { accessKeyOptionsQueryOptions } from '@/app/resources/access-keys'
import { listChannels, type ChannelDto } from '@/app/resources/channels'
import { controlQueryKeys } from '@/app/query-keys'
import { groupOptionsQueryOptions } from '@/app/resources/groups'
import {
  requestLogQueryOptions,
  type RequestLogFilters,
  type RequestLogItemDto,
  type RequestLogPageSize,
} from '@/app/resources/request-logs'
import { logsLocation } from '@/app/route-locations'
import LedgerRecordList from '@/components/collection/LedgerRecordList.vue'
import AsyncRefreshIndicator from '@/components/ui/AsyncRefreshIndicator.vue'
import AppTooltip from '@/components/ui/AppTooltip.vue'
import EmptyState from '@/components/ui/EmptyState.vue'
import IconButton from '@/components/ui/IconButton.vue'
import InlineFeedback from '@/components/ui/InlineFeedback.vue'
import OverflowTooltip from '@/components/ui/OverflowTooltip.vue'
import PaginationBar from '@/components/ui/PaginationBar.vue'
import QueryFeedback from '@/components/ui/QueryFeedback.vue'
import SkeletonSurface from '@/components/ui/SkeletonSurface.vue'
import StatusBadge from '@/components/ui/StatusBadge.vue'

import { formatEstimatedCost, formatISOInstant, formatLocalInstantWithSeconds } from '@/lib/format'
import { useAuthSession } from '@/features/auth/auth-session'
import type { AccessProtocol } from '@/api/control/types'

import {
  applyLogFilterDraft,
  createLogFilterDraft,
  defaultRequestLogFilters,
  parseAppliedLogFilterState,
  serializeAppliedLogFilters,
  validateLogFilterDraft,
  type LogFilterDraft,
  type LogFilterErrors,
} from './log-filters'
import { cacheHitRate } from '@/lib/cache-rate'
import { currentTimeZone } from '@/lib/time'
import {
  formatLogDuration,
  formatLogReasoning,
  formatLogTokenCount,
  hasRequestLogCache,
  reasoningBudgetSemantic,
  requestLogCostDisplayState,
  requestLogResponseTooltip,
  requestLogResponseTooltipVisible,
  requestLogUsageDisplayState,
} from './log-format'
import LogDetailDrawer from './LogDetailDrawer.vue'
import LogProtocolConversion from './LogProtocolConversion.vue'
import LogRouteIdentity from './LogRouteIdentity.vue'
import LogsFilterForm from './LogsFilterForm.vue'
import PricingModeIndicator from './PricingModeIndicator.vue'
import { isValidRequestLogAffinityKey } from './request-log-affinity'
import {
  logsMonitorQuery,
  parseLogsMonitorState,
  scopeAccessKeyLogFilters,
  type LogsMonitorState,
} from '../logs/logs-route'

const client = useApiClient()
const session = useAuthSession()
const route = useRoute()
const router = useRouter()
const { locale, t } = useI18n()
const logPageSizes = [20, 50, 100] as const
const protocolLabels: Record<AccessProtocol, string> = {
  'openai-completions': 'Completions',
  'openai-responses': 'Responses',
  'openai-images': 'Images',
  'openai-embeddings': 'Embeddings',
  rerank: 'Rerank',
  anthropic: 'Anthropic',
  gemini: 'Gemini',
}
const isAccessKey = computed(() => session.state.principalType === 'access_key')
const appliedFilterState = computed(() => parseAppliedLogFilterState(route.query))
const invalidAffinityKey = computed(() =>
  isAccessKey.value ? undefined : appliedFilterState.value.invalidAffinityKey,
)
const appliedFilters = computed(() => {
  const filters = appliedFilterState.value.filters
  return isAccessKey.value ? scopeAccessKeyLogFilters(filters) : filters
})
const routeState = computed(() => parseLogsMonitorState(route.query))
const selectedRequestID = computed(() => routeState.value.selectedRequestID)
const advancedOpen = computed(() => routeState.value.filtersOpen)
const detailClosing = ref(false)
const draft = ref(createLogFilterDraft(appliedFilters.value))
const filterErrors = ref<LogFilterErrors>(
  invalidAffinityKey.value !== undefined ? { affinity_key: 'monitor.logs.errors.affinityKey' } : {},
)
const paginationPending = ref(false)
const pageTransitionOrigin = ref<LogsMonitorState | null>(null)
const currentCursor = computed(() => routeState.value.cursorHistory.at(-1))
let detailFocusTimer: number | undefined
let pendingDetailNavigation: Promise<unknown> | undefined

const groupsQuery = useQuery(groupOptionsQueryOptions(client, () => !isAccessKey.value))
const channelsQuery = useQuery({
  queryKey: controlQueryKeys.channels.list(''),
  queryFn: ({ signal }) => listChannels(client, '', signal),
  enabled: computed(() => !isAccessKey.value),
  staleTime: 5 * 60 * 1_000,
})
const accessKeyOptionsQuery = useQuery(
  accessKeyOptionsQueryOptions(client, () => !isAccessKey.value),
)
const groupNames = computed<Record<number, string>>(() =>
  Object.fromEntries((groupsQuery.data.value ?? []).map((group) => [group.id, group.name])),
)
const groupProviderUrls = computed<Record<number, string>>(() =>
  Object.fromEntries(
    (groupsQuery.data.value ?? [])
      .filter((group) => group.provider_url !== null)
      .map((group) => [group.id, group.provider_url as string]),
  ),
)
const channelsByID = computed<Record<string, ChannelDto>>(() =>
  Object.fromEntries(
    (channelsQuery.data.value?.items ?? []).map((channel) => [channel.channel_id, channel]),
  ),
)
const logsQuery = useQuery({
  ...requestLogQueryOptions(client, appliedFilters, currentCursor),
  enabled: computed(() => invalidAffinityKey.value === undefined),
})
const logs = computed(() => logsQuery.data.value?.items ?? [])
const {
  initial: initialLoading,
  transition: collectionTransition,
  refreshing: collectionRefreshing,
  rows: skeletonRows,
} = useCollectionLoading(
  {
    pending: () => logsQuery.isPending.value,
    placeholder: () => logsQuery.isPlaceholderData.value,
    fetching: () => logsQuery.isFetching.value,
    hasData: () => logsQuery.data.value !== undefined,
    itemCount: () => logs.value.length,
  },
  { fallbackRows: 20 },
)
const logsRefreshing = computed(
  () =>
    collectionRefreshing.value ||
    (!isAccessKey.value && groupsQuery.data.value !== undefined && groupsQuery.isFetching.value) ||
    (!isAccessKey.value &&
      channelsQuery.data.value !== undefined &&
      channelsQuery.isFetching.value) ||
    (!isAccessKey.value &&
      accessKeyOptionsQuery.data.value !== undefined &&
      accessKeyOptionsQuery.isFetching.value),
)
const currentPage = computed(() => routeState.value.cursorHistory.length + 1)
const paginationBusy = computed(() => paginationPending.value || logsQuery.isFetching.value)
const filterSignature = computed(() =>
  JSON.stringify([serializeAppliedLogFilters(appliedFilters.value), invalidAffinityKey.value]),
)
const allAdvancedFilterKeys: readonly (keyof RequestLogFilters)[] = [
  'channel_id',
  'credential_id',
  'upstream_model',
  'model_consistency',
  'access_key_id',
  'request_id',
  'protocol',
  'operation',
  'stream',
  'final_status_code',
  'usage_state',
  'cost_state',
  'pricing_completeness',
  'cache_present',
  'attempt_status_code',
  'failure_category',
  'error_code',
  'retry_state',
  'retry_count_min',
  'retry_count_max',
  'first_response_min_ms',
  'first_response_max_ms',
  'duration_min_ms',
  'duration_max_ms',
  'input_tokens_min',
  'input_tokens_max',
  'output_tokens_min',
  'output_tokens_max',
  'cost_min_nano_usd',
  'cost_max_nano_usd',
  'affinity_key',
]
const accessKeyForbiddenFilterKeys = new Set<keyof RequestLogFilters>([
  'group_id',
  'channel_id',
  'credential_id',
  'upstream_model',
  'model_consistency',
  'access_key_id',
  'attempt_status_code',
  'failure_category',
  'error_code',
  'retry_state',
  'retry_count_min',
  'retry_count_max',
  'affinity_key',
])
const advancedFilterKeys = computed(() =>
  isAccessKey.value
    ? allAdvancedFilterKeys.filter((key) => !accessKeyForbiddenFilterKeys.has(key))
    : allAdvancedFilterKeys,
)
const advancedCount = computed(
  () => advancedFilterKeys.value.filter((key) => appliedFilters.value[key] !== undefined).length,
)
const hasNonTimeFilters = computed(() =>
  Object.keys(appliedFilters.value).some(
    (key) => key !== 'from_ms' && key !== 'to_ms' && key !== 'limit',
  ),
)
// chips 只承载快速表单上不可见的筛选（高级抽屉项）；时间/分组/模型/状态已显示在
// 可见控件里，不再重复回显。
const appliedChips = computed(() => {
  const filters = appliedFilters.value
  const values: Array<{ key: string; label: string }> = []
  for (const key of advancedFilterKeys.value) {
    const value = filters[key]
    if (value === undefined) continue
    values.push({ key, label: advancedChipLabel(key, value) })
  }
  return values
})

watch(
  () => selectedRequestID.value,
  () => {
    detailClosing.value = false
  },
)

watch(
  filterSignature,
  () => {
    draft.value = createLogFilterDraft(appliedFilters.value)
    filterErrors.value =
      invalidAffinityKey.value !== undefined
        ? { affinity_key: 'monitor.logs.errors.affinityKey' }
        : {}
    paginationPending.value = false
    pageTransitionOrigin.value = null
  },
  { immediate: true },
)

watch(
  () => logsQuery.dataUpdatedAt.value,
  (updatedAt, previousUpdatedAt) => {
    if (updatedAt <= 0 || updatedAt === previousUpdatedAt) return
    paginationPending.value = false
    pageTransitionOrigin.value = null
  },
)

watch(
  () => logsQuery.isError.value,
  (failed) => {
    if (!failed || pageTransitionOrigin.value === null) return
    const origin = pageTransitionOrigin.value
    paginationPending.value = false
    pageTransitionOrigin.value = null
    void router.replace(logsLocation(logsMonitorQuery(appliedFilters.value, origin)))
  },
)

function timestampLabel(value: number): string {
  return `${formatLocalInstantWithSeconds(value)} · ${currentTimeZone()}`
}

function cacheRateLabel(log: RequestLogItemDto): string {
  const rate = cacheHitRate(log.cache_read_tokens, log.input_tokens)
  return rate === null
    ? '—'
    : new Intl.NumberFormat(locale.value, {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 1,
      }).format(rate)
}

function advancedChipLabel(key: keyof RequestLogFilters, value: unknown): string {
  if (key === 'access_key_id') {
    const accessKey = accessKeyOptionsQuery.data.value?.find(({ id }) => id === value)
    return t('monitor.logs.filters.appliedAccessKey', {
      value: accessKey?.name ?? `#${value}`,
    })
  }
  if (key === 'channel_id') {
    const channel = channelsQuery.data.value?.items.find(({ channel_id }) => channel_id === value)
    return t('monitor.logs.filters.appliedChannel', {
      value: channel?.name ?? String(value),
    })
  }
  if (key === 'credential_id') {
    return t('monitor.logs.filters.appliedCredential', { value })
  }
  if (key === 'upstream_model') return t('monitor.logs.filters.appliedUpstreamModel', { value })
  if (key === 'model_consistency') {
    return t('monitor.logs.filters.appliedModelConsistency', {
      value: t(`monitor.logs.filters.modelConsistency.${String(value)}`),
    })
  }
  if (key === 'request_id') return t('monitor.logs.filters.appliedRequestId', { value })
  if (key === 'affinity_key') return t('monitor.logs.filters.appliedAffinityKey', { value })
  if (key === 'protocol') return String(value)
  if (key === 'operation') {
    return t('monitor.logs.filters.appliedOperation', {
      value: t(`monitor.logs.operation.${String(value)}`),
    })
  }
  if (key === 'failure_category') return t(`monitor.logs.failureCategory.${String(value)}`)
  if (key === 'retry_state') return t(`monitor.logs.filters.retryState.${String(value)}`)
  if (key === 'usage_state') {
    return t('monitor.logs.filters.appliedUsageState', {
      value: t(`monitor.logs.filters.usageState.${String(value)}`),
    })
  }
  if (key === 'cost_state') {
    return t('monitor.logs.filters.appliedCostState', {
      value: t(`monitor.logs.filters.costState.${String(value)}`),
    })
  }
  if (key === 'pricing_completeness') {
    return t('monitor.logs.filters.appliedCompleteness', {
      value: t(`monitor.logs.filters.completeness.${String(value)}`),
    })
  }
  const labelKeys: Partial<Record<keyof RequestLogFilters, string>> = {
    stream: 'stream',
    final_status_code: 'finalStatusCode',
    usage_state: 'usageStateLabel',
    cost_state: 'costStateLabel',
    pricing_completeness: 'completenessLabel',
    cache_present: 'cachePresent',
    channel_id: 'channel',
    credential_id: 'credential',
    attempt_status_code: 'attemptStatusCode',
    error_code: 'errorCode',
  }
  const rangeKey = key.replace(/_nano_usd$/u, '_usd')
  const label = labelKeys[key]
    ? t(`monitor.logs.filters.${labelKeys[key]}`)
    : t(`monitor.logs.filters.rangeFields.${rangeKey}`)
  const display =
    typeof value === 'boolean' ? t(value ? 'monitor.logs.yes' : 'monitor.logs.no') : value
  return `${label} ${String(display)}`
}

function updateDraftField(field: keyof LogFilterDraft, value: string): void {
  draft.value = { ...draft.value, [field]: value }
}

async function commitFilters(filters: RequestLogFilters): Promise<void> {
  if (pendingDetailNavigation) await pendingDetailNavigation
  if (isAccessKey.value) filters = scopeAccessKeyLogFilters(filters)
  const serialized = serializeAppliedLogFilters(filters)
  const nextSignature = JSON.stringify([serialized, undefined])
  draft.value = createLogFilterDraft(filters)
  filterErrors.value = {}

  if (
    nextSignature === filterSignature.value &&
    routeState.value.cursorHistory.length === 0 &&
    routeState.value.selectedRequestID === undefined &&
    !routeState.value.filtersOpen
  ) {
    await logsQuery.refetch()
    return
  }

  await router.push(logsLocation(logsMonitorQuery(filters)))
}

// 就地收窄而非跳转：排查时要看的是同一维度的其他请求，且目标可能已删。
async function filterByGroup(groupID: number): Promise<void> {
  await commitFilters({ ...appliedFilters.value, group_id: groupID })
}

async function filterByCredential(credentialID: number): Promise<void> {
  await commitFilters({ ...appliedFilters.value, credential_id: credentialID })
}

async function filterByClientModel(clientModel: string): Promise<void> {
  await commitFilters({ ...appliedFilters.value, client_model: clientModel })
}

async function filterByAffinityKey(affinityKey: string | null): Promise<void> {
  if (isAccessKey.value || affinityKey === null || !isValidRequestLogAffinityKey(affinityKey))
    return
  await commitFilters({ ...appliedFilters.value, affinity_key: affinityKey })
}

async function applyFilters(): Promise<void> {
  const errors = validateLogFilterDraft(draft.value)
  if (invalidAffinityKey.value !== undefined) {
    errors.affinity_key = 'monitor.logs.errors.affinityKey'
  }
  filterErrors.value = errors
  if (Object.keys(errors).length > 0) return

  await commitFilters({
    ...applyLogFilterDraft(draft.value),
    limit: appliedFilters.value.limit ?? 20,
  })
}

async function resetFilters(): Promise<void> {
  await commitFilters({
    ...defaultRequestLogFilters(),
    limit: appliedFilters.value.limit ?? 20,
  })
}

function setPageSize(pageSize: RequestLogPageSize): void {
  if (paginationBusy.value) return
  void commitFilters({ ...appliedFilters.value, limit: pageSize })
}

async function removeFilter(key: string): Promise<void> {
  const filters = { ...appliedFilters.value }
  delete filters[key as keyof RequestLogFilters]
  await commitFilters(filters)
}

function nextPage(): void {
  if (paginationBusy.value) return
  const cursor = logsQuery.data.value?.next_cursor
  if (!cursor || cursor === currentCursor.value) return
  if (routeState.value.cursorHistory.includes(cursor)) return
  pageTransitionOrigin.value = {
    ...routeState.value,
    cursorHistory: [...routeState.value.cursorHistory],
  }
  paginationPending.value = true
  void router.push(
    logsLocation(
      logsMonitorQuery(appliedFilters.value, {
        filtersOpen: false,
        cursorHistory: [...routeState.value.cursorHistory, cursor],
      }),
    ),
  )
}

function previousPage(): void {
  if (paginationBusy.value || routeState.value.cursorHistory.length === 0) return
  pageTransitionOrigin.value = {
    ...routeState.value,
    cursorHistory: [...routeState.value.cursorHistory],
  }
  paginationPending.value = true
  void router.push(
    logsLocation(
      logsMonitorQuery(appliedFilters.value, {
        filtersOpen: false,
        cursorHistory: routeState.value.cursorHistory.slice(0, -1),
      }),
    ),
  )
}

function setAdvancedOpen(open: boolean): void {
  void router.push(
    logsLocation(
      logsMonitorQuery(appliedFilters.value, {
        ...routeState.value,
        filtersOpen: open,
        selectedRequestID: undefined,
      }),
    ),
  )
}

async function setDetailOpen(requestID: string | undefined, open: boolean): Promise<void> {
  const closingID = selectedRequestID.value
  detailClosing.value = !open
  const navigation = router.push(
    logsLocation(
      logsMonitorQuery(appliedFilters.value, {
        ...routeState.value,
        filtersOpen: false,
        selectedRequestID: open ? requestID : undefined,
      }),
    ),
  )
  pendingDetailNavigation = navigation
  try {
    await navigation
  } finally {
    if (pendingDetailNavigation === navigation) pendingDetailNavigation = undefined
  }
  if (open || !closingID) return
  window.clearTimeout(detailFocusTimer)
  detailFocusTimer = window.setTimeout(() => {
    if (document.activeElement && document.activeElement !== document.body) return
    document.getElementById(`log-details-${closingID}`)?.focus()
  }, 30)
}

function affinityKeyFilterable(log: RequestLogItemDto): boolean {
  return !isAccessKey.value && log.affinity_key !== null
}

// 分组名靠 options 反查：查询就绪后仍找不到，才能断定分组已被删除。
function groupDeleted(log: RequestLogItemDto): boolean {
  return log.group_id !== null && groupsQuery.isSuccess.value && groupName(log) === null
}

function groupName(log: RequestLogItemDto): string | null {
  if (log.group_id === null) return null
  const group = groupsQuery.data.value?.find(({ id }) => id === log.group_id)
  return group?.name ?? null
}

function channelDefinition(log: RequestLogItemDto): ChannelDto | null {
  if (log.channel_id === null) return null
  return channelsByID.value[log.channel_id] ?? null
}

function responseLabel(log: RequestLogItemDto): string {
  if (log.status === 'success') return t('monitor.logs.response.normal')
  if (log.status === 'error') {
    return log.stream && log.status_code === 200
      ? t('monitor.logs.response.streamError')
      : t('monitor.logs.response.errorWithCode', { code: log.status_code })
  }
  return t(`monitor.logs.status.${log.status}`)
}

// 状态行只在承载新信息时出现：HTTP 码仅当徽标未展示它（流式错误、或非 200 的
// 非常态码）才补；尝试次数 >1 与 error_code 追加在同一行，避免与徽标重复。
function responseMeta(log: RequestLogItemDto): string {
  const parts: string[] = []
  const badgeCarriesCode = log.status === 'error' && !(log.stream && log.status_code === 200)
  if (!badgeCarriesCode && (log.status === 'error' || log.status_code !== 200)) {
    parts.push(t('monitor.logs.response.httpStatus', { code: log.status_code }))
  }
  if (log.attempt_count > 1) {
    parts.push(t('monitor.logs.attemptCount', { count: log.attempt_count }))
  }
  if (log.error_code) parts.push(log.error_code)
  return parts.join(' · ')
}

function statusTone(
  status: RequestLogItemDto['status'],
): 'success' | 'danger' | 'warning' | 'neutral' {
  if (status === 'success') return 'success'
  if (status === 'error') return 'danger'
  if (status === 'incomplete') return 'warning'
  return 'neutral'
}

// 列表首屏只揭示最终状态、尝试次数和关键原因；供应商原始证据仍由详情抽屉与
// debug capture 承担，提示里不出现任何 raw body/headers。
// 翻译函数只暴露纯函数需要的形状，取值规则本身在 log-format 内固定并已测试。
function translateLogMessage(key: string, named?: Record<string, string | number>): string {
  return named ? t(key, named) : t(key)
}

function responseTooltipVisible(log: RequestLogItemDto): boolean {
  return requestLogResponseTooltipVisible(log)
}

function responseTooltip(log: RequestLogItemDto): string {
  return requestLogResponseTooltip(log, translateLogMessage)
}

function modelConsistencyTooltip(log: RequestLogItemDto): string {
  const key =
    log.model_consistency === 'mismatch'
      ? 'monitor.logs.modelConsistency.mismatchTooltip'
      : 'monitor.logs.modelConsistency.unknownTooltip'
  return t(key, {
    upstream: log.upstream_model ?? '—',
    reported: log.upstream_reported_model ?? t('monitor.logs.modelConsistency.notObserved'),
  })
}

function modelConsistencyLabel(log: RequestLogItemDto): string {
  return t(
    log.model_consistency === 'mismatch'
      ? 'monitor.logs.modelConsistency.mismatchLabel'
      : 'monitor.logs.modelConsistency.unknownLabel',
  )
}

// showAffinityObservation 保留现有响应单元格提示槽位，仅当存在真实的亲和或
// 连续性观测时才显示它。零值（source "none" / state "no_signal"）保持隐藏，
// 因此普通行不受影响。
function showAffinityObservation(log: RequestLogItemDto): boolean {
  return (
    log.affinity_hit ||
    log.continuity_hit ||
    log.affinity_source !== 'none' ||
    log.affinity_state !== 'no_signal'
  )
}

// affinityTooltip 将 bounded source/state 对渲染为可读文本。它
// 永远不会接收或展示原始 prompt_cache_key、派生 key 或 HMAC 输入。
function affinityTooltip(log: RequestLogItemDto): string {
  const source = t(`monitor.logs.affinitySource.${log.affinity_source}`)
  const state = t(`monitor.logs.affinityState.${log.affinity_state}`)
  return [
    t('monitor.logs.affinitySourceLabel', { source }),
    t('monitor.logs.affinityStateLabel', { state }),
    log.continuity_hit ? t('monitor.logs.continuityHit') : '',
    log.affinity_hit ? t('monitor.logs.drawer.affinity') : '',
  ]
    .filter(Boolean)
    .join('\n')
}

function reasoningLabel(log: RequestLogItemDto): string {
  if (log.reasoning === null) return ''
  if (
    log.reasoning.mode === 'disabled' ||
    log.reasoning.effort === 'none' ||
    (log.reasoning.budget_tokens !== null &&
      reasoningBudgetSemantic(log.reasoning.budget_tokens) === 'disabled')
  ) {
    return t('monitor.logs.reasoning.compact', {
      value: 'disabled',
    })
  }
  const value = formatLogReasoning(log, locale.value)
  return t('monitor.logs.reasoning.compact', {
    value,
  })
}

function cacheTooltip(log: RequestLogItemDto): string {
  const details = [
    [t('monitor.logs.tokens.cacheRead'), log.cache_read_tokens],
    [t('monitor.logs.tokens.cacheWrite5m'), log.cache_write_5m_tokens],
    [t('monitor.logs.tokens.cacheWrite1h'), log.cache_write_1h_tokens],
    [t('monitor.logs.tokens.cacheWrite'), log.cache_write_unknown_tokens],
  ]
    .filter(([, value]) => value !== '0')
    .map(([label, value]) => `${label} ${formatLogTokenCount(value, locale.value)}`)
  details.push(`${t('monitor.logs.tokens.cacheHitRate')} ${cacheRateLabel(log)}`)
  details.push(t('monitor.logs.tokens.cacheRecordedHint'))
  return details.join('\n')
}

function timingPrimary(log: RequestLogItemDto): string {
  if (!log.stream || log.first_response_ms === null) return formatLogDuration(log.duration_ms)
  return `${formatLogDuration(log.first_response_ms)} / ${formatLogDuration(log.duration_ms)}`
}

function timingClass(log: RequestLogItemDto): string {
  return log.feedback_status === 'slow' || log.feedback_status === 'faulty'
    ? `logs-list__timing--${log.feedback_status}`
    : ''
}

function costLabel(log: RequestLogItemDto): string {
  const state = requestLogCostDisplayState(log)
  if (state === 'complete') {
    return formatEstimatedCost(log.estimated_cost_nano_usd, locale.value)
  }
  return '—'
}
</script>

<template>
  <div class="logs-tab">
    <LogsFilterForm
      :draft="draft"
      :errors="filterErrors"
      :groups="groupsQuery.data.value ?? []"
      :channels="channelsQuery.data.value?.items ?? []"
      :access-keys="accessKeyOptionsQuery.data.value ?? []"
      :groups-failed="groupsQuery.isError.value"
      :channels-failed="channelsQuery.isError.value"
      :access-keys-failed="accessKeyOptionsQuery.isError.value"
      :applied-chips="appliedChips"
      :advanced-count="advancedCount"
      :advanced-open="advancedOpen"
      :self-scoped="isAccessKey"
      @update:advanced-open="setAdvancedOpen"
      @update-field="updateDraftField"
      @remove-filter="removeFilter"
      @apply="applyFilters"
      @reset="resetFilters"
    />

    <InlineFeedback
      v-if="
        !isAccessKey &&
        (groupsQuery.isError.value ||
          channelsQuery.isError.value ||
          accessKeyOptionsQuery.isError.value)
      "
      tone="warning"
    >
      {{ t('monitor.logs.options.partialFailed') }}
    </InlineFeedback>

    <AsyncRefreshIndicator :active="logsRefreshing" :label="t('monitor.logs.loading')" />

    <SkeletonSurface
      v-if="invalidAffinityKey === undefined && (logsQuery.isPending.value || initialLoading)"
      variant="collection"
      :rows="appliedFilters.limit ?? 20"
      :columns="isAccessKey ? 7 : 9"
      row-height="52px"
      mobile-row-height="150px"
      :concealed="!initialLoading"
      :label="t('monitor.logs.loading')"
    />
    <QueryFeedback
      v-else-if="
        invalidAffinityKey === undefined && logsQuery.isError.value && !logsQuery.data.value
      "
      state="error"
      :message="t('monitor.logs.loadFailed')"
      :retry-label="t('common.retry')"
      @retry="logsQuery.refetch()"
    />
    <template v-else-if="invalidAffinityKey === undefined && logsQuery.data.value">
      <QueryFeedback
        v-if="logsQuery.isError.value"
        state="stale"
        :message="t('monitor.logs.stale')"
        :retry-label="t('common.retry')"
        @retry="logsQuery.refetch()"
      />
      <SkeletonSurface
        v-if="collectionTransition"
        variant="collection"
        :rows="skeletonRows"
        :columns="isAccessKey ? 7 : 9"
        row-height="52px"
        mobile-row-height="150px"
        :label="t('monitor.logs.loading')"
      />
      <template v-else>
        <p v-if="logs.length" class="logs-list__summary" data-testid="logs-result-summary">
          {{ t('monitor.logs.resultSummary', { count: logs.length }) }}
        </p>
        <LedgerRecordList
          v-if="logs.length"
          :grid-class="isAccessKey ? 'logs-list logs-list--scoped' : 'logs-list'"
          :label="t('monitor.logs.caption')"
          :row-count="logs.length + 1"
          :scroll-hint="t('monitor.scrollHint')"
        >
          <template #header>
            <span role="columnheader" :aria-label="t('monitor.logs.columns.timeNewestFirst')">{{
              t('monitor.logs.columns.time')
            }}</span>
            <span v-if="!isAccessKey" role="columnheader">{{
              t('monitor.logs.columns.affinityKey')
            }}</span>
            <span v-if="!isAccessKey" role="columnheader">{{
              t('monitor.logs.columns.route')
            }}</span>
            <span role="columnheader">{{ t('monitor.logs.columns.modelProtocol') }}</span>
            <span role="columnheader">{{ t('monitor.logs.columns.response') }}</span>
            <span role="columnheader">{{ t('monitor.logs.columns.cost') }}</span>
            <span
              class="logs-list__tokens-header"
              role="columnheader"
              :aria-label="t('monitor.logs.columns.tokensDetail')"
            >
              {{ t('monitor.logs.columns.tokens') }}
            </span>
            <span role="columnheader" :aria-label="t('monitor.logs.columns.timingDetail')">{{
              t('monitor.logs.columns.timing')
            }}</span>
            <span role="columnheader">{{ t('monitor.logs.columns.actions') }}</span>
          </template>

          <article
            v-for="(log, index) in logs"
            :key="log.request_id"
            class="ledger-record-list__record logs-list__record"
            role="row"
            :aria-rowindex="index + 2"
          >
            <div
              class="ledger-record-list__cell logs-list__cell logs-list__time"
              role="cell"
              :data-label="t('monitor.logs.columns.time')"
            >
              <small>{{ formatLocalInstantWithSeconds(log.completed_at_ms).slice(0, 10) }}</small>
              <AppTooltip :content="timestampLabel(log.completed_at_ms)">
                <time
                  :datetime="formatISOInstant(log.completed_at_ms)"
                  :title="timestampLabel(log.completed_at_ms)"
                  :aria-label="timestampLabel(log.completed_at_ms)"
                  tabindex="0"
                >
                  {{ formatLocalInstantWithSeconds(log.completed_at_ms).slice(11) }}
                </time>
              </AppTooltip>
            </div>
            <div
              v-if="!isAccessKey"
              class="ledger-record-list__cell logs-list__cell logs-list__affinity-key-cell"
              role="cell"
              :data-label="t('monitor.logs.columns.affinityKey')"
            >
              <AppTooltip v-if="affinityKeyFilterable(log)" :content="log.affinity_key ?? ''">
                <button
                  type="button"
                  class="logs-list__affinity-key filterable-value"
                  :aria-label="t('monitor.logs.filterAffinityKey', { value: log.affinity_key })"
                  :title="log.affinity_key ?? ''"
                  data-testid="logs-affinity-key-filter"
                  @click="filterByAffinityKey(log.affinity_key)"
                >
                  …{{ log.affinity_key?.slice(-6) ?? '' }}
                </button>
              </AppTooltip>
              <code v-else class="logs-list__affinity-key">—</code>
            </div>
            <div
              v-if="!isAccessKey"
              class="ledger-record-list__cell logs-list__cell"
              role="cell"
              :data-label="t('monitor.logs.columns.route')"
            >
              <LogRouteIdentity
                :group-id="log.group_id"
                :group-name="groupName(log)"
                :provider-url="
                  log.group_id === null ? null : (groupProviderUrls[log.group_id] ?? null)
                "
                :channel-id="log.channel_id"
                :channel="channelDefinition(log)"
                :credential-id="log.credential_id"
                :credential-name="log.credential_name"
                :group-deleted="groupDeleted(log)"
                :credential-deleted="log.credential_id !== null && log.credential_name === ''"
                filterable
                @filter-group="filterByGroup"
                @filter-credential="filterByCredential"
              />
            </div>
            <div
              class="ledger-record-list__cell logs-list__cell"
              role="cell"
              :data-label="t('monitor.logs.columns.modelProtocol')"
            >
              <span class="logs-list__inline">
                <OverflowTooltip
                  v-if="log.client_model"
                  as="button"
                  type="button"
                  class="logs-list__model filterable-value"
                  :content="log.client_model"
                  :aria-label="t('monitor.logs.filterModel', { name: log.client_model })"
                  @click="filterByClientModel(log.client_model)"
                >
                  {{ log.client_model }}
                </OverflowTooltip>
                <code v-else class="logs-list__model">—</code>
                <OverflowTooltip
                  v-if="log.upstream_model && log.upstream_model !== log.client_model"
                  as="span"
                  class="logs-list__model-mapping"
                  :content="log.upstream_model"
                >
                  -&gt;{{ log.upstream_model }}
                </OverflowTooltip>

                <AppTooltip
                  v-if="log.model_consistency === 'unknown' || log.model_consistency === 'mismatch'"
                  :content="modelConsistencyTooltip(log)"
                >
                  <button
                    type="button"
                    class="logs-list__hint logs-list__model-consistency"
                    :class="`logs-list__model-consistency--${log.model_consistency}`"
                    :aria-label="modelConsistencyLabel(log)"
                  >
                    <TriangleAlert
                      v-if="log.model_consistency === 'mismatch'"
                      :size="14"
                      aria-hidden="true"
                    />
                    <CircleHelp v-else :size="13" aria-hidden="true" />
                  </button>
                </AppTooltip>
              </span>
              <span class="logs-list__protocol-line">
                <AppTooltip :content="log.protocol">
                  <small class="logs-list__protocol" tabindex="0">{{
                    protocolLabels[log.protocol]
                  }}</small>
                </AppTooltip>
                <OverflowTooltip
                  v-if="reasoningLabel(log)"
                  as="small"
                  class="logs-list__reasoning"
                  :content="reasoningLabel(log)"
                >
                  {{ reasoningLabel(log) }}
                </OverflowTooltip>
                <LogProtocolConversion
                  :mode="log.route_mode"
                  :client-protocol="log.protocol"
                  :upstream-protocol="log.upstream_protocol"
                />
              </span>
            </div>
            <div
              class="ledger-record-list__cell logs-list__cell"
              role="cell"
              :data-label="t('monitor.logs.columns.response')"
            >
              <div class="logs-list__response-primary">
                <AppTooltip v-if="responseTooltipVisible(log)" :content="responseTooltip(log)">
                  <span data-testid="request-outcome"
                    ><StatusBadge :tone="statusTone(log.status)" size="compact">{{
                      responseLabel(log)
                    }}</StatusBadge></span
                  >
                </AppTooltip>
                <OverflowTooltip
                  v-else
                  as="span"
                  data-testid="request-outcome"
                  :content="responseLabel(log)"
                >
                  <StatusBadge :tone="statusTone(log.status)" size="compact">
                    {{ responseLabel(log) }}
                  </StatusBadge>
                </OverflowTooltip>

                <AppTooltip v-if="showAffinityObservation(log)" :content="affinityTooltip(log)">
                  <span
                    class="logs-list__hint logs-list__affinity"
                    tabindex="0"
                    :aria-label="affinityTooltip(log)"
                  >
                    <Magnet v-if="log.affinity_hit" :size="13" aria-hidden="true" />
                    <ArrowRight v-else-if="log.continuity_hit" :size="13" aria-hidden="true" />
                    <Info v-else :size="13" aria-hidden="true" />
                  </span>
                </AppTooltip>
              </div>
              <OverflowTooltip
                v-if="responseMeta(log)"
                as="small"
                class="logs-list__response-meta"
                :content="responseTooltip(log)"
              >
                {{ responseMeta(log) }}
              </OverflowTooltip>
            </div>
            <div
              class="ledger-record-list__cell logs-list__cell"
              role="cell"
              :data-label="t('monitor.logs.columns.cost')"
            >
              <span class="logs-list__cost-line">
                <OverflowTooltip
                  as="span"
                  :content="costLabel(log)"
                  :class="{
                    'logs-list__state--warning': requestLogCostDisplayState(log) !== 'complete',
                  }"
                >
                  {{ costLabel(log) }}
                </OverflowTooltip>
                <PricingModeIndicator
                  :mode="log.pricing_mode"
                  :context-threshold-tokens="log.context_threshold_tokens"
                />
              </span>
            </div>
            <div
              class="ledger-record-list__cell logs-list__cell"
              role="cell"
              :data-label="t('monitor.logs.columns.tokens')"
            >
              <div v-if="requestLogUsageDisplayState(log) === 'reported'" class="logs-list__tokens">
                <span class="logs-list__token-values">
                  <OverflowTooltip
                    as="span"
                    class="logs-list__token-line"
                    :content="`${t('monitor.logs.tokens.input')}: ${formatLogTokenCount(log.input_tokens, locale)}\n${t('monitor.logs.tokens.output')}: ${formatLogTokenCount(log.output_tokens, locale)}`"
                  >
                    <span
                      >{{ formatLogTokenCount(log.input_tokens, locale) }}
                      <small>{{ t('monitor.logs.tokens.input') }}</small></span
                    >
                    <span class="logs-list__token-separator" aria-hidden="true">·</span>
                    <span
                      >{{ formatLogTokenCount(log.output_tokens, locale) }}
                      <small>{{ t('monitor.logs.tokens.output') }}</small></span
                    >
                  </OverflowTooltip>
                  <AppTooltip
                    v-if="log.usage_state === 'partial'"
                    :content="t('monitor.logs.tokens.partial')"
                  >
                    <button
                      type="button"
                      class="logs-list__hint"
                      :aria-label="t('monitor.logs.tokens.partial')"
                    >
                      <CircleHelp :size="13" aria-hidden="true" />
                    </button>
                  </AppTooltip>
                </span>
                <AppTooltip
                  v-if="log.usage_state === 'complete' || hasRequestLogCache(log)"
                  :content="cacheTooltip(log)"
                >
                  <button
                    type="button"
                    class="logs-list__hint logs-list__cache-rate"
                    :aria-label="`${t('monitor.logs.tokens.cacheHitRate')} ${cacheRateLabel(log)} · ${t('monitor.logs.tokens.cacheDetails')}`"
                  >
                    {{ t('monitor.logs.tokens.cacheHitRate') }} {{ cacheRateLabel(log) }}
                  </button>
                </AppTooltip>
                <small v-else class="logs-list__cache-state">{{
                  t('monitor.logs.tokens.cacheUnavailable')
                }}</small>
              </div>
              <template v-else>
                <span class="logs-list__state--warning">—</span>
                <small class="logs-list__cache-state">{{
                  t(
                    log.usage_state === 'not_applicable'
                      ? 'monitor.logs.filters.usageState.not_applicable'
                      : 'monitor.logs.tokens.cacheUnavailable',
                  )
                }}</small>
              </template>
            </div>
            <div
              class="ledger-record-list__cell logs-list__cell"
              role="cell"
              :data-label="t('monitor.logs.columns.timing')"
            >
              <OverflowTooltip as="span" :content="timingPrimary(log)">
                <span :class="timingClass(log)">
                  <template v-if="log.stream && log.first_response_ms !== null">
                    {{ formatLogDuration(log.first_response_ms) }}
                    <span aria-hidden="true"> / </span>{{ formatLogDuration(log.duration_ms) }}
                  </template>
                  <template v-else>{{ formatLogDuration(log.duration_ms) }}</template>
                </span>
              </OverflowTooltip>
            </div>
            <div
              class="ledger-record-list__cell logs-list__action"
              role="cell"
              :data-label="t('monitor.logs.columns.actions')"
            >
              <AppTooltip :content="t('monitor.logs.details')">
                <IconButton
                  :id="`log-details-${log.request_id}`"
                  variant="ghost"
                  size="compact"
                  :label="t('monitor.logs.details')"
                  @click="setDetailOpen(log.request_id, true)"
                >
                  <ArrowRight :size="16" aria-hidden="true" />
                </IconButton>
              </AppTooltip>
            </div>
          </article>
        </LedgerRecordList>
        <EmptyState
          v-else
          variant="ledger"
          :title="
            t(hasNonTimeFilters ? 'monitor.logs.empty.filteredTitle' : 'monitor.logs.empty.title')
          "
          :description="
            t(
              hasNonTimeFilters
                ? 'monitor.logs.empty.filteredDescription'
                : 'monitor.logs.empty.description',
            )
          "
        >
          <template #icon><Search :size="20" /></template>
        </EmptyState>
        <PaginationBar
          cursor
          :page="currentPage"
          :page-size="appliedFilters.limit ?? 20"
          :page-sizes="logPageSizes"
          show-page-size
          appearance="detail"
          :has-previous="routeState.cursorHistory.length > 0"
          :has-next="Boolean(logsQuery.data.value?.next_cursor)"
          :pending="paginationBusy"
          @previous="previousPage"
          @next="nextPage"
          @update:page-size="setPageSize"
        />
      </template>
    </template>

    <LogDetailDrawer
      :open="Boolean(selectedRequestID) && !detailClosing && invalidAffinityKey === undefined"
      :request-id="invalidAffinityKey === undefined ? selectedRequestID : undefined"
      :self-scoped="isAccessKey"
      :group-names="groupNames"
      :groups-loaded="groupsQuery.isSuccess.value"
      :provider-urls="groupProviderUrls"
      :channels="channelsByID"
      @update:open="setDetailOpen(undefined, $event)"
    />
  </div>
</template>

<style scoped>
.logs-tab {
  display: grid;
  min-width: 0;
  gap: 14px;
}

.logs-list {
  --ledger-record-list-grid: 80px 80px minmax(160px, 1fr) minmax(190px, 1.2fr) 88px
    minmax(76px, 0.42fr) minmax(150px, 0.7fr) 100px 48px;
  --ledger-record-list-column-gap: 12px;
  --ledger-record-list-record-min-height: 52px;
  --ledger-record-list-record-padding: 8px 0;
}

.logs-list--scoped {
  --ledger-record-list-grid: 80px minmax(210px, 1.2fr) 96px minmax(76px, 0.42fr)
    minmax(150px, 0.7fr) 100px 48px;
}

.logs-list__cell {
  display: grid;
  min-width: 0;
  gap: 4px;
  color: var(--color-text);
  font-size: var(--text-sm);
  font-weight: 400;
}

.logs-list__cell > span,
.logs-list__cell code,
.logs-list__cell .filterable-value {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.logs-list__cell small {
  overflow: hidden;
  color: var(--color-text-muted);
  font-size: var(--text-label-xs);
  font-weight: 400;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.logs-list__summary {
  margin: 0;
  color: var(--color-text-muted);
  font-size: var(--text-sm);
}

.logs-list__time {
  font-family: var(--font-mono);
  font-size: var(--text-label-xs);
}

.logs-list__affinity-key {
  display: block;
  min-width: 0;
  overflow: hidden;
  font-family: var(--font-mono);
  font-size: var(--text-label-xs);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.logs-list__affinity-key-cell {
  align-content: center;
}

.logs-list__affinity-key.filterable-value {
  width: 100%;
  border: 0;
  background: transparent;
  color: var(--color-action);
  padding: 0;
  text-align: left;
}

.logs-list__affinity-key.filterable-value:hover {
  color: var(--color-text);
  text-decoration: underline;
  text-underline-offset: 3px;
}

.logs-list__inline,
.logs-list__tokens {
  min-width: 0;
}

.logs-list__response-primary {
  display: flex;
  min-width: 0;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
}

.logs-list__inline {
  display: flex;
  align-items: center;
  gap: 0;
}

.logs-list__timing--slow,
.log-detail__timing--slow {
  color: var(--color-warning);
}

.logs-list__timing--faulty,
.log-detail__timing--faulty {
  color: var(--color-danger);
}

.logs-list__protocol {
  min-width: 0;
}

.logs-list__protocol-line {
  display: flex;
  align-items: center;
  gap: 8px;
}

.logs-list__cost-line {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 5px;
}

.logs-list__cost-line > :first-child {
  min-width: 0;
}

.logs-list__tokens-header {
  text-align: left;
}

.logs-list__tokens {
  display: grid;
  gap: 4px;
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}

.logs-list__model {
  flex: 1 1 0;
  font-family: var(--font-mono);
}

.logs-list__model-mapping {
  flex: 1 1 0;
  min-width: 0;
  overflow: hidden;
  color: var(--color-text-muted);
  font-family: var(--font-mono);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.logs-list__reasoning {
  flex: 0 0 auto;
}

.logs-list__inline > .logs-list__hint {
  margin-left: 5px;
}

.logs-list__token-line {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
  align-items: center;
  gap: 5px;
  white-space: nowrap;
}

.logs-list__token-values {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 5px;
}

.logs-list__token-separator {
  color: var(--color-text-muted);
  margin: 0 2px;
}

.logs-list__hint {
  display: inline-flex;
  width: 24px;
  height: 24px;
  flex: 0 0 24px;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: var(--radius-tag);
  background: transparent;
  color: var(--color-text-muted);
  padding: 0;
  cursor: help;
}

.logs-list__cache-rate {
  width: auto;
  max-width: 100%;
  min-height: 24px;
  flex-basis: auto;
  justify-content: flex-start;
  gap: 3px;
  font: inherit;
  font-size: var(--text-label-xs);
}

.logs-list__hint:hover {
  background: var(--color-surface-sunken);
  color: var(--color-text);
}

.logs-list__model-consistency--mismatch,
.logs-list__model-consistency--mismatch:hover {
  color: var(--color-warning);
}

.logs-list__state--warning {
  color: var(--color-warning);
}

.logs-list__action {
  display: flex;
  justify-self: stretch;
  justify-content: flex-end;
}

.logs-list__action :deep(.icon-button) {
  color: var(--color-text-muted);
}

.logs-list__hint:focus-visible,
.logs-list__protocol:focus-visible,
.logs-list__time time:focus-visible {
  outline: 2px solid var(--color-focus);
  outline-offset: -2px;
}

.logs-list :deep(.ledger-record-list__header) {
  color: var(--color-text-muted);
}

.logs-tab :deep(.status-badge) {
  width: max-content;
  font-weight: 400;
}

@media (max-width: 1080px) {
  .logs-list {
    --ledger-record-list-column-gap: 10px;
    --ledger-record-list-grid: 80px 80px minmax(160px, 1fr) minmax(190px, 1.2fr) 92px 76px
      minmax(150px, 0.7fr) 96px 48px;
  }

  .logs-list--scoped {
    --ledger-record-list-grid: 80px minmax(190px, 1.2fr) 92px 76px minmax(150px, 0.7fr) 96px 48px;
  }
}

@media (max-width: 860px) {
  .logs-list {
    --ledger-record-list-card-grid: minmax(104px, 0.42fr) minmax(0, 1.58fr);
  }

  .logs-list__record {
    gap: 10px 14px;
  }

  .logs-list__cell,
  .logs-list__action {
    display: grid;
    grid-column: 1 / -1;
    grid-template-columns: subgrid;
    align-items: start;
  }

  .logs-list__cell::before,
  .logs-list__action::before {
    content: attr(data-label);
    color: var(--color-text-muted);
    font-size: var(--text-label-xs);
  }

  .logs-list__cell > *,
  .logs-list__action > * {
    grid-column: 2;
  }

  .logs-list__action {
    justify-self: stretch;
  }

  .logs-list__action :deep(.icon-button) {
    justify-self: end;
  }
}
</style>
