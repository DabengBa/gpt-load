<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import type { GroupOptionDto } from '@/api/control/types'
import type { ChannelDto } from '@/app/resources/channels'
import type {
  UsageAggregateDto,
  UsageBreakdownDto,
  UsageBreakdownRowDto,
} from '@/app/resources/usage'
import DataTable from '@/components/ui/DataTable.vue'
import { formatEstimatedCost, formatInteger, formatPercent, formatTokens } from '@/lib/format'

const props = defineProps<{
  breakdown: UsageBreakdownDto
  groups: GroupOptionDto[]
  channels: ChannelDto[]
}>()

const { locale, t } = useI18n()
type SortKey =
  'model' | 'group' | 'channel' | 'success_rate' | 'average_latency' | keyof UsageAggregateDto

type SortDirection = 'asc' | 'desc'

const sortKey = ref<SortKey>('estimated_cost_nano_usd')
const sortDirection = ref<SortDirection>('desc')

const isAdmin = computed(() => props.breakdown.scope === 'admin')
const sortedRows = computed(() =>
  [...props.breakdown.rows].sort((left, right) => {
    const result = compareRows(left, right, sortKey.value)
    if (result !== 0) return sortDirection.value === 'asc' ? result : -result
    return rowKey(left).localeCompare(rowKey(right))
  }),
)

function rowKey(row: UsageBreakdownRowDto): string {
  return JSON.stringify([
    props.breakdown.scope,
    row.model,
    row.group_id ?? null,
    row.channel_id ?? null,
  ])
}

function groupName(row: UsageBreakdownRowDto): string {
  if (row.group_id === undefined) return '—'
  return (
    props.groups.find((group) => group.id === row.group_id)?.name ??
    t('monitor.usage.filters.deletedOrUnknown', { id: row.group_id })
  )
}

function channelName(row: UsageBreakdownRowDto): string {
  if (row.channel_id === undefined) return '—'
  return (
    props.channels.find((channel) => channel.channel_id === row.channel_id)?.name ??
    t('monitor.usage.breakdown.deletedOrUnknownChannel', { id: row.channel_id })
  )
}

function compareRows(
  left: UsageBreakdownRowDto,
  right: UsageBreakdownRowDto,
  key: SortKey,
): number {
  if (key === 'model') return left.model.localeCompare(right.model)
  if (key === 'group') return groupName(left).localeCompare(groupName(right))
  if (key === 'channel') return channelName(left).localeCompare(channelName(right))
  if (key === 'success_rate') {
    return compareBigInt(
      BigInt(left.success_count) * BigInt(right.request_count),
      BigInt(right.success_count) * BigInt(left.request_count),
    )
  }
  if (key === 'average_latency') {
    if (left.duration_sample_count === 0 || right.duration_sample_count === 0) {
      return left.duration_sample_count - right.duration_sample_count
    }
    return compareBigInt(
      BigInt(left.duration_ms_total) * BigInt(right.duration_sample_count),
      BigInt(right.duration_ms_total) * BigInt(left.duration_sample_count),
    )
  }
  if (key === 'estimated_cost_nano_usd') {
    return compareBigInt(BigInt(left[key]), BigInt(right[key]))
  }
  if (key === 'duration_ms_total' || key === 'duration_sample_count') {
    return compareBigInt(BigInt(left[key]), BigInt(right[key]))
  }
  const leftValue = left[key]
  const rightValue = right[key]
  if (typeof leftValue !== 'number' || typeof rightValue !== 'number') return 0
  return leftValue - rightValue
}

function compareBigInt(left: bigint, right: bigint): number {
  return left < right ? -1 : left > right ? 1 : 0
}

function setSort(key: SortKey): void {
  if (sortKey.value === key) {
    sortDirection.value = sortDirection.value === 'asc' ? 'desc' : 'asc'
    return
  }
  sortKey.value = key
  sortDirection.value = key === 'model' || key === 'group' || key === 'channel' ? 'asc' : 'desc'
}

function ariaSort(key: SortKey): 'ascending' | 'descending' | 'none' {
  return sortKey.value === key
    ? sortDirection.value === 'asc'
      ? 'ascending'
      : 'descending'
    : 'none'
}

function averageLatency(aggregate: UsageAggregateDto): string {
  if (aggregate.duration_sample_count === 0) return '—'
  if (aggregate.duration_ms_total === 0) return '0 ms'
  return `${new Intl.NumberFormat(locale.value, { maximumFractionDigits: 1 }).format(
    aggregate.duration_ms_total / aggregate.duration_sample_count,
  )} ms`
}

function successRate(aggregate: UsageAggregateDto): string {
  return formatPercent(aggregate.success_count, aggregate.request_count, locale.value)
}

function quality(aggregate: UsageAggregateDto): string {
  return t('monitor.usage.columns.qualityCompact', {
    missing: formatInteger(aggregate.usage_missing_count, locale.value),
    partial: formatInteger(aggregate.partial_count, locale.value),
    unpriced: formatInteger(aggregate.unpriced_request_count, locale.value),
    pricingPartial: formatInteger(aggregate.pricing_partial_count, locale.value),
  })
}
</script>

<template>
  <DataTable
    :caption="t('monitor.usage.breakdown.caption')"
    :scroll-hint="t('monitor.scrollHint')"
    appearance="editorial"
  >
    <thead>
      <tr>
        <th scope="col" :aria-sort="ariaSort('model')">
          <button type="button" class="usage-breakdown__sort" @click="setSort('model')">
            {{ t('monitor.usage.breakdown.columns.model') }}
          </button>
        </th>
        <th v-if="isAdmin" scope="col" :aria-sort="ariaSort('group')">
          <button type="button" class="usage-breakdown__sort" @click="setSort('group')">
            {{ t('monitor.usage.breakdown.columns.group') }}
          </button>
        </th>
        <th v-if="isAdmin" scope="col" :aria-sort="ariaSort('channel')">
          <button type="button" class="usage-breakdown__sort" @click="setSort('channel')">
            {{ t('monitor.usage.breakdown.columns.channel') }}
          </button>
        </th>
        <th scope="col" :aria-sort="ariaSort('request_count')">
          <button type="button" class="usage-breakdown__sort" @click="setSort('request_count')">
            {{ t('monitor.usage.columns.requests') }}
          </button>
        </th>
        <th scope="col" :aria-sort="ariaSort('success_count')">
          <button type="button" class="usage-breakdown__sort" @click="setSort('success_count')">
            {{ t('monitor.usage.columns.success') }}
          </button>
        </th>
        <th scope="col" :aria-sort="ariaSort('failure_count')">
          <button type="button" class="usage-breakdown__sort" @click="setSort('failure_count')">
            {{ t('monitor.usage.columns.failure') }}
          </button>
        </th>
        <th scope="col" :aria-sort="ariaSort('success_rate')">
          <button type="button" class="usage-breakdown__sort" @click="setSort('success_rate')">
            {{ t('monitor.usage.breakdown.columns.successRate') }}
          </button>
        </th>
        <th scope="col" :aria-sort="ariaSort('average_latency')">
          <button type="button" class="usage-breakdown__sort" @click="setSort('average_latency')">
            {{ t('monitor.usage.breakdown.columns.averageLatency') }}
          </button>
        </th>
        <th scope="col" :aria-sort="ariaSort('uncached_input_tokens')">
          <button
            type="button"
            class="usage-breakdown__sort"
            @click="setSort('uncached_input_tokens')"
          >
            {{ t('monitor.usage.breakdown.columns.uncachedInput') }}
          </button>
        </th>
        <th scope="col" :aria-sort="ariaSort('cache_read_tokens')">
          <button type="button" class="usage-breakdown__sort" @click="setSort('cache_read_tokens')">
            {{ t('monitor.usage.breakdown.columns.cacheRead') }}
          </button>
        </th>
        <th scope="col" :aria-sort="ariaSort('cache_write_5m_tokens')">
          <button
            type="button"
            class="usage-breakdown__sort"
            @click="setSort('cache_write_5m_tokens')"
          >
            {{ t('monitor.usage.breakdown.columns.cacheWrite5m') }}
          </button>
        </th>
        <th scope="col" :aria-sort="ariaSort('cache_write_1h_tokens')">
          <button
            type="button"
            class="usage-breakdown__sort"
            @click="setSort('cache_write_1h_tokens')"
          >
            {{ t('monitor.usage.breakdown.columns.cacheWrite1h') }}
          </button>
        </th>
        <th scope="col" :aria-sort="ariaSort('cache_write_unknown_tokens')">
          <button
            type="button"
            class="usage-breakdown__sort"
            @click="setSort('cache_write_unknown_tokens')"
          >
            {{ t('monitor.usage.breakdown.columns.cacheWriteUnknown') }}
          </button>
        </th>
        <th scope="col" :aria-sort="ariaSort('output_tokens')">
          <button type="button" class="usage-breakdown__sort" @click="setSort('output_tokens')">
            {{ t('monitor.usage.breakdown.columns.output') }}
          </button>
        </th>
        <th scope="col" :aria-sort="ariaSort('total_tokens')">
          <button type="button" class="usage-breakdown__sort" @click="setSort('total_tokens')">
            {{ t('monitor.usage.columns.totalTokens') }}
          </button>
        </th>
        <th scope="col" :aria-sort="ariaSort('estimated_cost_nano_usd')">
          <button
            type="button"
            class="usage-breakdown__sort"
            @click="setSort('estimated_cost_nano_usd')"
          >
            {{ t('monitor.usage.columns.estimatedCost') }}
          </button>
        </th>
        <th scope="col">{{ t('monitor.usage.columns.quality') }}</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="row in sortedRows" :key="rowKey(row)">
        <td class="usage-breakdown__identity">{{ row.model }}</td>
        <td v-if="isAdmin">{{ groupName(row) }}</td>
        <td v-if="isAdmin">{{ channelName(row) }}</td>
        <td>{{ formatInteger(row.request_count, locale) }}</td>
        <td>{{ formatInteger(row.success_count, locale) }}</td>
        <td>{{ formatInteger(row.failure_count, locale) }}</td>
        <td>{{ successRate(row) }}</td>
        <td>{{ averageLatency(row) }}</td>
        <td>{{ formatTokens(row.uncached_input_tokens, locale) }}</td>
        <td>{{ formatTokens(row.cache_read_tokens, locale) }}</td>
        <td>{{ formatTokens(row.cache_write_5m_tokens, locale) }}</td>
        <td>{{ formatTokens(row.cache_write_1h_tokens, locale) }}</td>
        <td>{{ formatTokens(row.cache_write_unknown_tokens, locale) }}</td>
        <td>{{ formatTokens(row.output_tokens, locale) }}</td>
        <td>{{ formatTokens(row.total_tokens, locale) }}</td>
        <td>{{ formatEstimatedCost(row.estimated_cost_nano_usd, locale) }}</td>
        <td :title="quality(row)">{{ quality(row) }}</td>
      </tr>
    </tbody>
    <tfoot>
      <tr class="usage-breakdown__total">
        <th scope="row">{{ t('monitor.usage.breakdown.total') }}</th>
        <td v-if="isAdmin">—</td>
        <td v-if="isAdmin">—</td>
        <td>{{ formatInteger(breakdown.total.request_count, locale) }}</td>
        <td>{{ formatInteger(breakdown.total.success_count, locale) }}</td>
        <td>{{ formatInteger(breakdown.total.failure_count, locale) }}</td>
        <td>{{ successRate(breakdown.total) }}</td>
        <td>{{ averageLatency(breakdown.total) }}</td>
        <td>{{ formatTokens(breakdown.total.uncached_input_tokens, locale) }}</td>
        <td>{{ formatTokens(breakdown.total.cache_read_tokens, locale) }}</td>
        <td>{{ formatTokens(breakdown.total.cache_write_5m_tokens, locale) }}</td>
        <td>{{ formatTokens(breakdown.total.cache_write_1h_tokens, locale) }}</td>
        <td>{{ formatTokens(breakdown.total.cache_write_unknown_tokens, locale) }}</td>
        <td>{{ formatTokens(breakdown.total.output_tokens, locale) }}</td>
        <td>{{ formatTokens(breakdown.total.total_tokens, locale) }}</td>
        <td>{{ formatEstimatedCost(breakdown.total.estimated_cost_nano_usd, locale) }}</td>
        <td :title="quality(breakdown.total)">{{ quality(breakdown.total) }}</td>
      </tr>
    </tfoot>
  </DataTable>
</template>

<style scoped>
.usage-breakdown__sort {
  border: 0;
  background: transparent;
  color: inherit;
  cursor: pointer;
  font: inherit;
  letter-spacing: inherit;
  padding: 0;
  text-align: left;
  text-transform: inherit;
}

.usage-breakdown__sort:hover,
.usage-breakdown__sort:focus-visible {
  color: var(--color-text);
  text-decoration: underline;
  text-underline-offset: 3px;
}

.usage-breakdown__identity {
  max-width: 230px;
  overflow: hidden;
  font-weight: 620;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.usage-breakdown__total th,
.usage-breakdown__total td {
  border-top: 2px solid var(--color-border-subtle);
  background: var(--color-surface-sunken);
  font-weight: 700;
}

.usage-breakdown__total td,
.usage-breakdown__total th,
.usage-breakdown__sort {
  font-variant-numeric: tabular-nums;
}
</style>
