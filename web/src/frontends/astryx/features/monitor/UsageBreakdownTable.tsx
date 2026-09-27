import * as stylex from '@stylexjs/stylex'
import { Pagination } from '@astryxdesign/core'
import { useEffect, useId, useRef, useState } from 'react'
import { useIntl } from 'react-intl'

import type { GroupOptionDto } from '@shared/control/types'
import type { ChannelDto } from '@shared/control/resources/channels'
import {
  defaultUsageBreakdownSortDirectionFor,
  type UsageAggregateDto,
  type UsageBreakdownDto,
  type UsageBreakdownPageSize,
  type UsageBreakdownRowDto,
  type UsageBreakdownSort,
  type UsageBreakdownSortDirection,
} from '@shared/control/resources/usage'
import {
  formatEstimatedCost,
  formatInteger,
  formatPercent,
  formatTokens,
} from '@shared/lib/format'
import type { MessageId } from '@shared/i18n/message-ids'

import { useT } from '../../app/i18n'

const styles = stylex.create({
  // DataTable port (appearance="editorial"): a horizontal-scroll container
  // that gains tabindex/label/hint wiring only while it overflows.
  container: {
    maxWidth: '100%',
    overflowX: 'auto',
    overscrollBehaviorInline: 'contain',
  },
  table: {
    width: '100%',
    minWidth: 'var(--table-editorial-min-width)',
    borderCollapse: 'collapse',
    fontSize: 12.5,
  },
  srOnly: {
    position: 'absolute',
    width: 1,
    height: 1,
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
  headCell: {
    paddingTop: 0,
    paddingBottom: 8,
    paddingInlineStart: 0,
    paddingInlineEnd: 12,
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    backgroundColor: 'transparent',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 500,
    letterSpacing: '0.07em',
    lineHeight: 1.3,
    textAlign: 'left',
    textTransform: 'uppercase',
    whiteSpace: 'nowrap',
  },
  cell: {
    paddingTop: 10,
    paddingBottom: 10,
    paddingInlineStart: 0,
    paddingInlineEnd: 12,
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    lineHeight: 1.3,
    verticalAlign: 'middle',
    whiteSpace: 'nowrap',
  },
  // Editorial tables keep the outermost column flush with the container edge:
  // classic does it with :first-child/:last-child padding overrides, which
  // StyleX disallows, so first/last columns apply these by index.
  cellFirst: {
    paddingInlineStart: 0,
  },
  cellLast: {
    paddingInlineEnd: 0,
  },
  bodyRow: {
    backgroundColor: {
      default: 'transparent',
      ':hover': 'var(--color-surface-sunken)',
    },
  },
  cellNoBorder: {
    borderBottomWidth: 0,
  },
  sort: {
    borderWidth: 0,
    // The astryx reset already applies `font: inherit` + `color: inherit` +
    // zero margin/padding to buttons (classic set font/padding manually).
    backgroundColor: 'transparent',
    cursor: 'pointer',
    letterSpacing: 'inherit',
    textAlign: 'left',
    textTransform: 'inherit',
    fontVariantNumeric: 'tabular-nums',
    color: {
      default: 'inherit',
      ':hover': 'var(--color-text)',
      ':focus-visible': 'var(--color-text)',
    },
    textDecorationLine: {
      default: 'none',
      ':hover': 'underline',
      ':focus-visible': 'underline',
    },
    textUnderlineOffset: {
      default: 3,
      ':hover': 3,
      ':focus-visible': 3,
    },
  },
  identity: {
    maxWidth: 230,
    overflow: 'hidden',
    fontWeight: 620,
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  // Classic `.usage-breakdown__total th/td`: layered on top of the regular
  // headCell/cell rules, so tfoot cells keep their border-bottom and padding.
  totalAccent: {
    borderTopWidth: 2,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    backgroundColor: 'var(--color-surface-sunken)',
    fontWeight: 700,
    fontVariantNumeric: 'tabular-nums',
  },
})

interface BreakdownColumn {
  key: string
  labelId: MessageId
  sortKey?: UsageBreakdownSort
  adminOnly?: boolean
  identity?: boolean
  cell(row: UsageBreakdownRowDto): string
  total(breakdown: UsageBreakdownDto): string
}

/**
 * Paginated, sortable breakdown table — classic UsageBreakdownTable.vue with
 * DataTable (editorial) + PaginationBar. Emits become `onPage`, `onPageSize`,
 * `onSort` props.
 */
export function UsageBreakdownTable({
  breakdown,
  groups,
  channels,
  sort,
  sortDirection,
  onPage,
  onPageSize,
  onSort,
}: {
  breakdown: UsageBreakdownDto
  groups: GroupOptionDto[]
  channels: ChannelDto[]
  sort: UsageBreakdownSort
  sortDirection: UsageBreakdownSortDirection
  onPage: (page: number) => void
  onPageSize: (pageSize: UsageBreakdownPageSize) => void
  onSort: (sort: UsageBreakdownSort, direction: UsageBreakdownSortDirection) => void
}) {
  const intl = useIntl()
  const t = useT()
  const locale = intl.locale
  const isAdmin = breakdown.scope === 'admin'

  function rowKey(row: UsageBreakdownRowDto): string {
    return JSON.stringify([
      breakdown.scope,
      row.model,
      row.group_id ?? null,
      row.channel_id ?? null,
    ])
  }

  function groupName(row: UsageBreakdownRowDto): string {
    if (row.group_id === undefined) return '—'
    return (
      groups.find((group) => group.id === row.group_id)?.name ??
      t('monitor.usage.filters.deletedOrUnknown', { id: row.group_id })
    )
  }

  function channelName(row: UsageBreakdownRowDto): string {
    if (row.channel_id === undefined) return '—'
    return (
      channels.find((channel) => channel.channel_id === row.channel_id)?.name ??
      t('monitor.usage.breakdown.deletedOrUnknownChannel', { id: row.channel_id })
    )
  }

  function averageMilliseconds(totalMs: number, sampleCount: number): string {
    if (sampleCount === 0) return '—'
    if (totalMs === 0) return '0 ms'
    return `${intl.formatNumber(totalMs / sampleCount, { maximumFractionDigits: 1 })} ms`
  }

  function averageDuration(aggregate: UsageAggregateDto): string {
    return averageMilliseconds(aggregate.duration_ms_total, aggregate.duration_sample_count)
  }

  function averageFirstResponse(aggregate: UsageAggregateDto): string {
    return averageMilliseconds(
      aggregate.first_response_ms_total,
      aggregate.first_response_sample_count,
    )
  }

  function successRate(aggregate: UsageAggregateDto): string {
    return formatPercent(aggregate.success_count, aggregate.request_count, locale)
  }

  const columns: BreakdownColumn[] = [
    {
      key: 'model',
      labelId: 'monitor.usage.breakdown.columns.model',
      sortKey: 'model',
      identity: true,
      cell: (row) => row.model,
      total: () => t('monitor.usage.breakdown.total'),
    },
    {
      key: 'group',
      labelId: 'monitor.usage.breakdown.columns.group',
      sortKey: 'group',
      adminOnly: true,
      cell: (row) => groupName(row),
      total: () => '—',
    },
    {
      key: 'channel',
      labelId: 'monitor.usage.breakdown.columns.channel',
      sortKey: 'channel',
      adminOnly: true,
      cell: (row) => channelName(row),
      total: () => '—',
    },
    {
      key: 'request_count',
      labelId: 'monitor.usage.columns.requests',
      sortKey: 'request_count',
      cell: (row) => formatInteger(row.request_count, locale),
      total: (data) => formatInteger(data.total.request_count, locale),
    },
    {
      key: 'success_count',
      labelId: 'monitor.usage.columns.success',
      sortKey: 'success_count',
      cell: (row) => formatInteger(row.success_count, locale),
      total: (data) => formatInteger(data.total.success_count, locale),
    },
    {
      key: 'failure_count',
      labelId: 'monitor.usage.columns.failure',
      sortKey: 'failure_count',
      cell: (row) => formatInteger(row.failure_count, locale),
      total: (data) => formatInteger(data.total.failure_count, locale),
    },
    {
      key: 'attempts',
      labelId: 'monitor.usage.breakdown.columns.attempts',
      cell: (row) => formatInteger(row.attempt_count, locale),
      total: (data) => formatInteger(data.attempt_total.attempt_count, locale),
    },
    {
      key: 'attemptFailures',
      labelId: 'monitor.usage.breakdown.columns.attemptFailures',
      cell: (row) => formatInteger(row.attempt_failure_count, locale),
      total: (data) => formatInteger(data.attempt_total.attempt_failure_count, locale),
    },
    {
      key: 'success_rate',
      labelId: 'monitor.usage.breakdown.columns.successRate',
      sortKey: 'success_rate',
      cell: (row) => successRate(row),
      total: (data) => successRate(data.total),
    },
    {
      key: 'average_duration_ms',
      labelId: 'monitor.usage.breakdown.columns.averageDuration',
      sortKey: 'average_duration_ms',
      cell: (row) => averageDuration(row),
      total: (data) => averageDuration(data.total),
    },
    {
      key: 'average_first_response_ms',
      labelId: 'monitor.usage.breakdown.columns.averageFirstResponse',
      sortKey: 'average_first_response_ms',
      cell: (row) => averageFirstResponse(row),
      total: (data) => averageFirstResponse(data.total),
    },
    {
      key: 'uncached_input_tokens',
      labelId: 'monitor.usage.breakdown.columns.uncachedInput',
      sortKey: 'uncached_input_tokens',
      cell: (row) => formatTokens(row.uncached_input_tokens, locale),
      total: (data) => formatTokens(data.total.uncached_input_tokens, locale),
    },
    {
      key: 'cache_read_tokens',
      labelId: 'monitor.usage.breakdown.columns.cacheRead',
      sortKey: 'cache_read_tokens',
      cell: (row) => formatTokens(row.cache_read_tokens, locale),
      total: (data) => formatTokens(data.total.cache_read_tokens, locale),
    },
    {
      key: 'output_tokens',
      labelId: 'monitor.usage.breakdown.columns.output',
      sortKey: 'output_tokens',
      cell: (row) => formatTokens(row.output_tokens, locale),
      total: (data) => formatTokens(data.total.output_tokens, locale),
    },
    {
      key: 'total_tokens',
      labelId: 'monitor.usage.columns.totalTokens',
      sortKey: 'total_tokens',
      cell: (row) => formatTokens(row.total_tokens, locale),
      total: (data) => formatTokens(data.total.total_tokens, locale),
    },
    {
      key: 'estimated_cost_nano_usd',
      labelId: 'monitor.usage.columns.estimatedCost',
      sortKey: 'estimated_cost_nano_usd',
      cell: (row) => formatEstimatedCost(row.estimated_cost_nano_usd, locale),
      total: (data) => formatEstimatedCost(data.total.estimated_cost_nano_usd, locale),
    },
  ]
  const visibleColumns = columns.filter((column) => !column.adminOnly || isAdmin)
  const lastColumnIndex = visibleColumns.length - 1

  function setSort(key: UsageBreakdownSort): void {
    const direction =
      sort === key
        ? sortDirection === 'asc'
          ? 'desc'
          : 'asc'
        : defaultUsageBreakdownSortDirectionFor(key)
    onSort(key, direction)
  }

  function ariaSort(key: UsageBreakdownSort): 'ascending' | 'descending' | 'none' {
    return sort === key ? (sortDirection === 'asc' ? 'ascending' : 'descending') : 'none'
  }

  function setPage(page: number): void {
    if (page < 1 || page > breakdown.pagination.total_pages) return
    onPage(page)
  }

  function setPageSize(pageSize: number): void {
    if (pageSize === 20 || pageSize === 50 || pageSize === 100) onPageSize(pageSize)
  }

  // DataTable overflow wiring: the classic measured on mount, on table resize,
  // and on window resize. ResizeObserver on both the container and the table
  // reproduces all three (it fires once on observe for the mount measure).
  const containerRef = useRef<HTMLDivElement | null>(null)
  const tableRef = useRef<HTMLTableElement | null>(null)
  const [overflowing, setOverflowing] = useState(false)
  useEffect(() => {
    const container = containerRef.current
    if (container === null) return
    const update = (): void => {
      setOverflowing(container.scrollWidth > container.clientWidth + 1)
    }
    const observer =
      typeof ResizeObserver === 'function' ? new ResizeObserver(update) : undefined
    observer?.observe(container)
    const table = tableRef.current
    if (table !== null) observer?.observe(table)
    const fallbackFrame =
      observer === undefined ? requestAnimationFrame(update) : undefined
    window.addEventListener('resize', update)
    return () => {
      observer?.disconnect()
      if (fallbackFrame !== undefined) cancelAnimationFrame(fallbackFrame)
      window.removeEventListener('resize', update)
    }
  }, [])

  const caption = t('monitor.usage.breakdown.caption')
  const scrollHint = t('monitor.scrollHint')
  const identity = useId().replace(/[^a-zA-Z0-9_-]/g, '-')
  const scrollHintId = `data-table-${identity}-scroll-hint`

  return (
    <>
      <div
        ref={containerRef}
        {...stylex.props(styles.container)}
        data-table-scroll
        tabIndex={overflowing ? 0 : undefined}
        aria-label={overflowing ? caption : undefined}
        aria-describedby={overflowing && scrollHint !== '' ? scrollHintId : undefined}
      >
        <table ref={tableRef} {...stylex.props(styles.table)}>
          <caption {...stylex.props(styles.srOnly)}>{caption}</caption>
          <thead>
            <tr>
              {visibleColumns.map((column, columnIndex) => {
                const sortKey = column.sortKey
                return (
                  <th
                    key={column.key}
                    scope="col"
                    aria-sort={sortKey !== undefined ? ariaSort(sortKey) : undefined}
                    {...stylex.props(
                      styles.headCell,
                      columnIndex === 0 && styles.cellFirst,
                      columnIndex === lastColumnIndex && styles.cellLast,
                    )}
                  >
                    {sortKey !== undefined ? (
                      <button
                        type="button"
                        {...stylex.props(styles.sort)}
                        onClick={() => setSort(sortKey)}
                      >
                        {t(column.labelId)}
                      </button>
                    ) : (
                      t(column.labelId)
                    )}
                  </th>
                )
              })}
            </tr>
          </thead>
          <tbody>
            {breakdown.rows.map((row, rowIndex) => (
              <tr
                key={rowKey(row)}
                {...stylex.props(styles.bodyRow)}
              >
                {visibleColumns.map((column, columnIndex) => (
                  <td
                    key={column.key}
                    {...stylex.props(
                      styles.cell,
                      column.identity === true && styles.identity,
                      columnIndex === 0 && styles.cellFirst,
                      columnIndex === lastColumnIndex && styles.cellLast,
                      rowIndex === breakdown.rows.length - 1 && styles.cellNoBorder,
                    )}
                  >
                    {column.cell(row)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              {visibleColumns.map((column, columnIndex) =>
                columnIndex === 0 ? (
                  <th
                    key={column.key}
                    scope="row"
                    {...stylex.props(
                      styles.headCell,
                      styles.totalAccent,
                      styles.cellFirst,
                      columnIndex === lastColumnIndex && styles.cellLast,
                    )}
                  >
                    {column.total(breakdown)}
                  </th>
                ) : (
                  <td
                    key={column.key}
                    {...stylex.props(
                      styles.cell,
                      styles.totalAccent,
                      columnIndex === lastColumnIndex && styles.cellLast,
                    )}
                  >
                    {column.total(breakdown)}
                  </td>
                ),
              )}
            </tr>
          </tfoot>
        </table>
        <span id={scrollHintId} {...stylex.props(styles.srOnly)}>
          {scrollHint}
        </span>
      </div>
      {/* PaginationBar port: compact variant renders the same prev/next +
          "N / M" arrangement; pageSizeOptions supplies the 20/50/100 picker. */}
      <Pagination
        page={breakdown.pagination.page}
        totalItems={breakdown.pagination.total_items}
        totalPages={breakdown.pagination.total_pages}
        pageSize={breakdown.pagination.page_size}
        pageSizeOptions={[20, 50, 100]}
        onPageSizeChange={setPageSize}
        onChange={setPage}
        size="sm"
        variant="compact"
        label={t('common.pagination.label')}
        data-testid="pagination-bar"
      />
    </>
  )
}
