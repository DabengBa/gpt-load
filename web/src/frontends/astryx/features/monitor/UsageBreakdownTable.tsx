import * as stylex from '@stylexjs/stylex'
import { Pagination } from '@astryxdesign/core'
import { useEffect, useId, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { useIntl } from 'react-intl'

import type { GroupOptionDto } from '@shared/control/types'
import {
  type UsageAggregateDto,
  type UsageBreakdownDto,
  type UsageBreakdownPageSize,
  type UsageBreakdownRowDto,
  type UsageBreakdownSort,
  type UsageBreakdownSortDirection,
} from '@shared/control/resources/usage'
import { formatEstimatedCost, formatInteger, formatPercent, formatTokens } from '@shared/lib/format'
import type { MessageId } from '@shared/i18n/message-ids'

import { useT } from '../../app/i18n'
import { RouteLink } from '../../app/route-link'

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

    lineHeight: 1.3,
    textAlign: 'left',

    whiteSpace: 'nowrap',
  },
  numericHeadCell: {
    textAlign: 'center',
    paddingInlineStart: 12,
    paddingInlineEnd: 12,
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
  numericCell: {
    textAlign: 'center',
    fontVariantNumeric: 'tabular-nums',
    paddingInlineStart: 12,
    paddingInlineEnd: 12,
  },
  primaryNumericCell: {
    fontWeight: 620,
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
  groupLink: {
    textDecorationLine: 'underline',
    textUnderlineOffset: 3,
  },
})

interface BreakdownColumn {
  key: string
  labelId: MessageId
  sortKey?: UsageBreakdownSort
  adminOnly?: boolean
  identity?: boolean
  numeric?: boolean
  primary?: boolean
  cell(row: UsageBreakdownRowDto): ReactNode
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
  sort,
  sortDirection,
  onPage,
  onPageSize,
  onSort,
}: {
  breakdown: UsageBreakdownDto
  groups: GroupOptionDto[]
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
      row.client_model,
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
  function averageSeconds(totalMs: number, sampleCount: number): string {
    if (sampleCount === 0) return '—'
    return `${intl.formatNumber(totalMs / (sampleCount * 1000), {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    })} s`
  }

  function averageDuration(aggregate: UsageAggregateDto): string {
    return averageSeconds(aggregate.duration_ms_total, aggregate.duration_sample_count)
  }

  function averageFirstResponse(aggregate: UsageAggregateDto): string {
    return averageSeconds(aggregate.first_response_ms_total, aggregate.first_response_sample_count)
  }

  function successRate(aggregate: UsageAggregateDto): string {
    return formatPercent(aggregate.success_count, aggregate.request_count, locale)
  }

  function cacheRate(row: UsageBreakdownRowDto): string {
    const inputTokens = row.uncached_input_tokens + row.cache_read_tokens
    return inputTokens === 0 ? '—' : formatPercent(row.cache_read_tokens, inputTokens, locale)
  }

  const columns: BreakdownColumn[] = [
    {
      key: 'client_model',
      labelId: 'monitor.usage.breakdown.columns.clientModel',
      identity: true,
      numeric: false,
      cell: (row) => row.client_model ?? t('monitor.usage.breakdown.unknownModel'),
      total: () => '—',
    },
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
      cell: (row) => {
        const knownGroup = groups.find((group) => group.id === row.group_id)
        if (knownGroup === undefined || row.group_id === undefined) return groupName(row)
        return (
          <RouteLink to={`/groups/${row.group_id}`} {...stylex.props(styles.groupLink)}>
            {knownGroup.name}
          </RouteLink>
        )
      },
      total: () => '—',
    },

    {
      key: 'attempts',
      sortKey: 'attempt_count',
      labelId: 'monitor.usage.breakdown.columns.attempts',
      cell: (row) => formatInteger(row.attempt_count, locale),
      total: (data) => formatInteger(data.attempt_total.attempt_count, locale),
    },
    {
      key: 'attemptFailures',
      sortKey: 'attempt_failure_count',
      labelId: 'monitor.usage.breakdown.columns.attemptFailures',
      cell: (row) => formatInteger(row.attempt_failure_count, locale),
      total: (data) => formatInteger(data.attempt_total.attempt_failure_count, locale),
    },
    {
      key: 'normalAttempts',
      sortKey: 'normal_attempt_count',
      labelId: 'monitor.usage.breakdown.columns.normalAttempts',
      cell: (row) => formatInteger(row.normal_attempt_count, locale),
      total: (data) => formatInteger(data.attempt_total.normal_attempt_count, locale),
    },
    {
      key: 'slowAttempts',
      sortKey: 'slow_attempt_count',
      labelId: 'monitor.usage.breakdown.columns.slowAttempts',
      cell: (row) => formatInteger(row.slow_attempt_count, locale),
      total: (data) => formatInteger(data.attempt_total.slow_attempt_count, locale),
    },
    {
      key: 'faultyAttempts',
      sortKey: 'faulty_attempt_count',
      labelId: 'monitor.usage.breakdown.columns.faultyAttempts',
      cell: (row) => formatInteger(row.faulty_attempt_count, locale),
      total: (data) => formatInteger(data.attempt_total.faulty_attempt_count, locale),
    },
    {
      key: 'success_rate',
      labelId: 'monitor.usage.breakdown.columns.successRate',
      sortKey: 'success_rate',
      cell: (row) => successRate(row),
      total: (data) => successRate(data.total),
      primary: true,
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
      key: 'cache_rate',
      labelId: 'monitor.usage.breakdown.columns.cacheRate',
      sortKey: 'cache_rate',
      cell: (row) => cacheRate(row),
      total: () => '—',
    },
    {
      key: 'token_efficiency',
      labelId: 'monitor.usage.breakdown.columns.tokenEfficiency',
      sortKey: 'token_efficiency',
      cell: (row) => {
        const {
          success_output_tokens,
          success_duration_ms_total,
          success_efficiency_sample_count,
          success_count,
        } = row
        if (
          success_count === 0 ||
          success_efficiency_sample_count !== success_count ||
          success_output_tokens <= 0 ||
          success_duration_ms_total <= 0
        )
          return '—'
        return `${intl.formatNumber((success_output_tokens * 1000) / success_duration_ms_total, {
          maximumFractionDigits: 1,
        })} /s`
      },
      total: () => '—',
    },
    {
      key: 'total_tokens',
      labelId: 'monitor.usage.columns.totalTokens',
      sortKey: 'total_tokens',
      cell: (row) => formatTokens(row.total_tokens, locale),
      total: (data) => formatTokens(data.total.total_tokens, locale),
      primary: true,
    },
    {
      key: 'estimated_cost_nano_usd',
      labelId: 'monitor.usage.columns.estimatedCost',
      sortKey: 'estimated_cost_nano_usd',
      cell: (row) => formatEstimatedCost(row.estimated_cost_nano_usd, locale),
      total: (data) => formatEstimatedCost(data.total.estimated_cost_nano_usd, locale),
      primary: true,
    },
  ]
  const visibleColumns = columns.filter((column) => !column.adminOnly || isAdmin)
  const lastColumnIndex = visibleColumns.length - 1

  function setSort(key: UsageBreakdownSort): void {
    const direction = sort === key ? (sortDirection === 'asc' ? 'desc' : 'asc') : 'desc'
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
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(update) : undefined
    observer?.observe(container)
    const table = tableRef.current
    if (table !== null) observer?.observe(table)
    const fallbackFrame = observer === undefined ? requestAnimationFrame(update) : undefined
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
                const isNumeric =
                  column.numeric ?? (column.key !== 'model' && column.key !== 'group')
                return (
                  <th
                    key={column.key}
                    scope="col"
                    aria-sort={sortKey !== undefined ? ariaSort(sortKey) : undefined}
                    {...stylex.props(
                      styles.headCell,
                      isNumeric && styles.numericHeadCell,
                      !isNumeric && columnIndex === 0 && styles.cellFirst,
                      !isNumeric && columnIndex === lastColumnIndex && styles.cellLast,
                      isNumeric && styles.numericHeadCell,
                    )}
                  >
                    {sortKey !== undefined ? (
                      <button
                        type="button"
                        {...stylex.props(styles.sort)}
                        onClick={() => setSort(sortKey)}
                      >
                        {t(column.labelId)}
                        {sort === sortKey && (
                          <span aria-hidden="true">{sortDirection === 'desc' ? ' ↓' : ' ↑'}</span>
                        )}
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
              <tr key={rowKey(row)}>
                {visibleColumns.map((column, columnIndex) => (
                  <td
                    key={column.key}
                    {...stylex.props(
                      styles.cell,
                      column.identity === true && styles.identity,
                      (column.numeric ?? (column.key !== 'model' && column.key !== 'group')) &&
                        styles.numericCell,
                      column.primary === true && styles.primaryNumericCell,
                      !(column.numeric ?? (column.key !== 'model' && column.key !== 'group')) &&
                        columnIndex === 0 &&
                        styles.cellFirst,
                      !(column.numeric ?? (column.key !== 'model' && column.key !== 'group')) &&
                        columnIndex === lastColumnIndex &&
                        styles.cellLast,
                      (column.numeric ?? (column.key !== 'model' && column.key !== 'group')) &&
                        styles.numericCell,
                      rowIndex === breakdown.rows.length - 1 && styles.cellNoBorder,
                    )}
                  >
                    {column.cell(row)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
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
