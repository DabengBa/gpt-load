import * as stylex from '@stylexjs/stylex'
import { Badge, IconButton, Tooltip } from '@astryxdesign/core'
import {
  ArrowRight,
  CircleAlert,
  CircleCheck,
  CircleHelp,
  CircleOff,
  Info,
  Layers,
  Magnet,
  TriangleAlert,
  type LucideIcon,
} from 'lucide-react'
import { memo, type ReactNode } from 'react'
import { useIntl } from 'react-intl'

import type { ChannelDto } from '@shared/control/resources/channels'
import type { RequestLogItemDto } from '@shared/control/resources/request-logs'
import {
  formatLogDuration,
  formatLogOutputRate,
  formatLogReasoning,
  formatLogTokenCount,
  hasRequestLogCache,
  reasoningBudgetSemantic,
  requestLogCostDisplayState,
  requestLogProtocolLabel,
  requestLogResponseTooltip,
  requestLogResponseTooltipVisible,
  requestLogUsageDisplayState,
} from '@shared/domain/monitor/log-format'
import { cacheHitRate } from '@shared/lib/cache-rate'
import {
  formatEstimatedCost,
  formatISOInstant,
  formatLocalInstantWithSeconds,
} from '@shared/lib/format'
import { currentTimeZone } from '@shared/lib/time'
import type { MessageId } from '@shared/i18n/message-ids'

import { useT } from '../../app/i18n'
import { LogProtocolConversion } from '../monitor/LogProtocolConversion'
import { LogRouteIdentity } from '../monitor/LogRouteIdentity'
import { PricingModeIndicator } from '../monitor/PricingModeIndicator'

const TABLET = '@media (max-width: 1080px)'
const CARD = '@media (max-width: 860px)'

// Desktop grids — classic `--ledger-record-list-grid` values verbatim; TABLET
// is the classic 1080px tighten, CARD is the 860px card layout.
const ADMIN_GRID =
  '72px minmax(120px, 0.82fr) minmax(120px, 0.82fr) minmax(170px, 1.15fr) 96px minmax(76px, 0.42fr) minmax(104px, 0.6fr) 100px 34px'
const ADMIN_GRID_TABLET =
  '68px minmax(108px, 0.78fr) minmax(108px, 0.78fr) minmax(150px, 1.1fr) 92px minmax(72px, 0.42fr) minmax(96px, 0.58fr) 96px 32px'
const SCOPED_GRID =
  '72px minmax(170px, 1.15fr) 96px minmax(76px, 0.42fr) minmax(104px, 0.6fr) 100px 34px'
const SCOPED_GRID_TABLET =
  '68px minmax(150px, 1.1fr) 92px minmax(72px, 0.42fr) minmax(96px, 0.58fr) 96px 32px'

const styles = stylex.create({
  // LedgerRecordList port: the container is the grid.
  list: {
    display: 'grid',
    minWidth: 0,
    columnGap: { default: 14, [CARD]: 14 },
    rowGap: { [CARD]: 10 },
    overflowX: { default: 'auto', [CARD]: 'visible' },
    overflowY: { default: 'hidden', [CARD]: 'visible' },
    borderBottomWidth: { default: 1, [CARD]: 0 },
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-control)',
    paddingTop: { [CARD]: 10 },
  },
  listAdmin: {
    gridTemplateColumns: {
      default: ADMIN_GRID,
      [TABLET]: ADMIN_GRID_TABLET,
      [CARD]: 'minmax(0, 1fr)',
    },
  },
  listScoped: {
    gridTemplateColumns: {
      default: SCOPED_GRID,
      [TABLET]: SCOPED_GRID_TABLET,
      [CARD]: 'minmax(0, 1fr)',
    },
  },
  headRow: {
    display: { default: 'grid', [CARD]: 'none' },
    gridColumn: '1 / -1',
    gridTemplateColumns: 'subgrid',
    alignItems: 'center',
    minHeight: 38,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
    fontWeight: 500,
    letterSpacing: '0.04em',
  },
  headCell: {
    justifySelf: 'stretch',
    textAlign: 'left',
  },
  record: {
    position: 'relative',
    display: 'grid',
    gridColumn: { default: '1 / -1', [CARD]: 1 },
    gridTemplateColumns: {
      default: 'subgrid',
      // Classic card grid: label column + value column; each cell spans both
      // columns and splits label/value through its own subgrid.
      [CARD]: 'minmax(104px, 0.42fr) minmax(0, 1.58fr)',
    },
    alignItems: { default: 'center', [CARD]: 'start' },
    minHeight: { default: 52, [CARD]: 0 },
    rowGap: { [CARD]: 10 },
    columnGap: { [CARD]: 14 },
    borderStyle: 'solid',
    borderTopWidth: 1,
    borderRightWidth: { default: 0, [CARD]: 1 },
    borderBottomWidth: { default: 0, [CARD]: 1 },
    borderLeftWidth: { default: 0, [CARD]: 1 },
    borderColor: 'var(--color-border-subtle)',
    borderRadius: { [CARD]: 'var(--radius-control)' },
    backgroundColor: {
      default: 'transparent',
      ':hover': 'var(--color-surface-sunken)',
    },
    paddingBlock: { default: 8, [CARD]: 14 },
    paddingInline: { [CARD]: 14 },
    transitionProperty: 'background-color',
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
  },
  recordFirst: {
    borderTopColor: {
      default: 'var(--color-border-control)',
      [CARD]: 'var(--color-border-subtle)',
    },
  },
  cell: {
    display: 'grid',
    minWidth: 0,
    gap: 4,
    color: 'var(--color-text)',
    fontSize: 'var(--text-sm)',
    fontWeight: 400,
    gridColumn: { [CARD]: '1 / -1' },
    gridTemplateColumns: { [CARD]: 'subgrid' },
    alignItems: { [CARD]: 'start' },
  },
  // Mobile card label — the classic `data-label` ::before becomes a real
  // element sitting in the label column.
  cellLabel: {
    display: { default: 'none', [CARD]: 'block' },
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  cellValue: {
    display: 'grid',
    minWidth: 0,
    gap: 4,
    gridColumn: { [CARD]: 2 },
  },
  ellipsis: {
    minWidth: 0,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  time: {
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
  },
  dayLabel: {
    display: 'block',
    overflow: 'hidden',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 400,
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  affinityKeyCell: {
    alignContent: 'center',
  },
  affinityKey: {
    display: 'block',
    minWidth: 0,
    overflow: 'hidden',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  affinityKeyButton: {
    width: '100%',
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: { default: 'var(--color-action)', ':hover': 'var(--color-text)' },
    padding: 0,
    fontSize: 'inherit',
    fontFamily: 'inherit',
    textAlign: 'left',
    cursor: 'pointer',
    textDecoration: { default: 'none', ':hover': 'underline' },
    textUnderlineOffset: { ':hover': 3 },
  },
  inline: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 0,
  },
  model: {
    flexShrink: 1,
    flexGrow: 0,
    flexBasis: 'auto',
    minWidth: 0,
    overflow: 'hidden',
    fontFamily: 'var(--font-mono)',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  modelButton: {
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: { default: 'var(--color-action)', ':hover': 'var(--color-text)' },
    padding: 0,
    fontSize: 'inherit',
    fontFamily: 'var(--font-mono)',
    cursor: 'pointer',
    textDecoration: { default: 'none', ':hover': 'underline' },
    textUnderlineOffset: { ':hover': 3 },
  },
  modelMapping: {
    flexShrink: 1,
    flexGrow: 0,
    flexBasis: 'auto',
    minWidth: 0,
    overflow: 'hidden',
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  reasoning: {
    flexShrink: 0,
    flexGrow: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  protocolLine: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 8,
  },
  protocol: {
    minWidth: 0,
    color: 'var(--color-text-faint)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  responsePrimary: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 4,
  },
  responseMeta: {
    display: 'block',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  hint: {
    display: 'inline-flex',
    width: 20,
    height: 20,
    flexShrink: 0,
    flexBasis: 20,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 0,
    borderRadius: 'var(--radius-tag)',
    padding: 0,
    cursor: 'help',
    marginLeft: 5,
    backgroundColor: { default: 'transparent', ':hover': 'var(--color-surface-sunken)' },
    color: { default: 'var(--color-text-faint)', ':hover': 'var(--color-text)' },
  },
  hintCompact: {
    width: 18,
    height: 18,
    flexBasis: 18,
    marginLeft: 0,
  },
  hintMismatch: {
    color: { default: 'var(--color-warning)', ':hover': 'var(--color-warning)' },
  },
  costLine: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 5,
  },
  stateWarning: {
    color: 'var(--color-warning)',
  },
  tokens: {
    display: 'flex',
    minWidth: 0,
    justifyContent: 'flex-start',
    fontFamily: 'var(--font-mono)',
  },
  tokenValues: {
    display: 'flex',
    minWidth: 0,
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 5,
  },
  tokenLine: {
    display: 'flex',
    minWidth: 0,
    flexShrink: 0,
    alignItems: 'center',
    gap: 5,
    whiteSpace: 'nowrap',
  },
  tokenSeparator: {
    color: 'var(--color-text-faint)',
    marginInline: 2,
  },
  tokenHints: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 3,
  },
  cacheRate: {
    width: 'auto',
    maxWidth: '100%',
    height: 20,
    flexBasis: 'auto',
    justifyContent: 'flex-start',
    gap: 3,
    fontSize: 'var(--text-label-xs)',
    marginLeft: 0,
    whiteSpace: 'nowrap',
  },
  cacheState: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    whiteSpace: 'nowrap',
  },
  timingSlow: {
    color: 'var(--color-warning)',
  },
  timingFaulty: {
    color: 'var(--color-danger)',
  },
  timingMeta: {
    display: 'block',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  action: {
    display: 'grid',
    minWidth: 0,
    justifySelf: { default: 'end', [CARD]: 'stretch' },
    gridColumn: { [CARD]: '1 / -1' },
    gridTemplateColumns: { [CARD]: 'subgrid' },
    alignItems: { [CARD]: 'start' },
  },
  actionValue: {
    justifySelf: { [CARD]: 'end' },
  },
})

function formatLogDay(value: number): string {
  const date = new Date(value)
  const pad = (part: number) => String(part).padStart(2, '0')
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`
}

function formatLogTime(value: number): string {
  const date = new Date(value)
  const pad = (part: number) => String(part).padStart(2, '0')
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
}

function CellLabel({ children }: { children: ReactNode }) {
  return (
    <span aria-hidden {...stylex.props(styles.cellLabel)}>
      {children}
    </span>
  )
}

// Classic StatusBadge tones map onto DS Badge variants; classic compact badges
// always carry the tone icon.
const badgeVariants = {
  success: 'success',
  error: 'error',
  warning: 'warning',
  neutral: 'neutral',
} as const
const badgeIcons = {
  success: CircleCheck,
  error: CircleOff,
  warning: CircleAlert,
  neutral: CircleHelp,
} as const
type StatusTone = keyof typeof badgeVariants

function StatusBadge({ tone, label }: { tone: StatusTone; label: string }) {
  const ToneIcon: LucideIcon = badgeIcons[tone]
  return (
    <Badge variant={badgeVariants[tone]} icon={<ToneIcon size={12} aria-hidden />} label={label} />
  )
}

export interface LogsTableProps {
  logs: readonly RequestLogItemDto[]
  isAccessKey: boolean
  groupNames: Record<number, string>
  groupProviderUrls: Record<number, string>
  /** Groups options query resolved — a name still missing means deleted. */
  groupsLoaded: boolean
  channelsByID: Record<string, ChannelDto>
  onFilterGroup: (groupId: number) => void
  onFilterCredential: (credentialId: number) => void
  onFilterClientModel: (clientModel: string) => void
  onFilterAffinityKey: (affinityKey: string | null) => void
  onOpenDetail: (requestId: string) => void
}

type LogRowContext = Omit<LogsTableProps, 'logs' | 'isAccessKey'>

const LogRow = memo(function LogRow({
  log,
  rowIndex,
  newDay,
  isAccessKey,
  ctx,
}: {
  log: RequestLogItemDto
  rowIndex: number
  newDay: boolean
  isAccessKey: boolean
  ctx: LogRowContext
}) {
  const t = useT()
  const intl = useIntl()
  const locale = intl.locale
  const translate = (key: string, named?: Record<string, string | number>) =>
    t(key as MessageId, named)

  const groupName = log.group_id === null ? null : (ctx.groupNames[log.group_id] ?? null)
  const groupDeleted = log.group_id !== null && ctx.groupsLoaded && groupName === null
  const channel = log.channel_id === null ? null : (ctx.channelsByID[log.channel_id] ?? null)

  const responseLabel = (() => {
    if (log.status === 'success') return t('monitor.logs.response.normal')
    if (log.status === 'error') {
      return log.stream && log.status_code === 200
        ? t('monitor.logs.response.streamError')
        : t('monitor.logs.response.errorWithCode', { code: log.status_code })
    }
    return t(`monitor.logs.status.${log.status}` as MessageId)
  })()

  // The meta line only carries information the badge does not already show.
  const responseMeta = (() => {
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
  })()

  const modelConsistencyTooltip = t(
    log.model_consistency === 'mismatch'
      ? 'monitor.logs.modelConsistency.mismatchTooltip'
      : 'monitor.logs.modelConsistency.unknownTooltip',
    {
      upstream: log.upstream_model ?? '—',
      reported: log.upstream_reported_model ?? t('monitor.logs.modelConsistency.notObserved'),
    },
  )
  const modelConsistencyLabel = t(
    log.model_consistency === 'mismatch'
      ? 'monitor.logs.modelConsistency.mismatchLabel'
      : 'monitor.logs.modelConsistency.unknownLabel',
  )

  const showAffinityObservation =
    log.affinity_hit ||
    log.continuity_hit ||
    log.affinity_source !== 'none' ||
    log.affinity_state !== 'no_signal'

  // Bounded source/state pair only — raw prompt_cache_key, derived keys and
  // HMAC inputs never reach the UI (same rule as classic affinityTooltip).
  const affinityTooltip = [
    t('monitor.logs.affinitySourceLabel', {
      source: t(`monitor.logs.affinitySource.${log.affinity_source}` as MessageId),
    }),
    t('monitor.logs.affinityStateLabel', {
      state: t(`monitor.logs.affinityState.${log.affinity_state}` as MessageId),
    }),
    log.continuity_hit ? t('monitor.logs.continuityHit') : '',
    log.affinity_hit ? t('monitor.logs.drawer.affinity') : '',
  ]
    .filter(Boolean)
    .join('\n')

  const reasoningLabel = (() => {
    if (log.reasoning === null) return ''
    if (
      log.reasoning.mode === 'disabled' ||
      log.reasoning.effort === 'none' ||
      (log.reasoning.budget_tokens !== null &&
        reasoningBudgetSemantic(log.reasoning.budget_tokens) === 'disabled')
    ) {
      return t('monitor.logs.reasoning.compact', { value: 'disabled' })
    }
    return t('monitor.logs.reasoning.compact', {
      value: formatLogReasoning(log, locale),
    })
  })()

  // 列表行比例固定一位小数（classic cacheRateLabel）；抽屉与汇总仍用 shared
  // formatCacheHitRate 的零位裁剪格式。
  const cacheRateLabel = (): string => {
    const rate = cacheHitRate(log.cache_read_tokens, log.input_tokens)
    return rate === null
      ? '—'
      : new Intl.NumberFormat(locale, {
          style: 'percent',
          minimumFractionDigits: 1,
          maximumFractionDigits: 1,
        }).format(rate)
  }

  const cacheTooltip = () => {
    const details = (
      [
        ['monitor.logs.tokens.cacheRead', log.cache_read_tokens],
        ['monitor.logs.tokens.cacheWrite5m', log.cache_write_5m_tokens],
        ['monitor.logs.tokens.cacheWrite1h', log.cache_write_1h_tokens],
        ['monitor.logs.tokens.cacheWrite', log.cache_write_unknown_tokens],
      ] as const
    )
      .filter(([, value]) => value !== '0')
      .map(([key, value]) => `${t(key)} ${formatLogTokenCount(value, locale)}`)
    details.push(`${t('monitor.logs.tokens.cacheHitRate')} ${cacheRateLabel()}`)
    details.push(t('monitor.logs.tokens.cacheRecordedHint'))
    return details.join('\n')
  }

  const timingPrimary =
    log.stream && log.first_response_ms !== null
      ? `${formatLogDuration(log.first_response_ms)} / ${formatLogDuration(log.duration_ms)}`
      : formatLogDuration(log.duration_ms)

  const costState = requestLogCostDisplayState(log)
  const costLabel =
    costState === 'complete' ? formatEstimatedCost(log.estimated_cost_nano_usd, locale) : '—'
  const usageState = requestLogUsageDisplayState(log)
  const outputRate = formatLogOutputRate(log, locale)
  const responseTip = requestLogResponseTooltip(log, translate)
  const responseTipVisible = requestLogResponseTooltipVisible(log)

  const statusTone: StatusTone =
    log.status === 'success'
      ? 'success'
      : log.status === 'error'
        ? 'error'
        : log.status === 'incomplete'
          ? 'warning'
          : 'neutral'

  return (
    <article
      {...stylex.props(styles.record, rowIndex === 2 && styles.recordFirst)}
      data-testid="logs-list__record"
      role="row"
      aria-rowindex={rowIndex}
    >
      <div role="cell" {...stylex.props(styles.cell, styles.time)} data-testid="logs-list__time">
        <CellLabel>{t('monitor.logs.columns.time')}</CellLabel>
        <span {...stylex.props(styles.cellValue)}>
          {newDay && (
            <small {...stylex.props(styles.dayLabel)}>{formatLogDay(log.completed_at_ms)}</small>
          )}
          <time
            dateTime={formatISOInstant(log.completed_at_ms)}
            title={`${formatLocalInstantWithSeconds(log.completed_at_ms)} · ${currentTimeZone()}`}
            aria-label={`${formatLocalInstantWithSeconds(log.completed_at_ms)} · ${currentTimeZone()}`}
            tabIndex={0}
          >
            {formatLogTime(log.completed_at_ms)}
          </time>
        </span>
      </div>

      {!isAccessKey && (
        <div
          role="cell"
          {...stylex.props(styles.cell, styles.affinityKeyCell)}
          data-testid="logs-list__affinity-key-cell"
        >
          <CellLabel>{t('monitor.logs.columns.affinityKey')}</CellLabel>
          <span {...stylex.props(styles.cellValue)}>
            {log.affinity_key !== null ? (
              <Tooltip content={log.affinity_key}>
                <button
                  type="button"
                  {...stylex.props(styles.affinityKey, styles.affinityKeyButton)}
                  aria-label={t('monitor.logs.filterAffinityKey', {
                    value: log.affinity_key,
                  })}
                  title={log.affinity_key}
                  data-testid="logs-affinity-key-filter"
                  onClick={() => ctx.onFilterAffinityKey(log.affinity_key)}
                >
                  …{log.affinity_key.slice(-6)}
                </button>
              </Tooltip>
            ) : (
              <code {...stylex.props(styles.affinityKey)}>—</code>
            )}
          </span>
        </div>
      )}

      {!isAccessKey && (
        <div role="cell" {...stylex.props(styles.cell)}>
          <CellLabel>{t('monitor.logs.columns.route')}</CellLabel>
          <span {...stylex.props(styles.cellValue)}>
            <LogRouteIdentity
              groupId={log.group_id}
              groupName={groupName}
              providerUrl={
                log.group_id === null ? null : (ctx.groupProviderUrls[log.group_id] ?? null)
              }
              channelId={log.channel_id}
              channel={channel}
              credentialId={log.credential_id}
              credentialName={log.credential_name}
              groupDeleted={groupDeleted}
              groupResolved={ctx.groupsLoaded}
              credentialDeleted={log.credential_id !== null && log.credential_name === ''}
              filterable
              onFilterGroup={ctx.onFilterGroup}
              onFilterCredential={ctx.onFilterCredential}
            />
          </span>
        </div>
      )}

      <div role="cell" {...stylex.props(styles.cell)}>
        <CellLabel>{t('monitor.logs.columns.modelProtocol')}</CellLabel>
        <span {...stylex.props(styles.cellValue, styles.inline)}>
          {log.client_model ? (
            <Tooltip content={log.client_model}>
              <button
                type="button"
                {...stylex.props(styles.model, styles.modelButton)}
                data-testid="logs-list__model"
                aria-label={t('monitor.logs.filterModel', { name: log.client_model })}
                onClick={() =>
                  log.client_model !== null && ctx.onFilterClientModel(log.client_model)
                }
              >
                {log.client_model}
              </button>
            </Tooltip>
          ) : (
            <code {...stylex.props(styles.model)} data-testid="logs-list__model">
              —
            </code>
          )}
          {log.upstream_model !== null && log.upstream_model !== log.client_model && (
            <Tooltip content={log.upstream_model}>
              <span {...stylex.props(styles.modelMapping)} data-testid="logs-list__model-mapping">
                -&gt;{log.upstream_model}
              </span>
            </Tooltip>
          )}
          {(log.model_consistency === 'unknown' || log.model_consistency === 'mismatch') && (
            <Tooltip content={modelConsistencyTooltip}>
              <button
                type="button"
                {...stylex.props(
                  styles.hint,
                  log.model_consistency === 'mismatch' && styles.hintMismatch,
                )}
                aria-label={modelConsistencyLabel}
              >
                {log.model_consistency === 'mismatch' ? (
                  <TriangleAlert size={14} aria-hidden />
                ) : (
                  <CircleHelp size={13} aria-hidden />
                )}
              </button>
            </Tooltip>
          )}
        </span>
        <span {...stylex.props(styles.protocolLine)}>
          <Tooltip content={log.protocol}>
            <span {...stylex.props(styles.protocol)} data-testid="logs-list__protocol" tabIndex={0}>
              {requestLogProtocolLabel(log.protocol)}
            </span>
          </Tooltip>
          {reasoningLabel !== '' && (
            <Tooltip content={reasoningLabel}>
              <small {...stylex.props(styles.reasoning)}>{reasoningLabel}</small>
            </Tooltip>
          )}
          <LogProtocolConversion
            mode={log.route_mode}
            clientProtocol={log.protocol}
            upstreamProtocol={log.upstream_protocol}
          />
        </span>
      </div>

      <div role="cell" {...stylex.props(styles.cell)}>
        <CellLabel>{t('monitor.logs.columns.response')}</CellLabel>
        <span {...stylex.props(styles.cellValue)}>
          <span {...stylex.props(styles.responsePrimary)}>
            {responseTipVisible ? (
              <Tooltip content={responseTip}>
                <span>
                  <StatusBadge tone={statusTone} label={responseLabel} />
                </span>
              </Tooltip>
            ) : (
              <StatusBadge tone={statusTone} label={responseLabel} />
            )}
            {showAffinityObservation && (
              <Tooltip content={affinityTooltip}>
                <span
                  {...stylex.props(styles.hint, styles.hintCompact)}
                  tabIndex={0}
                  aria-label={affinityTooltip}
                >
                  {log.affinity_hit ? (
                    <Magnet size={13} aria-hidden />
                  ) : log.continuity_hit ? (
                    <ArrowRight size={13} aria-hidden />
                  ) : (
                    <Info size={13} aria-hidden />
                  )}
                </span>
              </Tooltip>
            )}
          </span>
          {responseMeta !== '' && (
            <Tooltip content={responseTip}>
              <small {...stylex.props(styles.responseMeta)} data-testid="logs-list__response-meta">
                {responseMeta}
              </small>
            </Tooltip>
          )}
        </span>
      </div>

      <div role="cell" {...stylex.props(styles.cell)}>
        <CellLabel>{t('monitor.logs.columns.cost')}</CellLabel>
        <span {...stylex.props(styles.cellValue, styles.costLine)}>
          <Tooltip content={costLabel}>
            <span
              data-tone={costState !== 'complete' ? 'warning' : undefined}
              {...stylex.props(styles.ellipsis, costState !== 'complete' && styles.stateWarning)}
            >
              {costLabel}
            </span>
          </Tooltip>
          <PricingModeIndicator
            mode={log.pricing_mode}
            contextThresholdTokens={log.context_threshold_tokens}
            xstyle={[styles.hint, styles.hintCompact]}
          />
        </span>
      </div>

      <div role="cell" {...stylex.props(styles.cell)}>
        <CellLabel>{t('monitor.logs.columns.tokens')}</CellLabel>
        <span {...stylex.props(styles.cellValue)}>
          {usageState === 'reported' ? (
            <Tooltip
              content={`${t('monitor.logs.tokens.input')}: ${formatLogTokenCount(log.input_tokens, locale)}\n${t('monitor.logs.tokens.output')}: ${formatLogTokenCount(log.output_tokens, locale)}`}
            >
              <span
                {...stylex.props(styles.tokens, styles.tokenValues)}
                data-testid="logs-list__token-values"
              >
                <span {...stylex.props(styles.tokenLine)}>
                  {formatLogTokenCount(log.input_tokens, locale)}
                  <span {...stylex.props(styles.tokenSeparator)} aria-hidden>
                    /
                  </span>
                  {formatLogTokenCount(log.output_tokens, locale)}
                </span>
                <span {...stylex.props(styles.tokenHints)}>
                  {log.usage_state === 'partial' && (
                    <Tooltip content={t('monitor.logs.tokens.partial')}>
                      <button
                        type="button"
                        {...stylex.props(styles.hint, styles.hintCompact)}
                        aria-label={t('monitor.logs.tokens.partial')}
                      >
                        <CircleHelp size={13} aria-hidden />
                      </button>
                    </Tooltip>
                  )}
                  {log.usage_state === 'complete' || hasRequestLogCache(log) ? (
                    <Tooltip content={cacheTooltip()}>
                      <button
                        type="button"
                        {...stylex.props(styles.hint, styles.cacheRate)}
                        data-testid="logs-list__cache-rate"
                        aria-label={`${t('monitor.logs.tokens.cacheHitRate')} ${cacheRateLabel()} · ${t('monitor.logs.tokens.cacheDetails')}`}
                      >
                        <Layers size={12} aria-hidden />
                        {t('monitor.logs.tokens.cacheHitRate')} {cacheRateLabel()}
                      </button>
                    </Tooltip>
                  ) : (
                    <small
                      {...stylex.props(styles.cacheState)}
                      data-testid="logs-list__cache-state"
                    >
                      {t('monitor.logs.tokens.cacheUnavailable')}
                    </small>
                  )}
                </span>
              </span>
            </Tooltip>
          ) : (
            <>
              <span
                {...stylex.props(styles.stateWarning)}
                data-testid="logs-list__state"
                data-tone="warning"
              >
                —
              </span>
              <small {...stylex.props(styles.cacheState)} data-testid="logs-list__cache-state">
                {t(
                  log.usage_state === 'not_applicable'
                    ? 'monitor.logs.filters.usageState.not_applicable'
                    : 'monitor.logs.tokens.cacheUnavailable',
                )}
              </small>
            </>
          )}
        </span>
      </div>

      <div role="cell" {...stylex.props(styles.cell)}>
        <CellLabel>{t('monitor.logs.columns.timing')}</CellLabel>
        <span {...stylex.props(styles.cellValue)}>
          <Tooltip content={timingPrimary}>
            <span
              data-testid="logs-list__timing"
              data-tone={
                log.feedback_status === 'slow' || log.feedback_status === 'faulty'
                  ? log.feedback_status
                  : undefined
              }
              {...stylex.props(
                log.feedback_status === 'slow'
                  ? styles.timingSlow
                  : log.feedback_status === 'faulty'
                    ? styles.timingFaulty
                    : null,
              )}
            >
              {log.stream && log.first_response_ms !== null ? (
                <>
                  {formatLogDuration(log.first_response_ms)}
                  <span aria-hidden> / </span>
                  {formatLogDuration(log.duration_ms)}
                </>
              ) : (
                formatLogDuration(log.duration_ms)
              )}
            </span>
          </Tooltip>
          {outputRate !== '—' && (
            <Tooltip content={outputRate}>
              <small {...stylex.props(styles.timingMeta)}>{outputRate}</small>
            </Tooltip>
          )}
        </span>
      </div>

      <div role="cell" {...stylex.props(styles.action)}>
        <CellLabel>{t('monitor.logs.columns.actions')}</CellLabel>
        <span {...stylex.props(styles.cellValue, styles.actionValue)}>
          <IconButton
            id={`log-details-${log.request_id}`}
            variant="ghost"
            size="sm"
            label={t('monitor.logs.details')}
            icon={<ArrowRight size={16} aria-hidden />}
            onClick={() => ctx.onOpenDetail(log.request_id)}
          />
        </span>
      </div>
    </article>
  )
})

export function LogsTable(props: LogsTableProps) {
  const t = useT()
  const { logs, isAccessKey, ...ctx } = props

  return (
    <div
      {...stylex.props(styles.list, isAccessKey ? styles.listScoped : styles.listAdmin)}
      role="table"
      data-testid="logs-list"
      aria-label={t('monitor.logs.caption')}
      aria-rowcount={logs.length + 1}
    >
      <div {...stylex.props(styles.headRow)} role="row" aria-rowindex={1}>
        <span
          role="columnheader"
          aria-label={t('monitor.logs.columns.timeNewestFirst')}
          {...stylex.props(styles.headCell)}
        >
          {t('monitor.logs.columns.time')}
        </span>
        {!isAccessKey && (
          <span role="columnheader" {...stylex.props(styles.headCell)}>
            {t('monitor.logs.columns.affinityKey')}
          </span>
        )}
        {!isAccessKey && (
          <span role="columnheader" {...stylex.props(styles.headCell)}>
            {t('monitor.logs.columns.route')}
          </span>
        )}
        <span role="columnheader" {...stylex.props(styles.headCell)}>
          {t('monitor.logs.columns.modelProtocol')}
        </span>
        <span role="columnheader" {...stylex.props(styles.headCell)}>
          {t('monitor.logs.columns.response')}
        </span>
        <span role="columnheader" {...stylex.props(styles.headCell)}>
          {t('monitor.logs.columns.cost')}
        </span>
        <span
          role="columnheader"
          aria-label={t('monitor.logs.columns.tokensDetail')}
          {...stylex.props(styles.headCell)}
        >
          {t('monitor.logs.columns.tokens')}
        </span>
        <span
          role="columnheader"
          aria-label={t('monitor.logs.columns.timingDetail')}
          {...stylex.props(styles.headCell)}
        >
          {t('monitor.logs.columns.timing')}
        </span>
        <span role="columnheader" {...stylex.props(styles.headCell)}>
          {t('monitor.logs.columns.actions')}
        </span>
      </div>

      {logs.map((log, index) => {
        const previous = logs[index - 1]?.completed_at_ms
        const a = new Date(log.completed_at_ms)
        const b = previous === undefined ? null : new Date(previous)
        const newDay =
          index === 0 ||
          b === null ||
          a.getFullYear() !== b.getFullYear() ||
          a.getMonth() !== b.getMonth() ||
          a.getDate() !== b.getDate()
        return (
          <LogRow
            key={log.request_id}
            log={log}
            rowIndex={index + 2}
            newDay={newDay}
            isAccessKey={isAccessKey}
            ctx={ctx}
          />
        )
      })}
    </div>
  )
}
