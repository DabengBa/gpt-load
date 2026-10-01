import * as stylex from '@stylexjs/stylex'
import { Badge } from '@astryxdesign/core'
import { CircleAlert, CircleCheck, CircleHelp, CircleOff, type LucideIcon } from 'lucide-react'
import { useIntl } from 'react-intl'

import type { RequestLogHealthDto } from '@shared/control/resources/health'

import { useT } from '../../app/i18n'
import { RelativeInstant } from '../../components/RelativeInstant'
import { MonitorSectionHeading } from './MonitorSectionHeading'

// Classic media queries: the card leaves the shared focus grid below 1100px,
// the metrics collapse to one row of four in the 761–1099px band, and the
// status/checkpoint rows stack below 760px.
const BAND = '@media (max-width: 1099px) and (min-width: 761px)'
const STACKED = '@media (max-width: 1099px)'
const TIGHT = '@media (max-width: 760px)'

const styles = stylex.create({
  section: {
    display: 'grid',
    minWidth: 0,
    gridTemplateRows: {
      default: 'auto var(--health-focus-content-height, 266px)',
      [STACKED]: 'auto auto',
    },
    gap: 'var(--space-4)',
  },
  // Classic Surface padded=false + `.request-log-health__card`.
  card: {
    display: 'flex',
    minWidth: 0,
    height: { default: '100%', [STACKED]: 'auto' },
    flexDirection: 'column',
    overflow: 'hidden',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-card)',
    backgroundColor: 'var(--color-surface)',
  },
  status: {
    display: 'flex',
    minWidth: 0,
    minHeight: 52,
    alignItems: { default: 'center', [TIGHT]: 'flex-start' },
    flexDirection: { [TIGHT]: 'column' },
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    backgroundColor: 'color-mix(in srgb, var(--color-surface-sunken) 48%, var(--color-surface))',
    paddingBlock: 10,
    paddingInline: 14,
  },
  failures: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
    whiteSpace: 'nowrap',
  },
  count: {
    color: 'var(--color-text)',
    fontFamily: 'var(--font-mono)',
    fontWeight: 600,
  },
  countDanger: {
    color: 'var(--color-danger)',
  },
  checkpoint: {
    display: 'flex',
    minWidth: 0,
    alignItems: { default: 'center', [TIGHT]: 'flex-start' },
    flexDirection: { [TIGHT]: 'column' },
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    paddingBlock: 8,
    paddingInline: 14,
  },
  checkpointDetail: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  checkpointRisk: {
    minWidth: 0,
    marginTop: 0,
    marginRight: 0,
    marginBottom: 0,
    marginLeft: { default: 'auto', [TIGHT]: 0 },
    color: 'var(--color-danger)',
    fontSize: 'var(--text-xs)',
    lineHeight: 1.35,
    textAlign: { default: 'right', [TIGHT]: 'left' },
  },
  metrics: {
    display: 'grid',
    flex: '1 1 auto',
    minHeight: 0,
    gridTemplateColumns: {
      default: 'repeat(2, minmax(0, 1fr))',
      [BAND]: 'repeat(4, minmax(0, 1fr))',
    },
    gridTemplateRows: {
      default: 'repeat(2, minmax(68px, 1fr))',
      [BAND]: 'minmax(72px, auto)',
    },
    margin: 0,
    backgroundColor: 'var(--color-surface)',
  },
  metric: {
    display: 'grid',
    minWidth: 0,
    alignContent: 'center',
    gap: 'var(--space-2)',
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    borderLeftWidth: 1,
    borderLeftStyle: 'solid',
    borderLeftColor: 'var(--color-border-subtle)',
    paddingBlock: 10,
    paddingInline: 12,
    minHeight: { [TIGHT]: 76 },
  },
  metricTerm: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  metricValue: {
    margin: 0,
    color: 'var(--color-text)',
    fontFamily: 'var(--font-mono)',
    fontSize: 22,
    fontVariantNumeric: 'tabular-nums',
    fontWeight: 560,
    letterSpacing: '-0.02em',
    lineHeight: 1,
    overflowWrap: 'anywhere',
  },
  metricValueDanger: {
    color: 'var(--color-danger)',
  },
})

// The classic borders come from :nth-child(-n+2)/:nth-child(odd) plus the
// 761–1099px band overrides (:nth-child(n) / :first-child). Positional
// pseudo-classes are not in the stylex allowlist, so each cell gets its own
// deterministic border style; stylex merges per-property, so a flat value
// here replaces the base value under every condition.
const metricBorders = stylex.create({
  // First cell: never a top border (first row) and never a left border
  // (odd column in the 2-up layout, :first-child in the 4-up band).
  first: { borderTopWidth: 0, borderLeftWidth: 0 },
  second: { borderTopWidth: 0 },
  // Third cell: top border only in the 2-up layout; in the band it loses the
  // top border and gains a left border (it is no longer first in the row).
  third: {
    borderTopWidth: { default: 1, [BAND]: 0 },
    borderLeftWidth: { default: 0, [BAND]: 1 },
  },
  // Fourth cell: left border everywhere; top border only in the 2-up layout.
  fourth: { borderTopWidth: { default: 1, [BAND]: 0 } },
})
const metricBorderStyles = [
  metricBorders.first,
  metricBorders.second,
  metricBorders.third,
  metricBorders.fourth,
] as const

type StatusTone = 'neutral' | 'success' | 'warning' | 'danger'

const badgeVariants = {
  neutral: 'neutral',
  success: 'success',
  warning: 'warning',
  danger: 'error',
} as const

// Classic StatusBadge always renders a tone icon (12px at compact size).
const badgeIcons: Record<StatusTone, LucideIcon> = {
  neutral: CircleHelp,
  success: CircleCheck,
  warning: CircleAlert,
  danger: CircleOff,
}

/**
 * Request-log collection health — classic RequestLogHealthCard.vue: status
 * header with write-failure count, the quota-checkpoint degradation strip,
 * and the 2×2 metric grid (4-up in the 761–1099px band).
 */
export function RequestLogHealthCard({ stats }: { stats: RequestLogHealthDto }) {
  const t = useT()
  const intl = useIntl()
  const n = (value: number): string => intl.formatNumber(value)

  // Danger outranks warning: dropped/write failures are abnormal first,
  // retention failures only degrade to warning when nothing else is wrong.
  const state: { tone: StatusTone; label: string } =
    stats.dropped_total > 0 || stats.write_failure_total > 0
      ? { tone: 'danger', label: t('monitor.health.requestLog.abnormal') }
      : stats.retention_delete_failure_total > 0
        ? { tone: 'warning', label: t('monitor.health.requestLog.retentionAbnormal') }
        : { tone: 'success', label: t('monitor.health.requestLog.normal') }

  const metrics = [
    {
      key: 'queue',
      label: t('monitor.health.requestLog.queue'),
      value: `${n(stats.queue_depth)} / ${n(stats.queue_capacity)}`,
      danger: false,
    },
    {
      key: 'enqueued',
      label: t('monitor.health.requestLog.enqueued'),
      value: n(stats.enqueued_total),
      danger: false,
    },
    {
      key: 'persisted',
      label: t('monitor.health.requestLog.persisted'),
      value: n(stats.persisted_total),
      danger: false,
    },
    {
      key: 'dropped',
      label: t('monitor.health.requestLog.droppedTotal'),
      value: n(stats.dropped_total),
      danger: stats.dropped_total > 0,
    },
  ]

  const StateIcon = badgeIcons[state.tone]

  return (
    <section {...stylex.props(styles.section)} aria-labelledby="request-log-health-title">
      <MonitorSectionHeading
        id="request-log-health-title"
        title={t('monitor.health.requestLog.title')}
        description={t('monitor.health.requestLog.description')}
      />

      <div {...stylex.props(styles.card)}>
        <header {...stylex.props(styles.status)}>
          <Badge
            variant={badgeVariants[state.tone]}
            icon={<StateIcon size={12} aria-hidden="true" />}
            label={state.label}
          />
          <span {...stylex.props(styles.failures)}>
            {t('monitor.health.requestLog.writeFailures')}{' '}
            <strong
              {...stylex.props(styles.count, stats.write_failure_total > 0 && styles.countDanger)}
            >
              {n(stats.write_failure_total)}
            </strong>
          </span>
        </header>

        {stats.access_quota_checkpoint_degraded && (
          <div {...stylex.props(styles.checkpoint)}>
            <Badge
              variant="error"
              icon={<CircleOff size={12} aria-hidden="true" />}
              label={t('monitor.health.requestLog.checkpointAbnormal')}
            />
            <span {...stylex.props(styles.checkpointDetail)}>
              {t('monitor.health.requestLog.checkpointFailures')}{' '}
              {/* The classic danger class is conditional on the degraded flag,
                  which is always true inside this block. */}
              <strong {...stylex.props(styles.count, styles.countDanger)}>
                {n(stats.access_quota_checkpoint_write_failure_total)}
              </strong>
              {stats.last_access_quota_checkpoint_write_failure_at_ms !== null && (
                <>
                  {' · '}
                  {t('monitor.health.requestLog.lastCheckpointFailureAt')}{' '}
                  <RelativeInstant
                    instant={stats.last_access_quota_checkpoint_write_failure_at_ms}
                    emptyLabel="—"
                    hint
                  />
                </>
              )}
            </span>
            {/* Same tautological v-if as the strong above — always rendered
                inside the degraded block. */}
            <p {...stylex.props(styles.checkpointRisk)}>
              {t('monitor.health.requestLog.checkpointRisk')}
            </p>
          </div>
        )}

        <dl {...stylex.props(styles.metrics)}>
          {metrics.map((metric, index) => (
            <div key={metric.key} {...stylex.props(styles.metric, metricBorderStyles[index])}>
              <dt {...stylex.props(styles.metricTerm)}>{metric.label}</dt>
              <dd {...stylex.props(styles.metricValue, metric.danger && styles.metricValueDanger)}>
                {metric.value}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  )
}
