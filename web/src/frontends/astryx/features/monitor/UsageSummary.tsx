import * as stylex from '@stylexjs/stylex'
import type { CSSProperties } from 'react'
import { useIntl } from 'react-intl'

import type { UsageAggregateDto } from '@shared/control/resources/usage'
import { formatCacheHitRate } from '@shared/lib/cache-rate'
import { formatEstimatedCost, formatInteger, formatPercent, formatTokens } from '@shared/lib/format'

import { useT } from '../../app/i18n'

const MEDIUM = '@media (max-width: 900px)'
const NARROW = '@media (max-width: 560px)'

const styles = stylex.create({
  kpis: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'repeat(4, minmax(0, 1fr))',
      [MEDIUM]: 'repeat(2, minmax(0, 1fr))',
      [NARROW]: 'minmax(0, 1fr)',
    },
    overflow: 'hidden',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-card)',
    backgroundColor: 'var(--color-border-subtle)',
    gap: 1,
  },
  kpi: {
    minWidth: 0,
    minHeight: {
      default: 108,
      [NARROW]: 94,
    },
    backgroundColor: 'var(--color-surface)',
    paddingBlock: 16,
    paddingInline: 18,
  },
  label: {
    display: 'flex',
    alignItems: 'center',
    gap: 7,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 560,
  },
  // Classic paints the dot via a ::before on the label; a real aria-hidden
  // span keeps the DOM count identical for screen readers.
  labelDot: {
    width: 6,
    height: 6,
    flex: '0 0 6px',
    borderRadius: '50%',
    backgroundColor: 'var(--usage-kpi-dot)',
  },
  value: {
    display: 'block',
    marginTop: 8,
    color: 'var(--usage-kpi-color)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'clamp(1.45rem, 2.1vw, 1.9rem)',
    fontVariantNumeric: 'tabular-nums',
    fontWeight: 580,
    letterSpacing: '-0.045em',
    lineHeight: 1,
    overflowWrap: 'anywhere',
  },
  detail: {
    display: 'block',
    marginTop: 9,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    lineHeight: 1.4,
  },
  // Classic declares this display:flex !important to beat the `small` block
  // rule; stylex merge order makes the same override without !important.
  outcomes: {
    display: 'flex',
    alignItems: 'baseline',
    gap: 5,
    whiteSpace: 'nowrap',
  },
  success: {
    color: 'var(--color-success)',
  },
  failure: {
    color: 'var(--color-danger)',
  },
  faint: {
    color: 'var(--color-text-faint)',
  },
})

type KpiTone = 'accent' | 'cache' | 'cost'

// Classic carries the tone through per-tile --usage-kpi-color /
// --usage-kpi-dot custom properties; the same vars are set inline here.
const toneVars: Record<KpiTone, { color: string; dot: string }> = {
  accent: { color: 'var(--color-action)', dot: 'var(--color-action)' },
  cache: { color: 'var(--color-success)', dot: 'var(--color-success)' },
  cost: { color: 'var(--color-warning)', dot: 'var(--color-warning)' },
}

function kpiToneStyle(tone: KpiTone): CSSProperties {
  const vars = toneVars[tone]
  return {
    '--usage-kpi-color': vars.color,
    '--usage-kpi-dot': vars.dot,
  } as CSSProperties
}

/**
 * Four-tile usage KPI summary — classic UsageSummary.vue. Requests with
 * success/failure outcomes, cache hit rate, total tokens, estimated cost.
 */
export function UsageSummary({ summary }: { summary: UsageAggregateDto }) {
  const intl = useIntl()
  const t = useT()
  const locale = intl.locale

  const inputTokens =
    summary.uncached_input_tokens +
    summary.cache_read_tokens +
    summary.cache_write_5m_tokens +
    summary.cache_write_1h_tokens +
    summary.cache_write_unknown_tokens

  return (
    <section {...stylex.props(styles.kpis)} aria-label={t('monitor.usage.kpi.title')}>
      <article {...stylex.props(styles.kpi)} style={kpiToneStyle('accent')}>
        <span {...stylex.props(styles.label)}>
          <span {...stylex.props(styles.labelDot)} aria-hidden="true" />
          {t('monitor.usage.kpi.requests')}
        </span>
        <strong {...stylex.props(styles.value)}>
          {formatInteger(summary.request_count, locale)}
        </strong>
        <small {...stylex.props(styles.detail, styles.outcomes)}>
          <span {...stylex.props(styles.success)}>
            {t('monitor.usage.columns.success')} {formatInteger(summary.success_count, locale)}
          </span>
          <span {...stylex.props(styles.faint)}>/</span>
          <span {...stylex.props(styles.failure)}>
            {t('monitor.usage.columns.failure')} {formatInteger(summary.failure_count, locale)}
          </span>
          <span {...stylex.props(styles.faint)}>
            ({formatPercent(summary.failure_count, summary.request_count, locale)})
          </span>
        </small>
      </article>
      <article {...stylex.props(styles.kpi)} style={kpiToneStyle('cache')}>
        <span {...stylex.props(styles.label)}>
          <span {...stylex.props(styles.labelDot)} aria-hidden="true" />
          {t('monitor.usage.kpi.cacheHitRate')}
        </span>
        <strong {...stylex.props(styles.value)}>
          {formatCacheHitRate(summary.cache_read_tokens, inputTokens, locale)}
        </strong>
        <small {...stylex.props(styles.detail)}>
          {t('monitor.usage.kpi.cacheTokenSummary', {
            read: formatTokens(summary.cache_read_tokens, locale),
            input: formatTokens(inputTokens, locale),
          })}
        </small>
      </article>
      <article {...stylex.props(styles.kpi)} style={kpiToneStyle('accent')}>
        <span {...stylex.props(styles.label)}>
          <span {...stylex.props(styles.labelDot)} aria-hidden="true" />
          {t('monitor.usage.kpi.totalTokens')}
        </span>
        <strong {...stylex.props(styles.value)}>
          {formatTokens(summary.total_tokens, locale)}
        </strong>
        <small {...stylex.props(styles.detail)}>{t('monitor.usage.kpi.persistedWindow')}</small>
      </article>
      <article {...stylex.props(styles.kpi)} style={kpiToneStyle('cost')}>
        <span {...stylex.props(styles.label)}>
          <span {...stylex.props(styles.labelDot)} aria-hidden="true" />
          {t('monitor.usage.kpi.estimatedCost')}
        </span>
        <strong {...stylex.props(styles.value)}>
          {formatEstimatedCost(summary.estimated_cost_nano_usd, locale)}
        </strong>
        <small {...stylex.props(styles.detail)}>{t('monitor.usage.kpi.estimatedCostBasis')}</small>
      </article>
    </section>
  )
}
