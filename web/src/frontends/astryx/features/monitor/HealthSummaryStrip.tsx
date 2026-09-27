import * as stylex from '@stylexjs/stylex'
import { Tooltip } from '@astryxdesign/core'
import type { CSSProperties } from 'react'
import { useIntl } from 'react-intl'

import type { HealthCredentialCountsDto } from '@shared/control/resources/health'

import { useT } from '../../app/i18n'

const MEDIUM = '@media (max-width: 900px)'
const NARROW = '@media (max-width: 560px)'

type StatusTone = 'neutral' | 'success' | 'warning' | 'danger'

interface CooldownRecovery {
  relative: string
  exact: string
}

interface HealthOverviewItem {
  key: string
  label: string
  value: string
  detail: string
  tooltip: string | undefined
  tone: StatusTone
}

const styles = stylex.create({
  overview: {
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
  item: {
    display: 'grid',
    minWidth: 0,
    minHeight: {
      default: 108,
      [MEDIUM]: 104,
      [NARROW]: 94,
    },
    alignContent: 'start',
    gap: 0,
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
    backgroundColor: 'var(--health-overview-dot)',
  },
  value: {
    display: 'block',
    marginTop: 8,
    color: 'var(--health-overview-color)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'clamp(1.45rem, 2.1vw, 1.9rem)',
    fontVariantNumeric: 'tabular-nums',
    fontWeight: 580,
    letterSpacing: '-0.045em',
    lineHeight: 1,
  },
  detail: {
    minWidth: 0,
    marginTop: 9,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    lineHeight: 1.4,
    overflowWrap: 'anywhere',
  },
  detailTooltip: {
    width: 'fit-content',
    borderRadius: 3,
  },
})

// Classic carries the tone through per-item --health-overview-color /
// --health-overview-dot custom properties; the same vars are set inline here.
const toneColors: Record<StatusTone, { color: string; dot: string }> = {
  neutral: { color: 'var(--color-text)', dot: 'var(--color-text-faint)' },
  success: { color: 'var(--color-success)', dot: 'var(--color-success)' },
  warning: { color: 'var(--color-warning)', dot: 'var(--color-warning)' },
  danger: { color: 'var(--color-danger)', dot: 'var(--color-danger)' },
}

/**
 * Four-tile health overview — classic HealthSummaryStrip.vue. Credentials /
 * available / cooldown / blacklisted with neutral/success/warning/danger tones.
 */
export function HealthSummaryStrip({
  counts,
  earliestCooldown,
}: {
  counts: HealthCredentialCountsDto
  earliestCooldown: CooldownRecovery | null
}) {
  const intl = useIntl()
  const t = useT()
  const n = (value: number): string => intl.formatNumber(value)

  const items: HealthOverviewItem[] = [
    {
      key: 'credentials',
      label: t('monitor.health.overview.credentials'),
      value: n(counts.credentials),
      detail: t('monitor.health.overview.credentialsDescription'),
      tooltip: undefined,
      tone: 'neutral',
    },
    {
      key: 'available',
      label: t('monitor.health.overview.available'),
      value: n(counts.available),
      detail: t('monitor.health.overview.availableDescription'),
      tooltip: undefined,
      tone: counts.available > 0 ? 'success' : counts.credentials === 0 ? 'neutral' : 'danger',
    },
    {
      key: 'cooldown',
      label: t('monitor.health.overview.cooldown'),
      value: n(counts.cooldown),
      detail:
        earliestCooldown === null
          ? t('monitor.health.overview.cooldownClear')
          : t('monitor.health.overview.cooldownRecovery', {
              time: earliestCooldown.relative,
            }),
      tooltip: earliestCooldown?.exact,
      tone: counts.cooldown > 0 ? 'warning' : 'neutral',
    },
    {
      key: 'blacklisted',
      label: t('monitor.health.overview.blacklisted'),
      value: n(counts.blacklisted),
      detail:
        counts.blacklisted > 0
          ? t('monitor.health.overview.blacklistedRecovery')
          : t('monitor.health.overview.blacklistedClear'),
      tooltip: undefined,
      tone: counts.blacklisted > 0 ? 'danger' : 'neutral',
    },
  ]

  return (
    <section {...stylex.props(styles.overview)} aria-label={t('monitor.health.overview.label')}>
      {items.map((item) => {
        const colors = toneColors[item.tone]
        return (
          <article
            key={item.key}
            {...stylex.props(styles.item)}
            style={
              {
                '--health-overview-color': colors.color,
                '--health-overview-dot': colors.dot,
              } as CSSProperties
            }
          >
            <span {...stylex.props(styles.label)}>
              <span {...stylex.props(styles.labelDot)} aria-hidden="true" />
              {item.label}
            </span>
            <strong {...stylex.props(styles.value)}>{item.value}</strong>
            {item.tooltip !== undefined ? (
              <Tooltip content={item.tooltip} placement="below">
                <span {...stylex.props(styles.detail, styles.detailTooltip)} tabIndex={0}>
                  {item.detail}
                </span>
              </Tooltip>
            ) : (
              <span {...stylex.props(styles.detail)}>{item.detail}</span>
            )}
          </article>
        )
      })}
    </section>
  )
}
