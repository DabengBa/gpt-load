import * as stylex from '@stylexjs/stylex'
import { Tooltip } from '@astryxdesign/core'
import { Ban, CircleCheck, CirclePause, Clock3 } from 'lucide-react'
import { useIntl } from 'react-intl'

import type { CredentialCounts, HealthCredentialCountsDto } from '@shared/control/types'

const styles = stylex.create({
  figure: {
    display: 'grid',
    width: 'min(100%, 250px)',
    gap: 7,
    borderRadius: 4,
  },
  counts: {
    display: 'flex',
    alignItems: 'center',
    gap: 7,
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    fontVariantNumeric: 'tabular-nums',
    lineHeight: 1,
  },
  total: {
    color: 'var(--color-text)',
    fontSize: 13,
    fontWeight: 650,
  },
  divider: {
    width: 1,
    height: 14,
    backgroundColor: 'var(--color-border-control)',
  },
  count: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 3,
    fontWeight: 600,
    whiteSpace: 'nowrap',
  },
  available: { color: 'var(--color-success)' },
  cooldown: { color: 'var(--color-warning)' },
  blacklisted: { color: 'var(--color-danger)' },
  disabled: { color: 'var(--color-neutral)' },
  bar: {
    display: 'flex',
    width: '100%',
    height: 'var(--health-bar-height)',
    overflow: 'hidden',
    borderRadius: 999,
    backgroundColor: 'var(--color-neutral-bg)',
  },
  segment: {
    minWidth: 0,
    flexBasis: 0,
  },
})

export function CredentialHealthBar({
  counts,
  label,
}: {
  counts: CredentialCounts | HealthCredentialCountsDto
  label: string
}) {
  const intl = useIntl()
  const normalized: CredentialCounts =
    'total' in counts
      ? counts
      : {
          total: counts.credentials,
          available: counts.available,
          cooldown: counts.cooldown,
          blacklisted: counts.blacklisted,
          disabled: 0,
        }
  const hasVisibleStatus =
    normalized.available + normalized.cooldown + normalized.blacklisted + normalized.disabled > 0
  const n = (value: number) => intl.formatNumber(value)

  const segments = [
    {
      key: 'available',
      value: normalized.available,
      style: styles.available,
      icon: <CircleCheck size={12} />,
    },
    {
      key: 'cooldown',
      value: normalized.cooldown,
      style: styles.cooldown,
      icon: <Clock3 size={12} />,
    },
    {
      key: 'blacklisted',
      value: normalized.blacklisted,
      style: styles.blacklisted,
      icon: <Ban size={12} />,
    },
    {
      key: 'disabled',
      value: normalized.disabled,
      style: styles.disabled,
      icon: <CirclePause size={12} />,
    },
  ] as const

  return (
    <Tooltip content={label}>
      <div {...stylex.props(styles.figure)} role="img" tabIndex={0} aria-label={label}>
        <div {...stylex.props(styles.counts)} aria-hidden="true">
          <strong {...stylex.props(styles.total)}>{n(normalized.total)}</strong>
          {hasVisibleStatus && <span {...stylex.props(styles.divider)} />}
          {segments.map(
            (segment) =>
              segment.value > 0 && (
                <span key={segment.key} {...stylex.props(styles.count, segment.style)}>
                  {segment.icon}
                  <span>{n(segment.value)}</span>
                </span>
              ),
          )}
        </div>
        <div {...stylex.props(styles.bar)} aria-hidden="true">
          {segments.map(
            (segment) =>
              segment.value > 0 && (
                <span
                  key={segment.key}
                  {...stylex.props(styles.segment, segment.style)}
                  style={{ flexGrow: segment.value }}
                />
              ),
          )}
        </div>
      </div>
    </Tooltip>
  )
}
