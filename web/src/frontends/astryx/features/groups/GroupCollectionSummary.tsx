import * as stylex from '@stylexjs/stylex'
import { useIntl } from 'react-intl'

import type { GroupCollectionStatus } from '@shared/control/types'

import { useT } from '../../app/i18n'
import type { GroupSummaryItem } from './use-group-summary-items'

/**
 * Summary chips double as the status filter — one tap re-queries the server
 * with `status`, so the counts and the table never disagree.
 */
export function GroupCollectionSummary({
  items,
  status,
  onSelect,
}: {
  items: readonly GroupSummaryItem[]
  status: GroupCollectionStatus | undefined
  onSelect(status: GroupCollectionStatus | undefined): void
}) {
  const t = useT()
  const intl = useIntl()

  return (
    <section {...stylex.props(styles.summary)} aria-label={t('groups.collection.summary.region')}>
      {items.map((item) => (
        <button
          key={item.value ?? 'all'}
          type="button"
          aria-pressed={status === item.value}
          {...stylex.props(
            styles.summaryFilter,
            status === item.value && styles.summaryFilterActive,
          )}
          onClick={() => onSelect(item.value)}
        >
          <span
            {...stylex.props(
              styles.summaryDot,
              item.tone === 'success'
                ? styles.dotSuccess
                : item.tone === 'danger'
                  ? styles.dotDanger
                  : undefined,
            )}
            aria-hidden="true"
          />
          <span>{item.label}</span>
          <span {...stylex.props(styles.summaryCount)}>{intl.formatNumber(item.count)}</span>
        </button>
      ))}
    </section>
  )
}

const styles = stylex.create({
  summary: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'repeat(4, minmax(0, 1fr))',
      '@media (max-width: 480px)': 'repeat(2, minmax(0, 1fr))',
    },
    gap: 'var(--space-1)',
  },
  summaryFilter: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    minWidth: 0,
    minHeight: 'var(--touch-target)',
    borderWidth: 0,
    padding: 'var(--space-2) var(--space-3)',
    backgroundColor: {
      default: 'transparent',
      ':hover': 'var(--color-interactive-hover)',
    },
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    fontFamily: 'inherit',
    cursor: 'pointer',
    borderRadius: 'var(--radius-control)',
    outline: {
      default: 'none',
      ':focus-visible': '2px solid var(--color-focus)',
    },
    outlineOffset: 2,
  },
  summaryFilterActive: {
    backgroundColor: {
      default: 'var(--color-action-soft)',
      ':hover': 'var(--color-action-soft)',
    },
    color: 'var(--color-action)',
    fontWeight: 600,
  },
  summaryDot: {
    width: 8,
    height: 8,
    borderRadius: 999,
    backgroundColor: 'var(--color-neutral-fg)',
    flexShrink: 0,
  },
  dotSuccess: { backgroundColor: 'var(--color-success)' },
  dotDanger: { backgroundColor: 'var(--color-danger)' },
  summaryCount: {
    marginLeft: 'auto',
    fontVariantNumeric: 'tabular-nums',
    fontSize: 'var(--text-body)',
    fontWeight: 600,
  },
})
