import * as stylex from '@stylexjs/stylex'
import { useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'

/**
 * Astryx counterpart of classic `components/collection/LedgerRecordList.vue` —
 * the subgrid ledger table shell (role=table, sticky-alignment header row,
 * horizontal scroll with accessible hint). The classic component exposes its
 * geometry through `--ledger-record-list-*` custom properties on the root;
 * the React version takes the same tunables as props and emits them as CSS
 * vars so consumers keep the exact grid contracts:
 *
 *   grid            → --ledger-record-list-grid (desktop columns)
 *   cardGrid        → --ledger-record-list-card-grid (≤860px card columns)
 *   recordMinHeight → --ledger-record-list-record-min-height
 *   recordPadding   → --ledger-record-list-record-padding
 *   columnGap       → --ledger-record-list-column-gap
 *
 * Rows are consumer-rendered `<article role="row">` children; the header slot
 * takes `role="columnheader"` spans. Record/cell chrome (borders, min-height,
 * hover, mobile card frame) is provided by the companion `ledgerRecordStyles`
 * export — spread it on the row/cell elements.
 */

const narrow = '@media (max-width: 860px)'
const smallest = '@media (max-width: 560px)'

export const ledgerListStyles = stylex.create({
  list: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'var(--ledger-record-list-grid, minmax(0, 1fr))',
      [narrow]: 'minmax(0, 1fr)',
    },
    columnGap: 'var(--ledger-record-list-column-gap, 16px)',
    gap: { [narrow]: '10px' },
    overflowX: { default: 'auto', [narrow]: 'visible' },
    overflowY: 'hidden',
    borderBottomWidth: { default: '1px', [narrow]: 0 },
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-control)',
    paddingTop: { [narrow]: '10px' },
  },
  header: {
    display: { default: 'grid', [narrow]: 'none' },
    gridColumn: '1 / -1',
    gridTemplateColumns: 'subgrid',
    alignItems: 'center',
    minHeight: '38px',
  },
  headerCell: {
    justifySelf: 'stretch',
    textAlign: 'start',
  },
})

export const ledgerRecordStyles = stylex.create({
  record: {
    display: 'grid',
    gridColumn: { default: '1 / -1', [narrow]: '1' },
    gridTemplateColumns: {
      default: 'subgrid',
      [narrow]: 'var(--ledger-record-list-card-grid, minmax(0, 0.48fr) minmax(0, 1.52fr))',
    },
    alignItems: { default: 'center', [narrow]: 'start' },
    gap: { [narrow]: '14px 16px' },
    position: 'relative',
    minHeight: {
      default: 'var(--ledger-record-list-record-min-height, 96px)',
      [narrow]: '0',
    },
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: {
      // Classic `.record:first-of-type` strengthens the top border — records
      // render as <article> after a <div> header, so first-of-type lands on
      // the first record regardless of row index.
      default: 'var(--color-border-subtle)',
      ':first-of-type': 'var(--color-border-control)',
      [narrow]: 'var(--color-border-subtle)',
    },
    borderWidth: { [narrow]: '1px' },
    borderStyle: { [narrow]: 'solid' },
    borderColor: { [narrow]: 'var(--color-border-subtle)' },
    borderRadius: { [narrow]: 'var(--radius-control)' },

    padding: {
      default: 'var(--ledger-record-list-record-padding, 14px 0)',
      [narrow]: '16px',
      [smallest]: '14px 13px',
    },
    transitionProperty: 'background-color',
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
  },
  cell: {
    minWidth: 0,
    justifySelf: 'stretch',
    textAlign: 'start',
  },
})

export interface LedgerRecordListProps {
  label: string
  rowCount?: number
  scrollHint?: string
  /** Desktop grid template (maps to --ledger-record-list-grid). */
  grid?: string
  /** ≤860px card grid template (maps to --ledger-record-list-card-grid). */
  cardGrid?: string
  recordMinHeight?: string
  recordPadding?: string
  columnGap?: string
  header?: ReactNode
  children?: ReactNode
}

export function LedgerRecordList({
  label,
  rowCount,
  scrollHint,
  grid,
  cardGrid,
  recordMinHeight,
  recordPadding,
  columnGap,
  header,
  children,
}: LedgerRecordListProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [overflowing, setOverflowing] = useState(false)
  const accessibleLabel = overflowing && scrollHint ? `${label} · ${scrollHint}` : label

  const updateOverflow = useCallback(() => {
    const element = containerRef.current
    setOverflowing(Boolean(element && element.scrollWidth > element.clientWidth + 1))
  }, [])

  // ResizeObserver fires once on observe() — no rAF fallback needed; the
  // window listener covers the non-RO path (same contract as classic).
  useEffect(() => {
    const element = containerRef.current
    const observer =
      typeof ResizeObserver === 'function' ? new ResizeObserver(updateOverflow) : undefined
    if (element) observer?.observe(element)
    if (observer === undefined) updateOverflow()
    window.addEventListener('resize', updateOverflow)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', updateOverflow)
    }
  }, [updateOverflow])

  // Classic re-measures on every child update (onUpdated) — a layout effect
  // keyed on children covers row/record churn.
  useEffect(() => {
    updateOverflow()
  })

  const vars = {
    ...(grid !== undefined ? { '--ledger-record-list-grid': grid } : {}),
    ...(cardGrid !== undefined ? { '--ledger-record-list-card-grid': cardGrid } : {}),
    ...(recordMinHeight !== undefined
      ? { '--ledger-record-list-record-min-height': recordMinHeight }
      : {}),
    ...(recordPadding !== undefined
      ? { '--ledger-record-list-record-padding': recordPadding }
      : {}),
    ...(columnGap !== undefined ? { '--ledger-record-list-column-gap': columnGap } : {}),
  } as CSSProperties

  return (
    <div
      ref={containerRef}
      {...stylex.props(ledgerListStyles.list)}
      style={vars}
      data-testid="ledger-record-list"
      role="table"
      aria-label={accessibleLabel}
      aria-rowcount={rowCount}
      tabIndex={overflowing ? 0 : undefined}
    >
      <div
        {...stylex.props(ledgerListStyles.header)}
        role="row"
        data-testid="ledger-record-list__header"
        aria-rowindex={1}
      >
        {header}
      </div>
      {children}
    </div>
  )
}
