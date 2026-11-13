import * as stylex from '@stylexjs/stylex'
import type { ReactNode } from 'react'

const CARD_LAYOUT = '@media (max-width: 860px)'

const styles = stylex.create({
  panel: {
    minWidth: 0,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-card)',
    backgroundColor: 'var(--color-surface)',
    boxShadow: 'var(--shadow-card)',
    padding: 'var(--space-4)',
  },
  // Log rows already become individual cards at this breakpoint.
  unframedCards: {
    borderWidth: { default: 1, [CARD_LAYOUT]: 0 },
    borderRadius: { default: 'var(--radius-card)', [CARD_LAYOUT]: 0 },
    backgroundColor: { default: 'var(--color-surface)', [CARD_LAYOUT]: 'transparent' },
    boxShadow: { default: 'var(--shadow-card)', [CARD_LAYOUT]: 'none' },
    padding: { default: 'var(--space-4)', [CARD_LAYOUT]: 0 },
  },
})

export function ContentPanel({
  children,
  unframedCards = false,
}: {
  children: ReactNode
  unframedCards?: boolean
}) {
  return (
    <div
      data-content-panel=""
      {...stylex.props(styles.panel, unframedCards && styles.unframedCards)}
    >
      {children}
    </div>
  )
}
