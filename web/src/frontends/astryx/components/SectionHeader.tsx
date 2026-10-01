import * as stylex from '@stylexjs/stylex'
import type { ReactNode } from 'react'

/**
 * Port of the classic ui/PanelHeader: bordered section heading with an
 * optional step chip and trailing actions slot.
 */
export function SectionHeader({
  headingId,
  title,
  description,
  step,
  actions,
}: {
  headingId: string
  title: ReactNode
  description?: ReactNode
  step?: number
  actions?: ReactNode
}) {
  return (
    <header {...stylex.props(styles.header)}>
      <div {...stylex.props(styles.copy)}>
        <h2 id={headingId} {...stylex.props(styles.heading)}>
          {step !== undefined && (
            <span {...stylex.props(styles.step)} aria-hidden="true">
              {step}
            </span>
          )}
          <span {...stylex.props(styles.title)}>{title}</span>
        </h2>
        {description !== undefined && <p {...stylex.props(styles.description)}>{description}</p>}
      </div>
      {actions !== undefined && <div {...stylex.props(styles.actions)}>{actions}</div>}
    </header>
  )
}

const styles = stylex.create({
  header: {
    display: 'flex',
    minHeight: 'var(--surface-header-min-height)',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-5)',
    marginBottom: 'var(--detail-panel-header-spacing)',
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    paddingBottom: 'var(--space-4)',
  },
  copy: {
    minWidth: 0,
  },
  heading: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    margin: 0,
    fontSize: 'var(--title-section)',
    fontWeight: 650,
    letterSpacing: '-0.01em',
  },
  title: {
    minWidth: 0,
  },
  step: {
    display: 'grid',
    width: '20px',
    height: '20px',
    flexShrink: 0,
    placeItems: 'center',
    borderRadius: '50%',
    backgroundColor: 'var(--color-action-soft)',
    color: 'var(--color-action)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 700,
    fontVariantNumeric: 'tabular-nums',
  },
  description: {
    maxWidth: '680px',
    marginTop: 'var(--space-1)',
    marginBottom: 0,
    marginLeft: 0,
    marginRight: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
  },
  actions: {
    flexShrink: 0,
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
})
