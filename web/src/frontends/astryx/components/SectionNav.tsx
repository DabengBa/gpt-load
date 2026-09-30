import * as stylex from '@stylexjs/stylex'
import { useId, useState } from 'react'

export interface SectionNavItem {
  id: string
  label: string
  disabled?: boolean
}

export interface SectionNavProps {
  items: readonly SectionNavItem[]
  value: string
  label: string
  caption?: string
  appearance?: 'default' | 'ledger'
  onSelect(id: string): void
}

const desktop = '@media (min-width: 768px)'
const ledgerNarrow = '@media (max-width: 860px)'

const styles = stylex.create({
  nav: {
    minWidth: 0,
    position: { [desktop]: 'sticky' },
    top: { [desktop]: 'calc(var(--topbar-height) + var(--space-4))' },
    alignSelf: { [desktop]: 'start' },
  },
  navLedger: {
    position: { default: 'sticky', [ledgerNarrow]: 'static' },
    top: '76px',
    alignSelf: 'start',
    maxWidth: { [ledgerNarrow]: '100%' },
  },
  mobile: {
    position: 'relative',
    display: { [desktop]: 'none' },
  },
  mobileLedger: {
    display: 'none',
  },
  toggle: {
    width: '100%',
    minHeight: 'var(--touch-target)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text)',
    font: 'inherit',
    textAlign: 'left',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-2)',
    paddingTop: 0,
    paddingBottom: 0,
    paddingInline: 'var(--space-3)',
    cursor: 'pointer',
  },
  options: {
    position: 'absolute',
    zIndex: 80, // --z-popover (stylex requires a numeric literal)
    top: 'calc(100% + var(--space-1))',
    right: 0,
    left: 0,
    display: 'grid',
    gap: 'var(--space-1)',
    margin: 0,
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-raised)',
    boxShadow: 'var(--shadow-overlay)',
    padding: 'var(--space-1)',
    listStyle: 'none',
  },
  optionButton: {
    width: '100%',
    minHeight: 'var(--touch-target)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'transparent',
    borderRadius: 'var(--radius-control)',
    color: 'var(--color-text)',
    font: 'inherit',
    textAlign: 'left',
    paddingTop: 0,
    paddingBottom: 0,
    paddingInline: 'var(--space-3)',
    cursor: 'pointer',
    backgroundColor: {
      default: 'var(--color-surface)',
      ':hover:not(:disabled)': 'var(--color-action-soft)',
    },
  },
  optionButtonCurrent: {
    backgroundColor: 'var(--color-action-soft)',
  },
  caption: {
    display: 'none',
  },
  captionLedger: {
    display: { default: 'block', [ledgerNarrow]: 'none' },
    marginBottom: '7px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
  },
  desktopList: {
    display: { default: 'none', [desktop]: 'grid' },
    margin: 0,
    borderLeftWidth: '1px',
    borderLeftStyle: 'solid',
    borderLeftColor: 'var(--color-border-subtle)',
    padding: 0,
    listStyle: 'none',
  },
  desktopListLedger: {
    display: { default: 'grid', [ledgerNarrow]: 'flex' },
    gap: '3px',
    borderLeftWidth: 0,
    overflowX: { [ledgerNarrow]: 'auto' },
    borderBottomWidth: { [ledgerNarrow]: '1px' },
    borderBottomStyle: { [ledgerNarrow]: 'solid' },
    borderBottomColor: { [ledgerNarrow]: 'var(--color-border-subtle)' },
    paddingBottom: { [ledgerNarrow]: '7px' },
    scrollbarWidth: { [ledgerNarrow]: 'none' },
  },
  link: {
    display: 'flex',
    minHeight: 'var(--touch-target)',
    alignItems: 'center',
    borderLeftWidth: '2px',
    borderLeftStyle: 'solid',
    borderLeftColor: {
      default: 'transparent',
      ':hover': 'var(--color-action)',
    },
    color: {
      default: 'var(--color-text-muted)',
      ':hover': 'var(--color-text)',
    },
    paddingInline: 'var(--space-3)',
    textDecoration: 'none',
  },
  linkCurrent: {
    borderLeftColor: 'var(--color-action)',
    color: 'var(--color-text)',
  },
  linkDisabled: {
    cursor: 'not-allowed',
    opacity: 0.55,
  },
  linkLedger: {
    minHeight: { default: '36px', [ledgerNarrow]: 'var(--touch-target)' },
    minWidth: { [ledgerNarrow]: 'max-content' },
    paddingTop: { default: '6px', [ledgerNarrow]: '7px' },
    paddingBottom: { default: '6px', [ledgerNarrow]: '7px' },
    paddingInline: { default: '11px 8px', [ledgerNarrow]: '10px' },
    fontSize: 'var(--text-meta)',
    backgroundColor: { ':hover': 'var(--color-surface-sunken)' },
    borderBottomWidth: { [ledgerNarrow]: '2px' },
    borderBottomStyle: { [ledgerNarrow]: 'solid' },
    borderBottomColor: {
      [ledgerNarrow]: {
        default: 'transparent',
        ':hover': 'var(--color-action)',
      },
    },
    borderLeftWidth: { [ledgerNarrow]: 0 },
  },
  linkLedgerCurrent: {
    backgroundColor: 'var(--color-surface-sunken)',
    borderLeftColor: 'var(--color-action)',
    color: 'var(--color-text)',
    borderBottomColor: {
      [ledgerNarrow]: 'var(--color-action)',
    },
  },
})

export function SectionNav({
  items,
  value,
  label,
  caption,
  appearance = 'default',
  onSelect,
}: SectionNavProps) {
  const [expanded, setExpanded] = useState(false)
  const optionsId = `${useId()}-options`
  const current = items.find((item) => item.id === value) ?? items[0]
  const ledger = appearance === 'ledger'

  function select(id: string): void {
    onSelect(id)
    setExpanded(false)
  }

  return (
    <nav {...stylex.props(styles.nav, ledger && styles.navLedger)} aria-label={label}>
      <div {...stylex.props(styles.mobile, ledger && styles.mobileLedger)}>
        <button
          {...stylex.props(styles.toggle)}
          type="button"
          aria-expanded={expanded}
          aria-controls={optionsId}
          onClick={() => setExpanded((open) => !open)}
        >
          <span>{current?.label}</span>
          <span aria-hidden="true">⌄</span>
        </button>
        {expanded && (
          <ul {...stylex.props(styles.options)} id={optionsId}>
            {items.map((item) => (
              <li key={item.id}>
                <button
                  {...stylex.props(
                    styles.optionButton,
                    item.id === value && styles.optionButtonCurrent,
                  )}
                  type="button"
                  aria-current={item.id === value ? 'page' : undefined}
                  disabled={item.disabled}
                  onClick={() => select(item.id)}
                >
                  {item.label}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {caption !== undefined && (
        <span {...stylex.props(styles.caption, ledger && styles.captionLedger)}>{caption}</span>
      )}
      <ul {...stylex.props(styles.desktopList, ledger && styles.desktopListLedger)}>
        {items.map((item) => {
          const isCurrent = item.id === value
          return (
            <li key={item.id}>
              <a
                {...stylex.props(
                  styles.link,
                  ledger && styles.linkLedger,
                  isCurrent && !ledger && styles.linkCurrent,
                  isCurrent && ledger && styles.linkLedgerCurrent,
                  item.disabled === true && styles.linkDisabled,
                )}
                href={`#${item.id}`}
                aria-current={isCurrent ? 'location' : undefined}
                aria-disabled={item.disabled === true ? 'true' : undefined}
                onClick={(event) => {
                  event.preventDefault()
                  if (item.disabled !== true) select(item.id)
                }}
              >
                {item.label}
              </a>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}
