import * as stylex from '@stylexjs/stylex'
import { Button } from '@astryxdesign/core'
import { TriangleAlert } from 'lucide-react'
import type { ReactNode } from 'react'

export type SaveBarStatus = 'idle' | 'saved' | 'error' | 'indeterminate'

export interface StickySaveBarProps {
  dirty: boolean
  pending: boolean
  status?: SaveBarStatus
  error?: string
  errorActionLabel?: string
  errorPlacement?: 'inline' | 'floating'
  appearance?: 'default' | 'ledger'
  alwaysVisible?: boolean
  onErrorAction?(): void
  statusContent?: ReactNode
  actions?: ReactNode
  xstyle?: stylex.StyleXStyles
}

const small = '@media (max-width: 480px)'
const ledgerNarrow = '@media (max-width: 800px)'

const styles = stylex.create({
  bar: {
    position: 'sticky',
    zIndex: 20, // --z-sticky (stylex requires a numeric literal)
    bottom: 0,
    display: 'grid',
    gridTemplateColumns: {
      default: 'minmax(0, 1fr) auto',
      [small]: '1fr',
    },
    alignItems: 'center',
    gap: 'var(--space-3)',
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-control)',
    backgroundColor: 'var(--color-surface-raised)',
    boxShadow: 'var(--shadow-sheet)',
    paddingTop: 'var(--space-3)',
    paddingInline: 'var(--space-4)',
    paddingBottom: 'calc(var(--space-3) + env(safe-area-inset-bottom))',
  },
  barLedger: {
    bottom: { default: '12px', [ledgerNarrow]: '8px' },
    display: 'flex',
    minHeight: '58px',
    alignItems: { default: 'center', [ledgerNarrow]: 'stretch' },
    flexDirection: { default: 'row', [ledgerNarrow]: 'column' },
    justifyContent: 'space-between',
    gap: { default: '18px', [ledgerNarrow]: '9px' },
    marginTop: '48px',
    marginBottom: '-11px',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: '9px',
    backgroundColor: 'color-mix(in srgb, var(--color-surface-raised) 94%, transparent)',
    boxShadow:
      '0 2px 8px light-dark(rgba(0, 0, 0, 0.08), rgba(0, 0, 0, 0.4)), 0 12px 28px light-dark(rgba(0, 0, 0, 0.08), rgba(0, 0, 0, 0.28))',
    paddingTop: '9px',
    paddingBottom: '9px',
    paddingInline: '14px 11px',
    backdropFilter: 'blur(10px)',
  },
  status: {
    minWidth: 0,
    margin: 0,
    color: 'var(--color-text-muted)',
  },
  statusLedger: {
    display: 'flex',
    alignItems: 'center',
    gap: '9px',
    '::before': {
      width: '7px',
      height: '7px',
      flex: 'none',
      borderRadius: '50%',
      backgroundColor: 'var(--color-neutral)',
      content: '""',
    },
  },
  statusLedgerDirty: { '::before': { backgroundColor: 'var(--color-warning)' } },
  statusLedgerSaving: { '::before': { backgroundColor: 'var(--color-action)' } },
  statusLedgerSaved: { '::before': { backgroundColor: 'var(--color-success)' } },
  statusLedgerError: { '::before': { backgroundColor: 'var(--color-danger)' } },
  statusLedgerIndeterminate: { '::before': { backgroundColor: 'var(--color-warning)' } },
  error: {
    minWidth: 0,
    margin: 0,
    gridColumn: '1 / -1',
    color: 'var(--color-danger)',
  },
  errorLedger: {
    flex: '1',
    fontSize: 'var(--text-sm)',
  },
  errorFloating: {
    position: 'absolute',
    zIndex: 1,
    bottom: 'calc(100% + var(--space-2))',
    left: 0,
    right: { default: 'auto', [ledgerNarrow]: 0 },
    display: 'flex',
    width: { default: 'max-content', [ledgerNarrow]: 'auto' },
    maxWidth: { default: '100%', [ledgerNarrow]: 'none' },
    alignItems: 'flex-start',
    gap: 'var(--space-2)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-feedback-danger-border)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-danger-bg)',
    boxShadow: 'var(--shadow-feedback)',
    paddingTop: '8px',
    paddingBottom: '8px',
    paddingInline: '10px',
    color: 'var(--color-danger)',
    lineHeight: 'var(--line-normal)',
    pointerEvents: 'auto',
  },
  actions: {
    display: 'flex',
    flexWrap: 'wrap',
    justifyContent: { default: 'flex-end', [small]: 'stretch' },
    gap: 'var(--space-2)',
  },
  actionsLedger: {
    flex: 'none',
    gap: '7px',
    display: { default: 'flex', [ledgerNarrow]: 'grid' },
    gridTemplateColumns: { [ledgerNarrow]: '1fr 1fr' },
  },
})

const ledgerDotClass: Record<SaveBarStatus | 'saving' | 'dirty', stylex.StyleXStyles> = {
  idle: styles.statusLedger,
  dirty: styles.statusLedgerDirty,
  saving: styles.statusLedgerSaving,
  saved: styles.statusLedgerSaved,
  error: styles.statusLedgerError,
  indeterminate: styles.statusLedgerIndeterminate,
}

export function StickySaveBar({
  dirty,
  pending,
  status = 'idle',
  error = '',
  errorActionLabel = '',
  errorPlacement = 'inline',
  appearance = 'default',
  alwaysVisible = false,
  onErrorAction,
  statusContent,
  actions,
  xstyle,
}: StickySaveBarProps) {
  const visible = alwaysVisible || dirty || pending || status !== 'idle' || error !== ''
  if (!visible) return null

  const ledger = appearance === 'ledger'
  const visualState =
    status === 'indeterminate'
      ? 'indeterminate'
      : pending
        ? 'saving'
        : status === 'error' || error !== ''
          ? 'error'
          : dirty
            ? 'dirty'
            : status === 'saved'
              ? 'saved'
              : 'idle'

  return (
    <footer
      {...stylex.props(styles.bar, ledger && styles.barLedger, xstyle)}
      data-status={visualState}
      aria-busy={pending || undefined}
    >
      <div
        {...stylex.props(
          styles.status,
          ledger && styles.statusLedger,
          ledger && ledgerDotClass[visualState],
        )}
        aria-live="polite"
      >
        {statusContent}
      </div>
      {error !== '' && (
        <p
          {...stylex.props(
            styles.error,
            ledger && styles.errorLedger,
            errorPlacement === 'floating' && styles.errorFloating,
          )}
          role="alert"
        >
          {errorPlacement === 'floating' && <TriangleAlert size={16} aria-hidden />}
          <span>{error}</span>
          {errorActionLabel !== '' && (
            <Button variant="ghost" size="sm" label={errorActionLabel} onClick={onErrorAction} />
          )}
        </p>
      )}
      <div {...stylex.props(styles.actions, ledger && styles.actionsLedger)}>{actions}</div>
    </footer>
  )
}
