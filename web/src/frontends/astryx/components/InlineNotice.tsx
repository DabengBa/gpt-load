import * as stylex from '@stylexjs/stylex'
import type { ReactNode } from 'react'

export type InlineNoticeTone = 'neutral' | 'info' | 'success' | 'warning' | 'danger'
export type InlineNoticeAppearance =
  | 'default'
  | 'hint'
  | 'ledger'
  | 'ledger-hint'
  | 'auth'

export interface InlineNoticeProps {
  tone?: InlineNoticeTone
  appearance?: InlineNoticeAppearance
  glyph?: string
  action?: ReactNode
  children?: ReactNode
}

const styles = stylex.create({
  base: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 'var(--space-2)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'transparent',
    borderRadius: 'var(--radius-control)',
    paddingBlock: '9px',
    paddingInline: '10px',
    fontSize: 'var(--text-meta)',
    lineHeight: 'var(--line-normal)',
  },
  glyph: {
    display: 'grid',
    width: '18px',
    height: '18px',
    flexShrink: 0,
    placeItems: 'center',
    fontWeight: 600,
    lineHeight: 1,
  },
  message: {
    minWidth: 0,
    flexGrow: 1,
  },
  action: {
    display: 'inline-flex',
    flexShrink: 0,
    alignItems: 'center',
    alignSelf: 'flex-start',
  },
  toneInfo: {
    borderColor: 'var(--color-border-subtle)',
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text-muted)',
  },
  toneNeutral: {
    borderColor: 'var(--color-border-subtle)',
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text-muted)',
  },
  toneSuccess: {
    borderColor: 'var(--color-success)',
    backgroundColor: 'var(--color-success-bg)',
    color: 'var(--color-success)',
  },
  toneWarning: {
    borderColor: 'var(--color-warning)',
    backgroundColor: 'var(--color-warning-bg)',
    color: 'var(--color-warning)',
  },
  toneDanger: {
    borderColor: 'var(--color-danger)',
    backgroundColor: 'var(--color-danger-bg)',
    color: 'var(--color-danger)',
  },
  appearanceHint: {
    gap: '6px',
    borderWidth: 0,
    backgroundColor: 'transparent',
    padding: 0,
    fontSize: 'var(--text-sm)',
  },
  appearanceHintGlyph: {
    width: '14px',
    height: '14px',
  },
  hintInfo: {
    color: 'var(--color-action)',
  },
  hintNeutral: {
    color: 'var(--color-text-muted)',
  },
  appearanceLedger: {
    borderRadius: '6px',
    paddingBlock: '9px',
    paddingInline: '11px',
    fontSize: '11px',
    lineHeight: 1.55,
  },
  ledgerGlyph: {
    width: '17px',
    height: '17px',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'currentColor',
    borderRadius: '50%',
    fontFamily: 'var(--font-serif)',
    fontSize: '11px',
    fontWeight: 700,
  },
  ledgerInfo: {
    borderColor:
      'color-mix(in srgb, var(--color-action) 33%, var(--color-border-subtle))',
    backgroundColor: 'var(--color-action-soft)',
    color: 'var(--color-action)',
  },
  ledgerWarning: {
    borderColor:
      'color-mix(in srgb, var(--color-warning) 36%, var(--color-border-subtle))',
  },
  ledgerDanger: {
    borderColor:
      'color-mix(in srgb, var(--color-danger) 32%, var(--color-border-subtle))',
  },
  appearanceLedgerHint: {
    gap: 'var(--space-2)',
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: 'var(--color-text-faint)',
    padding: 0,
    fontSize: '10.8px',
    lineHeight: 1.6,
  },
  appearanceAuth: {
    gap: 0,
    borderColor:
      'color-mix(in srgb, var(--color-danger) 34%, var(--color-border-subtle))',
    backgroundColor: 'var(--color-danger-bg)',
    color: 'var(--color-danger)',
    paddingBlock: '8px',
    paddingInline: '10px',
    fontSize: 'var(--text-label-xs)',
    lineHeight: 1.55,
  },
  authGlyphHidden: {
    display: 'none',
  },
})

const toneStyles = {
  neutral: styles.toneNeutral,
  info: styles.toneInfo,
  success: styles.toneSuccess,
  warning: styles.toneWarning,
  danger: styles.toneDanger,
} as const

function defaultGlyph(tone: InlineNoticeTone): string {
  if (tone === 'success') return '✓'
  if (tone === 'danger' || tone === 'warning') return '▲'
  return 'i'
}

/**
 * Astryx counterpart of classic InlineFeedback (minus the `toast` appearance —
 * transient toasts go through the app toast host instead). Same tone → ARIA
 * mapping: soft tones announce politely via role=status, warning/danger use
 * role=alert.
 */
export function InlineNotice({
  tone = 'info',
  appearance = 'default',
  glyph,
  action,
  children,
}: InlineNoticeProps) {
  const soft = tone === 'neutral' || tone === 'info' || tone === 'success'
  const ledgerGlyph =
    appearance === 'ledger' || appearance === 'ledger-hint'
  const hintGlyph = appearance === 'hint'

  return (
    <div
      {...stylex.props(
        styles.base,
        toneStyles[tone],
        appearance === 'hint' && styles.appearanceHint,
        appearance === 'ledger' && styles.appearanceLedger,
        appearance === 'ledger-hint' && styles.appearanceLedgerHint,
        appearance === 'auth' && styles.appearanceAuth,
        appearance === 'hint' && tone === 'info' && styles.hintInfo,
        appearance === 'hint' && tone === 'neutral' && styles.hintNeutral,
        appearance === 'ledger' && tone === 'info' && styles.ledgerInfo,
        appearance === 'ledger' && tone === 'warning' && styles.ledgerWarning,
        appearance === 'ledger' && tone === 'danger' && styles.ledgerDanger,
      )}
      role={soft ? 'status' : 'alert'}
      aria-live={soft ? 'polite' : 'assertive'}
      aria-atomic="true"
    >
      <span
        {...stylex.props(
          styles.glyph,
          hintGlyph && styles.appearanceHintGlyph,
          ledgerGlyph && styles.ledgerGlyph,
          appearance === 'auth' && styles.authGlyphHidden,
        )}
        aria-hidden="true"
      >
        {glyph ?? defaultGlyph(tone)}
      </span>
      <span {...stylex.props(styles.message)}>{children}</span>
      {action !== undefined && (
        <span {...stylex.props(styles.action)}>{action}</span>
      )}
    </div>
  )
}
