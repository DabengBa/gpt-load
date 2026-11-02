import * as stylex from '@stylexjs/stylex'
import { LoaderCircle, RefreshCw, TriangleAlert } from 'lucide-react'
import type { JSX } from 'react'

const touch = '@media (pointer: coarse)'
const reduceMotion = '@media (prefers-reduced-motion: reduce)'

const spin = stylex.keyframes({ to: { transform: 'rotate(360deg)' } })

const styles = stylex.create({
  // QueryFeedback port (loading/error/stale strips).
  feedback: {
    display: 'flex',
    minHeight: 48,
    alignItems: 'center',
    gap: 'var(--space-2)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text-muted)',
    padding: 'var(--space-3)',
  },
  feedbackError: {
    borderColor: 'var(--color-danger)',
    backgroundColor: 'var(--color-danger-bg)',
    color: 'var(--color-danger)',
  },
  feedbackStale: {
    borderColor: 'color-mix(in srgb, var(--color-warning) 32%, var(--color-border-subtle))',
    backgroundColor: 'var(--color-warning-bg)',
    color: 'var(--color-warning)',
    paddingTop: 9,
    paddingBottom: 9,
    paddingInline: 11,
    fontSize: 'var(--text-sm)',
  },
  feedbackText: {
    minWidth: 0,
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: 'auto',
  },
  feedbackRetry: {
    display: 'inline-flex',
    flex: 'none',
    minHeight: { default: 44, [touch]: 'var(--touch-target)' },
    alignItems: 'center',
    gap: 'var(--space-1)',
    marginLeft: 'auto',
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: 'inherit',
    fontWeight: 650,
    whiteSpace: 'nowrap',
    cursor: 'pointer',
  },
  feedbackRetryStale: {
    minHeight: { default: 'var(--control-compact)', [touch]: 'var(--touch-target)' },
  },
  spin: {
    animationName: { default: spin, [reduceMotion]: 'none' },
    animationDuration: '1s',
    animationTimingFunction: 'linear',
    animationIterationCount: 'infinite',
  },
})

export function ScheduleFeedback({
  state,
  message,
  retryLabel,
  onRetry,
}: {
  state: 'loading' | 'error' | 'stale'
  message: string
  retryLabel?: string
  onRetry?(): void
}): JSX.Element {
  return (
    <div
      {...stylex.props(
        styles.feedback,
        state === 'error' && styles.feedbackError,
        state === 'stale' && styles.feedbackStale,
      )}
      role={state === 'error' ? 'alert' : 'status'}
    >
      {state === 'loading' ? (
        <LoaderCircle size={18} aria-hidden {...stylex.props(styles.spin)} />
      ) : (
        <TriangleAlert size={state === 'stale' ? 14 : 18} aria-hidden />
      )}
      <span {...stylex.props(styles.feedbackText)}>{message}</span>
      {state !== 'loading' && retryLabel !== undefined && (
        <button
          type="button"
          {...stylex.props(styles.feedbackRetry, state === 'stale' && styles.feedbackRetryStale)}
          onClick={onRetry}
        >
          <RefreshCw size={15} aria-hidden />
          {retryLabel}
        </button>
      )}
    </div>
  )
}
