import * as stylex from '@stylexjs/stylex'
import { Copy } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { useClipboardCopy } from '../../app/use-clipboard-copy'

const styles = stylex.create({
  wrap: {
    position: 'relative',
    display: 'inline-flex',
    width: 'auto',
    maxWidth: '100%',
    minWidth: 0,
  },
  chip: {
    display: 'inline-flex',
    width: 'auto',
    maxWidth: '100%',
    minWidth: 0,
    minHeight: 'var(--control-compact, 28px)',
    alignItems: 'center',
    gap: 6,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text-muted)',
    paddingBlock: 0,
    paddingInline: 'var(--space-2)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
    cursor: 'pointer',
  },
  value: {
    minWidth: 0,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  feedback: {
    position: 'absolute',
    zIndex: 80,
    top: 'calc(100% + var(--space-1))',
    insetInlineStart: 0,
    width: 'max-content',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-tag)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text)',
    paddingBlock: 'var(--space-1)',
    paddingInline: 'var(--space-2)',
    boxShadow: 'var(--shadow-card)',
    fontSize: 'var(--text-label-xs)',
  },
})

/**
 * Masked-value copy chip — the Astryx counterpart of classic CopyChip with
 * layout="trailing". `resolveValue` (reveal) runs on copy, never on render;
 * the revealed secret lives only inside the copy hook's transient state.
 */
export function CopyChip({
  value,
  label,
  successLabel,
  failureLabel,
  resolveValue,
}: {
  value: string
  label: string
  successLabel: string
  failureLabel: string
  resolveValue?: () => string | Promise<string>
}) {
  const { copy, pending, reset, dialog } = useClipboardCopy()
  const [state, setState] = useState<'idle' | 'success' | 'failure'>('idle')
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  // Classic watches props.value and drops feedback/fallback state — the
  // collection re-keys the chip per record so this is mainly rotation reuse.
  const [lastValue, setLastValue] = useState(value)
  if (lastValue !== value) {
    setLastValue(value)
    reset()
    setState('idle')
  }

  useEffect(
    () => () => {
      clearTimeout(timerRef.current)
    },
    [],
  )

  const copyValue = async (): Promise<void> => {
    if (pending) return
    setState('idle')
    try {
      const result = await copy(resolveValue ?? value)
      if (result === 'cancelled') return
      setState(result === 'success' ? 'success' : 'idle')
    } catch {
      setState('failure')
    }
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setState('idle'), 2_000)
  }

  return (
    <span {...stylex.props(styles.wrap)}>
      <button
        {...stylex.props(styles.chip)}
        type="button"
        aria-label={label}
        aria-busy={pending}
        disabled={pending}
        onClick={() => void copyValue()}
      >
        <span {...stylex.props(styles.value)}>{value}</span>
        <Copy size={14} aria-hidden="true" />
      </button>
      {state !== 'idle' && (
        <span
          {...stylex.props(styles.feedback)}
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          {state === 'success' ? successLabel : failureLabel}
        </span>
      )}
      {dialog}
    </span>
  )
}
