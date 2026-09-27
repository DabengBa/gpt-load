import { Button, Dialog, DialogHeader, Layout, LayoutContent, LayoutFooter } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import { Check, Copy } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { useT } from '../app/i18n'
import { copyText } from '@shared/lib/clipboard'

const styles = stylex.create({
  control: {
    position: 'relative',
    display: 'inline-flex',
  },
  feedback: {
    position: 'absolute',
    zIndex: 80, // --z-popover (stylex requires a numeric literal)
    top: 'calc(100% + var(--space-1))',
    insetInlineEnd: 0,
    width: 'max-content',
    borderWidth: '1px',
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
  fallbackBody: {
    display: 'grid',
    gap: 'var(--space-3)',
  },
  fallbackField: {
    display: 'grid',
    gap: 'var(--space-2)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  fallbackValue: {
    width: '100%',
    minHeight: 'var(--control-md)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text)',
    paddingBlock: 0,
    paddingInline: 'var(--space-3)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
    lineHeight: 'var(--line-normal)',
  },
  fallbackSuccess: {
    color: 'var(--color-success)',
    fontSize: 'var(--text-sm)',
  },
  fallbackFailure: {
    color: 'var(--color-warning)',
    fontSize: 'var(--text-sm)',
  },
})

/** Manual-copy fallback when the clipboard API can't write (insecure context). */
export function CopyFallbackDialog({
  value,
  onClose,
}: {
  value: string | undefined
  onClose(): void
}) {
  const t = useT()
  const [pending, setPending] = useState(false)
  const [state, setState] = useState<'idle' | 'success' | 'failure'>('idle')
  const attemptRef = useRef(0)

  const close = () => {
    attemptRef.current += 1
    setPending(false)
    setState('idle')
    onClose()
  }

  const retry = async () => {
    if (value === undefined || pending) return
    const attempt = ++attemptRef.current
    setPending(true)
    setState('idle')
    try {
      const copied = await copyText(value, undefined, () => attemptRef.current === attempt)
      if (attemptRef.current === attempt) setState(copied ? 'success' : 'failure')
    } catch {
      if (attemptRef.current === attempt) setState('failure')
    } finally {
      if (attemptRef.current === attempt) setPending(false)
    }
  }

  // Single-line inputs strip newlines; restore the real payload on select-all
  // copy so multiline values stay intact.
  const preserveMultilineCopy = (event: React.ClipboardEvent<HTMLInputElement>) => {
    if (value === undefined || !/[\r\n]/.test(value) || !event.clipboardData) return
    const input = event.currentTarget
    if (input.selectionStart !== 0 || input.selectionEnd !== input.value.length) return
    event.clipboardData.setData('text/plain', value)
    event.preventDefault()
  }

  return (
    <Dialog isOpen={value !== undefined} onOpenChange={(open) => !open && close()} width={420}>
      <Layout
        header={
          <DialogHeader
            title={t('common.copyFallback.title')}
            subtitle={t('common.copyFallback.description')}
            onOpenChange={(open) => !open && close()}
            hasDivider
          />
        }
        content={
          <LayoutContent isScrollable>
            <div {...stylex.props(styles.fallbackBody)}>
              <label {...stylex.props(styles.fallbackField)}>
                <span>{t('common.copyFallback.valueLabel')}</span>
                <input
                  {...stylex.props(styles.fallbackValue)}
                  type="text"
                  value={value ?? ''}
                  readOnly
                  autoComplete="off"
                  spellCheck={false}
                  onCopy={preserveMultilineCopy}
                />
              </label>
              {state === 'success' && (
                <p {...stylex.props(styles.fallbackSuccess)} role="status">
                  {t('common.copied')}
                </p>
              )}
              {state === 'failure' && (
                <p {...stylex.props(styles.fallbackFailure)} role="alert">
                  {t('common.copyFallback.failed')}
                </p>
              )}
            </div>
          </LayoutContent>
        }
        footer={
          <LayoutFooter hasDivider>
            <Button variant="secondary" label={t('common.close')} onClick={close} />
            <Button
              label={t('common.copy')}
              isLoading={pending}
              onClick={() => void retry()}
            />
          </LayoutFooter>
        }
      />
    </Dialog>
  )
}

/**
 * Copy icon button with transient feedback — the React port of the classic
 * ui/CopyButton (44px touch target, 2s feedback window, clipboard fallback
 * dialog for non-secure contexts).
 */
export function CopyButton({
  value,
  label,
  successLabel,
  failureLabel,
}: {
  value: string
  label: string
  successLabel: string
  failureLabel: string
}) {
  const [state, setState] = useState<'idle' | 'success' | 'failure'>('idle')
  const [pending, setPending] = useState(false)
  const [fallbackText, setFallbackText] = useState<string | undefined>(undefined)
  const sequenceRef = useRef(0)
  const mountedRef = useRef(true)
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  useEffect(
    () => () => {
      mountedRef.current = false
      clearTimeout(timerRef.current)
    },
    [],
  )

  const copy = async (): Promise<void> => {
    if (pending) return
    const sequence = ++sequenceRef.current
    const isCurrent = () => mountedRef.current && sequenceRef.current === sequence
    setFallbackText(undefined)
    setState('idle')
    setPending(true)
    try {
      const copied = await copyText(value, undefined, isCurrent)
      if (!isCurrent()) return
      if (copied) {
        setState('success')
      } else {
        setFallbackText(value)
      }
    } catch {
      if (isCurrent()) setState('failure')
    } finally {
      if (isCurrent()) setPending(false)
    }
    if (!isCurrent()) return
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setState('idle'), 2000)
  }

  return (
    <span {...stylex.props(styles.control)}>
      <Button
        variant="secondary"
        size="sm"
        isIconOnly
        label={label}
        icon={state === 'success' ? <Check size={16} /> : <Copy size={16} />}
        isLoading={pending}
        onClick={() => void copy()}
      />
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
      <CopyFallbackDialog value={fallbackText} onClose={() => setFallbackText(undefined)} />
    </span>
  )
}
