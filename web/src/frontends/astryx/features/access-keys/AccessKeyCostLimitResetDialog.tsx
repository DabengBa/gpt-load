import * as stylex from '@stylexjs/stylex'
import {
  Button,
  Dialog,
  DialogHeader,
  Layout,
  LayoutContent,
  LayoutFooter,
} from '@astryxdesign/core'
import { useEffect, useRef, useState } from 'react'
import { useIntl } from 'react-intl'

import { resetAccessKeyCostLimits } from '@shared/control/resources/access-keys'
import type { AccessKeyCostLimitRuleDto, AccessKeyDto } from '@shared/control/types'
import { RequestCancelledError } from '@shared/http/errors'
import { formatUSD } from '@shared/lib/format'

import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'

const styles = stylex.create({
  body: {
    display: 'grid',
    gap: 'var(--space-2)',
  },
  rules: {
    display: 'grid',
    gap: 'var(--space-2)',
    overflow: 'hidden',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
  },
  rule: {
    display: 'grid',
    minHeight: '44px',
    gridTemplateColumns: 'auto minmax(0, 1fr)',
    alignItems: 'center',
    gap: 'var(--space-3)',
    backgroundColor: 'var(--color-surface)',
    paddingBlock: '7px',
    paddingInline: '10px',
    cursor: 'pointer',
  },
  // `label + label` divider — StyleX has no sibling selectors.
  ruleDivider: {
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
  },
  // `label:has(input:checked)` — selection state is already in React state, so
  // the checked highlight is applied directly rather than through `:has()`.
  ruleSelected: {
    backgroundColor: 'color-mix(in srgb, var(--color-action-soft) 48%, var(--color-surface))',
  },
  ruleInput: {
    width: '16px',
    height: '16px',
    margin: 0,
    accentColor: 'var(--color-action)',
    cursor: { ':disabled': 'not-allowed' },
  },
  ruleText: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'baseline',
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
  },
  ruleName: {
    fontSize: 'var(--text-sm)',
  },
  ruleAmount: {
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
    fontVariantNumeric: 'tabular-nums',
    whiteSpace: 'nowrap',
  },
  feedback: {
    borderRadius: 'var(--radius-control)',
    borderWidth: 1,
    borderStyle: 'solid',
    paddingBlock: '9px',
    paddingInline: '12px',
    fontSize: 'var(--text-meta)',
  },
  warning: {
    borderColor: 'var(--color-warning)',
    backgroundColor:
      'var(--color-warning-bg, color-mix(in srgb, var(--color-warning) 12%, transparent))',
  },
  danger: {
    borderColor: 'var(--color-danger)',
    backgroundColor:
      'var(--color-danger-bg, color-mix(in srgb, var(--color-danger) 10%, transparent))',
  },
})

/**
 * Port of classic `AccessKeyCostLimitResetDialog.vue`. Where the classic used a
 * `trigger` slot, the Astryx frontend controls dialog open state from the
 * parent (`open`/`onOpenChange`), matching `AccessKeyDeleteDialog` and
 * `AccessKeyRotateDialog`. The `reset(name)` emit maps to `onReset(name)` —
 * the parent applies the reset invalidation plan and shows the success toast,
 * matching the classic contract.
 *
 * Classic `setOpen` semantics map to the `open` prop transition: opening
 * selects every rule and clears the failure flag; closing (or requesting a
 * close while pending, which is refused) clears the selection and aborts any
 * in-flight request.
 */
export function AccessKeyCostLimitResetDialog({
  accessKey,
  open,
  onOpenChange,
  onReset,
}: {
  accessKey: AccessKeyDto
  open: boolean
  onOpenChange(open: boolean): void
  onReset(name: string): void
}) {
  const t = useT()
  const intl = useIntl()
  const { apiClient } = useAppServices()
  const [pending, setPending] = useState(false)
  const [failed, setFailed] = useState(false)
  const [selectedRuleIDs, setSelectedRuleIDs] = useState<ReadonlySet<number>>(new Set())
  const ruleListRef = useRef<HTMLDivElement>(null)
  const requestRef = useRef<AbortController | undefined>(undefined)
  const [wasOpen, setWasOpen] = useState(open)

  const selectedCount = selectedRuleIDs.size

  function periodLabel(seconds: number): string {
    if (seconds % 86_400 === 0) {
      return t('accessKeys.reset.periodDays', { count: seconds / 86_400 })
    }
    if (seconds % 3_600 === 0) {
      return t('accessKeys.reset.periodHours', { count: seconds / 3_600 })
    }
    if (seconds % 60 === 0) {
      return t('accessKeys.reset.periodMinutes', { count: seconds / 60 })
    }
    return t('accessKeys.reset.periodSeconds', { count: seconds })
  }

  function ruleLabel(rule: AccessKeyCostLimitRuleDto): string {
    return rule.kind === 'total'
      ? t('accessKeys.reset.total')
      : t('accessKeys.reset.periodic', { period: periodLabel(rule.period_seconds) })
  }

  function setRuleSelected(ruleID: number, selected: boolean): void {
    const next = new Set(selectedRuleIDs)
    if (selected) next.add(ruleID)
    else next.delete(ruleID)
    setSelectedRuleIDs(next)
  }

  function requestOpenChange(value: boolean): void {
    // Classic `setOpen`: refuse to close while the request is in flight.
    if (!value && pending) return
    onOpenChange(value)
  }

  // Classic `setOpen(true)` seeds the selection with every rule and clears the
  // failure flag; `setOpen(false)` clears it again. Render-phase adjustment is
  // the React-recommended replacement for Vue's `watch(open)` mirroring.
  if (wasOpen !== open) {
    setWasOpen(open)
    if (open) {
      setSelectedRuleIDs(new Set(accessKey.cost_limit_rules.map(({ id }) => id)))
    } else {
      setSelectedRuleIDs(new Set())
    }
    setFailed(false)
  }

  // Closing also aborts an in-flight request (external-system side effect,
  // so it stays in an effect rather than the render-phase adjustment above).
  useEffect(() => {
    if (open) return
    requestRef.current?.abort()
    requestRef.current = undefined
  }, [open])

  // Classic `focusFirstRule`: move focus to the first checkbox once the dialog
  // has mounted. Deferred a frame so it lands after the dialog's own
  // focus-on-open management.
  useEffect(() => {
    if (!open) return
    const frame = requestAnimationFrame(() => {
      ruleListRef.current?.querySelector<HTMLInputElement>('input[type="checkbox"]')?.focus()
    })
    return () => cancelAnimationFrame(frame)
  }, [open])

  // Abort an in-flight request when the component unmounts.
  useEffect(
    () => () => {
      requestRef.current?.abort()
      requestRef.current = undefined
    },
    [],
  )

  async function confirmReset(): Promise<void> {
    if (selectedCount === 0 || pending) return
    setPending(true)
    setFailed(false)
    const controller = new AbortController()
    requestRef.current = controller
    try {
      await resetAccessKeyCostLimits(
        apiClient,
        accessKey.id,
        [...selectedRuleIDs],
        controller.signal,
      )
      if (requestRef.current !== controller) return
      setSelectedRuleIDs(new Set())
      onOpenChange(false)
      onReset(accessKey.name)
    } catch (error: unknown) {
      if (
        requestRef.current === controller &&
        !controller.signal.aborted &&
        !(error instanceof RequestCancelledError)
      ) {
        setFailed(true)
      }
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = undefined
        setPending(false)
      }
    }
  }

  return (
    <Dialog isOpen={open} onOpenChange={requestOpenChange} width={440}>
      <Layout
        header={
          <DialogHeader
            title={t('accessKeys.reset.title')}
            subtitle={t('accessKeys.reset.description', { name: accessKey.name })}
            onOpenChange={requestOpenChange}
            hasDivider
          />
        }
        content={
          <LayoutContent>
            <div {...stylex.props(styles.body)}>
              <div ref={ruleListRef} {...stylex.props(styles.rules)}>
                {accessKey.cost_limit_rules.map((rule, index) => (
                  <label
                    key={rule.id}
                    {...stylex.props(
                      styles.rule,
                      index > 0 && styles.ruleDivider,
                      selectedRuleIDs.has(rule.id) && styles.ruleSelected,
                    )}
                  >
                    <input
                      type="checkbox"
                      {...stylex.props(styles.ruleInput)}
                      checked={selectedRuleIDs.has(rule.id)}
                      disabled={pending}
                      onChange={(event) => setRuleSelected(rule.id, event.target.checked)}
                    />
                    <span {...stylex.props(styles.ruleText)}>
                      <strong {...stylex.props(styles.ruleName)}>{ruleLabel(rule)}</strong>
                      <small {...stylex.props(styles.ruleAmount)}>
                        {formatUSD(rule.limit_usd, intl.locale)}
                      </small>
                    </span>
                  </label>
                ))}
              </div>
              <div {...stylex.props(styles.feedback, styles.warning)} role="status">
                {t('accessKeys.reset.impact')}
              </div>
              {failed && (
                <div {...stylex.props(styles.feedback, styles.danger)} role="alert">
                  {t('accessKeys.reset.failed')}
                </div>
              )}
            </div>
          </LayoutContent>
        }
        footer={
          <LayoutFooter hasDivider>
            <Button
              variant="secondary"
              size="sm"
              isDisabled={pending}
              label={t('common.cancel')}
              onClick={() => requestOpenChange(false)}
            />
            <Button
              variant="primary"
              size="sm"
              isLoading={pending}
              isDisabled={selectedCount === 0}
              label={t('accessKeys.reset.confirm', { count: selectedCount })}
              onClick={() => void confirmReset()}
            />
          </LayoutFooter>
        }
      />
    </Dialog>
  )
}
