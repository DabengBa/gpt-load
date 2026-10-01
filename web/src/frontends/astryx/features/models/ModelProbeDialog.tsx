import * as stylex from '@stylexjs/stylex'
import {
  Badge,
  Banner,
  Button,
  Dialog,
  DialogHeader,
  Layout,
  LayoutContent,
  LayoutFooter,
  Switch,
} from '@astryxdesign/core'
import { Copy, LoaderCircle } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useIntl } from 'react-intl'

import type { ModelProbeOutcome, ModelProbeResultDto } from '@shared/control/resources/model-probe'
import { formatLocalInstant } from '@shared/lib/format'

import { useT } from '../../app/i18n'
import { useClipboardCopy } from '../../app/use-clipboard-copy'

const NARROW = '@media (max-width: 480px)'
const TOUCH = '@media (max-width: 860px)'

const spin = stylex.keyframes({
  to: { transform: 'rotate(360deg)' },
})

const styles = stylex.create({
  body: {
    display: 'grid',
    gap: 'var(--space-3)',
  },
  // Classic QueryFeedback state="loading": bordered sunken box + spinner,
  // role="status".
  loading: {
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
  loadingSpin: {
    animationName: {
      default: spin,
      '@media (prefers-reduced-motion: reduce)': 'none',
    },
    animationDuration: '1s',
    animationTimingFunction: 'linear',
    animationIterationCount: 'infinite',
  },
  details: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'max-content minmax(0, 1fr)',
      [NARROW]: '1fr',
    },
    gap: { default: '8px var(--space-3)', [NARROW]: 'var(--space-1)' },
    margin: 0,
    fontSize: 'var(--text-sm)',
    lineHeight: 'var(--line-normal)',
  },
  detailsTerm: {
    color: 'var(--color-text-muted)',
  },
  // Classic `dd + dt { margin-top: var(--space-2) }` under 480px — StyleX
  // cannot express the sibling selector, so every dt after the first carries
  // the spacing explicitly.
  detailsTermSpaced: {
    marginTop: { default: 0, [NARROW]: 'var(--space-2)' },
  },
  detailsDesc: {
    display: 'flex',
    minWidth: 0,
    margin: 0,
    alignItems: 'center',
    gap: 'var(--space-2)',
    color: 'var(--color-text)',
    fontVariantNumeric: 'tabular-nums',
    overflowWrap: 'anywhere',
  },
  list: {
    display: 'grid',
    gap: 'var(--space-3)',
    margin: 0,
    padding: 0,
    listStyle: 'none',
  },
  row: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'baseline',
    justifyContent: 'space-between',
    gap: 'var(--space-2)',
  },
  identity: {
    minWidth: 0,
    color: 'var(--color-text)',
    fontWeight: 600,
    overflowWrap: 'anywhere',
  },
  outcome: {
    flex: 'none',
    fontSize: 'var(--text-sm)',
  },
  // The --color-text-* status tokens are undefined in both themes today; the
  // classic fallback chain is kept verbatim so a future token lands identically.
  outcomeSuccess: {
    color: 'var(--color-text-success, var(--color-text))',
  },
  outcomeDanger: {
    color: 'var(--color-text-danger, var(--color-text))',
  },
  outcomeWarning: {
    color: 'var(--color-text-warning, var(--color-text))',
  },
  disabledBadge: {
    marginLeft: 'var(--space-2)',
  },
  meta: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 'var(--space-2)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  toggle: {
    display: 'grid',
    gap: 'var(--space-3)',
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 'var(--space-3)',
  },
  toggleTitle: {
    margin: 0,
    color: 'var(--color-text)',
    // Classic declares var(--text-base), which is undefined — it resolves to
    // inherit, i.e. --text-body. Use the resolved value directly.
    fontSize: 'var(--text-body)',
    fontWeight: 600,
  },
  toggleHint: {
    margin: 0,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
    lineHeight: 'var(--line-normal)',
  },
  toggleActions: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
  },
  toggleList: {
    display: 'grid',
    gap: 'var(--space-2)',
    margin: 0,
    padding: 0,
    listStyle: 'none',
  },
  toggleRow: {
    display: 'flex',
    alignItems: { default: 'center', [NARROW]: 'flex-start' },
    flexDirection: { default: 'row', [NARROW]: 'column' },
    justifyContent: 'space-between',
    gap: { default: 'var(--space-2)', [NARROW]: 'var(--space-1)' },
    minWidth: 0,
  },
  toggleIdentity: {
    minWidth: 0,
    color: 'var(--color-text)',
    fontWeight: 600,
    overflowWrap: 'anywhere',
  },
  toggleState: {
    flex: 'none',
    fontSize: 'var(--text-sm)',
  },
  toggleStateEnabled: {
    color: 'var(--color-text-success, var(--color-text))',
  },
  toggleStateDisabled: {
    color: 'var(--color-text-muted)',
  },
  toggleFooter: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  toggleNote: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  // Classic AppButton variant="link" size="inline" — baseline-aligned text
  // button that underlines on hover (same pattern as LogDetailDrawer).
  linkButton: {
    display: 'inline',
    minHeight: 0,
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: { default: 'var(--color-action)', ':hover': 'var(--color-action-hover)' },
    padding: 0,
    verticalAlign: 'baseline',
    cursor: 'pointer',
    textDecorationLine: { default: 'none', ':hover': 'underline' },
  },
  // Classic CopyChip — unbordered mono value+icon button with transient
  // feedback popover beneath it.
  copyChipWrap: {
    position: 'relative',
    display: 'inline-flex',
    width: 'auto',
    maxWidth: '100%',
    minWidth: 0,
  },
  copyChip: {
    display: 'inline-flex',
    width: 'auto',
    maxWidth: '100%',
    minWidth: 0,
    minHeight: { default: 'var(--control-compact)', [TOUCH]: 'var(--touch-target)' },
    alignItems: 'center',
    justifyContent: 'flex-start',
    gap: 7,
    borderWidth: 0,
    borderRadius: 'var(--radius-tag)',
    backgroundColor: { default: 'transparent', ':hover': 'var(--color-surface-sunken)' },
    color: { default: 'var(--color-text-faint)', ':hover': 'var(--color-action)' },
    paddingBlock: 3,
    paddingInline: 0,
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
    cursor: 'pointer',
    transitionProperty: 'color, background-color',
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
  },
  copyChipValue: {
    display: 'block',
    minWidth: 0,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  // [data-state] wins over :hover in the classic (equal specificity, later
  // rule) — so success/failure pin both slots.
  copyChipSuccess: {
    backgroundColor: {
      default: 'var(--color-surface-sunken)',
      ':hover': 'var(--color-surface-sunken)',
    },
    color: { default: 'var(--color-success)', ':hover': 'var(--color-success)' },
  },
  copyChipFailure: {
    backgroundColor: {
      default: 'var(--color-surface-sunken)',
      ':hover': 'var(--color-surface-sunken)',
    },
    color: { default: 'var(--color-danger)', ':hover': 'var(--color-danger)' },
  },
  copyChipFeedback: {
    position: 'absolute',
    zIndex: 80, // --z-popover (stylex requires a numeric literal)
    top: 'calc(100% + var(--space-1))',
    insetInlineStart: 0,
    width: 'max-content',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-feedback-success-border, var(--color-border-subtle))',
    borderRadius: 'var(--radius-tag)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-success)',
    paddingBlock: 5,
    paddingInline: 7,
    boxShadow: 'var(--shadow-card)',
    fontSize: 'var(--text-sm)',
    whiteSpace: 'nowrap',
    pointerEvents: 'none',
  },
  copyChipFeedbackFailure: {
    borderColor: 'var(--color-feedback-danger-border, var(--color-danger))',
    color: 'var(--color-danger)',
  },
})

type CopyChipState = 'idle' | 'success' | 'failure'

/**
 * Classic ui/CopyChip (leading/trailing icon layouts): a borderless mono
 * button that copies `value`, shows a 2s status bubble, and falls back to the
 * manual-copy dialog when the clipboard API is unavailable.
 */
function LogIdCopyChip({
  value,
  label,
  successLabel,
  failureLabel,
  layout = 'leading',
}: {
  value: string
  label: string
  successLabel: string
  failureLabel: string
  layout?: 'leading' | 'trailing'
}) {
  const { copy, pending, reset, dialog } = useClipboardCopy()
  const [state, setState] = useState<CopyChipState>('idle')
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  // Classic watch(props.value, flush: 'sync'): value churn drops pending
  // feedback and any open fallback — render-time adjustment, not an effect.
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
      const result = await copy(value)
      if (result === 'cancelled') return
      // 'fallback' leaves the chip idle: the fallback dialog owns the flow.
      setState(result === 'success' ? 'success' : 'idle')
    } catch {
      setState('failure')
    }
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setState('idle'), 2_000)
  }

  return (
    <span {...stylex.props(styles.copyChipWrap)}>
      <button
        type="button"
        {...stylex.props(
          styles.copyChip,
          state === 'success' && styles.copyChipSuccess,
          state === 'failure' && styles.copyChipFailure,
        )}
        data-state={state}
        aria-label={label}
        aria-busy={pending}
        disabled={pending}
        onClick={() => void copyValue()}
      >
        {layout === 'leading' && <Copy size={14} aria-hidden="true" />}
        <span {...stylex.props(styles.copyChipValue)}>{value}</span>
        {layout === 'trailing' && <Copy size={14} aria-hidden="true" />}
      </button>
      {state !== 'idle' && (
        <span
          {...stylex.props(
            styles.copyChipFeedback,
            state === 'failure' && styles.copyChipFeedbackFailure,
          )}
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

interface ProbeGroup {
  group_id: number
  group_name: string
  outcome: ModelProbeOutcome
}

export interface ModelProbeDialogProps {
  open: boolean
  pending: boolean
  failed: boolean
  stopped: boolean
  results: readonly ModelProbeResultDto[]
  disabledGroupIds: readonly number[]
  completed: number
  total: number
  groupEnabledById?: ReadonlyMap<number, boolean>
  applying: boolean
  hideGroupControls?: boolean
  /** classic update:open */
  onOpenChange(open: boolean): void
  onStop(): void
  onViewLog(logId: string): void
  /** classic apply-enabled */
  onApplyEnabled?(changes: Map<number, boolean>): void
}

/** Classic features/models/ModelProbeDialog.vue — the model-liveness probe dialog. */
export function ModelProbeDialog({
  open,
  pending,
  failed,
  stopped,
  results,
  disabledGroupIds,
  completed,
  total,
  groupEnabledById,
  applying,
  hideGroupControls = false,
  onOpenChange,
  onStop,
  onViewLog,
  onApplyEnabled,
}: ModelProbeDialogProps) {
  const t = useT()
  const intl = useIntl()

  const detail = results.length === 1 ? results[0] : undefined
  const summary = { passed: 0, failed: 0, inconclusive: 0 }
  for (const result of results) summary[result.outcome] += 1

  const disabledGroupIdSet = new Set(disabledGroupIds)

  const [proposedEnabled, setProposedEnabled] = useState<ReadonlyMap<number, boolean>>(
    () => new Map<number, boolean>(),
  )
  const [initialized, setInitialized] = useState(false)

  // Classic watch(props.open): opening re-arms the one-shot seed; closing drops
  // every proposal. Render-time adjustments, not effects.
  const [wasOpen, setWasOpen] = useState(open)
  if (wasOpen !== open) {
    setWasOpen(open)
    if (open) {
      setInitialized(false)
    } else {
      setProposedEnabled(new Map())
    }
  }

  // Classic watch(props.pending): a fresh run discards proposals so a new
  // round of results never inherits the previous proposal.
  const [wasPending, setWasPending] = useState(pending)
  if (wasPending !== pending) {
    setWasPending(pending)
    if (pending) {
      setInitialized(false)
      setProposedEnabled(new Map())
    }
  }

  // Classic watch([open, pending, results], immediate): once a run settles
  // with results (and no request failure), seed the proposal map from the
  // current group states exactly once per session.
  if (open && !pending && !failed && results.length > 0 && !initialized) {
    const next = new Map<number, boolean>()
    for (const result of results) {
      next.set(result.group_id, groupEnabledById?.get(result.group_id) ?? true)
    }
    setProposedEnabled(next)
    setInitialized(true)
  }

  function currentEnabled(groupId: number): boolean {
    return groupEnabledById?.get(groupId) ?? true
  }

  const groups: ProbeGroup[] = (() => {
    const byId = new Map<
      number,
      { group_id: number; group_name: string; outcomes: ModelProbeOutcome[] }
    >()
    for (const result of results) {
      const entry = byId.get(result.group_id)
      if (entry) {
        entry.outcomes.push(result.outcome)
      } else {
        byId.set(result.group_id, {
          group_id: result.group_id,
          group_name: result.group_name,
          outcomes: [result.outcome],
        })
      }
    }
    return [...byId.values()].map(({ group_id, group_name, outcomes }) => {
      let outcome: ModelProbeOutcome
      if (outcomes.some((o) => o === 'failed')) outcome = 'failed'
      else if (outcomes.every((o) => o === 'passed')) outcome = 'passed'
      else outcome = 'inconclusive'
      return { group_id, group_name, outcome }
    })
  })()

  // Classic computed `diff`: proposals that differ from the current state.
  const changes = new Map<number, boolean>()
  for (const [id, value] of proposedEnabled) {
    if (value !== currentEnabled(id)) changes.set(id, value)
  }
  const hasChanges = changes.size > 0

  function setOpen(next: boolean): void {
    // Classic setOpen + dismissible guard: closing is blocked while pending
    // (Escape, scrim, header X, and the footer Close all land here).
    if (!next && pending) return
    onOpenChange(next)
  }

  function tone(result: ModelProbeResultDto): 'success' | 'danger' | 'warning' {
    if (result.outcome === 'passed') return 'success'
    if (result.outcome === 'failed') return 'danger'
    return 'warning'
  }

  function outcomeLabel(result: ModelProbeResultDto): string {
    return t(`monitor.modelProbe.outcome.${result.outcome}`)
  }

  function reasonLabel(result: ModelProbeResultDto): string {
    return result.reason === null ? '' : t(`monitor.modelProbe.reason.${result.reason}`)
  }

  function routeModeLabel(result: ModelProbeResultDto): string {
    return result.route_mode === null
      ? t('monitor.modelProbe.unknownValue')
      : t(`monitor.modelProbe.routeMode.${result.route_mode}`)
  }

  function groupLabel(result: ModelProbeResultDto): string {
    return result.group_name === '' ? `#${result.group_id}` : result.group_name
  }

  function isDisabledGroup(result: ModelProbeResultDto): boolean {
    return disabledGroupIdSet.has(result.group_id)
  }

  function groupToggleLabel(groupId: number, groupName: string): string {
    const label = groupName === '' ? `#${groupId}` : groupName
    return `${label}: ${t('monitor.modelProbe.toggle.title')}`
  }

  function setProposed(groupId: number, value: boolean): void {
    const next = new Map(proposedEnabled)
    next.set(groupId, value)
    setProposedEnabled(next)
  }

  function enablePassedGroups(): void {
    const next = new Map(proposedEnabled)
    for (const group of groups) {
      if (group.outcome === 'passed') next.set(group.group_id, true)
    }
    setProposedEnabled(next)
  }

  function disableNotPassedGroups(): void {
    const next = new Map(proposedEnabled)
    for (const group of groups) {
      if (group.outcome !== 'passed') next.set(group.group_id, false)
    }
    setProposedEnabled(next)
  }

  function resetProposedGroups(): void {
    const next = new Map<number, boolean>()
    for (const group of groups) {
      next.set(group.group_id, currentEnabled(group.group_id))
    }
    setProposedEnabled(next)
  }

  function apply(): void {
    if (applying || changes.size === 0) return
    onApplyEnabled?.(new Map(changes))
  }

  return (
    <Dialog
      isOpen={open}
      onOpenChange={setOpen}
      // Classic model-probe-dialog__surface: 720px ledger-width dialog whose
      // body owns the only scroll region while the footer stays pinned.
      width={720}
      maxHeight="calc(100dvh - 36px)"
    >
      <Layout
        header={
          <DialogHeader
            title={t('monitor.modelProbe.title')}
            subtitle={t('monitor.modelProbe.description')}
            onOpenChange={setOpen}
            hasDivider
          />
        }
        content={
          <LayoutContent isScrollable>
            <div {...stylex.props(styles.body)}>
              {pending && results.length === 0 && (
                <div {...stylex.props(styles.loading)} role="status">
                  <LoaderCircle
                    size={18}
                    aria-hidden="true"
                    {...stylex.props(styles.loadingSpin)}
                  />
                  <span>{t('monitor.modelProbe.loading')}</span>
                </div>
              )}
              {failed && <Banner status="error" title={t('monitor.modelProbe.requestFailed')} />}
              {stopped && <Banner status="warning" title={t('monitor.modelProbe.stopped')} />}
              {total > 1 && (
                <Banner
                  status="warning"
                  title={t('monitor.modelProbe.progress', { completed, total })}
                />
              )}
              {total > 1 && results.length > 0 && (
                <Banner
                  status="warning"
                  title={t('monitor.modelProbe.summary', {
                    passed: summary.passed,
                    failed: summary.failed,
                    inconclusive: summary.inconclusive,
                  })}
                />
              )}

              {detail !== undefined ? (
                <dl {...stylex.props(styles.details)}>
                  <dt {...stylex.props(styles.detailsTerm)}>
                    {t('monitor.modelProbe.fields.group')}
                  </dt>
                  <dd {...stylex.props(styles.detailsDesc)}>
                    {groupLabel(detail)}
                    {isDisabledGroup(detail) && (
                      <Badge
                        variant="neutral"
                        label={t('monitor.modelProbe.disabledBadge')}
                        xstyle={styles.disabledBadge}
                      />
                    )}
                  </dd>
                  <dt {...stylex.props(styles.detailsTerm, styles.detailsTermSpaced)}>
                    {t('monitor.modelProbe.fields.model')}
                  </dt>
                  <dd {...stylex.props(styles.detailsDesc)}>{detail.model}</dd>
                  <dt {...stylex.props(styles.detailsTerm, styles.detailsTermSpaced)}>
                    {t('monitor.modelProbe.fields.outcome')}
                  </dt>
                  <dd {...stylex.props(styles.detailsDesc)}>{outcomeLabel(detail)}</dd>
                  {detail.outcome === 'passed' && (
                    <>
                      <dt {...stylex.props(styles.detailsTerm, styles.detailsTermSpaced)}>
                        {t('monitor.modelProbe.fields.recovered')}
                      </dt>
                      <dd {...stylex.props(styles.detailsDesc)}>
                        {detail.recovered
                          ? t('monitor.modelProbe.recovered')
                          : t('monitor.modelProbe.recoveryNotNeeded')}
                      </dd>
                    </>
                  )}
                  {detail.reason && (
                    <>
                      <dt {...stylex.props(styles.detailsTerm, styles.detailsTermSpaced)}>
                        {t('monitor.modelProbe.fields.reason')}
                      </dt>
                      <dd {...stylex.props(styles.detailsDesc)}>{reasonLabel(detail)}</dd>
                    </>
                  )}
                  {detail.protocol && (
                    <>
                      <dt {...stylex.props(styles.detailsTerm, styles.detailsTermSpaced)}>
                        {t('monitor.modelProbe.fields.protocol')}
                      </dt>
                      <dd {...stylex.props(styles.detailsDesc)}>{detail.protocol}</dd>
                    </>
                  )}
                  <dt {...stylex.props(styles.detailsTerm, styles.detailsTermSpaced)}>
                    {t('monitor.modelProbe.fields.routeMode')}
                  </dt>
                  <dd {...stylex.props(styles.detailsDesc)}>{routeModeLabel(detail)}</dd>
                  {detail.status_code !== null && (
                    <>
                      <dt {...stylex.props(styles.detailsTerm, styles.detailsTermSpaced)}>
                        {t('monitor.modelProbe.fields.statusCode')}
                      </dt>
                      <dd {...stylex.props(styles.detailsDesc)}>{detail.status_code}</dd>
                    </>
                  )}
                  {detail.latency_ms !== null && (
                    <>
                      <dt {...stylex.props(styles.detailsTerm, styles.detailsTermSpaced)}>
                        {t('monitor.modelProbe.fields.latency')}
                      </dt>
                      <dd {...stylex.props(styles.detailsDesc)}>
                        {t('monitor.modelProbe.latency', {
                          value: intl.formatNumber(detail.latency_ms),
                        })}
                      </dd>
                    </>
                  )}
                  {detail.credential_label && (
                    <>
                      <dt {...stylex.props(styles.detailsTerm, styles.detailsTermSpaced)}>
                        {t('monitor.modelProbe.fields.credential')}
                      </dt>
                      <dd {...stylex.props(styles.detailsDesc)}>{detail.credential_label}</dd>
                    </>
                  )}
                  <dt {...stylex.props(styles.detailsTerm, styles.detailsTermSpaced)}>
                    {t('monitor.modelProbe.fields.logId')}
                  </dt>
                  <dd {...stylex.props(styles.detailsDesc)}>
                    {(() => {
                      const logId = detail.log_id
                      return logId ? (
                        <>
                          <LogIdCopyChip
                            value={logId}
                            label={t('monitor.modelProbe.fields.logId')}
                            successLabel={t('common.copied')}
                            failureLabel={t('common.copyFailed')}
                          />
                          <button
                            type="button"
                            {...stylex.props(styles.linkButton)}
                            onClick={() => onViewLog(logId)}
                          >
                            {t('monitor.modelProbe.viewLog')}
                          </button>
                        </>
                      ) : (
                        t('monitor.modelProbe.notExecuted')
                      )
                    })()}
                  </dd>
                  <dt {...stylex.props(styles.detailsTerm, styles.detailsTermSpaced)}>
                    {t('monitor.modelProbe.fields.testedAt')}
                  </dt>
                  <dd {...stylex.props(styles.detailsDesc)}>
                    {formatLocalInstant(detail.tested_at_ms, intl.locale)}
                  </dd>
                </dl>
              ) : results.length > 1 ? (
                <ul {...stylex.props(styles.list)}>
                  {results.map((result) => {
                    const logId = result.log_id
                    return (
                      <li key={`${result.group_id}:${result.model}`}>
                        <div {...stylex.props(styles.row)}>
                          <span {...stylex.props(styles.identity)}>
                            {groupLabel(result)} · {result.model}
                            {isDisabledGroup(result) && (
                              <Badge
                                variant="neutral"
                                label={t('monitor.modelProbe.disabledBadge')}
                                xstyle={styles.disabledBadge}
                              />
                            )}
                          </span>
                          <span
                            {...stylex.props(
                              styles.outcome,
                              tone(result) === 'success' && styles.outcomeSuccess,
                              tone(result) === 'danger' && styles.outcomeDanger,
                              tone(result) === 'warning' && styles.outcomeWarning,
                            )}
                          >
                            {outcomeLabel(result)}
                          </span>
                        </div>
                        <div {...stylex.props(styles.meta)}>
                          <span>
                            {t('monitor.modelProbe.fields.protocol')}:{' '}
                            {result.protocol ?? t('monitor.modelProbe.unknownValue')}
                          </span>
                          <span>
                            {t('monitor.modelProbe.fields.routeMode')}: {routeModeLabel(result)}
                          </span>
                          {result.reason && <span>{reasonLabel(result)}</span>}
                          {result.credential_label && (
                            <span>
                              {t('monitor.modelProbe.credential', {
                                credential: result.credential_label,
                              })}
                            </span>
                          )}
                          {result.outcome === 'passed' && (
                            <span>
                              {result.recovered
                                ? t('monitor.modelProbe.recovered')
                                : t('monitor.modelProbe.recoveryNotNeeded')}
                            </span>
                          )}
                          {logId && (
                            <>
                              <LogIdCopyChip
                                value={logId}
                                layout="trailing"
                                label={t('monitor.modelProbe.fields.logId')}
                                successLabel={t('common.copied')}
                                failureLabel={t('common.copyFailed')}
                              />
                              <button
                                type="button"
                                {...stylex.props(styles.linkButton)}
                                onClick={() => onViewLog(logId)}
                              >
                                {t('monitor.modelProbe.viewLog')}
                              </button>
                            </>
                          )}
                        </div>
                      </li>
                    )
                  })}
                </ul>
              ) : null}

              {!hideGroupControls && !pending && results.length > 0 && !failed && (
                <section
                  {...stylex.props(styles.toggle)}
                  aria-labelledby="model-probe-toggle-title"
                >
                  <h3 id="model-probe-toggle-title" {...stylex.props(styles.toggleTitle)}>
                    {t('monitor.modelProbe.toggle.title')}
                  </h3>
                  <p {...stylex.props(styles.toggleHint)}>{t('monitor.modelProbe.toggle.hint')}</p>

                  <div {...stylex.props(styles.toggleActions)}>
                    <Button
                      variant="secondary"
                      size="sm"
                      label={t('monitor.modelProbe.toggle.enablePassed')}
                      onClick={enablePassedGroups}
                    />
                    <Button
                      variant="secondary"
                      size="sm"
                      label={t('monitor.modelProbe.toggle.disableNotPassed')}
                      onClick={disableNotPassedGroups}
                    />
                    <Button
                      variant="secondary"
                      size="sm"
                      label={t('monitor.modelProbe.toggle.reset')}
                      onClick={resetProposedGroups}
                    />
                  </div>

                  <ul {...stylex.props(styles.toggleList)}>
                    {groups.map((group) => (
                      <li key={group.group_id} {...stylex.props(styles.toggleRow)}>
                        <span {...stylex.props(styles.toggleIdentity)}>
                          {group.group_name === '' ? `#${group.group_id}` : group.group_name}
                        </span>
                        <span
                          {...stylex.props(
                            styles.toggleState,
                            proposedEnabled.get(group.group_id)
                              ? styles.toggleStateEnabled
                              : styles.toggleStateDisabled,
                          )}
                        >
                          {proposedEnabled.get(group.group_id)
                            ? t('monitor.modelProbe.toggle.groupEnabled')
                            : t('monitor.modelProbe.toggle.groupDisabled')}
                        </span>
                        <Switch
                          label={groupToggleLabel(group.group_id, group.group_name)}
                          isLabelHidden
                          size="sm"
                          value={proposedEnabled.get(group.group_id) ?? true}
                          isDisabled={applying}
                          onChange={(checked) => setProposed(group.group_id, checked)}
                        />
                      </li>
                    ))}
                  </ul>

                  <div {...stylex.props(styles.toggleFooter)}>
                    <Button
                      variant="primary"
                      size="sm"
                      isDisabled={applying || !hasChanges}
                      label={t('monitor.modelProbe.toggle.apply')}
                      onClick={apply}
                    />
                    {!hasChanges && (
                      <span {...stylex.props(styles.toggleNote)}>
                        {t('monitor.modelProbe.toggle.noChanges')}
                      </span>
                    )}
                  </div>
                </section>
              )}
            </div>
          </LayoutContent>
        }
        footer={
          <LayoutFooter hasDivider>
            {pending && (
              <Button
                variant="secondary"
                size="sm"
                label={t('monitor.modelProbe.stop')}
                onClick={onStop}
              />
            )}
            <Button
              variant="secondary"
              size="sm"
              isDisabled={pending}
              label={t('monitor.modelProbe.close')}
              onClick={() => setOpen(false)}
            />
          </LayoutFooter>
        }
      />
    </Dialog>
  )
}
