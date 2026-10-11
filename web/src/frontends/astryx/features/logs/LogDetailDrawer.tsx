import * as stylex from '@stylexjs/stylex'
import { Badge, Skeleton, Tooltip } from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import {
  ChevronRight,
  CircleAlert,
  CircleCheck,
  CircleHelp,
  CircleOff,
  Copy,
  RefreshCw,
  TriangleAlert,
  type LucideIcon,
} from 'lucide-react'
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { useIntl } from 'react-intl'

import type { ChannelDto } from '@shared/control/resources/channels'
import { revealCredential } from '@shared/control/resources/credentials'
import {
  requestLogDetailQueryOptions,
  type RequestLogAttemptDto,
  type RequestLogPricingLineDto,
  type RequestLogReasoningDto,
  type RequestLogFeedbackStatus,
  type RequestLogRouteMode,
  type RequestLogStatus,
} from '@shared/control/resources/request-logs'
import {
  formatLogDuration,
  formatLogTokenCount,
  formatRequestLogReasoning,
  requestLogAttemptReasonSequences,
  requestLogAttemptReasonText,
  requestLogCostDisplayState,
  requestLogFirstScreen,
  requestLogUsageDisplayState,
} from '@shared/domain/monitor/log-format'
import type { MessageId } from '@shared/i18n/message-ids'
import { formatReceiptFormulaLine } from '@shared/domain/monitor/receipt-formula'
import { formatCacheHitRate } from '@shared/lib/cache-rate'
import {
  formatEstimatedCost,
  formatExactNanoUSD,
  formatISOInstant,
  formatLocalInstantWithSeconds,
} from '@shared/lib/format'

import { useStableLoading } from '../../app/collection-loading'
import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'
import { useClipboardCopy } from '../../app/use-clipboard-copy'
import { CopyButton } from '../../components/CopyButton'
import { DetailPanel } from '../../components/DetailPanel'
import { LogRouteIdentity } from './LogRouteIdentity'
import { PricingModeIndicator } from './PricingModeIndicator'

const NARROW = '@media (max-width: 520px)'
const TOUCH = '@media (max-width: 860px)'

type StatusTone = 'neutral' | 'success' | 'warning' | 'danger'

const badgeVariants = {
  neutral: 'neutral',
  success: 'success',
  warning: 'warning',
  danger: 'error',
} as const

// Classic StatusBadge always renders a tone icon (12px at compact size).
const badgeIcons: Record<StatusTone, LucideIcon> = {
  neutral: CircleHelp,
  success: CircleCheck,
  warning: CircleAlert,
  danger: CircleOff,
}

const styles = stylex.create({
  detail: {
    display: 'grid',
    minWidth: 0,
  },
  summary: {
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    rowGap: 8,
    columnGap: 12,
    paddingBlock: 16,
    paddingInline: 0,
  },
  time: {
    marginLeft: 'auto',
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
  },
  requestId: {
    display: 'flex',
    width: '100%',
    minWidth: 0,
    alignItems: 'center',
    gap: 6,
  },
  requestIdCode: {
    minWidth: 0,
    overflow: 'hidden',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  requestIdCopy: {
    width: 28,
    height: 28,
    borderColor: 'transparent',
  },
  route: {
    display: 'flex',
    minWidth: 0,
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 4,
  },
  routeCopyChip: {
    minHeight: 22,
    padding: 0,
  },
  section: {
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingBlock: 16,
    paddingInline: 0,
  },
  attemptSection: {
    paddingBlock: 8,
  },
  sectionTitle: {
    marginTop: 0,
    marginBottom: 12,
    marginInline: 0,
    fontSize: 'var(--text-sm)',
    fontWeight: 650,
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'repeat(2, minmax(0, 1fr))',
      [NARROW]: 'minmax(0, 1fr)',
    },
    rowGap: 12,
    columnGap: 18,
    margin: 0,
  },
  gridCell: {
    minWidth: 0,
  },
  term: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  description: {
    marginTop: 3,
    marginBottom: 0,
    marginInline: 0,
    color: 'var(--color-text)',
    fontSize: 'var(--text-sm)',
    overflowWrap: 'anywhere',
  },
  reasoning: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 400,
  },
  wide: {
    gridColumn: {
      default: '1 / -1',
      [NARROW]: 'auto',
    },
  },
  errorMessage: {
    display: 'grid',
    gap: 4,
    marginTop: 16,
    borderLeftWidth: 2,
    borderLeftStyle: 'solid',
    borderLeftColor: 'var(--color-danger)',
    backgroundColor: 'var(--color-danger-bg)',
    paddingBlock: 10,
    paddingInline: 12,
  },
  errorMessageAttempt: {
    marginTop: 12,
    borderLeftColor: 'var(--color-border-control)',
    backgroundColor: 'transparent',
    paddingBottom: 0,
    paddingInlineEnd: 0,
  },
  errorMessageLabel: {
    margin: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  errorMessageCode: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'baseline',
    gap: 8,
    margin: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  errorMessageCodeValue: {
    color: 'var(--color-danger)',
    overflowWrap: 'anywhere',
  },
  errorMessageContent: {
    margin: 0,
    color: 'var(--color-text)',
    fontSize: 'var(--text-sm)',
    lineHeight: 1.6,
    overflowWrap: 'anywhere',
    whiteSpace: 'pre-wrap',
  },
  errorMessageCollapsed: {
    display: '-webkit-box',
    overflow: 'hidden',
    WebkitBoxOrient: 'vertical',
    WebkitLineClamp: 3,
  },
  // .log-attempt__reason .log-error-message__content--collapsed
  errorMessageCollapsedAttempt: {
    WebkitLineClamp: 2,
  },
  errorMessageToggle: {
    justifySelf: 'start',
    marginTop: 2,
  },
  formula: {
    display: 'grid',
    gap: 4,
    fontFamily: 'var(--font-mono)',
    lineHeight: 1.6,
  },
  cost: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  timingSlow: {
    color: 'var(--color-warning)',
  },
  timingFaulty: {
    color: 'var(--color-danger)',
  },
  modelObservation: {
    display: 'grid',
    gap: 10,
    marginTop: 12,
    borderLeftWidth: 2,
    borderLeftStyle: 'solid',
    borderLeftColor: 'var(--color-border-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    paddingBlock: 10,
    paddingInline: 12,
  },
  modelObservationMismatch: {
    borderLeftColor: 'var(--color-warning)',
  },
  modelObservationHeading: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
    fontSize: 'var(--text-label-xs)',
  },
  attemptChain: {
    margin: 0,
  },
  // `.log-attempt-chain summary` — marker removal lands via
  // [data-details-chevron-scope] in entry.css (WebKit pseudo-element).
  chainSummary: {
    display: 'flex',
    minHeight: 24,
    alignItems: 'center',
    gap: 6,
    color: 'var(--color-text)',
    cursor: 'pointer',
    fontSize: 'var(--text-sm)',
    fontWeight: 650,
    listStyleType: 'none',
  },
  chevron: {
    color: 'var(--color-text-faint)',
    transitionProperty: 'transform',
    transitionDuration: '140ms',
    transitionTimingFunction: 'ease',
  },
  chainCount: {
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 400,
  },
  attemptFirst: {
    marginTop: 4,
  },
  attemptBordered: {
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
  },
  attempt: {
    paddingBlock: 8,
    paddingInline: 0,
  },
  attemptHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
    marginBottom: 8,
    color: 'var(--color-text)',
    fontSize: 'var(--text-sm)',
  },
  attemptPrimary: {
    display: 'grid',
    minWidth: 0,
    gap: 4,
  },
  attemptAction: {
    display: 'flex',
    minWidth: 0,
    flexWrap: 'wrap',
    alignItems: 'baseline',
    gap: 6,
    color: 'var(--color-text)',
    fontSize: 'var(--text-label-xs)',
  },
  attemptActionLabel: {
    color: 'var(--color-text-faint)',
  },
  attemptReason: {
    marginTop: 8,
    borderLeftWidth: 2,
    borderLeftStyle: 'solid',
    borderLeftColor: 'var(--color-border-control)',
    paddingLeft: 12,
  },
  attemptDetails: {
    marginTop: 8,
  },
  attemptDetailsSummary: {
    display: 'flex',
    minHeight: 22,
    alignItems: 'center',
    gap: 6,
    color: 'var(--color-text-faint)',
    cursor: 'pointer',
    fontSize: 'var(--text-label-xs)',
    listStyleType: 'none',
  },
  usageChainGrid: {
    marginTop: 12,
  },
  attemptDetailsGrid: {
    marginTop: 10,
  },
  empty: {
    margin: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  badge: {
    fontWeight: 400,
  },
  skeletonSurface: {
    display: 'grid',
    minWidth: 0,
    alignContent: 'start',
    gap: 'var(--space-4)',
    minHeight: 660,
  },
  skeletonConcealed: {
    visibility: 'hidden',
  },
  srOnly: {
    position: 'absolute',
    width: 1,
    height: 1,
    margin: -1,
    padding: 0,
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
    borderWidth: 0,
  },
  queryFeedback: {
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
  queryFeedbackError: {
    borderColor: 'var(--color-danger)',
    backgroundColor: 'var(--color-danger-bg)',
    color: 'var(--color-danger)',
  },
  queryFeedbackText: {
    minWidth: 0,
    flex: '1 1 auto',
  },
  queryFeedbackRetry: {
    display: 'inline-flex',
    flex: 'none',
    minHeight: 44,
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
  // Classic AppButton variant="link" size="inline" — a baseline-aligned text
  // button that underlines on hover.
  linkToggle: {
    display: 'inline',
    minHeight: 0,
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: { default: 'var(--color-action)', ':hover': 'var(--color-action-hover)' },
    padding: 0,
    // The astryx reset already applies `font: inherit` to buttons.
    verticalAlign: 'baseline',
    cursor: 'pointer',
    textDecorationLine: { default: 'none', ':hover': 'underline' },
  },
  pricingModeIndicator: {
    display: 'inline-flex',
    width: 18,
    height: 18,
    flex: '0 0 18px',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 0,
    borderRadius: 'var(--radius-tag)',
    backgroundColor: { default: 'transparent', ':hover': 'var(--color-surface-sunken)' },
    color: { default: 'var(--color-text-faint)', ':hover': 'var(--color-text)' },
    padding: 0,
    cursor: 'help',
    outlineWidth: { ':focus-visible': '2px' },
    outlineStyle: { ':focus-visible': 'solid' },
    outlineColor: { ':focus-visible': 'var(--color-focus)' },
    outlineOffset: { ':focus-visible': '2px' },
  },
  invalidInstant: {
    color: 'var(--color-danger)',
    overflowWrap: 'anywhere',
  },
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
  // 图标模式没有文字撑开点击区，补足到与其他紧凑控件一致的尺寸。
  copyChipIcon: {
    width: 'var(--control-compact)',
    minWidth: 'var(--control-compact)',
    justifyContent: 'center',
    padding: 0,
  },
  copyChipSuccess: {
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-success)',
  },
  copyChipFailure: {
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-danger)',
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

/** Classic AppDateTime with precision="second". */
function LogInstant({ instant }: { instant: number }) {
  const dateTime = formatISOInstant(instant)
  if (dateTime === undefined) {
    return <span {...stylex.props(styles.invalidInstant)}>{String(instant)}</span>
  }
  return <time dateTime={dateTime}>{formatLocalInstantWithSeconds(instant)}</time>
}

/**
 * Classic OverflowTooltip: the tooltip only exists while the trigger actually
 * clips its content; a clipping trigger becomes tabbable so keyboard users can
 * reach the full text.
 */
function OverflowTip({
  content,
  xstyle,
  children,
}: {
  content: string
  xstyle?: stylex.StyleXStyles
  children: ReactNode
}) {
  const [element, setElement] = useState<HTMLElement | null>(null)
  const [overflowing, setOverflowing] = useState(false)

  useEffect(() => {
    if (element === null) return
    const update = (): void => {
      // The classic falls back to the trigger's own text when `content` is
      // empty; every caller here passes content, but keep the guard identical.
      const effective = content !== '' ? content : (element.textContent?.trim() ?? '')
      setOverflowing(
        effective.length > 0 &&
          (element.scrollWidth > element.clientWidth + 1 ||
            element.scrollHeight > element.clientHeight + 1),
      )
    }
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(update) : undefined
    observer?.observe(element)
    const fallbackFrame = observer === undefined ? requestAnimationFrame(update) : undefined
    return () => {
      observer?.disconnect()
      if (fallbackFrame !== undefined) cancelAnimationFrame(fallbackFrame)
    }
  }, [element, content])

  return (
    <Tooltip content={content} isEnabled={overflowing} placement="above">
      <code ref={setElement} tabIndex={overflowing ? 0 : undefined} {...stylex.props(xstyle)}>
        {children}
      </code>
    </Tooltip>
  )
}

/** Classic PricingModeIndicator.vue moved to ./PricingModeIndicator (shared with LogsTable). */

/**
 * Classic CopyChip layout="icon": masked values copy directly while
 * `resolveValue` (only wired for api_key channels) fetches the real credential
 * first — the chip never displays the revealed secret.
 */
function CredentialCopyChip({
  value,
  label,
  successLabel,
  failureLabel,
  resolveValue,
  xstyle,
}: {
  value: string
  label: string
  successLabel: string
  failureLabel: string
  resolveValue?: () => string | Promise<string>
  xstyle?: stylex.StyleXStyles
}) {
  const { copy, pending, dialog } = useClipboardCopy()
  const [state, setState] = useState<'idle' | 'success' | 'failure'>('idle')
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  useEffect(
    () => () => {
      if (timerRef.current !== undefined) clearTimeout(timerRef.current)
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
    if (timerRef.current !== undefined) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setState('idle'), 2_000)
  }

  return (
    <span {...stylex.props(styles.copyChipWrap)}>
      <button
        type="button"
        {...stylex.props(
          styles.copyChip,
          styles.copyChipIcon,
          state === 'success' && styles.copyChipSuccess,
          state === 'failure' && styles.copyChipFailure,
          xstyle,
        )}
        data-state={state}
        aria-label={label}
        aria-busy={pending}
        disabled={pending}
        onClick={() => void copyValue()}
      >
        <Copy size={14} aria-hidden="true" />
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

function logTranslator(t: (id: MessageId, values?: Record<string, string | number>) => string) {
  return (key: string, named?: Record<string, string | number>) => t(key as MessageId, named)
}

function statusTone(status: RequestLogStatus): StatusTone {
  if (status === 'success') return 'success'
  if (status === 'error') return 'danger'
  if (status === 'incomplete') return 'warning'
  return 'neutral'
}

function attemptTone(attempt: RequestLogAttemptDto): 'success' | 'danger' | 'warning' {
  if (attempt.failure_category === 'ok') return 'success'
  return attempt.will_retry ? 'warning' : 'danger'
}

function timingTone(status: RequestLogFeedbackStatus | null) {
  if (status === 'slow') return styles.timingSlow
  if (status === 'faulty') return styles.timingFaulty
  return null
}

function errorMessageNeedsDisclosure(message: string): boolean {
  return message.length > 240
}

export interface LogDetailDrawerProps {
  open: boolean
  requestId: string | undefined
  selfScoped?: boolean
  groupNames?: Record<number, string>
  groupsLoaded?: boolean
  providerUrls?: Record<number, string>
  channels?: Record<string, ChannelDto>
  onOpenChange: (open: boolean) => void
}

/** Classic LogDetailDrawer.vue — the ledger-style request log detail drawer. */
export function LogDetailDrawer({
  open,
  requestId,
  selfScoped = false,
  groupNames,
  groupsLoaded,
  providerUrls,
  channels,
  onOpenChange,
}: LogDetailDrawerProps) {
  const t = useT()
  const { locale } = useIntl()
  const { apiClient } = useAppServices()
  const detailQuery = useQuery(requestLogDetailQueryOptions(apiClient, requestId))
  const initialLoading = useStableLoading(open && detailQuery.isPending)
  const log = detailQuery.data

  const [errorMessageExpanded, setErrorMessageExpanded] = useState(true)
  const [expandedAttemptErrorMessages, setExpandedAttemptErrorMessages] = useState<Set<number>>(
    new Set(),
  )

  // Classic useAbortControllerPool: pending credential reveals die with the
  // drawer, the selected request, or a close.
  const copyControllersRef = useRef(new Set<AbortController>())
  const abortCopies = useCallback(() => {
    for (const controller of copyControllersRef.current) controller.abort()
    copyControllersRef.current.clear()
  }, [])
  useEffect(() => abortCopies, [abortCopies])
  useEffect(() => {
    abortCopies()
  }, [requestId, abortCopies])
  useEffect(() => {
    if (!open) abortCopies()
  }, [open, abortCopies])

  // Render-phase mirrors of the classic watchers — the React-sanctioned
  // "adjust state during render" replacement for watch() resets.
  const logRequestId = log?.request_id
  const [seenLogRequestId, setSeenLogRequestId] = useState(logRequestId)
  if (logRequestId !== seenLogRequestId) {
    setSeenLogRequestId(logRequestId)
    setExpandedAttemptErrorMessages(new Set(log?.attempts.map(({ sequence }) => sequence) ?? []))
  }

  const [seenRequestId, setSeenRequestId] = useState(requestId)
  if (requestId !== seenRequestId) {
    setSeenRequestId(requestId)
    setErrorMessageExpanded(true)
    setExpandedAttemptErrorMessages(new Set())
  }

  const [wasOpen, setWasOpen] = useState(open)
  if (open !== wasOpen) {
    setWasOpen(open)
    if (open) {
      setErrorMessageExpanded(true)
      setExpandedAttemptErrorMessages(new Set(log?.attempts.map(({ sequence }) => sequence) ?? []))
    }
  }

  let finalAttempt: RequestLogAttemptDto | null = null
  if (log !== undefined && log.attempts.length > 0) {
    const reversed = [...log.attempts].reverse()
    finalAttempt =
      reversed.find(
        (attempt) =>
          attempt.group_id === log.group_id &&
          attempt.channel_id === log.channel_id &&
          attempt.credential_id === log.credential_id,
      ) ?? reversed[0]
  }

  // 首屏结论（最终状态 / 尝试次数 / 关键原因）的取值顺序与去重都由 log-format 的
  // 纯函数固定，这里只把翻译函数传进去并暴露给模板。
  const firstScreen = log ? requestLogFirstScreen(log, logTranslator(t)) : null
  const keyReasonLabel = firstScreen?.key_reason_label ?? ''
  const keyReasonText = firstScreen?.key_reason_text ?? ''
  const keyReasonCode = firstScreen?.key_reason_code ?? ''
  const requestErrorCode = firstScreen?.request_error_code ?? ''

  const drawerDescription = t(
    selfScoped ? 'monitor.logs.drawer.descriptionSelfScoped' : 'monitor.logs.drawer.description',
  )
  const usageDisplayState = log ? requestLogUsageDisplayState(log) : 'not_applicable'
  const costDisplayState = log ? requestLogCostDisplayState(log) : 'not_applicable'
  const receipt =
    log?.attempts.find((attempt) => attempt.committed && attempt.pricing_receipt)
      ?.pricing_receipt ??
    log?.attempts.find((attempt) => attempt.pricing_receipt)?.pricing_receipt ??
    undefined

  function channelDefinition(channelID: string | null | undefined): ChannelDto | null {
    if (!channelID) return null
    return channels?.[channelID] ?? null
  }

  function channelName(channelID: string | null | undefined): string {
    if (!channelID) return '—'
    return channelDefinition(channelID)?.name.trim() || channelID
  }

  const pricingIdentity = (() => {
    if (!receipt) return '—'
    return `${channelName(receipt.rule.channel_id)} · ${receipt.rule.model_id}`
  })()

  const cacheRows =
    !log || usageDisplayState !== 'reported'
      ? []
      : [
          { label: t('monitor.logs.tokens.cacheRead'), value: log.cache_read_tokens },
          { label: t('monitor.logs.tokens.cacheWrite5m'), value: log.cache_write_5m_tokens },
          { label: t('monitor.logs.tokens.cacheWrite1h'), value: log.cache_write_1h_tokens },
          { label: t('monitor.logs.tokens.cacheWrite'), value: log.cache_write_unknown_tokens },
        ].filter(({ value }) => value !== '0')

  const cacheRateLabel =
    !log || cacheRows.length === 0
      ? '—'
      : formatCacheHitRate(log.cache_read_tokens, log.input_tokens, locale)

  function formatFormulaLine(line: RequestLogPricingLineDto): string {
    return formatReceiptFormulaLine(line, locale)
  }

  const formula = (() => {
    const lines = receipt?.line_items ?? []
    const input = lines
      .filter((line) => line.code !== 'output')
      .map(formatFormulaLine)
      .join(' + ')
    const output = lines
      .filter((line) => line.code === 'output')
      .map(formatFormulaLine)
      .join(' + ')
    return {
      input: input || '—',
      output: output || '—',
    }
  })()

  const usageStateLabel = !log
    ? '—'
    : t(`monitor.logs.drawer.usage.state.${log.usage_state}` as MessageId)

  const costStateLabel =
    costDisplayState === 'complete'
      ? t('monitor.logs.drawer.usage.costState.priced')
      : t(`monitor.logs.drawer.usage.costState.${costDisplayState}` as MessageId)

  const costAmountLabel = (() => {
    if (!log) return '—'
    if (costDisplayState === 'complete') {
      return formatEstimatedCost(log.estimated_cost_nano_usd, locale)
    }
    if (costDisplayState === 'unpriced') return t('monitor.logs.cost.unpriced')
    return t('monitor.logs.cost.not_applicable')
  })()

  function operationLabel(operation: RequestLogAttemptDto['operation']): string {
    if (operation === null) return t('monitor.logs.protocolConversion.notRecorded')
    return t(`monitor.logs.operation.${operation}` as MessageId)
  }

  function upstreamProtocolLabel(
    upstreamProtocol: RequestLogAttemptDto['upstream_protocol'],
  ): string {
    if (upstreamProtocol === null) return t('monitor.logs.protocolConversion.notRecorded')
    return upstreamProtocol
  }

  function reasoningLabel(reasoning: RequestLogReasoningDto | null): string {
    const value = formatRequestLogReasoning(reasoning, locale)
    return value ? `[${value}]` : ''
  }

  function upstreamReasoningLabel(
    reasoning: RequestLogAttemptDto['reasoning'],
    routeMode: RequestLogRouteMode | null,
  ): string {
    return reasoningLabel(routeMode === 'converted' ? reasoning : (log?.reasoning ?? null))
  }

  function isFinalAttempt(attempt: RequestLogAttemptDto): boolean {
    return finalAttempt?.sequence === attempt.sequence
  }

  function showAttemptOperation(attempt: RequestLogAttemptDto): boolean {
    const requestOperation = log?.operation ?? null
    return (
      attempt.operation !== null &&
      (requestOperation === null || attempt.operation !== requestOperation)
    )
  }

  function dispatchStateLabel(attempt: RequestLogAttemptDto): string {
    if (attempt.response_started) return t('monitor.logs.dispatchState.response_started')
    if (attempt.dispatch_state === null) return t('monitor.logs.protocolConversion.notRecorded')
    return t(`monitor.logs.dispatchState.${attempt.dispatch_state}` as MessageId)
  }

  function accessKeyLabel(): string {
    const key = log?.access_key
    if (!key) return '—'
    // 控制面观察（access_key_id = 0）没有访问密钥，不能读成“引用的密钥已删除”。
    if (key.id === 0) return '—'
    if (key.deleted) return t('monitor.logs.deletedRef', { id: key.id })
    return key.name ? `${key.name} · #${key.id}` : `#${key.id}`
  }

  function finalGroupName(): string | null {
    const groupID = log?.group_id
    if (groupID === null || groupID === undefined) return null
    return (
      [...(log?.attempts ?? [])].reverse().find(({ group_id }) => group_id === groupID)
        ?.group_name ??
      groupNames?.[groupID] ??
      null
    )
  }

  function groupDeleted(groupID: number | null): boolean {
    return groupID !== null && groupsLoaded === true && groupNames?.[groupID] === undefined
  }

  function groupResolved(groupID: number | null): boolean {
    return groupID !== null && groupsLoaded === true && groupNames?.[groupID] !== undefined
  }

  // 订阅账号展示的就是完整邮箱，直接复制即可；密钥展示的是掩码，需取真值。
  const revealsCredential = channelDefinition(log?.channel_id)?.connection.type === 'api_key'

  // 密钥的 reveal 被订阅渠道拒绝，故仅密钥类走这条取值路径。
  async function resolveCredentialCopyValue(): Promise<string> {
    const record = log
    if (!record || record.group_id === null || record.credential_id === null) return ''
    const controller = new AbortController()
    copyControllersRef.current.add(controller)
    try {
      const result = await revealCredential(
        apiClient,
        record.group_id,
        record.credential_id,
        controller.signal,
      )
      const values = Object.values(result.credential)
      return values.length === 1 ? values[0] : JSON.stringify(result.credential)
    } finally {
      copyControllersRef.current.delete(controller)
    }
  }

  function finalChannel(): ChannelDto | null {
    return channelDefinition(log?.channel_id)
  }

  // 请求级已汇总的原因不在尝试首屏重复，同样的摘要也只在最早一个尝试上展示。
  const visibleAttemptReasonSequences = log
    ? requestLogAttemptReasonSequences(log)
    : new Set<number>()

  function isAttemptReasonVisible(attempt: RequestLogAttemptDto): boolean {
    return visibleAttemptReasonSequences.has(attempt.sequence)
  }

  // attempt 首屏原因与请求级关键原因共享同一身份规则：摘要优先，摘要为空时降级为错误码。
  function attemptReasonText(attempt: RequestLogAttemptDto): string {
    return isAttemptReasonVisible(attempt) ? requestLogAttemptReasonText(attempt) : ''
  }

  function attemptReasonLabel(attempt: RequestLogAttemptDto): string {
    if (!isAttemptReasonVisible(attempt)) return ''
    return attempt.error_summary.trim() !== ''
      ? t('monitor.logs.drawer.errorSummary')
      : t('monitor.logs.drawer.errorCode')
  }

  // 首屏已用降级后的错误码表达该原因时，折叠详情不再重复同一个码。
  function attemptErrorCodeNeedsDetails(attempt: RequestLogAttemptDto): boolean {
    const code = attempt.error_code.trim()
    return code !== '' && code !== attemptReasonText(attempt).trim()
  }

  function isAttemptErrorMessageExpanded(sequence: number): boolean {
    return expandedAttemptErrorMessages.has(sequence)
  }

  function toggleAttemptErrorMessage(sequence: number): void {
    setExpandedAttemptErrorMessages((current) => {
      const next = new Set(current)
      if (next.has(sequence)) next.delete(sequence)
      else next.add(sequence)
      return next
    })
  }

  return (
    // isOpen toggles rather than unmounting: Dialog's close path runs the
    // invoker focus restore only when the element transitions to closed.
    <DetailPanel
      isOpen={open}
      onOpenChange={onOpenChange}
      title={t('monitor.logs.drawer.title')}
      subtitle={drawerDescription}
    >
      {(open && detailQuery.isPending) || initialLoading ? (
        <section
          data-testid="skeleton-surface"
          data-variant="detail"
          {...stylex.props(styles.skeletonSurface, !initialLoading && styles.skeletonConcealed)}
          role={initialLoading ? 'status' : undefined}
          aria-label={initialLoading ? t('monitor.logs.drawer.loading') : undefined}
          aria-busy={initialLoading ? true : undefined}
          aria-hidden={initialLoading ? undefined : true}
        >
          <span {...stylex.props(styles.srOnly)}>{t('monitor.logs.drawer.loading')}</span>
          <Skeleton width="46%" height={32} radius={2} aria-hidden="true" />
          <Skeleton height={72} radius={2} aria-hidden="true" />
          <Skeleton height={50} radius={2} aria-hidden="true" />
          <Skeleton height={132} radius={2} aria-hidden="true" />
          <Skeleton height={132} radius={2} aria-hidden="true" />
          <Skeleton height={132} radius={2} aria-hidden="true" />
        </section>
      ) : detailQuery.isError || log === undefined ? (
        <div role="alert" {...stylex.props(styles.queryFeedback, styles.queryFeedbackError)}>
          <TriangleAlert size={18} aria-hidden="true" />
          <span {...stylex.props(styles.queryFeedbackText)}>
            {t('monitor.logs.drawer.loadFailed')}
          </span>
          <button
            type="button"
            {...stylex.props(styles.queryFeedbackRetry)}
            onClick={() => void detailQuery.refetch()}
          >
            <RefreshCw size={15} aria-hidden="true" />
            {t('common.retry')}
          </button>
        </div>
      ) : (
        <div data-testid="log-detail" {...stylex.props(styles.detail)}>
          <header {...stylex.props(styles.summary)}>
            <Badge
              variant={badgeVariants[statusTone(log.status)]}
              icon={(() => {
                const StatusIcon = badgeIcons[statusTone(log.status)]
                return <StatusIcon size={12} aria-hidden="true" />
              })()}
              label={`${t(`monitor.logs.status.${log.status}` as MessageId)}${
                log.status !== 'success' && log.status_code ? ` · ${log.status_code}` : ''
              }`}
              xstyle={styles.badge}
            />
            <span {...stylex.props(styles.time)}>
              <LogInstant instant={log.completed_at_ms} />
            </span>
            <span {...stylex.props(styles.requestId)}>
              <OverflowTip content={log.request_id} xstyle={styles.requestIdCode}>
                {log.request_id}
              </OverflowTip>
              <CopyButton
                value={log.request_id}
                label={t('monitor.logs.drawer.copyRequestId')}
                successLabel={t('common.copied')}
                failureLabel={t('common.copyFailed')}
                xstyle={styles.requestIdCopy}
              />
            </span>
          </header>

          <section {...stylex.props(styles.section)} data-testid="log-detail__section">
            <h3 {...stylex.props(styles.sectionTitle)}>{t('monitor.logs.drawer.summary')}</h3>
            <dl {...stylex.props(styles.grid)}>
              <div {...stylex.props(styles.gridCell)}>
                <dt {...stylex.props(styles.term)}>{t('monitor.logs.drawer.status')}</dt>
                <dd {...stylex.props(styles.description)}>
                  {t(`monitor.logs.status.${log.status}` as MessageId)}
                  {log.status_code ? ` · ${log.status_code}` : ''}
                </dd>
              </div>
              {!selfScoped && (
                <div {...stylex.props(styles.gridCell)}>
                  <dt {...stylex.props(styles.term)}>{t('monitor.logs.drawer.attemptCount')}</dt>
                  <dd {...stylex.props(styles.description)}>{log.attempt_count}</dd>
                </div>
              )}
              {log.stream && (
                <div {...stylex.props(styles.gridCell)}>
                  <dt {...stylex.props(styles.term)}>
                    {t('monitor.logs.drawer.requestFirstResponse')}
                  </dt>
                  <dd {...stylex.props(styles.description, timingTone(log.feedback_status))}>
                    {log.first_response_ms === null
                      ? '—'
                      : formatLogDuration(log.first_response_ms)}
                  </dd>
                </div>
              )}
              <div {...stylex.props(styles.gridCell)}>
                <dt {...stylex.props(styles.term)}>{t('monitor.logs.drawer.requestDuration')}</dt>
                <dd {...stylex.props(styles.description, timingTone(log.feedback_status))}>
                  {formatLogDuration(log.duration_ms)}
                </dd>
              </div>
            </dl>
            {keyReasonText !== '' && (
              <div {...stylex.props(styles.errorMessage)}>
                <p {...stylex.props(styles.errorMessageLabel)}>{keyReasonLabel}</p>
                <p
                  {...stylex.props(
                    styles.errorMessageContent,
                    !errorMessageExpanded &&
                      errorMessageNeedsDisclosure(keyReasonText) &&
                      styles.errorMessageCollapsed,
                  )}
                >
                  {keyReasonText}
                </p>
                {errorMessageNeedsDisclosure(keyReasonText) && (
                  <button
                    type="button"
                    {...stylex.props(styles.linkToggle, styles.errorMessageToggle)}
                    aria-expanded={errorMessageExpanded}
                    onClick={() => setErrorMessageExpanded(!errorMessageExpanded)}
                  >
                    {errorMessageExpanded
                      ? t('monitor.logs.drawer.collapseErrorMessage')
                      : t('monitor.logs.drawer.expandErrorMessage')}
                  </button>
                )}
                {keyReasonCode !== '' && (
                  <p {...stylex.props(styles.errorMessageCode)}>
                    <span>{t('monitor.logs.drawer.errorCode')}</span>
                    <code {...stylex.props(styles.errorMessageCodeValue)}>{keyReasonCode}</code>
                  </p>
                )}
                {requestErrorCode !== '' && (
                  <p {...stylex.props(styles.errorMessageCode)}>
                    <span>{t('monitor.logs.drawer.requestErrorCode')}</span>
                    <code {...stylex.props(styles.errorMessageCodeValue)}>{requestErrorCode}</code>
                  </p>
                )}
              </div>
            )}
          </section>

          <section {...stylex.props(styles.section)} data-testid="log-detail__section">
            <h3 {...stylex.props(styles.sectionTitle)}>{t('monitor.logs.drawer.request')}</h3>
            <dl {...stylex.props(styles.grid)}>
              {!selfScoped && (
                <div {...stylex.props(styles.gridCell)}>
                  <dt {...stylex.props(styles.term)}>{t('monitor.logs.drawer.accessKey')}</dt>
                  <dd {...stylex.props(styles.description)}>{accessKeyLabel()}</dd>
                </div>
              )}
              <div {...stylex.props(styles.gridCell)}>
                <dt {...stylex.props(styles.term)}>{t('monitor.logs.drawer.protocol')}</dt>
                <dd {...stylex.props(styles.description)}>
                  <code>{log.protocol}</code>
                </dd>
              </div>
              <div {...stylex.props(styles.gridCell)}>
                <dt {...stylex.props(styles.term)}>{t('monitor.logs.drawer.operation')}</dt>
                <dd {...stylex.props(styles.description)}>{operationLabel(log.operation)}</dd>
              </div>
              <div {...stylex.props(styles.gridCell)}>
                <dt {...stylex.props(styles.term)}>{t('monitor.logs.drawer.clientModel')}</dt>
                <dd {...stylex.props(styles.description)}>
                  <code>{log.client_model ?? '—'}</code>
                  {reasoningLabel(log.reasoning) !== '' && (
                    <small {...stylex.props(styles.reasoning)}>
                      {reasoningLabel(log.reasoning)}
                    </small>
                  )}
                </dd>
              </div>
            </dl>
          </section>

          {!selfScoped && (
            <section {...stylex.props(styles.section)} data-testid="log-detail__section">
              <h3 {...stylex.props(styles.sectionTitle)}>
                {t('monitor.logs.drawer.finalExecution')}
              </h3>
              <dl {...stylex.props(styles.grid)}>
                <div {...stylex.props(styles.gridCell, styles.wide)}>
                  <dt {...stylex.props(styles.term)}>{t('monitor.logs.drawer.routeIdentity')}</dt>
                  <dd {...stylex.props(styles.description, styles.route)}>
                    <LogRouteIdentity
                      groupId={log.group_id}
                      groupName={finalGroupName()}
                      groupDeleted={groupDeleted(log.group_id)}
                      groupResolved={groupResolved(log.group_id)}
                      providerUrl={
                        log.group_id === null ? null : (providerUrls?.[log.group_id] ?? null)
                      }
                      channelId={log.channel_id}
                      channel={finalChannel()}
                      credentialId={log.credential_id}
                      credentialName={log.credential_name}
                      credentialDeleted={log.credential_id !== null && log.credential_name === ''}
                      appearance="plain"
                    />
                    {open && log.credential_name !== '' && (
                      <CredentialCopyChip
                        key={`${requestId}:${log.group_id}:${log.credential_id}`}
                        value={log.credential_name}
                        label={t('monitor.logs.drawer.copyCredential')}
                        successLabel={t('common.copied')}
                        failureLabel={t('common.copyFailed')}
                        resolveValue={revealsCredential ? resolveCredentialCopyValue : undefined}
                        xstyle={styles.routeCopyChip}
                      />
                    )}
                  </dd>
                </div>
                <div {...stylex.props(styles.gridCell)}>
                  <dt {...stylex.props(styles.term)}>
                    {t('monitor.logs.drawer.upstreamProtocol')}
                  </dt>
                  <dd {...stylex.props(styles.description)}>
                    {upstreamProtocolLabel(log.upstream_protocol)}
                  </dd>
                </div>
                <div {...stylex.props(styles.gridCell)}>
                  <dt {...stylex.props(styles.term)}>{t('monitor.logs.drawer.upstreamModel')}</dt>
                  <dd {...stylex.props(styles.description)}>
                    <code>{log.upstream_model ?? '—'}</code>
                    {upstreamReasoningLabel(
                      finalAttempt?.reasoning ?? null,
                      finalAttempt?.route_mode ?? null,
                    ) !== '' && (
                      <small {...stylex.props(styles.reasoning)}>
                        {upstreamReasoningLabel(
                          finalAttempt?.reasoning ?? null,
                          finalAttempt?.route_mode ?? null,
                        )}
                      </small>
                    )}
                  </dd>
                </div>
              </dl>
              {(log.model_consistency === 'unknown' || log.model_consistency === 'mismatch') && (
                <div
                  {...stylex.props(
                    styles.modelObservation,
                    log.model_consistency === 'mismatch' && styles.modelObservationMismatch,
                  )}
                >
                  <div {...stylex.props(styles.modelObservationHeading)}>
                    <strong>{t('monitor.logs.drawer.modelObservation')}</strong>
                    <Badge
                      variant={
                        badgeVariants[log.model_consistency === 'mismatch' ? 'warning' : 'neutral']
                      }
                      icon={(() => {
                        const ObservationIcon =
                          badgeIcons[log.model_consistency === 'mismatch' ? 'warning' : 'neutral']
                        return <ObservationIcon size={12} aria-hidden="true" />
                      })()}
                      label={t(
                        log.model_consistency === 'mismatch'
                          ? 'monitor.logs.modelConsistency.mismatchLabel'
                          : 'monitor.logs.modelConsistency.unknownLabel',
                      )}
                      xstyle={styles.badge}
                    />
                  </div>
                  <dl {...stylex.props(styles.grid)}>
                    <div {...stylex.props(styles.gridCell)}>
                      <dt {...stylex.props(styles.term)}>
                        {t('monitor.logs.drawer.requestedModel')}
                      </dt>
                      <dd {...stylex.props(styles.description)}>
                        <code>{log.upstream_model ?? '—'}</code>
                      </dd>
                    </div>
                    <div {...stylex.props(styles.gridCell)}>
                      <dt {...stylex.props(styles.term)}>
                        {t('monitor.logs.drawer.reportedModel')}
                      </dt>
                      <dd {...stylex.props(styles.description)}>
                        <code>
                          {log.upstream_reported_model ??
                            t('monitor.logs.modelConsistency.notObserved')}
                        </code>
                      </dd>
                    </div>
                  </dl>
                </div>
              )}
            </section>
          )}

          <section {...stylex.props(styles.section)} data-testid="log-detail__section">
            <details
              data-testid="log-usage-chain"
              data-details-chevron-scope=""
              {...stylex.props(styles.attemptChain)}
            >
              <summary {...stylex.props(styles.chainSummary)}>
                <ChevronRight
                  size={15}
                  aria-hidden="true"
                  data-details-chevron=""
                  {...stylex.props(styles.chevron)}
                />
                <span>{t('monitor.logs.drawer.usage.title')}</span>
              </summary>
              <dl {...stylex.props(styles.grid, styles.usageChainGrid)}>
                <div {...stylex.props(styles.gridCell)}>
                  <dt {...stylex.props(styles.term)}>
                    {t('monitor.logs.drawer.usage.usageStateLabel')}
                  </dt>
                  <dd {...stylex.props(styles.description)}>{usageStateLabel}</dd>
                </div>
                <div {...stylex.props(styles.gridCell)}>
                  <dt {...stylex.props(styles.term)}>
                    {t('monitor.logs.drawer.usage.costStateLabel')}
                  </dt>
                  <dd {...stylex.props(styles.description)}>{costStateLabel}</dd>
                </div>
                {usageDisplayState === 'reported' && (
                  <div {...stylex.props(styles.gridCell)}>
                    <dt {...stylex.props(styles.term)}>{t('monitor.logs.tokens.input')}</dt>
                    <dd {...stylex.props(styles.description)}>
                      {formatLogTokenCount(log.input_tokens, locale)}
                    </dd>
                  </div>
                )}
                {usageDisplayState === 'reported' && (
                  <div {...stylex.props(styles.gridCell)}>
                    <dt {...stylex.props(styles.term)}>{t('monitor.logs.tokens.output')}</dt>
                    <dd {...stylex.props(styles.description)}>
                      {formatLogTokenCount(log.output_tokens, locale)}
                    </dd>
                  </div>
                )}
                {cacheRows.map((row) => (
                  <div key={row.label} {...stylex.props(styles.gridCell)}>
                    <dt {...stylex.props(styles.term)}>{row.label}</dt>
                    <dd {...stylex.props(styles.description)}>
                      {formatLogTokenCount(row.value, locale)}
                    </dd>
                  </div>
                ))}
                {cacheRows.length > 0 && (
                  <div {...stylex.props(styles.gridCell)}>
                    <dt {...stylex.props(styles.term)}>{t('monitor.logs.tokens.cacheHitRate')}</dt>
                    <dd {...stylex.props(styles.description)}>{cacheRateLabel}</dd>
                  </div>
                )}
                {costDisplayState !== 'unpriced' && (
                  <div {...stylex.props(styles.gridCell)}>
                    <dt {...stylex.props(styles.term)}>
                      {t('monitor.logs.drawer.usage.estimatedCost')}
                    </dt>
                    <dd {...stylex.props(styles.description, styles.cost)}>
                      <span>{costAmountLabel}</span>
                      <PricingModeIndicator
                        mode={log.pricing_mode}
                        contextThresholdTokens={log.context_threshold_tokens}
                        xstyle={[styles.pricingModeIndicator]}
                      />
                    </dd>
                  </div>
                )}
                {!selfScoped && receipt !== undefined && (
                  <div {...stylex.props(styles.gridCell)}>
                    <dt {...stylex.props(styles.term)}>{t('monitor.logs.receipt.identity')}</dt>
                    <dd {...stylex.props(styles.description)}>
                      <code>{pricingIdentity}</code>
                    </dd>
                  </div>
                )}

                {!selfScoped &&
                  costDisplayState !== 'unpriced' &&
                  receipt !== undefined &&
                  usageDisplayState === 'reported' && (
                    <div {...stylex.props(styles.gridCell, styles.wide)}>
                      <dt {...stylex.props(styles.term)}>{t('monitor.logs.receipt.formula')}</dt>
                      <dd {...stylex.props(styles.description, styles.formula)}>
                        <span>
                          {t('monitor.logs.receipt.input')} = {formula.input}
                        </span>
                        <span>
                          {t('monitor.logs.receipt.output')} = {formula.output}
                        </span>
                        <>
                          <span>
                            {t('monitor.logs.receipt.total')} ={' '}
                            {formatExactNanoUSD(receipt.total_nano_usd, locale)}
                          </span>
                          <small>{t('monitor.logs.receipt.rounding')}</small>
                        </>
                      </dd>
                    </div>
                  )}
              </dl>
            </details>
          </section>

          {!selfScoped && (
            <section
              {...stylex.props(styles.section, styles.attemptSection)}
              data-testid="log-detail__section"
            >
              {log.attempts.length > 0 ? (
                <details open data-details-chevron-scope="" {...stylex.props(styles.attemptChain)}>
                  <summary {...stylex.props(styles.chainSummary)}>
                    <ChevronRight
                      size={15}
                      aria-hidden="true"
                      data-details-chevron=""
                      {...stylex.props(styles.chevron)}
                    />
                    <span>{t('monitor.logs.drawer.attempts')}</span>
                    <span {...stylex.props(styles.chainCount)}>{log.attempts.length}</span>
                  </summary>
                  {log.attempts.map((attempt, index) => {
                    const tone = attemptTone(attempt)
                    const AttemptIcon = badgeIcons[tone]
                    const reasonText = attemptReasonText(attempt)
                    const reasonExpanded = isAttemptErrorMessageExpanded(attempt.sequence)
                    return (
                      <article
                        key={attempt.sequence}
                        data-testid="log-attempt"
                        {...stylex.props(
                          styles.attempt,
                          index === 0 && styles.attemptFirst,
                          index > 0 && styles.attemptBordered,
                        )}
                      >
                        <header {...stylex.props(styles.attemptHeader)}>
                          <span>
                            {t('monitor.logs.drawer.attempt', { sequence: attempt.sequence })}
                          </span>
                          <Badge
                            variant={badgeVariants[tone]}
                            icon={<AttemptIcon size={12} aria-hidden="true" />}
                            label={`${t(
                              `monitor.logs.failureCategory.${attempt.failure_category}` as MessageId,
                            )}${attempt.status_code ? ` · ${attempt.status_code}` : ''}`}
                            xstyle={styles.badge}
                          />
                        </header>
                        <div {...stylex.props(styles.attemptPrimary)}>
                          <LogRouteIdentity
                            groupId={attempt.group_id}
                            groupName={attempt.group_name}
                            groupDeleted={groupDeleted(attempt.group_id)}
                            groupResolved={groupResolved(attempt.group_id)}
                            providerUrl={providerUrls?.[attempt.group_id] ?? null}
                            channelId={attempt.channel_id}
                            channel={channelDefinition(attempt.channel_id)}
                            credentialId={attempt.credential_id}
                            credentialName={attempt.credential_name}
                            credentialDeleted={
                              attempt.credential_id !== null && attempt.credential_name === ''
                            }
                            appearance="plain"
                          />
                          <span {...stylex.props(styles.attemptAction)}>
                            <span {...stylex.props(styles.attemptActionLabel)}>
                              {t('monitor.logs.drawer.gatewayAction')}
                            </span>
                            {t(`monitor.logs.action.${attempt.action}` as MessageId)}
                          </span>
                        </div>
                        {reasonText !== '' && (
                          <div {...stylex.props(styles.attemptReason)}>
                            <p {...stylex.props(styles.errorMessageLabel)}>
                              {attemptReasonLabel(attempt)}
                            </p>
                            <p
                              {...stylex.props(
                                styles.errorMessageContent,
                                !reasonExpanded &&
                                  errorMessageNeedsDisclosure(reasonText) &&
                                  styles.errorMessageCollapsed,
                                !reasonExpanded &&
                                  errorMessageNeedsDisclosure(reasonText) &&
                                  styles.errorMessageCollapsedAttempt,
                              )}
                            >
                              {reasonText}
                            </p>
                            {errorMessageNeedsDisclosure(reasonText) && (
                              <button
                                type="button"
                                {...stylex.props(styles.linkToggle, styles.errorMessageToggle)}
                                aria-expanded={reasonExpanded}
                                onClick={() => toggleAttemptErrorMessage(attempt.sequence)}
                              >
                                {reasonExpanded
                                  ? t('monitor.logs.drawer.collapseErrorMessage')
                                  : t('monitor.logs.drawer.expandErrorMessage')}
                              </button>
                            )}
                          </div>
                        )}
                        <details
                          open
                          data-details-chevron-scope=""
                          {...stylex.props(styles.attemptDetails)}
                        >
                          <summary {...stylex.props(styles.attemptDetailsSummary)}>
                            <ChevronRight
                              size={13}
                              aria-hidden="true"
                              data-details-chevron=""
                              {...stylex.props(styles.chevron)}
                            />
                            <span>{t('monitor.logs.drawer.attemptDetails')}</span>
                          </summary>
                          <dl {...stylex.props(styles.grid, styles.attemptDetailsGrid)}>
                            {!isFinalAttempt(attempt) && (
                              <>
                                <div {...stylex.props(styles.gridCell)}>
                                  <dt {...stylex.props(styles.term)}>
                                    {t('monitor.logs.drawer.upstreamProtocol')}
                                  </dt>
                                  <dd {...stylex.props(styles.description)}>
                                    {upstreamProtocolLabel(attempt.upstream_protocol)}
                                  </dd>
                                </div>
                                <div {...stylex.props(styles.gridCell)}>
                                  <dt {...stylex.props(styles.term)}>
                                    {t('monitor.logs.drawer.upstreamModel')}
                                  </dt>
                                  <dd {...stylex.props(styles.description)}>
                                    <code>{attempt.upstream_model ?? '—'}</code>
                                  </dd>
                                </div>
                              </>
                            )}
                            {showAttemptOperation(attempt) && (
                              <div {...stylex.props(styles.gridCell)}>
                                <dt {...stylex.props(styles.term)}>
                                  {t('monitor.logs.drawer.operation')}
                                </dt>
                                <dd {...stylex.props(styles.description)}>
                                  {operationLabel(attempt.operation)}
                                </dd>
                              </div>
                            )}
                            <div {...stylex.props(styles.gridCell)}>
                              <dt {...stylex.props(styles.term)}>
                                {t('monitor.logs.drawer.dispatchState')}
                              </dt>
                              <dd {...stylex.props(styles.description)}>
                                {dispatchStateLabel(attempt)}
                              </dd>
                            </div>
                            {log.stream && (
                              <div {...stylex.props(styles.gridCell)}>
                                <dt {...stylex.props(styles.term)}>
                                  {t('monitor.logs.drawer.clientStreamState')}
                                </dt>
                                <dd {...stylex.props(styles.description)}>
                                  {attempt.committed
                                    ? t('monitor.logs.drawer.clientStreamStarted')
                                    : t('monitor.logs.drawer.clientStreamNotStarted')}
                                </dd>
                              </div>
                            )}
                            <div {...stylex.props(styles.gridCell)}>
                              <dt {...stylex.props(styles.term)}>
                                {t('monitor.logs.drawer.attemptDuration')}
                              </dt>
                              <dd
                                {...stylex.props(
                                  styles.description,
                                  timingTone(attempt.feedback_status),
                                )}
                              >
                                {formatLogDuration(attempt.duration_ms)}
                              </dd>
                            </div>
                            {attempt.failure_origin && (
                              <div {...stylex.props(styles.gridCell)}>
                                <dt {...stylex.props(styles.term)}>
                                  {t('monitor.logs.drawer.failureOrigin')}
                                </dt>
                                <dd {...stylex.props(styles.description)}>
                                  {t(
                                    `monitor.logs.failureOrigin.${attempt.failure_origin}` as MessageId,
                                  )}
                                </dd>
                              </div>
                            )}
                            {attempt.failure_scope && (
                              <div {...stylex.props(styles.gridCell)}>
                                <dt {...stylex.props(styles.term)}>
                                  {t('monitor.logs.drawer.failureScope')}
                                </dt>
                                <dd {...stylex.props(styles.description)}>
                                  {t(
                                    `monitor.logs.failureScope.${attempt.failure_scope}` as MessageId,
                                  )}
                                </dd>
                              </div>
                            )}
                            {attempt.retry_directive && (
                              <div {...stylex.props(styles.gridCell)}>
                                <dt {...stylex.props(styles.term)}>
                                  {t('monitor.logs.drawer.retryDirective')}
                                </dt>
                                <dd {...stylex.props(styles.description)}>
                                  {t(
                                    `monitor.logs.retryDirective.${attempt.retry_directive}` as MessageId,
                                  )}
                                </dd>
                              </div>
                            )}
                            {attempt.effect && (
                              <div {...stylex.props(styles.gridCell)}>
                                <dt {...stylex.props(styles.term)}>
                                  {t('monitor.logs.drawer.effect')}
                                </dt>
                                <dd {...stylex.props(styles.description)}>
                                  {t(`monitor.logs.effect.${attempt.effect}` as MessageId)}
                                </dd>
                              </div>
                            )}
                            {attempt.rule_id && (
                              <div {...stylex.props(styles.gridCell)}>
                                <dt {...stylex.props(styles.term)}>
                                  {t('monitor.logs.drawer.ruleId')}
                                </dt>
                                <dd {...stylex.props(styles.description)}>
                                  <code>{attempt.rule_id}</code>
                                </dd>
                              </div>
                            )}
                            <div {...stylex.props(styles.gridCell)}>
                              <dt {...stylex.props(styles.term)}>
                                {t('monitor.logs.drawer.subsequentAttempt')}
                              </dt>
                              <dd {...stylex.props(styles.description)}>
                                {attempt.will_retry
                                  ? t('monitor.logs.drawer.subsequentAttemptOccurred')
                                  : t('monitor.logs.drawer.noSubsequentAttempt')}
                              </dd>
                            </div>
                          </dl>
                          {attemptErrorCodeNeedsDetails(attempt) && (
                            <div {...stylex.props(styles.errorMessage, styles.errorMessageAttempt)}>
                              <p {...stylex.props(styles.errorMessageCode)}>
                                <span>{t('monitor.logs.drawer.errorCode')}</span>
                                <code {...stylex.props(styles.errorMessageCodeValue)}>
                                  {attempt.error_code}
                                </code>
                              </p>
                            </div>
                          )}
                        </details>
                      </article>
                    )
                  })}
                </details>
              ) : (
                <p {...stylex.props(styles.empty)}>{t('monitor.logs.drawer.noAttempts')}</p>
              )}
            </section>
          )}
        </div>
      )}
    </DetailPanel>
  )
}
