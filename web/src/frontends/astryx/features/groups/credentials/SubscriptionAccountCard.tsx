import {
  Badge,
  Button,
  IconButton,
  Popover,
  Skeleton,
  Tooltip,
} from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import {
  Check,
  Download,
  Ellipsis,
  Gauge,
  KeyRound,
  LoaderCircle,
  RefreshCw,
  RotateCcw,
  Trash2,
} from 'lucide-react'
import { useIntl } from 'react-intl'
import { useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'

import type {
  CredentialItemDto,
  CredentialQuotaLabelKey,
  CredentialQuotaWindowDto,
} from '@shared/control/types'
import type { ChannelCapabilitiesDto } from '@shared/control/resources/channels'
import { ChannelIcon } from '../../../components/ChannelIcon'
import { RelativeInstant } from '../../../components/RelativeInstant'
import { formatEstimatedCost, formatLocalInstant, formatTokens } from '@shared/lib/format'
import { quotaProgressTone } from '@shared/lib/quota-progress'
import type { MessageId } from '@shared/i18n/message-ids'
import { useT } from '../../../app/i18n'

import { presentCredentialFailureCategory } from '@shared/domain/groups/credentials/credential-failure-presenter'

const quotaSubjectKeys: Readonly<Record<string, CredentialQuotaLabelKey>> = {
  session: 'session',
  weekly: 'weekly',
  'extra usage': 'extra_usage',
  'included usage': 'included_usage',
  'pay as you go': 'pay_as_you_go',
  'oauth apps': 'oauth_apps',
}

const authErrorKeys: Readonly<Record<string, MessageId>> = {
  refresh_rejected: 'group.credentials.subscription.authError.refreshRejected',
  refresh_identity_changed: 'group.credentials.subscription.authError.identityChanged',
  refresh_outcome_unknown: 'group.credentials.subscription.authError.outcomeUnknown',
  refresh_persist_failed: 'group.credentials.subscription.authError.persistFailed',
  refresh_commit_failed: 'group.credentials.subscription.authError.persistFailed',
  refresh_registry_mismatch: 'group.credentials.subscription.authError.runtimeMismatch',
  refresh_start_failed: 'group.credentials.subscription.authError.refreshStartFailed',
  refresh_state_commit_failed: 'group.credentials.subscription.authError.persistFailed',
}

const observationErrorKeys: Readonly<Record<string, MessageId>> = {
  observation_access_denied: 'group.credentials.subscription.observationError.accessDenied',
  observation_authorization_failed:
    'group.credentials.subscription.observationError.authorizationFailed',
  observation_partial: 'group.credentials.subscription.observationError.partial',
  observation_upstream_failed: 'group.credentials.subscription.observationError.upstreamFailed',
  observation_payload_invalid: 'group.credentials.subscription.observationError.payloadInvalid',
}

type UnifiedStatus =
  | 'available'
  | 'quota_exhausted'
  | 'cooldown'
  | 'blacklisted'
  | 'refreshing'
  | 'needs_reauth'
  | 'outcome_unknown'

type ResetCreditDotTone = 'default' | 'warning' | 'danger'

export function SubscriptionAccountCard({
  item,
  selected,
  busy,
  refreshingObservation,
  observationError,
  detailBusy,
  detailLoaded,
  detailError,
  channelIcon,
  channelMark,
  capabilities,
  onSelectedChange,
  onRestore,
  onRefresh,
  onLoadDetails,
  onReset,
  onDownload,
  onRefreshCredential,
  onRemove,
}: {
  item: CredentialItemDto
  selected: boolean
  busy: boolean
  refreshingObservation: boolean
  observationError: string
  detailBusy: boolean
  detailLoaded: boolean
  detailError: string
  channelIcon?: string
  channelMark?: string
  capabilities: ChannelCapabilitiesDto
  onSelectedChange(selected: boolean): void
  onRestore(item: CredentialItemDto): void
  onRefresh(item: CredentialItemDto): void
  onLoadDetails(item: CredentialItemDto): void
  onReset(item: CredentialItemDto): void
  onDownload(item: CredentialItemDto): void
  onRefreshCredential(item: CredentialItemDto): void
  onRemove(item: CredentialItemDto): void
}) {
  const t = useT()
  const intl = useIntl()
  const locale = intl.locale
  const n = (value: number): string => intl.formatNumber(value)
  const [menuOpen, setMenuOpen] = useState(false)
  const [detailsExpanded, setDetailsExpanded] = useState(false)
  const [nowMs, setNowMs] = useState(() => Date.now())
  const itemRef = useRef(item)
  const onLoadDetailsRef = useRef(onLoadDetails)

  useEffect(() => {
    itemRef.current = item
    onLoadDetailsRef.current = onLoadDetails
  })

  useEffect(() => {
    const clockTimer = window.setInterval(() => {
      setNowMs(Date.now())
    }, 30_000)
    return () => window.clearInterval(clockTimer)
  }, [])

  useEffect(() => {
    if (detailsExpanded && !detailLoaded && !detailBusy && !detailError && !busy) {
      onLoadDetailsRef.current(itemRef.current)
    }
  }, [detailsExpanded, detailLoaded, detailBusy, detailError, busy])

  const observation = item.observation
  const supportsQuotaObservation = capabilities.quota_observation
  const needsInitialQuotaSync =
    supportsQuotaObservation &&
    (observation === undefined ||
      (observation.state === 'unavailable' &&
        observation.observed_at_ms === null &&
        observation.last_attempt_at_ms === null))
  const supportsResetCredit = capabilities.credential_actions.includes('reset_credit')
  const snapshot = observation?.snapshot
  function isAccountWideQuotaWindow(window: CredentialQuotaWindowDto): boolean {
    return window.scope === 'account'
  }

  function quotaWindowDuration(window: CredentialQuotaWindowDto): number {
    const seconds = window.window_seconds
    return seconds !== undefined && Number.isFinite(seconds) && seconds > 0
      ? seconds
      : Number.MAX_SAFE_INTEGER
  }

  function quotaWindowGroupKey(window: CredentialQuotaWindowDto): string {
    const modelIDs = (window.model_ids ?? [])
      .map((model) => model.trim())
      .filter((model) => model !== '')
      .sort()
    if (modelIDs.length > 0) return `models:${modelIDs.join('\u0000')}`

    const period = quotaWindowPeriodLabel(window.window_seconds)
    const labelParts = window.label
      .split('·')
      .map((part) => part.trim())
      .filter((part) => part !== '')
    const finalPart = labelParts.at(-1)
    if (
      period &&
      labelParts.length > 1 &&
      finalPart !== undefined &&
      normalizedQuotaLabelPart(finalPart) === normalizedQuotaLabelPart(period)
    ) {
      const subject = normalizedQuotaLabelPart(labelParts.slice(0, -1).join('\u0000'))
      if (subject) return `subject:${subject}`
    }
    return `scope:${normalizedQuotaLabelPart(window.scope) || window.id}`
  }

  // 呈现层统一排序：账号全局窗口优先；其余按同一模型/专属窗口成组，组内按时长升序。
  const quotaWindows = (() => {
    const windows = snapshot?.quota_windows ?? []
    const groupOrder = new Map<string, number>()
    for (const [index, window] of windows.entries()) {
      const key = quotaWindowGroupKey(window)
      if (!groupOrder.has(key)) groupOrder.set(key, index)
    }
    return [...windows].sort((left, right) => {
      const scopeDifference =
        Number(!isAccountWideQuotaWindow(left)) - Number(!isAccountWideQuotaWindow(right))
      if (scopeDifference !== 0) return scopeDifference
      const groupDifference =
        (groupOrder.get(quotaWindowGroupKey(left)) ?? 0) -
        (groupOrder.get(quotaWindowGroupKey(right)) ?? 0)
      if (groupDifference !== 0) return groupDifference
      return quotaWindowDuration(left) - quotaWindowDuration(right)
    })
  })()
  const accountQuotaWindows = quotaWindows.filter((window) => window.scope === 'account')
  const usageQuotaWindows = accountQuotaWindows.filter(
    (window) => quotaWindowPeriod(window) !== undefined,
  )
  const hasUsageQuotaWindows = usageQuotaWindows.length > 0
  const windowSkeletonHeight = `${24 + usageQuotaWindows.length * 32}px`
  const refreshSkeletonRows = Math.min(4, quotaWindows.length)
  const constrainedModels = Array.from(
    new Set(quotaWindows.flatMap((window) => window.model_ids ?? [])),
  )

  const accountName = item.account.email ?? item.mask
  const planLabel = snapshot?.plan_summary.name?.trim() ?? ''
  const planLevel = snapshot?.plan_summary.level ?? 'unknown'
  const credentialExpiryTooltip = (() => {
    const expiresAtMS = item.account.expires_at_ms
    if (expiresAtMS === undefined) return undefined
    const exact = formatLocalInstant(expiresAtMS, locale)
    return item.auth_state === 'ready'
      ? `${exact}\n${t('group.credentials.subscription.autoRenews')}`
      : exact
  })()
  const syncTimeTooltip = (() => {
    const observedAtMS = observation?.observed_at_ms
    if (observedAtMS === undefined || observedAtMS === null) return undefined
    return t('group.credentials.subscription.syncTime', {
      time: formatLocalInstant(observedAtMS, locale),
    })
  })()
  const syncExactTimeTooltip = (() => {
    const observedAtMS = observation?.observed_at_ms
    if (observedAtMS === undefined || observedAtMS === null) return undefined
    return formatLocalInstant(observedAtMS, locale)
  })()
  const quotaResetPrefix = t('group.credentials.subscription.quotaResetPrefix')
  const quotaResetSuffix = t('group.credentials.subscription.quotaResetSuffix')
  const resetCreditsAvailable = snapshot?.reset_credits_available ?? 0
  const resetCredits = snapshot?.reset_credits ?? []
  const availableResetCreditDetails = resetCredits.slice(0, resetCreditsAvailable)
  const hasResetCredits = supportsResetCredit && resetCreditsAvailable > 0
  const resetCreditDots = Array.from(
    { length: Math.min(resetCreditsAvailable, 5) },
    (_, index) => {
      const expiresAtMS = availableResetCreditDetails[index]?.expires_at_ms
      let tone: ResetCreditDotTone = 'default'
      if (expiresAtMS !== undefined) {
        const remainingMS = expiresAtMS - nowMs
        tone =
          remainingMS <= 24 * 60 * 60 * 1_000
            ? 'danger'
            : remainingMS <= 48 * 60 * 60 * 1_000
              ? 'warning'
              : 'default'
      }
      return { index, tone }
    },
  )
  const nearestResetCredit = availableResetCreditDetails.reduce<
    (typeof resetCredits)[number] | undefined
  >((nearest, credit) => {
    if (credit.expires_at_ms === undefined || credit.expires_at_ms <= nowMs) return nearest
    return nearest === undefined ||
      nearest.expires_at_ms === undefined ||
      credit.expires_at_ms < nearest.expires_at_ms
      ? credit
      : nearest
  }, undefined)
  function resetCreditExpiryLabel(credit: (typeof resetCredits)[number]): string {
    if (credit.expires_at_ms === undefined) {
      return t('group.credentials.subscription.resetCreditPermanent')
    }
    if (credit.expires_at_ms <= nowMs) {
      return t('group.credentials.subscription.resetCreditExpired')
    }
    return formatLocalInstant(credit.expires_at_ms, locale)
  }
  const resetCreditsTooltip = (() => {
    const lines = availableResetCreditDetails.length
      ? availableResetCreditDetails.map((credit, index) =>
          t('group.credentials.subscription.resetCreditsTooltipItem', {
            index: index + 1,
            expires: resetCreditExpiryLabel(credit),
          }),
        )
      : [t('group.credentials.subscription.resetCreditsTooltipNoDetails')]
    const knownCount = availableResetCreditDetails.length
    if (knownCount < resetCreditsAvailable) {
      lines.push(
        t('group.credentials.subscription.resetCreditsTooltipMore', {
          count: n(resetCreditsAvailable - knownCount),
        }),
      )
    }
    return [t('group.credentials.subscription.resetCreditsTooltipTitle'), ...lines].join('\n')
  })()
  const isProblem =
    item.effective_status === 'cooldown' || item.effective_status === 'blacklisted'

  interface QuotaWindowPeriod {
    startMS: number
    endMS: number
  }

  function quotaWindowPeriod(window: CredentialQuotaWindowDto): QuotaWindowPeriod | undefined {
    const endMS = window.reset_at_ms
    const seconds = window.window_seconds
    if (
      endMS === undefined ||
      seconds === undefined ||
      !Number.isSafeInteger(endMS) ||
      !Number.isSafeInteger(seconds) ||
      endMS <= 0 ||
      seconds <= 0
    ) {
      return undefined
    }
    const durationMS = seconds * 1_000
    if (!Number.isSafeInteger(durationMS) || durationMS > endMS) return undefined
    return { startMS: endMS - durationMS, endMS }
  }

  function quotaWindowNeedsRefresh(window: CredentialQuotaWindowDto): boolean {
    return window.reset_at_ms !== undefined && window.reset_at_ms <= nowMs
  }

  function quotaWindowPeriodLabel(seconds: number | undefined): string {
    if (seconds === undefined || !Number.isSafeInteger(seconds) || seconds <= 0) return ''
    const day = 24 * 60 * 60
    const hour = 60 * 60
    const minute = 60
    if (seconds % day === 0) return `${seconds / day}d`
    if (seconds % hour === 0) return `${seconds / hour}h`
    if (seconds % minute === 0) return `${seconds / minute}min`
    return `${seconds}s`
  }

  function normalizedQuotaLabelPart(value: string): string {
    return value.trim().toLowerCase().replaceAll('_', ' ').replaceAll('-', ' ')
  }

  function translatedQuotaLabel(labelKey: CredentialQuotaLabelKey, fallback: string): string {
    const key = `group.credentials.subscription.quotaLabels.${labelKey}` as MessageId
    const translated = t(key)
    return translated === key ? fallback : translated
  }

  function quotaWindowLabel(window: CredentialQuotaWindowDto): string {
    const period = quotaWindowPeriodLabel(window.window_seconds)
    if (period && window.scope === 'account') return period

    if (window.label_key) {
      const subject = translatedQuotaLabel(window.label_key, window.label)
      return window.label_key === 'oauth_apps' && period ? `${subject} · ${period}` : subject
    }

    const parts = window.label
      .split('·')
      .map((part) => part.trim())
      .filter(Boolean)
    if (parts.length === 0) return period

    const firstPart = normalizedQuotaLabelPart(parts[0] ?? '')
    if (period && (firstPart === 'session' || firstPart === 'weekly')) return period

    return parts
      .map((part) => {
        const labelKey = quotaSubjectKeys[normalizedQuotaLabelPart(part)]
        return labelKey ? translatedQuotaLabel(labelKey, part) : part
      })
      .join(' · ')
  }

  const unifiedStatus: UnifiedStatus = (() => {
    if (item.auth_state === 'refreshing') return 'refreshing'
    if (item.auth_state === 'reauthorization_required') return 'needs_reauth'
    if (item.auth_state === 'outcome_unknown') return 'outcome_unknown'
    if (item.effective_status === 'blacklisted') return 'blacklisted'
    if (item.effective_status === 'cooldown') return 'cooldown'
    if (
      supportsQuotaObservation &&
      quotaWindows.some((window) => window.scope === 'account' && window.state === 'exhausted')
    ) {
      return 'quota_exhausted'
    }
    return 'available'
  })()
  const statusTone = (
    {
      available: 'success',
      quota_exhausted: 'warning',
      cooldown: 'warning',
      blacklisted: 'danger',
      refreshing: 'neutral',
      needs_reauth: 'danger',
      outcome_unknown: 'danger',
    } as const
  )[unifiedStatus]
  const statusLabel = t(`group.credentials.subscription.status.${unifiedStatus}` as MessageId)
  const authIssue = (() => {
    if (item.auth_state !== 'reauthorization_required' && item.auth_state !== 'outcome_unknown') {
      return ''
    }
    const key = item.auth_error_code ? authErrorKeys[item.auth_error_code] : undefined
    return key ? t(key) : t(`group.credentials.subscription.auth.${item.auth_state}` as MessageId)
  })()
  // 额度同步需要可用的 access token；凭据刷新本身是异常账号的恢复入口，不受此限制。
  const observationRefreshBlocked = item.auth_state !== 'ready'
  const dailyUsage = item.daily_usage
  const dailyIncompleteHint =
    dailyUsage && !dailyUsage.data_complete
      ? t('group.credentials.subscription.dailyIncomplete')
      : undefined
  const failureLabel =
    item.recent_failure_count === 0
      ? t('group.credentials.none')
      : `${presentCredentialFailureCategory((key) => t(key as MessageId), item.last_failure_category)}${
          item.last_status_code === null ? '' : ` · ${item.last_status_code}`
        }`

  function observationErrorLabel(code: string | undefined): string {
    if (!code) return '—'
    const key = observationErrorKeys[code]
    return key ? t(key) : code
  }

  function estimateTitles(...keys: (MessageId | '')[]): string | undefined {
    const titles = keys.filter(Boolean).map((key) => t(key as MessageId))
    return titles.length === 0 ? undefined : titles.join(' · ')
  }

  function requestCountTitle(window: CredentialQuotaWindowDto): string | undefined {
    return window.observed_usage?.data_complete === false
      ? t('group.credentials.subscription.estimate.dataIncomplete')
      : undefined
  }

  function tokenCountTitle(window: CredentialQuotaWindowDto): string | undefined {
    const observed = window.observed_usage
    if (!observed) return undefined
    return estimateTitles(
      observed.data_complete ? '' : 'group.credentials.subscription.estimate.dataIncomplete',
      observed.usage_complete ? '' : 'group.credentials.subscription.estimate.usageIncomplete',
    )
  }

  function referenceCostTitle(window: CredentialQuotaWindowDto): string | undefined {
    const observed = window.observed_usage
    if (!observed) return undefined
    return estimateTitles(
      'group.credentials.subscription.estimate.priceMultiplierBasis',
      observed.data_complete ? '' : 'group.credentials.subscription.estimate.dataIncomplete',
      observed.pricing_complete
        ? ''
        : 'group.credentials.subscription.estimate.pricingIncomplete',
    )
  }

  function remainingPercent(window: CredentialQuotaWindowDto): number | undefined {
    if (window.utilization !== undefined) return Math.round((1 - window.utilization) * 100)
    if (window.remaining !== undefined && window.limit && window.limit > 0) {
      return Math.max(0, Math.min(100, Math.round((window.remaining / window.limit) * 100)))
    }
    return undefined
  }

  function quotaValueLabel(window: CredentialQuotaWindowDto): string {
    const value = remainingPercent(window)
    if (value !== undefined) {
      return t('group.credentials.subscription.remainingPercent', { value: n(value) })
    }
    if (window.remaining !== undefined && window.limit !== undefined) {
      return t('group.credentials.subscription.remaining', {
        remaining: n(window.remaining),
        limit: n(window.limit),
      })
    }
    if (window.remaining !== undefined) {
      return t('group.credentials.subscription.remainingAmount', {
        remaining: n(window.remaining),
      })
    }
    return t('group.credentials.subscription.unknown')
  }

  function quotaPeriodTooltip(window: CredentialQuotaWindowDto): string | undefined {
    const resetAtMS = window.reset_at_ms
    if (resetAtMS === undefined) return undefined
    const resetAt = formatLocalInstant(resetAtMS, locale)
    const period = quotaWindowPeriod(window)
    const periodLabel = period
      ? t('group.credentials.subscription.quotaPeriod', {
          start: formatLocalInstant(period.startMS, locale),
          end: resetAt,
        })
      : resetAt
    return quotaWindowNeedsRefresh(window)
      ? t('group.credentials.subscription.quotaPendingHint', { period: periodLabel })
      : periodLabel
  }

  function usedPercentValue(window: CredentialQuotaWindowDto): string {
    const remaining = remainingPercent(window)
    return remaining === undefined ? '—' : `${n(100 - remaining)}%`
  }

  function quotaTone(
    window: CredentialQuotaWindowDto,
  ): 'success' | 'warning' | 'danger' | undefined {
    const value = remainingPercent(window)
    if (value === undefined) return undefined
    return quotaProgressTone(value, window.state === 'exhausted')
  }

  function quotaFillWidth(window: CredentialQuotaWindowDto): string | undefined {
    const value = remainingPercent(window)
    return value === undefined ? undefined : `${value}%`
  }

  function toggleDetails(): void {
    if (detailBusy) return
    setDetailsExpanded((expanded) => !expanded)
  }

  function retryDetails(): void {
    onLoadDetails(item)
  }

  function runMenuAction(action: 'download' | 'refresh-credential' | 'restore' | 'remove'): void {
    setMenuOpen(false)
    switch (action) {
      case 'download':
        onDownload(item)
        return
      case 'refresh-credential':
        onRefreshCredential(item)
        return
      case 'restore':
        onRestore(item)
        return
      case 'remove':
        onRemove(item)
    }
  }

  const detailRegionId = `credential-detail-${item.credential_id}`

  return (
    <article
      {...stylex.props(styles.root, accountToneStyles[statusTone])}
      aria-busy={refreshingObservation ? true : undefined}
    >
      {refreshingObservation && (
        <div {...stylex.props(styles.refreshSkeleton)} role="status" aria-live="polite">
          <span {...stylex.props(styles.srOnly)}>
            {t('group.credentials.subscription.syncingQuota')}
          </span>
          <div {...stylex.props(styles.refreshSkeletonHeader)}>
            <div {...stylex.props(styles.refreshSkeletonTop)}>
              <div {...stylex.props(styles.refreshSkeletonCluster)}>
                <Skeleton width="20px" height="20px" />
                <Skeleton width="92px" height="24px" />
                <Skeleton width="64px" height="24px" />
              </div>
              <div {...stylex.props(styles.refreshSkeletonCluster)}>
                <Skeleton width="58px" height="12px" />
                <Skeleton width="32px" height="32px" />
                <Skeleton width="32px" height="32px" />
              </div>
            </div>
            <Skeleton width="62%" height="20px" aria-hidden="true" />
          </div>
          {supportsQuotaObservation && quotaWindows.length > 0 && (
            <div {...stylex.props(styles.refreshSkeletonQuotas)} aria-hidden="true">
              {Array.from({ length: refreshSkeletonRows }, (_, index) => (
                <div key={index} {...stylex.props(styles.refreshSkeletonQuota)}>
                  <Skeleton width="44%" height="13px" />
                  <span {...stylex.props(styles.refreshSkeletonQuotaMeta)}>
                    <Skeleton width="64px" height="12px" />
                    <Skeleton width="82px" height="12px" />
                  </span>
                </div>
              ))}
            </div>
          )}
          {supportsQuotaObservation && quotaWindows.length === 0 && (
            <Skeleton
              width="48%"
              height="16px"
              xstyle={styles.refreshSkeletonCentered}
              aria-hidden="true"
            />
          )}
          {unifiedStatus === 'quota_exhausted' && (
            <Skeleton
              width="82%"
              height="16px"
              xstyle={styles.refreshSkeletonCentered}
              aria-hidden="true"
            />
          )}
          {hasResetCredits && (
            <div {...stylex.props(styles.refreshSkeletonCredits)} aria-hidden="true">
              <Skeleton width="52px" height="14px" />
              <Skeleton width="72px" height="14px" />
              {nearestResetCredit && nearestResetCredit.expires_at_ms !== undefined && (
                <Skeleton
                  width="68px"
                  height="14px"
                  xstyle={styles.refreshSkeletonCreditsExpiry}
                />
              )}
              <Skeleton width="26px" height="26px" xstyle={styles.refreshSkeletonCreditsLast} />
            </div>
          )}
          <div {...stylex.props(styles.refreshSkeletonDetailControl)} aria-hidden="true">
            <span {...stylex.props(styles.refreshSkeletonRule)} />
            <Skeleton width="30px" height="30px" />
            <span {...stylex.props(styles.refreshSkeletonRule)} />
          </div>
          {detailsExpanded && (
            <div {...stylex.props(styles.refreshSkeletonDetail)} aria-hidden="true">
              {supportsQuotaObservation && hasUsageQuotaWindows && (
                <div {...stylex.props(styles.skeletonSection)}>
                  <span {...stylex.props(styles.skeletonTitle)}>
                    <Skeleton width="140px" height="11px" />
                  </span>
                  <Skeleton height={windowSkeletonHeight} />
                </div>
              )}
              <div {...stylex.props(styles.skeletonSection)}>
                <span {...stylex.props(styles.skeletonTitle)}>
                  <Skeleton width="88px" height="11px" />
                </span>
                <Skeleton height="var(--subscription-detail-overview-height)" />
              </div>
            </div>
          )}
        </div>
      )}
      <div {...stylex.props(styles.main, refreshingObservation && styles.hiddenDuringRefresh)}>
        <header {...stylex.props(styles.top)}>
          <div {...stylex.props(styles.topRow)}>
            <label {...stylex.props(styles.select)}>
              <span {...stylex.props(styles.srOnly)}>
                {t('group.credentials.subscription.selectAccount', { account: accountName })}
              </span>
              <input
                {...stylex.props(styles.selectInput)}
                type="checkbox"
                checked={selected}
                disabled={busy}
                onChange={(event) => onSelectedChange(event.target.checked)}
              />
              <span {...stylex.props(styles.selectBox)} aria-hidden="true">
                {selected && <Check size={16} strokeWidth={2.5} />}
              </span>
            </label>
            <div {...stylex.props(styles.badges)}>
              {planLabel !== '' && (
                <span
                  {...stylex.props(
                    styles.plan,
                    planLevel in planLevelStyles
                      ? planLevelStyles[planLevel as keyof typeof planLevelStyles]
                      : null,
                  )}
                >
                  {channelIcon && channelMark && (
                    <ChannelIcon icon={channelIcon} mark={channelMark} />
                  )}
                  <span>{planLabel}</span>
                </span>
              )}
              <Badge
                variant={
                  statusTone === 'success'
                    ? 'success'
                    : statusTone === 'warning'
                      ? 'warning'
                      : statusTone === 'danger'
                        ? 'error'
                        : 'neutral'
                }
                label={statusLabel}
              />
            </div>
            <div {...stylex.props(styles.actions)}>
              {supportsQuotaObservation && observation?.observed_at_ms != null && (
                <span {...stylex.props(styles.syncAge)}>
                  <RelativeInstant
                    instant={observation.observed_at_ms}
                    emptyLabel={t('group.credentials.subscription.unknown')}
                    tooltipContent={syncTimeTooltip}
                    hint
                  />
                </span>
              )}
              {supportsQuotaObservation && (
                <Tooltip content={t('group.credentials.subscription.sync')}>
                  <IconButton
                    variant="ghost"
                    size="sm"
                    label={t('group.credentials.subscription.sync')}
                    isLoading={refreshingObservation}
                    isDisabled={busy || observationRefreshBlocked}
                    onClick={() => onRefresh(item)}
                    icon={
                      <RefreshCw
                        size={15}
                        aria-hidden="true"
                        {...stylex.props(refreshingObservation && styles.syncIconSpinning)}
                      />
                    }
                  />
                </Tooltip>
              )}
              <Popover
                isOpen={menuOpen}
                onOpenChange={setMenuOpen}
                placement="below"
                alignment="end"
                xstyle={styles.menuSurface}
                content={
                  <div {...stylex.props(styles.menu)}>
                    <button
                      {...stylex.props(styles.menuItem)}
                      type="button"
                      disabled={busy}
                      onClick={() => runMenuAction('download')}
                    >
                      <Download size={15} aria-hidden="true" />
                      {t('group.credentials.subscription.download')}
                    </button>
                    <button
                      {...stylex.props(styles.menuItem)}
                      type="button"
                      disabled={busy}
                      onClick={() => runMenuAction('refresh-credential')}
                    >
                      <KeyRound size={15} aria-hidden="true" />
                      {t('group.credentials.subscription.refreshCredential')}
                    </button>
                    {isProblem && (
                      <button
                        {...stylex.props(styles.menuItem)}
                        type="button"
                        disabled={busy}
                        onClick={() => runMenuAction('restore')}
                      >
                        <RotateCcw size={15} aria-hidden="true" />
                        {t('group.credentials.restore')}
                      </button>
                    )}
                    <div {...stylex.props(styles.menuDivider)} />
                    <button
                      {...stylex.props(styles.menuItem, styles.menuDanger)}
                      type="button"
                      disabled={busy}
                      onClick={() => runMenuAction('remove')}
                    >
                      <Trash2 size={15} aria-hidden="true" />
                      {t('group.credentials.delete')}
                    </button>
                  </div>
                }
              >
                <IconButton
                  variant="ghost"
                  size="sm"
                  label={t('group.credentials.subscription.moreActions')}
                  isDisabled={busy}
                  icon={<Ellipsis size={16} aria-hidden="true" />}
                />
              </Popover>
            </div>
          </div>
          <div {...stylex.props(styles.topRow)}>
            <span {...stylex.props(styles.mail)} title={accountName}>
              {accountName}
            </span>
          </div>
        </header>

        {authIssue !== '' && (
          <div {...stylex.props(styles.alert)}>
            <span>{authIssue}</span>
          </div>
        )}
        {observationError !== '' && (
          <div {...stylex.props(styles.alert)} role="alert">
            <span>{observationError}</span>
          </div>
        )}

        {supportsQuotaObservation && quotaWindows.length > 0 ? (
          <div {...stylex.props(styles.quotas)}>
            {quotaWindows.map((window) => (
              <div
                key={window.id}
                {...stylex.props(styles.quota)}
                style={quotaToneStyles[quotaTone(window) ?? 'unknown']}
              >
                <span
                  {...stylex.props(styles.quotaMeter)}
                  role={remainingPercent(window) === undefined ? 'img' : 'progressbar'}
                  aria-label={
                    remainingPercent(window) === undefined
                      ? `${quotaWindowLabel(window)}: ${quotaValueLabel(window)}`
                      : quotaWindowLabel(window)
                  }
                  aria-valuemin={remainingPercent(window) === undefined ? undefined : 0}
                  aria-valuemax={remainingPercent(window) === undefined ? undefined : 100}
                  aria-valuenow={remainingPercent(window)}
                  aria-valuetext={
                    remainingPercent(window) === undefined ? undefined : quotaValueLabel(window)
                  }
                >
                  {remainingPercent(window) !== undefined && (
                    <span
                      {...stylex.props(styles.quotaFill)}
                      style={{ width: quotaFillWidth(window) }}
                      aria-hidden="true"
                    />
                  )}
                </span>
                <span {...stylex.props(styles.quotaName)} title={quotaWindowLabel(window)}>
                  {quotaWindowLabel(window)}
                </span>
                <span {...stylex.props(styles.quotaMeta)}>
                  <strong>{quotaValueLabel(window)}</strong>
                  <span aria-hidden="true">·</span>
                  <span {...stylex.props(styles.quotaReset)}>
                    {window.reset_at_ms && quotaWindowNeedsRefresh(window) ? (
                      <Tooltip content={quotaPeriodTooltip(window) ?? ''}>
                        <span {...stylex.props(styles.quotaResetPending)} tabIndex={0}>
                          {t('group.credentials.subscription.quotaPendingRefresh')}
                        </span>
                      </Tooltip>
                    ) : window.reset_at_ms ? (
                      <span {...stylex.props(styles.quotaResetTime)}>
                        {quotaResetPrefix !== '' && (
                          <span {...stylex.props(styles.quotaResetPrefix)}>
                            {quotaResetPrefix}
                          </span>
                        )}
                        <RelativeInstant
                          instant={window.reset_at_ms}
                          emptyLabel={t('group.credentials.subscription.unknown')}
                          tooltipContent={quotaPeriodTooltip(window)}
                          hint
                        />
                        {quotaResetSuffix !== '' && <span>{quotaResetSuffix}</span>}
                      </span>
                    ) : (
                      '—'
                    )}
                  </span>
                </span>
              </div>
            ))}
          </div>
        ) : needsInitialQuotaSync ? (
          <div {...stylex.props(styles.initialSync)}>
            <span {...stylex.props(styles.initialSyncIcon)} aria-hidden="true">
              <Gauge size={17} />
            </span>
            <span {...stylex.props(styles.initialSyncCopy)}>
              <strong>{t('group.credentials.subscription.initialSyncTitle')}</strong>
              <span>{t('group.credentials.subscription.initialSyncDescription')}</span>
            </span>
            <Button
              variant="secondary"
              size="sm"
              isDisabled={busy || observationRefreshBlocked}
              onClick={() => onRefresh(item)}
              icon={<RefreshCw size={14} aria-hidden="true" />}
              label={t('group.credentials.subscription.initialSyncAction')}
              xstyle={styles.initialSyncAction}
            />
          </div>
        ) : supportsQuotaObservation ? (
          <p {...stylex.props(styles.faint)}>
            {t('group.credentials.subscription.noQuota')}
          </p>
        ) : null}

        {unifiedStatus === 'quota_exhausted' && (
          <p {...stylex.props(styles.hint)}>
            {t('group.credentials.subscription.quotaExhaustedHint')}
          </p>
        )}

        {hasResetCredits && (
          <div {...stylex.props(styles.credits)}>
            <span>{t('group.credentials.subscription.resetCredits')}</span>
            <Tooltip content={resetCreditsTooltip}>
              <span
                {...stylex.props(styles.creditsSummary)}
                tabIndex={0}
                aria-label={resetCreditsTooltip}
              >
                <span {...stylex.props(styles.creditsDots)} aria-hidden="true">
                  {resetCreditDots.map((dot) => (
                    <i key={dot.index} {...stylex.props(creditDotStyles[dot.tone])} />
                  ))}
                </span>
                <strong>
                  {t('group.credentials.subscription.resetCreditsCount', {
                    count: n(resetCreditsAvailable),
                  })}
                </strong>
              </span>
            </Tooltip>
            {nearestResetCredit && nearestResetCredit.expires_at_ms !== undefined && (
              <span {...stylex.props(styles.creditsExpiry)}>
                {t('group.credentials.subscription.nearestResetCredit')}{' '}
                <RelativeInstant
                  instant={nearestResetCredit.expires_at_ms}
                  emptyLabel={t('group.credentials.subscription.unknown')}
                  hint
                />
              </span>
            )}
            <span {...stylex.props(styles.spacer)} />
            <Tooltip content={t('group.credentials.subscription.resetCreditsActionTooltip')}>
              <IconButton
                variant="ghost"
                size="sm"
                label={t('group.credentials.subscription.resetCreditsActionTooltip')}
                isDisabled={busy}
                onClick={() => onReset(item)}
                icon={<RotateCcw size={15} aria-hidden="true" />}
              />
            </Tooltip>
          </div>
        )}

        <div {...stylex.props(styles.detailControl)}>
          <span {...stylex.props(styles.detailRule)} />
          <Tooltip
            content={t(
              detailsExpanded
                ? 'group.credentials.subscription.collapseDetails'
                : 'group.credentials.subscription.expandDetails',
            )}
          >
            <button
              {...stylex.props(styles.detailToggle)}
              type="button"
              disabled={detailBusy}
              aria-expanded={detailsExpanded}
              aria-controls={detailRegionId}
              aria-label={t(
                detailsExpanded
                  ? 'group.credentials.subscription.collapseDetails'
                  : 'group.credentials.subscription.expandDetails',
              )}
              aria-busy={detailBusy ? true : undefined}
              onClick={toggleDetails}
            >
              <span {...stylex.props(styles.detailDisc)}>
                {detailBusy ? (
                  <LoaderCircle
                    size={15}
                    aria-hidden="true"
                    {...stylex.props(styles.detailSpinner)}
                  />
                ) : (
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                    width="15"
                    height="15"
                  >
                    <path d="M5 4.5h14A1.5 1.5 0 0 1 20.5 6v12a1.5 1.5 0 0 1-1.5 1.5H5A1.5 1.5 0 0 1 3.5 18V6A1.5 1.5 0 0 1 5 4.5Z" />
                    <path d="M3.5 10h17" />
                    <path d={detailsExpanded ? 'm9 17 3-3 3 3' : 'm9 14 3 3 3-3'} />
                  </svg>
                )}
              </span>
            </button>
          </Tooltip>
          <span {...stylex.props(styles.detailRule)} />
        </div>
      </div>

      {detailsExpanded && (
        <section
          id={detailRegionId}
          {...stylex.props(
            styles.detail,
            refreshingObservation && styles.hiddenDuringRefresh,
          )}
          aria-live="polite"
        >
          {!detailLoaded && !detailError ? (
            <div {...stylex.props(styles.skeleton)}>
              {supportsQuotaObservation && hasUsageQuotaWindows && (
                <div {...stylex.props(styles.skeletonSection)}>
                  <span {...stylex.props(styles.skeletonTitle)}>
                    <Skeleton width="140px" height="11px" />
                  </span>
                  <Skeleton height={windowSkeletonHeight} />
                </div>
              )}
              <div {...stylex.props(styles.skeletonSection)}>
                <span {...stylex.props(styles.skeletonTitle)}>
                  <Skeleton width="88px" height="11px" />
                </span>
                <Skeleton height="var(--subscription-detail-overview-height)" />
              </div>
            </div>
          ) : detailError !== '' ? (
            <div {...stylex.props(styles.detailError)} role="alert">
              <span>{detailError}</span>
              <Button
                variant="secondary"
                size="sm"
                isDisabled={busy}
                onClick={retryDetails}
                label={t('common.retry')}
              />
            </div>
          ) : (
            <div {...stylex.props(styles.detailContent)}>
              {supportsQuotaObservation && hasUsageQuotaWindows && (
                <section {...stylex.props(styles.detailSection)}>
                  <h3 {...stylex.props(styles.detailHeading)}>
                    {t('group.credentials.subscription.estimate.title')}
                  </h3>
                  <div {...stylex.props(styles.windowTable)} role="table">
                    <div {...stylex.props(styles.windowRow, styles.windowHead)} role="row">
                      <span role="columnheader">
                        {t('group.credentials.subscription.estimate.window')}
                      </span>
                      <span role="columnheader">
                        {t('group.credentials.subscription.estimate.used')}
                      </span>
                      <span role="columnheader">
                        {t('group.credentials.subscription.estimate.requests')}
                      </span>
                      <span role="columnheader">
                        {t('group.credentials.subscription.estimate.tokens')}
                      </span>
                      <span role="columnheader">
                        {t('group.credentials.subscription.estimate.referenceCost')}
                      </span>
                    </div>
                    {usageQuotaWindows.map((window) => (
                      <div key={window.id} {...stylex.props(styles.windowRow)} role="row">
                        <span
                          {...stylex.props(styles.windowName)}
                          role="cell"
                          title={quotaWindowLabel(window)}
                        >
                          {quotaWindowLabel(window)}
                        </span>
                        <span
                          {...stylex.props(styles.windowValue, styles.windowUsed)}
                          role="cell"
                        >
                          {usedPercentValue(window)}
                        </span>
                        <span
                          {...stylex.props(styles.windowValue)}
                          title={requestCountTitle(window)}
                          role="cell"
                        >
                          {window.observed_usage ? n(window.observed_usage.request_count) : '—'}
                        </span>
                        <span
                          {...stylex.props(styles.windowValue)}
                          title={tokenCountTitle(window)}
                          role="cell"
                        >
                          {window.observed_usage
                            ? formatTokens(window.observed_usage.total_tokens, locale)
                            : '—'}
                        </span>
                        <span
                          {...stylex.props(styles.windowValue)}
                          title={referenceCostTitle(window)}
                          role="cell"
                        >
                          {window.observed_usage
                            ? formatEstimatedCost(
                                window.observed_usage.estimated_reference_cost_nano_usd,
                                locale,
                              )
                            : '—'}
                        </span>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              <section {...stylex.props(styles.detailSection)}>
                <h3 {...stylex.props(styles.detailHeading)}>
                  {t('group.credentials.subscription.overview')}
                </h3>
                <div {...stylex.props(styles.diagnostics)}>
                  <dl {...stylex.props(styles.diagnosticItem)}>
                    <dt {...stylex.props(styles.diagnosticTerm)}>
                      {t('group.credentials.subscription.lastUsed')}
                    </dt>
                    <dd {...stylex.props(styles.diagnosticValue)}>
                      <RelativeInstant
                        instant={item.last_used_at_ms ?? null}
                        emptyLabel={t('group.credentials.subscription.unknown')}
                        hint
                      />
                    </dd>
                  </dl>
                  <dl {...stylex.props(styles.diagnosticItem)}>
                    <dt {...stylex.props(styles.diagnosticTerm)}>
                      {t('group.credentials.subscription.dailySuccessSummary')}
                    </dt>
                    <dd
                      {...stylex.props(
                        styles.diagnosticValue,
                        dailyUsage && styles.dailySuccess,
                      )}
                      title={dailyIncompleteHint}
                    >
                      {dailyUsage
                        ? n(dailyUsage.success_count)
                        : t('group.credentials.subscription.estimate.unavailable')}
                    </dd>
                  </dl>
                  <dl {...stylex.props(styles.diagnosticItem)}>
                    <dt {...stylex.props(styles.diagnosticTerm)}>
                      {t('group.credentials.subscription.dailyFailureSummary')}
                    </dt>
                    <dd
                      {...stylex.props(
                        styles.diagnosticValue,
                        dailyUsage && styles.dailyFailure,
                      )}
                      title={dailyIncompleteHint}
                    >
                      {dailyUsage
                        ? n(dailyUsage.failure_count)
                        : t('group.credentials.subscription.estimate.unavailable')}
                    </dd>
                  </dl>
                  <dl {...stylex.props(styles.diagnosticItem)}>
                    <dt {...stylex.props(styles.diagnosticTerm)}>
                      {t('group.credentials.detailsFailure')}
                    </dt>
                    <dd {...stylex.props(styles.diagnosticValue)}>{failureLabel}</dd>
                  </dl>
                  <dl {...stylex.props(styles.diagnosticItem)}>
                    <dt {...stylex.props(styles.diagnosticTerm)}>
                      {t('group.credentials.detailsConsecutive')}
                    </dt>
                    <dd {...stylex.props(styles.diagnosticValue)}>
                      {n(item.consecutive_failure_count)}
                    </dd>
                  </dl>
                  <dl {...stylex.props(styles.diagnosticItem)}>
                    <dt {...stylex.props(styles.diagnosticTerm)}>
                      {t('group.credentials.subscription.lastError')}
                    </dt>
                    <dd {...stylex.props(styles.diagnosticValue)}>
                      {observationErrorLabel(observation?.last_error_code)}
                    </dd>
                  </dl>
                  <dl {...stylex.props(styles.diagnosticItem)}>
                    <dt {...stylex.props(styles.diagnosticTerm)}>
                      {t('group.credentials.subscription.lastTokenRefresh')}
                    </dt>
                    <dd {...stylex.props(styles.diagnosticValue)}>
                      <RelativeInstant
                        instant={item.account.last_refresh_at_ms ?? null}
                        emptyLabel={t('group.credentials.subscription.unknown')}
                        hint
                      />
                    </dd>
                  </dl>
                  <dl {...stylex.props(styles.diagnosticItem)}>
                    <dt {...stylex.props(styles.diagnosticTerm)}>
                      {t('group.credentials.subscription.tokenExpiresAt')}
                    </dt>
                    <dd {...stylex.props(styles.diagnosticValue)}>
                      <RelativeInstant
                        instant={item.account.expires_at_ms ?? null}
                        emptyLabel={t('group.credentials.subscription.unknown')}
                        tooltipContent={credentialExpiryTooltip}
                        hint
                      />
                    </dd>
                  </dl>
                  <dl {...stylex.props(styles.diagnosticItem)}>
                    <dt {...stylex.props(styles.diagnosticTerm)}>
                      {t('group.credentials.subscription.lastQuotaSync')}
                    </dt>
                    <dd {...stylex.props(styles.diagnosticValue)}>
                      <RelativeInstant
                        instant={observation?.observed_at_ms ?? null}
                        emptyLabel={t('group.credentials.subscription.unknown')}
                        tooltipContent={syncExactTimeTooltip}
                        hint
                      />
                    </dd>
                  </dl>
                </div>
                {supportsQuotaObservation && constrainedModels.length > 0 && (
                  <div {...stylex.props(styles.models)}>
                    <span>{t('group.credentials.subscription.modelConstraints')}</span>
                    {constrainedModels.map((model) => (
                      <code key={model} {...stylex.props(styles.modelCode)}>
                        {model}
                      </code>
                    ))}
                  </div>
                )}
              </section>
            </div>
          )}
        </section>
      )}
    </article>
  )
}

const styles = stylex.create({
  root: {
    containerType: 'inline-size',
    position: 'relative',
    overflow: 'hidden',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'color-mix(in srgb, var(--color-action) 24%, var(--color-border-subtle))',
    borderLeftWidth: '3px',
    borderLeftStyle: 'solid',
    borderLeftColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'color-mix(in srgb, var(--color-action-soft) 46%, var(--color-surface))',
    boxShadow:
      '0 1px 2px color-mix(in srgb, var(--color-action) 14%, transparent), 0 8px 24px color-mix(in srgb, var(--color-action) 10%, transparent)',
  },
  hiddenDuringRefresh: {
    visibility: 'hidden',
  },
  refreshSkeleton: {
    position: 'absolute',
    zIndex: 2,
    inset: 0,
    display: 'grid',
    alignContent: 'start',
    gap: 'var(--space-3)',
    backgroundColor: 'color-mix(in srgb, var(--color-action-soft) 46%, var(--color-surface))',
    paddingTop: 'var(--space-3)',
    paddingBottom: 'var(--space-3)',
    paddingLeft: 'var(--space-4)',
    paddingRight: 'var(--space-4)',
  },
  refreshSkeletonHeader: {
    display: 'grid',
    gap: 'var(--space-2)',
    minWidth: 0,
  },
  refreshSkeletonTop: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
  },
  refreshSkeletonCluster: {
    display: 'flex',
    minWidth: 0,
    gap: 'var(--space-2)',
  },
  refreshSkeletonQuotas: {
    display: 'grid',
    gap: 'var(--space-2)',
  },
  refreshSkeletonQuota: {
    display: 'grid',
    minHeight: '34px',
    gridTemplateColumns: 'minmax(0, 1fr) auto',
    alignItems: 'center',
    gap: 'var(--space-3)',
    overflow: 'hidden',
    borderRadius: '6px',
    backgroundColor: 'var(--color-surface-sunken)',
    paddingTop: '7px',
    paddingBottom: '7px',
    paddingLeft: '10px',
    paddingRight: '10px',
  },
  refreshSkeletonQuotaMeta: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
  },
  refreshSkeletonCentered: {
    alignSelf: 'center',
  },
  refreshSkeletonCredits: {
    display: 'flex',
    minHeight: 'var(--control-compact)',
    alignItems: 'center',
    gap: 'var(--space-2)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    paddingTop: '4px',
    paddingBottom: '4px',
    paddingLeft: '6px',
    paddingRight: '6px',
  },
  refreshSkeletonCreditsExpiry: {
    marginLeft: 'var(--space-1)',
  },
  refreshSkeletonCreditsLast: {
    marginLeft: 'auto',
  },
  refreshSkeletonDetailControl: {
    display: 'grid',
    height: '44px',
    gridTemplateColumns: '1fr 44px 1fr',
    alignItems: 'center',
    marginTop: '2px',
  },
  refreshSkeletonRule: {
    height: '1px',
    backgroundColor: 'var(--color-border-subtle)',
  },
  refreshSkeletonDetail: {
    display: 'grid',
    gap: '13px',
    marginTop: 'calc(-1 * var(--space-3))',
    marginLeft: 'calc(-1 * var(--space-4))',
    marginRight: 'calc(-1 * var(--space-4))',
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'color-mix(in srgb, var(--color-action) 18%, var(--color-border-subtle))',
    backgroundColor: 'color-mix(in srgb, var(--color-action-soft) 72%, var(--color-surface))',
    paddingTop: '14px',
    paddingBottom: '16px',
    paddingLeft: '18px',
    paddingRight: '18px',
  },
  main: {
    display: 'grid',
    gap: 'var(--space-3)',
    paddingTop: 'var(--space-3)',
    paddingLeft: 'var(--space-4)',
    paddingRight: 'var(--space-4)',
    paddingBottom: 0,
  },
  top: {
    display: 'grid',
    gap: 'var(--space-2)',
    minWidth: 0,
  },
  topRow: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  select: {
    position: 'relative',
    display: 'grid',
    width: '20px',
    height: '20px',
    flexShrink: 0,
    alignItems: 'center',
    justifyItems: 'start',
    borderWidth: 0,
    backgroundColor: 'transparent',
    padding: 0,
    cursor: 'pointer',
  },
  selectInput: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    opacity: 0,
  },
  selectBox: {
    display: 'grid',
    width: '20px',
    height: '20px',
    placeItems: 'center',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'color-mix(in srgb, var(--color-text-faint) 42%, transparent)',
    borderRadius: '3px',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-action)',
  },
  badges: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  plan: {
    display: 'inline-flex',
    minHeight: '24px',
    flexShrink: 0,
    minWidth: 0,
    alignItems: 'center',
    gap: '5px',
    borderRadius: 'var(--radius-tag)',
    backgroundColor: 'var(--color-neutral-bg)',
    color: 'var(--color-neutral)',
    paddingTop: '3px',
    paddingBottom: '3px',
    paddingLeft: '8px',
    paddingRight: '8px',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
    fontWeight: 600,
    whiteSpace: 'nowrap',
  },
  mail: {
    minWidth: 0,
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: 'auto',
    overflow: 'hidden',
    marginRight: '2px',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-body)',
    fontWeight: 650,
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  syncAge: {
    flexShrink: 0,
    marginRight: 'var(--space-1)',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
  },
  syncIconSpinning: {
    animationName: stylex.keyframes({ to: { transform: 'rotate(360deg)' } }),
    animationDuration: '0.9s',
    animationTimingFunction: 'linear',
    animationIterationCount: 'infinite',
  },
  actions: {
    display: 'flex',
    flexShrink: 0,
    marginLeft: 'auto',
    alignItems: 'center',
    gap: '2px',
  },
  alert: {
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 'var(--space-2) var(--space-3)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-feedback-danger-border)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-danger-bg)',
    color: 'var(--color-text)',
    paddingTop: '8px',
    paddingBottom: '8px',
    paddingLeft: '10px',
    paddingRight: '10px',
    fontSize: 'var(--text-meta)',
  },
  quotas: {
    display: 'grid',
    gap: 'var(--space-2)',
  },
  /* 强调色用于左竖条与行底进度线，单值同时适配明暗；淡底只做状态提示 */
  quota: {
    position: 'relative',
    display: 'grid',
    minHeight: '34px',
    gridTemplateColumns: 'minmax(0, 1fr) auto',
    alignItems: 'center',
    gap: 'var(--space-3)',
    overflow: 'hidden',
    borderRadius: '6px',
    backgroundColor: 'var(--quota-tint)',
    /* 左竖条用 inset 阴影而非 border：border 会把 inset:0 的进度线整体右推 3px，
       导致左下圆角处出现断口。阴影不占盒模型，细线可贯通到最左侧与竖条重叠。 */
    boxShadow: 'inset 3px 0 0 var(--quota-accent)',
    paddingTop: '7px',
    paddingBottom: '7px',
    paddingLeft: '13px',
    paddingRight: '10px',
  },
  quotaMeter: {
    position: 'absolute',
    zIndex: 0,
    inset: 0,
    overflow: 'hidden',
    borderRadius: 'inherit',
    pointerEvents: 'none',
  },
  quotaFill: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    height: '3px',
    backgroundColor: 'var(--quota-accent)',
    transitionProperty: 'width',
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
  },
  quotaName: {
    position: 'relative',
    zIndex: 1,
    minWidth: 0,
    overflow: 'hidden',
    color: 'var(--color-text)',
    fontSize: 'var(--text-sm)',
    fontWeight: 600,
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  quotaMeta: {
    position: 'relative',
    zIndex: 1,
    display: 'inline-flex',
    minWidth: 'max-content',
    alignItems: 'center',
    gap: '6px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    fontVariantNumeric: 'tabular-nums',
    whiteSpace: 'nowrap',
  },
  quotaReset: {
    display: 'inline-flex',
    alignItems: 'center',
    minWidth: 0,
  },
  quotaResetPending: {
    color: 'var(--color-warning)',
    fontWeight: 600,
  },
  quotaResetTime: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '4px',
    minWidth: 0,
  },
  quotaResetPrefix: {
    flexShrink: 0,
  },
  initialSync: {
    display: 'grid',
    gridTemplateColumns: 'auto minmax(0, 1fr) auto',
    alignItems: 'center',
    gap: 'var(--space-3)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    paddingTop: '10px',
    paddingBottom: '10px',
    paddingLeft: '12px',
    paddingRight: '12px',
  },
  initialSyncIcon: {
    display: 'grid',
    width: '30px',
    height: '30px',
    placeItems: 'center',
    borderRadius: '50%',
    backgroundColor: 'var(--color-action-soft)',
    color: 'var(--color-action)',
  },
  initialSyncCopy: {
    display: 'grid',
    minWidth: 0,
    gap: '2px',
    fontSize: 'var(--text-label-xs)',
  },
  initialSyncAction: {
    flexShrink: 0,
  },
  faint: {
    margin: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
  },
  hint: {
    margin: 0,
    color: 'var(--color-warning)',
    fontSize: 'var(--text-meta)',
  },
  credits: {
    display: 'flex',
    minHeight: 'var(--control-compact)',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    paddingTop: '4px',
    paddingBottom: '4px',
    paddingLeft: '6px',
    paddingRight: '6px',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
  },
  creditsSummary: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    cursor: 'default',
  },
  creditsDots: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '3px',
  },
  creditsExpiry: {
    marginLeft: 'var(--space-1)',
    color: 'var(--color-text-faint)',
  },
  spacer: {
    flexGrow: 1,
  },
  detailControl: {
    display: 'grid',
    gridTemplateColumns: '1fr 44px 1fr',
    alignItems: 'center',
    marginTop: '2px',
  },
  detailRule: {
    height: '1px',
    backgroundColor: 'var(--color-border-subtle)',
  },
  detailToggle: {
    display: 'grid',
    width: '44px',
    height: '44px',
    placeItems: 'center',
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: 'var(--color-text-faint)',
    cursor: 'pointer',
    padding: 0,
  },
  detailDisc: {
    display: 'grid',
    width: '30px',
    height: '30px',
    placeItems: 'center',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: '50%',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text-muted)',
    boxShadow: '0 1px 2px rgb(24 29 33 / 5%)',
  },
  detailSpinner: {
    animationName: stylex.keyframes({ to: { transform: 'rotate(360deg)' } }),
    animationDuration: '0.9s',
    animationTimingFunction: 'linear',
    animationIterationCount: 'infinite',
  },
  detail: {
    display: 'grid',
    gap: '13px',
    marginTop: 'calc(-1 * var(--space-3))',
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'color-mix(in srgb, var(--color-action) 18%, var(--color-border-subtle))',
    backgroundColor: 'color-mix(in srgb, var(--color-action-soft) 72%, var(--color-surface))',
    paddingTop: '14px',
    paddingBottom: '16px',
    paddingLeft: '18px',
    paddingRight: '18px',
  },
  skeleton: {
    '--subscription-detail-overview-height': {
      default: '92px',
      '@media (max-width: 680px)': '154px',
    },
    display: 'grid',
    gap: '13px',
  },
  skeletonSection: {
    display: 'grid',
    gap: '8px',
  },
  skeletonTitle: {
    display: 'block',
  },
  detailError: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-2) var(--space-3)',
    flexWrap: 'wrap',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-danger-bg)',
    paddingTop: '8px',
    paddingBottom: '8px',
    paddingLeft: '10px',
    paddingRight: '10px',
    color: 'var(--color-danger)',
    fontSize: 'var(--text-meta)',
  },
  detailContent: {
    display: 'grid',
    gap: '13px',
  },
  detailSection: {
    display: 'grid',
    gap: '8px',
  },
  detailHeading: {
    margin: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 700,
    letterSpacing: '0.04em',
    textTransform: 'uppercase',
  },
  windowTable: {
    display: 'grid',
    borderRadius: '6px',
    backgroundColor: 'var(--color-surface)',
    overflow: 'hidden',
  },
  windowRow: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1.4fr) repeat(4, minmax(0, 1fr))',
    alignItems: 'center',
    gap: 'var(--space-2)',
    paddingTop: '6px',
    paddingBottom: '6px',
    paddingLeft: '10px',
    paddingRight: '10px',
    fontSize: 'var(--text-label-xs)',
  },
  windowHead: {
    color: 'var(--color-text-faint)',
    fontWeight: 600,
  },
  windowName: {
    minWidth: 0,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  windowValue: {
    fontVariantNumeric: 'tabular-nums',
    textAlign: 'right',
    whiteSpace: 'nowrap',
  },
  windowUsed: {
    fontWeight: 650,
  },
  diagnostics: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
    gap: '8px 14px',
    borderRadius: '6px',
    backgroundColor: 'var(--color-surface)',
    paddingTop: '10px',
    paddingBottom: '10px',
    paddingLeft: '12px',
    paddingRight: '12px',
  },
  diagnosticItem: {
    display: 'grid',
    gap: '2px',
    minWidth: 0,
    margin: 0,
  },
  diagnosticTerm: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  diagnosticValue: {
    margin: 0,
    fontSize: 'var(--text-meta)',
    fontVariantNumeric: 'tabular-nums',
  },
  dailySuccess: {
    color: 'var(--color-success)',
  },
  dailyFailure: {
    color: 'var(--color-danger)',
  },
  models: {
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: '6px',
    borderRadius: '6px',
    backgroundColor: 'var(--color-surface)',
    paddingTop: '8px',
    paddingBottom: '8px',
    paddingLeft: '12px',
    paddingRight: '12px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  modelCode: {
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-tag)',
    backgroundColor: 'var(--color-surface-sunken)',
    paddingTop: '2px',
    paddingBottom: '2px',
    paddingLeft: '6px',
    paddingRight: '6px',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
  },
  menuSurface: {
    minWidth: '180px',
    borderRadius: '10px',
    padding: '8px',
  },
  menu: {
    display: 'grid',
    width: '100%',
    gap: '1px',
  },
  menuItem: {
    display: 'flex',
    width: '100%',
    alignItems: 'center',
    gap: 'var(--space-2)',
    borderWidth: 0,
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'transparent',
    color: 'var(--color-text)',
    paddingTop: '7px',
    paddingBottom: '7px',
    paddingLeft: '6px',
    paddingRight: '6px',
    fontSize: 'var(--text-button)',
    fontFamily: 'inherit',
    textAlign: 'left',
    cursor: 'pointer',
  },
  menuDanger: {
    color: 'var(--color-danger)',
  },
  menuDivider: {
    height: '1px',
    marginTop: '4px',
    marginBottom: '4px',
    marginLeft: '-8px',
    marginRight: '-8px',
    backgroundColor: 'var(--color-border-subtle)',
  },
  srOnly: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
})

const accountToneStyles = stylex.create({
  success: { borderLeftColor: 'var(--color-success)' },
  warning: { borderLeftColor: 'var(--color-warning)' },
  danger: { borderLeftColor: 'var(--color-danger)' },
  neutral: { borderLeftColor: 'var(--color-border-control)' },
})

// Quota tone rides per-row CSS custom properties exactly like the classic
// modifier classes: --quota-accent drives the left bar + bottom fill,
// --quota-tint the row background.
const quotaToneStyles: Record<'success' | 'warning' | 'danger' | 'unknown', CSSProperties> = {
  success: {
    '--quota-accent': 'oklch(70% 0.16 158)',
    '--quota-tint': 'light-dark(#dcfeea, #112b21)',
  } as CSSProperties,
  warning: {
    '--quota-accent': 'oklch(75% 0.152 75)',
    '--quota-tint': 'light-dark(#fff2e2, #302212)',
  } as CSSProperties,
  danger: {
    '--quota-accent': 'oklch(65% 0.2 22)',
    '--quota-tint': 'light-dark(#fef0f0, #371a1d)',
  } as CSSProperties,
  unknown: {
    '--quota-accent': 'var(--color-border-control)',
    '--quota-tint': 'var(--color-surface-sunken)',
  } as CSSProperties,
}

const planLevelStyles = stylex.create({
  free: {
    backgroundColor: 'var(--color-neutral-bg)',
    color: 'var(--color-neutral)',
  },
  standard: {
    backgroundColor: 'var(--color-success-bg)',
    color: 'var(--color-success)',
  },
  premium: {
    backgroundColor: 'var(--color-action-soft)',
    color: 'var(--color-action)',
  },
  elite: {
    backgroundColor: 'var(--color-warning-bg)',
    color: 'var(--color-warning)',
  },
})

const creditDotStyles = stylex.create({
  default: {
    display: 'inline-block',
    width: '7px',
    height: '7px',
    borderRadius: '50%',
    backgroundColor: 'var(--color-action)',
  },
  warning: {
    display: 'inline-block',
    width: '7px',
    height: '7px',
    borderRadius: '50%',
    backgroundColor: 'var(--color-warning)',
  },
  danger: {
    display: 'inline-block',
    width: '7px',
    height: '7px',
    borderRadius: '50%',
    backgroundColor: 'var(--color-danger)',
  },
})
