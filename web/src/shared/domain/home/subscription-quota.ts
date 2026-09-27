import type { HomeSubscriptionAccountDto } from '@shared/control/resources/home'
import type {
  CredentialItemDto,
  CredentialQuotaLabelKey,
  CredentialQuotaWindowDto,
  CredentialResetCreditDto,
} from '@shared/control/types'
import type { QuotaProgressTone } from '@shared/lib/quota-progress'

import { formatLocalInstant } from '@shared/lib/format'
import { quotaProgressTone } from '@shared/lib/quota-progress'

// Framework-free ports of the classic HomeSubscriptionAccountMiniCard helpers.
// The translator is parameterized so vue-i18n and react-intl consumers share
// the exact label/tooltip/status semantics.

export interface SubscriptionQuotaTranslator {
  t(key: string, values?: Record<string, string | number>): string
  has(key: string): boolean
  n(value: number): string
  locale: string
}

export type SubscriptionUnifiedStatus =
  | 'available'
  | 'quota_exhausted'
  | 'cooldown'
  | 'blacklisted'
  | 'refreshing'
  | 'needs_reauth'
  | 'outcome_unknown'
export type SubscriptionCardTone = 'success' | 'warning' | 'danger' | 'neutral'
export type ResetCreditDotTone = 'default' | 'warning' | 'danger'

export function isAccountWideQuotaWindow(window: CredentialQuotaWindowDto): boolean {
  return window.scope === 'account'
}

function quotaWindowDuration(window: CredentialQuotaWindowDto): number {
  const seconds = window.window_seconds
  return seconds !== undefined && Number.isFinite(seconds) && seconds > 0
    ? seconds
    : Number.MAX_SAFE_INTEGER
}

// Account-scoped windows first, then ascending duration — matches classic.
export function sortQuotaWindows(
  windows: readonly CredentialQuotaWindowDto[],
): CredentialQuotaWindowDto[] {
  return [...windows].sort((left, right) => {
    const scopeDifference =
      Number(!isAccountWideQuotaWindow(left)) - Number(!isAccountWideQuotaWindow(right))
    return scopeDifference !== 0
      ? scopeDifference
      : quotaWindowDuration(left) - quotaWindowDuration(right)
  })
}

// The strip shows at most 4 segments; a lead window pushed past the cap is
// force-merged so the headline number and strip always reference one window.
export function capQuotaWindows(
  sorted: readonly CredentialQuotaWindowDto[],
  leadWindow: CredentialQuotaWindowDto | undefined,
): CredentialQuotaWindowDto[] {
  const capped = sorted.slice(0, 4)
  if (leadWindow && !capped.some((window) => window.id === leadWindow.id)) {
    return [leadWindow, ...capped.slice(0, 3)]
  }
  return capped
}

export function remainingQuotaPercent(window: CredentialQuotaWindowDto): number | undefined {
  if (window.utilization !== undefined) return Math.round((1 - window.utilization) * 100)
  if (window.remaining !== undefined && window.limit && window.limit > 0) {
    return Math.max(0, Math.min(100, Math.round((window.remaining / window.limit) * 100)))
  }
  return undefined
}

export function quotaWindowTone(
  window: CredentialQuotaWindowDto,
): QuotaProgressTone | undefined {
  const value = remainingQuotaPercent(window)
  return value === undefined ? undefined : quotaProgressTone(value, window.state === 'exhausted')
}

// Day/hour get compact suffixes; minutes/seconds stay so odd windows still label.
export function quotaWindowPeriodLabel(seconds: number | undefined): string {
  if (seconds === undefined || !Number.isSafeInteger(seconds) || seconds <= 0) return ''
  const day = 24 * 60 * 60
  const hour = 60 * 60
  const minute = 60
  if (seconds % day === 0) return `${seconds / day}d`
  if (seconds % hour === 0) return `${seconds / hour}h`
  if (seconds % minute === 0) return `${seconds / minute}min`
  return `${seconds}s`
}

const quotaSubjectKeys: Readonly<Record<string, CredentialQuotaLabelKey>> = {
  session: 'session',
  weekly: 'weekly',
  'extra usage': 'extra_usage',
  'included usage': 'included_usage',
  'pay as you go': 'pay_as_you_go',
  'oauth apps': 'oauth_apps',
}

function normalizedQuotaLabelPart(value: string): string {
  return value.trim().toLowerCase().replaceAll('_', ' ').replaceAll('-', ' ')
}

function translatedQuotaLabel(
  labelKey: CredentialQuotaLabelKey,
  fallback: string,
  tr: SubscriptionQuotaTranslator,
): string {
  const key = `group.credentials.subscription.quotaLabels.${labelKey}`
  return tr.has(key) ? tr.t(key) : fallback
}

// label_key 'oauth_apps' keeps the period suffix so same-kind windows at
// different periods do not collapse into identical labels.
export function quotaWindowLabel(
  window: CredentialQuotaWindowDto,
  tr: SubscriptionQuotaTranslator,
): string {
  const period = quotaWindowPeriodLabel(window.window_seconds)
  if (period && window.scope === 'account') return period

  if (window.label_key) {
    const subject = translatedQuotaLabel(window.label_key, window.label, tr)
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
      return labelKey ? translatedQuotaLabel(labelKey, part, tr) : part
    })
    .join(' · ')
}

export function quotaValueLabel(
  window: CredentialQuotaWindowDto,
  tr: SubscriptionQuotaTranslator,
): string {
  const value = remainingQuotaPercent(window)
  if (value !== undefined) {
    return tr.t('group.credentials.subscription.remainingPercent', { value: tr.n(value) })
  }
  if (window.remaining !== undefined && window.limit !== undefined) {
    return tr.t('group.credentials.subscription.remaining', {
      remaining: tr.n(window.remaining),
      limit: tr.n(window.limit),
    })
  }
  if (window.remaining !== undefined) {
    return tr.t('group.credentials.subscription.remainingAmount', {
      remaining: tr.n(window.remaining),
    })
  }
  return tr.t('group.credentials.subscription.unknown')
}

export interface QuotaWindowPeriod {
  startMS: number
  endMS: number
}

export function quotaWindowPeriod(
  window: CredentialQuotaWindowDto,
): QuotaWindowPeriod | undefined {
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

export function quotaWindowNeedsRefresh(
  window: CredentialQuotaWindowDto,
  nowMs: number,
): boolean {
  return window.reset_at_ms !== undefined && window.reset_at_ms <= nowMs
}

export function quotaPeriodTooltip(
  window: CredentialQuotaWindowDto,
  tr: SubscriptionQuotaTranslator,
  nowMs: number,
): string | undefined {
  const resetAtMS = window.reset_at_ms
  if (resetAtMS === undefined) return undefined
  const resetAt = formatLocalInstant(resetAtMS, tr.locale)
  const period = quotaWindowPeriod(window)
  const periodLabel = period
    ? tr.t('group.credentials.subscription.quotaPeriod', {
        start: formatLocalInstant(period.startMS, tr.locale),
        end: resetAt,
      })
    : resetAt
  return quotaWindowNeedsRefresh(window, nowMs)
    ? tr.t('group.credentials.subscription.quotaPendingHint', { period: periodLabel })
    : periodLabel
}

export function quotaTooltip(
  window: CredentialQuotaWindowDto,
  tr: SubscriptionQuotaTranslator,
  nowMs: number,
): string {
  return [quotaWindowLabel(window, tr), quotaValueLabel(window, tr), quotaPeriodTooltip(window, tr, nowMs)]
    .filter((line): line is string => Boolean(line))
    .join('\n')
}

export interface LeadQuotaWindow {
  window: CredentialQuotaWindowDto
  percent: number | undefined
}

// Explicit exhausted windows lead; otherwise the lowest remaining percent
// across the full sorted list (not the capped strip).
export function selectLeadQuotaWindow(
  sorted: readonly CredentialQuotaWindowDto[],
): LeadQuotaWindow | undefined {
  const exhausted = sorted.find((window) => window.state === 'exhausted')
  if (exhausted) return { window: exhausted, percent: remainingQuotaPercent(exhausted) }

  let best: { window: CredentialQuotaWindowDto; percent: number } | undefined
  for (const window of sorted) {
    const percent = remainingQuotaPercent(window)
    if (percent === undefined) continue
    if (best === undefined || percent < best.percent) best = { window, percent }
  }
  if (best) return best
  const [first] = sorted
  return first ? { window: first, percent: undefined } : undefined
}

export function subscriptionUnifiedStatus(
  credential: CredentialItemDto,
  quotaObservation: boolean,
  sortedWindows: readonly CredentialQuotaWindowDto[],
): SubscriptionUnifiedStatus {
  if (credential.auth_state === 'refreshing') return 'refreshing'
  if (credential.auth_state === 'reauthorization_required') return 'needs_reauth'
  if (credential.auth_state === 'outcome_unknown') return 'outcome_unknown'
  if (credential.effective_status === 'blacklisted') return 'blacklisted'
  if (credential.effective_status === 'cooldown') return 'cooldown'
  if (
    quotaObservation &&
    sortedWindows.some((window) => window.scope === 'account' && window.state === 'exhausted')
  ) {
    return 'quota_exhausted'
  }
  return 'available'
}

export function showAggregateAvailability(
  account: Pick<HomeSubscriptionAccountDto, 'group_count' | 'available_group_count'>,
  status: SubscriptionUnifiedStatus,
): boolean {
  return (
    account.group_count > 1 &&
    (account.available_group_count < account.group_count || status === 'available')
  )
}

export function subscriptionCardTone(
  aggregate: boolean,
  account: Pick<HomeSubscriptionAccountDto, 'group_count' | 'available_group_count'>,
  status: SubscriptionUnifiedStatus,
): SubscriptionCardTone {
  if (aggregate) {
    if (account.available_group_count === account.group_count) return 'success'
    return account.available_group_count > 0 ? 'warning' : 'danger'
  }
  const tones: Record<SubscriptionUnifiedStatus, SubscriptionCardTone> = {
    available: 'success',
    quota_exhausted: 'warning',
    cooldown: 'warning',
    blacklisted: 'danger',
    refreshing: 'neutral',
    needs_reauth: 'danger',
    outcome_unknown: 'danger',
  }
  return tones[status]
}

export function subscriptionStatusLabel(
  aggregate: boolean,
  account: Pick<HomeSubscriptionAccountDto, 'group_count' | 'available_group_count'>,
  status: SubscriptionUnifiedStatus,
  tr: SubscriptionQuotaTranslator,
): string {
  return aggregate
    ? tr.t('home.ledger.subscriptionAccounts.availableGroups', {
        available: tr.n(account.available_group_count),
        total: tr.n(account.group_count),
      })
    : tr.t(`group.credentials.subscription.status.${status}`)
}

export function showSubscriptionStatusChip(
  aggregate: boolean,
  status: SubscriptionUnifiedStatus,
): boolean {
  return aggregate || status !== 'available'
}

export interface ResetCreditDot {
  index: number
  tone: ResetCreditDotTone
}

// Dots are capped at 5; exact counts live in the tooltip/screen-reader text.
export function resetCreditDots(
  credits: readonly CredentialResetCreditDto[],
  available: number,
  nowMs: number,
): ResetCreditDot[] {
  const details = credits.slice(0, available)
  return Array.from({ length: Math.min(available, 5) }, (_, index) => {
    const expiresAtMS = details[index]?.expires_at_ms
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
  })
}

function resetCreditExpiryLabel(
  credit: CredentialResetCreditDto,
  tr: SubscriptionQuotaTranslator,
  nowMs: number,
): string {
  if (credit.expires_at_ms === undefined) {
    return tr.t('group.credentials.subscription.resetCreditPermanent')
  }
  if (credit.expires_at_ms <= nowMs) {
    return tr.t('group.credentials.subscription.resetCreditExpired')
  }
  return formatLocalInstant(credit.expires_at_ms, tr.locale)
}

// One tooltip covers every reset credit (title + per-credit expiry).
export function resetCreditsTooltip(
  credits: readonly CredentialResetCreditDto[],
  available: number,
  tr: SubscriptionQuotaTranslator,
  nowMs: number,
): string {
  const details = credits.slice(0, available)
  const lines = details.length
    ? details.map((credit, index) =>
        tr.t('group.credentials.subscription.resetCreditsTooltipItem', {
          index: index + 1,
          expires: resetCreditExpiryLabel(credit, tr, nowMs),
        }),
      )
    : [tr.t('group.credentials.subscription.resetCreditsTooltipNoDetails')]
  if (details.length < available) {
    lines.push(
      tr.t('group.credentials.subscription.resetCreditsTooltipMore', {
        count: tr.n(available - details.length),
      }),
    )
  }
  return [tr.t('group.credentials.subscription.resetCreditsTooltipTitle'), ...lines].join('\n')
}
