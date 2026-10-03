import * as stylex from '@stylexjs/stylex'
import { Badge, Tooltip } from '@astryxdesign/core'
import { useEffect, useState, type CSSProperties } from 'react'
import { useIntl } from 'react-intl'

import type { HomeSubscriptionAccountDto } from '@shared/control/resources/home'
import {
  capQuotaWindows,
  quotaPeriodTooltip,
  quotaTooltip,
  quotaValueLabel,
  quotaWindowLabel,
  quotaWindowNeedsRefresh,
  quotaWindowTone,
  remainingQuotaPercent,
  resetCreditDots,
  resetCreditsTooltip,
  selectLeadQuotaWindow,
  showAggregateAvailability,
  showSubscriptionStatusChip,
  sortQuotaWindows,
  subscriptionCardTone,
  subscriptionStatusLabel,
  subscriptionUnifiedStatus,
  type SubscriptionQuotaTranslator,
} from '@shared/domain/home/subscription-quota'

import { useT } from '../../app/i18n'
import { ChannelIcon } from '../../components/ChannelIcon'
import { RelativeInstant } from '../../components/RelativeInstant'

const REDUCED_MOTION = '@media (prefers-reduced-motion: reduce)'

const styles = stylex.create({
  card: {
    display: 'flex',
    minWidth: 0,
    flexDirection: 'column',
    gap: 8,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    boxShadow: 'var(--shadow-card)',
    paddingTop: 10,
    paddingBottom: 10,
    paddingInline: 11,
  },
  srOnly: {
    position: 'absolute',
    width: 1,
    height: 1,
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
  top: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 7,
  },
  channel: {
    display: 'inline-flex',
    width: 16,
    height: 16,
    flex: 'none',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 4,
    fontSize: 16,
  },
  account: {
    minWidth: 0,
    flex: '1 1 auto',
    overflow: 'hidden',
    color: 'var(--color-text)',
    fontSize: 'var(--text-sm)',
    fontWeight: 600,
    lineHeight: '16px',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  plan: {
    flex: 'none',
    borderRadius: 5,
    paddingTop: 1,
    paddingBottom: 1,
    paddingInline: 6,
    fontSize: 'var(--text-label-xs)',
    fontWeight: 600,
    whiteSpace: 'nowrap',
  },
  planFree: {
    backgroundColor: 'var(--color-neutral-bg)',
    color: 'var(--color-neutral-fg)',
  },
  planStandard: {
    backgroundColor: 'var(--color-success-bg)',
    color: 'var(--color-success)',
  },
  planPremium: {
    backgroundColor: 'var(--color-action-soft)',
    color: 'var(--color-action)',
  },
  planElite: {
    backgroundColor: 'var(--color-warning-bg)',
    color: 'var(--color-warning)',
  },
  lead: {
    display: 'flex',
    alignItems: 'flex-end',
    gap: 7,
  },
  num: {
    display: 'inline-flex',
    flex: 'none',
    alignItems: 'baseline',
    gap: 1,
    fontVariantNumeric: 'tabular-nums',
  },
  numStrong: {
    fontSize: 25,
    fontWeight: 640,
    letterSpacing: '-0.02em',
    lineHeight: 1,
  },
  numSuffix: {
    fontSize: 12,
    fontWeight: 600,
  },
  numAmount: {
    maxWidth: '9em',
    overflow: 'hidden',
    fontSize: 'var(--text-sm)',
    fontWeight: 650,
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  // Same oklch accent set as the group-detail subscription quota bar — these
  // numbers are quota-remaining amounts, not status semantics.
  toneSuccess: { color: 'oklch(70% 0.16 158)' },
  toneWarning: { color: 'oklch(75% 0.152 75)' },
  toneDanger: { color: 'oklch(65% 0.2 22)' },
  toneUnknown: { color: 'var(--color-text-faint)' },
  numEmpty: {
    color: 'var(--color-text-faint)',
    fontSize: 21,
    fontWeight: 500,
    lineHeight: 1,
  },
  meta: {
    display: 'block',
    flex: '1 1 auto',
    minWidth: 0,
    overflow: 'hidden',
    paddingBottom: 3,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    lineHeight: 1.3,
    textAlign: 'right',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  pending: {
    cursor: 'help',
    textDecorationLine: 'underline',
    textDecorationStyle: 'dotted',
    textDecorationColor: 'var(--color-border-control)',
    textUnderlineOffset: 3,
    borderRadius: { ':focus-visible': 3 },
    outline: { ':focus-visible': '2px solid var(--color-focus)' },
    outlineOffset: { ':focus-visible': 2 },
  },
  resources: {
    display: 'flex',
    alignItems: 'center',
    columnGap: 8,
    rowGap: 6,
    flexWrap: 'wrap',
  },
  status: {
    flex: 'none',
  },
  strip: {
    display: 'flex',
    minWidth: 32,
    flex: '1 1 32px',
    alignItems: 'center',
    gap: 4,
  },
  spacer: {
    flex: '1 1 auto',
  },
  seg: {
    position: 'relative',
    display: 'block',
    flex: '1 1 0',
    height: 4,
    overflow: 'hidden',
    borderRadius: 999,
    cursor: 'help',
  },
  segTrack: {
    backgroundColor: 'var(--color-surface-sunken)',
  },
  segTrackSuccess: { backgroundColor: 'light-dark(#dcfeea, #112b21)' },
  segTrackWarning: { backgroundColor: 'light-dark(#fff2e2, #302212)' },
  segTrackDanger: { backgroundColor: 'light-dark(#fef0f0, #371a1d)' },
  segFill: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    borderRadius: 'inherit',
    transitionProperty: {
      default: 'width',
      [REDUCED_MOTION]: 'none',
    },
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
  },
  segFillUnknown: { backgroundColor: 'var(--color-border-control)' },
  segFillSuccess: { backgroundColor: 'oklch(70% 0.16 158)' },
  segFillWarning: { backgroundColor: 'oklch(75% 0.152 75)' },
  segFillDanger: { backgroundColor: 'oklch(65% 0.2 22)' },
  credits: {
    display: 'inline-flex',
    flex: 'none',
    alignItems: 'center',
    gap: 3,
    cursor: 'help',
    outline: { ':focus-visible': '2px solid var(--color-focus)' },
    outlineOffset: { ':focus-visible': 2 },
    borderRadius: { ':focus-visible': 3 },
  },
  creditDot: {
    width: 6,
    height: 6,
    flex: 'none',
    borderRadius: '50%',
    backgroundColor: 'var(--color-action)',
  },
  creditDotWarning: { backgroundColor: 'var(--color-warning)' },
  creditDotDanger: { backgroundColor: 'var(--color-danger)' },
  creditMore: {
    color: 'var(--color-text-faint)',
    fontSize: 9,
    fontWeight: 700,
    lineHeight: 1,
  },
})

const planStyles = {
  free: styles.planFree,
  standard: styles.planStandard,
  premium: styles.planPremium,
  elite: styles.planElite,
} as const

const toneStyles = {
  success: styles.toneSuccess,
  warning: styles.toneWarning,
  danger: styles.toneDanger,
} as const

const segTrackStyles = {
  success: styles.segTrackSuccess,
  warning: styles.segTrackWarning,
  danger: styles.segTrackDanger,
} as const

const segFillStyles = {
  success: styles.segFillSuccess,
  warning: styles.segFillWarning,
  danger: styles.segFillDanger,
} as const

const badgeVariants = {
  success: 'success',
  warning: 'warning',
  danger: 'error',
  neutral: 'neutral',
} as const

const creditDotStyles = {
  warning: styles.creditDotWarning,
  danger: styles.creditDotDanger,
} as const

export function SubscriptionAccountMiniCard({ account }: { account: HomeSubscriptionAccountDto }) {
  const intl = useIntl()
  const t = useT()
  // Real-clock ticking (30s): reset-credit dot tones and quota pending state
  // must not freeze at the mount instant — classic card keeps the same timer.
  const [nowMs, setNowMs] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 30_000)
    return () => window.clearInterval(timer)
  }, [])

  const tr: SubscriptionQuotaTranslator = {
    t: (key, values) => intl.formatMessage({ id: key }, values) as string,
    has: (key) => key in intl.messages,
    n: (value) => intl.formatNumber(value),
    locale: intl.locale,
  }

  const credential = account.credential
  const snapshot = credential.observation?.snapshot
  const accountName = credential.account.email ?? credential.mask
  const planLabel = snapshot?.plan_summary.name?.trim() ?? ''
  const planLevel = snapshot?.plan_summary.level ?? 'unknown'
  const channelTooltip = [account.channel_name, planLabel].filter(Boolean).join(' · ')

  const sortedWindows = sortQuotaWindows(snapshot?.quota_windows ?? [])
  const lead = selectLeadQuotaWindow(sortedWindows)
  const quotaWindows = capQuotaWindows(sortedWindows, lead?.window)
  const leadTone = lead ? quotaWindowTone(lead.window) : undefined

  const status = subscriptionUnifiedStatus(
    credential,
    account.capabilities.quota_observation,
    sortedWindows,
  )
  const aggregate = showAggregateAvailability(account, status)
  const cardTone = subscriptionCardTone(aggregate, account, status)
  const statusLabel = subscriptionStatusLabel(aggregate, account, status, tr)
  const showStatus = showSubscriptionStatusChip(aggregate, status)

  const quotaResetPrefix = t('group.credentials.subscription.quotaResetPrefix')
  const quotaResetSuffix = t('group.credentials.subscription.quotaResetSuffix')

  const resetCreditsAvailable = snapshot?.reset_credits_available ?? 0
  const resetCredits = snapshot?.reset_credits ?? []
  const hasResetCredits =
    account.capabilities.credential_actions.includes('reset_credit') && resetCreditsAvailable > 0
  const dots = resetCreditDots(resetCredits, resetCreditsAvailable, nowMs)
  const creditsTooltip = resetCreditsTooltip(resetCredits, resetCreditsAvailable, tr, nowMs)

  const numToneStyle = leadTone ? toneStyles[leadTone] : styles.toneUnknown

  return (
    <article {...stylex.props(styles.card)} aria-label={`${accountName} · ${statusLabel}`}>
      <span {...stylex.props(styles.srOnly)}>{statusLabel}</span>

      <div {...stylex.props(styles.top)}>
        <span {...stylex.props(styles.channel)} role="img" aria-label={channelTooltip}>
          <ChannelIcon icon={account.channel_icon} mark={account.channel_mark} />
        </span>
        <Tooltip content={accountName}>
          <span {...stylex.props(styles.account)}>{accountName}</span>
        </Tooltip>
        {planLabel !== '' && (
          <span
            {...stylex.props(
              styles.plan,
              planStyles[planLevel as keyof typeof planStyles] ?? styles.planFree,
            )}
          >
            {planLabel}
          </span>
        )}
      </div>

      <div {...stylex.props(styles.lead)}>
        {showStatus ? (
          <Badge variant={badgeVariants[cardTone]} label={statusLabel} xstyle={styles.status} />
        ) : lead !== undefined && lead.percent !== undefined ? (
          <span {...stylex.props(styles.num, numToneStyle)}>
            <strong {...stylex.props(styles.numStrong)}>{intl.formatNumber(lead.percent)}</strong>
            <span {...stylex.props(styles.numSuffix)} aria-hidden="true">
              %
            </span>
          </span>
        ) : lead !== undefined && lead.window.remaining !== undefined ? (
          <span {...stylex.props(styles.num, styles.numAmount, numToneStyle)}>
            {quotaValueLabel(lead.window, tr)}
          </span>
        ) : (
          <span {...stylex.props(styles.num, styles.numEmpty)}>—</span>
        )}

        <span {...stylex.props(styles.meta)}>
          {lead !== undefined ? (
            <>
              {quotaWindowLabel(lead.window, tr)}
              {lead.window.reset_at_ms !== undefined && (
                <>
                  {' · '}
                  {quotaWindowNeedsRefresh(lead.window, nowMs) ? (
                    <Tooltip content={quotaPeriodTooltip(lead.window, tr, nowMs) ?? ''}>
                      <span {...stylex.props(styles.pending)} tabIndex={0}>
                        {t('group.credentials.subscription.quotaPendingRefresh')}
                      </span>
                    </Tooltip>
                  ) : (
                    <>
                      {quotaResetPrefix !== '' && quotaResetPrefix}
                      <RelativeInstant
                        instant={lead.window.reset_at_ms}
                        emptyLabel={t('group.credentials.subscription.unknown')}
                        hint
                        tooltipContent={quotaPeriodTooltip(lead.window, tr, nowMs)}
                      />
                      {quotaResetSuffix !== '' && quotaResetSuffix}
                    </>
                  )}
                </>
              )}
            </>
          ) : (
            t('group.credentials.subscription.noQuota')
          )}
        </span>
      </div>

      {(quotaWindows.length > 1 || hasResetCredits) && (
        <div {...stylex.props(styles.resources)}>
          {quotaWindows.length > 1 ? (
            <div
              {...stylex.props(styles.strip)}
              role="group"
              aria-label={t('group.credentials.subscription.title')}
            >
              {quotaWindows.map((window) => {
                const percent = remainingQuotaPercent(window)
                const tone = quotaWindowTone(window)
                const tooltip = quotaTooltip(window, tr, nowMs)
                const fillStyle: CSSProperties | undefined =
                  percent === undefined ? undefined : { width: `${percent}%` }
                return (
                  <Tooltip key={window.id} content={tooltip}>
                    <span
                      {...stylex.props(
                        styles.seg,
                        tone !== undefined ? segTrackStyles[tone] : styles.segTrack,
                      )}
                      role={percent === undefined ? 'img' : 'progressbar'}
                      aria-label={tooltip}
                      aria-valuemin={percent === undefined ? undefined : 0}
                      aria-valuemax={percent === undefined ? undefined : 100}
                      aria-valuenow={percent}
                      tabIndex={0}
                    >
                      {percent !== undefined && (
                        <span
                          {...stylex.props(
                            styles.segFill,
                            tone !== undefined ? segFillStyles[tone] : styles.segFillUnknown,
                          )}
                          style={fillStyle}
                          aria-hidden="true"
                        />
                      )}
                    </span>
                  </Tooltip>
                )
              })}
            </div>
          ) : (
            <span {...stylex.props(styles.spacer)} aria-hidden="true" />
          )}

          {hasResetCredits && (
            <Tooltip content={creditsTooltip}>
              <span {...stylex.props(styles.credits)} tabIndex={0} aria-label={creditsTooltip}>
                {dots.map((dot) => (
                  <i
                    key={dot.index}
                    {...stylex.props(
                      styles.creditDot,
                      dot.tone !== 'default' ? creditDotStyles[dot.tone] : undefined,
                    )}
                  />
                ))}
                {resetCreditsAvailable > dots.length && (
                  <span {...stylex.props(styles.creditMore)} aria-hidden="true">
                    +
                  </span>
                )}
              </span>
            </Tooltip>
          )}
        </div>
      )}
    </article>
  )
}
