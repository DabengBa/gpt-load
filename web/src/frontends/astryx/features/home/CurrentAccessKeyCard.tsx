import * as stylex from '@stylexjs/stylex'
import { Badge, Tooltip } from '@astryxdesign/core'
import { Gauge, KeyRound, LockKeyhole } from 'lucide-react'
import { useIntl } from 'react-intl'

import type {
  AccessKeyCollectionItemDto,
  AccessKeyCostLimitRuleStatusDto,
} from '@shared/control/types'
import type { MessageId } from '@shared/i18n/message-ids'
import { formatInteger, formatLocalInstant, formatUSD } from '@shared/lib/format'
import { quotaProgressTone } from '@shared/lib/quota-progress'

import { useT } from '../../app/i18n'
import { CostLimitWindowTime } from './CostLimitWindowTime'

const NARROW = '@media (max-width: 860px)'
const TIGHT = '@media (max-width: 560px)'
const REDUCED_MOTION = '@media (prefers-reduced-motion: reduce)'

const styles = stylex.create({
  section: {
    display: 'grid',
    gap: 14,
    marginTop: 'var(--space-4)',
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 18,
    paddingBottom: 18,
  },
  header: {
    display: 'flex',
    alignItems: {
      default: 'center',
      [TIGHT]: 'flex-start',
    },
    flexDirection: {
      default: 'row',
      [TIGHT]: 'column',
    },
    justifyContent: 'space-between',
    gap: 'var(--space-4)',
  },
  title: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 10,
  },
  titleIcon: {
    color: 'var(--color-action)',
  },
  eyebrow: {
    margin: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  name: {
    margin: 0,
    marginTop: 2,
    fontFamily: 'var(--font-serif)',
    fontSize: 'var(--text-lg)',
    fontWeight: 550,
  },
  identity: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  maskedKey: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  boundary: {
    display: 'flex',
    alignItems: 'center',
    gap: 7,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  boundaryIcon: {
    flex: '0 0 auto',
    color: 'var(--color-text-faint)',
  },
  facts: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'repeat(6, minmax(0, 1fr))',
      [NARROW]: 'repeat(2, minmax(0, 1fr))',
      [TIGHT]: 'minmax(0, 1fr)',
    },
    margin: 0,
    gap: 1,
    backgroundColor: 'var(--color-border-subtle)',
  },
  fact: {
    minWidth: 0,
    backgroundColor: 'var(--color-surface)',
    paddingTop: 10,
    paddingBottom: 10,
    paddingInline: 12,
  },
  factTerm: {
    margin: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  factValue: {
    display: 'block',
    minWidth: 0,
    overflow: 'hidden',
    margin: '5px 0 0',
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  limits: {
    display: 'grid',
    gap: 'var(--space-3)',
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 'var(--space-4)',
  },
  limitsHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
  },
  limitsTitle: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  limitsHeading: {
    margin: 0,
    fontSize: 'var(--text-meta)',
  },
  limitList: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'repeat(2, minmax(0, 1fr))',
      [TIGHT]: 'minmax(0, 1fr)',
    },
    gap: 'var(--space-3)',
  },
  limitCard: {
    display: 'grid',
    gap: 'var(--space-2)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'color-mix(in srgb, var(--color-action) 24%, var(--color-border-subtle))',
    borderLeftWidth: 3,
    borderLeftStyle: 'solid',
    borderLeftColor: 'var(--color-success)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'color-mix(in srgb, var(--color-action-soft) 46%, var(--color-surface))',
    padding: 'var(--space-3)',
    boxShadow:
      '0 1px 2px color-mix(in srgb, var(--color-action) 14%, transparent), 0 8px 24px color-mix(in srgb, var(--color-action) 10%, transparent)',
  },
  limitCardWarning: {
    borderLeftColor: 'var(--color-warning)',
  },
  limitCardDanger: {
    borderColor: 'var(--color-feedback-danger-border)',
    borderLeftColor: 'var(--color-danger)',
    backgroundColor: 'color-mix(in srgb, var(--color-danger-bg) 82%, var(--color-surface))',
    boxShadow:
      '0 1px 2px color-mix(in srgb, var(--color-danger) 14%, transparent), 0 8px 24px color-mix(in srgb, var(--color-danger) 9%, transparent)',
  },
  limitHeading: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-2)',
  },
  limitProgress: {
    position: 'relative',
    display: 'grid',
    minHeight: 34,
    gridTemplateColumns: 'minmax(0, 1fr) auto',
    alignItems: 'center',
    gap: 'var(--space-3)',
    overflow: 'hidden',
    borderRadius: 6,
    paddingTop: 7,
    paddingBottom: 7,
    paddingRight: 10,
    paddingLeft: 13,
  },
  limitProgressSuccess: {
    backgroundColor: 'light-dark(#dcfeea, #112b21)',
    boxShadow: 'inset 3px 0 0 oklch(70% 0.16 158)',
  },
  limitProgressWarning: {
    backgroundColor: 'light-dark(#fff2e2, #302212)',
    boxShadow: 'inset 3px 0 0 oklch(75% 0.152 75)',
  },
  limitProgressDanger: {
    backgroundColor: 'light-dark(#fef0f0, #371a1d)',
    boxShadow: 'inset 3px 0 0 oklch(65% 0.2 22)',
  },
  limitMeter: {
    position: 'absolute',
    zIndex: 0,
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    overflow: 'hidden',
    borderRadius: 'inherit',
    pointerEvents: 'none',
  },
  limitFillBase: { backgroundColor: 'var(--color-border-control)' },
  limitFillSuccess: { backgroundColor: 'oklch(70% 0.16 158)' },
  limitFillWarning: { backgroundColor: 'oklch(75% 0.152 75)' },
  limitFillDanger: { backgroundColor: 'oklch(65% 0.2 22)' },
  limitFill: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    height: 3,
    transitionProperty: {
      default: 'width',
      [REDUCED_MOTION]: 'none',
    },
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
  },
  limitUsage: {
    position: 'relative',
    zIndex: 1,
    minWidth: 0,
    overflow: 'hidden',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  limitPercent: {
    position: 'relative',
    zIndex: 1,
    minWidth: 36,
    color: 'var(--color-text)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
    fontVariantNumeric: 'tabular-nums',
    fontWeight: 650,
    textAlign: 'right',
  },
  limitRecovery: {
    display: 'flex',
    minHeight: 20,
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 4,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
  },
})

function remainingPercent(rule: AccessKeyCostLimitRuleStatusDto): number {
  if (rule.status === 'inactive') return 100
  const limit = Number(rule.limit_usd)
  const remaining = Number(rule.remaining_usd)
  if (!Number.isFinite(limit) || !Number.isFinite(remaining) || limit <= 0) return 0
  return Math.round(Math.max(0, Math.min(100, (remaining / limit) * 100)))
}

const limitCardTones = {
  warning: styles.limitCardWarning,
  danger: styles.limitCardDanger,
} as const

const limitProgressStyles = {
  success: styles.limitProgressSuccess,
  warning: styles.limitProgressWarning,
  danger: styles.limitProgressDanger,
} as const

const limitFillStyles = {
  success: styles.limitFillSuccess,
  warning: styles.limitFillWarning,
  danger: styles.limitFillDanger,
} as const

const ruleStateVariants = {
  available: 'success',
  inactive: 'neutral',
  exhausted: 'error',
} as const

export function CurrentAccessKeyCard({ accessKey }: { accessKey: AccessKeyCollectionItemDto }) {
  const intl = useIntl()
  const t = useT()

  const rpm =
    accessKey.rpm_limit === 0
      ? t('home.ledger.currentAccessKey.unlimited')
      : t('home.ledger.currentAccessKey.rpmValue', {
          count: formatInteger(accessKey.rpm_limit, intl.locale),
        })
  const protocols =
    accessKey.filters.protocols.length === 0
      ? t('home.ledger.currentAccessKey.allProtocols')
      : accessKey.filters.protocols.join(', ')
  const groups =
    accessKey.filters.groups.length === 0
      ? t('home.ledger.currentAccessKey.allGroups')
      : accessKey.filters.groups.map((id) => `#${id}`).join(', ')
  const models =
    accessKey.filters.models.length === 0
      ? t('home.ledger.currentAccessKey.allModels')
      : accessKey.filters.models.join(', ')
  const costLimits = accessKey.cost_limit_status

  function periodLabel(seconds: number): string {
    if (seconds % 86_400 === 0) {
      return t('home.ledger.currentAccessKey.costLimits.periodDays', {
        count: seconds / 86_400,
      })
    }
    if (seconds % 3_600 === 0) {
      return t('home.ledger.currentAccessKey.costLimits.periodHours', {
        count: seconds / 3_600,
      })
    }
    if (seconds % 60 === 0) {
      return t('home.ledger.currentAccessKey.costLimits.periodMinutes', {
        count: seconds / 60,
      })
    }
    return t('home.ledger.currentAccessKey.costLimits.periodSeconds', { count: seconds })
  }

  function ruleLabel(kind: 'total' | 'periodic', periodSeconds: number): string {
    return kind === 'total'
      ? t('home.ledger.currentAccessKey.costLimits.total')
      : t('home.ledger.currentAccessKey.costLimits.periodic', {
          period: periodLabel(periodSeconds),
        })
  }

  function ruleUsageLabel(rule: AccessKeyCostLimitRuleStatusDto): string {
    return t('home.ledger.currentAccessKey.costLimits.usage', {
      used: formatUSD(rule.used_usd, intl.locale),
      limit: formatUSD(rule.limit_usd, intl.locale),
      remaining: formatUSD(rule.remaining_usd, intl.locale),
    })
  }

  function ruleTone(rule: AccessKeyCostLimitRuleStatusDto): 'success' | 'warning' | 'danger' {
    return quotaProgressTone(remainingPercent(rule), rule.status === 'exhausted')
  }

  return (
    <section {...stylex.props(styles.section)} aria-labelledby="current-access-key-title">
      <header {...stylex.props(styles.header)}>
        <div {...stylex.props(styles.title)}>
          <KeyRound {...stylex.props(styles.titleIcon)} size={16} aria-hidden="true" />
          <div>
            <p {...stylex.props(styles.eyebrow)}>{t('home.ledger.currentAccessKey.eyebrow')}</p>
            <h2 id="current-access-key-title" {...stylex.props(styles.name)}>
              {accessKey.name}
            </h2>
          </div>
        </div>
        <div {...stylex.props(styles.identity)}>
          <Badge variant="success" label={t('home.ledger.currentAccessKey.active')} />
          <code {...stylex.props(styles.maskedKey)}>{accessKey.masked_key}</code>
        </div>
      </header>

      <div {...stylex.props(styles.boundary)}>
        <LockKeyhole {...stylex.props(styles.boundaryIcon)} size={14} aria-hidden="true" />
        <span>{t('home.ledger.currentAccessKey.readOnly')}</span>
      </div>

      <dl {...stylex.props(styles.facts)}>
        <div {...stylex.props(styles.fact)}>
          <dt {...stylex.props(styles.factTerm)}>{t('home.ledger.currentAccessKey.rpm')}</dt>
          <dd {...stylex.props(styles.factValue)}>{rpm}</dd>
        </div>
        <div {...stylex.props(styles.fact)}>
          <dt {...stylex.props(styles.factTerm)}>{t('home.ledger.currentAccessKey.protocols')}</dt>
          <dd {...stylex.props(styles.factValue)}>
            <Tooltip content={protocols}>
              <span>{protocols}</span>
            </Tooltip>
          </dd>
        </div>
        <div {...stylex.props(styles.fact)}>
          <dt {...stylex.props(styles.factTerm)}>{t('home.ledger.currentAccessKey.groups')}</dt>
          <dd {...stylex.props(styles.factValue)}>
            <Tooltip content={groups}>
              <span>{groups}</span>
            </Tooltip>
          </dd>
        </div>
        <div {...stylex.props(styles.fact)}>
          <dt {...stylex.props(styles.factTerm)}>{t('home.ledger.currentAccessKey.models')}</dt>
          <dd {...stylex.props(styles.factValue)}>
            <Tooltip content={models}>
              <span>{models}</span>
            </Tooltip>
          </dd>
        </div>
        <div {...stylex.props(styles.fact)}>
          <dt {...stylex.props(styles.factTerm)}>
            {t('home.ledger.currentAccessKey.lastRequest')}
          </dt>
          <dd {...stylex.props(styles.factValue)}>
            {accessKey.last_request_at_ms !== null ? (
              <time dateTime={new Date(accessKey.last_request_at_ms).toISOString()}>
                {formatLocalInstant(accessKey.last_request_at_ms, intl.locale)}
              </time>
            ) : (
              <span>{t('home.ledger.currentAccessKey.neverRequested')}</span>
            )}
          </dd>
        </div>
      </dl>

      {costLimits !== null && costLimits.rules.length > 0 && (
        <section {...stylex.props(styles.limits)} aria-labelledby="cost-limits-title">
          <header {...stylex.props(styles.limitsHeader)}>
            <div {...stylex.props(styles.limitsTitle)}>
              <Gauge size={15} aria-hidden="true" />
              <h3 id="cost-limits-title" {...stylex.props(styles.limitsHeading)}>
                {t('home.ledger.currentAccessKey.costLimits.title')}
              </h3>
            </div>
            <Badge
              variant={costLimits.allowed ? 'success' : 'error'}
              label={t(
                costLimits.allowed
                  ? 'home.ledger.currentAccessKey.costLimits.available'
                  : 'home.ledger.currentAccessKey.costLimits.blocked',
              )}
            />
          </header>

          <div {...stylex.props(styles.limitList)}>
            {costLimits.rules.map((rule) => {
              const tone = ruleTone(rule)
              const percent = remainingPercent(rule)
              const label = ruleLabel(rule.kind, rule.period_seconds)
              const usage = ruleUsageLabel(rule)
              return (
                <article
                  key={rule.id}
                  {...stylex.props(
                    styles.limitCard,
                    tone !== 'success' ? limitCardTones[tone] : undefined,
                  )}
                >
                  <div {...stylex.props(styles.limitHeading)}>
                    <strong>{label}</strong>
                    <Badge
                      variant={ruleStateVariants[rule.status]}
                      label={t(`accessKeys.costLimits.status.${rule.status}` as MessageId)}
                    />
                  </div>
                  <div
                    {...stylex.props(
                      styles.limitProgress,
                      tone === 'success' ? styles.limitProgressSuccess : limitProgressStyles[tone],
                    )}
                  >
                    <span
                      {...stylex.props(styles.limitMeter)}
                      role="progressbar"
                      aria-label={label}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={percent}
                      aria-valuetext={t('accessKeys.costLimits.remainingPercent', {
                        value: intl.formatNumber(percent),
                      })}
                    >
                      <span
                        {...stylex.props(styles.limitFill, limitFillStyles[tone])}
                        style={{ width: `${percent}%` }}
                        aria-hidden="true"
                      />
                    </span>
                    <Tooltip content={usage}>
                      <span {...stylex.props(styles.limitUsage)}>{usage}</span>
                    </Tooltip>
                    <strong {...stylex.props(styles.limitPercent)}>
                      {intl.formatNumber(percent)}%
                    </strong>
                  </div>
                  <div {...stylex.props(styles.limitRecovery)}>
                    {rule.kind === 'periodic' ? (
                      <>
                        {t(
                          rule.status === 'inactive'
                            ? 'home.ledger.currentAccessKey.costLimits.previewEndsAt'
                            : rule.status === 'exhausted'
                              ? 'home.ledger.currentAccessKey.costLimits.availableAgain'
                              : 'home.ledger.currentAccessKey.costLimits.resetsAt',
                        )}{' '}
                        <CostLimitWindowTime rule={rule} />
                      </>
                    ) : (
                      rule.status === 'exhausted' &&
                      t('home.ledger.currentAccessKey.costLimits.notAutomatic')
                    )}
                  </div>
                </article>
              )
            })}
          </div>
        </section>
      )}
    </section>
  )
}
