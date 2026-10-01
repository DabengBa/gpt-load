import * as stylex from '@stylexjs/stylex'
import { CircleAlert, TriangleAlert } from 'lucide-react'
import { useIntl } from 'react-intl'

import type { RuntimeHealthDto } from '@shared/control/resources/health'
import {
  attentionRowLimit,
  attentionTotal,
  collectAttentionItems,
  type AttentionItem,
} from '@shared/domain/home/attention'
import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../../app/i18n'
import { RouteLink } from '../../app/route-link'
import { RelativeInstant } from '../../components/RelativeInstant'

const styles = stylex.create({
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
    paddingTop: 14,
  },
  srOnly: {
    position: 'absolute',
    width: 1,
    height: 1,
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    gap: 9,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'transparent',
    borderRadius: 'var(--radius-control)',
    paddingTop: 8,
    paddingBottom: 8,
    paddingInline: 11,
    fontSize: 'var(--text-meta)',
    color: 'inherit',
    textDecoration: 'none',
    transitionProperty: 'border-color, background-color',
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
  },
  rowWarning: {
    borderColor: {
      default: 'color-mix(in srgb, var(--color-warning) 30%, var(--color-border-subtle))',
      ':hover': 'var(--color-warning)',
    },
    backgroundColor: 'var(--color-warning-bg)',
    color: 'var(--color-warning)',
  },
  rowDanger: {
    borderColor: {
      default: 'color-mix(in srgb, var(--color-danger) 30%, var(--color-border-subtle))',
      ':hover': 'var(--color-danger)',
    },
    backgroundColor: 'var(--color-danger-bg)',
    color: 'var(--color-danger)',
  },
  icon: {
    flex: 'none',
  },
  detail: {
    display: 'inline-flex',
    alignItems: 'baseline',
    flexWrap: 'wrap',
    gap: 5,
    minWidth: 0,
  },
  go: {
    marginLeft: 'auto',
    fontWeight: 650,
    whiteSpace: 'nowrap',
  },
})

function itemHref(item: AttentionItem): string {
  if (item.kind === 'blacklisted') {
    return `${pagePath('groups')}/${item.groupID}?tab=credentials&credential_status=blacklisted`
  }
  if (item.kind === 'billing') {
    return `${pagePath('groups')}/${item.groupID}?tab=credentials&credential_status=cooldown`
  }
  return `${pagePath('groups')}/${item.groupID}?tab=credentials`
}

function itemTone(item: AttentionItem, health: RuntimeHealthDto | null): 'danger' | 'warning' {
  if (item.kind === 'blacklisted' || item.kind === 'billing') return 'danger'
  if (
    item.kind === 'expiringResetCredit' &&
    item.expiresAtMS !== undefined &&
    health !== null &&
    item.expiresAtMS - health.observed_at_ms <= 24 * 60 * 60 * 1_000
  ) {
    return 'danger'
  }
  return 'warning'
}

// Conditional block: nothing actionable means no section at all — no heading,
// no border, no "all good" strip (classic HomeAttention).
export function HomeAttention({ health }: { health: RuntimeHealthDto | null }) {
  const intl = useIntl()
  const t = useT()

  const items =
    health === null
      ? []
      : collectAttentionItems(
          health.blacklisted_credentials,
          health.expiring_reset_credits,
          health.low_quota_credentials,
          health.cooldown_credentials,
        )
  if (items.length === 0) return null

  const overflowing = items.length > attentionRowLimit
  const total = attentionTotal(items)

  const quotaPercent = (remaining: number): string =>
    new Intl.NumberFormat(intl.locale, {
      style: 'percent',
      maximumFractionDigits: 0,
    }).format(remaining)

  if (overflowing) {
    return (
      <section {...stylex.props(styles.section)} aria-labelledby="home-attention-title">
        <h2 id="home-attention-title" {...stylex.props(styles.srOnly)}>
          {t('home.ledger.attention.title')}
        </h2>
        <RouteLink
          to={`${pagePath('monitor')}?tab=health`}
          {...stylex.props(styles.row, styles.rowDanger)}
        >
          <CircleAlert {...stylex.props(styles.icon)} size={14} aria-hidden="true" />
          <span>{t('home.ledger.attention.summary', { count: total })}</span>
          <span {...stylex.props(styles.go)} aria-hidden="true">
            {t('home.ledger.attention.action')} →
          </span>
        </RouteLink>
      </section>
    )
  }

  return (
    <section {...stylex.props(styles.section)} aria-labelledby="home-attention-title">
      <h2 id="home-attention-title" {...stylex.props(styles.srOnly)}>
        {t('home.ledger.attention.title')}
      </h2>
      {items.map((item) => {
        const tone = itemTone(item, health)
        return (
          <RouteLink
            key={`${item.kind}-${item.groupID}`}
            to={itemHref(item)}
            {...stylex.props(styles.row, tone === 'danger' ? styles.rowDanger : styles.rowWarning)}
          >
            {item.kind === 'blacklisted' || item.kind === 'billing' ? (
              <CircleAlert {...stylex.props(styles.icon)} size={14} aria-hidden="true" />
            ) : (
              <TriangleAlert {...stylex.props(styles.icon)} size={14} aria-hidden="true" />
            )}
            {item.kind === 'blacklisted' && (
              <span>
                {t('home.ledger.attention.blacklisted', {
                  group: item.groupName,
                  count: item.value,
                })}
              </span>
            )}
            {item.kind === 'billing' && (
              <span {...stylex.props(styles.detail)}>
                {t('home.ledger.attention.billing', {
                  group: item.groupName,
                  count: item.value,
                })}
                {item.cooldownUntilMS !== undefined && (
                  <RelativeInstant instant={item.cooldownUntilMS} emptyLabel="" />
                )}
              </span>
            )}
            {item.kind === 'expiringResetCredit' && (
              <span {...stylex.props(styles.detail)}>
                {t('home.ledger.attention.resetCreditExpiring', {
                  group: item.groupName,
                  count: item.value,
                })}
                <RelativeInstant instant={item.expiresAtMS ?? null} emptyLabel="" />
              </span>
            )}
            {item.kind === 'lowQuota' && (
              <span {...stylex.props(styles.detail)}>
                {t('home.ledger.attention.lowQuota', {
                  group: item.groupName,
                  remaining: quotaPercent(item.value),
                })}
                {item.resetAtMS !== undefined && (
                  <RelativeInstant instant={item.resetAtMS} emptyLabel="" />
                )}
              </span>
            )}
            <span {...stylex.props(styles.go)} aria-hidden="true">
              {t('home.ledger.attention.action')} →
            </span>
          </RouteLink>
        )
      })}
    </section>
  )
}
