import * as stylex from '@stylexjs/stylex'
import { Badge } from '@astryxdesign/core'
import { ArrowRight, CircleAlert, CircleOff, KeyRound } from 'lucide-react'
import { useIntl } from 'react-intl'

import type { HealthAccessKeyCostLimitDto } from '@shared/control/types'
import { formatUSD } from '@shared/lib/format'
import { pagePath } from '@shared/routing/page-routes'
import { defaultAccessKeyCollectionFilters } from '@shared/routing/access-key-collection-route'
import { serializeSettingsRouteQuery } from '@shared/routing/settings-route'

import { useT } from '../../app/i18n'
import { stringifySharedRouteSearch } from '../../app/search-codec'
import { RouteLink } from '../../app/route-link'
import { RelativeInstant } from '../../components/RelativeInstant'
import { MonitorSectionHeading } from './MonitorSectionHeading'

const WIDE = '@media (max-width: 900px)'

const styles = stylex.create({
  section: {
    display: 'grid',
    gap: 'var(--space-4)',
  },
  // Classic Surface variant="default" padded=false with the page's
  // `overflow: hidden` on the surface itself.
  surface: {
    overflow: 'hidden',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-card)',
    backgroundColor: 'var(--color-surface)',
  },
  row: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'minmax(180px, 0.8fr) minmax(240px, 1.4fr) minmax(220px, 1fr) auto',
      [WIDE]: 'minmax(0, 1fr)',
    },
    alignItems: 'center',
    gap: 'var(--space-4)',
    paddingBlock: 12,
    paddingInline: 14,
  },
  // Classic `article + article` sibling border — applied on index > 0 because
  // sibling combinators are not in the stylex allowlist.
  rowSibling: {
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
  },
  identity: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  // Classic `.access-key-limit-health__identity > svg`.
  identityIcon: {
    flex: '0 0 auto',
    color: 'var(--color-danger)',
  },
  identityText: {
    minWidth: 0,
  },
  identityName: {
    display: 'block',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  identityCode: {
    display: 'block',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  rules: {
    display: 'grid',
    gap: 3,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
    fontFamily: 'var(--font-mono)',
  },
  recovery: {
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  action: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    color: 'var(--color-action)',
    fontSize: 'var(--text-sm)',
    fontWeight: 600,
    borderRadius: 'var(--radius-control)',
    textDecorationLine: { default: 'none', ':hover': 'underline' },
    outlineWidth: { ':focus-visible': 2 },
    outlineStyle: { ':focus-visible': 'solid' },
    outlineColor: { ':focus-visible': 'var(--color-focus)' },
    outlineOffset: { ':focus-visible': 3 },
  },
})

function editHref(accessKeyID: number): string {
  const query = serializeSettingsRouteQuery('credentials', {
    collection: defaultAccessKeyCollectionFilters,
    drawer: { mode: 'edit', accessKeyID },
  })
  return `${pagePath('settings')}${stringifySharedRouteSearch(query)}`
}

/**
 * Access keys blocked by exhausted cost-limit rules — classic
 * AccessKeyCostLimitHealth.vue. The manage link opens the access-keys drawer
 * (`?action=edit&access_key_id=`), which the astryx collection route already
 * parses.
 */
export function AccessKeyCostLimitHealth({
  accessKeys,
}: {
  accessKeys: HealthAccessKeyCostLimitDto[]
}) {
  const t = useT()
  const intl = useIntl()

  function ruleLabel(rule: HealthAccessKeyCostLimitDto['blocking_rules'][number]): string {
    if (rule.kind === 'total') return t('monitor.health.accessKeyLimits.total')
    let period: string
    if (rule.period_seconds % 86_400 === 0) {
      period = t('monitor.health.accessKeyLimits.periodDays', {
        count: rule.period_seconds / 86_400,
      })
    } else if (rule.period_seconds % 3_600 === 0) {
      period = t('monitor.health.accessKeyLimits.periodHours', {
        count: rule.period_seconds / 3_600,
      })
    } else if (rule.period_seconds % 60 === 0) {
      period = t('monitor.health.accessKeyLimits.periodMinutes', {
        count: rule.period_seconds / 60,
      })
    } else {
      period = t('monitor.health.accessKeyLimits.periodSeconds', {
        count: rule.period_seconds,
      })
    }
    return t('monitor.health.accessKeyLimits.periodic', { period })
  }

  return (
    <section {...stylex.props(styles.section)} aria-labelledby="access-key-limit-health-title">
      <MonitorSectionHeading
        id="access-key-limit-health-title"
        title={t('monitor.health.accessKeyLimits.title')}
        description={t('monitor.health.accessKeyLimits.description')}
      />

      <div {...stylex.props(styles.surface)}>
        {accessKeys.map((accessKey, index) => {
          const StatusIcon = accessKey.recoverable ? CircleAlert : CircleOff
          return (
            <article
              key={accessKey.access_key_id}
              {...stylex.props(styles.row, index > 0 && styles.rowSibling)}
            >
              <div {...stylex.props(styles.identity)}>
                <KeyRound size={15} aria-hidden="true" {...stylex.props(styles.identityIcon)} />
                <div {...stylex.props(styles.identityText)}>
                  <strong {...stylex.props(styles.identityName)}>{accessKey.name}</strong>
                  <code {...stylex.props(styles.identityCode)}>{accessKey.masked_key}</code>
                </div>
              </div>

              <div {...stylex.props(styles.rules)}>
                {accessKey.blocking_rules.map((rule) => (
                  <span key={rule.id}>
                    {ruleLabel(rule)} · {formatUSD(rule.used_usd, intl.locale)} /{' '}
                    {formatUSD(rule.limit_usd, intl.locale)}
                  </span>
                ))}
              </div>

              <div {...stylex.props(styles.recovery)}>
                <Badge
                  variant={accessKey.recoverable ? 'warning' : 'error'}
                  icon={<StatusIcon size={12} aria-hidden="true" />}
                  label={
                    accessKey.recoverable
                      ? t('monitor.health.accessKeyLimits.temporary')
                      : t('monitor.health.accessKeyLimits.manual')
                  }
                />
                {accessKey.next_available_at_ms !== null ? (
                  <span>
                    {t('monitor.health.accessKeyLimits.availableAgain')}{' '}
                    <RelativeInstant
                      instant={accessKey.next_available_at_ms}
                      emptyLabel={t('monitor.health.accessKeyLimits.notAutomatic')}
                      hint
                    />
                  </span>
                ) : (
                  <span>{t('monitor.health.accessKeyLimits.notAutomatic')}</span>
                )}
              </div>

              <RouteLink to={editHref(accessKey.access_key_id)} {...stylex.props(styles.action)}>
                {t('monitor.health.accessKeyLimits.manage')}
                <ArrowRight size={14} aria-hidden="true" />
              </RouteLink>
            </article>
          )
        })}
      </div>
    </section>
  )
}
