import * as stylex from '@stylexjs/stylex'
import { useIntl } from 'react-intl'

import type { HomeBaseDto } from '@shared/control/resources/home'
import type { ReleaseUpdateDto } from '@shared/control/resources/system-update'
import {
  formatDuration,
  formatInteger,
  formatLocalInstant,
  formatLocalTime,
} from '@shared/lib/format'

import { useT } from '../../app/i18n'
import { HomeReleaseUpdateLink } from './home-chrome'

const NARROW = '@media (max-width: 860px)'

const styles = stylex.create({
  header: {
    display: 'flex',
    alignItems: {
      default: 'center',
      [NARROW]: 'start',
    },
    justifyContent: 'space-between',
    gap: 22,
    flexWrap: 'wrap',
    // No bottom border: separators are always owned by the next section's
    // top border, and spacing comes from that section's margin — otherwise a
    // quiet page would show a double rule here.
  },
  facts: {
    maxWidth: 'none',
    margin: 0,
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-serif)',
    fontSize: 'var(--title-lede)',
    fontWeight: 500,
    lineHeight: 'var(--line-compact)',
    letterSpacing: '-0.015em',
  },
  factStrong: {
    color: 'var(--color-text)',
    fontWeight: 650,
  },
  factLabel: {
    color: 'var(--color-text-muted)',
    fontWeight: 500,
  },
  separator: {
    color: 'var(--color-text-muted)',
    fontWeight: 500,
  },
  stamp: {
    display: 'grid',
    gridTemplateColumns: 'max-content max-content',
    gap: '5px 1ch',
    margin: 0,
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
    lineHeight: 'var(--line-compact)',
    whiteSpace: 'nowrap',
    justifyContent: {
      default: 'end',
      [NARROW]: 'start',
    },
  },
  stampRow: {
    display: 'grid',
    gridColumn: '1 / -1',
    gridTemplateColumns: 'subgrid',
    alignItems: 'baseline',
  },
  stampTerm: {
    margin: 0,
    textAlign: 'right',
  },
  stampValue: {
    margin: 0,
    color: 'var(--color-text-muted)',
    fontVariantNumeric: 'tabular-nums',
    fontWeight: 500,
    textAlign: 'left',
  },
  version: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 'var(--space-1)',
  },
})

export function HomeSummary({
  base,
  update,
  observedAtMs,
  uptimeNowMs,
}: {
  base: HomeBaseDto
  update: ReleaseUpdateDto | null
  observedAtMs: number | null
  uptimeNowMs: number
}) {
  const intl = useIntl()
  const t = useT()
  const updated = observedAtMs === null ? '—' : formatLocalTime(observedAtMs, intl.locale)
  const updatedTitle =
    observedAtMs === null ? undefined : formatLocalInstant(observedAtMs, intl.locale)

  return (
    <header {...stylex.props(styles.header)}>
      <div>
        <h1 id="home-title" {...stylex.props(styles.facts)}>
          <span>
            <strong {...stylex.props(styles.factStrong)}>
              {formatInteger(base.inventory.group_count, intl.locale)}
            </strong>{' '}
            <span {...stylex.props(styles.factLabel)}>{t('home.ledger.factGroups')}</span>
          </span>
          <span {...stylex.props(styles.separator)} aria-hidden="true">
            {' '}
            ·{' '}
          </span>
          <span>
            <strong {...stylex.props(styles.factStrong)}>
              {formatInteger(base.inventory.available_credential_count, intl.locale)}/
              {formatInteger(base.inventory.credential_count, intl.locale)}
            </strong>{' '}
            <span {...stylex.props(styles.factLabel)}>
              {t('home.ledger.factAvailableCredentials')}
            </span>
          </span>
          <span {...stylex.props(styles.separator)} aria-hidden="true">
            {' '}
            ·{' '}
          </span>
          <span>
            <strong {...stylex.props(styles.factStrong)}>
              {formatInteger(base.inventory.model_count, intl.locale)}
            </strong>{' '}
            <span {...stylex.props(styles.factLabel)}>{t('home.ledger.factModels')}</span>
          </span>
        </h1>
      </div>
      <dl {...stylex.props(styles.stamp)}>
        <div {...stylex.props(styles.stampRow)}>
          <dt {...stylex.props(styles.stampTerm)}>{t('home.ledger.updated')}</dt>
          <dd {...stylex.props(styles.stampValue)} title={updatedTitle}>
            {updated}
          </dd>
        </div>
        <div {...stylex.props(styles.stampRow)}>
          <dt {...stylex.props(styles.stampTerm)}>{t('home.ledger.version')}</dt>
          <dd {...stylex.props(styles.stampValue)}>
            <span {...stylex.props(styles.version)}>
              <span>{base.version}</span>
              {update !== null && (
                <HomeReleaseUpdateLink currentVersion={base.version} update={update} />
              )}
            </span>
          </dd>
        </div>
        <div {...stylex.props(styles.stampRow)}>
          <dt {...stylex.props(styles.stampTerm)}>{t('home.ledger.uptime')}</dt>
          <dd {...stylex.props(styles.stampValue)}>
            {formatDuration(base.started_at_ms, uptimeNowMs, intl.locale)}
          </dd>
        </div>
      </dl>
    </header>
  )
}
