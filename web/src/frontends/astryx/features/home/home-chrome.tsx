import * as stylex from '@stylexjs/stylex'
import { Tooltip } from '@astryxdesign/core'
import { CircleArrowUp } from 'lucide-react'
import type { ReactNode } from 'react'
import { useIntl } from 'react-intl'

import type { ReleaseUpdateDto } from '@shared/control/resources/system-update'
import { formatLocalInstant } from '@shared/lib/format'

import { useT } from '../../app/i18n'

const styles = stylex.create({
  heading: {
    display: 'flex',
    minWidth: 0,
    minHeight: 26,
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
    flexWrap: 'wrap',
  },
  title: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 'var(--space-2-5)',
    margin: 0,
    fontSize: 'var(--title-section)',
    fontWeight: 650,
    letterSpacing: '-0.01em',
    lineHeight: 'var(--line-compact)',
  },
  dot: {
    flex: '0 0 auto',
    width: 6,
    height: 6,
    borderRadius: '50%',
    backgroundColor: 'var(--color-action)',
  },
  actions: {
    display: 'flex',
    flex: '0 0 auto',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  updateLink: {
    display: 'inline-flex',
    width: 24,
    height: 24,
    flex: '0 0 auto',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: '50%',
    color: {
      default: 'var(--color-warning)',
      ':hover': 'var(--color-text)',
    },
    transitionProperty: 'color, background-color',
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
    backgroundColor: { ':hover': 'var(--color-warning-bg)' },
  },
})

// Section heading with the same "dot + bold" grammar as the monitor page's
// section titles (classic HomeSectionHeading).
export function HomeSectionHeading({
  id,
  title,
  actions,
}: {
  id?: string
  title: string
  actions?: ReactNode
}) {
  return (
    <header {...stylex.props(styles.heading)}>
      <h2 id={id} {...stylex.props(styles.title)}>
        <span {...stylex.props(styles.dot)} aria-hidden="true" />
        {title}
      </h2>
      {actions !== undefined && <div {...stylex.props(styles.actions)}>{actions}</div>}
    </header>
  )
}

export function HomeReleaseUpdateLink({
  currentVersion,
  update,
}: {
  currentVersion: string
  update: ReleaseUpdateDto
}) {
  const intl = useIntl()
  const t = useT()
  const label = t('home.ledger.updateAvailableLabel', { version: update.version })
  const tooltip = t('home.ledger.updateAvailableTooltip', {
    current: currentVersion,
    latest: update.version,
    published: formatLocalInstant(update.published_at_ms, intl.locale),
  })
  return (
    <Tooltip content={tooltip}>
      <a
        {...stylex.props(styles.updateLink)}
        href={update.release_url}
        target="_blank"
        rel="noopener noreferrer"
        aria-label={label}
      >
        <CircleArrowUp size={14} strokeWidth={2} aria-hidden="true" />
      </a>
    </Tooltip>
  )
}
