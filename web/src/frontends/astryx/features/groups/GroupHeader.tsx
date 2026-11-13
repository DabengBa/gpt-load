import * as stylex from '@stylexjs/stylex'
import { Badge } from '@astryxdesign/core'
import { ArrowLeft } from 'lucide-react'

import type { GroupSummaryDto } from '@shared/control/resources/groups'
import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../../app/i18n'
import { RouteLink } from '../../app/route-link'

const styles = stylex.create({
  header: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 'var(--space-2)',
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-control)',
  },
  back: {
    display: 'inline-flex',
    width: 'fit-content',
    minHeight: { default: 'var(--control-sm)', '@media (max-width: 520px)': 'var(--touch-target)' },
    alignItems: 'center',
    gap: 6,
    color: {
      default: 'var(--color-text-faint)',
      ':hover': 'var(--color-action)',
    },
    fontSize: 'var(--text-meta)',
  },
  body: {
    display: 'flex',
    flex: '1 1 640px',
    flexWrap: 'wrap',
    minWidth: 0,

    alignItems: 'center',
    gap: 'var(--space-2) var(--space-3)',
    paddingTop: 0,
    paddingBottom: 'var(--space-2)',
  },
  topline: {
    display: 'flex',
    flexGrow: 1,
    minWidth: 0,

    alignItems: 'center',
  },
  title: {
    display: 'flex',
    minWidth: 0,
    flexWrap: 'wrap',
    alignItems: { default: 'center', '@media (max-width: 520px)': 'flex-start' },
    flexDirection: 'row',
    gap: { default: '10px', '@media (max-width: 520px)': '7px' },
  },
  h1: {
    maxWidth: 'none',
    margin: 0,
    fontSize: '24px',
    fontWeight: 650,
    letterSpacing: 0,
    lineHeight: 1.35,
    overflowWrap: 'anywhere',
  },
  statusReason: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
  },
  details: {
    display: 'flex',
    minWidth: 0,

    flexWrap: 'wrap',

    alignItems: 'center',
    justifyContent: { default: 'flex-end', '@media (max-width: 1000px)': 'flex-start' },
    gap: { default: '7px 12px', '@media (max-width: 520px)': '8px 12px' },
    marginTop: 0,
  },

  actionsTarget: {
    display: 'inline-flex',
    minHeight: '24px',
    alignItems: 'center',
  },
})

function serviceBadgeVariant(
  status: GroupSummaryDto['service_status'],
): 'success' | 'error' | 'neutral' {
  return status === 'available' ? 'success' : status === 'unavailable' ? 'error' : 'neutral'
}

export function GroupHeader({ group }: { group: GroupSummaryDto }) {
  const t = useT()

  return (
    <header {...stylex.props(styles.header)}>
      <RouteLink to={pagePath('groups')} {...stylex.props(styles.back)}>
        <ArrowLeft size={16} aria-hidden="true" />
        {t('group.backToGroups')}
      </RouteLink>
      <div {...stylex.props(styles.body)}>
        <div {...stylex.props(styles.topline)}>
          <div {...stylex.props(styles.title)}>
            <h1 id="group-detail-title" {...stylex.props(styles.h1)}>
              {group.name}
            </h1>
            <Badge
              variant={serviceBadgeVariant(group.service_status)}
              label={t(`groups.collection.status.${group.service_status}`)}
            />
            {group.service_status === 'unavailable' && group.service_status_reason && (
              <span {...stylex.props(styles.statusReason)}>
                {t(`groups.collection.statusReason.${group.service_status_reason}`)}
              </span>
            )}
          </div>
        </div>
        <div {...stylex.props(styles.details)}>
          {/* Unified-mode portal target: GroupSettingsBaseForm teleports its
              enable switch here via createPortal. The id is contractual. */}
          <div id="group-header-actions" {...stylex.props(styles.actionsTarget)} />
        </div>
      </div>
    </header>
  )
}
