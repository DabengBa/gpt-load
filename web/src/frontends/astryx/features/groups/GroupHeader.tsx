import * as stylex from '@stylexjs/stylex'
import { Badge } from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, ExternalLink } from 'lucide-react'

import type { GroupSummaryDto } from '@shared/control/resources/groups'
import { channelsQueryOptions } from '@shared/control/resources/channels'
import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../../app/i18n'
import { RouteLink } from '../../app/route-link'
import { useAppServices } from '../../app/services'
import { ChannelIcon } from '../../components/ChannelIcon'
import { CopyChip } from '../access-keys/AccessKeyCopyChip'

const styles = stylex.create({
  header: {
    display: 'grid',
    gap: 'var(--space-2)',
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-control)',
  },
  back: {
    display: 'inline-flex',
    width: 'fit-content',
    minHeight: { default: '38px', '@media (max-width: 520px)': 'var(--touch-target)' },
    alignItems: 'center',
    gap: 6,
    color: {
      default: 'var(--color-text-faint)',
      ':hover': 'var(--color-action)',
    },
    fontSize: 'var(--text-meta)',
  },
  body: {
    display: 'grid',
    minWidth: 0,
    gridTemplateColumns: {
      default: 'minmax(0, 1fr) minmax(0, auto)',
      '@media (max-width: 1000px)': 'minmax(0, 1fr)',
    },
    alignItems: 'center',
    columnGap: 'var(--space-5)',
    paddingTop: 'var(--space-2)',
    paddingBottom: { default: '16px', '@media (max-width: 520px)': '18px' },
  },
  topline: {
    display: 'flex',
    minWidth: 0,
    gridColumn: '1',
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
    gridColumn: { default: '2', '@media (max-width: 1000px)': '1' },
    gridRow: { default: '1', '@media (max-width: 1000px)': 'auto' },
    flexWrap: 'wrap',
    flexDirection: { default: 'row', '@media (max-width: 520px)': 'row' },
    alignItems: 'center',
    justifyContent: { default: 'flex-end', '@media (max-width: 1000px)': 'flex-start' },
    gap: { default: '7px 12px', '@media (max-width: 520px)': '8px 12px' },
    marginTop: { default: 0, '@media (max-width: 1000px)': 'var(--space-3)' },
  },
  metaTag: {
    display: 'inline-flex',
    minHeight: '24px',
    alignItems: 'center',
    gap: 'var(--space-1)',
    maxWidth: '100%',
    overflowWrap: 'anywhere',
    backgroundColor: 'var(--color-surface-sunken)',
    paddingBlock: '3px',
    paddingInline: '7px',
    fontSize: 'var(--text-label-xs)',
  },
  channelIcon: {
    flexShrink: 0,
    fontSize: '15px',
  },
  actionsTarget: {
    display: 'inline-flex',
    minHeight: '24px',
    alignItems: 'center',
  },
  providerLink: {
    display: 'inline-flex',
    minHeight: {
      default: 'var(--control-compact)',
      '@media (max-width: 860px)': 'var(--touch-target)',
    },
    alignItems: 'center',
    gap: 4,
    borderRadius: 'var(--radius-tag)',
    color: { default: 'var(--color-text-muted)', ':hover': 'var(--color-action)' },
    backgroundColor: { default: 'transparent', ':hover': 'var(--color-surface-sunken)' },
    paddingBlock: '3px',
    paddingInline: '7px',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 560,
    transitionProperty: 'color, background-color',
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
  },
})

function serviceBadgeVariant(
  status: GroupSummaryDto['service_status'],
): 'success' | 'error' | 'neutral' {
  return status === 'available' ? 'success' : status === 'unavailable' ? 'error' : 'neutral'
}

export function GroupHeader({ group }: { group: GroupSummaryDto }) {
  const t = useT()
  const { apiClient } = useAppServices()
  const channelsQuery = useQuery(channelsQueryOptions(apiClient, ''))
  const channel = channelsQuery.data?.items.find(
    ({ channel_id }) => channel_id === group.channel_id,
  )
  const channelName = channel?.name.trim() || group.channel_id

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
          <span {...stylex.props(styles.metaTag)}>
            {channel && <ChannelIcon icon={channel.icon} mark={channel.mark} />}
            <span>{channelName}</span>
          </span>
          {/* Unified-mode portal target: GroupSettingsBaseForm teleports its
              enable switch here via createPortal. The id is contractual. */}
          <div id="group-header-actions" {...stylex.props(styles.actionsTarget)} />
          {group.provider_url && (
            <a
              {...stylex.props(styles.providerLink)}
              href={group.provider_url}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={t('group.openProviderUrl', { url: group.provider_url })}
            >
              <ExternalLink size={13} aria-hidden="true" />
              <span>{t('group.settings.base.providerUrl')}</span>
            </a>
          )}
          {group.params.base_url && (
            <CopyChip
              value={group.params.base_url}
              label={t('group.copyUpstreamUrl', { url: group.params.base_url })}
              successLabel={t('group.copySuccess')}
              failureLabel={t('group.copyFailure')}
            />
          )}
        </div>
      </div>
    </header>
  )
}
