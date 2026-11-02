import * as stylex from '@stylexjs/stylex'
import {
  Badge,
  IconButton,
  Switch,
  Table,
  Tooltip,
  pixel,
  proportional,
  useTablePagination,
  useTableStickyColumns,
  type TableColumn,
} from '@astryxdesign/core'
import { useNavigate } from '@tanstack/react-router'
import { ArrowRight, Copy, ExternalLink, KeyRound } from 'lucide-react'
import { useMemo, type ReactNode } from 'react'
import { useIntl } from 'react-intl'

import type { MessageId } from '@shared/i18n/message-ids'

import type {
  CredentialStatus,
  GroupCollectionItemDto,
  GroupCollectionStatus,
  GroupCollectionPaginationDto,
} from '@shared/control/types'
import type { ChannelDto } from '@shared/control/resources/channels'

import { useT } from '../../app/i18n'
import { RouteLink } from '../../app/route-link'
import { useMediaQuery } from '../../app/use-media-query'
import { ChannelIcon } from '../../components/ChannelIcon'
import { GroupDeleteDialog } from './settings/GroupDeleteDialog'
import { groupDetailHref } from './group-detail-href'

type GroupRow = GroupCollectionItemDto & Record<string, unknown>

const statusLabelIds: Record<GroupCollectionStatus, MessageId> = {
  available: 'groups.collection.status.available',
  unavailable: 'groups.collection.status.unavailable',
  disabled: 'groups.collection.status.disabled',
}

const styles = stylex.create({
  tableWrap: {
    marginTop: 'var(--space-1)',
    minWidth: 0,
  },
  nameCell: {
    minWidth: 0,
  },
  nameLink: {
    color: 'var(--color-action)',
    fontWeight: 600,
    textDecoration: { default: 'none', ':hover': 'underline' },
    display: 'inline-block',
    maxWidth: '100%',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    verticalAlign: 'bottom',
    textUnderlineOffset: 3,
  },
  statusCell: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  },
  channelCell: {
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
    minWidth: 0,
  },
  channelHeading: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    minWidth: 0,
    flexWrap: 'wrap',
  },
  channelName: {
    fontWeight: 600,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  providerLink: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    color: 'var(--color-text-muted)',
    fontSize: 11,
    textDecoration: 'none',
  },
  modelCount: {
    fontVariantNumeric: 'tabular-nums',
    fontWeight: 600,
  },
  modelTotal: {
    color: 'var(--color-text-muted)',
    fontWeight: 400,
  },
  credentialCell: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    minWidth: 0,
    maxWidth: '100%',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  credentialDot: {
    width: 8,
    height: 8,
    borderRadius: 999,
    backgroundColor: 'var(--color-neutral-fg)',
    flexShrink: 0,
  },
  credentialDotSuccess: { backgroundColor: 'var(--color-success)' },
  credentialDotDanger: { backgroundColor: 'var(--color-danger)' },
  credentialDotWarning: { backgroundColor: 'var(--color-warning)' },
  actionsCell: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    justifyContent: 'flex-end',
  },
})

function groupEnabled(
  group: GroupCollectionItemDto,
  optimistic: ReadonlyMap<number, boolean>,
): boolean {
  return optimistic.get(group.id) ?? group.status !== 'disabled'
}

function statusBadgeVariant(status: GroupCollectionStatus): 'success' | 'error' | 'neutral' {
  return status === 'available' ? 'success' : status === 'unavailable' ? 'error' : 'neutral'
}

function credentialDotStyle(status: CredentialStatus | null, configured: boolean) {
  if (configured && status === 'available') return styles.credentialDotSuccess
  if (status === 'blacklisted') return styles.credentialDotDanger
  if (status === 'cooldown') return styles.credentialDotWarning
  return undefined
}

export interface GroupCollectionTableProps {
  items: GroupCollectionItemDto[]
  pagination: GroupCollectionPaginationDto
  page: number
  channelsByID: Record<string, ChannelDto>
  optimisticEnabled: ReadonlyMap<number, boolean>
  togglingGroupIDs: ReadonlySet<number>
  copyingGroupIDs: ReadonlySet<number>
  onPageChange(page: number): void
  onPrefetchDetail(groupId: number): void
  onToggleEnabled(group: GroupCollectionItemDto, next: boolean): void
  onCopy(group: GroupCollectionItemDto): void
}

/**
 * The collection table: column contract, per-row affordances and the two table
 * plugins. Row-scoped state comes in as props so the page above owns the
 * mutations and this stays a pure projection of server data.
 */
export function GroupCollectionTable({
  items,
  pagination,
  page,
  channelsByID,
  optimisticEnabled,
  togglingGroupIDs,
  copyingGroupIDs,
  onPageChange,
  onPrefetchDetail,
  onToggleEnabled,
  onCopy,
}: GroupCollectionTableProps) {
  const t = useT()
  const intl = useIntl()
  const navigate = useNavigate()
  const narrowViewport = useMediaQuery('(max-width: 680px)')

  function channelName(channelID: string): string {
    return channelsByID[channelID]?.name ?? channelID
  }

  // Sorting is intentionally controlled by the single directionless server
  // sort selector. The table must not advertise a second click direction that
  // the API cannot represent.
  const paginationPlugin = useTablePagination<GroupRow>({
    page,
    onPageChange,
    totalItems: pagination.total_items,
    totalPages: pagination.total_pages,
    pageSize: pagination.page_size,
  })
  const stickyPlugin = useTableStickyColumns<GroupRow>({
    startKeys: narrowViewport ? [] : ['group'],
    endKeys: narrowViewport ? [] : ['actions'],
  })

  const columns: TableColumn<GroupRow>[] = useMemo(
    () => [
      {
        key: 'group',
        header: t('groups.collection.columns.group'),
        width: proportional(1),
        renderCell: (group): ReactNode => (
          <span {...stylex.props(styles.nameCell)}>
            <RouteLink
              to={groupDetailHref(group.id)}
              {...stylex.props(styles.nameLink)}
              aria-label={t('groups.collection.openDetail', { name: group.name })}
              title={group.name}
              onPointerEnter={() => onPrefetchDetail(group.id)}
              onFocus={() => onPrefetchDetail(group.id)}
            >
              {group.name}
            </RouteLink>
          </span>
        ),
      },
      {
        key: 'status',
        header: t('groups.collection.columns.status'),
        width: pixel(180),
        renderCell: (group): ReactNode => (
          <span {...stylex.props(styles.statusCell)}>
            <Switch
              value={groupEnabled(group, optimisticEnabled)}
              isDisabled={togglingGroupIDs.has(group.id)}
              label={t('groups.collection.toggleEnabled', { name: group.name })}
              isLabelHidden
              onChange={(next) => onToggleEnabled(group, next)}
            />
            <Badge
              variant={statusBadgeVariant(group.status)}
              label={t(statusLabelIds[group.status])}
            />
          </span>
        ),
      },
      {
        key: 'channel',
        header: t('groups.collection.columns.channel'),
        width: proportional(1.4),
        renderCell: (group): ReactNode => {
          const channel = channelsByID[group.channel_id]
          return (
            <span {...stylex.props(styles.channelCell)}>
              <span {...stylex.props(styles.channelHeading)}>
                {channel !== undefined && <ChannelIcon icon={channel.icon} mark={channel.mark} />}
                <span {...stylex.props(styles.channelName)} title={channelName(group.channel_id)}>
                  {channelName(group.channel_id)}
                </span>
              </span>
              {group.provider_url !== null && (
                <a
                  {...stylex.props(styles.providerLink)}
                  href={group.provider_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={t('group.openProviderUrl', { url: group.provider_url })}
                >
                  <ExternalLink size={11} aria-hidden />
                  <span>{t('group.settings.base.providerUrl')}</span>
                </a>
              )}
            </span>
          )
        },
      },
      {
        key: 'models',
        header: (
          <Tooltip content={t('groups.collection.modelsHelp')}>
            <span tabIndex={0}>{t('groups.collection.columns.models')}</span>
          </Tooltip>
        ),
        width: pixel(130),
        align: 'end',
        renderCell: (group): ReactNode => (
          <span {...stylex.props(styles.modelCount)}>
            <Tooltip
              content={t('groups.collection.modelsHelpValue', {
                client: intl.formatNumber(group.client_model_count),
                total: intl.formatNumber(group.model_count),
              })}
            >
              <span tabIndex={0}>{intl.formatNumber(group.client_model_count)}</span>
            </Tooltip>
            <span {...stylex.props(styles.modelTotal)}>
              {' / '}
              {intl.formatNumber(group.model_count)}
            </span>
          </span>
        ),
      },
      {
        key: 'credentialHealth',
        header: t('groups.collection.columns.credentialHealth'),
        width: proportional(1.2),
        renderCell: (group): ReactNode => (
          <span {...stylex.props(styles.credentialCell)}>
            <span
              {...stylex.props(
                styles.credentialDot,
                credentialDotStyle(group.credential_status, group.credential_configured),
              )}
              aria-hidden="true"
            />
            {group.credential_configured && group.credential_status !== null
              ? t(`group.credentials.effective.${group.credential_status}`)
              : t('group.unified.noCredentials')}
          </span>
        ),
      },
      {
        key: 'actions',
        header: t('groups.collection.columns.actions'),
        width: pixel(narrowViewport ? 260 : 212),
        align: 'end',
        renderCell: (group): ReactNode => (
          <span {...stylex.props(styles.actionsCell)}>
            <IconButton
              variant="ghost"
              size="sm"
              label={t('groups.collection.manageCredentialFor', { name: group.name })}
              icon={<KeyRound size={15} />}
              onClick={() =>
                void navigate({
                  to: groupDetailHref(group.id),
                  search: { tab: 'credentials' } as never,
                })
              }
            />
            <IconButton
              variant="ghost"
              size="sm"
              label={t('groups.collection.copyFor', { name: group.name })}
              icon={<Copy size={15} />}
              isLoading={copyingGroupIDs.has(group.id)}
              onClick={() => onCopy(group)}
            />
            <IconButton
              variant="ghost"
              size="sm"
              label={t('groups.collection.openDetail', { name: group.name })}
              icon={<ArrowRight size={15} />}
              href={groupDetailHref(group.id)}
            />
            <GroupDeleteDialog groupId={group.id} groupName={group.name} context="list" />
          </span>
        ),
      },
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps -- stable callbacks close over current state
    [
      t,
      intl,
      channelsByID,
      optimisticEnabled,
      togglingGroupIDs,
      copyingGroupIDs,
      narrowViewport,
      onPrefetchDetail,
      onToggleEnabled,
      onCopy,
    ],
  )

  return (
    <div {...stylex.props(styles.tableWrap)}>
      <Table<GroupRow>
        data={items as GroupRow[]}
        columns={columns}
        density="compact"
        dividers="rows"
        hasHover
        textOverflow="truncate"
        aria-label={t('groups.collection.tableLabel')}
        rowIndexStart={(pagination.page - 1) * pagination.page_size + 1}
        rowCount={pagination.total_items}
        plugins={{
          pagination: paginationPlugin,
          sticky: stickyPlugin,
        }}
      />
    </div>
  )
}
