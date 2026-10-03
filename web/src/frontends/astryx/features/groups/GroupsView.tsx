import * as stylex from '@stylexjs/stylex'
import {
  Badge,
  Button,
  EmptyState,
  IconButton,
  Selector,
  Skeleton,
  Switch,
  Table,
  TextInput,
  proportional,
  pixel,
  useTablePagination,
  useTableSortable,
  useTableStickyColumns,
  type TableColumn,
  type TableSortState,
} from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import {
  ArrowRight,
  Copy,
  ExternalLink,
  KeyRound,
  Layers3,
  RefreshCw,
  Search,
  TriangleAlert,
  UserRound,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useIntl } from 'react-intl'

import type { MessageId } from '@shared/i18n/message-ids'

import type {
  ConnectionType,
  GroupCollectionFilters,
  GroupCollectionItemDto,
  GroupCollectionSort,
  GroupCollectionStatus,
} from '@shared/control/types'
import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import {
  cacheGroupSettings,
  copyGroup,
  groupCollectionQueryOptions,
  invalidateGroupSettingsDependents,
  updateGroupSettings,
} from '@shared/control/resources/groups'
import { channelsQueryOptions, type ChannelDto } from '@shared/control/resources/channels'
import { createUUID } from '@shared/lib/uuid'
import { pagePath } from '@shared/routing/page-routes'
import {
  constrainGroupCollectionSearchQuery,
  isCanonicalGroupCollectionRouteQuery,
  parseGroupCollectionRouteQuery,
  serializeGroupCollectionRouteQuery,
} from '@shared/routing/group-collection-route'
import type { SharedRouteQuery } from '@shared/routing/route-query'

import { useT } from '../../app/i18n'
import { RouteLink } from '../../app/route-link'
import { useAppServices } from '../../app/services'
import { useDebouncedAction } from '../../app/use-debounced-action'
import { useMediaQuery } from '../../app/use-media-query'
import { useVisibleRefetch } from '../../app/use-visible-refetch'
import { useCollectionLoading } from '../../app/collection-loading'
import { ChannelIcon } from '../../components/ChannelIcon'

// The server sort enum is directionless (see ADR-0001 read model): each named
// sort implies its own direction. Column sort keys map onto that enum; a
// direction change on an already-active column snaps back to the default.
const columnForSort: Partial<Record<GroupCollectionSort, string>> = {
  name: 'group',
  status: 'status',
}
const sortForColumn: Record<string, GroupCollectionSort> = {
  group: 'name',
  status: 'status',
}
const directionForSort: Record<GroupCollectionSort, 'ascending' | 'descending'> = {
  recent: 'descending',
  created: 'descending',

  name: 'ascending',
  status: 'ascending',
}

const sortOptions: readonly GroupCollectionSort[] = ['recent', 'status', 'name', 'created']

const sortLabelIds: Record<GroupCollectionSort, MessageId> = {
  recent: 'groups.collection.sort.recent',
  status: 'groups.collection.sort.status',
  name: 'groups.collection.sort.name',

  created: 'groups.collection.sort.created',
}
const statusLabelIds: Record<GroupCollectionStatus, MessageId> = {
  available: 'groups.collection.status.available',
  unavailable: 'groups.collection.status.unavailable',
  disabled: 'groups.collection.status.disabled',
}
const connectionTypeLabelIds: Record<ConnectionType, MessageId> = {
  api_key: 'groups.collection.connectionType.apiKey',
  subscription: 'groups.collection.connectionType.subscription',
}

type GroupRow = GroupCollectionItemDto & Record<string, unknown>

// Manifest-derived hrefs; the dynamic route tree keeps `to`/`search` loosely
// typed, so plain `string` paths are the contract here.
function groupDetailHref(id: number): string {
  return `${pagePath('groups')}/${id}`
}

const styles = stylex.create({
  page: {
    width: '100%',
    paddingTop: 'var(--stage-padding-top)',
    paddingBottom: 'var(--stage-padding-bottom)',
    paddingInline: {
      default: 'var(--stage-padding-inline)',
      '@media (max-width: 680px)': 'var(--stage-padding-inline-compact)',
    },
  },
  pageInner: {
    width: 'min(100%, 1240px)',
    marginInline: 'auto',
  },
  collection: {
    position: 'relative',
    minWidth: 0,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: 'var(--space-3)',
    paddingBottom: 'var(--space-4)',
  },
  heading: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  headerActions: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  title: {
    margin: 0,
    fontSize: 'var(--text-heading-2-size, 20px)',
    fontWeight: 650,
  },
  refreshing: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
  summary: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'repeat(4, minmax(0, 1fr))',
      '@media (max-width: 480px)': 'repeat(2, minmax(0, 1fr))',
    },
    gap: 'var(--space-1)',
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-control)',
    paddingBottom: 'var(--space-2)',
  },
  summaryFilter: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    minWidth: 0,
    minHeight: 'var(--touch-target)',
    borderWidth: 0,
    padding: 'var(--space-2) var(--space-3)',
    backgroundColor: {
      default: 'transparent',
      ':hover': 'var(--color-interactive-hover)',
    },
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    fontFamily: 'inherit',
    cursor: 'pointer',
    borderRadius: 'var(--radius-control)',
    outline: {
      default: 'none',
      ':focus-visible': '2px solid var(--color-focus)',
    },
    outlineOffset: 2,
  },
  summaryFilterActive: {
    backgroundColor: {
      default: 'var(--color-action-soft)',
      ':hover': 'var(--color-action-soft)',
    },
    color: 'var(--color-action)',
    fontWeight: 600,
  },
  summaryDot: {
    width: 8,
    height: 8,
    borderRadius: 999,
    backgroundColor: 'var(--color-neutral-fg)',
    flexShrink: 0,
  },
  dotSuccess: { backgroundColor: 'var(--color-success)' },
  dotDanger: { backgroundColor: 'var(--color-danger)' },
  summaryCount: {
    marginLeft: 'auto',
    fontVariantNumeric: 'tabular-nums',
    fontSize: 'var(--text-body)',
    fontWeight: 600,
  },
  toolbar: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'minmax(220px, 1fr) repeat(2, minmax(180px, 0.5fr))',
      '@media (max-width: 880px)': 'repeat(2, minmax(0, 1fr))',
      '@media (max-width: 480px)': 'minmax(0, 1fr)',
    },
    alignItems: 'end',
    gap: 'var(--space-3)',
    paddingTop: 'var(--space-4)',
  },
  toolbarField: {
    minWidth: 0,
  },
  toolbarSearch: {
    minWidth: 0,
    gridColumn: {
      default: 'auto',
      '@media (max-width: 880px)': '1 / -1',
    },
  },
  toolbarResult: {
    gridColumn: '1 / -1',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
    minHeight: 34,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    fontVariantNumeric: 'tabular-nums',
  },
  staleBanner: {
    marginTop: 14,
    borderRadius: 'var(--radius-control, 6px)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-warning)',
    backgroundColor:
      'var(--color-warning-bg, color-mix(in srgb, var(--color-warning) 12%, transparent))',
    color: 'var(--color-text)',
    padding: '9px 12px',
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    fontSize: 'var(--text-meta)',
  },
  skeletonGrid: {
    display: 'grid',
    gap: 10,
    paddingTop: 14,
  },
  tableWrap: {
    backgroundColor: 'var(--color-surface)',
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-control)',
    marginTop: 'var(--space-2)',
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
  typeBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 3,
    borderRadius: 'var(--radius-tag)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    padding: '0 5px',
    fontSize: 11,
    lineHeight: '16px',
    color: 'var(--color-text-muted)',
    whiteSpace: 'nowrap',
    flex: 'none',
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
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
  },
  dotWarning: { backgroundColor: 'var(--color-warning)' },
  actionsCell: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    justifyContent: 'flex-end',
  },
  errorBox: {
    paddingTop: 14,
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

export function GroupsView() {
  const t = useT()
  const intl = useIntl()
  const { apiClient, queryClient, toast } = useAppServices()
  const navigate = useNavigate()
  const { rawSearch, searchStr, pathname } = useRouterState({
    select: (state) => ({
      rawSearch: state.location.search as SharedRouteQuery,
      searchStr: state.location.searchStr,
      pathname: state.location.pathname,
    }),
  })
  const filters = useMemo(() => parseGroupCollectionRouteQuery(rawSearch), [rawSearch])
  const [searchDraft, setSearchDraft] = useState(filters.q ?? '')
  // The debounced callback must read the draft at fire time, not at schedule
  // time — a ref mirrors the Vue `searchDraft.value` read.
  const searchDraftRef = useRef(searchDraft)
  useEffect(() => {
    searchDraftRef.current = searchDraft
  }, [searchDraft])
  const debounce = useDebouncedAction(250)
  const groupsPath = pagePath('groups')
  const narrowViewport = useMediaQuery('(max-width: 680px)')

  const groupsQuery = useQuery(groupCollectionQueryOptions(apiClient, filters))
  const channelsQuery = useQuery(channelsQueryOptions(apiClient, ''))
  const channelsByID = useMemo<Record<string, ChannelDto>>(
    () =>
      Object.fromEntries(
        (channelsQuery.data?.items ?? []).map((channel) => [channel.channel_id, channel]),
      ),
    [channelsQuery.data],
  )

  const data = groupsQuery.data
  const [togglingGroupIDs, setTogglingGroupIDs] = useState<ReadonlySet<number>>(new Set())
  const [copyingGroupIDs, setCopyingGroupIDs] = useState<ReadonlySet<number>>(new Set())
  const [optimisticEnabled, setOptimisticEnabled] = useState<ReadonlyMap<number, boolean>>(
    new Map(),
  )

  // Route → draft sync via the render-adjustment pattern (set-state during
  // render is the sanctioned alternative to an effect that only mirrors
  // external state back into local state).
  const [lastSearchStr, setLastSearchStr] = useState(searchStr)
  if (lastSearchStr !== searchStr) {
    setLastSearchStr(searchStr)
    debounce.cancel()
    setSearchDraft(filters.q ?? '')
  }

  // Canonicalize non-canonical query params — mirrors the classic router
  // watch: junk/duplicated keys are dropped via a history replace.
  useEffect(() => {
    // During a pending transition the outgoing route keeps rendering while
    // location already moved — a late canonicalization must not resurrect it.
    if (pathname !== groupsPath) return
    if (!isCanonicalGroupCollectionRouteQuery(rawSearch, filters)) {
      void navigate({
        to: groupsPath,
        search: serializeGroupCollectionRouteQuery(filters),
        replace: true,
        resetScroll: false,
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the URL string
  }, [searchStr])

  // Page correction: when the server clamps a deep page, mirror the
  // correction into the URL (classic parity).
  const totalPages = data?.pagination.total_pages
  const requestedPage = filters.page
  const isPlaceholder = groupsQuery.isPlaceholderData
  useEffect(() => {
    if (pathname !== groupsPath) return
    if (
      !isPlaceholder &&
      totalPages !== undefined &&
      totalPages > 0 &&
      requestedPage > totalPages
    ) {
      void navigate({
        to: groupsPath,
        search: serializeGroupCollectionRouteQuery({ ...filters, page: totalPages }),
        replace: true,
        resetScroll: false,
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on server pagination
  }, [totalPages, requestedPage, isPlaceholder])

  useVisibleRefetch([groupsQuery.refetch])

  function routeWithFilters(next: GroupCollectionFilters, replace = false): void {
    void navigate({
      to: groupsPath,
      search: serializeGroupCollectionRouteQuery(next),
      replace,
      resetScroll: false,
    })
  }

  function updateConditions(
    patch: Partial<Pick<GroupCollectionFilters, 'q' | 'status' | 'connection_type' | 'sort'>>,
  ): void {
    const q = constrainGroupCollectionSearchQuery(searchDraftRef.current)
    routeWithFilters({ ...filters, q, ...patch, page: 1 })
  }

  function onSearchChange(value: string): void {
    debounce.cancel()
    setSearchDraft(value)
    // The clear affordance emits '' through onChange; applying it immediately
    // keeps the classic `@clear` contract (no debounce on explicit clear).
    if (value === '') {
      updateConditions({ q: undefined })
      return
    }
    debounce.schedule(() => {
      updateConditions({ q: constrainGroupCollectionSearchQuery(searchDraftRef.current) })
    })
  }

  function setStatus(status: GroupCollectionStatus | undefined): void {
    updateConditions({ status })
  }

  function setConnectionType(value: string | null): void {
    if (value === null || value === '') {
      updateConditions({ connection_type: undefined })
      return
    }
    if (value !== 'api_key' && value !== 'subscription') return
    updateConditions({ connection_type: value as ConnectionType })
  }

  function setSort(value: string): void {
    updateConditions({ sort: value as GroupCollectionSort })
  }

  function resetConditions(): void {
    debounce.cancel()
    setSearchDraft('')
    routeWithFilters({ sort: 'recent', page: 1, page_size: 100 })
  }

  function setPage(page: number): void {
    routeWithFilters({ ...filters, page })
  }

  // Controlled plugin states — all bound to the typed route filters so the
  // server stays the single source of truth (ADR-0001: no client-side paging).
  const sortState: TableSortState = useMemo(() => {
    const column = columnForSort[filters.sort]
    return column === undefined
      ? []
      : [{ sortKey: column, direction: directionForSort[filters.sort] }]
  }, [filters.sort])

  function onSortChange(next: TableSortState): void {
    const entry = next[0]
    if (entry === undefined) {
      setSort('recent')
      return
    }
    const mapped = sortForColumn[entry.sortKey]
    if (mapped === undefined) return
    // Second click on the active column cycles direction; the directionless
    // enum has nowhere to go, so it releases back to the default sort.
    if (mapped === filters.sort && directionForSort[mapped] !== entry.direction) {
      setSort('recent')
      return
    }
    setSort(mapped)
  }

  const sortablePlugin = useTableSortable<GroupRow>({
    sort: sortState,
    onSortChange,
    isMultiSortEnabled: false,
    allowUnsortedState: true,
  })
  const paginationPlugin = useTablePagination<GroupRow>({
    page: filters.page,
    onPageChange: setPage,
    totalItems: data?.pagination.total_items,
    totalPages: data?.pagination.total_pages,
    pageSize: filters.page_size,
  })
  const stickyPlugin = useTableStickyColumns<GroupRow>({
    startKeys: narrowViewport ? [] : ['group'],
    endKeys: narrowViewport ? [] : ['actions'],
  })

  async function toggleGroupEnabled(group: GroupCollectionItemDto, next: boolean): Promise<void> {
    if (togglingGroupIDs.has(group.id)) return
    setOptimisticEnabled((prev) => new Map(prev).set(group.id, next))
    setTogglingGroupIDs((prev) => new Set(prev).add(group.id))
    try {
      const settings = await updateGroupSettings(apiClient, group.id, { enabled: next })
      cacheGroupSettings(queryClient, group.id, settings)
      await invalidateGroupSettingsDependents(queryClient, group.id)
      toast.show({
        message: t(next ? 'groups.collection.enabledOn' : 'groups.collection.enabledOff', {
          name: group.name,
        }),
        tone: next ? 'success' : 'warning',
      })
    } catch {
      toast.show({ message: t('groups.collection.toggleFailed'), tone: 'danger' })
    } finally {
      setOptimisticEnabled((prev) => {
        const nextMap = new Map(prev)
        nextMap.delete(group.id)
        return nextMap
      })
      setTogglingGroupIDs((prev) => {
        const nextSet = new Set(prev)
        nextSet.delete(group.id)
        return nextSet
      })
    }
  }

  async function copyGroupRecord(group: GroupCollectionItemDto): Promise<void> {
    if (copyingGroupIDs.has(group.id)) return
    setCopyingGroupIDs((prev) => new Set(prev).add(group.id))
    try {
      const result = await copyGroup(apiClient, group.id, createUUID())
      await applyInvalidationPlan(queryClient, mutationInvalidationPlans.group.create)
      toast.show({
        message: t('groups.collection.copySucceeded', { name: result.group_name }),
        tone: 'success',
      })
      // Group detail is astryx-owned — stay in the SPA.
      void navigate({ to: groupDetailHref(result.group_id) })
    } catch {
      toast.show({ message: t('groups.collection.copyFailed'), tone: 'danger' })
    } finally {
      setCopyingGroupIDs((prev) => {
        const nextSet = new Set(prev)
        nextSet.delete(group.id)
        return nextSet
      })
    }
  }

  const hasFilterCriteria =
    filters.q !== undefined || filters.status !== undefined || filters.connection_type !== undefined
  const hasChangedConditions = hasFilterCriteria || filters.sort !== 'recent'
  const collectionBusy = data !== undefined && groupsQuery.isFetching
  const {
    initial: initialLoading,
    transition: collectionTransition,
    refreshing: collectionRefreshing,
    rows: skeletonRows,
  } = useCollectionLoading(
    {
      pending: groupsQuery.isPending,
      placeholder: groupsQuery.isPlaceholderData,
      fetching: groupsQuery.isFetching,
      hasData: data !== undefined,
      itemCount: data?.items.length ?? 0,
    },
    { fallbackRows: 20 },
  )

  const connectionTypeOptions = [
    { value: '', label: t('groups.collection.connectionType.all') },
    { value: 'api_key', label: t('groups.collection.connectionType.apiKey') },
    { value: 'subscription', label: t('groups.collection.connectionType.subscription') },
  ]
  const sortSelectOptions = sortOptions.map((sort) => ({
    value: sort,
    label: t(sortLabelIds[sort]),
  }))

  function channelName(channelID: string): string {
    return channelsByID[channelID]?.name ?? channelID
  }

  const summaryItems = useMemo(() => {
    const summary = data?.summary
    if (!summary) return []
    return [
      {
        value: undefined as GroupCollectionStatus | undefined,
        label: t('groups.collection.status.all'),
        count: summary.total,
        tone: 'neutral' as const,
      },
      {
        value: 'available' as const,
        label: t('groups.collection.status.available'),
        count: summary.available,
        tone: 'success' as const,
      },
      {
        value: 'unavailable' as const,
        label: t('groups.collection.status.unavailable'),
        count: summary.unavailable,
        tone: 'danger' as const,
      },
      {
        value: 'disabled' as const,
        label: t('groups.collection.status.disabled'),
        count: summary.disabled,
        tone: 'neutral' as const,
      },
    ]
  }, [data?.summary, t])

  const columns: TableColumn<GroupRow>[] = useMemo(
    () => [
      {
        key: 'group',
        header: t('groups.collection.columns.group'),
        width: proportional(1),
        sortable: true,
        renderCell: (group): ReactNode => (
          <span {...stylex.props(styles.nameCell)}>
            <RouteLink
              to={groupDetailHref(group.id)}
              {...stylex.props(styles.nameLink)}
              aria-label={t('groups.collection.openDetail', { name: group.name })}
              title={group.name}
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
        sortable: true,
        renderCell: (group): ReactNode => (
          <span {...stylex.props(styles.statusCell)}>
            <Switch
              value={groupEnabled(group, optimisticEnabled)}
              isDisabled={togglingGroupIDs.has(group.id)}
              label={t('groups.collection.toggleEnabled', { name: group.name })}
              isLabelHidden
              onChange={(next) => void toggleGroupEnabled(group, next)}
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

                <span {...stylex.props(styles.typeBadge)}>
                  {group.connection_type === 'api_key' ? (
                    <KeyRound size={10} aria-hidden />
                  ) : (
                    <UserRound size={10} aria-hidden />
                  )}
                  {t(connectionTypeLabelIds[group.connection_type])}
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
        header: t('groups.collection.columns.models'),
        width: pixel(130),
        align: 'end',
        renderCell: (group): ReactNode => (
          <span {...stylex.props(styles.modelCount)}>
            {intl.formatNumber(group.client_model_count)}
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
                styles.summaryDot,
                group.credential_configured && group.credential_status === 'available'
                  ? styles.dotSuccess
                  : group.credential_status === 'blacklisted'
                    ? styles.dotDanger
                    : group.credential_status === 'cooldown'
                      ? styles.dotWarning
                      : undefined,
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
        width: pixel(narrowViewport ? 156 : 108),
        align: 'end',
        renderCell: (group): ReactNode => (
          <span {...stylex.props(styles.actionsCell)}>
            <IconButton
              variant="ghost"
              size="sm"
              label={t('groups.collection.manageCredentialFor', { name: group.name })}
              icon={<KeyRound size={15} />}
              href={`${groupDetailHref(group.id)}?tab=credentials`}
            />
            <IconButton
              variant="ghost"
              size="sm"
              label={t('groups.collection.copyFor', { name: group.name })}
              icon={<Copy size={15} />}
              isLoading={copyingGroupIDs.has(group.id)}
              onClick={() => void copyGroupRecord(group)}
            />
            <IconButton
              variant="ghost"
              size="sm"
              label={t('groups.collection.openDetail', { name: group.name })}
              icon={<ArrowRight size={15} />}
              href={groupDetailHref(group.id)}
            />
          </span>
        ),
      },
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps -- stable callbacks close over current state
    [t, intl, channelsByID, optimisticEnabled, togglingGroupIDs, copyingGroupIDs, narrowViewport],
  )

  return (
    <section {...stylex.props(styles.page)} aria-labelledby="groups-title">
      <div {...stylex.props(styles.pageInner)}>
        <div {...stylex.props(styles.collection)} aria-busy={collectionBusy || undefined}>
          <header {...stylex.props(styles.header)}>
            <div {...stylex.props(styles.heading)}>
              <h1 id="groups-title" {...stylex.props(styles.title)}>
                {t('groups.title')}
              </h1>
              {data !== undefined && (
                <Badge variant="neutral" label={intl.formatNumber(data.summary.total)} />
              )}
            </div>
            <div {...stylex.props(styles.headerActions)}>
              <IconButton
                variant="ghost"
                size="sm"
                label={t('groups.collection.retry')}
                icon={<RefreshCw size={15} />}
                isLoading={groupsQuery.isFetching}
                onClick={() => void groupsQuery.refetch()}
              />
              {data !== undefined && data.summary.total > 0 && (
                <Button
                  variant="primary"
                  size="sm"
                  label={t('groups.collection.importCredentials')}
                  icon={<KeyRound size={15} />}
                  href={pagePath('import')}
                />
              )}
            </div>
          </header>
          <span aria-live="polite" {...stylex.props(styles.refreshing)}>
            {collectionRefreshing ? t('groups.collection.loading') : ''}
          </span>

          {groupsQuery.isPending || initialLoading ? (
            <div
              {...stylex.props(styles.skeletonGrid)}
              role="status"
              aria-label={t('groups.collection.loading')}
            >
              {Array.from({ length: 8 }, (_, index) => (
                <Skeleton key={index} height={68} radius={2} />
              ))}
            </div>
          ) : groupsQuery.isError && !data ? (
            <div {...stylex.props(styles.errorBox)} role="alert">
              <EmptyState
                title={t('groups.collection.errorTitle')}
                description={t('groups.collection.errorDescription')}
                icon={<TriangleAlert size={20} />}
                actions={
                  <Button
                    variant="secondary"
                    size="sm"
                    label={t('groups.collection.retry')}
                    onClick={() => void groupsQuery.refetch()}
                  />
                }
              />
            </div>
          ) : data !== undefined ? (
            <>
              {data.summary.total > 0 && (
                <section
                  {...stylex.props(styles.summary)}
                  aria-label={t('groups.collection.summary.region')}
                >
                  {summaryItems.map((item) => (
                    <button
                      key={item.value ?? 'all'}
                      type="button"
                      aria-pressed={filters.status === item.value}
                      {...stylex.props(
                        styles.summaryFilter,
                        filters.status === item.value && styles.summaryFilterActive,
                      )}
                      onClick={() => setStatus(item.value)}
                    >
                      <span
                        {...stylex.props(
                          styles.summaryDot,
                          item.tone === 'success'
                            ? styles.dotSuccess
                            : item.tone === 'danger'
                              ? styles.dotDanger
                              : undefined,
                        )}
                        aria-hidden="true"
                      />
                      <span>{item.label}</span>
                      <span {...stylex.props(styles.summaryCount)}>
                        {intl.formatNumber(item.count)}
                      </span>
                    </button>
                  ))}
                </section>
              )}

              {groupsQuery.isError && (
                <div {...stylex.props(styles.staleBanner)} role="status">
                  <TriangleAlert size={13} aria-hidden />
                  <span>{t('groups.collection.stale')}</span>
                  <Button
                    variant="secondary"
                    size="sm"
                    label={t('groups.collection.retry')}
                    onClick={() => void groupsQuery.refetch()}
                  />
                </div>
              )}

              {data.summary.total > 0 && (
                <div
                  {...stylex.props(styles.toolbar)}
                  role="group"
                  aria-label={t('groups.collection.filters.region')}
                >
                  <div {...stylex.props(styles.toolbarSearch)}>
                    <TextInput
                      size="sm"
                      label={t('groups.collection.filters.searchLabel')}
                      placeholder={t('groups.collection.filters.searchPlaceholder')}
                      startIcon={<Search size={14} />}
                      value={searchDraft}
                      hasClear
                      onChange={onSearchChange}
                    />
                  </div>
                  <div {...stylex.props(styles.toolbarField)}>
                    <Selector
                      size="sm"
                      variant="input"
                      label={t('groups.collection.connectionType.label')}
                      options={connectionTypeOptions}
                      value={filters.connection_type ?? ''}
                      onChange={setConnectionType}
                    />
                  </div>
                  <div {...stylex.props(styles.toolbarField)}>
                    <Selector
                      size="sm"
                      variant="input"
                      label={t('groups.collection.filters.sortLabel')}
                      options={sortSelectOptions}
                      value={filters.sort}
                      onChange={setSort}
                    />
                  </div>
                  <div {...stylex.props(styles.toolbarResult)}>
                    <span aria-live="polite">
                      {t('groups.collection.result', {
                        shown: intl.formatNumber(data.items.length),
                        total: intl.formatNumber(data.pagination.total_items),
                      })}
                    </span>
                    {hasChangedConditions && (
                      <Button
                        variant="ghost"
                        size="sm"
                        label={t('groups.collection.filters.reset')}
                        onClick={resetConditions}
                      />
                    )}
                  </div>
                </div>
              )}

              {collectionTransition ? (
                <div
                  {...stylex.props(styles.skeletonGrid)}
                  role="status"
                  aria-label={t('groups.collection.loading')}
                >
                  {Array.from({ length: Math.min(skeletonRows, 8) }, (_, index) => (
                    <Skeleton key={index} height={52} radius={2} />
                  ))}
                </div>
              ) : data.summary.total === 0 ? (
                <EmptyState
                  title={t('groups.collection.emptyTitle')}
                  description={t('groups.collection.emptyDescription')}
                  icon={<Layers3 size={20} />}
                  actions={
                    <Button
                      variant="secondary"
                      size="sm"
                      label={t('groups.collection.importCredentials')}
                      icon={<KeyRound size={15} />}
                      href={pagePath('import')}
                    />
                  }
                />
              ) : data.pagination.total_items === 0 && hasFilterCriteria ? (
                <EmptyState
                  title={t('groups.collection.noResultsTitle')}
                  description={t('groups.collection.noResultsDescription')}
                  icon={<Search size={20} />}
                  actions={
                    <Button
                      variant="secondary"
                      size="sm"
                      label={t('groups.collection.filters.reset')}
                      onClick={resetConditions}
                    />
                  }
                />
              ) : data.items.length > 0 ? (
                <div {...stylex.props(styles.tableWrap)}>
                  <Table<GroupRow>
                    data={data.items as GroupRow[]}
                    columns={columns}
                    density="balanced"
                    dividers="rows"
                    hasHover
                    textOverflow="truncate"
                    aria-label={t('groups.collection.tableLabel')}
                    rowIndexStart={(data.pagination.page - 1) * data.pagination.page_size + 1}
                    rowCount={data.pagination.total_items}
                    plugins={{
                      sortable: sortablePlugin,
                      pagination: paginationPlugin,
                      sticky: stickyPlugin,
                    }}
                  />
                </div>
              ) : null}
            </>
          ) : null}
        </div>
      </div>
    </section>
  )
}
