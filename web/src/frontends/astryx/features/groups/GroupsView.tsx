import * as stylex from '@stylexjs/stylex'

import type { MessageId } from '@shared/i18n/message-ids'

import type { GroupCollectionSort } from '@shared/control/types'

import { useT } from '../../app/i18n'
import { GroupCollectionHeader } from './GroupCollectionHeader'
import {
  GroupCollectionEmpty,
  GroupCollectionError,
  GroupCollectionSkeleton,
  GroupCollectionStaleBanner,
} from './GroupCollectionStates'
import { GroupCollectionSummary } from './GroupCollectionSummary'
import { GroupCollectionTable } from './GroupCollectionTable'
import { GroupCollectionToolbar, type GroupSortOption } from './GroupCollectionToolbar'
import { useGroupCollection } from './use-group-collection'

// The server sort enum is directionless (see ADR-0001 read model): each named
// sort implies its own direction. Column sort keys map onto that enum; a
// direction change on an already-active column snaps back to the default.

const sortOptions: readonly GroupCollectionSort[] = ['recent', 'status', 'name', 'created']

const sortLabelIds: Record<GroupCollectionSort, MessageId> = {
  recent: 'groups.collection.sort.recent',
  status: 'groups.collection.sort.status',
  name: 'groups.collection.sort.name',
  created: 'groups.collection.sort.created',
}

/**
 * Composition root for the group collection: the hook owns data, routing and
 * mutations, the children own layout. Nothing here decides what a group is.
 */
export function GroupsView() {
  const t = useT()
  const collection = useGroupCollection()
  const data = collection.data

  const sortSelectOptions: GroupSortOption[] = sortOptions.map((sort) => ({
    value: sort,
    label: t(sortLabelIds[sort]),
  }))

  return (
    <section {...stylex.props(styles.page)} aria-labelledby="groups-title">
      <div {...stylex.props(styles.pageInner)}>
        <div
          {...stylex.props(styles.collection)}
          aria-busy={collection.collectionBusy || undefined}
        >
          <GroupCollectionHeader
            total={data?.summary.total}
            refreshing={collection.groupsQuery.isFetching}
            refreshingLabel={collection.collectionRefreshing ? t('groups.collection.loading') : ''}
            onRefresh={() => void collection.groupsQuery.refetch()}
          />
          {collection.groupsQuery.isPending || collection.initialLoading ? (
            <GroupCollectionSkeleton rows={8} height={68} />
          ) : collection.groupsQuery.isError && !data ? (
            <GroupCollectionError onRetry={() => void collection.groupsQuery.refetch()} />
          ) : data !== undefined ? (
            <>
              {data.summary.total > 0 && (
                <GroupCollectionSummary
                  items={collection.summaryItems}
                  status={collection.filters.status}
                  onSelect={collection.setStatus}
                />
              )}
              {collection.groupsQuery.isError && (
                <GroupCollectionStaleBanner onRetry={() => void collection.groupsQuery.refetch()} />
              )}
              {data.summary.total > 0 && (
                <GroupCollectionToolbar
                  searchDraft={collection.searchDraft}
                  sort={collection.filters.sort}
                  sortOptions={sortSelectOptions}
                  itemsShown={data.items}
                  totalItems={data.pagination.total_items}
                  hasChangedConditions={collection.hasChangedConditions}
                  onSearchChange={collection.onSearchChange}
                  onSortChange={collection.setSort}
                  onReset={collection.resetConditions}
                />
              )}
              {collection.collectionTransition ? (
                <GroupCollectionSkeleton rows={Math.min(collection.skeletonRows, 8)} height={52} />
              ) : data.summary.total === 0 ? (
                <GroupCollectionEmpty variant="empty" onReset={collection.resetConditions} />
              ) : data.pagination.total_items === 0 && collection.hasFilterCriteria ? (
                <GroupCollectionEmpty variant="noResults" onReset={collection.resetConditions} />
              ) : data.items.length > 0 ? (
                <GroupCollectionTable
                  items={data.items}
                  pagination={data.pagination}
                  page={collection.filters.page}
                  channelsByID={collection.channelsByID}
                  optimisticEnabled={collection.optimisticEnabled}
                  togglingGroupIDs={collection.togglingGroupIDs}
                  copyingGroupIDs={collection.copyingGroupIDs}
                  onPageChange={collection.setPage}
                  onPrefetchDetail={collection.prefetchGroupSummary}
                  onToggleEnabled={collection.toggleGroupEnabled}
                  onCopy={collection.copyGroupRecord}
                />
              ) : null}
            </>
          ) : null}
        </div>
      </div>
    </section>
  )
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
    width: '100%',
    marginInline: 'auto',
  },
  collection: {
    position: 'relative',
    minWidth: 0,
  },
})
