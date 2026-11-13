import * as stylex from '@stylexjs/stylex'
import { Button, EmptyState, Skeleton } from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import { useRouterState } from '@tanstack/react-router'
import { RefreshCw, TriangleAlert } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import {
  groupModelsQueryOptions,
  groupSettingsQueryOptions,
  groupSummaryQueryOptions,
} from '@shared/control/resources/groups'
import { credentialQueryOptions } from '@shared/control/resources/credentials'
import { pagePath } from '@shared/routing/page-routes'
import { parsePositiveId } from '@shared/routing/group-detail-route'

import { useStableLoading } from '../../app/collection-loading'
import { useT } from '../../app/i18n'
import { RouteLink } from '../../app/route-link'
import { useAppServices } from '../../app/services'
import { ContentPanel } from '../../components/ContentPanel'
import { GroupCredentialsTab } from './credentials/GroupCredentialsTab'
import { GroupSettingsTab } from './settings/GroupSettingsTab'
import { GroupModelsTab } from './models/GroupModelsTab'
import { GroupHeader } from './GroupHeader'
import { GroupUnifiedSaveBar } from './GroupUnifiedSaveBar'
import type {
  GroupSettingsEditorHandle,
  GroupEditorState,
  GroupModelsEditorHandle,
} from './editor-handles'

const idleEditorState: GroupEditorState = {
  dirty: false,
  pending: false,
  error: '',
  saved: false,
}

const spin = stylex.keyframes({
  to: { transform: 'rotate(360deg)' },
})

const styles = stylex.create({
  page: {
    display: 'grid',
    width: '100%',
    minWidth: 0,
    marginInline: 'auto',
    alignContent: 'start',
    gap: 'var(--space-4)',
    paddingTop: 'var(--stage-padding-top)',
    paddingBottom: {
      default: 'var(--stage-padding-bottom)',
    },
    paddingInline: {
      default: 'var(--stage-padding-inline)',
      '@media (max-width: 680px)': 'var(--stage-padding-inline-compact)',
    },
  },
  invalid: {
    display: 'grid',
    maxWidth: '640px',
    gap: 'var(--space-3)',
  },
  invalidHeading: {
    margin: 0,
  },
  invalidText: {
    margin: 0,
    color: 'var(--color-text-muted)',
  },
  credentials: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'auto minmax(0, 1fr)',
      '@media (max-width: 640px)': 'minmax(0, 1fr)',
    },
    alignItems: 'center',
    columnGap: 'var(--space-3)',
    rowGap: '8px',
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 'var(--space-2)',
    paddingBottom: 'var(--space-2)',
  },
  credentialList: {
    display: 'grid',
    minWidth: 0,
    gridColumn: { default: '2', '@media (max-width: 640px)': '1' },
    gap: 6,
  },
  credentialRow: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'flex-end',
    gap: 'var(--space-3)',
    flexWrap: { default: 'nowrap', '@media (max-width: 520px)': 'wrap' },
  },
  credentialField: {
    display: 'grid',
    minWidth: 0,
    flexGrow: 1,
    gap: 4,
  },
  credentialFieldLabel: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  credentialFieldInput: {
    width: '100%',
    minHeight: 'var(--control-md)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text)',
    paddingTop: 0,
    paddingBottom: 0,
    paddingInline: 'var(--space-3)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'inherit',
  },
  credentialRowEnd: {
    flexShrink: 0,
    marginBottom: '1px',
  },
  credentialHeading: {
    margin: 0,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--title-section)',
    fontWeight: 650,
  },
  empty: {
    gridColumn: { default: '2', '@media (max-width: 640px)': '1' },
    margin: 0,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  staleBanner: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    color: 'var(--color-warning)',
    fontSize: 'var(--text-sm)',
  },
  advancedPanel: {
    display: { default: 'block', ':has(#group-settings-advanced-target:empty)': 'none' },
  },
  refreshing: {
    color: 'var(--color-text-faint)',
    animationName: {
      default: spin,
      '@media (prefers-reduced-motion: reduce)': 'none',
    },
    animationDuration: '900ms',
    animationTimingFunction: 'linear',
    animationIterationCount: 'infinite',
  },
})

export function GroupDetailView() {
  const t = useT()
  const { apiClient, queryClient } = useAppServices()
  const { id } = useRouterState({
    select: (state) => ({
      id: (state.matches.at(-1)?.params as { id?: string } | undefined)?.id,
    }),
  })
  const groupId = parsePositiveId(id)
  const summaryQuery = useQuery(groupSummaryQueryOptions(apiClient, groupId))
  // The summary omits the enabled flag, so the models tab reads it from the
  // settings query (same cache entry the settings tab prefetches).
  const settingsQuery = useQuery(groupSettingsQueryOptions(apiClient, groupId))

  const initialLoading = useStableLoading(summaryQuery.isPending && summaryQuery.data === undefined)
  const summaryRefreshing = summaryQuery.data !== undefined && summaryQuery.isFetching

  const [settingsState, setSettingsState] = useState<GroupEditorState>(idleEditorState)
  const [modelsState, setModelsState] = useState<GroupEditorState>(idleEditorState)
  const [deletePending, setDeletePending] = useState(false)
  const [combinedPending, setCombinedPending] = useState(false)
  const settingsEditorRef = useRef<GroupSettingsEditorHandle>(null)
  const modelsEditorRef = useRef<GroupModelsEditorHandle>(null)

  // Warm the tab queries as soon as the id resolves — classic prefetches on
  // the same immediate groupId watch.
  useEffect(() => {
    if (groupId === undefined) return
    void Promise.allSettled([
      queryClient.prefetchQuery(credentialQueryOptions(apiClient, groupId)),
      queryClient.prefetchQuery(groupModelsQueryOptions(apiClient, groupId)),
      queryClient.prefetchQuery(groupSettingsQueryOptions(apiClient, groupId)),
    ])
  }, [groupId, apiClient, queryClient])

  return (
    <section {...stylex.props(styles.page)} aria-labelledby="group-detail-title">
      {groupId === undefined ? (
        <div {...stylex.props(styles.invalid)} role="alert">
          <h1 id="group-detail-title" {...stylex.props(styles.invalidHeading)}>
            {t('group.invalidTitle')}
          </h1>
          <p {...stylex.props(styles.invalidText)}>{t('group.invalidDescription')}</p>
          <RouteLink to={pagePath('groups')}>{t('group.backToGroups')}</RouteLink>
        </div>
      ) : (
        <>
          {summaryRefreshing && (
            <div role="status" aria-label={t('group.loading')} {...stylex.props(styles.refreshing)}>
              <RefreshCw size={13} aria-hidden />
            </div>
          )}
          {summaryQuery.isPending || initialLoading ? (
            <div role="status" aria-label={t('group.loading')}>
              <Skeleton height={220} radius={2} />
            </div>
          ) : summaryQuery.isError && !summaryQuery.data ? (
            <div role="alert">
              <EmptyState
                title={t('group.loadFailed')}
                icon={<TriangleAlert size={20} />}
                actions={
                  <Button
                    variant="secondary"
                    size="sm"
                    label={t('common.retry')}
                    onClick={() => void summaryQuery.refetch()}
                  />
                }
              />
            </div>
          ) : summaryQuery.data ? (
            <>
              {summaryQuery.isError && (
                <div {...stylex.props(styles.staleBanner)} role="status">
                  <TriangleAlert size={13} aria-hidden />
                  <span>{t('group.stale')}</span>
                  <Button
                    variant="secondary"
                    size="sm"
                    label={t('common.retry')}
                    onClick={() => void summaryQuery.refetch()}
                  />
                </div>
              )}
              <GroupHeader group={summaryQuery.data} />
              <>
                <ContentPanel>
                  <GroupSettingsTab
                    ref={settingsEditorRef}
                    key={`settings-${groupId}`}
                    groupId={groupId}
                    unified
                    blocked={deletePending || combinedPending}
                    onStateChange={setSettingsState}
                  />
                </ContentPanel>
                <ContentPanel>
                  <GroupCredentialsTab key={`credentials-${groupId}`} groupId={groupId} />
                </ContentPanel>
                <ContentPanel>
                  <GroupModelsTab
                    ref={modelsEditorRef}
                    key={`models-${groupId}`}
                    groupId={groupId}
                    channelId={summaryQuery.data.channel_id}
                    scheduleChannelId={settingsState.channelId ?? summaryQuery.data.channel_id}
                    enabled={settingsQuery.data?.enabled ?? true}
                    unified
                    blocked={deletePending || combinedPending}
                    onStateChange={setModelsState}
                  />
                </ContentPanel>
                {/* Unified-mode portal target: GroupSettingsTab portals its
                      advanced section here via createPortal. Contractual id. */}
                <div {...stylex.props(styles.advancedPanel)}>
                  <ContentPanel>
                    <div id="group-settings-advanced-target" />
                  </ContentPanel>
                </div>
                <GroupUnifiedSaveBar
                  groupId={groupId}
                  groupName={summaryQuery.data.name}
                  settingsState={settingsState}
                  modelsState={modelsState}
                  deletePending={deletePending}
                  settingsEditorRef={settingsEditorRef}
                  modelsEditorRef={modelsEditorRef}
                  onDeletePendingChange={setDeletePending}
                  combinedPending={combinedPending}
                  onCombinedPendingChange={setCombinedPending}
                />
              </>
            </>
          ) : null}
        </>
      )}
    </section>
  )
}
