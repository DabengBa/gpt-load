import * as stylex from '@stylexjs/stylex'
import { Badge, Button, EmptyState, Skeleton } from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import { useRouterState } from '@tanstack/react-router'
import { RefreshCw, TriangleAlert } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import type { CredentialItemDto } from '@shared/control/types'
import { credentialQueryOptions } from '@shared/control/resources/credentials'
import {
  groupModelsQueryOptions,
  groupSettingsQueryOptions,
  groupSummaryQueryOptions,
} from '@shared/control/resources/groups'
import { pagePath } from '@shared/routing/page-routes'
import { parsePositiveId } from '@shared/routing/group-detail-route'
import { scalarRouteQuery, type SharedRouteQuery } from '@shared/routing/route-query'

import { useStableLoading } from '../../app/collection-loading'
import { useT } from '../../app/i18n'
import { RouteLink } from '../../app/route-link'
import { useAppServices } from '../../app/services'
import { StickySaveBar } from '../../components/StickySaveBar'
import { GroupApiKeyEditor } from './credentials/GroupApiKeyEditor'
import { GroupCredentialsTab } from './credentials/GroupCredentialsTab'
import { GroupDeleteDialog } from './settings/GroupDeleteDialog'
import { GroupSettingsTab } from './settings/GroupSettingsTab'
import { GroupModelsTab } from './models/GroupModelsTab'
import { GroupHeader } from './GroupHeader'
import type { GroupEditorHandle, GroupEditorState, GroupModelsEditorHandle } from './editor-handles'
import { credentialStatusBadgeVariant, type OperationalStatus } from './credential-status'

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
    alignContent: 'start',
    gap: 'var(--space-3)',
    paddingBottom: 'var(--space-3)',
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
    gridTemplateColumns: 'minmax(0, 1fr)',
    rowGap: '8px',
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: { default: '12px', '@media (max-width: 640px)': '10px' },
    paddingBottom: { default: '12px', '@media (max-width: 640px)': '10px' },
  },
  credentialList: {
    display: 'grid',
    minWidth: 0,
    gridColumn: '1',
    gap: 6,
  },
  credentialRow: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'flex-end',
    gap: 'var(--space-3)',
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
  empty: {
    gridColumn: '1',
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

function unifiedCredentialSummary(
  credential: CredentialItemDto,
  t: ReturnType<typeof useT>,
): { status: OperationalStatus; label: string } {
  if (credential.auth_state === 'refreshing') {
    return { status: 'unknown', label: t('group.credentials.subscription.status.refreshing') }
  }
  if (credential.auth_state === 'reauthorization_required') {
    return {
      status: 'unavailable',
      label: t('group.credentials.subscription.status.needs_reauth'),
    }
  }
  if (credential.auth_state === 'outcome_unknown') {
    return { status: 'unknown', label: t('group.credentials.subscription.status.outcome_unknown') }
  }
  return {
    status: credential.effective_status,
    label: t(`group.credentials.effective.${credential.effective_status}`),
  }
}

export function GroupDetailView() {
  const t = useT()
  const { apiClient, queryClient } = useAppServices()
  const { id, rawSearch } = useRouterState({
    select: (state) => ({
      id: (state.matches.at(-1)?.params as { id?: string } | undefined)?.id,
      rawSearch: state.location.search as SharedRouteQuery,
    }),
  })
  const groupId = parsePositiveId(id)
  const summaryQuery = useQuery(groupSummaryQueryOptions(apiClient, groupId))
  // The summary omits the enabled flag, so the models tab reads it from the
  // settings query (same cache entry the settings tab prefetches).
  const settingsQuery = useQuery(groupSettingsQueryOptions(apiClient, groupId))
  const credentialsQuery = useQuery(credentialQueryOptions(apiClient, groupId ?? 0))
  // managementOpen reads the RAW query — an absent/unknown tab renders the
  // unified settings+models view, matching classic.
  const managementOpen = scalarRouteQuery(rawSearch.tab) === 'credentials'
  const initialLoading = useStableLoading(summaryQuery.isPending && summaryQuery.data === undefined)
  const summaryRefreshing = summaryQuery.data !== undefined && summaryQuery.isFetching

  const [settingsState, setSettingsState] = useState<GroupEditorState>(idleEditorState)
  const [modelsState, setModelsState] = useState<GroupEditorState>(idleEditorState)
  const [deletePending, setDeletePending] = useState(false)
  const settingsEditorRef = useRef<GroupEditorHandle>(null)
  const modelsEditorRef = useRef<GroupModelsEditorHandle>(null)

  const unifiedDirty = settingsState.dirty || modelsState.dirty
  const unifiedPending = settingsState.pending || modelsState.pending || deletePending
  const unifiedInvalid = settingsState.invalid === true || (modelsState.invalidRowCount ?? 0) > 0
  const unifiedError = settingsState.error || modelsState.error
  const unifiedSaved = !unifiedDirty && (settingsState.saved || modelsState.saved)
  const unifiedSaveStatus: 'idle' | 'saved' | 'error' = unifiedError
    ? 'error'
    : unifiedSaved
      ? 'saved'
      : 'idle'
  const unifiedErrorActionLabel = modelsState.invalidRowCount
    ? t('group.modelEditor.locateFirstInvalid')
    : ''

  function saveUnified(): void {
    if (settingsState.dirty) settingsEditorRef.current?.requestSave()
    if (modelsState.dirty) modelsEditorRef.current?.requestSave()
  }

  function discardUnified(): void {
    if (settingsState.dirty) settingsEditorRef.current?.discard()
    if (modelsState.dirty) modelsEditorRef.current?.discard()
  }

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
              {managementOpen ? (
                <GroupCredentialsTab
                  key={`management-${groupId}`}
                  groupId={groupId}
                  channelId={summaryQuery.data.channel_id}
                  connectionType={summaryQuery.data.connection_type}
                />
              ) : (
                <>
                  <GroupSettingsTab
                    ref={settingsEditorRef}
                    key={`settings-${groupId}`}
                    groupId={groupId}
                    unified
                    blocked={deletePending}
                    onStateChange={setSettingsState}
                  />
                  <section
                    aria-label={t('group.credentials.title')}
                    {...stylex.props(styles.credentials)}
                  >
                    {credentialsQuery.data?.credential ? (
                      <div {...stylex.props(styles.credentialList)}>
                        {[credentialsQuery.data.credential].map((credential) => {
                          const summary = unifiedCredentialSummary(credential, t)
                          return (
                            <div key={credential.mask} {...stylex.props(styles.credentialRow)}>
                              {credential.connection_type === 'subscription' ? (
                                <label {...stylex.props(styles.credentialField)}>
                                  <span {...stylex.props(styles.credentialFieldLabel)}>
                                    {t('group.credentials.full.kind.account')}
                                  </span>
                                  <input
                                    {...stylex.props(styles.credentialFieldInput)}
                                    value={credential.mask}
                                    readOnly
                                    autoComplete="off"
                                  />
                                </label>
                              ) : (
                                <GroupApiKeyEditor
                                  groupId={groupId}
                                  credential={credential}
                                  disabled={unifiedPending}
                                />
                              )}
                              <span {...stylex.props(styles.credentialRowEnd)}>
                                <Badge
                                  variant={credentialStatusBadgeVariant(summary.status)}
                                  label={summary.label}
                                />
                              </span>
                            </div>
                          )
                        })}
                      </div>
                    ) : (
                      <p {...stylex.props(styles.empty)}>{t('group.unified.noCredentials')}</p>
                    )}
                  </section>
                  <GroupModelsTab
                    ref={modelsEditorRef}
                    key={`models-${groupId}`}
                    groupId={groupId}
                    channelId={summaryQuery.data.channel_id}
                    enabled={settingsQuery.data?.enabled ?? true}
                    unified
                    blocked={deletePending}
                    readonlyRouteFields
                    onStateChange={setModelsState}
                  />
                  {/* Unified-mode portal target: GroupSettingsTab portals its
                      advanced section here via createPortal. Contractual id. */}
                  <div id="group-settings-advanced-target" />
                  <StickySaveBar
                    appearance="ledger"
                    alwaysVisible
                    dirty={unifiedDirty}
                    pending={unifiedPending}
                    status={unifiedSaveStatus}
                    error={unifiedError}
                    errorActionLabel={unifiedErrorActionLabel}
                    onErrorAction={() => void modelsEditorRef.current?.focusFirstInvalid()}
                    statusContent={
                      <div>
                        <strong>
                          {unifiedPending
                            ? t('group.settings.saving')
                            : unifiedSaved
                              ? t('group.settings.savedFeedback')
                              : unifiedDirty
                                ? t('group.settings.unsaved')
                                : t('group.settings.saved')}
                        </strong>
                        <span>{unifiedError || t('group.settings.saveNote')}</span>
                      </div>
                    }
                    actions={
                      <>
                        <Button
                          variant="ghost"
                          size="sm"
                          label={t('common.discard')}
                          isDisabled={unifiedPending || !unifiedDirty}
                          onClick={discardUnified}
                        />
                        <GroupDeleteDialog
                          groupId={groupId}
                          groupName={summaryQuery.data.name}
                          disabled={unifiedPending || unifiedDirty}
                          onPendingChange={setDeletePending}
                        />
                        <Button
                          size="sm"
                          label={t('group.settings.save')}
                          isDisabled={unifiedPending || !unifiedDirty || unifiedInvalid}
                          onClick={saveUnified}
                        />
                      </>
                    }
                  />
                </>
              )}
            </>
          ) : null}
        </>
      )}
    </section>
  )
}
