import { Button, EmptyState, Skeleton } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { RefreshCw, TriangleAlert } from 'lucide-react'
import { useEffect, useImperativeHandle, useRef, useState, type Ref } from 'react'
import { createPortal } from 'react-dom'

import type { AccessProtocol, GroupSettingsDto } from '@shared/control/types'
import type { ChannelFieldDto } from '@shared/control/resources/channels'
import { channelsQueryOptions } from '@shared/control/resources/channels'
import {
  cacheGroupSettings,
  groupModelsQueryOptions,
  groupSettingsQueryOptions,
  invalidateGroupSettingsDependents,
  updateGroupSettings,
} from '@shared/control/resources/groups'
import { ApiError, RequestCancelledError } from '@shared/http/errors'
import { isValidUpstreamBaseURL } from '@shared/lib/upstream-base-url'

import { useStableLoading } from '../../../app/collection-loading'
import { useT } from '../../../app/i18n'
import { useAppServices } from '../../../app/services'
import { usePortalTarget } from '../../../app/use-portal-target'
import { useTransientFlag } from '../../../app/use-transient-flag'
import { useUnsavedChanges } from '../../../app/use-unsaved-changes'
import { InlineNotice } from '../../../components/InlineNotice'
import { SectionHeader } from '../../../components/SectionHeader'
import type { GroupEditorHandle, GroupEditorState } from '../editor-handles'
import { GroupSettingsAdvanced } from './GroupSettingsAdvanced'
import { GroupSettingsBaseForm } from './GroupSettingsBaseForm'
import { GroupSettingsSiteAccessSection } from './GroupSettingsSiteAccessSection'
import { GroupSettingsSaveBar } from './GroupSettingsSaveBar'
import { useGroupOverrideFlags, useGroupProxyDraft } from './use-group-settings-overrides'
import {
  useGroupSettingsFieldsMutators,
  useGroupSettingsOverrideMutators,
} from './use-group-settings-mutators'
import {
  buildGroupSettingsPatch,
  createGroupSettingsDraft,
  groupPolicyCountKeys,
  groupTimeoutKeys,
  requiresSiteRootConfirmation,
  extractSiteRootUrl,
  isValidProviderUrlForWrite,
  type GroupSettingsDraft,
} from '@shared/domain/groups/settings/group-settings-patch'

const parameterOverrideOperations = new Set([
  'chat_completion',
  'responses_create',
  'images_generate',
  'embeddings_create',
  'rerank',
])

/**
 * Classic features/groups/settings/GroupSettingsTab.vue — group settings draft
 * editor with runtime overrides, header rules, parameter overrides, and the
 * unified-mode advanced portal into #group-settings-advanced-target.
 */
export function GroupSettingsTab({
  groupId,
  unified = false,
  blocked = false,
  onStateChange,
  ref,
}: {
  groupId: number
  unified?: boolean
  blocked?: boolean
  onStateChange(state: GroupEditorState): void
  ref?: Ref<GroupEditorHandle>
}) {
  const t = useT()
  const { apiClient } = useAppServices()
  const queryClient = useQueryClient()
  const query = useQuery(groupSettingsQueryOptions(apiClient, groupId))
  const modelsQuery = useQuery(groupModelsQueryOptions(apiClient, groupId))
  const channelsQuery = useQuery(channelsQueryOptions(apiClient, ''))
  const initialLoading = useStableLoading(query.isPending && query.data === undefined)
  const queryRefreshing = query.data !== undefined && query.isFetching
  const [saved, setSaved] = useState<GroupSettingsDto | undefined>(undefined)
  const [draft, setDraft] = useState<GroupSettingsDraft | undefined>(undefined)
  const [pending, setPending] = useState(false)
  const [enabledPending, setEnabledPending] = useState(false)
  const [deletePending, setDeletePending] = useState(false)
  const [deleted, setDeleted] = useState(false)
  const [error, setError] = useState('')
  const [forceSiteRootConfirmation, setForceSiteRootConfirmation] = useState(false)
  const [headerRulesValid, setHeaderRulesValid] = useState(true)
  const [headerRulesInvalidEdits, setHeaderRulesInvalidEdits] = useState(false)
  const [headerRulesEditorRevision, setHeaderRulesEditorRevision] = useState(0)
  const [parameterOverridesValid, setParameterOverridesValid] = useState(true)
  const [parameterOverridesInvalidEdits, setParameterOverridesInvalidEdits] = useState(false)
  const [parameterOverridesEditorRevision, setParameterOverridesEditorRevision] = useState(0)
  const {
    value: savedFeedback,
    clear: clearSavedFeedback,
    show: showSavedFeedback,
  } = useTransientFlag(1_600)
  const controllerRef = useRef<AbortController | undefined>(undefined)

  const headerActionsTarget = usePortalTarget('group-header-actions')
  const advancedTarget = usePortalTarget('group-settings-advanced-target')

  const timeoutKeys = groupTimeoutKeys
  const policyCountKeys = groupPolicyCountKeys

  const selectedChannel = channelsQuery.data?.items.find(
    ({ channel_id }) => channel_id === draft?.channel_id,
  )
  const channelParamFields: ChannelFieldDto[] = selectedChannel?.param_fields ?? []
  const channelParamsDisabled = selectedChannel === undefined
  const proxySupported = selectedChannel?.capabilities.outbound_proxy ?? false
  const parameterOverrideProtocols: AccessProtocol[] = [
    ...new Set(
      (selectedChannel?.routes ?? [])
        .filter(({ operation }) => parameterOverrideOperations.has(operation))
        .map(({ client_protocol }) => client_protocol),
    ),
  ]

  const patch =
    saved && draft ? buildGroupSettingsPatch(saved, draft, forceSiteRootConfirmation) : {}
  const {
    proxyMode,
    proxyEndpoint,
    setProxyEndpoint,
    proxyState,
    proxyOverridden,
    proxyPendingRestore,
    proxyValue,
    toggleProxyOverride,
    resetProxyDraft,
  } = useGroupProxyDraft({ saved, proxySupported, t })

  const dirty =
    !deleted &&
    (Object.keys(patch).length > 0 ||
      headerRulesInvalidEdits ||
      parameterOverridesInvalidEdits ||
      proxyState.dirty)
  const mutationPending = pending || deletePending || blocked

  const nameError = draft?.name.trim() ? '' : t('group.settings.base.nameError')
  const paramErrors: Record<string, string> = {}
  for (const field of channelParamFields) {
    const value = draft?.params[field.key]?.trim() ?? ''
    if (field.required && !value) {
      paramErrors[field.key] = t('group.settings.base.paramRequired', { field: field.label })
    } else if (field.input_kind === 'url' && value && !isValidUpstreamBaseURL(value)) {
      paramErrors[field.key] = t('group.settings.base.upstreamUrlError')
    }
  }
  const timeoutValid = timeoutKeys.every((key) => {
    const value = draft?.overrides[key]
    return value === undefined || (Number.isSafeInteger(value) && value > 0)
  })
  const policyCountsValid = policyCountKeys.every((key) => {
    const value = draft?.overrides[key]
    return value === undefined || (Number.isSafeInteger(value) && value >= 0)
  })
  const rootConfirmationRequired =
    !!saved &&
    !!draft &&
    (forceSiteRootConfirmation || requiresSiteRootConfirmation(saved, draft)) &&
    !draft.clearSiteAccessToken &&
    !draft.siteAccessToken.trim() &&
    !draft.confirmSiteRootChange
  const siteAccessInvalid =
    !!saved &&
    !!draft &&
    ((saved.site_access.token_configured && !draft.clearSiteAccessToken) ||
      !!draft.siteAccessToken.trim()) &&
    (!draft.siteUserId.trim() || !extractSiteRootUrl(draft.provider_url ?? ''))
  const providerUrlInvalid =
    !!draft?.provider_url?.trim() &&
    (draft.provider_url !== saved?.provider_url ||
      (!draft.clearSiteAccessToken &&
        (!!draft.siteAccessToken.trim() ||
          (saved?.site_access.token_configured &&
            draft.siteUserId.trim() !== saved.site_access.user_id)))) &&
    !isValidProviderUrlForWrite(draft.provider_url)
  const valid =
    !providerUrlInvalid &&
    !siteAccessInvalid &&
    !rootConfirmationRequired &&
    !nameError &&
    Object.keys(paramErrors).length === 0 &&
    timeoutValid &&
    policyCountsValid &&
    headerRulesValid &&
    parameterOverridesValid &&
    !proxyState.invalid

  const {
    isPendingRestore,
    headerRulesOverridden,
    headerRulesPendingRestore,
    displayedHeaderRules,
    affinity,
    websocket,
    reasoningStatusFilter,
  } = useGroupOverrideFlags(saved, draft, t)

  const {
    updateParam,
    selectChannel,
    setTimeoutOverride,
    setTimeoutValue,
    setPolicyCountOverride,
    setPolicyCountValue,
    policyCountError,
  } = useGroupSettingsFieldsMutators({
    saved,
    draft,
    setDraft,
    isMutationPending: () => stateRef.current.mutationPending,
    t,
  })

  const {
    updateHeaderRules,
    toggleHeaderRulesOverride,
    updateParameterOverrides,
    toggleAffinityOverride,
    setAffinityValue,
    toggleWebsocketOverride,
    setWebsocketValue,
    toggleReasoningStatusFilterOverride,
    setReasoningStatusFilterValue,
  } = useGroupSettingsOverrideMutators({
    saved,
    setDraft,
    overridden: {
      headerRules: headerRulesOverridden,
      affinity: affinity.overridden,
      websocket: websocket.overridden,
      reasoningStatusFilter: reasoningStatusFilter.overridden,
    },
    onHeaderRulesReset: () => {
      setHeaderRulesValid(true)
      setHeaderRulesInvalidEdits(false)
      setHeaderRulesEditorRevision((rev) => rev + 1)
    },
  })

  function resetSavedDraft(settings: GroupSettingsDto): void {
    setSaved(settings)
    setDraft(createGroupSettingsDraft(settings))
    setHeaderRulesValid(true)
    setHeaderRulesInvalidEdits(false)
    setHeaderRulesEditorRevision((rev) => rev + 1)
    setParameterOverridesValid(true)
    setParameterOverridesInvalidEdits(false)
    setParameterOverridesEditorRevision((rev) => rev + 1)
    resetProxyDraft(settings.proxy.configured_mode)
  }

  // Latest-value refs read by effects and async continuations. Written in the
  // first passive effect of every commit (react-hooks/refs forbids render-time
  // writes); declared before the effects that consume them.
  const resetSavedDraftRef = useRef(resetSavedDraft)
  const stateRef = useRef({ dirty, mutationPending, deleted, blocked })
  const savedRef = useRef(saved)
  const consumeCurrentQueryRef = useRef<() => void>(() => {})

  function consumeCurrentQuery(): void {
    const latest = query.data
    if (
      !latest ||
      latest === savedRef.current ||
      stateRef.current.dirty ||
      stateRef.current.mutationPending ||
      stateRef.current.deleted
    ) {
      return
    }
    resetSavedDraftRef.current(latest)
  }

  const { dialog: unsavedChangesDialog } = useUnsavedChanges({
    dirty,
    blocked: mutationPending,
    allowRouteUpdate: (current, next) => {
      const sameGroup =
        current.routeId === next.routeId && String(current.params.id) === String(next.params.id)
      if (!sameGroup) return false
      const nextTab = typeof next.search.tab === 'string' ? next.search.tab : undefined
      const currentTab = typeof current.search.tab === 'string' ? current.search.tab : undefined
      if (unified) return nextTab !== 'credentials'
      return nextTab === 'settings' && currentTab === 'settings'
    },
  })

  const onStateChangeRef = useRef(onStateChange)

  // Sync pass — must run before every effect that reads the refs above.
  useEffect(() => {
    resetSavedDraftRef.current = resetSavedDraft
    stateRef.current = { dirty, mutationPending, deleted, blocked }
    savedRef.current = saved
    consumeCurrentQueryRef.current = consumeCurrentQuery
    onStateChangeRef.current = onStateChange
  })

  // Classic watch(query.data, {immediate}): hydrate the draft whenever fresh
  // settings land and the surface is idle.
  useEffect(() => {
    const latest = query.data
    if (!latest) return
    const { dirty: isDirty, mutationPending: isPending, deleted: isDeleted } = stateRef.current
    if (latest === savedRef.current || isDirty || isPending || isDeleted) return
    resetSavedDraftRef.current(latest)
  }, [query.data])

  // Classic watch(dirty): a fresh edit clears the saved flash; leaving dirty
  // re-consumes any settings that arrived while editing.
  const wasDirtyRef = useRef(dirty)
  useEffect(() => {
    if (dirty) clearSavedFeedback()
    else if (wasDirtyRef.current) consumeCurrentQueryRef.current()
    wasDirtyRef.current = dirty
  }, [dirty, clearSavedFeedback])

  // Classic watch([mutationPending, deleted]): re-consume after a mutation
  // settles or the group is deleted.
  const pendingSnapshotRef = useRef({ mutationPending, deleted })
  useEffect(() => {
    const last = pendingSnapshotRef.current
    pendingSnapshotRef.current = { mutationPending, deleted }
    if (last.mutationPending === mutationPending && last.deleted === deleted) return
    consumeCurrentQueryRef.current()
  }, [mutationPending, deleted])

  function requestSave(): void {
    if (!dirty || !valid || stateRef.current.mutationPending) return
    void save()
  }

  async function save(): Promise<void> {
    if (!saved || !draft || stateRef.current.mutationPending || !valid) return
    const active = new AbortController()
    controllerRef.current = active
    setPending(true)
    clearSavedFeedback()
    setError('')
    try {
      const body = {
        ...patch,
        ...(proxyState.dirty && proxyState.value !== undefined ? { proxy: proxyState.value } : {}),
      }
      const result = await updateGroupSettings(apiClient, groupId, body, active.signal)
      if (controllerRef.current !== active) return
      setForceSiteRootConfirmation(false)
      resetSavedDraft(result)
      cacheGroupSettings(queryClient, groupId, result)
      await invalidateGroupSettingsDependents(queryClient, groupId)
      showSavedFeedback()
    } catch (cause: unknown) {
      if (cause instanceof RequestCancelledError || controllerRef.current !== active) return
      if (cause instanceof ApiError && cause.code === 'SITE_ROOT_CHANGE_CONFIRMATION_REQUIRED') {
        setForceSiteRootConfirmation(true)
        setDraft((current) => (current ? { ...current, confirmSiteRootChange: false } : current))
        setError(t('siteAccess.confirmError'))
      } else setError(t('group.settings.saveFailed'))
    } finally {
      if (controllerRef.current === active) {
        controllerRef.current = undefined
        setPending(false)
      }
    }
  }

  function discard(): void {
    if (!saved || stateRef.current.mutationPending) return
    setError('')
    setForceSiteRootConfirmation(false)
    clearSavedFeedback()
    resetSavedDraft(saved)
    consumeCurrentQuery()
  }

  // 分组总闸不再走表单草稿：它是立即生效的动作，语义与调度页条目开关一致。
  // 成功后只同步 enabled 字段，保留用户在表单里尚未保存的其它改动。
  async function setGroupEnabled(next: boolean): Promise<void> {
    const base = saved
    if (!base || enabledPending || next === base.enabled) return
    setEnabledPending(true)
    setError('')
    try {
      const result = await updateGroupSettings(apiClient, groupId, { enabled: next })
      setSaved(result)
      setDraft((current) => (current ? { ...current, enabled: result.enabled } : current))
      cacheGroupSettings(queryClient, groupId, result)
      await invalidateGroupSettingsDependents(queryClient, groupId)
    } catch {
      setError(t('group.settings.saveFailed'))
    } finally {
      setEnabledPending(false)
    }
  }

  function onDeleted(): void {
    setDeleted(true)
    setError('')
  }

  function headerSummary(): string {
    return t('group.settings.runtime.headerSummary', {
      set: Object.keys(displayedHeaderRules.set).length,
      remove: displayedHeaderRules.remove.length,
    })
  }

  useImperativeHandle(ref, () => ({ requestSave, discard }))

  // Classic watch([dirty, mutationPending, error, savedFeedback, valid]):
  // report aggregate state to the host's unified save bar.
  useEffect(() => {
    onStateChangeRef.current({
      dirty,
      pending: mutationPending,
      error,
      saved: savedFeedback,
      invalid: !valid,
    })
  }, [dirty, mutationPending, error, savedFeedback, valid])

  useEffect(
    () => () => {
      controllerRef.current?.abort()
    },
    [],
  )

  const advanced = saved && draft && (
    <GroupSettingsAdvanced
      saved={saved}
      draft={draft}
      pending={mutationPending}
      proxy={{
        mode: proxyMode,
        endpoint: proxyEndpoint,
        supported: proxySupported,
        overridden: proxyOverridden,
        pendingRestore: proxyPendingRestore,
        value: proxyValue,
        disabled: mutationPending || selectedChannel === undefined || !proxySupported,
      }}
      onProxyModeChange={resetProxyDraft}
      onProxyEndpointChange={setProxyEndpoint}
      onToggleProxy={toggleProxyOverride}
      websocket={{
        overridden: websocket.overridden,
        pendingRestore: websocket.pendingRestore,
        label: websocket.label,
      }}
      onToggleWebsocket={toggleWebsocketOverride}
      onSetWebsocket={setWebsocketValue}
      reasoningStatusFilter={{
        overridden: reasoningStatusFilter.overridden,
        label: reasoningStatusFilter.label,
      }}
      onToggleReasoningStatusFilter={toggleReasoningStatusFilterOverride}
      onSetReasoningStatusFilter={setReasoningStatusFilterValue}
      affinity={{
        overridden: affinity.overridden,
        pendingRestore: affinity.pendingRestore,
        label: affinity.label,
      }}
      onToggleAffinity={toggleAffinityOverride}
      onSetAffinity={setAffinityValue}
      isPendingRestore={isPendingRestore}
      onSetTimeoutOverride={setTimeoutOverride}
      onSetTimeoutValue={setTimeoutValue}
      onSetPolicyCountOverride={setPolicyCountOverride}
      onSetPolicyCountValue={setPolicyCountValue}
      onPolicyCountError={policyCountError}
      parameterOverrides={{
        revision: parameterOverridesEditorRevision,
        protocols: parameterOverrideProtocols,
        models: modelsQuery.data?.items ?? [],
        onValidChange: setParameterOverridesValid,
        onInvalidEditsChange: setParameterOverridesInvalidEdits,
        onChange: updateParameterOverrides,
      }}
      headerRules={{
        overridden: headerRulesOverridden,
        pendingRestore: headerRulesPendingRestore,
        revision: headerRulesEditorRevision,
        value: displayedHeaderRules,
        summary: headerSummary(),
        onToggle: toggleHeaderRulesOverride,
        onValidChange: setHeaderRulesValid,
        onInvalidEditsChange: setHeaderRulesInvalidEdits,
        onChange: updateHeaderRules,
      }}
    />
  )

  return (
    <section
      {...stylex.props(styles.root)}
      aria-labelledby={unified ? undefined : 'group-settings-heading'}
      aria-label={unified ? t('group.settings.title') : undefined}
    >
      {!unified && (
        <SectionHeader headingId="group-settings-heading" title={t('group.settings.title')} />
      )}

      {queryRefreshing && (
        <div
          role="status"
          aria-label={t('group.settings.loading')}
          {...stylex.props(styles.refreshing)}
        >
          <RefreshCw size={13} aria-hidden />
        </div>
      )}

      {(query.isPending && !query.data) || initialLoading ? (
        <div
          role="status"
          aria-label={t('group.settings.loading')}
          {...stylex.props(styles.skeleton)}
        >
          <Skeleton height={42} radius={2} />
          <Skeleton height={42} radius={2} />
          <Skeleton height={42} radius={2} />
          <Skeleton height={120} radius={2} />
        </div>
      ) : query.isError && !query.data ? (
        <div role="alert">
          <EmptyState
            title={t('group.settings.loadFailed')}
            icon={<TriangleAlert size={20} />}
            actions={
              <Button
                variant="secondary"
                size="sm"
                label={t('common.retry')}
                onClick={() => void query.refetch()}
              />
            }
          />
        </div>
      ) : saved && draft ? (
        <>
          {error !== '' && <InlineNotice tone="danger">{error}</InlineNotice>}
          {siteAccessInvalid && (
            <InlineNotice tone="danger">{t('siteAccess.required')}</InlineNotice>
          )}
          {rootConfirmationRequired && (
            <InlineNotice tone="danger">{t('siteAccess.confirmError')}</InlineNotice>
          )}
          {channelParamsDisabled && (
            <InlineNotice
              tone="danger"
              action={
                <Button
                  variant="secondary"
                  size="sm"
                  label={t('common.retry')}
                  onClick={() => void channelsQuery.refetch()}
                />
              }
            >
              {t('group.settings.base.channelCatalogUnavailable')}
            </InlineNotice>
          )}
          <div {...stylex.props(styles.layout)}>
            <div {...stylex.props(styles.content)}>
              <GroupSettingsBaseForm
                section="general"
                showTitle={!unified}
                showDescription={!unified}
                unified={unified}
                channelId={draft.channel_id}
                channels={channelsQuery.data?.items ?? []}
                selectedChannel={selectedChannel ?? null}
                channelsLoading={channelsQuery.isFetching}
                channelsError={channelsQuery.isError}
                paramFields={channelParamFields}
                params={draft.params}
                name={draft.name}
                providerUrl={draft.provider_url}
                providerUrlInvalid={providerUrlInvalid}

                enabled={saved.enabled}
                enabledPending={enabledPending}
                pending={mutationPending}
                paramsDisabled={channelParamsDisabled}
                nameError={nameError}
                paramErrors={paramErrors}
                headerActionsTarget={unified ? headerActionsTarget : null}
                onParamChange={updateParam}
                onChannelSelect={selectChannel}
                onChannelsRetry={() => void channelsQuery.refetch()}
                onNameChange={(value) =>
                  setDraft((current) => (current ? { ...current, name: value } : current))
                }
                onProviderUrlChange={(value) =>
                  setDraft((current) =>
                    current
                      ? { ...current, provider_url: value, confirmSiteRootChange: false }
                      : current,
                  )
                }

                onSetEnabled={(value) => void setGroupEnabled(value)}
              />
              <GroupSettingsSiteAccessSection
                groupId={groupId}
                saved={saved}
                draft={draft}
                access={query.data?.site_access ?? saved.site_access}
                forceConfirmation={forceSiteRootConfirmation}
                disabled={mutationPending}
                onChange={(value) =>
                  setDraft((current) => (current ? { ...current, ...value } : current))
                }
              />
              {unified && advancedTarget ? createPortal(advanced, advancedTarget) : advanced}
            </div>
          </div>
          {!unified && (
            <GroupSettingsSaveBar
              groupId={groupId}
              groupName={saved.name}
              dirty={dirty}
              pending={pending}
              blocked={mutationPending}
              error={error}
              savedFeedback={savedFeedback}
              valid={valid}
              deletePending={deletePending}
              deleted={deleted}
              onDeletePendingChange={setDeletePending}
              onDeleted={onDeleted}
              onDiscard={discard}
              onSave={requestSave}
            />
          )}
        </>
      ) : null}
      {unsavedChangesDialog}
    </section>
  )
}

const styles = stylex.create({
  root: {
    display: 'grid',
    gap: 0,
    minWidth: 0,
    paddingTop: {
      default: 'var(--detail-panel-padding-top)',
      '@media (max-width: 860px)': 'var(--detail-panel-padding-top-compact)',
    },
  },
  refreshing: {
    display: 'flex',
    justifyContent: 'flex-end',
    color: 'var(--color-text-faint)',
    paddingBottom: 'var(--space-2)',
  },
  skeleton: {
    display: 'grid',
    gap: 'var(--space-3)',
  },
  layout: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr)',
    alignItems: 'start',
    marginTop: 'var(--space-3)',
  },
  content: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-4)',
  },
})
