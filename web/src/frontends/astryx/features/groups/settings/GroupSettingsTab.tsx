import { Button, EmptyState, Skeleton, Switch, TextInput } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { RefreshCw, TriangleAlert } from 'lucide-react'
import { useEffect, useImperativeHandle, useRef, useState, type ReactNode, type Ref } from 'react'
import { createPortal } from 'react-dom'

import type {
  AccessProtocol,
  GroupSettingsDto,
  HeaderRulesDto,
  ParameterOverrideRuleDto,
  ProxyConfiguredMode,
} from '@shared/control/types'
import type { ChannelDto, ChannelFieldDto } from '@shared/control/resources/channels'
import { channelsQueryOptions } from '@shared/control/resources/channels'
import { proxyDraftState, proxyOverrideToggleMode } from '@shared/control/resources/proxy'
import {
  cacheGroupSettings,
  groupModelsQueryOptions,
  groupSettingsQueryOptions,
  invalidateGroupSettingsDependents,
  updateGroupSettings,
} from '@shared/control/resources/groups'
import { RequestCancelledError } from '@shared/http/errors'
import { isValidUpstreamBaseURL } from '@shared/lib/upstream-base-url'
import { isValidPriceMultiplier } from '@shared/lib/price-multiplier'
import type { MessageId } from '@shared/i18n/message-ids'

import { useStableLoading } from '../../../app/collection-loading'
import { useT } from '../../../app/i18n'
import { useAppServices } from '../../../app/services'
import { usePortalTarget } from '../../../app/use-portal-target'
import { useTransientFlag } from '../../../app/use-transient-flag'
import { useUnsavedChanges } from '../../../app/use-unsaved-changes'
import { HeaderRulesEditor } from '../../../components/HeaderRulesEditor'
import { InlineNotice } from '../../../components/InlineNotice'
import { ProxyOverrideControl } from '../../../components/ProxyOverrideControl'
import { SettingBlock, SettingRow } from '../../../components/setting-chrome'
import { SectionHeader } from '../../../components/SectionHeader'
import { StickySaveBar } from '../../../components/StickySaveBar'
import type { GroupEditorHandle, GroupEditorState } from '../editor-handles'
import { GroupDeleteDialog } from './GroupDeleteDialog'
import { GroupSettingsBaseForm } from './GroupSettingsBaseForm'
import { ParameterOverrideRulesEditor } from './ParameterOverrideRulesEditor'
import {
  buildGroupSettingsPatch,
  createGroupSettingsDraft,
  groupPolicyCountKeys,
  groupTimeoutKeys,
  preserveChannelParams,
  setGroupConfigOverride,
  setGroupPolicyCountOverride,
  type GroupPolicyCountKey,
  type GroupSettingsDraft,
  type GroupTimeoutKey,
} from '@shared/domain/groups/settings/group-settings-patch'

const policyRows = [
  {
    key: 'blacklist_threshold',
    helpKey: 'blacklistThresholdHelp',
  },
] as const

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
  const [headerRulesValid, setHeaderRulesValid] = useState(true)
  const [headerRulesInvalidEdits, setHeaderRulesInvalidEdits] = useState(false)
  const [headerRulesEditorRevision, setHeaderRulesEditorRevision] = useState(0)
  const [parameterOverridesValid, setParameterOverridesValid] = useState(true)
  const [parameterOverridesInvalidEdits, setParameterOverridesInvalidEdits] = useState(false)
  const [parameterOverridesEditorRevision, setParameterOverridesEditorRevision] = useState(0)
  const [proxyMode, setProxyMode] = useState<ProxyConfiguredMode>('inherit')
  const [proxyEndpoint, setProxyEndpoint] = useState('')
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
  const channelParamFields: ChannelFieldDto[] =
    selectedChannel?.connection.type === 'subscription' ? [] : (selectedChannel?.param_fields ?? [])
  const channelParamsDisabled = selectedChannel === undefined
  const parameterOverrideProtocols: AccessProtocol[] = [
    ...new Set(
      (selectedChannel?.routes ?? [])
        .filter(({ operation }) => parameterOverrideOperations.has(operation))
        .map(({ client_protocol }) => client_protocol),
    ),
  ]

  const patch = saved && draft ? buildGroupSettingsPatch(saved, draft) : {}
  const proxyState = saved
    ? proxyDraftState(saved.proxy, proxyMode, proxyEndpoint)
    : { dirty: false, invalid: false, value: undefined }
  // 代理沿用其它设置项的覆盖语义：inherit 即“继承全局”，direct/custom 即“本分组覆盖”。
  const proxyOverridden = proxyMode !== 'inherit'
  const proxyPendingRestore = saved?.proxy.configured_mode !== 'inherit' && proxyMode === 'inherit'
  const proxyEffectiveLabel = (() => {
    const view = saved?.proxy
    if (!view) return ''
    return view.display_url ?? t(`common.proxy.mode.${view.effective_mode}` as MessageId)
  })()
  const proxySupported = selectedChannel?.capabilities.outbound_proxy ?? false
  const proxyValue = (() => {
    if (!proxySupported) return t('common.proxy.unsupported')
    if (proxyPendingRestore) return t('group.settings.runtime.resetPending')
    return proxyEffectiveLabel
  })()

  function toggleProxyOverride(): void {
    const base = saved?.proxy
    if (base) setProxyMode(proxyOverrideToggleMode(base, proxyOverridden))
    setProxyEndpoint('')
  }

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
  const valid =
    !nameError &&
    Object.keys(paramErrors).length === 0 &&
    isValidPriceMultiplier(draft?.price_multiplier ?? '') &&
    timeoutValid &&
    policyCountsValid &&
    headerRulesValid &&
    parameterOverridesValid &&
    !proxyState.invalid

  function isPendingRestore(key: GroupTimeoutKey | GroupPolicyCountKey): boolean {
    return draft?.overrides[key] === undefined && saved?.overrides[key] !== undefined
  }

  const headerRulesOverridden = draft?.overrides.header_rules !== undefined
  const headerRulesPendingRestore =
    !headerRulesOverridden && saved?.overrides.header_rules !== undefined
  const displayedHeaderRules: HeaderRulesDto = (() => {
    if (draft?.overrides.header_rules !== undefined) return draft.overrides.header_rules
    if (headerRulesPendingRestore) return { set: {}, remove: [] }
    return saved?.effective.header_rules ?? { set: {}, remove: [] }
  })()
  const affinityOverridden = draft?.overrides.affinity_enabled !== undefined
  const affinityPendingRestore =
    !affinityOverridden && saved?.overrides.affinity_enabled !== undefined
  const affinityEnabledLabel = saved?.effective.affinity_enabled
    ? t('group.settings.runtime.enabledValue')
    : t('group.settings.runtime.disabledValue')
  const websocketOverridden = draft?.overrides.responses_websocket_enabled !== undefined
  const websocketPendingRestore =
    !websocketOverridden && saved?.overrides.responses_websocket_enabled !== undefined
  const websocketEnabledLabel = saved?.effective.responses_websocket_enabled
    ? t('group.settings.runtime.enabledValue')
    : t('group.settings.runtime.disabledValue')
  const reasoningStatusFilterOverridden =
    draft?.overrides.responses_reasoning_status_filter_enabled !== undefined
  const reasoningStatusFilterEnabledLabel = saved?.effective
    .responses_reasoning_status_filter_enabled
    ? t('group.settings.runtime.enabledValue')
    : t('group.settings.runtime.disabledValue')

  function resetSavedDraft(settings: GroupSettingsDto): void {
    setSaved(settings)
    setDraft(createGroupSettingsDraft(settings))
    setHeaderRulesValid(true)
    setHeaderRulesInvalidEdits(false)
    setHeaderRulesEditorRevision((rev) => rev + 1)
    setParameterOverridesValid(true)
    setParameterOverridesInvalidEdits(false)
    setParameterOverridesEditorRevision((rev) => rev + 1)
    setProxyMode(settings.proxy.configured_mode)
    setProxyEndpoint('')
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

  function updateParam(key: string, value: string | null): void {
    setDraft((current) => {
      if (!current) return current
      const params = { ...current.params }
      if (value === null) delete params[key]
      else params[key] = value
      return { ...current, params }
    })
  }

  function selectChannel(channel: ChannelDto): void {
    if (stateRef.current.mutationPending) return
    setDraft((current) => {
      if (!current) return current
      const params = preserveChannelParams(current.params, channel.param_fields)
      return {
        ...current,
        channel_id: channel.channel_id,
        connection_type: channel.connection.type,
        params,
      }
    })
  }

  function setTimeoutOverride(key: GroupTimeoutKey, enabled: boolean): void {
    if (!saved) return
    const effective = saved.effective[key]
    setDraft((current) =>
      current ? setGroupConfigOverride(current, key, enabled, effective) : current,
    )
  }

  function setTimeoutValue(key: GroupTimeoutKey, value: string): void {
    setDraft((current) =>
      current ? { ...current, overrides: { ...current.overrides, [key]: Number(value) } } : current,
    )
  }

  function setPolicyCountOverride(key: GroupPolicyCountKey, enabled: boolean): void {
    if (!saved) return
    const effective = saved.effective[key]
    setDraft((current) =>
      current ? setGroupPolicyCountOverride(current, key, enabled, effective) : current,
    )
  }

  function setPolicyCountValue(key: GroupPolicyCountKey, value: string): void {
    setDraft((current) =>
      current ? { ...current, overrides: { ...current.overrides, [key]: Number(value) } } : current,
    )
  }

  function policyCountError(key: GroupPolicyCountKey): string | undefined {
    const value = draft?.overrides[key]
    return value !== undefined && (!Number.isSafeInteger(value) || value < 0)
      ? t('group.settings.runtime.nonNegativeIntegerError')
      : undefined
  }

  function updateHeaderRules(value: HeaderRulesDto): void {
    setDraft((current) =>
      current
        ? {
            ...current,
            overrides: {
              ...current.overrides,
              header_rules: { set: { ...value.set }, remove: [...value.remove] },
            },
          }
        : current,
    )
  }

  function toggleHeaderRulesOverride(): void {
    if (!saved) return
    setDraft((current) => {
      if (!current) return current
      const overrides = { ...current.overrides }
      if (headerRulesOverridden) {
        delete overrides.header_rules
      } else {
        overrides.header_rules = {
          set: { ...saved.effective.header_rules.set },
          remove: [...saved.effective.header_rules.remove],
        }
      }
      return { ...current, overrides }
    })
    setHeaderRulesValid(true)
    setHeaderRulesInvalidEdits(false)
    setHeaderRulesEditorRevision((rev) => rev + 1)
  }

  function updateParameterOverrides(value: ParameterOverrideRuleDto[]): void {
    setDraft((current) => {
      if (!current) return current
      const overrides = { ...current.overrides }
      if (value.length === 0) delete overrides.parameter_overrides
      else overrides.parameter_overrides = value
      return { ...current, overrides }
    })
  }

  function toggleAffinityOverride(): void {
    if (!saved) return
    setDraft((current) => {
      if (!current) return current
      const overrides = { ...current.overrides }
      if (affinityOverridden) delete overrides.affinity_enabled
      else overrides.affinity_enabled = saved.effective.affinity_enabled
      return { ...current, overrides }
    })
  }

  function setAffinityValue(value: boolean): void {
    setDraft((current) =>
      current
        ? { ...current, overrides: { ...current.overrides, affinity_enabled: value } }
        : current,
    )
  }

  function toggleWebsocketOverride(): void {
    if (!saved) return
    setDraft((current) => {
      if (!current) return current
      const overrides = { ...current.overrides }
      if (websocketOverridden) delete overrides.responses_websocket_enabled
      else overrides.responses_websocket_enabled = saved.effective.responses_websocket_enabled
      return { ...current, overrides }
    })
  }

  function setWebsocketValue(value: boolean): void {
    setDraft((current) =>
      current
        ? {
            ...current,
            overrides: { ...current.overrides, responses_websocket_enabled: value },
          }
        : current,
    )
  }

  function toggleReasoningStatusFilterOverride(): void {
    if (!saved) return
    setDraft((current) => {
      if (!current) return current
      const overrides = { ...current.overrides }
      if (reasoningStatusFilterOverridden)
        delete overrides.responses_reasoning_status_filter_enabled
      else {
        overrides.responses_reasoning_status_filter_enabled =
          saved.effective.responses_reasoning_status_filter_enabled
      }
      return { ...current, overrides }
    })
  }

  function setReasoningStatusFilterValue(value: boolean): void {
    setDraft((current) =>
      current
        ? {
            ...current,
            overrides: {
              ...current.overrides,
              responses_reasoning_status_filter_enabled: value,
            },
          }
        : current,
    )
  }

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
      resetSavedDraft(result)
      cacheGroupSettings(queryClient, groupId, result)
      await invalidateGroupSettingsDependents(queryClient, groupId)
      showSavedFeedback()
    } catch (cause: unknown) {
      if (cause instanceof RequestCancelledError || controllerRef.current !== active) return
      setError(t('group.settings.saveFailed'))
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

  const advancedContent: ReactNode = saved && draft && (
    <details {...stylex.props(styles.advanced)}>
      <summary {...stylex.props(styles.advancedSummary)}>{t('group.settings.advanced')}</summary>
      <div {...stylex.props(styles.advancedContent)}>
        <div {...stylex.props(styles.advancedSections)}>
          <section id="settings-runtime" {...stylex.props(styles.section)}>
            <header>
              <h3 {...stylex.props(styles.sectionTitle)}>{t('group.settings.sections.runtime')}</h3>
              <p {...stylex.props(styles.sectionDescription)}>
                {t('group.settings.runtime.description')}
              </p>
            </header>
            <div {...stylex.props(styles.runtime)}>
              <SettingRow
                label={t('group.settings.runtime.responses_websocket_enabled')}
                value={
                  websocketPendingRestore
                    ? t('group.settings.runtime.resetPending')
                    : websocketEnabledLabel
                }
                help={t('group.settings.runtime.websocketHelp')}
                sourceLabel={
                  websocketOverridden
                    ? t('group.settings.runtime.override')
                    : websocketPendingRestore
                      ? t('group.settings.runtime.pendingRestoreSource')
                      : t('group.settings.runtime.inherited')
                }
                actionLabel={
                  websocketOverridden
                    ? t('group.settings.runtime.useInherited')
                    : t('group.settings.runtime.useOverride')
                }
                overridden={websocketOverridden}
                pendingRestore={websocketPendingRestore}
                disabled={mutationPending}
                onToggle={toggleWebsocketOverride}
                control={
                  <Switch
                    size="sm"
                    value={draft.overrides.responses_websocket_enabled ?? false}
                    isDisabled={mutationPending}
                    label={t('group.settings.runtime.responses_websocket_enabled')}
                    isLabelHidden
                    onChange={setWebsocketValue}
                  />
                }
              />
              <SettingRow
                label={t('group.settings.runtime.responses_reasoning_status_filter_enabled')}
                value={reasoningStatusFilterEnabledLabel}
                help={t('group.settings.runtime.reasoningStatusFilterHelp')}
                sourceLabel={
                  reasoningStatusFilterOverridden
                    ? t('group.settings.runtime.override')
                    : t('group.settings.runtime.groupDefault')
                }
                actionLabel={
                  reasoningStatusFilterOverridden
                    ? t('group.settings.runtime.useDefault')
                    : t('group.settings.runtime.useOverride')
                }
                overridden={reasoningStatusFilterOverridden}
                disabled={mutationPending}
                onToggle={toggleReasoningStatusFilterOverride}
                control={
                  <Switch
                    size="sm"
                    value={draft.overrides.responses_reasoning_status_filter_enabled ?? false}
                    isDisabled={mutationPending}
                    label={t('group.settings.runtime.responses_reasoning_status_filter_enabled')}
                    isLabelHidden
                    onChange={setReasoningStatusFilterValue}
                  />
                }
              />
              <SettingRow
                label={t('common.proxy.title')}
                value={proxyValue}
                help={proxySupported ? undefined : t('common.proxy.unsupportedHelp')}
                sourceLabel={
                  !proxySupported
                    ? t('common.proxy.unsupportedBadge')
                    : proxyOverridden
                      ? t('group.settings.runtime.override')
                      : proxyPendingRestore
                        ? t('group.settings.runtime.pendingRestoreSource')
                        : t('group.settings.runtime.inherited')
                }
                actionLabel={
                  proxyOverridden
                    ? t('group.settings.runtime.useInherited')
                    : t('group.settings.runtime.useOverride')
                }
                overridden={proxySupported && proxyOverridden}
                pendingRestore={proxySupported && proxyPendingRestore}
                locked={!proxySupported}
                disabled={mutationPending || selectedChannel === undefined || !proxySupported}
                onToggle={toggleProxyOverride}
                control={
                  <ProxyOverrideControl
                    base={saved.proxy}
                    mode={proxyMode}
                    endpoint={proxyEndpoint}
                    disabled={mutationPending}
                    onModeChange={setProxyMode}
                    onEndpointChange={setProxyEndpoint}
                  />
                }
              />
              {timeoutKeys.map((key) => (
                <SettingRow
                  key={key}
                  label={t(`group.settings.runtime.${key}` as MessageId)}
                  value={
                    isPendingRestore(key)
                      ? t('group.settings.runtime.resetPending')
                      : t('group.settings.runtime.effective', {
                          value: saved.effective[key],
                        })
                  }
                  sourceLabel={
                    draft.overrides[key] !== undefined
                      ? t('group.settings.runtime.override')
                      : isPendingRestore(key)
                        ? t('group.settings.runtime.pendingRestoreSource')
                        : t('group.settings.runtime.inherited')
                  }
                  actionLabel={
                    draft.overrides[key] === undefined
                      ? t('group.settings.runtime.useOverride')
                      : t('group.settings.runtime.useInherited')
                  }
                  overridden={draft.overrides[key] !== undefined}
                  pendingRestore={isPendingRestore(key)}
                  disabled={mutationPending}
                  onToggle={() => setTimeoutOverride(key, draft.overrides[key] === undefined)}
                  control={
                    <div {...stylex.props(styles.runtimeInput)}>
                      <TextInput
                        type="text"
                        size="sm"
                        value={String(draft.overrides[key])}
                        label={t('group.settings.runtime.valueFor', {
                          field: t(`group.settings.runtime.${key}` as MessageId),
                        })}
                        isLabelHidden
                        data-gptload-mono
                        isDisabled={mutationPending}
                        onChange={(value) => setTimeoutValue(key, value)}
                      />
                      <span aria-hidden="true" {...stylex.props(styles.runtimeUnit)}>
                        {t('group.settings.runtime.seconds')}
                      </span>
                    </div>
                  }
                />
              ))}
              {policyRows.map((policy) => (
                <SettingRow
                  key={policy.key}
                  label={t(`group.settings.runtime.${policy.key}` as MessageId)}
                  value={
                    isPendingRestore(policy.key)
                      ? t('group.settings.runtime.resetPending')
                      : t('group.settings.runtime.effectiveCount', {
                          value: saved.effective[policy.key],
                        })
                  }
                  help={t(`group.settings.runtime.${policy.helpKey}` as MessageId)}
                  sourceLabel={
                    draft.overrides[policy.key] !== undefined
                      ? t('group.settings.runtime.override')
                      : isPendingRestore(policy.key)
                        ? t('group.settings.runtime.pendingRestoreSource')
                        : t('group.settings.runtime.inherited')
                  }
                  actionLabel={
                    draft.overrides[policy.key] === undefined
                      ? t('group.settings.runtime.useOverride')
                      : t('group.settings.runtime.useInherited')
                  }
                  overridden={draft.overrides[policy.key] !== undefined}
                  pendingRestore={isPendingRestore(policy.key)}
                  disabled={mutationPending}
                  onToggle={() =>
                    setPolicyCountOverride(policy.key, draft.overrides[policy.key] === undefined)
                  }
                  control={
                    <div {...stylex.props(styles.runtimeInput)}>
                      <TextInput
                        id={`group-settings-${policy.key}`}
                        type="text"
                        size="sm"
                        value={String(draft.overrides[policy.key])}
                        label={t('group.settings.runtime.valueFor', {
                          field: t(`group.settings.runtime.${policy.key}` as MessageId),
                        })}
                        isLabelHidden
                        data-gptload-mono
                        isDisabled={mutationPending}
                        status={
                          policyCountError(policy.key) === undefined
                            ? undefined
                            : {
                                type: 'error',
                                message: policyCountError(policy.key) ?? '',
                              }
                        }
                        statusVariant="tooltip"
                        onChange={(value) => setPolicyCountValue(policy.key, value)}
                      />
                      <span aria-hidden="true" {...stylex.props(styles.runtimeUnit)}>
                        {t('group.settings.runtime.countUnit')}
                      </span>
                    </div>
                  }
                />
              ))}
              <SettingRow
                label={t('group.settings.runtime.affinity_enabled')}
                value={
                  affinityPendingRestore
                    ? t('group.settings.runtime.resetPending')
                    : affinityEnabledLabel
                }
                help={t('group.settings.runtime.affinityHelp')}
                sourceLabel={
                  affinityOverridden
                    ? t('group.settings.runtime.override')
                    : affinityPendingRestore
                      ? t('group.settings.runtime.pendingRestoreSource')
                      : t('group.settings.runtime.inherited')
                }
                actionLabel={
                  affinityOverridden
                    ? t('group.settings.runtime.useInherited')
                    : t('group.settings.runtime.useOverride')
                }
                overridden={affinityOverridden}
                pendingRestore={affinityPendingRestore}
                divided={false}
                disabled={mutationPending}
                onToggle={toggleAffinityOverride}
                control={
                  <Switch
                    size="sm"
                    value={draft.overrides.affinity_enabled ?? false}
                    isDisabled={mutationPending}
                    label={t('group.settings.runtime.affinity_enabled')}
                    isLabelHidden
                    onChange={setAffinityValue}
                  />
                }
              />
            </div>
          </section>
          <section id="settings-parameters" {...stylex.props(styles.section)}>
            <header>
              <h3 {...stylex.props(styles.sectionTitle)}>
                {t('group.settings.sections.parameters')}
              </h3>
              <p {...stylex.props(styles.sectionDescription)}>
                {t('group.settings.parameterOverrides.description')}
              </p>
            </header>
            <ParameterOverrideRulesEditor
              resetKey={parameterOverridesEditorRevision}
              value={draft.overrides.parameter_overrides ?? []}
              protocols={parameterOverrideProtocols}
              models={modelsQuery.data?.items ?? []}
              disabled={mutationPending}
              onValidChange={setParameterOverridesValid}
              onInvalidEditsChange={setParameterOverridesInvalidEdits}
              onChange={updateParameterOverrides}
            />
          </section>

          <section id="settings-headers" {...stylex.props(styles.section)}>
            <SettingBlock
              title={t('group.settings.sections.headers')}
              help={t('group.settings.headers.description')}
              meta={headerSummary()}
              sourceLabel={
                headerRulesOverridden
                  ? t('group.settings.runtime.override')
                  : headerRulesPendingRestore
                    ? t('group.settings.runtime.pendingRestoreSource')
                    : t('group.settings.runtime.inherited')
              }
              actionLabel={
                headerRulesOverridden
                  ? t('group.settings.runtime.useInherited')
                  : t('group.settings.runtime.useOverride')
              }
              overridden={headerRulesOverridden}
              pendingRestore={headerRulesPendingRestore}
              disabled={mutationPending}
              onToggle={() => void toggleHeaderRulesOverride()}
            >
              <HeaderRulesEditor
                resetKey={headerRulesEditorRevision}
                value={displayedHeaderRules}
                disabled={mutationPending || !headerRulesOverridden}
                showAdd={headerRulesOverridden}
                removeLabel={t('group.settings.runtime.headerRemove')}
                removeHint={t('group.settings.runtime.headerRemoveHint')}
                onValidChange={(value) => setHeaderRulesValid(value)}
                onInvalidEditsChange={(value) => setHeaderRulesInvalidEdits(value)}
                onChange={updateHeaderRules}
              />
            </SettingBlock>
          </section>
        </div>
      </div>
    </details>
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
                priceMultiplier={draft.price_multiplier}
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
                  setDraft((current) => (current ? { ...current, provider_url: value } : current))
                }
                onPriceMultiplierChange={(value) =>
                  setDraft((current) =>
                    current ? { ...current, price_multiplier: value } : current,
                  )
                }
                onSetEnabled={(value) => void setGroupEnabled(value)}
              />
              {unified && advancedTarget
                ? createPortal(advancedContent, advancedTarget)
                : advancedContent}
            </div>
          </div>
          {!unified && (
            <StickySaveBar
              appearance="ledger"
              alwaysVisible
              dirty={dirty}
              pending={mutationPending}
              status={error ? 'error' : savedFeedback ? 'saved' : 'idle'}
              error={error}
              statusContent={
                <div>
                  <strong>
                    {pending
                      ? t('group.settings.saving')
                      : savedFeedback
                        ? t('group.settings.savedFeedback')
                        : dirty
                          ? t('group.settings.unsaved')
                          : t('group.settings.saved')}
                  </strong>
                  <span>
                    {pending
                      ? t('group.settings.savingNote')
                      : savedFeedback
                        ? t('group.settings.savedFeedbackNote')
                        : dirty
                          ? t('group.settings.dirtyNote')
                          : t('group.settings.saveNote')}
                  </span>
                </div>
              }
              actions={
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    isDisabled={mutationPending || !dirty || deletePending}
                    onClick={discard}
                    label={t('common.discard')}
                  />
                  <GroupDeleteDialog
                    groupId={groupId}
                    groupName={saved.name}
                    disabled={mutationPending || dirty || deleted}
                    onPendingChange={setDeletePending}
                    onDeleted={onDeleted}
                  />
                  <Button
                    size="sm"
                    isDisabled={mutationPending || !dirty || !valid || deletePending}
                    onClick={requestSave}
                    label={t('group.settings.save')}
                  />
                </>
              }
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
  advanced: {
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 'var(--space-3)',
  },
  advancedSummary: {
    color: 'var(--color-text-muted)',
    cursor: 'pointer',
    fontSize: 'var(--text-sm)',
    fontWeight: 600,
  },
  advancedContent: {
    display: 'grid',
    gridTemplateColumns: '1fr',
    alignItems: 'start',
    gap: {
      default: 'var(--space-3)',
      '@media (max-width: 860px)': 'var(--space-2)',
    },
    marginTop: 'var(--space-3)',
  },
  advancedSections: {
    display: 'grid',
    minWidth: 0,
    gap: {
      default: 'var(--space-4)',
      '@media (max-width: 860px)': 'var(--space-3)',
    },
  },
  section: {
    display: 'grid',
    gap: '15px',
    scrollMarginTop: '76px',
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: '17px',
  },
  sectionTitle: {
    margin: 0,
    fontSize: 'var(--text-body)',
    fontWeight: 650,
  },
  sectionDescription: {
    maxWidth: '580px',
    marginTop: '3px',
    marginBottom: 0,
    marginLeft: 0,
    marginRight: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  runtime: {
    display: 'grid',
    gap: 'var(--space-1)',
  },
  runtimeInput: {
    display: 'flex',
    width: {
      default: 'min(100%, 190px)',
      '@media (max-width: 800px)': 'min(100%, 220px)',
    },
    minWidth: 0,
    alignItems: 'center',
    gap: '7px',
  },
  runtimeUnit: {
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: '11px',
    whiteSpace: 'nowrap',
  },
})
