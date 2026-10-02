import * as stylex from '@stylexjs/stylex'
import {
  Button,
  Dialog,
  DialogHeader,
  Layout,
  LayoutContent,
  LayoutFooter,
} from '@astryxdesign/core'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import { ArrowRight, Plus, RefreshCw } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { ApiError, RequestCancelledError } from '@shared/http/errors'
import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import { channelsQueryOptions, type ChannelDto } from '@shared/control/resources/channels'
import {
  getCredentialStage,
  type CredentialStage,
} from '@shared/control/resources/credential-stages'
import {
  createGroup,
  discoverModels,
  isSameTargetConflictData,
  readCredentialValidationData,
  type CredentialValidationData,
  type GroupCreateRequest,
  type ModelDiscoveryRequest,
  type SameTargetConflictData,
} from '@shared/control/resources/groups'
import type { ModelCandidate } from '@shared/control/resources/providers'
import { proxyMutation } from '@shared/control/resources/proxy'
import type { MessageId } from '@shared/i18n/message-ids'
import { isValidUpstreamBaseURL } from '@shared/lib/upstream-base-url'

import { constrainCollectionSearch } from '@shared/routing/route-query'
import type { SharedRouteQuery } from '@shared/routing/route-query'
import {
  parseImportRouteQuery,
  serializeImportRouteQuery,
  type ImportDiscoveryFilter,
  type ImportPanel,
  type ImportRouteState,
} from '@shared/routing/import-route'
import { pagePath } from '@shared/routing/page-routes'
import { readSingleCredential } from './single-credential-input'
import { mapConnectionToChannel, parseConnectionJSON } from '@shared/domain/import/connection-json'
import {
  appendSelectedCandidates,
  findModelNameConflicts,
  mergeCandidateMetadata,
  modelDraftValidity,
  readModelNameConflicts,
  type ModelAliasEditorLabels,
  type ModelDiscoveryDrawerLabels,
  type ModelNameConflict,
} from '@shared/domain/models/model-draft'
import {
  createDiscoveredModelDraft,
  toGroupModels,
  type ImportDraft,
  type ModelDraftItem,
} from '@shared/domain/import/model-draft'
import { presentSubscriptionErrorKey } from '@shared/domain/import/subscription-error-presenter'

import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'
import { useUnsavedChanges } from '../../app/use-unsaved-changes'
import { ChannelIcon } from '../../components/ChannelIcon'
import { InlineNotice } from '../../components/InlineNotice'
import { SectionHeader } from '../../components/SectionHeader'
import { StickySaveBar } from '../../components/StickySaveBar'
import { ModelAliasEditor, type ModelAliasEditorHandle } from '../groups/ModelAliasEditor'
import { ModelDiscoveryDrawer } from '../models/ModelDiscoveryDrawer'
import { ModelPricingStatus } from '../models/ModelPricingStatus'
import { ChannelPresetPicker } from './ChannelPresetPicker'
import { CredentialTextarea } from './CredentialTextarea'
import { ImportConnectionSection } from './ImportConnectionSection'
import { ImportOperationNotice } from './ImportOperationNotice'
import { SubscriptionCredentialStager } from './SubscriptionCredentialStager'
import { useImportOperationOwner, useOperationSnapshot } from './import-operation'

// Module scope keeps `Date.now()` reads out of render scope (react-hooks/purity).
function currentReadyStage(stage: CredentialStage | null): CredentialStage | null {
  return stage?.status === 'ready' && stage.expires_at_ms > Date.now() ? stage : null
}

function expireStaleReadyStage(stage: CredentialStage | null): CredentialStage | null {
  return stage?.status === 'ready' && stage.expires_at_ms <= Date.now()
    ? { ...stage, status: 'expired' }
    : stage
}

function freshDraft(): ImportDraft {
  return {
    mode: 'new',
    channel_id: '',
    connection_type: 'api_key',
    params: {},
    proxy: { mode: 'inherit', url: '' },
    name: '',
    provider_url: '',

    credentials: '',
    staged_credential: null,
    models: [],
  }
}

function cloneDraft(source: ImportDraft): ImportDraft {
  return {
    ...source,
    params: { ...source.params },
    proxy: { ...source.proxy },
    staged_credential:
      source.staged_credential === null ? null : structuredClone(source.staged_credential),
    models: source.models.map((model) => ({ ...model, sources: [...model.sources] })),
  }
}

function sameTargetConflict(cause: unknown): SameTargetConflictData | null {
  return cause instanceof ApiError &&
    cause.code === 'CHANNEL_TARGET_CONFLICT' &&
    isSameTargetConflictData(cause.data)
    ? cause.data
    : null
}

function initialChannelParams(channel: ChannelDto): Record<string, string> {
  return Object.fromEntries(
    channel.param_fields
      .filter(({ required, default_value: defaultValue }) => required || defaultValue !== null)
      .map(({ key, default_value: defaultValue }) => [key, defaultValue ?? '']),
  )
}

type ImportStepState = 'pending' | 'active' | 'ready' | 'error' | 'optional'

/**
 * Classic features/import/NewGroupImport.vue — three-step create flow:
 * channel preset → credentials (api_key textarea or subscription stager) →
 * model draft, then a stable idempotent create operation with same-target
 * conflict recovery and sessionStorage draft recovery.
 */
export function NewGroupImport({ initialDraft }: { initialDraft?: ImportDraft | null }) {
  const services = useAppServices()
  const apiClient = services.apiClient
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const t = useT()
  const toast = services.toast
  const { rawSearch } = useRouterState({
    select: (state) => ({ rawSearch: state.location.search as SharedRouteQuery }),
  })
  const routeState = parseImportRouteQuery(rawSearch)

  const owner = useImportOperationOwner()
  const createOperation = owner.createGroup
  const createSnap = useOperationSnapshot(createOperation)

  // Classic setup: a confirmed outcome clears before first paint — confirmed
  // never maps to a notice key, so a mount-effect reset is invisible.
  const mountedRef = useRef(true)
  useEffect(() => {
    mountedRef.current = true
    if (createOperation.getSnapshot().outcome?.kind === 'confirmed') createOperation.reset()
    return () => {
      mountedRef.current = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mount only
  }, [])

  const [initialOperationDraft] = useState(
    () => createOperation.getSnapshot().operation?.payload.draft ?? null,
  )
  const isFreshNewGroup = initialOperationDraft === null && initialDraft == null
  const [draft, setDraft] = useState<ImportDraft>(() =>
    cloneDraft(initialOperationDraft ?? initialDraft ?? freshDraft()),
  )
  // Classic `defaultDraft` baseline: freshDraft() JSON mutated by
  // default-channel adoption — recovered/locked drafts never match it and
  // stay dirty.
  const [baselineJson, setBaselineJson] = useState(() => JSON.stringify(freshDraft()))
  const [baseUrlOverrideEnabled, setBaseUrlOverrideEnabled] = useState(
    () => isFreshNewGroup || Boolean(draft.params.base_url?.trim()),
  )
  const [paramTouched, setParamTouched] = useState<Record<string, boolean>>({})
  const [visibleModelInvalidIndexes, setVisibleModelInvalidIndexes] = useState<Set<number>>(
    new Set(),
  )
  const [revealAllModelErrors, setRevealAllModelErrors] = useState(false)
  const nextModelKeyRef = useRef(Math.max(0, ...draft.models.map(({ key }) => key)) + 1)
  const [discoveryCandidates, setDiscoveryCandidates] = useState<ModelCandidate[]>([])
  const [shouldApplyDefaultChannel, setShouldApplyDefaultChannel] = useState(isFreshNewGroup)
  const allChannelsQuery = useQuery(channelsQueryOptions(apiClient, ''))
  const allChannels = allChannelsQuery.data?.items ?? []
  const allChannelsStable = allChannels
  const [selectedChannelCache, setSelectedChannelCache] = useState<ChannelDto | null>(null)
  const selectedChannel: ChannelDto | null =
    allChannelsStable.find(({ channel_id }) => channel_id === draft.channel_id) ??
    (selectedChannelCache?.channel_id === draft.channel_id ? selectedChannelCache : null)

  const [discoveryErrorKey, setDiscoveryErrorKey] = useState<MessageId | ''>('')
  const [discoveryLoading, setDiscoveryLoading] = useState(false)
  const discoveryDrawerOpen = routeState.panel === 'discovery'
  const modelEditorRef = useRef<ModelAliasEditorHandle>(null)
  const [errorKey, setErrorKey] = useState('')
  const [credentialValidation, setCredentialValidation] = useState<CredentialValidationData | null>(
    null,
  )
  const [connectionParsedSuccess, setConnectionParsedSuccess] = useState(false)
  // 识别到连接 JSON 但当前渠道无法完整映射 URL/API key 时，阻断 discover/create 提交原始 JSON。
  const [connectionUnsupported, setConnectionUnsupported] = useState(false)
  const submissionErrorRef = useRef<HTMLElement>(null)
  const [errorFocusToken, setErrorFocusToken] = useState(0)
  useEffect(() => {
    if (errorFocusToken > 0) submissionErrorRef.current?.focus()
  }, [errorFocusToken])
  const [conflict, setConflict] = useState<SameTargetConflictData | null>(() =>
    sameTargetConflict(createOperation.getSnapshot().lastError),
  )
  const [serverModelConflicts, setServerModelConflicts] = useState<ModelNameConflict[]>([])
  const [completed, setCompleted] = useState(false)

  const mutationPending = createSnap.pending
  const payloadLocked = createSnap.operation !== null
  const activeSnap = createSnap.operation ? createSnap : null
  const mutationOutcome = activeSnap?.outcome ?? null
  const operationNoticeKey: MessageId | '' = !mutationOutcome
    ? ''
    : mutationOutcome.kind === 'reconciling'
      ? 'import.operation.reconciling'
      : mutationOutcome.kind === 'indeterminate'
        ? 'import.operation.indeterminate'
        : mutationOutcome.kind === 'failed' && mutationOutcome.reason === 'retryable-precondition'
          ? 'import.operation.waiting'
          : mutationOutcome.kind === 'failed' && mutationOutcome.reason === 'expired-known'
            ? 'import.operation.expired'
            : ''
  const operationResourceIdentity =
    mutationOutcome?.kind === 'failed' && mutationOutcome.reason === 'expired-known'
      ? mutationOutcome.resource_identity
      : ''
  const canRetryOperation = activeSnap?.canRetry ?? false

  const discoveryControllerRef = useRef<AbortController | undefined>(undefined)
  const discoveryRequestIdentityRef = useRef(0)
  const discoveryPanelRunRef = useRef(false)
  const [autoFilledSkip, setAutoFilledSkip] = useState(false)

  const credential = readSingleCredential(draft.credentials)
  const readyStage = draft.staged_credential?.status === 'ready' ? draft.staged_credential : null

  async function syncDiscoveryStage(stageID: string, identity: number): Promise<void> {
    if (!mountedRef.current) return
    try {
      const stage = await getCredentialStage(apiClient, stageID)
      if (!mountedRef.current) return
      setDraft((current) =>
        current.staged_credential?.stage_id === stageID
          ? {
              ...current,
              staged_credential: {
                ...stage,
                authorization_url:
                  stage.authorization_url ?? current.staged_credential.authorization_url,
                redirect_uri: stage.redirect_uri ?? current.staged_credential.redirect_uri,
              },
            }
          : current,
      )
    } catch (cause) {
      if (mountedRef.current && discoveryRequestIdentityRef.current === identity) {
        setDiscoveryErrorKey(
          presentSubscriptionErrorKey(cause, 'common.modelDiscoveryFailed') as MessageId,
        )
      }
    }
  }

  const credentialCount =
    draft.connection_type === 'subscription'
      ? readyStage === null
        ? 0
        : 1
      : credential === null
        ? 0
        : 1
  const tooManyCredentials =
    draft.connection_type === 'subscription'
      ? false
      : draft.credentials.trim() !== '' && credential === null
  const connectionChannel = selectedChannel
  const isSubscription = draft.connection_type === 'subscription'
  const proxyLocked = isSubscription && draft.staged_credential !== null
  const draftProxyMutation = proxyMutation(draft.proxy.mode, draft.proxy.url)
  const draftProxyOverride =
    selectedChannel?.capabilities.outbound_proxy === true &&
    draftProxyMutation !== null &&
    draftProxyMutation !== undefined
      ? draftProxyMutation
      : undefined
  const proxyError =
    selectedChannel?.capabilities.outbound_proxy === true && draftProxyMutation === undefined
      ? t('common.proxy.invalid')
      : ''
  const structuredCredentials =
    selectedChannel !== null &&
    (selectedChannel.credential_fields.length !== 1 ||
      selectedChannel.credential_fields[0]?.key !== 'api_key')
  const connectionMappingBlocked = (() => {
    if (draft.connection_type !== 'api_key') return false
    const parsed = parseConnectionJSON(draft.credentials)
    if (!parsed) return connectionUnsupported
    const channel = selectedChannel
    if (!channel || channel.connection.type !== 'api_key') return true
    const mapping = mapConnectionToChannel(parsed, channel)
    return (
      mapping.credentials === null ||
      mapping.urlParamKey === null ||
      draft.credentials !== mapping.credentials
    )
  })()
  const allParamErrors: Record<string, string> = (() => {
    const errors: Record<string, string> = {}
    const channel = connectionChannel
    if (!channel) return errors
    for (const field of channel.param_fields) {
      const value = draft.params[field.key]?.trim() ?? ''
      if ((field.required || (field.key === 'base_url' && baseUrlOverrideEnabled)) && !value) {
        errors[field.key] = t('import.connection.paramRequired', { name: field.label })
        continue
      }
      if (field.input_kind === 'url' && value && !isValidUpstreamBaseURL(value)) {
        errors[field.key] = t('import.connection.urlError')
      }
    }
    return errors
  })()
  const paramErrors = Object.fromEntries(
    Object.entries(allParamErrors).filter(([key]) => paramTouched[key]),
  )
  const visibleParamError = Object.values(paramErrors)[0] ?? proxyError
  const paramsError =
    selectedChannel === null
      ? t('import.presets.channelRequired')
      : (Object.values(allParamErrors)[0] ?? proxyError)
  const modelConflicts = serverModelConflicts.length
    ? serverModelConflicts
    : findModelNameConflicts(toGroupModels(draft.models))
  const modelValidity = modelDraftValidity(draft.models, modelConflicts)
  const hasVisibleModelErrors =
    visibleModelInvalidIndexes.size > 0 ||
    (revealAllModelErrors && modelValidity.invalidIndexes.size > 0)
  const modelValidationSummary = [
    modelConflicts.length ? t('import.models.conflictSummary') : '',
    modelValidity.emptyIDIndexes.size ? t('import.models.manualIdRequired') : '',
    modelValidity.emptyAliasIndexes.size ? t('import.models.emptyAliasSummary') : '',
  ]
    .filter(Boolean)
    .join(' · ')
  const submissionErrorMessage =
    (hasVisibleModelErrors ? modelValidationSummary : '') ||
    (credentialValidation
      ? t('import.credentials.validation', {
          entry: credentialValidation.entry,
          field: credentialValidation.field,
          reason: t(
            `import.credentials.validationReasons.${credentialValidation.reason_code}` as MessageId,
          ),
        })
      : errorKey
        ? t(errorKey as MessageId)
        : '')
  const submitBlockedReason = (() => {
    if (payloadLocked || mutationPending) return ''

    if (paramsError) {
      if (selectedChannel === null) return t('import.presets.channelRequired')
      return visibleParamError || t('import.steps.channel.incomplete')
    }
    if (credentialCount === 0)
      return t(isSubscription ? 'import.subscription.required' : 'import.credentials.required')
    if (tooManyCredentials) return t('import.credentials.tooMany')
    if (modelValidity.invalidIndexes.size) return t('import.models.resolveErrors')
    return ''
  })()
  const canDiscover =
    selectedChannel?.capabilities.model_discovery === true &&
    !payloadLocked &&
    !paramsError &&
    !connectionMappingBlocked &&
    credentialCount > 0 &&
    !tooManyCredentials
  const canCreate =
    !payloadLocked &&
    !mutationPending &&
    !paramsError &&
    !connectionMappingBlocked &&
    credentialCount > 0 &&
    !tooManyCredentials &&
    modelValidity.invalidIndexes.size === 0
  const currentModelIDs = draft.models.map(({ id }) => id.trim()).filter(Boolean)
  const dirty = !completed && JSON.stringify(draft) !== baselineJson
  const summary = (() => {
    const models = draft.models.length
    if (isSubscription) {
      return t(models ? 'import.summaryAccounts' : 'import.summaryAccountsOptional', {
        accounts: credentialCount,
        models,
      })
    }
    return t(models ? 'import.summary' : 'import.summaryOptional', {
      credentials: credentialCount,
      models,
    })
  })()

  const connectionTypeLabel = t(
    isSubscription
      ? 'import.steps.channel.connectionTypes.subscription'
      : 'import.steps.channel.connectionTypes.apiKey',
  )
  const channelStepState: ImportStepState = visibleParamError
    ? 'error'
    : selectedChannel && !paramsError
      ? 'ready'
      : 'active'
  const channelStepSummary = (() => {
    if (visibleParamError) return visibleParamError
    if (!selectedChannel) return ''
    return t('import.steps.channel.summary', {
      channel: selectedChannel.name,
      connection: connectionTypeLabel,
    })
  })()
  const subscriptionStageError =
    isSubscription &&
    readyStage === null &&
    draft.staged_credential !== null &&
    ['failed', 'cancelled', 'expired', 'outcome_unknown'].includes(draft.staged_credential.status)
  const activeSubscriptionStage =
    draft.staged_credential !== null &&
    ['pending_authorization', 'exchanging'].includes(draft.staged_credential.status)
      ? draft.staged_credential
      : null
  const credentialStepState: ImportStepState =
    tooManyCredentials || credentialValidation !== null || subscriptionStageError
      ? 'error'
      : credentialCount > 0
        ? 'ready'
        : channelStepState === 'ready'
          ? 'active'
          : 'pending'
  const subscriptionChannelName = selectedChannel?.name ?? draft.channel_id
  const credentialStepTitle = t(
    isSubscription
      ? 'import.steps.credentials.subscriptionTitle'
      : structuredCredentials
        ? 'import.steps.credentials.structuredTitle'
        : 'import.steps.credentials.apiKeyTitle',
    { channel: subscriptionChannelName },
  )
  const credentialStepDescription = isSubscription
    ? undefined
    : t(
        structuredCredentials
          ? 'import.credentials.structuredDescription'
          : 'import.credentials.description',
      )
  const credentialStepSummary = (() => {
    if (tooManyCredentials) return t('import.credentials.tooMany')
    if (credentialValidation || subscriptionStageError)
      return t('import.steps.credentials.needsAttention')
    if (isSubscription && credentialCount > 0)
      return t('import.subscription.readyCount', { count: credentialCount })
    if (isSubscription && activeSubscriptionStage)
      return t(`import.subscription.status.${activeSubscriptionStage.status}` as MessageId)
    if (credentialCount > 0)
      return t(
        structuredCredentials
          ? 'import.steps.credentials.credentialCount'
          : 'import.steps.credentials.keyCount',
        { count: credentialCount },
      )
    return ''
  })()
  const modelStepState: ImportStepState = hasVisibleModelErrors ? 'error' : 'optional'
  const modelStepSummary = (() => {
    if (hasVisibleModelErrors) return t('import.models.resolveErrors')
    if (draft.models.length) return t('import.models.count', { count: draft.models.length })
    if (paramsError) return t('import.steps.models.availableAfterChannel')
    if (credentialCount === 0)
      return t(
        isSubscription
          ? 'import.steps.models.availableAfterAccount'
          : 'import.steps.models.availableAfterCredentials',
      )
    return t('import.steps.models.optionalSummary')
  })()
  const resolvedGroupName =
    draft.name.trim() || selectedChannel?.name || t('import.steps.create.unnamedGroup')
  const createStatusTitle = mutationPending
    ? t('import.steps.create.creating')
    : payloadLocked
      ? t('import.steps.create.checking')
      : canCreate
        ? summary
        : t('import.steps.create.incomplete')
  const createStatusDescription = mutationPending
    ? t('import.steps.create.creatingDescription')
    : payloadLocked
      ? t('import.steps.create.checkingDescription')
      : !canCreate
        ? submitBlockedReason
        : t('import.steps.create.readyDescription', { name: resolvedGroupName })
  const discoveryError = discoveryErrorKey ? t(discoveryErrorKey) : ''

  const aliasEditorLabels: ModelAliasEditorLabels = {
    tableLabel: t('import.models.tableLabel'),
    id: t('import.models.id'),
    alias: t('import.models.alias'),
    thirdColumn: t('import.models.source'),
    actions: t('import.models.actions'),
    search: t('import.models.search'),
    searchLabel: t('import.models.searchLabel'),
    clearSearch: t('import.models.clearSearch'),
    aliasEnabledFor: (id) => t('import.models.aliasEnabledFor', { id }),
    aliasFor: (id) => t('import.models.aliasFor', { id }),
    aliasPlaceholder: t('import.models.aliasPlaceholder'),
    aliasRequired: t('import.models.aliasRequired'),
    removeFor: (id) => t('import.models.removeFor', { id }),
    manualId: t('import.models.manualId'),
    manualIdRequired: t('import.models.manualIdRequired'),
    add: t('import.models.add'),
    addInline: t('import.models.addInline'),
    count: (count) => t('import.models.count', { count }),
    empty: t('import.models.empty'),
    noMatches: t('import.models.noMatches'),
    nameConflict: (name) => t('import.models.nameConflict', { name }),
    weight: t('import.models.weight'),
    priority: t('import.models.priority'),
    weightDisabled: t('import.models.weightDisabled'),
    invalidWeight: t('import.models.invalidWeight'),
    invalidPriority: t('import.models.invalidPriority'),
    zeroShare: t('import.models.zeroShareSummary'),
  }
  const discoveryDrawerLabels: ModelDiscoveryDrawerLabels = {
    title: t('import.models.drawer.title'),
    description: t('import.models.drawer.description'),
    close: t('import.models.drawer.close'),
    loading: t('import.models.drawer.loading'),
    search: t('import.models.drawer.search'),
    clearSearch: t('import.models.clearSearch'),
    filterLabel: t('import.models.drawer.filterLabel'),
    filterUnadded: t('import.models.drawer.filterUnadded'),
    filterAll: t('import.models.drawer.filterAll'),
    alreadyAdded: t('import.models.drawer.alreadyAdded'),
    unadded: t('import.models.drawer.unadded'),
    noMatches: t('import.models.drawer.noMatches'),
    empty: t('import.models.drawer.empty'),
    selected: (count) => t('import.models.drawer.selected', { count }),
    selectAll: t('import.models.drawer.selectAll'),
    deselectAll: t('import.models.drawer.deselectAll'),
    retry: t('common.retry'),
    cancel: t('common.cancel'),
    confirm: t('import.models.drawer.confirm'),
    pricingStatus: {
      pending: t('import.models.pricing.pending'),
      configured: t('import.models.pricing.configured'),
    },
    pricingDiscovered: (source) => t('import.models.pricing.discovered', { source }),
    sources: {
      catalog: t('import.models.sources.catalog'),
      live: t('import.models.sources.live'),
    },
  }

  const unsavedChanges = useUnsavedChanges({
    dirty,
    blocked: mutationPending,
    allowRouteUpdate: (current, next) =>
      next.pathname === current.pathname &&
      parseImportRouteQuery(next.search as SharedRouteQuery).mode === 'new' &&
      parseImportRouteQuery(current.search as SharedRouteQuery).mode === 'new',
  })

  // Classic recovery.register: prefer the stable operation draft, else the live form.
  const recoveryDraftRef = useRef(() => null as ImportDraft | null)
  useEffect(() => {
    recoveryDraftRef.current = () => {
      if (completed) return null
      const stableDraft = createOperation.getSnapshot().operation?.payload.draft
      return stableDraft?.mode === 'new'
        ? cloneDraft(stableDraft as ImportDraft)
        : cloneDraft(draft)
    }
  })
  useEffect(
    () => services.importRecovery.register(() => recoveryDraftRef.current()),
    [services.importRecovery],
  )

  function cancelDiscovery(): void {
    discoveryRequestIdentityRef.current += 1
    discoveryControllerRef.current?.abort()
    discoveryControllerRef.current = undefined
    setDiscoveryLoading(false)
  }

  function invalidateDiscovery(): void {
    cancelDiscovery()
    setDiscoveryCandidates([])
    setDiscoveryErrorKey('')
  }

  function cancelDefaultChannel(): void {
    setShouldApplyDefaultChannel(false)
  }

  function updateRoute(patch: Partial<ImportRouteState>, replace = false): void {
    const state: ImportRouteState = {
      ...routeState,
      ...patch,
      mode: 'new',
      groupID: undefined,
    }
    void navigate({
      to: pagePath('import'),
      search: serializeImportRouteQuery(state),
      replace,
    })
  }

  function setPanel(panel: ImportPanel | undefined): void {
    updateRoute(
      panel === 'discovery'
        ? { panel }
        : { panel, discoverySearch: undefined, discoveryFilter: 'unadded' },
    )
  }

  function setModelSearch(value: string): void {
    updateRoute({ modelSearch: constrainCollectionSearch(value) }, true)
  }

  function setDiscoverySearch(value: string): void {
    updateRoute({ discoverySearch: constrainCollectionSearch(value) }, true)
  }

  function setDiscoveryFilter(value: ImportDiscoveryFilter): void {
    updateRoute({ discoveryFilter: value })
  }

  function resetParamTouches(): void {
    setParamTouched({})
  }

  function touchChannelParam(key: string): void {
    setParamTouched((current) => (current[key] ? current : { ...current, [key]: true }))
  }

  function selectChannel(channel: ChannelDto): void {
    if (payloadLocked) return
    cancelDefaultChannel()
    resetParamTouches()
    setSelectedChannelCache(channel)
    setDraft((current) => ({
      ...current,
      channel_id: channel.channel_id,
      connection_type: channel.connection.type,
      params: initialChannelParams(channel),
    }))
    setBaseUrlOverrideEnabled(true)
    setPanel(undefined)
  }

  function setChannelParam(key: string, value: string): void {
    cancelDefaultChannel()
    setConnectionParsedSuccess(false)
    setDraft((current) => {
      const params = { ...current.params }
      if (key === 'base_url' && !value.trim()) delete params.base_url
      else params[key] = value
      return { ...current, params }
    })
    if (key === 'base_url' && value.trim()) setBaseUrlOverrideEnabled(true)
  }

  function setBaseURLOverride(enabled: boolean): void {
    cancelDefaultChannel()
    setParamTouched((current) => {
      if (!((field) => field in current)('base_url')) return current
      const next = { ...current }
      delete next.base_url
      return next
    })
    setBaseUrlOverrideEnabled(enabled)
    if (enabled) return
    setConnectionParsedSuccess(false)
    setDraft((current) => {
      const params = { ...current.params }
      delete params.base_url
      return { ...current, params }
    })
  }

  function createManualRow(): ModelDraftItem {
    const key = nextModelKeyRef.current++
    return {
      id: '',
      name: '',
      sources: [],
      pricing_status: 'pending',
      alias: '',
      alias_enabled: false,
      editable_id: true,
      key,
    }
  }

  function updateModels(models: ModelDraftItem[]): void {
    setServerModelConflicts([])
    setDraft((current) => {
      const previousByKey = new Map(current.models.map((item) => [item.key, item] as const))
      return {
        ...current,
        models: models.map((item) => {
          const previous = previousByKey.get(item.key)
          return previous && previous.id === item.id
            ? { ...item, sources: [...item.sources] }
            : {
                ...item,
                name: item.id,
                sources: [],
                pricing_status: 'pending',
              }
        }),
      }
    })
  }

  function requestDiscovery(): void {
    if (!ensureConnectionMapping() || !canDiscover) return
    if (!discoveryDrawerOpen) {
      setPanel('discovery')
      return
    }
    discoveryPanelRunRef.current = true
    startDiscovery()
  }

  function startDiscovery(): void {
    if (!ensureConnectionMapping() || !canDiscover || discoveryLoading) return
    const subscriptionStage =
      draft.connection_type === 'subscription'
        ? currentReadyStage(draft.staged_credential)
        : undefined
    if (draft.connection_type === 'subscription' && !subscriptionStage) {
      setDraft((current) => ({
        ...current,
        staged_credential: expireStaleReadyStage(current.staged_credential),
      }))
      // 让位给 draft 变更触发的 invalidateDiscovery，否则这条提示会被它清掉。
      queueMicrotask(() => {
        setDiscoveryErrorKey('common.subscriptionErrors.stageExpired')
      })
      return
    }
    const request: ModelDiscoveryRequest = {
      channel_id: draft.channel_id,
      connection_type: draft.connection_type,
      params: Object.fromEntries(
        Object.entries(draft.params).map(([key, value]) => [key, value.trim()]),
      ),
      ...(draftProxyOverride === undefined ? {} : { proxy: draftProxyOverride }),
      ...(draft.connection_type === 'subscription'
        ? { staged_credential_id: subscriptionStage?.stage_id }
        : { credentials: draft.credentials }),
    }
    cancelDiscovery()
    const controller = new AbortController()
    discoveryControllerRef.current = controller
    const identity = ++discoveryRequestIdentityRef.current
    setDiscoveryCandidates([])
    setDiscoveryErrorKey('')
    setDiscoveryLoading(true)
    void runDiscovery(request, controller, identity)
  }

  async function runDiscovery(
    request: ModelDiscoveryRequest,
    controller: AbortController,
    identity: number,
  ): Promise<void> {
    const stageID = request.staged_credential_id?.trim()
    try {
      const result = await discoverModels(apiClient, request, controller.signal)
      if (
        discoveryRequestIdentityRef.current !== identity ||
        discoveryControllerRef.current !== controller
      ) {
        return
      }
      setDiscoveryCandidates(result.models)
      setDraft((current) => ({
        ...current,
        models: mergeCandidateMetadata(current.models, result.models),
      }))
    } catch (cause: unknown) {
      if (
        cause instanceof RequestCancelledError ||
        discoveryRequestIdentityRef.current !== identity ||
        discoveryControllerRef.current !== controller
      ) {
        return
      }
      setDiscoveryErrorKey(
        (draft.connection_type === 'subscription'
          ? presentSubscriptionErrorKey(cause, 'common.modelDiscoveryFailed')
          : 'common.modelDiscoveryFailed') as MessageId,
      )
    } finally {
      if (stageID) await syncDiscoveryStage(stageID, identity)
      if (
        discoveryRequestIdentityRef.current === identity &&
        discoveryControllerRef.current === controller
      ) {
        discoveryControllerRef.current = undefined
        setDiscoveryLoading(false)
      }
    }
  }

  function confirmCandidates(selectedCandidates: ModelCandidate[]): void {
    setServerModelConflicts([])
    setDraft((current) => ({
      ...current,
      models: appendSelectedCandidates(current.models, selectedCandidates, (candidate) =>
        createDiscoveredModelDraft([candidate], () => nextModelKeyRef.current++).at(0)!,
      ),
    }))
    setPanel(undefined)
  }

  async function addManualModel(): Promise<void> {
    if (payloadLocked) return
    if (draft.models.length === 0) {
      updateModels([createManualRow()])
      return
    }
    await modelEditorRef.current?.addManual()
  }

  async function focusFirstInvalidModel(): Promise<void> {
    setRevealAllModelErrors(true)
    await modelEditorRef.current?.focusFirstInvalid()
  }

  function buildCreateBody(confirmSameTarget: boolean): GroupCreateRequest {
    const name = draft.name.trim()
    return {
      channel_id: draft.channel_id,
      connection_type: draft.connection_type,
      params: Object.fromEntries(
        Object.entries(draft.params).map(([key, value]) => [key, value.trim()]),
      ),
      ...(draftProxyOverride === undefined ? {} : { proxy: draftProxyOverride }),
      ...(name ? { name } : {}),
      provider_url: draft.provider_url.trim() || null,

      models: toGroupModels(draft.models),
      ...(draft.connection_type === 'subscription'
        ? {
            staged_credential_id: currentReadyStage(draft.staged_credential)?.stage_id,
          }
        : { credential: draft.credentials }),
      confirm_same_target: confirmSameTarget,
    }
  }

  async function finishSuccess(groupID: number): Promise<void> {
    setCompleted(true)
    setDraft((current) => ({ ...current, credentials: '', staged_credential: null }))
    services.importRecovery.clear()
    createOperation.reset()

    await applyInvalidationPlan(queryClient, mutationInvalidationPlans.group.create)
    if (!mountedRef.current) return
    toast.show({
      message: t('group.settings.savedFeedback'),
      tone: 'success',
      duration: 4_000,
    })
    await unsavedChanges.runWithoutPrompt(() =>
      navigate({ to: `${pagePath('groups')}/${groupID}` }),
    )
  }

  async function reportSubmissionError(key: string): Promise<void> {
    setCredentialValidation(null)
    setErrorKey(key)
    setErrorFocusToken((token) => token + 1)
  }

  async function submitCreate(): Promise<void> {
    if (!ensureConnectionMapping()) return
    if (
      draft.connection_type === 'subscription' &&
      readyStage !== null &&
      currentReadyStage(draft.staged_credential) === null
    ) {
      setDraft((current) => ({
        ...current,
        staged_credential: expireStaleReadyStage(current.staged_credential),
      }))
      // 同上：draft 变更会触发清空 errorKey 的 watcher，先让它跑完。
      queueMicrotask(() => {
        void reportSubmissionError('common.subscriptionErrors.stageExpired')
      })
      return
    }
    if (!canCreate) return
    cancelDiscovery()
    setConflict(null)
    setErrorKey('')
    setServerModelConflicts([])
    if (!owner.beginCreate(buildCreateBody(false), cloneDraft(draft))) return
    await executeCreateOperation()
  }

  async function executeCreateOperation(): Promise<void> {
    if (!createOperation.getSnapshot().operation) return
    setErrorKey('')
    setCredentialValidation(null)
    const outcome = await createOperation.execute((operation, signal) =>
      createGroup(apiClient, operation.payload.request, operation.idempotencyKey, signal),
    )
    if (!outcome) return
    if (outcome.kind === 'confirmed') {
      await finishSuccess(outcome.value.group_id)
      return
    }
    if (!mountedRef.current || outcome.kind !== 'failed' || outcome.reason !== 'rejected') return
    const cause = createOperation.getSnapshot().lastError
    const targetConflict = sameTargetConflict(cause)
    if (targetConflict) {
      setConflict(targetConflict)
      return
    }
    if (cause instanceof ApiError && cause.code === 'MODEL_NAME_CONFLICT') {
      const conflicts = readModelNameConflicts(cause.data)
      if (conflicts.length) {
        setServerModelConflicts(conflicts)
        setRevealAllModelErrors(true)
        createOperation.reset()
        setErrorFocusToken((token) => token + 1)
        return
      }
    }
    if (cause instanceof ApiError && cause.code === 'VALIDATION_FAILED') {
      const validation = readCredentialValidationData(cause.data)
      if (validation) {
        setCredentialValidation(validation)
        createOperation.reset()
        setErrorFocusToken((token) => token + 1)
        return
      }
    }
    createOperation.reset()
    await reportSubmissionError(
      cause instanceof ApiError && cause.code === 'SINGLE_CREDENTIAL_REQUIRED'
        ? 'import.credentials.tooMany'
        : draft.connection_type === 'subscription'
          ? presentSubscriptionErrorKey(cause, 'import.createFailed')
          : 'import.createFailed',
    )
  }

  async function submitSeparateGroup(): Promise<void> {
    const current = createOperation.getSnapshot().operation
    if (!current || !conflict || mutationPending) return
    const displayedConflict = conflict
    const payload = structuredClone(current.payload)
    createOperation.reset()
    if (
      !owner.beginCreate(
        {
          ...payload.request,
          confirm_same_target: true,
        },
        payload.draft,
      )
    ) {
      return
    }
    await executeCreateOperation()
    setConflict((current) => (current === displayedConflict ? null : current))
  }

  async function manageGroup(groupID: number): Promise<void> {
    if (mutationPending || !(await unsavedChanges.confirmDiscard())) return
    returnToEdit()
    services.importRecovery.clear()
    setCompleted(true)
    await unsavedChanges.runWithoutPrompt(() =>
      navigate({ to: `${pagePath('groups')}/${groupID}` }),
    )
  }

  async function retryOperation(): Promise<void> {
    if (createOperation.getSnapshot().operation) await executeCreateOperation()
  }

  async function abandonOperation(): Promise<void> {
    if (mutationPending || !payloadLocked) return
    if (!(await unsavedChanges.confirmDiscard()) || mutationPending) return
    createOperation.reset()
    setConflict(null)
    setServerModelConflicts([])
    setCredentialValidation(null)
    setErrorKey('')
  }

  function returnToEdit(): void {
    if (mutationPending) return
    createOperation.reset()

    setConflict(null)
    setCredentialValidation(null)
    setErrorKey('')
  }

  // 将连接 JSON 按当前渠道 schema 映射到既有字段。无法完整映射时保留用户可见
  // 原输入并阻断 discover/create，避免原始 JSON 越过请求边界。
  function applyConnectionMapping(): void {
    setConnectionParsedSuccess(false)
    setConnectionUnsupported(false)

    const channel = selectedChannel
    if (!channel || channel.connection.type !== 'api_key') return

    const parsed = parseConnectionJSON(draft.credentials)
    if (!parsed) return

    const mapping = mapConnectionToChannel(parsed, channel)
    if (mapping.credentials === null || mapping.urlParamKey === null) {
      // 无法同时映射 URL 与 api_key：保留原输入，阻断请求边界。
      setConnectionUnsupported(true)
      return
    }

    setAutoFilledSkip(true)
    if (mapping.urlParamKey !== null) {
      setChannelParam(mapping.urlParamKey, parsed.baseURL)
      touchChannelParam(mapping.urlParamKey)
    }
    setDraft((current) => ({ ...current, credentials: mapping.credentials ?? '' }))
    // 非法 URL 继续交给既有字段错误展示，不显示"已自动填入"的成功提示。
    setConnectionParsedSuccess(isValidUpstreamBaseURL(parsed.baseURL))
  }

  function ensureConnectionMapping(): boolean {
    if (draft.connection_type !== 'api_key') {
      setConnectionUnsupported(false)
      return true
    }

    const parsed = parseConnectionJSON(draft.credentials)
    if (!parsed) {
      setConnectionUnsupported(false)
      return true
    }

    const channel = selectedChannel
    if (!channel || channel.connection.type !== 'api_key') {
      setConnectionUnsupported(true)
      return false
    }

    // 请求构造前同步重试一次，覆盖 schema resolve 与 watcher 尚未收敛的窗口。
    applyConnectionMapping()
    return !connectionUnsupported && !connectionMappingBlocked
  }

  // Classic watch([channel_id, connection_type, params, credentials,
  // readyStageIds, proxy]) → invalidateDiscovery. Effect keyed on the composite
  // signature; the credential-mapping render adjustment converges first.
  const discoveryInvalidationSignature = [
    draft.channel_id,
    draft.connection_type,
    JSON.stringify(draft.params),
    draft.credentials,
    readyStage?.stage_id ?? '',
    JSON.stringify(draft.proxy),
  ].join('')
  const lastInvalidationSignatureRef = useRef(discoveryInvalidationSignature)
  useEffect(() => {
    if (lastInvalidationSignatureRef.current === discoveryInvalidationSignature) return
    lastInvalidationSignatureRef.current = discoveryInvalidationSignature
    invalidateDiscovery()
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the signature
  }, [discoveryInvalidationSignature])

  // Classic watch(allChannels, immediate): adopt the first channel as the
  // default for a fresh draft — and fold it into the baseline so `dirty` stays
  // false until the user edits.
  if (allChannelsStable.length > 0 && shouldApplyDefaultChannel) {
    setShouldApplyDefaultChannel(false)
    if (JSON.stringify(draft) === baselineJson) {
      const channel = allChannelsStable[0]
      const params = initialChannelParams(channel)
      setDraft((current) => ({
        ...current,
        channel_id: channel.channel_id,
        connection_type: channel.connection.type,
        params,
      }))
      setBaselineJson(
        JSON.stringify({
          ...freshDraft(),
          channel_id: channel.channel_id,
          connection_type: channel.connection.type,
          params,
        }),
      )
    }
  }

  // Classic watch(selectedChannel, immediate): sync proxy/connection_type,
  // then run the connection mapping (classic nextTick — here the batched
  // adjustment converges in the same re-render).
  const [lastSelectedChannel, setLastSelectedChannel] = useState<ChannelDto | null>(null)
  if (lastSelectedChannel !== selectedChannel) {
    setLastSelectedChannel(selectedChannel)
    if (selectedChannel && !payloadLocked) {
      if (!selectedChannel.capabilities.outbound_proxy && draft.proxy.mode !== 'inherit') {
        setDraft((current) => ({ ...current, proxy: { mode: 'inherit', url: '' } }))
      }
      if (draft.connection_type !== selectedChannel.connection.type) {
        setDraft((current) => ({
          ...current,
          connection_type: selectedChannel.connection.type,
          params: initialChannelParams(selectedChannel),
        }))
        setBaseUrlOverrideEnabled(true)
      }
      applyConnectionMapping()
    }
  }

  // Classic watch(channel_id): switching channels clears the autofill notice
  // and re-runs the connection mapping after channel state settles.
  const [lastDraftChannelId, setLastDraftChannelId] = useState(draft.channel_id)
  if (draft.channel_id !== lastDraftChannelId) {
    setLastDraftChannelId(draft.channel_id)
    setConnectionParsedSuccess(false)
  }

  // Classic watch(credentials, flush:'sync'): map a pasted connection JSON
  // before any draft-change watcher sees it. connectionAutoFilled skips the
  // adjustment for our own writes.
  const [lastCredentialsSeen, setLastCredentialsSeen] = useState(draft.credentials)
  if (draft.credentials !== lastCredentialsSeen) {
    setLastCredentialsSeen(draft.credentials)
    if (autoFilledSkip) {
      setAutoFilledSkip(false)
    } else {
      applyConnectionMapping()
    }
  }

  // Classic watch([channel_id, credentials, proxy]) → clear credentialValidation.
  const validationClearSignature = [
    draft.channel_id,
    draft.credentials,
    JSON.stringify(draft.proxy),
  ].join('')
  const [lastValidationSignature, setLastValidationSignature] = useState(validationClearSignature)
  if (validationClearSignature !== lastValidationSignature) {
    setLastValidationSignature(validationClearSignature)
    setCredentialValidation(null)
  }

  // Classic watch(draft JSON) → clear errorKey on any draft mutation.
  const draftJson = JSON.stringify(draft)
  const lastDraftJsonRef = useRef(draftJson)
  useEffect(() => {
    if (lastDraftJsonRef.current === draftJson) return
    lastDraftJsonRef.current = draftJson
    setErrorKey('')
  }, [draftJson])

  // Classic watch([discoveryDrawerOpen, canDiscover]): opening the drawer
  // auto-runs discovery once; closing cancels in-flight work. Deferred to a
  // microtask — route close is external state, so the setState lands in a
  // callback rather than the effect body.
  useEffect(() => {
    if (!discoveryDrawerOpen) {
      discoveryPanelRunRef.current = false
      if (discoveryLoading) queueMicrotask(cancelDiscovery)
      return
    }
    if (!canDiscover || discoveryPanelRunRef.current) return
    discoveryPanelRunRef.current = true
    queueMicrotask(startDiscovery)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mirrors the classic pair watch
  }, [discoveryDrawerOpen, canDiscover, discoveryLoading])

  // Unmount cleanup: abort the in-flight request; state cleanup is moot once
  // the component is gone.
  useEffect(
    () => () => {
      discoveryRequestIdentityRef.current += 1
      discoveryControllerRef.current?.abort()
    },
    [],
  )

  return (
    <div {...stylex.props(styles.root)}>
      <div {...stylex.props(styles.notice)}>
        <ImportOperationNotice
          messageKey={operationNoticeKey}
          resourceIdentity={operationResourceIdentity}
          canRetry={canRetryOperation}
          canAbandon={payloadLocked && !mutationPending}
          pending={mutationPending}
          onRetry={() => void retryOperation()}
          onAbandon={() => void abandonOperation()}
        />
      </div>

      <div {...stylex.props(styles.steps)}>
        <section
          {...stylex.props(styles.step)}
          data-state={channelStepState}
          aria-labelledby="import-channel-step-heading"
        >
          <div {...stylex.props(styles.stepHeader)}>
            <SectionHeader
              headingId="import-channel-step-heading"
              step={1}
              title={
                <>
                  {t('import.steps.channel.title')}
                  <span {...stylex.props(styles.requirement, styles.requirementRequired)}>
                    {t('import.required')}
                  </span>
                </>
              }
              actions={
                channelStepSummary ? (
                  <span
                    {...stylex.props(
                      styles.stepSummary,
                      visibleParamError ? styles.stepSummaryError : null,
                    )}
                  >
                    {selectedChannel && !visibleParamError && (
                      <ChannelIcon icon={selectedChannel.icon} mark={selectedChannel.mark} />
                    )}
                    <span>{channelStepSummary}</span>
                  </span>
                ) : undefined
              }
            />
          </div>

          <div {...stylex.props(styles.stepBody)}>
            <ChannelPresetPicker
              value={draft.channel_id}
              channels={allChannelsStable}
              selectedChannel={selectedChannel}
              loading={allChannelsQuery.isFetching}
              error={allChannelsQuery.isError}
              disabled={payloadLocked}
              hideHeader
              compact
              onSelect={selectChannel}
              onRetry={() => void allChannelsQuery.refetch()}
            />

            <ImportConnectionSection
              channel={connectionChannel}
              name={draft.name}
              providerUrl={draft.provider_url}

              params={draft.params}
              proxy={draft.proxy}
              proxyDisabled={proxyLocked}
              paramErrors={paramErrors}
              baseUrlOverrideEnabled={baseUrlOverrideEnabled}
              disabled={payloadLocked}
              onNameChange={(value) => setDraft((current) => ({ ...current, name: value }))}
              onProviderUrlChange={(value) =>
                setDraft((current) => ({ ...current, provider_url: value }))
              }

              onParamChange={setChannelParam}
              onProxyChange={(proxy) => setDraft((current) => ({ ...current, proxy }))}
              onBaseUrlOverrideChange={setBaseURLOverride}
              onParamBlur={touchChannelParam}
            />
          </div>
        </section>

        <section
          {...stylex.props(styles.step, styles.credentialsStep)}
          aria-labelledby="import-credentials-step-heading"
        >
          <div {...stylex.props(styles.stepHeader)}>
            <SectionHeader
              headingId="import-credentials-step-heading"
              step={2}
              title={
                <>
                  {credentialStepTitle}
                  <span {...stylex.props(styles.requirement, styles.requirementRequired)}>
                    {t('import.required')}
                  </span>
                </>
              }
              description={credentialStepDescription}
              actions={
                credentialStepSummary ? (
                  <span
                    {...stylex.props(
                      styles.stepSummary,
                      credentialStepState === 'error' ? styles.stepSummaryError : null,
                    )}
                  >
                    {credentialStepSummary}
                  </span>
                ) : undefined
              }
            />
          </div>

          <div {...stylex.props(styles.stepBody)}>
            {isSubscription ? (
              <SubscriptionCredentialStager
                stage={draft.staged_credential}
                onStageChange={(stage) =>
                  setDraft((current) => ({ ...current, staged_credential: stage }))
                }
                channelId={draft.channel_id}
                channelName={subscriptionChannelName}
                authorizationMethods={selectedChannel?.connection.authorization_methods ?? []}
                proxy={draftProxyOverride}
                notices={selectedChannel?.notices ?? []}
                context="create"
                disabled={payloadLocked}
                entryDisabled={draftProxyMutation === undefined || draft.staged_credential !== null}
                hideHeader
                compact
              />
            ) : (
              <CredentialTextarea
                value={draft.credentials}
                channel={selectedChannel}
                disabled={payloadLocked}
                hideHeader
                compact
                rows={4}
                onChange={(credentials) => setDraft((current) => ({ ...current, credentials }))}
              />
            )}
            {connectionParsedSuccess && (
              <InlineNotice tone="neutral" appearance="hint">
                {t('import.connection.connectionParsed')}
              </InlineNotice>
            )}
          </div>
        </section>

        <section
          {...stylex.props(styles.step)}
          data-state={modelStepState}
          aria-labelledby="import-models-heading"
        >
          <div {...stylex.props(styles.stepHeader)}>
            <SectionHeader
              headingId="import-models-heading"
              step={3}
              title={
                <>
                  {t('import.steps.models.title')}
                  <span {...stylex.props(styles.requirement, styles.requirementOptional)}>
                    {t('import.optional')}
                  </span>
                </>
              }
              actions={
                <span
                  {...stylex.props(
                    styles.stepSummary,
                    hasVisibleModelErrors ? styles.stepSummaryError : null,
                  )}
                >
                  {modelStepSummary}
                </span>
              }
            />
          </div>

          <div id="import-models-content" {...stylex.props(styles.stepBody, styles.modelsBody)}>
            {draft.models.length === 0 && (
              <div {...stylex.props(styles.modelsEmpty)}>
                <span>{t('import.models.empty')}</span>
                <div {...stylex.props(styles.modelsActions)}>
                  {selectedChannel?.capabilities.model_discovery && (
                    <Button
                      variant="secondary"
                      size="sm"
                      isLoading={discoveryLoading}
                      isDisabled={!canDiscover}
                      icon={<RefreshCw size={16} aria-hidden="true" />}
                      label={t('import.discover')}
                      onClick={requestDiscovery}
                    />
                  )}
                  <Button
                    variant="secondary"
                    size="sm"
                    isDisabled={payloadLocked}
                    icon={<Plus size={16} aria-hidden="true" />}
                    label={t('import.models.add')}
                    onClick={() => void addManualModel()}
                  />
                </div>
              </div>
            )}

            {draft.models.length > 0 && (
              <div {...stylex.props(styles.modelsToolbar)}>
                {selectedChannel?.capabilities.model_discovery && (
                  <Button
                    variant="secondary"
                    size="sm"
                    isLoading={discoveryLoading}
                    isDisabled={!canDiscover}
                    icon={<RefreshCw size={16} aria-hidden="true" />}
                    label={t('import.discover')}
                    onClick={requestDiscovery}
                  />
                )}
                <Button
                  variant="secondary"
                  size="sm"
                  isDisabled={payloadLocked}
                  icon={<Plus size={16} aria-hidden="true" />}
                  label={t('import.models.add')}
                  onClick={() => void addManualModel()}
                />
              </div>
            )}

            {draft.models.length > 0 && (
              <div {...stylex.props(styles.modelEditor)}>
                <ModelAliasEditor
                  ref={modelEditorRef}
                  value={draft.models}
                  conflicts={modelConflicts}
                  labels={aliasEditorLabels}
                  createRow={createManualRow}
                  disabled={payloadLocked}
                  search={routeState.modelSearch ?? ''}
                  addable={false}
                  validationMode="blur"
                  showAllErrors={revealAllModelErrors}
                  renderThirdColumn={(item) => (
                    <ModelPricingStatus
                      status={item.pricing_status}
                      labels={{
                        pending: t('import.models.pricing.pending'),
                        configured: t('import.models.pricing.configured'),
                      }}
                    />
                  )}
                  onChange={updateModels}
                  onSearchChange={setModelSearch}
                  onVisibleValidationChange={(indexes) =>
                    setVisibleModelInvalidIndexes(new Set(indexes))
                  }
                />
              </div>
            )}
          </div>
        </section>
      </div>

      {submissionErrorMessage && (
        <div
          ref={submissionErrorRef as React.RefObject<HTMLDivElement>}
          {...stylex.props(styles.error)}
          tabIndex={-1}
        >
          <InlineNotice tone="danger" appearance="ledger">
            <span {...stylex.props(styles.errorContent)}>
              <span>{submissionErrorMessage}</span>
              {modelValidity.invalidIndexes.size > 0 && (
                <button
                  type="button"
                  {...stylex.props(styles.linkButton)}
                  onClick={() => void focusFirstInvalidModel()}
                >
                  {t('import.models.locateFirstInvalid')}
                </button>
              )}
            </span>
          </InlineNotice>
        </div>
      )}

      <StickySaveBar
        appearance="ledger"
        alwaysVisible
        dirty={!canCreate && !mutationPending}
        pending={mutationPending}
        status={canCreate ? 'saved' : operationNoticeKey ? 'indeterminate' : 'idle'}
        statusContent={
          <div>
            <strong>{createStatusTitle}</strong>
            {createStatusDescription && <span>{createStatusDescription}</span>}
          </div>
        }
        actions={
          <Button
            size="sm"
            isLoading={mutationPending}
            isDisabled={!canCreate}
            label={t('import.create')}
            endContent={<ArrowRight size={16} aria-hidden="true" />}
            onClick={() => void submitCreate()}
          />
        }
      />

      <ModelDiscoveryDrawer
        open={discoveryDrawerOpen}
        candidates={discoveryCandidates}
        currentIds={currentModelIDs}
        loading={discoveryLoading}
        error={discoveryError}
        labels={discoveryDrawerLabels}
        dismissible={!discoveryLoading}
        search={routeState.discoverySearch ?? ''}
        filter={routeState.discoveryFilter}
        onOpenChange={(open) => setPanel(open ? 'discovery' : undefined)}
        onSearchChange={setDiscoverySearch}
        onFilterChange={setDiscoveryFilter}
        onRetry={requestDiscovery}
        onConfirm={confirmCandidates}
      />

      <Dialog
        isOpen={conflict !== null}
        onOpenChange={(open) => {
          if (!open && conflict !== null) returnToEdit()
        }}
        purpose={mutationPending ? 'required' : 'info'}
        width={520}
      >
        <Layout
          header={
            <DialogHeader
              title={t(
                isSubscription ? 'import.conflict.titleSubscription' : 'import.conflict.title',
              )}
              subtitle={t(
                isSubscription
                  ? 'import.conflict.descriptionSubscription'
                  : 'import.conflict.description',
              )}
              onOpenChange={(open) => {
                if (!open && conflict !== null) returnToEdit()
              }}
              hasDivider
            />
          }
          content={
            <LayoutContent>
              <div {...stylex.props(styles.conflictGroups)}>
                {conflict?.groups.map((group) => (
                  <div key={group.id} {...stylex.props(styles.conflictGroup)}>
                    <div>
                      <strong>{`#${group.id} · ${group.name}`}</strong>
                      <span {...stylex.props(styles.conflictHelp)}>
                        {t(
                          isSubscription
                            ? 'import.conflict.manageHelpSubscription'
                            : 'import.conflict.manageHelp',
                        )}
                      </span>
                    </div>
                    <Button
                      variant="secondary"
                      size="sm"
                      isDisabled={mutationPending}
                      label={t(
                        isSubscription
                          ? 'import.conflict.manageSubscription'
                          : 'import.conflict.manage',
                      )}
                      onClick={() => void manageGroup(group.id)}
                    />
                  </div>
                ))}
              </div>
            </LayoutContent>
          }
          footer={
            <LayoutFooter hasDivider>
              <Button
                variant="secondary"
                size="sm"
                isDisabled={mutationPending}
                label={t('import.conflict.edit')}
                onClick={returnToEdit}
              />
              <Button
                size="sm"
                isLoading={mutationPending}
                isDisabled={mutationPending}
                label={t('import.conflict.separate')}
                onClick={() => void submitSeparateGroup()}
              />
            </LayoutFooter>
          }
        />
      </Dialog>
      {unsavedChanges.dialog}
    </div>
  )
}

const styles = stylex.create({
  root: {
    minWidth: 0,
  },
  notice: {
    marginTop: { default: 'var(--space-5)', ':empty': 0 },
  },
  steps: {
    minWidth: 0,
  },
  step: {
    minWidth: 0,
    paddingTop: '14px',
    paddingBottom: '18px',
    selectors: {},
  },
  stepHeader: {
    minWidth: 0,
  },
  stepBody: {
    minWidth: 0,
    marginTop: '12px',
    marginLeft: '34px',
  },
  credentialsStep: {},
  modelsBody: {},
  requirement: {
    display: 'inline-flex',
    alignItems: 'center',
    borderRadius: 'var(--radius-tag)',
    paddingBlock: '2px',
    paddingInline: '8px',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 600,
    letterSpacing: '0.01em',
  },
  requirementRequired: {
    backgroundColor: 'var(--color-action-soft)',
    color: 'var(--color-action)',
  },
  requirementOptional: {
    backgroundColor: 'var(--color-neutral-bg)',
    color: 'var(--color-text-faint)',
  },
  stepSummary: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
    minWidth: 0,
  },
  stepSummaryError: {
    color: 'var(--color-danger)',
  },
  modelsEmpty: {
    display: 'grid',
    gap: 'var(--space-3)',
    justifyItems: 'start',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
    paddingBlock: 'var(--space-3)',
  },
  modelsActions: {
    display: 'flex',
    gap: 'var(--space-2)',
  },
  modelsToolbar: {
    display: 'flex',
    gap: 'var(--space-2)',
    marginBottom: 'var(--space-3)',
  },
  modelEditor: {
    minWidth: 0,
  },
  error: {
    marginTop: 'var(--space-5)',
    outlineStyle: 'none',
  },
  linkButton: {
    display: 'inline',
    minHeight: 0,
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: { default: 'var(--color-action)', ':hover': 'var(--color-action-hover)' },
    padding: 0,
    verticalAlign: 'baseline',
    cursor: 'pointer',
    fontSize: 'inherit',
    textDecorationLine: { default: 'none', ':hover': 'underline' },
  },
  errorContent: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 'var(--space-3)',
    flexWrap: 'wrap',
  },
  conflictGroups: {
    display: 'grid',
    gap: 'var(--space-3)',
  },
  conflictGroup: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
  },
  conflictHelp: {
    display: 'block',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
    marginTop: '2px',
  },
})
