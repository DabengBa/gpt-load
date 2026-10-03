import * as stylex from '@stylexjs/stylex'
import { Button, Selector, Skeleton } from '@astryxdesign/core'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import { useEffect, useRef, useState } from 'react'

import { ApiError, InvalidResponseError } from '@shared/http/errors'
import { credentialQueryOptions } from '@shared/control/resources/credentials'
import {
  connectGroupCredential,
  projectCredentialStage,
  type CredentialStage,
} from '@shared/control/resources/credential-stages'
import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import { channelsQueryOptions, type ChannelDto } from '@shared/control/resources/channels'
import {
  groupOptionsQueryOptions,
  groupSummaryQueryOptions,
  importGroupCredentials,
  readCredentialValidationData,
  type CredentialValidationData,
} from '@shared/control/resources/groups'
import { parsePositiveRouteInteger } from '@shared/routing/route-query'
import { parseImportRouteQuery, serializeImportRouteQuery } from '@shared/routing/import-route'
import { pagePath } from '@shared/routing/page-routes'
import type { SharedRouteQuery } from '@shared/routing/route-query'
import type { ExistingGroupImportDraft } from '@shared/domain/import/model-draft'
import type { MessageId } from '@shared/i18n/message-ids'

import { useStableLoading } from '../../app/collection-loading'
import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'
import { useUnsavedChanges } from '../../app/use-unsaved-changes'
import { InlineNotice } from '../../components/InlineNotice'
import { readSingleCredential } from './single-credential-input'
import { CredentialTextarea } from './CredentialTextarea'
import { SubscriptionCredentialStager } from './SubscriptionCredentialStager'
import { ImportOperationNotice } from './ImportOperationNotice'
import { useImportOperationOwner, useOperationSnapshot } from './import-operation'

const selectorPlaceholder = '__select_group__'
type ExistingStageDraft = ExistingGroupImportDraft & { staged_credential?: CredentialStage | null }

const narrow = '@media (max-width: 860px)'
const tiny = '@media (max-width: 640px)'

/**
 * Configure an empty group or reconnect its subscription account. The stable operation survives mode switches and page
 * remounts; the URL carries `mode=existing&group_id=N`.
 */
export function ExistingGroupImport({
  initialDraft,
}: {
  initialDraft?: ExistingGroupImportDraft | null
}) {
  const services = useAppServices()
  const apiClient = services.apiClient
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const t = useT()
  const toast = services.toast
  const { rawSearch } = useRouterState({
    select: (state) => ({ rawSearch: state.location.search as SharedRouteQuery }),
  })

  const owner = useImportOperationOwner()
  const importSnapshot = useOperationSnapshot(owner.importCredentials)
  const connectSnapshot = useOperationSnapshot(owner.connectCredentials)
  const operation = connectSnapshot.operation ? owner.connectCredentials : owner.importCredentials
  const snapshot = connectSnapshot.operation ? connectSnapshot : importSnapshot

  const [completed, setCompleted] = useState(false)
  const [errorKey, setErrorKey] = useState('')
  const [credentialValidation, setCredentialValidation] = useState<CredentialValidationData | null>(
    null,
  )
  const submissionErrorRef = useRef<HTMLElement>(null)
  const [errorFocusToken, setErrorFocusToken] = useState(0)
  useEffect(() => {
    if (errorFocusToken > 0) submissionErrorRef.current?.focus()
  }, [errorFocusToken])

  // Classic setup: a confirmed outcome clears the operation before first paint
  // (confirmed never maps to a notice key, so the one-frame window is invisible).
  const mountedRef = useRef(true)
  useEffect(() => {
    mountedRef.current = true
    if (operation.getSnapshot().outcome?.kind === 'confirmed') operation.reset()
    return () => {
      mountedRef.current = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mount only
  }, [])

  const [credentials, setCredentials] = useState(() => {
    const stableDraft = operation.getSnapshot().operation?.payload.draft
    return stableDraft?.mode === 'existing'
      ? stableDraft.credentials
      : (initialDraft?.credentials ?? '')
  })
  const [stage, setStage] = useState<CredentialStage | null>(() => {
    const stableDraft = owner.connectCredentials.getSnapshot().operation?.payload.draft
    const recovered = stableDraft?.mode === 'existing' ? stableDraft : initialDraft
    return recovered && 'staged_credential' in recovered && recovered.staged_credential
      ? projectCredentialStage(recovered.staged_credential)
      : null
  })

  const pending = snapshot.pending
  const payloadLocked = snapshot.operation !== null
  const outcome = snapshot.outcome
  const operationNoticeKey: MessageId | '' = !outcome
    ? ''
    : outcome.kind === 'reconciling'
      ? 'import.operation.reconciling'
      : outcome.kind === 'indeterminate'
        ? 'import.operation.indeterminate'
        : outcome.kind === 'failed' && outcome.reason === 'retryable-precondition'
          ? 'import.operation.waiting'
          : outcome.kind === 'failed' && outcome.reason === 'expired-known'
            ? 'import.operation.expired'
            : ''
  const operationResourceIdentity =
    outcome?.kind === 'failed' && outcome.reason === 'expired-known'
      ? outcome.resource_identity
      : ''

  const routeGroupID = Object.prototype.hasOwnProperty.call(rawSearch, 'group_id')
    ? parsePositiveRouteInteger(rawSearch.group_id)
    : undefined
  const operationGroupID = snapshot.operation?.payload.groupID
  const targetGroupID = operationGroupID ?? routeGroupID
  const summaryQuery = useQuery(groupSummaryQueryOptions(apiClient, targetGroupID))
  const hasCredential = summaryQuery.data?.credential_configured === true

  const groupsQuery = useQuery(groupOptionsQueryOptions(apiClient))
  const channelsQuery = useQuery(channelsQueryOptions(apiClient, ''))
  const groupsLoading = useStableLoading(groupsQuery.isPending && groupsQuery.data === undefined)
  const groupsRefreshing = groupsQuery.data !== undefined && groupsQuery.isFetching

  const selectedGroup =
    targetGroupID === undefined
      ? null
      : (groupsQuery.data?.find((group) => group.id === targetGroupID) ?? null)
  const targetGroups = groupsQuery.data ?? []
  const subscription = selectedGroup?.connection_type === 'subscription'
  const detailQuery = useQuery({
    ...credentialQueryOptions(apiClient, targetGroupID ?? 0),
    enabled: subscription && hasCredential && targetGroupID !== undefined,
  })
  const authState = detailQuery.data?.credential?.auth_state
  const reconnect =
    subscription &&
    !detailQuery.isError &&
    !detailQuery.isFetching &&
    (authState === 'reauthorization_required' || authState === 'outcome_unknown')
  const occupied = hasCredential && !reconnect
  const selectedChannel: ChannelDto | null = (() => {
    const channelID = selectedGroup?.channel_id
    return channelID
      ? (channelsQuery.data?.items.find((channel) => channel.channel_id === channelID) ?? null)
      : null
  })()
  const selectedGroupMissing =
    targetGroupID !== undefined && groupsQuery.data !== undefined && selectedGroup === null

  const credential = readSingleCredential(credentials)
  const canSubmit =
    !payloadLocked &&
    !pending &&
    selectedGroup !== null &&
    selectedChannel !== null &&
    (summaryQuery.data?.credential_configured === false || reconnect) &&
    !summaryQuery.isError &&
    !summaryQuery.isFetching &&
    (subscription ? stage?.status === 'ready' : credential !== null)
  const dirty = !completed && (credentials !== '' || stage !== null)
  const actionSummary = selectedGroup
    ? t('import.existing.actionSummary', { name: selectedGroup.name })
    : t('import.existing.actionSelectTarget')
  const submissionErrorMessage = credentialValidation
    ? t('import.credentials.validation', {
        entry: credentialValidation.entry,
        field: credentialValidation.field,
        reason: t(
          `import.credentials.validationReasons.${credentialValidation.reason_code}` as MessageId,
        ),
      })
    : errorKey
      ? t(errorKey as MessageId)
      : ''

  const unsavedChanges = useUnsavedChanges({
    dirty,
    blocked: pending,
    allowRouteUpdate: (current, next) =>
      next.pathname === current.pathname &&
      parseImportRouteQuery(next.search as SharedRouteQuery).mode === 'existing' &&
      parseImportRouteQuery(current.search as SharedRouteQuery).mode === 'existing',
  })

  // Classic recovery.register: snapshot the operation draft or the live form.
  const recoveryDraftRef = useRef(() => null as ExistingStageDraft | null)
  useEffect(() => {
    recoveryDraftRef.current = () =>
      completed
        ? null
        : operation.getSnapshot().operation?.payload.draft.mode === 'existing'
          ? (operation.getSnapshot().operation?.payload.draft as ExistingGroupImportDraft)
          : {
              mode: 'existing',
              group_id: targetGroupID ?? null,
              credentials,
              staged_credential: stage,
            }
  })
  useEffect(
    () => services.importRecovery.register(() => recoveryDraftRef.current()),
    [services.importRecovery],
  )

  // Classic watch(credentials): a fresh paste clears the validation panel.
  const [lastCredentials, setLastCredentials] = useState(credentials)
  if (lastCredentials !== credentials) {
    setLastCredentials(credentials)
    setCredentialValidation(null)
  }

  async function selectGroup(value: string): Promise<void> {
    if (payloadLocked) return
    setStage(null)
    setErrorKey('')
    if (value === selectorPlaceholder) {
      await unsavedChanges.runWithoutPrompt(() =>
        navigate({
          to: pagePath('import'),
          search: serializeImportRouteQuery({ mode: 'existing', discoveryFilter: 'unadded' }),
        }),
      )
      return
    }
    const id = parsePositiveRouteInteger(value)
    if (id === undefined) return
    await unsavedChanges.runWithoutPrompt(() =>
      navigate({
        to: pagePath('import'),
        search: serializeImportRouteQuery({
          mode: 'existing',
          groupID: id,
          discoveryFilter: 'unadded',
        }),
      }),
    )
  }

  async function executeImportOperation(): Promise<void> {
    const operation = owner.connectCredentials.getSnapshot().operation
      ? owner.connectCredentials
      : owner.importCredentials
    const current = operation.getSnapshot().operation
    if (!current) return
    setErrorKey('')
    setCredentialValidation(null)
    const outcome = owner.connectCredentials.getSnapshot().operation
      ? await owner.connectCredentials.execute(async (stableOperation, signal) => {
          const result = await connectGroupCredential(
            apiClient,
            stableOperation.payload.groupID,
            stableOperation.payload.stageID,
            stableOperation.idempotencyKey,
            stableOperation.payload.expectedCredentialID,
            signal,
          )
          if (result.group_id !== stableOperation.payload.groupID) throw new InvalidResponseError()
          return result
        })
      : await owner.importCredentials.execute(async (stableOperation, signal) => {
          const imported = await importGroupCredentials(
            apiClient,
            stableOperation.payload.groupID,
            { credential: stableOperation.payload.credentials },
            stableOperation.idempotencyKey,
            signal,
          )
          if (imported.group_id !== stableOperation.payload.groupID) {
            throw new InvalidResponseError()
          }
          return imported
        })
    if (!outcome) return
    if (outcome.kind === 'confirmed') {
      const targetID = current.payload.groupID
      setCompleted(true)
      setCredentials('')
      setStage(null)
      services.importRecovery.clear()
      operation.reset()
      await applyInvalidationPlan(
        queryClient,
        mutationInvalidationPlans.group.importCredentials(targetID),
      )
      if (!mountedRef.current) return
      toast.show({
        message: t('group.settings.savedFeedback'),
        tone: 'success',
        duration: 4_000,
      })
      await unsavedChanges.runWithoutPrompt(() =>
        navigate({ to: `${pagePath('groups')}/${targetID}` }),
      )
      return
    }
    if (!mountedRef.current) return
    if (outcome.kind === 'failed' && outcome.reason === 'rejected') {
      const cause = operation.getSnapshot().lastError
      const validation =
        cause instanceof ApiError && cause.code === 'VALIDATION_FAILED'
          ? readCredentialValidationData(cause.data)
          : null
      operation.reset()
      if (validation) {
        setCredentialValidation(validation)
      } else {
        setErrorKey(
          cause instanceof ApiError && cause.code === 'SINGLE_CREDENTIAL_REQUIRED'
            ? 'import.existing.populated'
            : 'import.existing.importFailed',
        )
      }
      setErrorFocusToken((token) => token + 1)
    }
  }

  async function submit(): Promise<void> {
    const groupID = targetGroupID
    if (groupID === undefined || !selectedGroup || !canSubmit) return
    if (subscription) {
      if (!stage || stage.expires_at_ms <= Date.now()) return
      const expectedCredentialID = reconnect ? detailQuery.data?.credential?.credential_id : 0
      if (expectedCredentialID === undefined) return
      const draft: ExistingStageDraft = {
        mode: 'existing',
        group_id: groupID,
        credentials: '',
        staged_credential: stage,
      }
      if (
        !owner.beginConnectCredentials(
          { groupID, stageID: stage.stage_id, expectedCredentialID },
          draft,
        )
      )
        return
      await executeImportOperation()
      return
    }
    if (
      !owner.beginImportCredentials({ groupID, credentials }, 'existing', {
        mode: 'existing',
        group_id: groupID,
        credentials,
      })
    ) {
      return
    }
    await executeImportOperation()
  }

  async function abandonOperation(): Promise<void> {
    if (pending || !payloadLocked) return
    if (!(await unsavedChanges.confirmDiscard()) || pending) return
    operation.reset()
    setCredentialValidation(null)
    setErrorKey('')
  }

  return (
    <div {...stylex.props(styles.root)}>
      <div {...stylex.props(styles.notice)}>
        <ImportOperationNotice
          messageKey={operationNoticeKey}
          resourceIdentity={operationResourceIdentity}
          canRetry={snapshot.canRetry}
          canAbandon={payloadLocked && !pending}
          pending={pending}
          onRetry={() => void executeImportOperation()}
          onAbandon={() => void abandonOperation()}
        />
      </div>

      <div {...stylex.props(styles.intro)}>
        <InlineNotice tone="neutral" appearance="ledger-hint" glyph="i">
          {t('import.existing.description')}
        </InlineNotice>
      </div>

      <section {...stylex.props(styles.target)} aria-labelledby="existing-target-heading">
        <header {...stylex.props(styles.sectionHeader)}>
          <h2 id="existing-target-heading" {...stylex.props(styles.sectionTitle)}>
            {t('import.existing.title')}
          </h2>
          <div {...stylex.props(styles.sectionActions)}>
            {groupsQuery.data !== undefined && (
              <span {...stylex.props(styles.groupCount)}>
                {t('import.existing.groupCount', { count: targetGroups.length })}
              </span>
            )}
          </div>
        </header>

        <div
          role="status"
          aria-live="polite"
          {...stylex.props(groupsRefreshing ? styles.refreshVisible : styles.refreshHidden)}
        >
          {groupsRefreshing ? t('import.existing.groupsLoading') : ''}
        </div>

        {(groupsQuery.isPending && groupsQuery.data === undefined) || groupsLoading ? (
          <div
            role="status"
            aria-label={t('import.existing.groupsLoading')}
            {...stylex.props(styles.skeleton)}
          >
            <Skeleton height={64} radius={2} />
          </div>
        ) : groupsQuery.isError && groupsQuery.data === undefined ? (
          <div {...stylex.props(styles.queryError)}>
            <InlineNotice tone="danger">{t('import.existing.groupsFailed')}</InlineNotice>
            <Button
              variant="secondary"
              size="sm"
              label={t('common.retry')}
              onClick={() => void groupsQuery.refetch()}
            />
          </div>
        ) : (
          <>
            {groupsQuery.isError && (
              <InlineNotice tone="warning">{t('import.existing.groupsStale')}</InlineNotice>
            )}
            {targetGroups.length === 0 && (
              <InlineNotice tone="info">{t('import.existing.groupsEmpty')}</InlineNotice>
            )}
            <div {...stylex.props(styles.targetBody)}>
              <Selector
                label={t('import.existing.groupLabel')}
                options={[
                  { value: selectorPlaceholder, label: t('import.existing.groupPlaceholder') },
                  ...targetGroups.map((group) => ({
                    value: String(group.id),
                    label: t('import.existing.groupOption', {
                      id: group.id,
                      name: group.name,
                    }),
                  })),
                ]}
                value={selectedGroup ? String(selectedGroup.id) : selectorPlaceholder}
                size="sm"
                isDisabled={payloadLocked || targetGroups.length === 0}
                onChange={(value) => void selectGroup(value)}
              />
              {selectedGroup && (
                <div {...stylex.props(styles.groupMeta)}>
                  <strong {...stylex.props(styles.groupMetaTitle)}>
                    {t('import.existing.groupMeta', {
                      id: selectedGroup.id,
                      models: selectedGroup.models.length,
                    })}
                  </strong>
                  <span>{t('import.existing.groupUnchanged')}</span>
                </div>
              )}
            </div>
            {selectedGroupMissing && (
              <div {...stylex.props(styles.missingGroup)}>
                <InlineNotice tone="danger">
                  {t('import.existing.groupNotFound', {
                    id: targetGroupID ?? 0,
                  })}
                </InlineNotice>
              </div>
            )}
          </>
        )}
      </section>

      {occupied && selectedGroup && (
        <div {...stylex.props(styles.credentials)}>
          <InlineNotice tone="info">{t('import.existing.populated')}</InlineNotice>
          <Button
            variant="secondary"
            size="sm"
            label={t('import.existing.manage')}
            href={`${pagePath('groups')}/${selectedGroup.id}`}
          />
        </div>
      )}
      {selectedGroup && summaryQuery.isPending && <div role="status">{t('group.loading')}</div>}
      {selectedGroup && summaryQuery.isError && (
        <div {...stylex.props(styles.queryError)}>
          <InlineNotice tone="danger">{t('group.loadFailed')}</InlineNotice>
          <Button
            variant="secondary"
            size="sm"
            label={t('common.retry')}
            onClick={() => void summaryQuery.refetch()}
          />
        </div>
      )}
      {subscription && hasCredential && detailQuery.isError && (
        <div {...stylex.props(styles.queryError)}>
          <InlineNotice tone="danger">{t('group.loadFailed')}</InlineNotice>
          <Button
            variant="secondary"
            size="sm"
            label={t('common.retry')}
            onClick={() => void detailQuery.refetch()}
          />
        </div>
      )}
      {!occupied && subscription && selectedChannel && (
        <SubscriptionCredentialStager
          stage={stage}
          onStageChange={setStage}
          channelId={selectedChannel.channel_id}
          channelName={selectedChannel.name}
          authorizationMethods={selectedChannel.connection.authorization_methods}
          groupId={targetGroupID}
          context="connect"
          disabled={
            payloadLocked ||
            pending ||
            summaryQuery.data === undefined ||
            summaryQuery.isFetching ||
            summaryQuery.isError
          }
        />
      )}
      {!occupied && !subscription && (
        <div {...stylex.props(styles.credentials)}>
          <CredentialTextarea
            value={credentials}
            channel={selectedChannel}
            disabled={
              payloadLocked ||
              summaryQuery.data === undefined ||
              summaryQuery.isError ||
              summaryQuery.isFetching
            }
            showHeaderDescription={false}
            storageDescription={t('import.existing.credentialStorageNotice')}
            showCredentialNotice={false}
            rows={8}
            onChange={setCredentials}
          />
        </div>
      )}

      {selectedGroup && channelsQuery.isError && channelsQuery.data === undefined && (
        <div {...stylex.props(styles.queryError)}>
          <InlineNotice tone="danger">{t('import.presets.loadFailed')}</InlineNotice>
          <Button
            variant="secondary"
            size="sm"
            label={t('common.retry')}
            onClick={() => void channelsQuery.refetch()}
          />
        </div>
      )}

      {submissionErrorMessage && (
        <div
          ref={submissionErrorRef as React.RefObject<HTMLDivElement>}
          {...stylex.props(styles.error)}
          tabIndex={-1}
        >
          <InlineNotice tone="danger">{submissionErrorMessage}</InlineNotice>
        </div>
      )}

      {!occupied && (
        <footer {...stylex.props(styles.actions)}>
          <div aria-live="polite" {...stylex.props(styles.actionsSummary)}>
            <strong {...stylex.props(styles.actionsSummaryTitle)}>{actionSummary}</strong>
            <span {...stylex.props(styles.actionsSummaryHelp)}>
              {t('import.existing.actionHelp')}
            </span>
          </div>
          <Button
            size="sm"
            isLoading={pending}
            isDisabled={!canSubmit}
            label={t('import.existing.submit')}
            onClick={() => void submit()}
          />
        </footer>
      )}
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
  intro: {
    marginTop: '22px',
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    color: 'var(--color-text-muted)',
    paddingBottom: '18px',
    fontSize: 'var(--text-sm)',
    lineHeight: 1.55,
  },
  target: {
    display: 'grid',
    gap: 'var(--space-3)',
    minWidth: 0,
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    paddingTop: '22px',
    paddingBottom: 'var(--space-6)',
  },
  sectionHeader: {
    display: 'flex',
    alignItems: { default: 'flex-start', [tiny]: 'stretch' },
    flexDirection: { default: 'row', [tiny]: 'column' },
    justifyContent: 'space-between',
    gap: 'var(--space-4)',
  },
  sectionTitle: {
    margin: 0,
    fontSize: 'var(--title-section)',
    fontWeight: 650,
    letterSpacing: 0,
  },
  sectionActions: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  groupCount: {
    display: 'inline-flex',
    minHeight: '23px',
    alignItems: 'center',
    gap: '6px',
    borderRadius: '999px',
    backgroundColor: 'var(--color-neutral-bg)',
    color: 'var(--color-neutral-fg)',
    paddingBlock: '2px',
    paddingInline: 'var(--space-2)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 590,
    whiteSpace: 'nowrap',
    '::before': {
      width: '6px',
      height: '6px',
      flexShrink: 0,
      borderRadius: '50%',
      backgroundColor: 'currentColor',
      content: '""',
    },
  },
  refreshVisible: {
    minHeight: '18px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  refreshHidden: {
    minHeight: '18px',
    visibility: 'hidden',
  },
  skeleton: {
    display: 'grid',
    gap: 'var(--space-2)',
    minHeight: '102px',
  },
  queryError: {
    display: 'flex',
    alignItems: { default: 'center', [tiny]: 'stretch' },
    flexDirection: { default: 'row', [tiny]: 'column' },
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
  },
  targetBody: {
    display: 'grid',
    gridTemplateColumns: { default: 'minmax(280px, 0.66fr) minmax(300px, 1fr)', [narrow]: '1fr' },
    alignItems: 'end',
    gap: '18px',
  },
  groupMeta: {
    display: 'flex',
    minHeight: 'var(--control-xs)',
    minWidth: 0,
    flexWrap: 'wrap',
    alignItems: 'center',
    justifyContent: 'flex-start',
    gap: 'var(--space-4)',
    borderLeftWidth: { default: '1px', [narrow]: 0 },
    borderLeftStyle: 'solid',
    borderLeftColor: 'var(--color-border-subtle)',
    color: 'var(--color-text-faint)',
    paddingLeft: { default: '18px', [narrow]: 0 },
    fontSize: 'var(--text-meta)',
  },
  groupMetaTitle: {
    fontWeight: 560,
  },
  missingGroup: {
    marginTop: 'var(--space-2)',
  },
  credentials: {
    marginTop: 'var(--space-2)',
  },
  error: {
    marginTop: 'var(--space-5)',
    outlineStyle: 'none',
  },
  actions: {
    display: 'flex',
    alignItems: { default: 'center', [narrow]: 'stretch' },
    flexDirection: { default: 'row', [narrow]: 'column' },
    justifyContent: 'space-between',
    gap: 'var(--space-4)',
    minHeight: '64px',
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    marginTop: 'var(--space-5)',
    paddingTop: 'var(--space-4)',
  },
  actionsSummary: {
    minWidth: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  actionsSummaryTitle: {
    display: 'block',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
    fontWeight: 560,
  },
  actionsSummaryHelp: {
    display: 'block',
    marginTop: '2px',
  },
})
