import * as stylex from '@stylexjs/stylex'
import { Button, Selector, Skeleton } from '@astryxdesign/core'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import { useEffect, useRef, useState } from 'react'

import { ApiError, InvalidResponseError } from '@shared/http/errors'
import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import { channelsQueryOptions, type ChannelDto } from '@shared/control/resources/channels'
import {
  groupOptionsQueryOptions,
  importGroupCredentials,
  readCredentialValidationData,
  type CredentialValidationData,
} from '@shared/control/resources/groups'
import { parsePositiveRouteInteger } from '@shared/routing/route-query'
import {
  parseImportRouteQuery,
  serializeImportRouteQuery,
} from '@shared/routing/import-route'
import { pagePath } from '@shared/routing/page-routes'
import type { SharedRouteQuery } from '@shared/routing/route-query'
import type { ExistingGroupImportDraft } from '@shared/domain/import/model-draft'
import type { MessageId } from '@shared/i18n/message-ids'

import { useStableLoading } from '../../app/collection-loading'
import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'
import { useUnsavedChanges } from '../../app/use-unsaved-changes'
import { InlineNotice } from '../../components/InlineNotice'
import { analyzeCredentials } from '@shared/domain/import/credential-analysis'
import { CredentialTextarea } from './CredentialTextarea'
import { ImportOperationNotice } from './ImportOperationNotice'
import { useImportOperationOwner, useOperationSnapshot } from './import-operation'

const selectorPlaceholder = '__select_group__'

const narrow = '@media (max-width: 860px)'
const tiny = '@media (max-width: 640px)'

/**
 * Classic features/import/ExistingGroupImport.vue — append credentials to an
 * existing api_key group. The stable operation survives mode switches and page
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
  const operation = owner.importCredentials
  const snapshot = useOperationSnapshot(operation)

  const [completed, setCompleted] = useState(false)
  const [errorKey, setErrorKey] = useState('')
  const [credentialValidation, setCredentialValidation] =
    useState<CredentialValidationData | null>(null)
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

  const groupsQuery = useQuery(groupOptionsQueryOptions(apiClient))
  const channelsQuery = useQuery(channelsQueryOptions(apiClient, ''))
  const groupsLoading = useStableLoading(
    groupsQuery.isPending && groupsQuery.data === undefined,
  )
  const groupsRefreshing = groupsQuery.data !== undefined && groupsQuery.isFetching

  const selectedGroup =
    targetGroupID === undefined
      ? null
      : (groupsQuery.data?.find((group) => group.id === targetGroupID) ?? null)
  const apiKeyGroups = (groupsQuery.data ?? []).filter(
    ({ connection_type }) => connection_type === 'api_key',
  )
  const selectedChannel: ChannelDto | null = (() => {
    const channelID = selectedGroup?.channel_id
    return channelID
      ? (channelsQuery.data?.items.find((channel) => channel.channel_id === channelID) ??
          null)
      : null
  })()
  const selectedGroupMissing =
    targetGroupID !== undefined &&
    groupsQuery.data !== undefined &&
    selectedGroup === null

  const credentialAnalysis = analyzeCredentials(
    credentials,
    selectedGroup?.channel_id,
  )
  const canSubmit =
    !payloadLocked &&
    !pending &&
    selectedGroup !== null &&
    selectedChannel !== null &&
    credentialAnalysis.nonEmptyCount > 0 &&
    !credentialAnalysis.tooManyCredentials
  const dirty = !completed && credentials !== ''
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
  const recoveryDraftRef = useRef(() => null as ExistingGroupImportDraft | null)
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
    const current = operation.getSnapshot().operation
    if (!current) return
    setErrorKey('')
    setCredentialValidation(null)
    const outcome = await operation.execute(async (stableOperation, signal) => {
      const imported = await importGroupCredentials(
        apiClient,
        stableOperation.payload.groupID,
        { credentials: stableOperation.payload.credentials },
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
      services.importRecovery.clear()
      operation.reset()
      await applyInvalidationPlan(
        queryClient,
        mutationInvalidationPlans.group.importCredentials(targetID),
      )
      if (!mountedRef.current) return
      toast.show({
        message: t('import.credentials.result', {
          added: outcome.value.credentials_added,
          duplicated: outcome.value.credentials_duplicated,
        }),
        tone: outcome.value.credentials_added === 0 ? 'warning' : 'success',
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
        setErrorKey('import.existing.importFailed')
      }
      setErrorFocusToken((token) => token + 1)
    }
  }

  async function submit(): Promise<void> {
    const groupID = targetGroupID
    if (groupID === undefined || !selectedGroup || !canSubmit) return
    if (
      !owner.beginImportCredentials(
        { groupID, credentials },
        'existing',
        { mode: 'existing', group_id: groupID, credentials },
      )
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
                {t('import.existing.groupCount', { count: apiKeyGroups.length })}
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
            {apiKeyGroups.length === 0 && (
              <InlineNotice tone="info">{t('import.existing.groupsEmpty')}</InlineNotice>
            )}
            <div {...stylex.props(styles.targetBody)}>
              <Selector
                label={t('import.existing.groupLabel')}
                options={[
                  { value: selectorPlaceholder, label: t('import.existing.groupPlaceholder') },
                  ...apiKeyGroups.map((group) => ({
                    value: String(group.id),
                    label: t('import.existing.groupOption', {
                      id: group.id,
                      name: group.name,
                    }),
                  })),
                ]}
                value={selectedGroup ? String(selectedGroup.id) : selectorPlaceholder}
                size="sm"
                isDisabled={payloadLocked || apiKeyGroups.length === 0}
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

      <div {...stylex.props(styles.credentials)}>
        <CredentialTextarea
          value={credentials}
          channel={selectedChannel}
          disabled={payloadLocked}
          showHeaderDescription={false}
          storageDescription={t('import.existing.credentialStorageNotice')}
          duplicateLabel={t('import.existing.batchDuplicates')}
          showCredentialNotice={false}
          rows={8}
          onChange={setCredentials}
        />
      </div>

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
    fontSize: '11px',
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
    letterSpacing: '-0.01em',
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
    color: 'var(--color-neutral)',
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
    alignItems: 'center',
    justifyContent: 'flex-start',
    gap: 'var(--space-4)',
    borderLeftWidth: { default: '1px', [narrow]: 0 },
    borderLeftStyle: 'solid',
    borderLeftColor: 'var(--color-border-subtle)',
    color: 'var(--color-text-faint)',
    paddingLeft: { default: '18px', [narrow]: 0 },
    fontSize: '10.8px',
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
