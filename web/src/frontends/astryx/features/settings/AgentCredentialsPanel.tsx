import * as stylex from '@stylexjs/stylex'
import {
  AlertDialog,
  Badge,
  Banner,
  Button,
  CheckboxInput,
  DateTimeInput,
  Dialog,
  DialogHeader,
  EmptyState,
  Layout,
  LayoutContent,
  LayoutFooter,
  Selector,
  Skeleton,
  Table,
  TextInput,
  proportional,
  pixel,
  type ISODateTimeString,
  type TableColumn,
} from '@astryxdesign/core'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Bot, Plus, TriangleAlert } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useIntl } from 'react-intl'

import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import {
  agentCredentialListQueryOptions,
  agentCredentialResources,
  agentCredentialScopes,
  createAgentCredential,
  disableAgentCredential,
} from '@shared/control/resources/agent-credentials'
import type { AgentCredentialDto, AgentCredentialScope } from '@shared/control/types'
import { RequestCancelledError } from '@shared/http/errors'
import type { MessageId } from '@shared/i18n/message-ids'
import { formatLocalInstant } from '@shared/lib/format'
import { currentTimeZone } from '@shared/lib/time'
import { createUUID } from '@shared/lib/uuid'

import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'
import { plainTextInputAttrs } from '../../components/input-attrs'
import { CopyChip } from '../access-keys/AccessKeyCopyChip'

const styles = stylex.create({
  panel: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-3)',
  },
  headerRow: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
  },
  heading: {
    margin: 0,
    fontSize: 'var(--text-meta)',
    fontWeight: 650,
  },
  headingDescription: {
    margin: 0,
    marginTop: 'var(--space-1)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  nameCell: {
    minWidth: 0,
    color: 'var(--color-text)',
    fontWeight: 600,
    overflowWrap: 'anywhere',
  },
  scopeCell: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 5,
  },
  statusCell: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 5,
  },
  timeCell: {
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
  },
  dialogBody: {
    display: 'grid',
    gap: 'var(--space-3)',
  },
  scopeGroup: {
    display: 'grid',
    gap: 'var(--space-2)',
    borderWidth: 0,
    padding: 0,
    margin: 0,
  },
  scopeLegend: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    marginBottom: 'var(--space-1)',
  },
  scopeRow: {
    display: 'grid',
    gridTemplateColumns: 'auto minmax(0, 1fr)',
    alignItems: 'start',
    gap: 'var(--space-2)',
  },
  scopeText: {
    display: 'grid',
    gap: '2px',
  },
  scopeName: {
    fontSize: 'var(--text-sm)',
    fontWeight: 600,
  },
  scopeHint: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
  },
  secretValue: {
    display: 'block',
    maxWidth: '100%',
    paddingBlock: 'var(--space-2)',
    paddingInline: 'var(--space-3)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-code)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
    overflowWrap: 'anywhere',
    userSelect: 'all',
  },
  skeletonGrid: {
    display: 'grid',
    gap: 'var(--space-2)',
  },
})

function pad(value: number): string {
  return String(value).padStart(2, '0')
}

function localDateTimeValue(epochMS: number | null): string {
  if (epochMS === null || !Number.isSafeInteger(epochMS) || epochMS <= 0) return ''
  const date = new Date(epochMS)
  if (Number.isNaN(date.getTime())) return ''
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
}

const scopeLabels: Record<AgentCredentialScope, { label: MessageId; hint: MessageId }> = {
  'diagnostics:read': {
    label: 'settings.credentials.agent.scopeDiagnostics',
    hint: 'settings.credentials.agent.scopeDiagnosticsHint',
  },
  'changes:propose': {
    label: 'settings.credentials.agent.scopePropose',
    hint: 'settings.credentials.agent.scopeProposeHint',
  },
  'changes:apply': {
    label: 'settings.credentials.agent.scopeApply',
    hint: 'settings.credentials.agent.scopeApplyHint',
  },
}

type AgentCredentialRow = AgentCredentialDto & Record<string, unknown>

function isExpired(credential: AgentCredentialDto): boolean {
  return credential.expires_at_ms !== null && credential.expires_at_ms <= Date.now()
}

export function AgentCredentialsPanel() {
  const t = useT()
  const intl = useIntl()
  const { apiClient, toast } = useAppServices()
  const queryClient = useQueryClient()

  const listQuery = useQuery(agentCredentialListQueryOptions(apiClient))
  const [createOpen, setCreateOpen] = useState(false)
  const [disableTarget, setDisableTarget] = useState<AgentCredentialDto | null>(null)
  const [disableBusy, setDisableBusy] = useState(false)
  const mountedRef = useRef(true)
  useEffect(
    () => () => {
      mountedRef.current = false
    },
    [],
  )
  useEffect(
    () => () => {
      queryClient.removeQueries({ queryKey: agentCredentialResources.list.queryKey })
    },
    [queryClient],
  )

  const items = useMemo(
    () => [...(listQuery.data?.items ?? [])].sort((a, b) => b.created_at_ms - a.created_at_ms),
    [listQuery.data],
  )

  const columns: TableColumn<AgentCredentialRow>[] = [
    {
      key: 'name',
      header: t('settings.credentials.agent.columnName'),
      width: proportional(1.1),
      renderCell: (record): ReactNode => (
        <span {...stylex.props(styles.nameCell)}>{record.name}</span>
      ),
    },
    {
      key: 'scopes',
      header: t('settings.credentials.agent.columnScopes'),
      width: proportional(1.6),
      renderCell: (record): ReactNode => (
        <span {...stylex.props(styles.scopeCell)}>
          {record.scopes.map((scope) => (
            <Badge
              key={scope}
              variant={scope === 'diagnostics:read' ? 'neutral' : 'warning'}
              label={t(scopeLabels[scope].label)}
            />
          ))}
        </span>
      ),
    },
    {
      key: 'status',
      header: t('settings.credentials.agent.columnStatus'),
      width: pixel(120),
      renderCell: (record): ReactNode => (
        <span {...stylex.props(styles.statusCell)}>
          <Badge
            variant={record.status === 'active' ? 'success' : 'neutral'}
            label={t(`accessKeys.status.${record.status}` as MessageId)}
          />
          {record.status === 'active' && isExpired(record) && (
            <Badge variant="error" label={t('accessKeys.status.expired')} />
          )}
        </span>
      ),
    },
    {
      key: 'expires',
      header: t('settings.credentials.agent.columnExpires'),
      width: pixel(170),
      renderCell: (record): ReactNode => (
        <span {...stylex.props(styles.timeCell)}>
          {record.expires_at_ms === null
            ? t('settings.credentials.agent.expiresNever')
            : formatLocalInstant(record.expires_at_ms, intl.locale)}
        </span>
      ),
    },
    {
      key: 'actions',
      header: t('settings.credentials.agent.columnActions'),
      width: pixel(110),
      renderCell: (record): ReactNode => (
        <Button
          variant="secondary"
          size="sm"
          isDisabled={record.status === 'disabled'}
          label={t('settings.credentials.agent.disable')}
          onClick={() => setDisableTarget(record)}
        />
      ),
    },
  ]

  async function confirmDisable(): Promise<void> {
    const target = disableTarget
    if (target === null || disableBusy) return
    setDisableBusy(true)
    try {
      await disableAgentCredential(apiClient, target.id)
    } catch (error: unknown) {
      if (!(error instanceof RequestCancelledError) && mountedRef.current) {
        toast.show({ message: t('settings.credentials.agent.disableFailed'), tone: 'danger' })
      }
      return
    } finally {
      if (mountedRef.current) setDisableBusy(false)
    }
    setDisableTarget(null)
    try {
      await applyInvalidationPlan(queryClient, mutationInvalidationPlans.agentCredential.disable)
    } catch {
      void queryClient.invalidateQueries({ queryKey: agentCredentialResources.list.queryKey })
    }
    if (mountedRef.current) {
      toast.show({ message: t('settings.credentials.agent.disabledToast', { name: target.name }) })
    }
  }

  return (
    <div {...stylex.props(styles.panel)}>
      <div {...stylex.props(styles.headerRow)}>
        <div>
          <h3 {...stylex.props(styles.heading)}>{t('settings.credentials.agent.title')}</h3>
          <p {...stylex.props(styles.headingDescription)}>
            {t('settings.credentials.agent.description')}
          </p>
        </div>
        <Button
          className="agent-credential-create"
          size="sm"
          icon={<Plus size={15} aria-hidden />}
          label={t('settings.credentials.agent.create')}
          onClick={() => setCreateOpen(true)}
        />
      </div>

      {listQuery.isPending ? (
        <div
          {...stylex.props(styles.skeletonGrid)}
          role="status"
          aria-label={t('settings.credentials.agent.loading')}
        >
          <Skeleton height={72} radius={2} />
        </div>
      ) : listQuery.isError ? (
        <div role="alert">
          <EmptyState
            title={t('settings.credentials.agent.errorTitle')}
            description={t('settings.credentials.agent.errorDescription')}
            icon={<TriangleAlert size={20} />}
            actions={
              <Button
                variant="secondary"
                size="sm"
                label={t('common.retry')}
                onClick={() => void listQuery.refetch()}
              />
            }
          />
        </div>
      ) : items.length === 0 ? (
        <EmptyState
          title={t('settings.credentials.agent.emptyTitle')}
          description={t('settings.credentials.agent.emptyDescription')}
          icon={<Bot size={20} />}
          actions={
            <Button
              size="sm"
              icon={<Plus size={15} aria-hidden />}
              label={t('settings.credentials.agent.create')}
              onClick={() => setCreateOpen(true)}
            />
          }
        />
      ) : (
        <Table
          data={items as AgentCredentialRow[]}
          columns={columns}
          density="compact"
          dividers="rows"
          hasHover
          aria-label={t('settings.credentials.agent.tableLabel')}
        />
      )}

      <AgentCredentialCreateDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        onCreated={(name, hasSecret) => {
          toast.show({
            message: hasSecret
              ? t('settings.credentials.agent.createdToast', { name })
              : t('settings.credentials.agent.replayedToast', { name }),
          })
        }}
      />

      <AlertDialog
        isOpen={disableTarget !== null}
        onOpenChange={(value) => {
          if (!value && !disableBusy) setDisableTarget(null)
        }}
        title={t('settings.credentials.agent.disableTitle')}
        description={t('settings.credentials.agent.disableDescription', {
          name: disableTarget?.name ?? '',
        })}
        cancelLabel={t('common.cancel')}
        actionLabel={t('settings.credentials.agent.disable')}
        isActionLoading={disableBusy}
        onAction={() => void confirmDisable()}
      />
    </div>
  )
}

function AgentCredentialCreateDialog({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean
  onOpenChange(open: boolean): void
  onCreated(name: string, hasSecret: boolean): void
}) {
  const t = useT()
  const { apiClient, queryClient } = useAppServices()
  const [name, setName] = useState('')
  const [scopes, setScopes] = useState<ReadonlySet<AgentCredentialScope>>(
    () => new Set<AgentCredentialScope>(['diagnostics:read']),
  )
  const [expirationMode, setExpirationMode] = useState<'never' | 'specified'>('never')
  const [expiresAt, setExpiresAt] = useState<number | null>(null)
  const [now, setNow] = useState(() => Date.now())
  const [pending, setPending] = useState(false)
  const [failed, setFailed] = useState(false)
  const [created, setCreated] = useState<{ name: string; secret: string } | null>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const nameInputRef = useRef<HTMLInputElement | null>(null)

  useEffect(
    () => () => {
      controllerRef.current?.abort()
    },
    [],
  )
  useEffect(() => {
    if (!open) return
    const frame = requestAnimationFrame(() => nameInputRef.current?.focus())
    return () => cancelAnimationFrame(frame)
  }, [open])

  const expirationError = (() => {
    if (expirationMode !== 'specified') return undefined
    if (expiresAt === null || expiresAt <= 0) {
      return t('settings.credentials.agent.expirationRequired')
    }
    if (expiresAt <= now) return t('settings.credentials.agent.expirationFuture')
    return undefined
  })()

  const nameBlank = name.trim().length === 0
  const valid = !nameBlank && scopes.size > 0 && expirationError === undefined

  const setDialogOpen = (value: boolean) => {
    if (!value && pending) return
    if (!value) {
      controllerRef.current?.abort()
      controllerRef.current = null
      setFailed(false)
      setName('')
      setScopes(new Set<AgentCredentialScope>(['diagnostics:read']))
      setExpirationMode('never')
      setExpiresAt(null)
      setCreated(null)
    }
    onOpenChange(value)
  }

  const setScope = (scope: AgentCredentialScope, checked: boolean) => {
    setScopes((previous) => {
      const next = new Set(previous)
      if (checked) next.add(scope)
      else next.delete(scope)
      return next
    })
  }

  const expirationOptions = [
    { value: 'never', label: t('settings.credentials.agent.expirationNever') },
    { value: 'specified', label: t('settings.credentials.agent.expirationSpecified') },
  ]

  const submit = async () => {
    if (!valid || pending) return
    setPending(true)
    setFailed(false)
    const controller = new AbortController()
    controllerRef.current = controller
    try {
      const result = await createAgentCredential(
        apiClient,
        {
          name: name.trim(),
          scopes: agentCredentialScopes.filter((scope) => scopes.has(scope)),
          expires_at_ms: expirationMode === 'specified' ? expiresAt : null,
        },
        createUUID(),
        controller.signal,
      )
      setCreated(null)
      if (result.secret !== undefined) {
        // One-time plaintext: keep it in dialog state only — the backend stores
        // no recoverable copy and replays never carry it again.
        setCreated({ name: result.name, secret: result.secret })
      } else {
        setDialogOpen(false)
      }
      try {
        await applyInvalidationPlan(queryClient, mutationInvalidationPlans.agentCredential.create)
      } catch {
        void queryClient.invalidateQueries({ queryKey: agentCredentialResources.list.queryKey })
      }
      onCreated(result.name, result.secret !== undefined)
    } catch (error: unknown) {
      if (!(error instanceof RequestCancelledError)) setFailed(true)
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null
      setPending(false)
    }
  }

  return (
    <Dialog isOpen={open} onOpenChange={setDialogOpen} width={520}>
      <Layout
        header={
          <DialogHeader
            title={t(
              created === null
                ? 'settings.credentials.agent.createTitle'
                : 'settings.credentials.agent.secretTitle',
            )}
            subtitle={t(
              created === null
                ? 'settings.credentials.agent.createDescription'
                : 'settings.credentials.agent.secretDescription',
            )}
            onOpenChange={setDialogOpen}
            hasDivider
          />
        }
        content={
          <LayoutContent isScrollable>
            {created !== null ? (
              <div {...stylex.props(styles.dialogBody)}>
                <Banner status="warning" title={t('settings.credentials.agent.secretWarning')} />
                <code {...stylex.props(styles.secretValue)}>{created.secret}</code>
                <div>
                  <CopyChip
                    value={t('settings.credentials.agent.secretCopy')}
                    label={t('settings.credentials.agent.secretCopy')}
                    successLabel={t('common.copied')}
                    failureLabel={t('common.copyFailed')}
                    resolveValue={() => created.secret}
                  />
                </div>
              </div>
            ) : (
              <form
                {...stylex.props(styles.dialogBody)}
                onSubmit={(event) => {
                  event.preventDefault()
                  void submit()
                }}
              >
                <TextInput
                  ref={nameInputRef}
                  label={t('settings.credentials.agent.nameLabel')}
                  placeholder={t('settings.credentials.agent.namePlaceholder')}
                  value={name}
                  onChange={setName}
                  isDisabled={pending}
                  autoComplete="off"
                  {...plainTextInputAttrs}
                />
                <fieldset {...stylex.props(styles.scopeGroup)}>
                  <legend {...stylex.props(styles.scopeLegend)}>
                    {t('settings.credentials.agent.scopesLegend')}
                  </legend>
                  {agentCredentialScopes.map((scope) => (
                    <label key={scope} {...stylex.props(styles.scopeRow)}>
                      <CheckboxInput
                        value={scopes.has(scope)}
                        isDisabled={pending}
                        label={t(scopeLabels[scope].label)}
                        isLabelHidden
                        onChange={(checked) => setScope(scope, checked)}
                      />
                      <span {...stylex.props(styles.scopeText)}>
                        <span {...stylex.props(styles.scopeName)}>
                          {t(scopeLabels[scope].label)}
                        </span>
                        <span {...stylex.props(styles.scopeHint)}>
                          {t(scopeLabels[scope].hint)}
                        </span>
                      </span>
                    </label>
                  ))}
                </fieldset>
                {scopes.size === 0 && (
                  <Banner status="warning" title={t('settings.credentials.agent.scopesRequired')} />
                )}
                <div>
                  <Selector
                    label={t('settings.credentials.agent.expiration')}
                    options={expirationOptions}
                    value={expirationMode}
                    size="sm"
                    isDisabled={pending}
                    onChange={(value) => {
                      if (value !== 'never' && value !== 'specified') return
                      setNow(Date.now())
                      setExpirationMode(value)
                    }}
                  />
                </div>
                {expirationMode === 'specified' && (
                  <DateTimeInput
                    label={t('settings.credentials.agent.expirationTime')}
                    description={t('settings.credentials.agent.expirationTimezone', {
                      timezone: currentTimeZone(),
                    })}
                    status={
                      expirationError !== undefined
                        ? { type: 'error', message: expirationError }
                        : undefined
                    }
                    size="sm"
                    hasSeconds
                    isDisabled={pending}
                    value={
                      localDateTimeValue(expiresAt) === ''
                        ? undefined
                        : (localDateTimeValue(expiresAt) as ISODateTimeString)
                    }
                    min={
                      localDateTimeValue(now + 1_000) === ''
                        ? undefined
                        : (localDateTimeValue(now + 1_000) as ISODateTimeString)
                    }
                    onChange={(value) => {
                      setNow(Date.now())
                      if (value === undefined || value === '') {
                        setExpiresAt(0)
                        return
                      }
                      const epochMS = new Date(value).getTime()
                      setExpiresAt(Number.isSafeInteger(epochMS) ? epochMS : 0)
                    }}
                  />
                )}
                {failed && (
                  <Banner status="error" title={t('settings.credentials.agent.createFailed')} />
                )}
              </form>
            )}
          </LayoutContent>
        }
        footer={
          <LayoutFooter hasDivider>
            {created !== null ? (
              <Button
                size="sm"
                label={t('settings.credentials.agent.secretDone')}
                onClick={() => setDialogOpen(false)}
              />
            ) : (
              <>
                <Button
                  variant="secondary"
                  size="sm"
                  label={t('common.cancel')}
                  isDisabled={pending}
                  onClick={() => setDialogOpen(false)}
                />
                <Button
                  size="sm"
                  label={t('settings.credentials.agent.createSubmit')}
                  isLoading={pending}
                  isDisabled={!valid}
                  onClick={() => void submit()}
                />
              </>
            )}
          </LayoutFooter>
        }
      />
    </Dialog>
  )
}
