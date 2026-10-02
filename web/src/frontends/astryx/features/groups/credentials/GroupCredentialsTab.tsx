import { AlertDialog, EmptyState, Skeleton, Button } from '@astryxdesign/core'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from '@tanstack/react-router'
import { Plus } from 'lucide-react'
import * as actions from '@shared/control/resources/credentials'
import { channelsQueryOptions } from '@shared/control/resources/channels'

import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import type { CredentialItemDto, CredentialTestResultDto } from '@shared/control/types'
import { createUUID } from '@shared/lib/uuid'
import { ApiError } from '@shared/http/errors'
import type { MessageId } from '@shared/i18n/message-ids'
import { pagePath } from '@shared/routing/page-routes'
import { useT } from '../../../app/i18n'
import { useAppServices } from '../../../app/services'
import { InlineNotice } from '../../../components/InlineNotice'
import { GroupCredentialRecord } from './GroupCredentialRecord'
import { SubscriptionAccountCard } from './SubscriptionAccountCard'
import { CredentialTestDialog } from './CredentialTestDialog'

export function GroupCredentialsTab({
  groupId,
  channelId,
  connectionType,
}: {
  groupId: number
  channelId: string
  connectionType: 'api_key' | 'subscription'
}) {
  return (
    <CredentialDetail
      key={groupId}
      groupId={groupId}
      channelId={channelId}
      connectionType={connectionType}
    />
  )
}

function CredentialDetail({
  groupId,
  channelId,
  connectionType,
}: {
  groupId: number
  channelId: string
  connectionType: 'api_key' | 'subscription'
}) {
  const t = useT()
  const { apiClient } = useAppServices()
  const client = useQueryClient()
  const navigate = useNavigate()
  const query = useQuery({
    ...actions.credentialQueryOptions(apiClient, groupId),
    refetchInterval: (current) =>
      current.state.data?.credential?.auth_state === 'refreshing' ||
      current.state.data?.credential?.observation?.state === 'refreshing'
        ? 2000
        : false,
  })
  const channels = useQuery(channelsQueryOptions(apiClient, ''))
  const channel = channels.data?.items.find((entry) => entry.channel_id === channelId)
  const [pending, setPending] = useState('')
  const [error, setError] = useState('')
  const [deleteOpen, setDeleteOpen] = useState(false)
  const [resetOpen, setResetOpen] = useState(false)
  const resetOperation = useRef<{ credentialId: number; key: string } | undefined>(undefined)
  const [testOpen, setTestOpen] = useState(false)
  const [testResult, setTestResult] = useState<CredentialTestResultDto>()
  const [testFailed, setTestFailed] = useState(false)
  const busyRef = useRef(false)
  const mountedRef = useRef(true)
  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  async function reconcile() {
    await applyInvalidationPlan(client, mutationInvalidationPlans.group.importCredentials(groupId))
  }
  async function run(name: string, operation: () => Promise<unknown>) {
    if (busyRef.current) return
    busyRef.current = true
    setPending(name)
    setError('')
    try {
      await operation()
      try {
        await reconcile()
      } catch {
        if (mountedRef.current) setError(t('group.credentials.reconcileFailed'))
      }
    } catch {
      const failureKeys: Record<string, MessageId> = {
        delete: 'group.credentials.deleteFailed',
        restore: 'group.credentials.restoreFailed',
        test: 'group.credentials.test.requestFailed',
        observation: 'group.credentials.subscription.syncFailed',
        refresh: 'group.credentials.subscription.refreshCredentialFailed',
        download: 'group.credentials.subscription.downloadFailed',
        reset: 'group.credentials.subscription.consumeResetCreditFailed',
      }
      if (mountedRef.current) setError(t(failureKeys[name] ?? 'group.credentials.reconcileFailed'))
    } finally {
      busyRef.current = false
      if (mountedRef.current) setPending('')
    }
  }
  function configure() {
    void navigate({
      to: pagePath('import'),
      search: { mode: 'existing', group_id: String(groupId) },
    })
  }
  async function test(item: CredentialItemDto) {
    if (busyRef.current) return
    setTestOpen(true)
    setTestFailed(false)
    setTestResult(undefined)
    await run('test', async () => {
      try {
        setTestResult(
          await actions.testCredentialConnection(apiClient, groupId, item.credential_id),
        )
      } catch (cause) {
        setTestFailed(true)
        throw cause
      }
    })
  }
  async function download(item: CredentialItemDto) {
    await run('download', async () => {
      const result = await actions.downloadCredential(apiClient, groupId, item.credential_id)
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(result.credential, null, 2)], { type: 'application/json' }),
      )
      const link = document.createElement('a')
      link.href = url
      link.download = result.filename
      link.click()
      URL.revokeObjectURL(url)
    })
  }
  if (query.isPending) return <Skeleton />
  if (query.isError)
    return (
      <EmptyState
        title={t('group.credentials.loadFailed')}
        actions={<Button label={t('common.retry')} onClick={() => void query.refetch()} />}
      />
    )
  const item = query.data.credential
  if (!item)
    return (
      <EmptyState
        title={t('group.credentials.emptyTitle')}
        description={t('group.credentials.emptyDescription')}
        actions={
          <Button
            label={t('group.credentials.add')}
            icon={<Plus size={16} />}
            onClick={configure}
          />
        }
      />
    )
  const restore = () =>
    void run('restore', () => actions.restoreCredential(apiClient, groupId, item.credential_id))
  return (
    <>
      {error && <InlineNotice tone="danger">{error}</InlineNotice>}
      {connectionType === 'subscription' ? (
        <>
          <SubscriptionAccountCard
            item={item}
            busy={!!pending}
            refreshingObservation={pending === 'observation'}
            observationError={error}
            detailBusy={query.isFetching}
            detailLoaded
            detailError=""
            capabilities={
              channel?.capabilities ?? {
                model_discovery: false,
                quota_observation: false,
                credential_actions: [],
                outbound_proxy: false,
              }
            }
            channelIcon={channel?.icon}
            onRestore={restore}
            onRefresh={() =>
              void run('observation', () =>
                actions.refreshCredentialObservation(apiClient, groupId, item.credential_id),
              )
            }
            onLoadDetails={() => void query.refetch()}
            onReset={() => setResetOpen(true)}
            onDownload={() => void download(item)}
            onRefreshCredential={() =>
              void run('refresh', () =>
                actions.refreshCredential(apiClient, groupId, item.credential_id),
              )
            }
            onRemove={() => setDeleteOpen(true)}
          />
          <Button
            label={t('group.credentials.subscription.connect')}
            icon={<Plus size={16} />}
            isDisabled={!!pending}
            onClick={configure}
          />
        </>
      ) : (
        <GroupCredentialRecord
          item={item}
          groupId={groupId}
          busy={!!pending}
          onTest={() => void test(item)}
          onRestore={restore}
          onRemove={() => setDeleteOpen(true)}
        />
      )}
      <CredentialTestDialog
        open={testOpen}
        mask={item.mask}
        pending={pending === 'test'}
        requestFailed={testFailed}
        result={testResult}
        onOpenChange={setTestOpen}
        onViewLog={(id) => void navigate({ to: pagePath('logs'), search: { log_id: id } })}
      />
      <AlertDialog
        isOpen={deleteOpen}
        onOpenChange={setDeleteOpen}
        title={t('group.credentials.delete')}
        description={item.mask}
        actionLabel={t('group.credentials.delete')}
        isActionLoading={pending === 'delete'}
        onAction={() =>
          void run('delete', async () => {
            await actions.deleteCredential(apiClient, groupId, item.credential_id)
            setDeleteOpen(false)
          })
        }
      />
      <AlertDialog
        isOpen={resetOpen}
        onOpenChange={(open) => {
          if (!busyRef.current) setResetOpen(open)
        }}
        title={t('group.credentials.subscription.consumeResetCreditTitle')}
        description={t('group.credentials.subscription.consumeResetCreditDescription')}
        actionLabel={t('group.credentials.subscription.consumeResetCredit')}
        isActionLoading={pending === 'reset'}
        onAction={() =>
          void run('reset', async () => {
            if (resetOperation.current?.credentialId !== item.credential_id)
              resetOperation.current = { credentialId: item.credential_id, key: createUUID() }
            const operation = resetOperation.current
            try {
              await actions.consumeCredentialResetCredit(
                apiClient,
                groupId,
                item.credential_id,
                operation.key,
              )
              resetOperation.current = undefined
              setResetOpen(false)
            } catch (cause) {
              if (
                cause instanceof ApiError &&
                (cause.code === 'RESET_CREDIT_REJECTED' ||
                  cause.code === 'RESET_CREDIT_UNAVAILABLE' ||
                  (cause.status >= 400 &&
                    cause.status < 500 &&
                    ![
                      'RESET_CREDIT_OUTCOME_UNKNOWN',
                      'CONTROL_OPERATION_INCOMPLETE',
                      'CONTROL_RECOVERY_PENDING',
                    ].includes(cause.code)))
              )
                resetOperation.current = undefined
              throw cause
            }
          })
        }
      />
    </>
  )
}
