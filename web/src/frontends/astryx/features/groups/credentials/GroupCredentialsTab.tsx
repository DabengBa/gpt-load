import { AlertDialog, EmptyState, Skeleton, Button } from '@astryxdesign/core'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from '@tanstack/react-router'
import { Plus } from 'lucide-react'
import * as actions from '@shared/control/resources/credentials'

import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import type { CredentialItemDto, CredentialTestResultDto } from '@shared/control/types'
import type { MessageId } from '@shared/i18n/message-ids'
import { pagePath } from '@shared/routing/page-routes'
import { useT } from '../../../app/i18n'
import { useAppServices } from '../../../app/services'
import { InlineNotice } from '../../../components/InlineNotice'
import { GroupCredentialRecord } from './GroupCredentialRecord'
import { CredentialTestDialog } from './CredentialTestDialog'

export function GroupCredentialsTab({ groupId }: { groupId: number }) {
  return <CredentialDetail key={groupId} groupId={groupId} />
}

function CredentialDetail({ groupId }: { groupId: number }) {
  const t = useT()
  const { apiClient } = useAppServices()
  const client = useQueryClient()
  const navigate = useNavigate()
  const query = useQuery(actions.credentialQueryOptions(apiClient, groupId))
  const [pending, setPending] = useState('')
  const [error, setError] = useState('')
  const [deleteOpen, setDeleteOpen] = useState(false)
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
      <GroupCredentialRecord
        item={item}
        groupId={groupId}
        busy={!!pending}
        onTest={() => void test(item)}
        onRestore={restore}
        onRemove={() => setDeleteOpen(true)}
      />
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
    </>
  )
}
