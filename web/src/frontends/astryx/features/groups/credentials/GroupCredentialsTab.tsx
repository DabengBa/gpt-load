import { EmptyState, Skeleton, Button } from '@astryxdesign/core'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from '@tanstack/react-router'
import { Plus } from 'lucide-react'
import * as stylex from '@stylexjs/stylex'
import * as actions from '@shared/control/resources/credentials'

import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import type { MessageId } from '@shared/i18n/message-ids'
import { pagePath } from '@shared/routing/page-routes'
import { useT } from '../../../app/i18n'
import { useAppServices } from '../../../app/services'
import { InlineNotice } from '../../../components/InlineNotice'
import { GroupCredentialRecord } from './GroupCredentialRecord'

const styles = stylex.create({
  empty: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
    paddingTop: 'var(--space-3)',
    paddingBottom: 'var(--space-3)',
  },
  emptyCopy: { minWidth: 0 },
  emptyTitle: { margin: 0, fontSize: 'var(--text-body)', fontWeight: 650 },
  emptyDescription: { margin: 0, color: 'var(--color-text-muted)', fontSize: 'var(--text-sm)' },
})

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
        restore: 'group.credentials.restoreFailed',
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
      <section {...stylex.props(styles.empty)}>
        <div {...stylex.props(styles.emptyCopy)}>
          <h2 {...stylex.props(styles.emptyTitle)}>{t('group.credentials.emptyTitle')}</h2>
          <p {...stylex.props(styles.emptyDescription)}>
            {t('group.credentials.emptyDescription')}
          </p>
        </div>
        <Button label={t('group.credentials.add')} icon={<Plus size={16} />} onClick={configure} />
      </section>
    )
  const restore = () =>
    void run('restore', () => actions.restoreCredential(apiClient, groupId, item.credential_id))
  return (
    <>
      {error && <InlineNotice tone="danger">{error}</InlineNotice>}
      <GroupCredentialRecord item={item} groupId={groupId} busy={!!pending} onRestore={restore} />
    </>
  )
}
