import { useNavigate } from '@tanstack/react-router'
import { useState } from 'react'

import type { GroupCollectionItemDto } from '@shared/control/types'
import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import {
  cacheGroupSettings,
  copyGroup,
  invalidateGroupSettingsDependents,
  updateGroupSettings,
} from '@shared/control/resources/groups'
import { createUUID } from '@shared/lib/uuid'

import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'
import { groupDetailHref } from './group-detail-href'

/**
 * Row-scoped mutations for the collection (enable toggle, copy). Optimistic
 * enable state lives here so the table can render the intended value before
 * the server round trip lands, then falls back to server truth.
 */
export function useGroupRowMutations() {
  const t = useT()
  const navigate = useNavigate()
  const { apiClient, queryClient, toast } = useAppServices()
  const [togglingGroupIDs, setTogglingGroupIDs] = useState<ReadonlySet<number>>(new Set())
  const [copyingGroupIDs, setCopyingGroupIDs] = useState<ReadonlySet<number>>(new Set())
  const [optimisticEnabled, setOptimisticEnabled] = useState<ReadonlyMap<number, boolean>>(
    new Map(),
  )

  async function toggleGroupEnabled(group: GroupCollectionItemDto, next: boolean): Promise<void> {
    if (togglingGroupIDs.has(group.id)) return
    setOptimisticEnabled((prev) => new Map(prev).set(group.id, next))
    setTogglingGroupIDs((prev) => new Set(prev).add(group.id))
    try {
      const settings = await updateGroupSettings(apiClient, group.id, { enabled: next })
      cacheGroupSettings(queryClient, group.id, settings)
      await invalidateGroupSettingsDependents(queryClient, group.id)
      toast.show({
        message: t(next ? 'groups.collection.enabledOn' : 'groups.collection.enabledOff', {
          name: group.name,
        }),
        tone: next ? 'success' : 'warning',
      })
    } catch {
      toast.show({ message: t('groups.collection.toggleFailed'), tone: 'danger' })
    } finally {
      setOptimisticEnabled((prev) => {
        const nextMap = new Map(prev)
        nextMap.delete(group.id)
        return nextMap
      })
      setTogglingGroupIDs((prev) => {
        const nextSet = new Set(prev)
        nextSet.delete(group.id)
        return nextSet
      })
    }
  }

  async function copyGroupRecord(group: GroupCollectionItemDto): Promise<void> {
    if (copyingGroupIDs.has(group.id)) return
    setCopyingGroupIDs((prev) => new Set(prev).add(group.id))
    try {
      const result = await copyGroup(apiClient, group.id, createUUID())
      await applyInvalidationPlan(queryClient, mutationInvalidationPlans.group.create)
      toast.show({
        message: t('groups.collection.copySucceeded', { name: result.group_name }),
        tone: 'success',
      })
      // Group detail is astryx-owned — stay in the SPA.
      void navigate({ to: groupDetailHref(result.group_id) })
    } catch {
      toast.show({ message: t('groups.collection.copyFailed'), tone: 'danger' })
    } finally {
      setCopyingGroupIDs((prev) => {
        const nextSet = new Set(prev)
        nextSet.delete(group.id)
        return nextSet
      })
    }
  }

  return {
    togglingGroupIDs,
    copyingGroupIDs,
    optimisticEnabled,
    toggleGroupEnabled,
    copyGroupRecord,
  }
}
