import { useMemo } from 'react'

import type { GroupCollectionStatus, GroupCollectionSummaryDto } from '@shared/control/types'

import { useT } from '../../app/i18n'

export interface GroupSummaryItem {
  value: GroupCollectionStatus | undefined
  label: string
  count: number
  tone: 'neutral' | 'success' | 'danger'
}

/**
 * Summary chips mirror the server's status breakdown; `undefined` is the
 * unfiltered total, which is why it doubles as the "all" chip.
 */
export function useGroupSummaryItems(
  summary: GroupCollectionSummaryDto | undefined,
): readonly GroupSummaryItem[] {
  const t = useT()
  return useMemo(() => {
    if (!summary) return []
    return [
      {
        value: undefined,
        label: t('groups.collection.status.all'),
        count: summary.total,
        tone: 'neutral' as const,
      },
      {
        value: 'available' as const,
        label: t('groups.collection.status.available'),
        count: summary.available,
        tone: 'success' as const,
      },
      {
        value: 'unavailable' as const,
        label: t('groups.collection.status.unavailable'),
        count: summary.unavailable,
        tone: 'danger' as const,
      },
      {
        value: 'disabled' as const,
        label: t('groups.collection.status.disabled'),
        count: summary.disabled,
        tone: 'neutral' as const,
      },
    ]
  }, [summary, t])
}
