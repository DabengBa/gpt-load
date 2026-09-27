import * as stylex from '@stylexjs/stylex'
import { Banner, Button, EmptyState, Skeleton } from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import { TriangleAlert } from 'lucide-react'
import { useEffect, useMemo, useState, useImperativeHandle, type Ref } from 'react'
import { useIntl } from 'react-intl'
import { useNavigate } from '@tanstack/react-router'

import {
  healthQueryOptions,
  type HealthProblemCredentialDto,
} from '@shared/control/resources/health'
import { formatLocalInstant } from '@shared/lib/format'
import { pagePath } from '@shared/routing/page-routes'
import { healthMonitorQuery } from '@shared/routing/monitor-route'

import { useAppServices } from '../../app/services'
import { useStableLoading } from '../../app/collection-loading'
import { useT } from '../../app/i18n'
import { AccessKeyCostLimitHealth } from './AccessKeyCostLimitHealth'
import { GroupHealthCollection } from './GroupHealthCollection'
import {
  HealthProblemCollection,
  type HealthProblemItem,
  type RecoveryDisplay,
} from './HealthProblemCollection'
import { HealthSummaryStrip } from './HealthSummaryStrip'
import { RequestLogHealthCard } from './RequestLogHealthCard'

const styles = stylex.create({
  root: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-6)',
  },
  focusGrid: {
    display: 'grid',
    minWidth: 0,
    gridTemplateColumns: {
      default: 'minmax(0, 2fr) minmax(320px, 0.82fr)',
      '@media (max-width: 1099px)': 'minmax(0, 1fr)',
    },
    alignItems: 'start',
    gap: '20px',
  },
  focusContent: {
    minWidth: 0,
  },
  skeleton: {
    display: 'grid',
    gap: 'var(--space-2)',
  },
  error: {
    paddingBlock: 'var(--space-6)',
  },
  refreshing: {
    minHeight: 'var(--space-4)',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
})

export interface HealthTabHandle {
  refresh(): Promise<void>
}

interface HealthTabProps {
  handleRef?: Ref<HealthTabHandle>
  groupsExpanded: boolean
}

export function HealthTab({ handleRef, groupsExpanded }: HealthTabProps) {
  const t = useT()
  const intl = useIntl()
  const navigate = useNavigate()
  const { apiClient } = useAppServices()

  const healthQuery = useQuery(healthQueryOptions(apiClient))
  const data = healthQuery.data
  const initialLoading = useStableLoading(healthQuery.isPending)
  const healthRefreshing = data !== undefined && healthQuery.isFetching
  const hasStaleData = healthQuery.isError && data !== undefined

  // Visibility-aware elapsed clock: a 1s interval advances the counter while
  // the tab is visible, and a fresh snapshot re-anchors it to zero via render
  // adjustment — matching the classic performance.now semantics.
  const [isVisible, setIsVisible] = useState(() => document.visibilityState !== 'hidden')
  const [elapsed, setElapsed] = useState({ at: 0, ms: 0 })
  if (healthQuery.dataUpdatedAt !== elapsed.at) {
    setElapsed({ at: healthQuery.dataUpdatedAt, ms: 0 })
  }
  const timerShouldRun = isVisible && data !== undefined && !healthQuery.isError
  useEffect(() => {
    const onVisibility = () => setIsVisible(document.visibilityState !== 'hidden')
    document.addEventListener('visibilitychange', onVisibility)
    return () => document.removeEventListener('visibilitychange', onVisibility)
  }, [])
  useEffect(() => {
    if (!timerShouldRun) return
    const id = setInterval(
      () => setElapsed((prev) => ({ at: prev.at, ms: prev.ms + 1_000 })),
      1_000,
    )
    return () => clearInterval(id)
  }, [timerShouldRun, healthQuery.dataUpdatedAt])
  const elapsedMs = data === undefined || elapsed.at !== healthQuery.dataUpdatedAt ? 0 : elapsed.ms

  const monitorPath = pagePath('monitor')

  const problemItems = useMemo<HealthProblemItem[]>(() => {
    if (!data) return []
    const groupCounts = new Map(data.groups.map((group) => [group.id, group.counts]))
    const items: HealthProblemItem[] = [
      ...data.cooldown_credentials.map((credential) => ({
        credential,
        kind: 'cooldown' as const,
        tone: 'warning' as const,
      })),
      ...data.blacklisted_credentials.map((credential) => ({
        credential,
        kind: 'blacklisted' as const,
        tone: 'danger' as const,
      })),
    ]
    return items.sort((left, right) => {
      const leftUnavailable = groupCounts.get(left.credential.group_id)?.available === 0 ? 0 : 1
      const rightUnavailable = groupCounts.get(right.credential.group_id)?.available === 0 ? 0 : 1
      const leftKind = left.kind === 'blacklisted' ? 0 : 1
      const rightKind = right.kind === 'blacklisted' ? 0 : 1
      const leftRecovery = left.credential.recovery.at_ms ?? Number.MAX_SAFE_INTEGER
      const rightRecovery = right.credential.recovery.at_ms ?? Number.MAX_SAFE_INTEGER
      return (
        leftUnavailable - rightUnavailable ||
        leftKind - rightKind ||
        leftRecovery - rightRecovery ||
        left.credential.group_name.localeCompare(right.credential.group_name) ||
        left.credential.credential_id - right.credential.credential_id
      )
    })
  }, [data])

  const focusContentHeight = useMemo(() => {
    const rowHeight = 76
    const headerHeight = 38
    const compactMinimum = 196
    const visibleRows = Math.min(problemItems.length, 3)
    return Math.max(compactMinimum, headerHeight + visibleRows * rowHeight)
  }, [problemItems.length])

  function remainingLabel(totalSeconds: number): string {
    if (totalSeconds >= 3_600) {
      return t('monitor.health.recovery.hoursMinutes', {
        hours: Math.floor(totalSeconds / 3_600),
        minutes: Math.floor((totalSeconds % 3_600) / 60),
      })
    }
    if (totalSeconds >= 60) {
      return t('monitor.health.recovery.minutesSeconds', {
        minutes: Math.floor(totalSeconds / 60),
        seconds: totalSeconds % 60,
      })
    }
    return t('monitor.health.recovery.seconds', { seconds: totalSeconds })
  }

  const recoveryByCredential = useMemo<Record<number, RecoveryDisplay | undefined>>(() => {
    const observedAtMS = data?.observed_at_ms
    const display = (credential: HealthProblemCredentialDto): RecoveryDisplay | undefined => {
      const recoveryAtMS = credential.recovery.at_ms
      if (recoveryAtMS === null || observedAtMS === undefined) return undefined
      const remainingSeconds = Math.max(
        0,
        Math.ceil((recoveryAtMS - (observedAtMS + elapsedMs)) / 1_000),
      )
      const scheduled = credential.recovery.mode === 'scheduled_release'
      return {
        relative: remainingLabel(remainingSeconds),
        exact: t('monitor.health.recovery.exact', {
          time: formatLocalInstant(recoveryAtMS, intl.locale),
        }),
        labelKey: scheduled ? 'scheduledReleaseRecovery' : 'cooldownRecovery',
        hintKey: scheduled ? 'scheduledReleaseHint' : 'cooldownHint',
      }
    }
    return Object.fromEntries(
      problemItems.map((item) => [item.credential.credential_id, display(item.credential)]),
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps -- recovery display re-derives on the 1s tick
  }, [data, problemItems, elapsedMs, intl.locale])

  const earliestCooldown = useMemo<RecoveryDisplay | null>(() => {
    const earliest = problemItems
      .filter((item) => item.kind === 'cooldown' && item.credential.cooldown_until_ms !== null)
      .sort(
        (left, right) =>
          (left.credential.cooldown_until_ms ?? Number.MAX_SAFE_INTEGER) -
          (right.credential.cooldown_until_ms ?? Number.MAX_SAFE_INTEGER),
      )[0]
    return earliest ? (recoveryByCredential[earliest.credential.credential_id] ?? null) : null
  }, [problemItems, recoveryByCredential])

  useImperativeHandle(handleRef, () => ({
    refresh: async () => {
      await healthQuery.refetch()
    },
  }))

  function toggleGroups(): void {
    void navigate({
      to: monitorPath,
      search: healthMonitorQuery({ groupsExpanded: !groupsExpanded }),
      resetScroll: false,
    })
  }

  return (
    <div {...stylex.props(styles.root)} aria-busy={healthQuery.isFetching || undefined}>
      <span aria-live="polite" {...stylex.props(styles.refreshing)}>
        {healthRefreshing ? t('monitor.health.loading') : ''}
      </span>

      {healthQuery.isPending || initialLoading ? (
        <div {...stylex.props(styles.skeleton)} aria-label={t('monitor.health.loading')}>
          <Skeleton height={40} radius={2} />
          <Skeleton height={160} radius={2} />
          <Skeleton height={200} radius={2} />
        </div>
      ) : healthQuery.isError && !data ? (
        <div {...stylex.props(styles.error)} role="alert">
          <EmptyState
            title={t('monitor.health.loadFailed')}
            icon={<TriangleAlert size={20} />}
            actions={
              <Button
                variant="secondary"
                size="sm"
                label={t('common.retry')}
                onClick={() => void healthQuery.refetch()}
              />
            }
          />
        </div>
      ) : data ? (
        <>
          {hasStaleData && (
            <Banner
              status="warning"
              title={t('monitor.health.stale')}
              endContent={
                <Button
                  variant="secondary"
                  size="sm"
                  label={t('common.retry')}
                  onClick={() => void healthQuery.refetch()}
                />
              }
            />
          )}

          <HealthSummaryStrip counts={data.counts} earliestCooldown={earliestCooldown} />

          <div
            {...stylex.props(styles.focusGrid)}
            style={
              { '--health-focus-content-height': `${focusContentHeight}px` } as React.CSSProperties
            }
          >
            <div {...stylex.props(styles.focusContent)}>
              <HealthProblemCollection
                items={problemItems}
                recoveryByCredential={recoveryByCredential}
                statsWindowSeconds={data.stats_window_seconds}
                availableCount={data.counts.available}
              />
            </div>
            <RequestLogHealthCard stats={data.request_log} />
          </div>

          {data.blocked_access_keys.length > 0 && (
            <AccessKeyCostLimitHealth accessKeys={data.blocked_access_keys} />
          )}

          <GroupHealthCollection
            groups={data.groups}
            expanded={groupsExpanded}
            onToggle={toggleGroups}
          />
        </>
      ) : null}
    </div>
  )
}
