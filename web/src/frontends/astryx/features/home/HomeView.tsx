import * as stylex from '@stylexjs/stylex'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import { Button, EmptyState, Skeleton } from '@astryxdesign/core'
import { TriangleAlert } from 'lucide-react'
import { useEffect, useMemo, useState, useSyncExternalStore } from 'react'

import { healthQueryOptions } from '@shared/control/resources/health'
import {
  homeBaseQueryOptions,
  homeSubscriptionAccountsQueryOptions,
} from '@shared/control/resources/home'
import { systemUpdateQueryOptions } from '@shared/control/resources/system-update'
import type { GatewayClientID } from '@shared/domain/home/gateway-clients'
import { pagePath } from '@shared/routing/page-routes'
import type { SharedRouteQuery } from '@shared/routing/route-query'
import {
  isCanonicalHomeRouteQuery,
  parseHomeRouteQuery,
  serializeHomeRouteQuery,
  type HomeRouteState,
} from '@shared/routing/home-route'

import { useStableLoading } from '../../app/collection-loading'
import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'
import { useHomeStatistics } from '../../app/use-home-statistics'
import { CurrentAccessKeyCard } from './CurrentAccessKeyCard'
import { GatewayConnection } from './GatewayConnection'
import { HomeAttention } from './HomeAttention'
import { HomeSpend } from './HomeSpend'
import { HomeSubscriptionAccounts } from './HomeSubscriptionAccounts'
import { HomeSummary } from './HomeSummary'
import { HomeWelcome } from './HomeWelcome'

const narrow = '@media (max-width: 860px)'

const styles = stylex.create({
  page: {
    width: '100%',
    paddingTop: 'var(--stage-padding-top)',
    paddingBottom: 'var(--stage-padding-bottom)',
    paddingInline: 'var(--stage-padding-inline)',
  },
  pageInner: {
    width: 'min(100%, 1240px)',
    marginInline: 'auto',
  },
  sheet: {
    position: 'relative',
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-5)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: {
      default: 'var(--radius-sheet)',
      [narrow]: '9px',
    },
    backgroundColor: 'var(--color-surface)',
    boxShadow: 'var(--shadow-sheet)',
    paddingTop: 'var(--sheet-padding-top)',
    paddingBottom: 'var(--sheet-padding-bottom)',
    paddingInline: 'var(--sheet-padding-inline)',
  },
  sheetWelcome: {
    minHeight: {
      default: 560,
      [narrow]: 0,
    },
  },
  refreshing: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
  error: {
    display: 'grid',
    minHeight: 420,
    gap: 'var(--space-5)',
  },
  title: {
    margin: 0,
    fontFamily: 'var(--font-serif)',
    fontSize: 'var(--title-panel)',
    fontWeight: 500,
  },
  skeletonBlock: {
    display: 'grid',
    gap: 'var(--space-4)',
  },
  banner: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-3)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderRadius: 'var(--radius-tag)',
    paddingTop: 'var(--space-2-5)',
    paddingBottom: 'var(--space-2-5)',
    paddingInline: 'var(--space-3)',
    fontSize: 'var(--text-sm)',
  },
  bannerWarning: {
    borderColor:
      'color-mix(in srgb, var(--color-warning) 36%, var(--color-border-subtle))',
    backgroundColor: 'var(--color-warning-bg)',
    color: 'var(--color-warning)',
  },
})

export function HomeView() {
  const t = useT()
  const navigate = useNavigate()
  const { apiClient, authSession } = useAppServices()
  const sessionState = useSyncExternalStore(authSession.subscribe, authSession.getState)
  const isAccessKey = sessionState.principalType === 'access_key'
  const isAdmin = sessionState.principalType === 'admin'

  const { rawSearch, searchStr } = useRouterState({
    select: (state) => ({
      rawSearch: state.location.search as SharedRouteQuery,
      searchStr: state.location.searchStr,
    }),
  })
  const routeState = useMemo<HomeRouteState>(() => {
    const state = parseHomeRouteQuery(rawSearch)
    return isAccessKey ? { ...state, accessKeyID: undefined } : state
  }, [rawSearch, isAccessKey])
  const homePath = pagePath('home')

  const baseQuery = useQuery(homeBaseQueryOptions(apiClient))
  // 订阅账号含完整管理身份与额度,仅管理员发起;模板侧再 gate 一次防会话切换瞬间
  // 复用旧查询缓存。
  const subscriptionAccountsQuery = useQuery(
    homeSubscriptionAccountsQueryOptions(apiClient, isAdmin),
  )
  // 更新检查与首页数据解耦,仅管理员进首页时按需触发一次。
  const updateQuery = useQuery(systemUpdateQueryOptions(apiClient, isAdmin))
  // /api/health 不在 AccessKey 白名单里,必须前端主动 gate。
  const healthQuery = useQuery(healthQueryOptions(apiClient, undefined, !isAccessKey))

  // 花费固定看近 30 天:首页不提供时间旋钮。
  const statistics = useHomeStatistics({ initialRange: '30d' })

  const [nowMS, setNowMS] = useState(() => Date.now())

  // Server clock offset: server_now_ms stamped at response time vs the
  // reactive dataUpdatedAt receipt stamp — pure render math, equivalent to
  // the classic watch on baseQuery.data.
  const base = baseQuery.data
  const clockOffsetMS = base ? base.server_now_ms - baseQuery.dataUpdatedAt : 0

  useEffect(() => {
    const timer = window.setInterval(() => setNowMS(Date.now()), 60_000)
    return () => window.clearInterval(timer)
  }, [])

  const uptimeNowMS = nowMS + clockOffsetMS
  const releaseUpdate = updateQuery.data?.update ?? null
  const statisticsSnapshot =
    statistics.snapshot.state.kind === 'initial' ? null : statistics.snapshot.state.snapshot
  const statisticsLoading = statistics.snapshot.state.kind === 'initial'
  const baseLoading = useStableLoading(baseQuery.isPending)
  const statisticsInitialLoading = useStableLoading(statisticsLoading)
  const baseRefreshing = base !== undefined && baseQuery.isFetching
  const homeRefreshing = baseRefreshing || statistics.refreshing
  const isEmpty =
    base?.inventory !== undefined &&
    base.inventory.group_count === 0 &&
    base.inventory.credential_count === 0
  const accessKeys = base?.access_keys ?? []
  const selectedAccessKeyID =
    accessKeys.find(({ id }) => id === routeState.accessKeyID)?.id ??
    accessKeys[0]?.id ??
    null

  // Canonicalize non-canonical query params (junk, duplicated keys, raw ints).
  useEffect(() => {
    if (!isCanonicalHomeRouteQuery(rawSearch, routeState)) {
      void navigate({
        to: homePath,
        search: serializeHomeRouteQuery(routeState),
        replace: true,
        resetScroll: false,
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the URL string
  }, [searchStr])

  // A requested access key that isn't in the loaded list drops from the URL —
  // the selector falls back to the first available key.
  const requestedAccessKeyID = routeState.accessKeyID
  useEffect(() => {
    if (!base || requestedAccessKeyID === undefined) return
    if (accessKeys.some(({ id }) => id === requestedAccessKeyID)) return
    void navigate({
      to: homePath,
      search: serializeHomeRouteQuery({ ...routeState, accessKeyID: undefined }),
      replace: true,
      resetScroll: false,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the loaded key list
  }, [base, requestedAccessKeyID])

  function navigateHome(patch: Partial<HomeRouteState>, replace = false): void {
    const next = { ...routeState, ...patch }
    void navigate({
      to: homePath,
      search: serializeHomeRouteQuery(next),
      replace,
      resetScroll: false,
    })
  }

  const welcomeMode = Boolean(isEmpty && !isAccessKey && base)

  return (
    <section {...stylex.props(styles.page)} aria-labelledby="home-title">
      <div {...stylex.props(styles.pageInner)}>
        <div
          {...stylex.props(styles.sheet, welcomeMode && styles.sheetWelcome)}
          aria-busy={baseRefreshing || undefined}
        >
          <span aria-live="polite" {...stylex.props(styles.refreshing)}>
            {homeRefreshing ? t('home.ledger.loading') : ''}
          </span>

          {baseQuery.isPending || baseLoading ? (
            <div
              {...stylex.props(styles.skeletonBlock)}
              role="status"
              aria-label={t('home.ledger.loading')}
            >
              <Skeleton height={40} radius={2} />
              <Skeleton height={16} radius={2} />
              <Skeleton height={120} radius={2} />
              <Skeleton height={200} radius={2} />
            </div>
          ) : baseQuery.isError && !base ? (
            <section {...stylex.props(styles.error)} aria-labelledby="home-title">
              <h1 id="home-title" {...stylex.props(styles.title)}>
                {t('home.ledger.title')}
              </h1>
              <div role="alert">
                <EmptyState
                  title={t('home.ledger.baseError')}
                  icon={<TriangleAlert size={20} />}
                  actions={
                    <Button
                      variant="secondary"
                      size="sm"
                      label={t('common.retry')}
                      onClick={() => void baseQuery.refetch()}
                    />
                  }
                />
              </div>
            </section>
          ) : welcomeMode && base ? (
            <HomeWelcome base={base} update={releaseUpdate} />
          ) : base ? (
            <>
              {baseQuery.isError && (
                <div
                  {...stylex.props(styles.banner, styles.bannerWarning)}
                  role="status"
                >
                  <TriangleAlert size={13} aria-hidden />
                  <span>{t('home.ledger.baseError')}</span>
                  <Button
                    variant="secondary"
                    size="sm"
                    label={t('common.retry')}
                    onClick={() => void baseQuery.refetch()}
                  />
                </div>
              )}
              <HomeSummary
                base={base}
                update={releaseUpdate}
                observedAtMs={statistics.snapshot.lastSuccessfulObservedAtMS}
                uptimeNowMs={uptimeNowMS}
              />

              {/* 紧贴事实行:它是「X/Y 个凭据可用」的注解。 */}
              {!isAccessKey && <HomeAttention health={healthQuery.data ?? null} />}

              {isAdmin &&
                Boolean(subscriptionAccountsQuery.data?.items.length) &&
                subscriptionAccountsQuery.data && (
                  <HomeSubscriptionAccounts accounts={subscriptionAccountsQuery.data} />
                )}

              {base.current_access_key && (
                <CurrentAccessKeyCard accessKey={base.current_access_key} />
              )}

              <GatewayConnection
                accessKeys={base.access_keys}
                selectedAccessKeyID={selectedAccessKeyID}
                clientID={routeState.client}
                credential={isAccessKey ? authSession.getAuthKey() : undefined}
                selfScoped={isAccessKey}
                onAccessKeyChange={(id) => {
                  if (isAccessKey) return
                  navigateHome({ accessKeyID: id })
                }}
                onClientChange={(id: GatewayClientID) => navigateHome({ client: id })}
              />

              <HomeSpend
                snapshot={statisticsSnapshot}
                loading={statisticsLoading || statisticsInitialLoading}
              />
            </>
          ) : null}
        </div>
      </div>
    </section>
  )
}
