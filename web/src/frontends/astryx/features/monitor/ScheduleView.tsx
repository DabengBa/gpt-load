import * as stylex from '@stylexjs/stylex'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import { useEffect, useMemo, useRef, useSyncExternalStore } from 'react'
import { useIntl } from 'react-intl'

import { pagePath } from '@shared/routing/page-routes'
import {
  parseScheduleMonitorState,
  scopeAccessKeyScheduleMonitorState,
  sameMonitorQuery,
  scheduleMonitorQuery,
  type ScheduleDrafts,
  type ScheduleMonitorState,
} from '@shared/routing/monitor-route'
import type { SharedRouteQuery } from '@shared/routing/route-query'
import type { MessageId } from '@shared/i18n/message-ids'

import { useAppServices } from '../../app/services'
import { useT } from '../../app/i18n'

import { SchedulePanel, type SchedulePanelLabels } from './SchedulePanel'
import { ScheduleAccessReadOnly } from './ScheduleAccessReadOnly'
import { ModelUpstreamDrawer, type ModelUpstreamDrawerHandle } from '../models/ModelUpstreamDrawer'

const styles = stylex.create({
  page: {
    display: 'grid',
    minWidth: 0,
    padding: {
      default: '12px 16px 16px',
      '@media (max-width: 800px)': '12px',
    },
  },
  sheet: {
    display: 'grid',
    width: '100%',
    marginInline: 'auto',
    minWidth: 0,
    alignContent: 'start',
    gap: 0,
  },
  title: {
    margin: 0,
    fontSize: '20px',
    fontWeight: 650,
    lineHeight: '28px',
  },
  panel: {
    minWidth: 0,
    paddingTop: '8px',
  },
})

const scheduleReasonCodes = [
  'access_key_disabled',
  'access_key_expired',
  'protocol_filtered',
  'model_filtered',
  'model_required_by_filter',
  'operation_unsupported',
  'native_route_required',
  'no_route_target',
  'group_disabled',
  'group_filtered',
  'no_available_group',
  'no_credentials',
  'credential_blacklisted',
  'credential_cooldown',
  'credential_auth_unavailable',
  'credential_not_allowed',
  'no_available_credential',
  'entry_disabled',
  'entry_blacklisted',
  'entry_cooldown',
  'entry_weight_zero',
  'tier_demoted',
] as const

export function ScheduleView() {
  const t = useT()
  const intl = useIntl()
  const navigate = useNavigate()
  const { authSession } = useAppServices()
  const sessionState = useSyncExternalStore(authSession.subscribe, authSession.getState)
  const isAdmin = sessionState.principalType === 'admin'

  const { rawSearch, searchStr, pathname } = useRouterState({
    select: (state) => ({
      rawSearch: state.location.search as SharedRouteQuery,
      searchStr: state.location.searchStr,
      pathname: state.location.pathname,
    }),
  })
  const schedulePath = pagePath('schedule')

  const scheduleState = useMemo(() => {
    const state = parseScheduleMonitorState(rawSearch)
    return isAdmin ? state : scopeAccessKeyScheduleMonitorState(state)
  }, [rawSearch, isAdmin])
  const canonicalQuery = useMemo(() => scheduleMonitorQuery(scheduleState), [scheduleState])
  const isCanonicalQuery = sameMonitorQuery(rawSearch, canonicalQuery)

  // Classic deep watch on route.query (immediate): canonicalize by replace.
  useEffect(() => {
    // Pending transition: the outgoing route still renders while location has
    // moved — a late canonicalization must not resurrect this page.
    if (pathname !== schedulePath) return
    if (!isCanonicalQuery) {
      void navigate({
        to: schedulePath,
        search: canonicalQuery,
        replace: true,
        resetScroll: false,
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the URL string
  }, [searchStr])

  // Classic pendingScheduleState: consecutive draft/row replaces must merge
  // onto the in-flight target, not the last committed URL. Kept in a ref —
  // rendering always follows the committed location.
  const pendingStateRef = useRef<ScheduleMonitorState | null>(null)
  const pendingGenerationRef = useRef(0)
  const drawerRef = useRef<ModelUpstreamDrawerHandle>(null)
  const latestStateRef = useRef(scheduleState)
  latestStateRef.current = scheduleState
  const latestPathRef = useRef(pathname)
  latestPathRef.current = pathname

  function withDrawerBypass(navigateState: () => Promise<void>): Promise<void> {
    return drawerRef.current?.runWithoutPrompt(navigateState) ?? navigateState()
  }

  async function openPrice(priceID: number): Promise<void> {
    const initial = pendingStateRef.current ?? latestStateRef.current
    if (priceID === initial.selectedPriceID) return
    if (drawerRef.current && !(await drawerRef.current.confirmDiscardSwitch())) return
    const current = pendingStateRef.current ?? latestStateRef.current
    if (latestPathRef.current !== schedulePath || current.externalModel !== initial.externalModel)
      return
    drawerRef.current?.discardChanges()
    await withDrawerBypass(() =>
      navigateScheduleState({ ...current, selectedPriceID: priceID }, 'push'),
    )
  }

  async function navigateScheduleState(
    next: ScheduleMonitorState,
    method: 'push' | 'replace',
  ): Promise<void> {
    pendingGenerationRef.current += 1
    const generation = pendingGenerationRef.current
    pendingStateRef.current = next
    await navigate({
      to: schedulePath,
      search: scheduleMonitorQuery(next),
      replace: method === 'replace',
      resetScroll: false,
    })
    if (generation !== pendingGenerationRef.current) return
    pendingStateRef.current = null
  }

  function updateScheduleContext(
    next: Partial<ScheduleMonitorState>,
    method: 'push' | 'replace',
  ): void {
    const current = pendingStateRef.current ?? latestStateRef.current
    const contextChanged =
      next.externalModel !== undefined && next.externalModel !== current.externalModel
    const nextState: ScheduleMonitorState = contextChanged
      ? { ...current, ...next, selectedRow: undefined, sourceGroupId: undefined, drafts: {} }
      : { ...current, ...next }
    void navigateScheduleState(nextState, method)
  }

  // Classic commitScheduleContext: context commits push; draft/row churn stays
  // replace-only so it never pollutes history.
  async function commitScheduleContext(context: { externalModel?: string }): Promise<void> {
    const current = pendingStateRef.current ?? scheduleState
    const externalModel = context.externalModel?.trim() || undefined
    const contextChanged = externalModel !== current.externalModel
    if (
      contextChanged &&
      Object.keys(current.drafts).length > 0 &&
      !window.confirm(t('monitor.schedule.detail.unsaved'))
    )
      return
    if (contextChanged && drawerRef.current && !(await drawerRef.current.confirmDiscardSwitch()))
      return
    if (contextChanged) drawerRef.current?.discardChanges()
    const next: ScheduleMonitorState = contextChanged
      ? {
          ...current,
          externalModel,
          selectedRow: undefined,
          sourceGroupId: undefined,
          drafts: {},
          selectedPriceID: undefined,
        }
      : { ...current, externalModel }
    if (sameMonitorQuery(rawSearch, scheduleMonitorQuery(next))) return
    void navigateScheduleState(next, 'push')
  }

  const scheduleLabels = useMemo<SchedulePanelLabels>(
    () => ({
      model: t('monitor.schedule.panel.model'),
      selectModel: t('monitor.schedule.panel.selectModel'),
      noMatchingModels: t('monitor.schedule.panel.noMatchingModels'),
      noModels: t('monitor.schedule.panel.noModels'),
      importModels: t('shell.import'),
      loadingOptions: t('monitor.schedule.panel.loadingOptions'),
      contextRequired: t('monitor.schedule.panel.contextRequired'),
      kicker: t('monitor.schedule.panel.kicker'),
      contextReady: t('monitor.schedule.panel.contextReady'),
      context: t('monitor.schedule.panel.context'),
      retry: t('monitor.schedule.panel.retry'),
      indexFailed: t('monitor.schedule.panel.indexFailed'),
      detailFailed: t('monitor.schedule.panel.detailFailed'),
      detail: {
        title: t('monitor.schedule.detail.title'),
        loading: t('monitor.schedule.detail.loading'),
        refresh: t('monitor.schedule.detail.refresh'),
        stale: t('monitor.schedule.detail.stale'),
        observedAt: t('monitor.schedule.detail.observedAt'),
        routeUnavailable: t('monitor.schedule.detail.routeUnavailable'),
        group: t('monitor.schedule.detail.group'),
        upstreamModel: t('monitor.schedule.detail.upstreamModel'),
        weight: t('monitor.schedule.detail.weight'),
        priority: t('monitor.schedule.detail.priority'),
        share: t('monitor.schedule.detail.share'),
        status: t('monitor.schedule.detail.status'),
        available: t('monitor.schedule.detail.available'),
        cooldown: t('monitor.schedule.detail.cooldown'),
        blacklisted: t('monitor.schedule.detail.blacklisted'),
        failures: t('monitor.schedule.detail.failures'),
        recover: t('monitor.schedule.detail.recover'),
        breakerRecovery: t('monitor.schedule.detail.breakerRecovery'),
        breakerThreshold: t('monitor.schedule.detail.breakerThreshold'),
        breakerCooldown: t('monitor.schedule.detail.breakerCooldown'),
        invalidValue: t('monitor.schedule.detail.invalidValue'),
        derivedReadOnly: t('monitor.schedule.detail.derivedReadOnly'),
        save: t('monitor.schedule.detail.save'),
        discard: t('monitor.schedule.detail.discard'),
        unsaved: t('monitor.schedule.detail.unsaved'),
        saved: t('monitor.schedule.detail.saved'),
        saveFailed: t('monitor.schedule.detail.saveFailed'),
        conflict: t('monitor.schedule.detail.conflict'),
        refreshToResolve: t('monitor.schedule.detail.refreshToResolve'),
        recoverFailed: t('monitor.schedule.detail.recoverFailed'),
        noEntries: t('monitor.schedule.detail.noEntries'),
        unknownReason: t('monitor.schedule.detail.unknownReason'),
        draftPreview: t('monitor.schedule.detail.draftPreview'),
        reasonLabels: Object.fromEntries(
          scheduleReasonCodes.map((code) => [
            code,
            t(`monitor.schedule.reasons.${code}` as MessageId),
          ]),
        ),
      },
    }),
    [t],
  )

  return (
    <section {...stylex.props(styles.page)} aria-labelledby="schedule-title">
      <div {...stylex.props(styles.sheet)} data-testid="schedule-page">
        <h1 id="schedule-title" {...stylex.props(styles.title)}>
          {t('shell.schedule')}
        </h1>
        {isAdmin && isCanonicalQuery && (
          <div {...stylex.props(styles.panel)}>
            <SchedulePanel
              externalModel={scheduleState.externalModel}
              selectedRow={scheduleState.selectedRow}
              sourceGroupId={scheduleState.sourceGroupId}
              drafts={scheduleState.drafts}
              labels={scheduleLabels}
              locale={intl.locale}
              onChangeContext={commitScheduleContext}
              onDraftChange={(drafts: ScheduleDrafts) =>
                updateScheduleContext({ drafts }, 'replace')
              }
              onRowChange={(row) => updateScheduleContext({ selectedRow: row }, 'replace')}
              onSaved={(_, clearDrafts) =>
                updateScheduleContext(clearDrafts ? { drafts: {} } : {}, 'replace')
              }
              onRecovered={() => updateScheduleContext({}, 'replace')}
              onRefresh={() => updateScheduleContext({}, 'replace')}
              onOpenPrice={(priceID) => void openPrice(priceID)}
            />
          </div>
        )}
        {!isAdmin && isCanonicalQuery && (
          <ScheduleAccessReadOnly
            key={scheduleState.externalModel}
            externalModel={scheduleState.externalModel}
          />
        )}
        {isAdmin && isCanonicalQuery && (
          <ModelUpstreamDrawer
            ref={drawerRef}
            isOpen={scheduleState.selectedPriceID !== undefined}
            priceId={scheduleState.selectedPriceID ?? null}
            onClose={() => {
              drawerRef.current?.discardChanges()
              void withDrawerBypass(() =>
                navigateScheduleState({ ...scheduleState, selectedPriceID: undefined }, 'replace'),
              )
            }}
          />
        )}
      </div>
    </section>
  )
}
