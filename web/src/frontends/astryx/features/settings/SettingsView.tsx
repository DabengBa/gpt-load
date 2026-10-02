import {
  Button,
  Dialog,
  DialogHeader,
  Layout,
  LayoutContent,
  LayoutFooter,
  Skeleton,
} from '@astryxdesign/core'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import * as stylex from '@stylexjs/stylex'
import { TriangleAlert } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useIntl } from 'react-intl'

import { SectionNav } from '../../components/SectionNav'
import { StickySaveBar } from '../../components/StickySaveBar'
import { useT } from '../../app/i18n'
import { useStableLoading } from '../../app/collection-loading'
import { useAppServices } from '../../app/services'
import { useSectionNavigation } from '../../app/use-section-navigation'
import { useSettingsDraftController } from '../../app/use-settings-draft'
import { useTransientFlag } from '../../app/use-transient-flag'
import { useUnsavedChanges } from '../../app/use-unsaved-changes'
import { controlQueryKeys } from '@shared/control/query-keys'
import type { ProxyConfiguredMode, ProxyViewDto } from '@shared/control/types'
import { proxyDraftState } from '@shared/control/resources/proxy'
import {
  runtimeSettingKeys,
  settingsQueryOptions,
  type RuntimeSettingKey,
  type SettingsPatch,
  type SettingsResource,
} from '@shared/control/resources/settings'
import { pagePath } from '@shared/routing/page-routes'
import {
  isValidAffinityCapacity,
  isValidNonNegativeInteger,
  isValidRetention,
  isValidTimeout,
  type SettingsDraft,
} from '@shared/domain/settings/settings-patch'
import {
  isCanonicalSettingsRouteQuery,
  parseSettingsCredentialsRoute,
  parseSettingsRouteSection,
  serializeSettingsRouteQuery,
  type SettingsRouteSection,
} from '@shared/routing/settings-route'
import type { SharedRouteQuery } from '@shared/routing/route-query'
import { formatLocalInstant } from '@shared/lib/format'

import { BrowserAccessSection } from './BrowserAccessSection'
import { ConnectionSection } from './ConnectionSection'
import { CredentialsSection } from './CredentialsSection'
import { DataMaintenanceSection } from './DataMaintenanceSection'
import { ReliabilitySection } from './ReliabilitySection'
import { RoutingSection } from './RoutingSection'
import { SystemInfoSection } from './SystemInfoSection'

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
    borderRadius: 'var(--radius-sheet)',
    backgroundColor: 'var(--color-surface)',
    boxShadow: 'var(--shadow-sheet)',
    paddingTop: 'var(--sheet-padding-top)',
    paddingBottom: 'var(--sheet-padding-bottom)',
    paddingInline: 'var(--sheet-padding-inline)',
  },
  title: {
    margin: 0,
    fontSize: 'var(--text-heading-2-size, 20px)',
    fontWeight: 650,
  },
  refreshing: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
  layout: {
    display: 'grid',
    gridTemplateColumns: {
      default: '176px minmax(0, 1fr)',
      [narrow]: '1fr',
    },
    alignItems: 'start',
    gap: '34px',
  },
  content: {
    display: 'grid',
    minWidth: 0,
    gap: '28px',
  },
  contentSection: {
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: '17px',
  },
  validation: {
    display: 'grid',
    gap: 'var(--space-2)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-danger)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-danger-bg)',
    paddingBlock: 'var(--space-3)',
    paddingInline: 'var(--space-4)',
    fontSize: 'var(--text-sm)',
  },
  validationTitle: {
    color: 'var(--color-danger)',
    fontWeight: 650,
  },
  validationList: {
    display: 'grid',
    gap: 'var(--space-1)',
    margin: 0,
    paddingInlineStart: 'var(--space-5)',
  },
  validationLink: {
    color: 'var(--color-danger)',
    textDecoration: 'underline',
  },
  errorBox: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-danger)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-danger-bg)',
    color: 'var(--color-danger)',
    paddingBlock: 'var(--space-3)',
    paddingInline: 'var(--space-4)',
    fontSize: 'var(--text-sm)',
  },
  staleBanner: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    color: 'var(--color-warning)',
    fontSize: 'var(--text-sm)',
  },
  skeleton: {
    display: 'grid',
    gap: 'var(--space-3)',
  },
  saveBar: {
    marginInlineStart: { default: '210px', [narrow]: 0 },
  },
  statusStack: {
    display: 'grid',
    gap: '2px',
  },
  statusStrong: {
    color: 'var(--color-text)',
    fontSize: 'var(--text-sm)',
    fontWeight: 650,
  },
  statusDetail: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
    overflowWrap: 'anywhere',
  },
  discardList: {
    display: 'grid',
    gap: 'var(--space-1)',
    margin: 0,
    paddingInlineStart: 'var(--space-5)',
  },
})

const timeoutKeys = [
  'first_byte_timeout',
  'request_timeout',
  'stream_idle_timeout',
  'blacklist_release_seconds',
  'affinity_ttl',
] as const
const browserAccessKeys: ReadonlySet<RuntimeSettingKey> = new Set([
  'header_rules',
  'cors',
  'response_header_rules',
])

function sectionID(section: SettingsRouteSection): string {
  return `settings-${section}`
}

function sectionFromID(id: string): SettingsRouteSection | undefined {
  const section = id.replace(/^settings-/u, '')
  return section === 'routing' ||
    section === 'connection' ||
    section === 'reliability' ||
    section === 'browser-access' ||
    section === 'credentials' ||
    section === 'data-maintenance' ||
    section === 'system'
    ? section
    : undefined
}

export function SettingsView() {
  const t = useT()
  const intl = useIntl()
  const { apiClient } = useAppServices()
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const rawSearch = useRouterState({
    select: (state) => state.location.search as SharedRouteQuery,
  })
  const searchStr = useRouterState({ select: (state) => state.location.searchStr })
  const pathname = useRouterState({ select: (state) => state.location.pathname })
  // Dynamic route tree → `to` only accepts a plain `string` (GroupsView idiom).
  const settingsPath = pagePath('settings')

  const settingsQuery = useQuery(settingsQueryOptions(apiClient, intl.locale))
  const resource = settingsQuery.data ?? null
  const initialLoading = useStableLoading(
    settingsQuery.isPending && settingsQuery.data === undefined,
  )
  const settingsRefreshing = settingsQuery.data !== undefined && settingsQuery.isFetching

  const controller = useSettingsDraftController({
    client: apiClient,
    resource,
    locale: intl.locale,
  })
  const base = controller.getBase()
  const draft = controller.getDraft()
  const patch = controller.getPatch()
  const pending = controller.isPending()
  const failed = controller.hasFailed()
  const operationLocked = controller.isOperationLocked()
  const savedAt = controller.getSavedAt()

  // --- proxy override (local draft state, published into saveAll's `extra`) ---
  const [proxyMode, setProxyMode] = useState<ProxyConfiguredMode>('inherit')
  const [proxyEndpoint, setProxyEndpoint] = useState('')
  const [proxyBaseView, setProxyBaseView] = useState<ProxyViewDto>()
  const proxyConfig = base?.settings.values.proxy_config
  // Render-time adjustment (GroupsView idiom): a fresh resource snapshot resets
  // the local proxy draft to its configured mode.
  const [lastProxyConfig, setLastProxyConfig] = useState(proxyConfig)
  if (lastProxyConfig !== proxyConfig) {
    setLastProxyConfig(proxyConfig)
    if (proxyConfig !== undefined) {
      setProxyMode(proxyConfig.configured_mode)
      setProxyEndpoint('')
      setProxyBaseView(proxyConfig)
    }
  }
  const proxyState = proxyBaseView
    ? proxyDraftState(proxyBaseView, proxyMode, proxyEndpoint)
    : { dirty: false, invalid: false, value: undefined }

  // --- browser-access reported state ---
  const [browserAccessValid, setBrowserAccessValid] = useState(true)
  const [headerRulesValid, setHeaderRulesValid] = useState(true)
  const [corsValid, setCorsValid] = useState(true)
  const [responseHeaderRulesValid, setResponseHeaderRulesValid] = useState(true)
  const [headerRulesInvalidEdits, setHeaderRulesInvalidEdits] = useState(false)
  const [responseRulesInvalidEdits, setResponseRulesInvalidEdits] = useState(false)
  const [browserAccessEditorRevision, setBrowserAccessEditorRevision] = useState(0)

  const hasLocalEdits = headerRulesInvalidEdits || responseRulesInvalidEdits || proxyState.dirty
  useEffect(() => controller.setLocalEdits(hasLocalEdits), [controller, hasLocalEdits])

  const dirty =
    controller.isDirty() || headerRulesInvalidEdits || responseRulesInvalidEdits || proxyState.dirty
  const valid = controller.isValid() && browserAccessValid && !proxyState.invalid

  // Leaving a header-rules override clears its local invalid-edit flag
  // (render-time adjustment — the same transition guard as Vue's watch).
  const headerRulesOverridden = draft?.overrides.has('header_rules')
  const [prevHeaderRulesOverridden, setPrevHeaderRulesOverridden] = useState(headerRulesOverridden)
  if (prevHeaderRulesOverridden !== headerRulesOverridden) {
    setPrevHeaderRulesOverridden(headerRulesOverridden)
    if (!headerRulesOverridden) setHeaderRulesInvalidEdits(false)
  }
  const responseRulesOverridden = draft?.overrides.has('response_header_rules')
  const [prevResponseRulesOverridden, setPrevResponseRulesOverridden] =
    useState(responseRulesOverridden)
  if (prevResponseRulesOverridden !== responseRulesOverridden) {
    setPrevResponseRulesOverridden(responseRulesOverridden)
    if (!responseRulesOverridden) setResponseRulesInvalidEdits(false)
  }

  // --- saved feedback flag ---
  const savedFeedback = useTransientFlag(1_600)
  useEffect(() => {
    if (savedAt !== null) savedFeedback.show()
    // eslint-disable-next-line react-hooks/exhaustive-deps -- show() is controller-stable
  }, [savedAt])
  useEffect(() => {
    if (dirty) savedFeedback.clear()
    // eslint-disable-next-line react-hooks/exhaustive-deps -- clear() is controller-stable
  }, [dirty])

  // --- section navigation + route sync ---
  const navItems = [
    { id: 'settings-routing', label: t('settings.navigation.routing') },
    { id: 'settings-connection', label: t('settings.navigation.connection') },
    { id: 'settings-reliability', label: t('settings.navigation.reliability') },
    { id: 'settings-browser-access', label: t('settings.navigation.browserAccess') },
    { id: 'settings-data-maintenance', label: t('settings.navigation.dataMaintenance') },
    { id: 'settings-credentials', label: t('settings.navigation.credentials') },
    { id: 'settings-system', label: t('settings.navigation.system') },
  ]
  const routeSection = parseSettingsRouteSection(rawSearch)
  const { activeSection, selectSection } = useSectionNavigation({
    ids: navItems.map(({ id }) => id),
    initialId: sectionID(routeSection),
    topOffset: 76,
  })

  // Canonicalize non-canonical query (invalid/repeated section), else scrollspy
  // follows URL changes (back/forward, shared links).
  useEffect(() => {
    // Pending transition: the outgoing route still renders while location has
    // moved — a late canonicalization must not resurrect this page.
    if (pathname !== settingsPath) return
    const section = parseSettingsRouteSection(rawSearch)
    const credentials =
      section === 'credentials' ? parseSettingsCredentialsRoute(rawSearch) : undefined
    if (!isCanonicalSettingsRouteQuery(rawSearch, section, credentials)) {
      void navigate({
        to: settingsPath,
        search: serializeSettingsRouteQuery(section, credentials),
        replace: true,
        resetScroll: false,
      })
      return
    }
    selectSection(sectionID(section))
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the search string
  }, [searchStr])

  // Deep-link settle: the scrollspy can't scroll to sections before they mount,
  // and the router's push scroll reset can interrupt — retry once both land.
  const ready = base !== null && draft !== null
  const initialSectionSettled = useRef(false)
  useEffect(() => {
    if (!ready || initialSectionSettled.current) return
    initialSectionSettled.current = true
    const target = sectionID(routeSection)
    const frame = requestAnimationFrame(() => selectSection(target, 'auto'))
    const retry = window.setTimeout(() => selectSection(target, 'auto'), 120)
    return () => {
      cancelAnimationFrame(frame)
      window.clearTimeout(retry)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- first-ready latch
  }, [ready])

  const navigateSection = (id: string): void => {
    const section = sectionFromID(id)
    if (section === undefined) return
    selectSection(id)
    if (section === routeSection) return
    void navigate({
      to: settingsPath,
      search: serializeSettingsRouteQuery(section),
      resetScroll: false,
    })
  }

  // --- dirty/valid bookkeeping ---
  const changedKeys = (() => {
    const changed = runtimeSettingKeys.filter((key) =>
      Object.prototype.hasOwnProperty.call(patch, key),
    ) as RuntimeSettingKey[]
    if (headerRulesInvalidEdits && !changed.includes('header_rules')) changed.push('header_rules')
    if (responseRulesInvalidEdits && !changed.includes('response_header_rules'))
      changed.push('response_header_rules')
    return changed
  })()

  const settingLabel = (key: RuntimeSettingKey): string => {
    if (key === 'affinity_enabled' || key === 'affinity_ttl' || key === 'affinity_capacity')
      return t(`settings.affinity.${key}`)
    if (key === 'request_log_retention_days') return t('settings.logs.retention')
    if (key === 'header_rules') return t('settings.headers.blockTitle')
    if (key === 'cors') return t('settings.browserAccess.cors.title')
    if (key === 'response_header_rules') return t('settings.browserAccess.responseHeaders.title')
    return t(`settings.runtime.${key}`)
  }
  const changedLabels = [
    ...changedKeys.map(settingLabel),
    ...(proxyState.dirty ? [t('common.proxy.title')] : []),
  ]

  const invalidKeys = (() => {
    if (draft === null) return [] as RuntimeSettingKey[]
    return runtimeSettingKeys.filter((key) => {
      if (!draft.overrides.has(key)) return false
      if ((timeoutKeys as readonly string[]).includes(key))
        return !isValidTimeout(draft.values[key as (typeof timeoutKeys)[number]])
      if (key === 'header_rules') return !headerRulesValid
      if (key === 'cors') return !corsValid
      if (key === 'response_header_rules') return !responseHeaderRulesValid
      if (key === 'request_log_retention_days')
        return !isValidRetention(draft.values.request_log_retention_days)
      if (key === 'affinity_capacity')
        return !isValidAffinityCapacity(draft.values.affinity_capacity)
      if (key === 'retry_count' || key === 'blacklist_threshold')
        return !isValidNonNegativeInteger(draft.values[key])
      return false
    })
  })()

  const savedAtLabel = savedAt ? formatLocalInstant(savedAt.getTime(), intl.locale) : ''
  const saveBarError = failed ? t('settings.saveFailed') : ''

  const unsaved = useUnsavedChanges({
    dirty,
    blocked: operationLocked,
    allowRouteUpdate: (to, from) => to.routeId === from.routeId,
  })

  // --- discard flow ---
  const [discardDialogOpen, setDiscardDialogOpen] = useState(false)
  const discard = (): void => {
    controller.discard()
    setHeaderRulesInvalidEdits(false)
    setResponseRulesInvalidEdits(false)
    setBrowserAccessEditorRevision((revision) => revision + 1)
    if (proxyBaseView) {
      setProxyMode(proxyBaseView.configured_mode)
      setProxyEndpoint('')
    }
  }
  const requestDiscard = (): void => {
    if (!dirty || operationLocked) return
    setDiscardDialogOpen(true)
  }
  const confirmDiscard = (): void => {
    discard()
    setDiscardDialogOpen(false)
  }

  // --- validation link focus ---
  const settingTarget = (key: RuntimeSettingKey): string =>
    browserAccessKeys.has(key) ? 'settings-browser-access' : `settings-value-${key}`
  const sectionForKey = (key: RuntimeSettingKey): SettingsRouteSection => {
    if (browserAccessKeys.has(key)) return 'browser-access'
    if (
      key === 'route_strategy' ||
      key === 'affinity_enabled' ||
      key === 'affinity_ttl' ||
      key === 'affinity_capacity'
    )
      return 'routing'
    if (
      key === 'first_byte_timeout' ||
      key === 'request_timeout' ||
      key === 'stream_idle_timeout' ||
      key === 'responses_websocket_enabled'
    )
      return 'connection'
    if (
      key === 'retry_count' ||
      key === 'blacklist_threshold' ||
      key === 'blacklist_release_seconds'
    )
      return 'reliability'
    return 'data-maintenance'
  }
  const focusTarget = (key: RuntimeSettingKey): void => {
    const id = settingTarget(key)
    const sectionElementId = sectionID(sectionForKey(key))
    navigateSection(sectionElementId)
    requestAnimationFrame(() => {
      const target = browserAccessKeys.has(key)
        ? (document
            .getElementById(sectionElementId)
            ?.querySelector<HTMLElement>('[aria-invalid="true"]') ??
          document.getElementById(sectionElementId))
        : document.getElementById(id)
      target?.focus()
    })
  }

  const handleSaveAll = async (): Promise<void> => {
    const extra: SettingsPatch =
      proxyState.dirty && proxyState.value !== undefined ? { proxy_config: proxyState.value } : {}
    await controller.saveAll(extra)
  }

  // Classic drops the settings query cache when leaving the page.
  useEffect(
    () => () => {
      queryClient.removeQueries({ queryKey: controlQueryKeys.settingsAll })
    },
    [queryClient],
  )

  const sectionProps = (base: SettingsResource, draft: SettingsDraft) => ({
    base,
    draft,
    disabled: operationLocked,
  })

  return (
    <div {...stylex.props(styles.page)}>
      <div {...stylex.props(styles.pageInner)}>
        <article {...stylex.props(styles.sheet)} aria-labelledby="settings-title">
          <h1 id="settings-title" {...stylex.props(styles.title)}>
            {t('settings.title')}
          </h1>
          <span aria-live="polite" {...stylex.props(styles.refreshing)}>
            {settingsRefreshing ? t('settings.loading') : ''}
          </span>

          <div {...stylex.props(styles.layout)}>
            <SectionNav
              value={activeSection}
              items={navItems}
              label={t('settings.navigation.label')}
              caption={t('settings.navigation.caption')}
              appearance="ledger"
              onSelect={navigateSection}
            />

            <div {...stylex.props(styles.content)}>
              {(settingsQuery.isPending && resource === null) || initialLoading ? (
                <div
                  {...stylex.props(styles.skeleton)}
                  role="status"
                  aria-label={t('settings.loading')}
                >
                  {Array.from({ length: 6 }, (_, index) => (
                    <Skeleton key={index} height={120} radius={2} />
                  ))}
                </div>
              ) : settingsQuery.isError && resource === null ? (
                <div {...stylex.props(styles.errorBox)} role="alert">
                  <TriangleAlert size={16} aria-hidden />
                  <span>{t('settings.loadFailed')}</span>
                  <Button
                    variant="secondary"
                    size="sm"
                    label={t('common.retry')}
                    onClick={() => void settingsQuery.refetch()}
                  />
                </div>
              ) : base !== null && draft !== null ? (
                <>
                  {settingsQuery.isError && (
                    <div {...stylex.props(styles.staleBanner)} role="status">
                      <TriangleAlert size={13} aria-hidden />
                      <span>{t('settings.stale')}</span>
                      <Button
                        variant="secondary"
                        size="sm"
                        label={t('common.retry')}
                        onClick={() => void settingsQuery.refetch()}
                      />
                    </div>
                  )}

                  {invalidKeys.length > 0 && (
                    <section {...stylex.props(styles.validation)} role="alert" tabIndex={-1}>
                      <strong {...stylex.props(styles.validationTitle)}>
                        {t('settings.validation.title')}
                      </strong>
                      <ul {...stylex.props(styles.validationList)}>
                        {invalidKeys.map((key) => (
                          <li key={key}>
                            <a
                              {...stylex.props(styles.validationLink)}
                              href={`#${settingTarget(key)}`}
                              onClick={(event) => {
                                event.preventDefault()
                                focusTarget(key)
                              }}
                            >
                              {settingLabel(key)}
                            </a>
                          </li>
                        ))}
                      </ul>
                    </section>
                  )}

                  <div>
                    <RoutingSection
                      {...sectionProps(base, draft)}
                      publish={(key, next) => controller.updateDraft({ key, draft: next })}
                    />
                  </div>
                  <div {...stylex.props(styles.contentSection)}>
                    <ConnectionSection
                      {...sectionProps(base, draft)}
                      publish={(key, next) => controller.updateDraft({ key, draft: next })}
                      proxy={base.settings.values.proxy_config}
                      proxyMode={proxyMode}
                      proxyEndpoint={proxyEndpoint}
                      onProxyModeChange={setProxyMode}
                      onProxyEndpointChange={setProxyEndpoint}
                    />
                  </div>
                  <div {...stylex.props(styles.contentSection)}>
                    <ReliabilitySection
                      {...sectionProps(base, draft)}
                      publish={(key, next) => controller.updateDraft({ key, draft: next })}
                    />
                  </div>
                  <div {...stylex.props(styles.contentSection)}>
                    <BrowserAccessSection
                      {...sectionProps(base, draft)}
                      publish={(key, next) => controller.updateDraft({ key, draft: next })}
                      resetKey={browserAccessEditorRevision}
                      onValidChange={setBrowserAccessValid}
                      onHeaderRulesValidChange={setHeaderRulesValid}
                      onCorsValidChange={setCorsValid}
                      onResponseRulesValidChange={setResponseHeaderRulesValid}
                      onHeaderRulesInvalidEditsChange={setHeaderRulesInvalidEdits}
                      onResponseRulesInvalidEditsChange={setResponseRulesInvalidEdits}
                    />
                  </div>
                  <div {...stylex.props(styles.contentSection)}>
                    <DataMaintenanceSection
                      {...sectionProps(base, draft)}
                      publish={(key, next) => controller.updateDraft({ key, draft: next })}
                    />
                  </div>
                </>
              ) : null}

              <div {...stylex.props(styles.contentSection)}>
                <CredentialsSection />
              </div>
              <div {...stylex.props(styles.contentSection)}>
                <SystemInfoSection />
              </div>
            </div>
          </div>

          <Dialog isOpen={discardDialogOpen} onOpenChange={setDiscardDialogOpen} width={440}>
            <Layout
              header={
                <DialogHeader
                  title={t('settings.discardConfirm.title')}
                  subtitle={t('settings.discardConfirm.description')}
                  onOpenChange={setDiscardDialogOpen}
                  hasDivider
                />
              }
              content={
                <LayoutContent isScrollable>
                  <ul {...stylex.props(styles.discardList)}>
                    {changedLabels.map((label) => (
                      <li key={label}>{label}</li>
                    ))}
                  </ul>
                </LayoutContent>
              }
              footer={
                <LayoutFooter hasDivider>
                  <Button
                    variant="secondary"
                    size="sm"
                    label={t('settings.discardConfirm.cancel')}
                    onClick={() => setDiscardDialogOpen(false)}
                  />
                  <Button
                    variant="destructive"
                    size="sm"
                    label={t('settings.discardConfirm.confirm')}
                    onClick={confirmDiscard}
                  />
                </LayoutFooter>
              }
            />
          </Dialog>

          {ready && (
            <StickySaveBar
              appearance="ledger"
              alwaysVisible
              dirty={dirty}
              pending={pending}
              status={failed ? 'error' : savedFeedback.value ? 'saved' : 'idle'}
              error={saveBarError}
              xstyle={styles.saveBar}
              statusContent={
                <span {...stylex.props(styles.statusStack)}>
                  <strong {...stylex.props(styles.statusStrong)}>
                    {pending
                      ? t('settings.saveState.saving')
                      : dirty
                        ? t('settings.dirtySummary', { count: changedLabels.length })
                        : savedFeedback.value
                          ? t('settings.saved')
                          : t('settings.saveState.baseline')}
                  </strong>
                  <span {...stylex.props(styles.statusDetail)}>
                    {pending
                      ? t('settings.saveState.savingNote')
                      : dirty
                        ? changedLabels.join(', ')
                        : savedFeedback.value
                          ? t('settings.savedAt', { time: savedAtLabel })
                          : t('settings.saveState.baselineNote')}
                  </span>
                </span>
              }
              actions={
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    label={t('settings.discard')}
                    isDisabled={!dirty || operationLocked}
                    onClick={requestDiscard}
                  />
                  <Button
                    variant="primary"
                    size="sm"
                    label={t('settings.save')}
                    isLoading={pending}
                    isDisabled={!dirty || !valid || operationLocked}
                    onClick={() => void handleSaveAll()}
                  />
                </>
              }
            />
          )}
        </article>
      </div>
      {unsaved.dialog}
    </div>
  )
}
