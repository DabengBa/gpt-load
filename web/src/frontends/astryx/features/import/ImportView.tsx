import * as stylex from '@stylexjs/stylex'
import { SegmentedControl, SegmentedControlItem } from '@astryxdesign/core'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import { useEffect, useRef, useState } from 'react'

import { pagePath } from '@shared/routing/page-routes'
import {
  isCanonicalImportRouteQuery,
  parseImportRouteQuery,
  serializeImportRouteQuery,
} from '@shared/routing/import-route'
import type { SharedRouteQuery } from '@shared/routing/route-query'
import type { ExistingGroupImportDraft, ImportDraft } from '@shared/domain/import/model-draft'

import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'
import { ExistingGroupImport } from './ExistingGroupImport'
import { useImportOperationMode, type ImportOperationMode } from './import-operation'
import { NewGroupImport } from './NewGroupImport'

/**
 * Classic features/import/ImportView.vue — mode switcher + recovery draft
 * hand-off. The shared codec canonicalizes the URL via validateSearch; the
 * recovery-mode redirect below mirrors the classic setup-time replace for the
 * cases validateSearch cannot invent (a recovered draft carrying group_id).
 */
export function ImportView() {
  const t = useT()
  const services = useAppServices()
  const navigate = useNavigate()
  const { rawSearch, searchStr, pathname } = useRouterState({
    select: (state) => ({
      rawSearch: state.location.search as SharedRouteQuery,
      searchStr: state.location.searchStr,
      pathname: state.location.pathname,
    }),
  })
  const importRoutePath = pagePath('import')
  const routeState = parseImportRouteQuery(rawSearch)
  const hasGroupContext = Object.prototype.hasOwnProperty.call(rawSearch, 'group_id')
  const operationMode = useImportOperationMode()

  // Classic `recoveredDraft = ref(recovery.consume())` — consumed exactly once
  // per page mount (initialDraft prop, not a reactive source afterwards).
  const [recoveredDraft] = useState(() => services.importRecovery.consume())

  const activeMode: ImportOperationMode =
    operationMode ??
    (hasGroupContext
      ? 'existing'
      : (recoveredDraft?.mode ?? (routeState.mode === 'existing' ? 'existing' : 'new')))

  // Classic setup-time + watch(operationMode, immediate) canonicalization that
  // validateSearch cannot cover: a recovered draft overrides the URL mode.
  const canonicalizedRef = useRef(false)
  useEffect(() => {
    if (pathname !== importRoutePath) return
    if (canonicalizedRef.current) return
    canonicalizedRef.current = true
    if (operationMode) return
    if (hasGroupContext) {
      if (routeState.mode !== 'existing') {
        void navigate({
          to: pagePath('import'),
          search: serializeImportRouteQuery({
            mode: 'existing',
            groupID: routeState.groupID,
            discoveryFilter: 'unadded',
          }),
          replace: true,
        })
      }
      return
    }
    if (recoveredDraft?.mode === 'existing') {
      void navigate({
        to: pagePath('import'),
        search: serializeImportRouteQuery({
          mode: 'existing',
          groupID: recoveredDraft.group_id ?? undefined,
          discoveryFilter: 'unadded',
        }),
        replace: true,
      })
      return
    }
    if (recoveredDraft?.mode === 'new' && routeState.mode !== 'new') {
      void navigate({
        to: pagePath('import'),
        search: serializeImportRouteQuery({
          mode: 'new',
          discoveryFilter: 'unadded',
        }),
        replace: true,
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot on mount
  }, [])

  // Classic watch(operationMode, immediate): a live operation pins the URL
  // mode (existing keeps its group_id context).
  useEffect(() => {
    // Guard: this effect can flush during the commit that unmounts ImportView
    // (e.g. a success navigation); a late pin must never pull the user back.
    if (pathname !== importRoutePath) return
    if (!operationMode || routeState.mode === operationMode) return
    void navigate({
      to: pagePath('import'),
      search: serializeImportRouteQuery(
        operationMode === 'existing' && hasGroupContext
          ? {
              mode: 'existing',
              groupID: routeState.groupID,
              discoveryFilter: 'unadded',
            }
          : { mode: operationMode, discoveryFilter: 'unadded' },
      ),
      replace: true,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mirrors the mode watch
  }, [operationMode])

  // Classic watch(route.query, deep, immediate): belt-and-braces replace for
  // anything validateSearch let through (defense in depth, no-op normally).
  useEffect(() => {
    if (pathname !== importRoutePath) return
    const next = parseImportRouteQuery(rawSearch)
    if (!isCanonicalImportRouteQuery(rawSearch, next)) {
      void navigate({
        to: pagePath('import'),
        search: serializeImportRouteQuery(next),
        replace: true,
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the URL string
  }, [searchStr])

  const [clearedRecoveryOnSwitch, setClearedRecoveryOnSwitch] = useState(false)

  function selectMode(mode: string): void {
    if (mode !== 'new' && mode !== 'existing') return
    if (operationMode || activeMode === mode) return
    // Classic `recoveredDraft.value = null` on manual switch.
    if (!clearedRecoveryOnSwitch) setClearedRecoveryOnSwitch(true)
    void navigate({
      to: pagePath('import'),
      search: serializeImportRouteQuery({ mode, discoveryFilter: 'unadded' }),
    })
  }

  const effectiveRecovered = clearedRecoveryOnSwitch ? null : recoveredDraft
  const recoveredNewDraft: ImportDraft | null =
    effectiveRecovered?.mode === 'new' ? effectiveRecovered : null
  const recoveredExistingDraft: ExistingGroupImportDraft | null =
    effectiveRecovered?.mode === 'existing' ? effectiveRecovered : null

  return (
    <section {...stylex.props(styles.page)} aria-labelledby="import-page-title">
      <div {...stylex.props(styles.pageInner)}>
        <div {...stylex.props(styles.content)}>
          <div {...stylex.props(styles.header)}>
            <h1 id="import-page-title" {...stylex.props(styles.title)}>
              {t('import.title')}
            </h1>
            <div {...stylex.props(styles.modeControl)}>
              <SegmentedControl
                value={activeMode}
                label={t('import.mode.label')}
                size="sm"
                onChange={selectMode}
              >
                <SegmentedControlItem
                  value="new"
                  label={t('import.mode.new')}
                  isDisabled={operationMode !== null && operationMode !== 'new'}
                />
                <SegmentedControlItem
                  value="existing"
                  label={t('import.mode.existing')}
                  isDisabled={operationMode !== null && operationMode !== 'existing'}
                />
              </SegmentedControl>
            </div>
          </div>
          {activeMode === 'new' ? (
            <NewGroupImport initialDraft={recoveredNewDraft} />
          ) : (
            <ExistingGroupImport initialDraft={recoveredExistingDraft} />
          )}
        </div>
      </div>
    </section>
  )
}

const narrow = '@media (max-width: 680px)'

const styles = stylex.create({
  page: {
    width: '100%',
    paddingTop: 'var(--stage-padding-top)',
    paddingBottom: 'var(--stage-padding-bottom)',
    paddingInline: {
      default: 'var(--stage-padding-inline)',
      [narrow]: 'var(--stage-padding-inline-compact)',
    },
  },
  pageInner: {
    // Classic PageFrame wide.
    width: 'min(100%, 1240px)',
    marginInline: 'auto',
  },
  content: {
    position: 'relative',
    minWidth: 0,
    minHeight: 0,
  },
  header: {
    display: 'flex',
    alignItems: { default: 'center', [narrow]: 'stretch' },
    flexDirection: { default: 'row', [narrow]: 'column' },
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-control)',
    paddingBottom: 'var(--space-4)',
  },
  title: {
    margin: 0,
    fontSize: 'var(--text-heading-2-size, 20px)',
    fontWeight: 650,
    lineHeight: 1.4,
  },
  modeControl: {
    minWidth: 0,
    width: { default: 'auto', [narrow]: '100%' },
  },
})
