import * as stylex from '@stylexjs/stylex'
import {
  AlertDialog,
  Button,
  CodeBlock,
  Selector,
  TextInput,
} from '@astryxdesign/core'
import { Check, Copy, Zap } from 'lucide-react'
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react'

import type { HomeBaseDto } from '@shared/control/resources/home'
import { revealAccessKey } from '@shared/control/resources/access-keys'
import type { MessageId } from '@shared/i18n/message-ids'
import { registerEphemeralStateCleaner } from '@shared/controllers/ephemeral-state'
import {
  createGatewayActionsController,
  type GatewayActionTarget,
} from '@shared/controllers/gateway-actions'
import {
  ccSwitchTargets,
  clientConfiguration,
  clientFields,
  clientQuickImportURL,
  clientRequiredProtocol,
  gatewayClients,
  type CCSwitchTargetID,
  type GatewayClientID,
} from '@shared/domain/home/gateway-clients'
import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../../app/i18n'
import { RouteLink } from '../../app/route-link'
import { useAppServices } from '../../app/services'
import { useClipboardCopy } from '../../app/use-clipboard-copy'
import { useMediaQuery } from '../../app/use-media-query'
import { ChannelIcon } from '../../components/ChannelIcon'
import { ClientPicker } from './ClientPicker'
import { HomeSectionHeading } from './home-chrome'

const NARROW = '@media (max-width: 860px)'
const TIGHT = '@media (max-width: 560px)'

const styles = stylex.create({
  section: {
    marginTop: 36,
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 20,
  },
  empty: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 'var(--space-3)',
    marginTop: 'var(--space-4)',
  },
  emptyText: {
    margin: 0,
    color: 'var(--color-text-muted)',
  },
  createLink: {
    color: 'var(--color-action)',
    fontWeight: 600,
    textDecoration: { default: 'none', ':hover': 'underline' },
  },
  toolbar: {
    display: 'flex',
    alignItems: 'flex-end',
    gap: 16,
    flexWrap: 'wrap',
    marginTop: 12,
  },
  keyField: {
    display: 'grid',
    minWidth: 0,
    gap: 6,
    flexGrow: { default: 0, [NARROW]: 1 },
    flexShrink: 1,
    flexBasis: { default: '300px', [NARROW]: '100%' },
  },
  label: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  keyControl: {
    display: 'flex',
    minWidth: 0,
    height: 'var(--control-md)',
    overflow: 'hidden',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
  },
  keySelector: {
    flex: '1',
    minWidth: 0,
  },
  keyCopy: {
    flex: 'none',
    borderLeftWidth: 1,
    borderLeftStyle: 'solid',
    borderLeftColor: 'var(--color-border-subtle)',
    borderRadius: 0,
  },
  clients: {
    display: 'grid',
    minWidth: 0,
    gap: 6,
  },
  panel: {
    display: 'grid',
    gap: 'var(--space-3)',
    marginTop: 12,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-sheet)',
    backgroundColor: 'var(--color-surface)',
    padding: 12,
  },
  // Fixed height: the quick-import button (~30px) + padding makes 42px; without
  // it a client with no import would collapse the header to 38px and jump.
  panelHeader: {
    display: 'flex',
    minHeight: 'var(--control-lg)',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
    flexWrap: 'wrap',
    borderRadius: 'var(--radius-tag)',
    backgroundColor:
      'color-mix(in srgb, var(--color-surface-sunken) 52%, var(--color-surface))',
    paddingTop: 6,
    paddingBottom: 6,
    paddingInline: 10,
  },
  panelTitle: {
    display: 'inline-flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 'var(--space-2-5)',
  },
  panelTitleStrong: {
    minWidth: 0,
    fontSize: 'var(--title-section)',
    fontWeight: 650,
    letterSpacing: '-0.01em',
    lineHeight: 'var(--line-compact)',
  },
  panelTitleKind: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
    fontWeight: 400,
  },
  panelBody: {
    display: 'grid',
    gap: 'var(--space-3)',
    paddingTop: 0,
    paddingBottom: 2,
    paddingInline: 2,
  },
  ccSwitchOptions: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'auto minmax(200px, 1fr)',
      [NARROW]: '1fr',
    },
    alignItems: 'end',
    gap: 'var(--space-3)',
  },
  ccSwitchTarget: {
    display: 'grid',
    minWidth: 0,
    gap: 6,
  },
  ccSwitchModel: {
    display: 'grid',
    minWidth: 0,
    gap: 6,
  },
  targets: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
  },
  target: {
    display: 'inline-flex',
    // 桌面紧凑控件,窄屏把触控目标还回 44px(与 classic 一致)。
    minHeight: { default: 'var(--control-sm)', [TIGHT]: 'var(--touch-target)' },
    alignItems: 'center',
    gap: 8,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: {
      default: 'var(--color-border-control)',
      ':hover:not(:disabled)': 'var(--color-text-faint)',
    },
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text)',
    paddingTop: 0,
    paddingBottom: 0,
    paddingInline: 13,
    fontFamily: 'inherit',
    fontSize: 'var(--text-button)',
    fontWeight: 560,
    whiteSpace: 'nowrap',
    cursor: { default: 'pointer', ':disabled': 'not-allowed' },
    transitionProperty: 'border-color, background-color, color',
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
    opacity: { ':disabled': 0.45 },
  },
  targetSelected: {
    borderColor: 'var(--color-action)',
    backgroundColor: 'var(--color-action-soft)',
    color: 'var(--color-action)',
    fontWeight: 620,
  },
  targetIcon: {
    display: 'inline-flex',
    width: 17,
    minWidth: 17,
    height: 17,
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 16,
  },
  modelInput: {
    fontFamily: 'var(--font-mono)',
  },
  modelInputTight: {
    minHeight: 'var(--touch-target)',
  },
  hint: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    fontSize: 'var(--text-sm)',
  },
  hintWarning: {
    color: 'var(--color-warning)',
  },
  hintGlyph: {
    display: 'inline-flex',
    width: 14,
    height: 14,
    flex: 'none',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: 700,
  },
  // Two columns: config (or fields) on the left, steps on the right. When the
  // client is GUI-only the steps take the whole row.
  guide: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'minmax(0, 1fr) minmax(240px, 320px)',
      [NARROW]: 'minmax(0, 1fr)',
    },
    alignItems: 'stretch',
    gap: 'var(--space-4)',
  },
  guideColumn: {
    display: 'flex',
    minWidth: 0,
    minHeight: 0,
    flexDirection: 'column',
  },
  guideCaption: {
    display: 'flex',
    minHeight: 24,
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-2)',
    marginTop: 0,
    marginBottom: 6,
    marginInline: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  guideCode: {
    flex: '1',
    backgroundColor:
      'color-mix(in srgb, var(--color-surface-sunken) 52%, var(--color-surface))',
  },
  fields: {
    display: 'grid',
    flex: '1',
    alignContent: 'start',
    margin: 0,
    gap: 'var(--space-2)',
  },
  fieldRow: {
    display: 'grid',
    gridTemplateColumns: 'minmax(80px, max-content) minmax(0, 1fr)',
    alignItems: 'center',
    gap: 'var(--space-3)',
  },
  fieldTerm: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  fieldValue: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 'var(--space-2)',
    margin: 0,
  },
  fieldCode: {
    flex: '1 1 auto',
    minWidth: 0,
    overflow: 'hidden',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-tag)',
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-code)',
    paddingTop: 7,
    paddingBottom: 7,
    paddingInline: 10,
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  steps: {
    display: 'grid',
    flex: '1',
    gap: 'var(--space-2)',
    alignContent: 'start',
    margin: 0,
    borderRadius: 'var(--radius-tag)',
    backgroundColor:
      'color-mix(in srgb, var(--color-surface-sunken) 52%, var(--color-surface))',
    paddingTop: 11,
    paddingBottom: 11,
    paddingInline: 13,
    listStyle: 'none',
  },
  step: {
    display: 'grid',
    gridTemplateColumns: 'auto minmax(0, 1fr)',
    alignItems: 'baseline',
    gap: 'var(--space-2-5)',
  },
  stepNumber: {
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
    fontVariantNumeric: 'tabular-nums',
    fontWeight: 700,
  },
  stepText: {
    margin: 0,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
    lineHeight: 'var(--line-editorial)',
  },
  feedback: {
    position: 'fixed',
    zIndex: 80, // --z-popover (stylex requires a numeric literal)
    bottom: 26,
    left: '50%',
    width: 'max-content',
    maxWidth: 'calc(100vw - 32px)',
    transform: 'translateX(-50%)',
    borderWidth: 0,
    borderRadius: 8,
    backgroundColor: 'var(--color-text)',
    color: 'var(--color-surface)',
    paddingTop: 8,
    paddingBottom: 8,
    paddingInline: 14,
    boxShadow: 'var(--shadow-overlay)',
    fontSize: 12.5,
    margin: 0,
  },
})

export function GatewayConnection({
  accessKeys,
  selectedAccessKeyID,
  clientID,
  credential,
  selfScoped = false,
  onAccessKeyChange,
  onClientChange,
}: {
  accessKeys: HomeBaseDto['access_keys']
  selectedAccessKeyID: number | null
  clientID: GatewayClientID
  credential?: string
  selfScoped?: boolean
  onAccessKeyChange(id: number): void
  onClientChange(id: GatewayClientID): void
}) {
  const { apiClient } = useAppServices()
  const t = useT()
  const { copy, reset: resetCopy, dialog: copyFallbackDialog } = useClipboardCopy()

  const [controller] = useState(() => createGatewayActionsController())
  const actions = useSyncExternalStore(
    useCallback((listener) => controller.subscribe(listener), [controller]),
    () => controller.getSnapshot(),
  )
  const actionBusy = actions.busy
  const confirmOpen = actions.confirmOpen

  const [ccSwitchTargetID, setCcSwitchTargetID] = useState<CCSwitchTargetID>('claude')
  const [ccSwitchModel, setCcSwitchModel] = useState('')
  // xstyle prop types reject media-query conditions — JS-driven swap for the
  // classic ≤560px touch-target bump on the model input shell.
  const tightViewport = useMediaQuery('(max-width: 560px)')

  const origin = useState(() => window.location.origin)[0]
  const selectedKey =
    accessKeys.find((accessKey) => accessKey.id === selectedAccessKeyID) ?? null
  const selectOptions = accessKeys.map((accessKey) => ({
    value: String(accessKey.id),
    label: `${accessKey.name} · ${accessKey.masked_key}`,
  }))
  const currentClient =
    gatewayClients.find((candidate) => candidate.id === clientID) ?? gatewayClients[0]!
  const currentCCSwitchTarget =
    ccSwitchTargets.find((candidate) => candidate.id === ccSwitchTargetID) ??
    ccSwitchTargets[0]!
  const currentRequiredProtocol = clientRequiredProtocol(
    currentClient,
    currentCCSwitchTarget,
  )
  const selectedKeySupportsClient = Boolean(
    !currentRequiredProtocol || selectedKey?.protocols.includes(currentRequiredProtocol),
  )
  const quickImportAvailable = Boolean(currentClient.quickImport)
  const quickImportRequiresModel =
    clientID === 'cc-switch' && currentCCSwitchTarget.requiresModel
  const quickImportReady =
    quickImportAvailable &&
    selectedKeySupportsClient &&
    (!quickImportRequiresModel || Boolean(ccSwitchModel.trim()))
  const maskedSnippet = selectedKey
    ? clientConfiguration(
        clientID,
        origin,
        selectedKey.masked_key,
        ccSwitchTargetID,
        ccSwitchModel,
        `GPT-Load · ${selectedKey.name}`,
      )
    : ''
  const clientFieldList =
    selectedKey && currentClient.configKind === 'fields'
      ? clientFields(clientID, origin, selectedKey.masked_key)
      : []

  const clientLabel = (id: GatewayClientID): string =>
    t(`home.ledger.connection.clients.${id}` as MessageId)
  const selectedClientKind = (id: GatewayClientID): string => {
    const kind = gatewayClients.find((candidate) => candidate.id === id)?.kind
    return kind ? t(`home.ledger.connection.clientKinds.${kind}` as MessageId) : ''
  }
  const quickImportClientLabel =
    clientID === 'cc-switch'
      ? `${clientLabel(clientID)} · ${t(`home.ledger.connection.ccSwitchTargets.${ccSwitchTargetID}` as MessageId)}`
      : clientLabel(clientID)
  const configurationLanguage =
    clientID === 'new-api'
      ? t('home.ledger.connection.connectionInfo')
      : clientID === 'cc-switch'
        ? t('home.ledger.connection.importParameters')
        : t('home.ledger.connection.configuration')
  const copyConfigurationLabel =
    clientID === 'new-api'
      ? t('home.ledger.connection.copyConnectionInfo')
      : clientID === 'cc-switch'
        ? t('home.ledger.connection.copyImportParameters')
        : t('home.ledger.connection.copyConfiguration')

  const visibleFeedback =
    actions.feedback !== null &&
    actions.feedback.accessKeyID === selectedAccessKeyID &&
    actions.feedback.clientID === clientID
      ? actions.feedback
      : null
  const feedbackMessage = (() => {
    const current = visibleFeedback
    if (!current) return ''
    if (current.target === 'key') {
      return t(
        current.kind === 'success'
          ? 'home.ledger.connection.keyCopied'
          : 'home.ledger.connection.keyCopyFailed',
      )
    }
    if (current.target === 'configuration') {
      return t(
        current.kind === 'success'
          ? 'home.ledger.connection.configurationCopied'
          : 'home.ledger.connection.configurationCopyFailed',
      )
    }
    if (current.target === 'baseUrl' || current.target === 'apiKey') {
      const field = t(`home.ledger.connection.fields.${current.target}` as MessageId)
      return current.kind === 'success'
        ? t('home.ledger.connection.fieldCopied', { field })
        : t('home.ledger.connection.fieldCopyFailed', { field })
    }
    if (current.kind === 'success') {
      return t('home.ledger.connection.quickImportRequested', {
        client: quickImportClientLabel,
      })
    }
    if (current.kind === 'popup-blocked') {
      return t('home.ledger.connection.popupBlocked', { client: quickImportClientLabel })
    }
    return t('home.ledger.connection.quickImportFailed', {
      client: quickImportClientLabel,
    })
  })()
  const copySucceeded = (target: GatewayActionTarget): boolean =>
    visibleFeedback?.target === target && visibleFeedback.kind === 'success'

  const selectedProtocols = selectedKey?.protocols

  // Controller inputs — post-commit values are what pending operations compare
  // against, the same slot the classic live props occupied.
  useEffect(() => {
    controller.setOps({
      reveal: (accessKeyID, signal) =>
        revealAccessKey(apiClient, accessKeyID, signal).then((result) => result.key),
      onInvalidate: resetCopy,
    })
  }, [controller, apiClient, resetCopy])

  useEffect(() => {
    controller.setInput({
      selectedAccessKeyID,
      selectedAccessKeyPresent: selectedKey !== null,
      clientID,
      selfScoped,
      credential,
    })
  }, [controller, selectedAccessKeyID, selectedKey, clientID, selfScoped, credential])

  // Selection vanished from the loaded key list → drop every pending sensitive
  // action and the open confirmation (classic immediate watch, mount included).
  useEffect(() => {
    if (
      selectedAccessKeyID !== null &&
      accessKeys.some((accessKey) => accessKey.id === selectedAccessKeyID)
    ) {
      return
    }
    controller.selectionLost()
  }, [controller, accessKeys, selectedAccessKeyID])

  // Key/client change: invalidate pending ops and close the confirm (the local
  // model-field reset is the render adjustment below).
  const prevSelectionRef = useRef({ key: selectedAccessKeyID, client: clientID })
  useEffect(() => {
    const previous = prevSelectionRef.current
    prevSelectionRef.current = { key: selectedAccessKeyID, client: clientID }
    if (previous.key !== selectedAccessKeyID || previous.client !== clientID) {
      controller.selectionChanged()
    }
  }, [controller, selectedAccessKeyID, clientID])

  // Render adjustments — pure local state mirroring the classic watches.
  // A key change clears the CC Switch model field.
  const [lastKeyID, setLastKeyID] = useState(selectedAccessKeyID)
  if (lastKeyID !== selectedAccessKeyID) {
    setLastKeyID(selectedAccessKeyID)
    setCcSwitchModel('')
  }
  // A protocol-set change re-picks a supported CC Switch target.
  const [lastProtocols, setLastProtocols] = useState(selectedProtocols)
  if (lastProtocols !== selectedProtocols) {
    setLastProtocols(selectedProtocols)
    const protocols = selectedProtocols ?? []
    const currentTarget = ccSwitchTargets.find(
      (candidate) => candidate.id === ccSwitchTargetID,
    )
    if (!currentTarget || !protocols.includes(currentTarget.requiredProtocol)) {
      const supported = ccSwitchTargets.find((candidate) =>
        protocols.includes(candidate.requiredProtocol),
      )
      if (supported) setCcSwitchTargetID(supported.id)
    }
  }

  // Any edit to the snippet inputs (target app, model, credential) invalidates
  // outstanding sensitive operations — the classic flush:sync watch.
  const prevConfigRef = useRef({ ccSwitchTargetID, ccSwitchModel, credential })
  useEffect(() => {
    const previous = prevConfigRef.current
    if (
      previous.ccSwitchTargetID === ccSwitchTargetID &&
      previous.ccSwitchModel === ccSwitchModel &&
      previous.credential === credential
    ) {
      return
    }
    prevConfigRef.current = { ccSwitchTargetID, ccSwitchModel, credential }
    controller.invalidate()
  }, [controller, ccSwitchTargetID, ccSwitchModel, credential])

  useEffect(() => {
    const unregister = registerEphemeralStateCleaner(() => controller.invalidate())
    return () => {
      unregister()
      controller.dispose()
    }
  }, [controller])

  function selectKey(value: string): void {
    if (actionBusy) return
    const id = Number(value)
    if (Number.isSafeInteger(id) && accessKeys.some((accessKey) => accessKey.id === id)) {
      onAccessKeyChange(id)
    }
  }

  function selectCCSwitchTarget(value: CCSwitchTargetID): void {
    if (actionBusy) return
    const target = ccSwitchTargets.find((candidate) => candidate.id === value)
    if (!target || !selectedKey?.protocols.includes(target.requiredProtocol)) return
    controller.invalidate()
    controller.setConfirmOpen(false)
    setCcSwitchTargetID(target.id)
    // Switching the target app keeps the model text — it usually carries over
    // to the other app and clearing it would force a retype.
  }

  async function copyAccessKey(): Promise<void> {
    await controller.withRevealedKey('key', async (key, isCurrent) => {
      if (!isCurrent()) return
      return copy(key)
    })
  }

  async function copyClientConfiguration(): Promise<void> {
    if (!selectedKeySupportsClient) return

    if (clientID === 'codex') {
      try {
        const result = await copy(maskedSnippet)
        if (result === 'success')
          controller.reportImmediate('configuration', 'success')
      } catch {
        controller.reportImmediate('configuration', 'failure')
      }
      return
    }

    await controller.withRevealedKey('configuration', async (key, isCurrent) => {
      if (!isCurrent()) return
      let configuration: string | undefined
      try {
        configuration = clientConfiguration(
          clientID,
          origin,
          key,
          ccSwitchTargetID,
          ccSwitchModel,
          `GPT-Load · ${selectedKey?.name ?? ''}`,
        )
        if (!isCurrent()) return
        return await copy(configuration)
      } finally {
        configuration = undefined
      }
    })
  }

  async function copyField(field: {
    id: 'baseUrl' | 'apiKey'
    secret?: boolean
  }): Promise<void> {
    if (!selectedKeySupportsClient) return
    if (!field.secret) {
      // Non-secret fields copy their displayed value directly — no reveal.
      const entry = clientFieldList.find((candidate) => candidate.id === field.id)
      if (!entry) return
      try {
        const result = await copy(entry.value)
        if (result === 'success') controller.reportImmediate(field.id, 'success')
      } catch {
        controller.reportImmediate(field.id, 'failure')
      }
      return
    }
    await controller.withRevealedKey(field.id, async (key, isCurrent) => {
      if (!isCurrent()) return
      return copy(key)
    })
  }

  async function openQuickImport(): Promise<void> {
    if (!quickImportReady || actionBusy) return

    const popup = window.open('about:blank', '_blank')
    if (!popup) {
      controller.reportImmediate('quick-import', 'popup-blocked')
      return
    }
    try {
      popup.opener = null
    } catch {
      popup.close()
      controller.reportImmediate('quick-import', 'failure')
      return
    }
    controller.setConfirmOpen(false)

    let target: string | undefined
    const opened = await controller.withRevealedKey('quick-import', (key, isCurrent) => {
      if (!isCurrent()) return
      target =
        clientQuickImportURL(
          clientID,
          origin,
          key,
          ccSwitchTargetID,
          ccSwitchModel,
          `GPT-Load · ${selectedKey?.name ?? ''}`,
        ) ?? undefined
      if (!target) throw new Error('QUICK_IMPORT_UNAVAILABLE')
      if (!isCurrent()) return
      popup.location.replace(target)
    })
    target = undefined
    if (!opened) popup.close()
  }

  return (
    <section {...stylex.props(styles.section)} aria-labelledby="gateway-connection-title">
      <HomeSectionHeading
        id="gateway-connection-title"
        title={t('home.ledger.connection.title')}
      />

      {selectedKey === null ? (
        <div {...stylex.props(styles.empty)}>
          <p {...stylex.props(styles.emptyText)}>
            {t('home.ledger.connection.noAccessKey')}
          </p>
          {!selfScoped && (
            <RouteLink to={pagePath('accessKeys')} {...stylex.props(styles.createLink)}>
              {t('home.ledger.connection.createAccessKey')}
            </RouteLink>
          )}
        </div>
      ) : (
        <>
          <div {...stylex.props(styles.toolbar)}>
            <div {...stylex.props(styles.keyField)}>
              <label {...stylex.props(styles.label)} htmlFor="gateway-access-key">
                {t('home.ledger.connection.accessKey')}
              </label>
              <div {...stylex.props(styles.keyControl)}>
                <Selector
                  id="gateway-access-key"
                  variant="ghost"
                  size="sm"
                  xstyle={styles.keySelector}
                  label={t('home.ledger.connection.accessKey')}
                  isLabelHidden
                  value={String(selectedKey.id)}
                  options={selectOptions}
                  isDisabled={actionBusy || selfScoped}
                  onChange={selectKey}
                />
                <span {...stylex.props(styles.keyCopy)}>
                  <Button
                    variant="ghost"
                    size="sm"
                    isIconOnly
                    label={t('home.ledger.connection.copyAccessKey')}
                    icon={
                      copySucceeded('key') ? (
                        <Check size={16} aria-hidden="true" />
                      ) : (
                        <Copy size={16} aria-hidden="true" />
                      )
                    }
                    isLoading={actionBusy}
                    isDisabled={actionBusy}
                    onClick={() => void copyAccessKey()}
                  />
                </span>
              </div>
            </div>

            <div {...stylex.props(styles.clients)}>
              <span id="gateway-client-label" {...stylex.props(styles.label)}>
                {t('home.ledger.connection.clients.label')}
              </span>
              <ClientPicker
                value={clientID}
                protocols={selectedKey.protocols}
                disabled={actionBusy}
                onChange={onClientChange}
              />
            </div>
          </div>

          <div {...stylex.props(styles.panel)}>
            <header {...stylex.props(styles.panelHeader)}>
              <span {...stylex.props(styles.panelTitle)}>
                <strong {...stylex.props(styles.panelTitleStrong)}>
                  {clientLabel(clientID)}
                </strong>
                <span {...stylex.props(styles.panelTitleKind)}>
                  {selectedClientKind(clientID)}
                </span>
              </span>
              {quickImportAvailable && (
                <Button
                  size="sm"
                  icon={<Zap size={14} aria-hidden="true" />}
                  isDisabled={!quickImportReady || actionBusy}
                  isLoading={actionBusy}
                  label={t(
                    clientID === 'cc-switch'
                      ? 'home.ledger.connection.importAndEnable'
                      : 'home.ledger.connection.quickImport',
                  )}
                  onClick={() => controller.setConfirmOpen(true)}
                />
              )}
            </header>

            <div {...stylex.props(styles.panelBody)}>
              {clientID === 'cc-switch' && (
                <div {...stylex.props(styles.ccSwitchOptions)}>
                  <div {...stylex.props(styles.ccSwitchTarget)}>
                    <span {...stylex.props(styles.label)}>
                      {t('home.ledger.connection.targetApplication')}
                    </span>
                    <div
                      {...stylex.props(styles.targets)}
                      role="group"
                      aria-label={t('home.ledger.connection.targetApplication')}
                    >
                      {ccSwitchTargets.map((target) => (
                        <button
                          key={target.id}
                          {...stylex.props(
                            styles.target,
                            target.id === ccSwitchTargetID && styles.targetSelected,
                          )}
                          type="button"
                          disabled={
                            actionBusy ||
                            !selectedKey.protocols.includes(target.requiredProtocol)
                          }
                          aria-pressed={target.id === ccSwitchTargetID}
                          onClick={() => selectCCSwitchTarget(target.id)}
                        >
                          <span {...stylex.props(styles.targetIcon)}>
                            <ChannelIcon icon={target.icon} mark={target.mark} />
                          </span>
                          <span>
                            {t(
                              `home.ledger.connection.ccSwitchTargets.${target.id}` as MessageId,
                            )}
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div {...stylex.props(styles.ccSwitchModel)}>
                    <span {...stylex.props(styles.label)}>
                      {t('home.ledger.connection.primaryModel')}
                      {quickImportRequiresModel &&
                        ` · ${t('home.ledger.connection.required')}`}
                    </span>
                    <TextInput
                      id="cc-switch-primary-model"
                      label={t('home.ledger.connection.primaryModel')}
                      isLabelHidden
                      placeholder={t('home.ledger.connection.modelPlaceholder')}
                      value={ccSwitchModel}
                      onChange={setCcSwitchModel}
                      isDisabled={actionBusy}
                      {...{ maxLength: 200, spellCheck: false } as const}
                      size="sm"
                      xstyle={[
                        styles.modelInput,
                        tightViewport && styles.modelInputTight,
                      ]}
                    />
                  </div>
                </div>
              )}

              {!selectedKeySupportsClient && (
                <p {...stylex.props(styles.hint, styles.hintWarning)} role="status">
                  <span {...stylex.props(styles.hintGlyph)} aria-hidden="true">
                    ▲
                  </span>
                  {t('home.ledger.connection.protocolUnavailable', {
                    client: quickImportClientLabel,
                    protocol: currentRequiredProtocol ?? '',
                  })}
                </p>
              )}

              {quickImportRequiresModel && !ccSwitchModel.trim() && (
                <p {...stylex.props(styles.hint, styles.hintWarning)} role="status">
                  <span {...stylex.props(styles.hintGlyph)} aria-hidden="true">
                    ▲
                  </span>
                  {t('home.ledger.connection.ccSwitchModelRequired')}
                </p>
              )}

              <div {...stylex.props(styles.guide)}>
                {currentClient.configKind === 'snippet' ? (
                  <div {...stylex.props(styles.guideColumn)}>
                    <p {...stylex.props(styles.guideCaption)}>
                      <span>{configurationLanguage}</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        isIconOnly
                        label={copyConfigurationLabel}
                        icon={
                          copySucceeded('configuration') ? (
                            <Check size={16} aria-hidden="true" />
                          ) : (
                            <Copy size={16} aria-hidden="true" />
                          )
                        }
                        isLoading={actionBusy}
                        isDisabled={
                          !selectedKeySupportsClient ||
                          actionBusy ||
                          (quickImportRequiresModel && !ccSwitchModel.trim())
                        }
                        onClick={() => void copyClientConfiguration()}
                      />
                    </p>
                    <CodeBlock
                      code={maskedSnippet}
                      language="json"
                      title={configurationLanguage}
                      hasLanguageLabel={false}
                      container="section"
                      width="100%"
                      isWrapped
                      xstyle={styles.guideCode}
                    />
                  </div>
                ) : (
                  <section {...stylex.props(styles.guideColumn)}>
                    <p {...stylex.props(styles.guideCaption)}>
                      {t('home.ledger.connection.fieldsTitle')}
                    </p>
                    <dl {...stylex.props(styles.fields)}>
                      {clientFieldList.map((field) => (
                        <div key={field.id} {...stylex.props(styles.fieldRow)}>
                          <dt {...stylex.props(styles.fieldTerm)}>
                            {t(`home.ledger.connection.fields.${field.id}` as MessageId)}
                          </dt>
                          <dd {...stylex.props(styles.fieldValue)}>
                            <code {...stylex.props(styles.fieldCode)}>{field.value}</code>
                            <Button
                              variant="ghost"
                              size="sm"
                              isIconOnly
                              label={t('home.ledger.connection.copyField', {
                                field: t(
                                  `home.ledger.connection.fields.${field.id}` as MessageId,
                                ),
                              })}
                              icon={
                                copySucceeded(field.id) ? (
                                  <Check size={16} aria-hidden="true" />
                                ) : (
                                  <Copy size={16} aria-hidden="true" />
                                )
                              }
                              isLoading={field.secret ? actionBusy : false}
                              isDisabled={!selectedKeySupportsClient || actionBusy}
                              onClick={() => void copyField(field)}
                            />
                          </dd>
                        </div>
                      ))}
                    </dl>
                  </section>
                )}

                <section {...stylex.props(styles.guideColumn)}>
                  <p {...stylex.props(styles.guideCaption)}>
                    {t('home.ledger.connection.stepsTitle')}
                  </p>
                  <ol {...stylex.props(styles.steps)}>
                    {Array.from({ length: currentClient.steps }, (_, index) => index + 1).map(
                      (step) => (
                        <li key={step} {...stylex.props(styles.step)}>
                          <span {...stylex.props(styles.stepNumber)} aria-hidden="true">
                            {String(step).padStart(2, '0')}
                          </span>
                          <p {...stylex.props(styles.stepText)}>
                            {t(
                              `home.ledger.connection.steps.${clientID}.s${step}` as MessageId,
                            )}
                          </p>
                        </li>
                      ),
                    )}
                  </ol>
                </section>
              </div>
            </div>
          </div>
        </>
      )}

      {visibleFeedback !== null && (
        <p {...stylex.props(styles.feedback)} role="status">
          {feedbackMessage}
        </p>
      )}

      {copyFallbackDialog}

      <AlertDialog
        isOpen={confirmOpen}
        onOpenChange={(open) => {
          if (!open) controller.setConfirmOpen(false)
        }}
        title={t('home.ledger.connection.quickImportConfirmTitle', {
          client: quickImportClientLabel,
        })}
        description={t('home.ledger.connection.quickImportConfirmDescription', {
          client: quickImportClientLabel,
        })}
        cancelLabel={t('common.cancel')}
        actionLabel={t(
          clientID === 'cc-switch'
            ? 'home.ledger.connection.importAndEnable'
            : 'home.ledger.connection.openAndImport',
        )}
        actionVariant="primary"
        isActionLoading={actionBusy}
        onAction={() => void openQuickImport()}
      />
    </section>
  )
}
