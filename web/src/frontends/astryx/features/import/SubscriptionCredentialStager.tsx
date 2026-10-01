import { Badge, Button, Collapsible, Field, TextInput } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import { ExternalLink, FileJson, Plus, Send } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'

import { useIntl } from 'react-intl'

import { useAppServices } from '../../app/services'
import { useT } from '../../app/i18n'
import { CopyButton } from '../../components/CopyButton'
import { InlineNotice } from '../../components/InlineNotice'
import { RelativeInstant } from '../../components/RelativeInstant'
import { SectionHeader } from '../../components/SectionHeader'
import {
  beginCredentialAuthorization,
  cancelCredentialStage,
  completeCredentialAuthorization,
  getCredentialStage,
  importCredentialStage,
  pollCredentialDeviceAuthorization,
  type CredentialStageNetworkInput,
  type CredentialStage,
} from '@shared/control/resources/credential-stages'
import type {
  ChannelAuthorizationMethod,
  ChannelNoticeDto,
} from '@shared/control/resources/channels'
import type { ProxyConfigInput } from '@shared/control/types'
import { presentSubscriptionErrorKey } from '@shared/domain/import/subscription-error-presenter'
import type { MessageId } from '@shared/i18n/message-ids'

const OAUTH_JSON_PLACEHOLDER = '{"access_token":"...","refresh_token":"..."}'
// 连续失败到这个次数就停止轮询：暂存已经被删除或服务端持续不可用时，
// 无限重试只会一直刷错误，用户反而看不出该重新授权。
const POLL_FAILURE_LIMIT = 5

type StageTone = 'success' | 'warning' | 'danger' | 'neutral'

export function SubscriptionCredentialStager({
  stages,
  onStagesChange,
  channelId,
  channelName = '',
  authorizationMethods,
  proxy,
  groupId,
  notices = [],
  disabled = false,
  entryDisabled = false,
  compact = false,
  hideHeader = false,
  step,
  context = 'create',
}: {
  stages: CredentialStage[]
  onStagesChange(stages: CredentialStage[]): void
  channelId: string
  channelName?: string
  authorizationMethods: ChannelAuthorizationMethod[]
  proxy?: ProxyConfigInput
  groupId?: number
  notices?: ChannelNoticeDto[]
  disabled?: boolean
  entryDisabled?: boolean
  compact?: boolean
  hideHeader?: boolean
  step?: number
  /**
   * create：账号在创建分组时写入；connect：账号确认后加入已有分组。
   * 只影响面向用户的措辞，不改变暂存行为。
   */
  context?: 'create' | 'connect'
}) {
  const { apiClient, toast } = useAppServices()
  const t = useT()
  const intl = useIntl()
  const [busyAction, setBusyAction] = useState<'authorize' | 'import' | `callback:${string}` | ''>(
    '',
  )
  const [feedbackKey, setFeedbackKey] = useState('')
  const [oauthJSON, setOauthJSON] = useState('')
  const [callbackURLs, setCallbackURLs] = useState<Record<string, string>>({})
  const [callbackErrorKeys, setCallbackErrorKeys] = useState<Record<string, string>>({})
  // 弹窗被拦截时授权已经开始，唯一出路是手动打开链接，这里额外提示一句。
  const [popupBlockedStages, setPopupBlockedStages] = useState<Record<string, boolean>>({})
  const [jsonImportOpen, setJsonImportOpen] = useState(false)
  const [nowMS, setNowMS] = useState(() => Date.now())
  const pollingRef = useRef(
    new Map<string, { timer?: number; controller?: AbortController; failures: number }>(),
  )
  const expiryTimersRef = useRef(new Map<string, number>())
  const stagesRef = useRef(stages)
  const disabledRef = useRef(disabled)
  const entryDisabledRef = useRef(entryDisabled)
  const busyActionRef = useRef(busyAction)

  const readyCount = stages.filter(({ status }) => status === 'ready').length
  const hasAccounts = stages.length > 0
  const entryBusy = disabled || entryDisabled || busyAction !== ''
  const stageNetwork: CredentialStageNetworkInput | undefined =
    groupId !== undefined ? { group_id: groupId } : proxy !== undefined ? { proxy } : undefined
  const supportsBrowserOAuth = authorizationMethods.includes('browser_oauth')
  const supportsDeviceOAuth = authorizationMethods.includes('device_oauth')
  const supportsOAuthFile = authorizationMethods.includes('oauth_file')
  const supportsInteractiveOAuth = supportsBrowserOAuth || supportsDeviceOAuth
  const hasEntryMethod = supportsInteractiveOAuth || supportsOAuthFile
  const channelLabel = channelName.trim() || channelId

  function clearCallbackURL(stageID: string): void {
    setCallbackURLs((current) => {
      if (!(stageID in current)) return current
      const next = { ...current }
      delete next[stageID]
      return next
    })
  }

  function clearCallbackError(stageID: string): void {
    setCallbackErrorKeys((current) => {
      if (!(stageID in current)) return current
      const next = { ...current }
      delete next[stageID]
      return next
    })
  }

  function clearStageFlags(stageID: string): void {
    setPopupBlockedStages((current) => {
      if (!(stageID in current)) return current
      const next = { ...current }
      delete next[stageID]
      return next
    })
  }

  function stopPolling(stageID: string): void {
    const state = pollingRef.current.get(stageID)
    if (!state) return
    if (state.timer !== undefined) window.clearTimeout(state.timer)
    state.controller?.abort()
    pollingRef.current.delete(stageID)
  }

  function stopExpiryTimer(stageID: string): void {
    const timer = expiryTimersRef.current.get(stageID)
    if (timer !== undefined) window.clearTimeout(timer)
    expiryTimersRef.current.delete(stageID)
  }

  function shouldPoll(stage: CredentialStage): boolean {
    return stage.status === 'pending_authorization' || stage.status === 'exchanging'
  }

  function replaceStage(stage: CredentialStage): void {
    const current = stagesRef.current
    const existing = current.find((item) => item.stage_id === stage.stage_id)
    const merged = existing
      ? {
          ...stage,
          authorization_url: stage.authorization_url ?? existing.authorization_url,
          redirect_uri: stage.redirect_uri ?? existing.redirect_uri,
          authorization_method: stage.authorization_method ?? existing.authorization_method,
          user_code: stage.user_code ?? existing.user_code,
          next_poll_at_ms: stage.next_poll_at_ms ?? existing.next_poll_at_ms,
          duplicate: stage.duplicate ?? existing.duplicate,
        }
      : stage
    const index = current.findIndex((item) => item.stage_id === stage.stage_id)
    onStagesChange(
      index === -1
        ? [...current, merged]
        : current.map((item, itemIndex) => (itemIndex === index ? merged : item)),
    )
    if (!shouldPoll(stage)) {
      clearCallbackURL(stage.stage_id)
      clearCallbackError(stage.stage_id)
      clearStageFlags(stage.stage_id)
    }
  }

  function expireReadyStage(stageID: string): void {
    stopExpiryTimer(stageID)
    const stage = stagesRef.current.find((item) => item.stage_id === stageID)
    if (!stage || stage.status !== 'ready') return
    if (stage.expires_at_ms > Date.now()) {
      scheduleExpiry(stage)
      return
    }
    replaceStage({ ...stage, status: 'expired' })
  }

  function scheduleExpiry(stage: CredentialStage): void {
    stopExpiryTimer(stage.stage_id)
    if (stage.status !== 'ready') return
    const delay = Math.max(0, Math.min(stage.expires_at_ms - Date.now() + 10, 2_147_483_647))
    expiryTimersRef.current.set(
      stage.stage_id,
      window.setTimeout(() => expireReadyStage(stage.stage_id), delay),
    )
  }

  function schedulePoll(stage: CredentialStage): void {
    if (!shouldPoll(stage) || pollingRef.current.has(stage.stage_id)) return
    const state: { timer?: number; controller?: AbortController; failures: number } = {
      failures: 0,
    }
    pollingRef.current.set(stage.stage_id, state)
    const poll = async () => {
      const controller = new AbortController()
      state.controller = controller
      let polledStage: CredentialStage | undefined
      try {
        const current = stagesRef.current.find((item) => item.stage_id === stage.stage_id) ?? stage
        const next =
          current.authorization_method === 'device_oauth' &&
          current.status === 'pending_authorization'
            ? await pollCredentialDeviceAuthorization(apiClient, stage.stage_id, controller.signal)
            : await getCredentialStage(apiClient, stage.stage_id, controller.signal)
        polledStage = next
        state.failures = 0
        setFeedbackKey((key) => (key === 'import.subscription.pollFailed' ? '' : key))
        replaceStage(next)
        if (!shouldPoll(next)) {
          stopPolling(stage.stage_id)
          return
        }
      } catch {
        if (!controller.signal.aborted) {
          state.failures += 1
          setFeedbackKey(
            state.failures >= POLL_FAILURE_LIMIT
              ? 'import.subscription.pollAbandoned'
              : 'import.subscription.pollFailed',
          )
          if (state.failures >= POLL_FAILURE_LIMIT) {
            stopPolling(stage.stage_id)
            return
          }
        }
      } finally {
        state.controller = undefined
      }
      if (pollingRef.current.has(stage.stage_id)) {
        const latest =
          polledStage ?? stagesRef.current.find((item) => item.stage_id === stage.stage_id)
        const providerDelay =
          latest?.authorization_method === 'device_oauth' && latest.next_poll_at_ms
            ? Math.max(250, latest.next_poll_at_ms - Date.now())
            : 1_200
        state.timer = window.setTimeout(
          poll,
          state.failures ? 1_200 * Math.min(state.failures + 1, 4) : providerDelay,
        )
      }
    }
    const initialDelay =
      stage.authorization_method === 'device_oauth' && stage.next_poll_at_ms
        ? Math.max(250, stage.next_poll_at_ms - Date.now())
        : 800
    state.timer = window.setTimeout(poll, initialDelay)
  }

  useEffect(() => {
    stagesRef.current = stages
    disabledRef.current = disabled
    entryDisabledRef.current = entryDisabled
    busyActionRef.current = busyAction
    for (const stage of stages) {
      schedulePoll(stage)
      scheduleExpiry(stage)
    }
    const live = new Set(stages.map(({ stage_id }) => stage_id))
    for (const stageID of pollingRef.current.keys()) {
      if (!live.has(stageID)) stopPolling(stageID)
    }
    for (const stageID of expiryTimersRef.current.keys()) {
      if (!live.has(stageID)) stopExpiryTimer(stageID)
    }
  })

  // 弹窗必须在用户手势所在的那一个任务里打开，任何 await 之后再开都会被拦截。
  function openAuthorizationPopup(): Window | null {
    return window.open(
      'about:blank',
      `gpt-load-${channelId}-oauth-${Date.now()}`,
      'popup,width=720,height=820,resizable=yes,scrollbars=yes',
    )
  }

  async function beginAuthorization(
    existingPopup?: Window | null,
    replacingExistingStage = false,
  ): Promise<void> {
    if (
      !supportsInteractiveOAuth ||
      disabled ||
      (entryDisabled && !replacingExistingStage) ||
      busyActionRef.current
    ) {
      existingPopup?.close()
      return
    }
    setFeedbackKey('')
    const popup = existingPopup === undefined ? openAuthorizationPopup() : existingPopup
    setBusyAction('authorize')
    try {
      const stage = await beginCredentialAuthorization(apiClient, channelId, stageNetwork)
      replaceStage(stage)
      schedulePoll(stage)
      if (popup && stage.authorization_url) popup.location.replace(stage.authorization_url)
      else setPopupBlockedStages((current) => ({ ...current, [stage.stage_id]: true }))
    } catch (cause) {
      popup?.close()
      setFeedbackKey(presentSubscriptionErrorKey(cause, 'import.subscription.authorizeFailed'))
    } finally {
      setBusyAction('')
    }
  }

  async function importFile(event: ChangeEvent<HTMLInputElement>): Promise<void> {
    const input = event.target
    const files = Array.from(input.files ?? [])
    input.value = ''
    if (
      files.length === 0 ||
      !supportsOAuthFile ||
      disabled ||
      entryDisabled ||
      busyActionRef.current
    ) {
      return
    }
    setFeedbackKey('')
    setBusyAction('import')
    const imported: CredentialStage[] = []
    let failed = 0
    try {
      for (const file of files) {
        try {
          imported.push(await importCredentialStage(apiClient, channelId, file, stageNetwork))
        } catch {
          failed += 1
        }
      }
      if (imported.length > 0) {
        const knownStageIDs = new Set(stagesRef.current.map(({ stage_id }) => stage_id))
        onStagesChange([
          ...stagesRef.current,
          ...imported.filter(({ stage_id }) => !knownStageIDs.has(stage_id)),
        ])
        setOauthJSON('')
        setJsonImportOpen(false)
      }
      toast.show({
        message: t('import.subscription.importResult', {
          succeeded: intl.formatNumber(imported.length),
          failed: intl.formatNumber(failed),
        }),
        tone: failed === 0 ? 'success' : imported.length === 0 ? 'danger' : 'warning',
        duration: 4_000,
      })
    } finally {
      setBusyAction('')
    }
  }

  async function importText(): Promise<void> {
    const value = oauthJSON.trim()
    if (!value) return
    await importOAuthJSON(
      new File([value], `${channelId}-credential.json`, { type: 'application/json' }),
    )
  }

  async function importOAuthJSON(file: File): Promise<void> {
    if (!supportsOAuthFile || disabled || entryDisabled || busyActionRef.current) return
    setFeedbackKey('')
    setBusyAction('import')
    try {
      replaceStage(await importCredentialStage(apiClient, channelId, file, stageNetwork))
      setOauthJSON('')
      setJsonImportOpen(false)
    } catch (cause) {
      setFeedbackKey(presentSubscriptionErrorKey(cause, 'import.subscription.importFailed'))
    } finally {
      setBusyAction('')
    }
  }

  async function submitCallback(stage: CredentialStage): Promise<void> {
    const callbackURL = callbackURLs[stage.stage_id]?.trim() ?? ''
    if (!callbackURL || disabled || busyActionRef.current) return
    setFeedbackKey('')
    clearCallbackError(stage.stage_id)
    setBusyAction(`callback:${stage.stage_id}`)
    try {
      replaceStage(await completeCredentialAuthorization(apiClient, stage.stage_id, callbackURL))
    } catch (cause) {
      setCallbackErrorKeys((current) => ({
        ...current,
        [stage.stage_id]: presentSubscriptionErrorKey(cause, 'import.subscription.callbackFailed'),
      }))
    } finally {
      setBusyAction('')
    }
  }

  // 授权会话的剩余时间用 m:ss 倒计时，而不是「10 分钟后」这类相对措辞——
  // 用户在这一步是在等一个明确的截止点，秒级读数才有参考价值。
  function remainingCountdown(stage: CredentialStage): string {
    const remainingSeconds = Math.max(0, Math.floor((stage.expires_at_ms - nowMS) / 1_000))
    const minutes = Math.floor(remainingSeconds / 60)
    const seconds = remainingSeconds % 60
    return `${minutes}:${String(seconds).padStart(2, '0')}`
  }

  function callbackEndpoint(stage: CredentialStage): string {
    return stage.redirect_uri ?? t('import.subscription.callbackEndpointFallback')
  }

  function callbackPlaceholder(stage: CredentialStage): string {
    if (!stage.redirect_uri)
      return t('import.subscription.callbackPlaceholder', {
        port: t('import.subscription.callbackPortToken'),
      })
    const separator = stage.redirect_uri.includes('?') ? '&' : '?'
    return `${stage.redirect_uri}${separator}code=...&state=...`
  }

  async function restartAuthorization(stage: CredentialStage): Promise<void> {
    if (!supportsInteractiveOAuth || disabled || busyActionRef.current) return
    const popup = openAuthorizationPopup()
    await removeStage(stage)
    await beginAuthorization(popup, true)
  }

  async function removeStage(stage: CredentialStage): Promise<void> {
    if (disabled) return
    setFeedbackKey('')
    if (stage.status === 'pending_authorization' || stage.status === 'ready') {
      try {
        await cancelCredentialStage(apiClient, stage.stage_id)
      } catch (cause) {
        // 取消失败也要把卡片摘掉：暂存自己会到期，留一张点不动的卡片只会把
        // 用户困在这一步。这里只作为提示，不阻断移除。
        setFeedbackKey(presentSubscriptionErrorKey(cause, 'import.subscription.cancelFailed'))
      }
    }
    stopPolling(stage.stage_id)
    stopExpiryTimer(stage.stage_id)
    clearCallbackURL(stage.stage_id)
    clearCallbackError(stage.stage_id)
    clearStageFlags(stage.stage_id)
    onStagesChange(stagesRef.current.filter(({ stage_id }) => stage_id !== stage.stage_id))
  }

  function statusTone(stage: CredentialStage): StageTone {
    if (stage.status === 'ready' && stage.duplicate) return 'warning'
    if (stage.status === 'ready') return 'success'
    if (stage.status === 'pending_authorization' || stage.status === 'exchanging') return 'warning'
    if (stage.status === 'consumed') return 'neutral'
    return 'danger'
  }

  function isAwaiting(stage: CredentialStage): boolean {
    return stage.status === 'pending_authorization' || stage.status === 'exchanging'
  }

  // 这些状态下暂存已经没救了，卡片必须自带「重新授权」出口，
  // 否则用户只剩「移除」一个动作，得自己想到再点一次登录。
  function isRecoverable(stage: CredentialStage): boolean {
    return (
      stage.status === 'failed' ||
      stage.status === 'cancelled' ||
      stage.status === 'expired' ||
      stage.status === 'outcome_unknown'
    )
  }

  function stageErrorKey(code: string): MessageId {
    const known: Readonly<Record<string, MessageId>> = {
      authorization_denied: 'import.subscription.stageError.authorizationDenied',
      authorization_failed: 'import.subscription.stageError.authorizationFailed',
      authorization_expired: 'import.subscription.stageError.authorizationExpired',
      authorization_exchange_rejected: 'import.subscription.stageError.exchangeRejected',
      authorization_exchange_unknown: 'import.subscription.stageError.exchangeUnknown',
      authorization_exchange_interrupted: 'import.subscription.stageError.exchangeInterrupted',
      credential_refresh_identity_changed: 'import.subscription.stageError.refreshIdentityChanged',
      credential_refresh_rejected: 'import.subscription.stageError.refreshRejected',
      credential_refresh_outcome_unknown: 'import.subscription.stageError.refreshUnknown',
      credential_refresh_persist_failed: 'import.subscription.stageError.refreshPersistFailed',
    }
    return known[code] ?? 'import.subscription.stageError.unknown'
  }

  function stageErrorMessage(stage: CredentialStage): string {
    if (stage.error_code) return t(stageErrorKey(stage.error_code))
    if (stage.status === 'expired') return t('import.subscription.stageError.expired')
    if (stage.status === 'cancelled') return t('import.subscription.stageError.cancelled')
    return t('import.subscription.stageError.unknown')
  }

  useEffect(() => {
    const countdownTimer = window.setInterval(() => {
      setNowMS(Date.now())
    }, 1_000)
    const polling = pollingRef.current
    const expiryTimers = expiryTimersRef.current
    return () => {
      for (const stageID of [...polling.keys()]) stopPolling(stageID)
      for (const stageID of [...expiryTimers.keys()]) stopExpiryTimer(stageID)
      window.clearInterval(countdownTimer)
    }
  }, [])

  return (
    <section
      {...stylex.props(styles.root, compact && styles.compact)}
      aria-labelledby={hideHeader ? undefined : 'subscription-stager-title'}
      aria-label={hideHeader ? t('import.subscription.title') : undefined}
    >
      {!hideHeader && (
        <SectionHeader
          headingId="subscription-stager-title"
          step={step}
          title={t('import.subscription.title')}
          actions={
            readyCount > 0 ? (
              <span {...stylex.props(styles.count)}>
                {t('import.subscription.readyCount', { count: intl.formatNumber(readyCount) })}
              </span>
            ) : undefined
          }
        />
      )}

      {!hasAccounts &&
        notices.map((notice) => (
          <InlineNotice key={notice.id} tone={notice.tone} appearance="ledger">
            {t(`import.subscription.channelNotice.${notice.id}` as MessageId)}
          </InlineNotice>
        ))}

      {hasAccounts && (
        <div {...stylex.props(styles.accounts)}>
          {stages.map((stage) => (
            <article
              key={stage.stage_id}
              {...stylex.props(styles.account, toneStyles[statusTone(stage)])}
            >
              {isAwaiting(stage) ? (
                <div {...stylex.props(styles.summary, styles.summaryAwaiting)} role="status">
                  <span {...stylex.props(styles.spinner)} aria-hidden="true" />
                  <div {...stylex.props(styles.identity)}>
                    <strong {...stylex.props(styles.awaitingTitle)}>
                      {stage.status === 'exchanging'
                        ? t('import.subscription.exchanging')
                        : stage.authorization_method === 'device_oauth'
                          ? t('import.subscription.deviceWaiting')
                          : t('import.subscription.waiting')}
                    </strong>
                    <span {...stylex.props(styles.identityDetail)}>
                      {stage.authorization_method === 'device_oauth'
                        ? t('import.subscription.deviceWaitingHelp', { channel: channelLabel })
                        : t('import.subscription.waitingHelp', { channel: channelLabel })}
                    </span>
                  </div>
                  {stage.status === 'pending_authorization' && (
                    <span
                      {...stylex.props(styles.countdown)}
                      aria-label={t('import.subscription.sessionRemaining')}
                    >
                      {remainingCountdown(stage)}
                    </span>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    isDisabled={disabled || stage.status === 'exchanging'}
                    onClick={() => void removeStage(stage)}
                    label={t('common.cancel')}
                  />
                </div>
              ) : (
                <div {...stylex.props(styles.summary)}>
                  <div {...stylex.props(styles.identity)}>
                    <strong {...stylex.props(styles.identityName)}>
                      {stage.account.email_mask || t('import.subscription.pendingAccount')}
                    </strong>
                    {stage.status === 'ready' && (
                      <span {...stylex.props(styles.identityDetail)}>
                        {stage.duplicate ? (
                          t('import.subscription.duplicateNotice')
                        ) : (
                          <>
                            {t(`import.subscription.readyNotice.${context}` as MessageId)}
                            {' · '}
                            {t('import.subscription.expires')}{' '}
                            <RelativeInstant
                              instant={stage.expires_at_ms}
                              emptyLabel={t('import.subscription.unknown')}
                              hint
                            />
                          </>
                        )}
                      </span>
                    )}
                  </div>
                  <Badge
                    variant={
                      statusTone(stage) === 'success'
                        ? 'success'
                        : statusTone(stage) === 'warning'
                          ? 'warning'
                          : statusTone(stage) === 'danger'
                            ? 'error'
                            : 'neutral'
                    }
                    label={
                      stage.status === 'ready' && stage.duplicate
                        ? t('import.subscription.duplicateStatus')
                        : t(`import.subscription.status.${stage.status}` as MessageId)
                    }
                  />
                  <Button
                    variant="ghost"
                    size="sm"
                    isDisabled={disabled}
                    onClick={() => void removeStage(stage)}
                    label={t('import.subscription.remove')}
                    xstyle={styles.dangerGhost}
                  />
                </div>
              )}

              {isRecoverable(stage) && (
                <div {...stylex.props(styles.recover)} role="alert">
                  <span {...stylex.props(styles.recoverText)}>{stageErrorMessage(stage)}</span>
                  {supportsInteractiveOAuth && (
                    <Button
                      size="sm"
                      isDisabled={disabled || busyAction !== ''}
                      isLoading={busyAction === 'authorize'}
                      onClick={() => void restartAuthorization(stage)}
                      label={t('import.subscription.restart')}
                    />
                  )}
                </div>
              )}

              {/* 远程部署时浏览器可能无法访问服务端声明的 loopback callback；手动授权是
                  常规路径而不是异常兜底，因此授权链接与回调输入始终展开。 */}
              {stage.status === 'pending_authorization' && stage.authorization_url && (
                <div {...stylex.props(styles.authorization)}>
                  {popupBlockedStages[stage.stage_id] && (
                    <InlineNotice tone="warning" appearance="ledger">
                      {t('import.subscription.popupBlocked')}
                    </InlineNotice>
                  )}
                  <p {...stylex.props(styles.manualHint)}>
                    {stage.authorization_method === 'device_oauth'
                      ? t('import.subscription.deviceInstructions')
                      : t('import.subscription.manualHint', {
                          redirectUri: callbackEndpoint(stage),
                        })}
                  </p>

                  <div {...stylex.props(styles.linkField)}>
                    <span {...stylex.props(styles.fieldLabel)}>
                      {t('import.subscription.authorizationLink')}
                    </span>
                    <div {...stylex.props(styles.authorizationLink)}>
                      <code {...stylex.props(styles.linkCode)}>{stage.authorization_url}</code>
                      <CopyButton
                        value={stage.authorization_url}
                        label={t('import.subscription.copyAuthorization')}
                        successLabel={t('common.copied')}
                        failureLabel={t('common.copyFailed')}
                      />
                      <a
                        {...stylex.props(styles.openLink)}
                        href={stage.authorization_url}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <ExternalLink size={15} aria-hidden="true" />
                        <span>{t('import.subscription.openAuthorization')}</span>
                      </a>
                    </div>
                  </div>

                  {stage.authorization_method === 'device_oauth' && stage.user_code && (
                    <div {...stylex.props(styles.linkField)}>
                      <span {...stylex.props(styles.fieldLabel)}>
                        {t('import.subscription.userCode')}
                      </span>
                      <div {...stylex.props(styles.deviceCode)}>
                        <code {...stylex.props(styles.deviceCodeValue)}>{stage.user_code}</code>
                        <CopyButton
                          value={stage.user_code}
                          label={t('import.subscription.copyUserCode')}
                          successLabel={t('common.copied')}
                          failureLabel={t('common.copyFailed')}
                        />
                      </div>
                    </div>
                  )}

                  {stage.authorization_method === 'browser_oauth' && (
                    <form
                      {...stylex.props(styles.callbackForm)}
                      onSubmit={(event) => {
                        event.preventDefault()
                        void submitCallback(stage)
                      }}
                    >
                      <TextInput
                        label={t('import.subscription.callbackLabel')}
                        description={t('import.subscription.callbackHelp')}
                        status={
                          callbackErrorKeys[stage.stage_id]
                            ? {
                                type: 'error',
                                message: t(callbackErrorKeys[stage.stage_id] as MessageId),
                              }
                            : undefined
                        }
                        value={callbackURLs[stage.stage_id] ?? ''}
                        onChange={(value) =>
                          setCallbackURLs((current) => ({
                            ...current,
                            [stage.stage_id]: value,
                          }))
                        }
                        type="text"
                        autoComplete="off"
                        isDisabled={disabled || busyAction !== ''}
                        placeholder={callbackPlaceholder(stage)}
                        xstyle={styles.monoInput}
                        onPaste={() => {
                          window.setTimeout(() => {
                            const latest = stagesRef.current.find(
                              (item) => item.stage_id === stage.stage_id,
                            )
                            const pasted = callbackURLs[stage.stage_id]?.trim()
                            if (latest && pasted) void submitCallback(latest)
                          })
                        }}
                      />
                      <Button
                        type="submit"
                        variant="secondary"
                        size="sm"
                        isLoading={busyAction === `callback:${stage.stage_id}`}
                        isDisabled={
                          disabled || busyAction !== '' || !callbackURLs[stage.stage_id]?.trim()
                        }
                        xstyle={styles.callbackSubmit}
                        icon={<Send size={15} aria-hidden="true" />}
                        label={t('import.subscription.submitCallback')}
                      />
                    </form>
                  )}

                  {supportsInteractiveOAuth && (
                    <Button
                      variant="ghost"
                      size="sm"
                      isDisabled={disabled || busyAction !== ''}
                      onClick={() => void restartAuthorization(stage)}
                      xstyle={styles.restart}
                      label={t('import.subscription.restart')}
                    />
                  )}
                </div>
              )}
            </article>
          ))}
        </div>
      )}

      {hasEntryMethod && (
        <div {...stylex.props(styles.entry)}>
          <div {...stylex.props(styles.entryActions)}>
            {supportsInteractiveOAuth && (
              <Button
                variant={hasAccounts ? 'secondary' : 'primary'}
                size={hasAccounts ? 'sm' : 'lg'}
                isLoading={busyAction === 'authorize'}
                isDisabled={entryBusy}
                onClick={() => void beginAuthorization()}
                icon={hasAccounts ? <Plus size={15} aria-hidden="true" /> : undefined}
                label={
                  hasAccounts
                    ? t('import.subscription.addAnother')
                    : t('import.subscription.authorize', { channel: channelLabel })
                }
              />
            )}
            {supportsOAuthFile && (
              <label
                {...stylex.props(
                  styles.file,
                  styles.entryFile,
                  hasAccounts && styles.entryFileCompact,
                  !supportsInteractiveOAuth && styles.entryFilePrimary,
                  entryBusy && styles.fileDisabled,
                )}
              >
                <FileJson size={16} aria-hidden="true" />
                {busyAction === 'import'
                  ? t('import.subscription.importing')
                  : t('import.subscription.importFile')}
                <input
                  {...stylex.props(styles.fileInput)}
                  type="file"
                  accept="application/json,.json"
                  multiple
                  disabled={entryBusy}
                  onChange={(event) => void importFile(event)}
                />
              </label>
            )}
          </div>

          {supportsOAuthFile && (
            <Collapsible
              isOpen={jsonImportOpen}
              onOpenChange={setJsonImportOpen}
              trigger={
                <span {...stylex.props(styles.jsonDisclosureSummary)}>
                  <FileJson size={16} aria-hidden="true" {...stylex.props(styles.jsonIcon)} />
                  <span>{t('import.subscription.pasteJSON')}</span>
                </span>
              }
            >
              <div {...stylex.props(styles.json)}>
                <Field
                  label={t('import.subscription.oauthJSONLabel')}
                  inputID="subscription-oauth-json"
                >
                  <textarea
                    {...stylex.props(styles.jsonTextarea)}
                    value={oauthJSON}
                    rows={5}
                    disabled={entryBusy}
                    autoComplete="off"
                    autoCapitalize="none"
                    spellCheck={false}
                    placeholder={OAUTH_JSON_PLACEHOLDER}
                    onChange={(event) => setOauthJSON(event.target.value)}
                  />
                </Field>

                <div {...stylex.props(styles.importActions)}>
                  <Button
                    variant="secondary"
                    isLoading={busyAction === 'import'}
                    isDisabled={entryBusy || !oauthJSON.trim()}
                    onClick={() => void importText()}
                    icon={<FileJson size={16} aria-hidden="true" />}
                    label={t('import.subscription.importText')}
                  />
                </div>
              </div>
            </Collapsible>
          )}
        </div>
      )}

      {!hasAccounts && hasEntryMethod && notices.length === 0 && (
        <InlineNotice tone="warning" appearance="ledger-hint">
          {t('import.subscription.riskNotice')}
        </InlineNotice>
      )}

      {feedbackKey !== '' && (
        <InlineNotice tone="danger" appearance="ledger">
          {t(feedbackKey as MessageId, { channel: channelLabel })}
        </InlineNotice>
      )}
    </section>
  )
}

const styles = stylex.create({
  root: {
    display: 'grid',
    gap: 'var(--space-4)',
    minWidth: 0,
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    paddingTop: '22px',
    paddingBottom: 'var(--space-6)',
  },
  /* compact 用于抽屉内部：外层容器已提供内边距与边界，这里不再叠加 */
  compact: {
    borderBottomWidth: 0,
    paddingTop: 0,
    paddingBottom: 0,
  },
  count: {
    flexShrink: 0,
    borderRadius: 'var(--radius-tag)',
    backgroundColor: 'var(--color-success-bg)',
    color: 'var(--color-text)',
    paddingTop: '4px',
    paddingBottom: '4px',
    paddingLeft: '8px',
    paddingRight: '8px',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 600,
  },
  accounts: {
    display: 'grid',
    gap: 'var(--space-2)',
    minWidth: 0,
  },
  account: {
    display: 'grid',
    gap: '10px',
    minWidth: 0,
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderLeftWidth: '3px',
    borderLeftStyle: 'solid',
    borderLeftColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    paddingTop: 'var(--space-3)',
    paddingBottom: 'var(--space-3)',
    paddingLeft: 'var(--space-3-5)',
    paddingRight: 'var(--space-3-5)',
  },
  summary: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) auto auto',
    alignItems: 'center',
    gap: 'var(--space-3)',
    minWidth: 0,
  },
  summaryAwaiting: {
    gridTemplateColumns: 'auto minmax(0, 1fr) auto auto',
  },
  identity: {
    display: 'grid',
    minWidth: 0,
    gap: '3px',
  },
  identityName: {
    overflow: 'hidden',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  /* 等待态的主行是一句话而不是账号地址，用正文字体读起来更自然 */
  awaitingTitle: {
    overflow: 'hidden',
    fontSize: 'var(--text-body)',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    fontWeight: 620,
  },
  identityDetail: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  countdown: {
    flexShrink: 0,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    fontVariantNumeric: 'tabular-nums',
    fontWeight: 620,
  },
  recover: {
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    gap: 'var(--space-2) var(--space-3)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-danger-bg)',
    paddingTop: '8px',
    paddingBottom: '8px',
    paddingLeft: '10px',
    paddingRight: '10px',
    color: 'var(--color-text)',
    fontSize: 'var(--text-meta)',
  },
  recoverText: {
    minWidth: 0,
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: 'auto',
  },
  restart: {
    justifySelf: 'start',
    fontSize: 'var(--text-sm)',
    fontWeight: 560,
    paddingLeft: 0,
  },
  dangerGhost: {
    color: 'var(--color-danger)',
  },
  spinner: {
    width: '14px',
    height: '14px',
    flexShrink: 0,
    borderWidth: '2px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderTopColor: 'var(--color-action)',
    borderRadius: '50%',
    animationName: stylex.keyframes({ to: { transform: 'rotate(360deg)' } }),
    animationDuration: '0.9s',
    animationTimingFunction: 'linear',
    animationIterationCount: 'infinite',
  },
  authorization: {
    display: 'grid',
    gap: 'var(--space-3)',
    minWidth: 0,
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 'var(--space-3)',
  },
  manualHint: {
    margin: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    lineHeight: 1.6,
  },
  linkField: {
    display: 'grid',
    gap: '5px',
    minWidth: 0,
  },
  fieldLabel: {
    display: 'block',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 600,
  },
  authorizationLink: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) auto auto',
    alignItems: 'center',
    gap: 'var(--space-1)',
  },
  linkCode: {
    overflow: 'hidden',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text-muted)',
    paddingTop: '9px',
    paddingBottom: '9px',
    paddingLeft: '10px',
    paddingRight: '10px',
    fontSize: 'var(--text-label-xs)',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  openLink: {
    display: 'inline-flex',
    minWidth: '44px',
    height: 'var(--control-compact)',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '6px',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    color: 'var(--color-action)',
    paddingLeft: '10px',
    paddingRight: '10px',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 600,
    textDecoration: 'none',
  },
  deviceCode: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, max-content) auto',
    alignItems: 'center',
    justifyContent: 'start',
    gap: 'var(--space-1)',
  },
  deviceCodeValue: {
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text)',
    paddingTop: '9px',
    paddingBottom: '9px',
    paddingLeft: '12px',
    paddingRight: '12px',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-body)',
    fontWeight: 700,
    letterSpacing: '0.08em',
    whiteSpace: 'nowrap',
  },
  callbackForm: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) auto',
    alignItems: 'start',
    gap: 'var(--space-2)',
    minWidth: 0,
  },
  monoInput: {
    fontFamily: 'var(--font-mono)',
  },
  /* 提交按钮与输入框顶端对齐：label 占一行，按钮下移同样的高度 */
  callbackSubmit: {
    marginTop: '22px',
  },
  entry: {
    display: 'grid',
    gap: 'var(--space-3)',
    minWidth: 0,
  },
  entryActions: {
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
  },
  json: {
    display: 'grid',
    gap: 'var(--space-3)',
  },
  jsonDisclosureSummary: {
    display: 'inline-flex',
    minWidth: 0,
    alignItems: 'center',
    gap: '8px',
  },
  jsonIcon: {
    flexShrink: 0,
    color: 'var(--color-action)',
  },
  jsonTextarea: {
    width: '100%',
    resize: 'vertical',
    fontFamily: 'var(--font-mono)',
    lineHeight: 1.55,
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text)',
    paddingTop: '8px',
    paddingBottom: '8px',
    paddingLeft: '10px',
    paddingRight: '10px',
    fontSize: 'var(--text-body)',
  },
  importActions: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
  },
  file: {
    display: 'inline-flex',
    minHeight: 'var(--control-md)',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '7px',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text-muted)',
    paddingLeft: '12px',
    paddingRight: '12px',
    fontSize: 'var(--text-button)',
    fontWeight: 560,
    cursor: 'pointer',
  },
  entryFile: {
    minHeight: 'var(--control-lg)',
    paddingLeft: '18px',
    paddingRight: '18px',
  },
  entryFileCompact: {
    minHeight: 'var(--control-sm)',
    paddingLeft: '12px',
    paddingRight: '12px',
    fontSize: 'var(--text-sm)',
  },
  entryFilePrimary: {
    borderColor: 'var(--color-action)',
    backgroundColor: 'var(--color-action)',
    color: 'var(--color-action-ink)',
    fontWeight: 600,
  },
  fileDisabled: {
    cursor: 'not-allowed',
    opacity: 0.46,
  },
  fileInput: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
  },
})

const toneStyles = stylex.create({
  success: { borderLeftColor: 'var(--color-success)' },
  warning: { borderLeftColor: 'var(--color-warning)' },
  danger: { borderLeftColor: 'var(--color-danger)' },
  neutral: { borderLeftColor: 'var(--color-border-control)' },
})
