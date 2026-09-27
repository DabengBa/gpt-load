import type { GatewayClientID } from '@shared/domain/home/gateway-clients'

// Framework-neutral port of the classic GatewayConnection sensitive-action
// state machine. Owns operation identity, abort control, busy/confirm/feedback
// state, and the feedback auto-clear timer; DOM-bound work (clipboard, reveal
// request, popup navigation) arrives through injected ops. Render code reads
// only `getSnapshot()` — React Compiler can freeze stable-reference method
// calls, so plain field getters are intentionally absent.

export type GatewayActionTarget =
  | 'key'
  | 'configuration'
  | 'quick-import'
  | 'baseUrl'
  | 'apiKey'

export type GatewayActionKind = 'success' | 'failure' | 'popup-blocked'

export type GatewayCopyOutcome = 'success' | 'fallback' | 'cancelled'

export interface GatewayActionIdentity {
  operationID: number
  accessKeyID: number
  clientID: GatewayClientID
}

export interface GatewayActionFeedback extends GatewayActionIdentity {
  target: GatewayActionTarget
  kind: GatewayActionKind
}

export interface GatewayActionsSnapshot {
  busy: boolean
  confirmOpen: boolean
  feedback: GatewayActionFeedback | null
}

export interface GatewayActionsInput {
  selectedAccessKeyID: number | null
  /** 选中项仍存在于已加载列表 —— 等价 classic 的 `selectedKey !== null`。 */
  selectedAccessKeyPresent: boolean
  clientID: GatewayClientID
  selfScoped: boolean
  credential?: string
}

export interface GatewayActionsOps {
  /** `/api/access-keys/:id/reveal` —— 管理员路径；AccessKey 会话直接用 credential。 */
  reveal(accessKeyID: number, signal: AbortSignal): Promise<string | undefined>
  /** 宿主把剪贴板对话框等瞬时状态一并清掉。 */
  onInvalidate(): void
}

export interface GatewayActionsController {
  subscribe(listener: () => void): () => void
  getSnapshot(): GatewayActionsSnapshot
  setInput(input: GatewayActionsInput): void
  setOps(ops: GatewayActionsOps): void
  setConfirmOpen(open: boolean): void
  /** 密钥/客户端/配置输入变化：终止进行中的敏感操作并清空反馈。 */
  invalidate(): void
  /** 选中项从已加载列表消失（immediate watch 语义）：invalidate + 关确认框。 */
  selectionLost(): void
  /** 访问密钥或客户端切换：invalidate + 关确认框。 */
  selectionChanged(): void
  /** 无需 reveal 的即时反馈（quick-import 弹窗被拦等）。 */
  reportImmediate(target: GatewayActionTarget, kind: GatewayActionKind): void
  /**
   * 取回密钥（管理员 reveal / AccessKey 会话 credential）后执行带身份守卫的
   * 敏感操作。操作返回 'fallback'/'cancelled' 时不报成功。
   */
  withRevealedKey(
    target: GatewayActionTarget,
    operation: (
      key: string,
      isCurrent: () => boolean,
    ) => Promise<GatewayCopyOutcome | void> | void,
  ): Promise<boolean>
  dispose(): void
}

export function createGatewayActionsController(
  feedbackDurationMs = 2_000,
): GatewayActionsController {
  let input: GatewayActionsInput = {
    selectedAccessKeyID: null,
    selectedAccessKeyPresent: false,
    clientID: 'cc-switch',
    selfScoped: false,
    credential: undefined,
  }
  let ops: GatewayActionsOps | null = null
  let busy = false
  let confirmOpen = false
  let feedback: GatewayActionFeedback | null = null
  let unmounted = false
  let operationSequence = 0
  let activeOperationID = 0
  let actionController: AbortController | undefined
  let feedbackTimer: ReturnType<typeof setTimeout> | undefined

  const listeners = new Set<() => void>()
  let snapshot: GatewayActionsSnapshot = { busy, confirmOpen, feedback }
  const notify = () => {
    snapshot = { busy, confirmOpen, feedback }
    for (const listener of listeners) listener()
  }

  function clearFeedbackTimer(): void {
    clearTimeout(feedbackTimer)
    feedbackTimer = undefined
  }

  function identityMatches(identity: GatewayActionIdentity): boolean {
    return (
      !unmounted &&
      activeOperationID === identity.operationID &&
      input.selectedAccessKeyID === identity.accessKeyID &&
      input.clientID === identity.clientID
    )
  }

  function operationIsCurrent(
    identity: GatewayActionIdentity,
    controller: AbortController,
  ): boolean {
    return (
      identityMatches(identity) &&
      actionController === controller &&
      !controller.signal.aborted
    )
  }

  function setFeedback(
    identity: GatewayActionIdentity,
    target: GatewayActionTarget,
    kind: GatewayActionKind,
  ): void {
    if (!identityMatches(identity)) return
    feedback = { ...identity, target, kind }
    notify()
    clearFeedbackTimer()
    feedbackTimer = setTimeout(() => {
      feedbackTimer = undefined
      if (feedback?.operationID === identity.operationID) {
        feedback = null
        notify()
      }
    }, feedbackDurationMs)
  }

  function invalidate(): void {
    ops?.onInvalidate()
    activeOperationID = ++operationSequence
    actionController?.abort()
    actionController = undefined
    if (busy || feedback !== null) {
      busy = false
      feedback = null
      notify()
    }
    clearFeedbackTimer()
  }

  async function withRevealedKey(
    target: GatewayActionTarget,
    operation: (
      key: string,
      isCurrent: () => boolean,
    ) => Promise<GatewayCopyOutcome | void> | void,
  ): Promise<boolean> {
    if (
      input.selectedAccessKeyID === null ||
      !input.selectedAccessKeyPresent ||
      busy ||
      unmounted ||
      !ops
    ) {
      return false
    }

    const identity: GatewayActionIdentity = {
      operationID: ++operationSequence,
      accessKeyID: input.selectedAccessKeyID,
      clientID: input.clientID,
    }
    const controller = new AbortController()
    activeOperationID = identity.operationID
    actionController = controller
    busy = true
    feedback = null
    notify()
    clearFeedbackTimer()

    let secret: string | undefined
    const isCurrent = () => operationIsCurrent(identity, controller)
    try {
      secret = input.selfScoped
        ? input.credential
        : await ops.reveal(identity.accessKeyID, controller.signal)
      if (!secret) throw new Error('ACCESS_KEY_UNAVAILABLE')
      if (!isCurrent()) return false
      const result = await operation(secret, isCurrent)
      if (!isCurrent()) return false
      if (result !== undefined && result !== 'success') return false
      setFeedback(identity, target, 'success')
      return true
    } catch {
      if (isCurrent()) setFeedback(identity, target, 'failure')
      return false
    } finally {
      secret = undefined
      if (actionController === controller) {
        actionController = undefined
        busy = false
        notify()
      }
    }
  }

  return {
    subscribe(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    getSnapshot: () => snapshot,
    setInput(next) {
      input = next
    },
    setOps(next) {
      ops = next
    },
    setConfirmOpen(open) {
      if (confirmOpen === open) return
      confirmOpen = open
      notify()
    },
    invalidate,
    selectionLost() {
      invalidate()
      if (confirmOpen) {
        confirmOpen = false
        notify()
      }
    },
    selectionChanged() {
      invalidate()
      if (confirmOpen) {
        confirmOpen = false
        notify()
      }
    },
    reportImmediate(target, kind) {
      if (input.selectedAccessKeyID === null || unmounted) return
      const identity: GatewayActionIdentity = {
        operationID: ++operationSequence,
        accessKeyID: input.selectedAccessKeyID,
        clientID: input.clientID,
      }
      activeOperationID = identity.operationID
      setFeedback(identity, target, kind)
    },
    withRevealedKey,
    dispose() {
      unmounted = true
      invalidate()
      listeners.clear()
    },
  }
}
