import type {
  RequestLogAttemptDto,
  RequestLogDetailDto,
  RequestLogItemDto,
  RequestLogReasoningDto,
} from '@/app/resources/request-logs'

export type RequestLogUsageDisplayState = 'reported' | 'missing' | 'not_applicable'
export type RequestLogCostDisplayState = 'complete' | 'unpriced' | 'not_applicable'

export function requestLogUsageDisplayState(log: RequestLogItemDto): RequestLogUsageDisplayState {
  if (log.usage_state === 'missing') return 'missing'
  if (log.usage_state === 'not_applicable') return 'not_applicable'
  return 'reported'
}

export function requestLogCostDisplayState(log: RequestLogItemDto): RequestLogCostDisplayState {
  if (log.cost_state === 'not_applicable') return 'not_applicable'
  if (log.cost_state === 'unpriced') return 'unpriced'
  return 'complete'
}

export function formatLogDuration(milliseconds: number): string {
  if (!Number.isSafeInteger(milliseconds) || milliseconds < 0) return '—'
  if (milliseconds < 1_000) return `${milliseconds}ms`
  if (milliseconds < 60_000) {
    const seconds = milliseconds / 1_000
    return `${seconds.toFixed(seconds < 10 ? 2 : 1).replace(/\.0+$/u, '')}s`
  }
  const totalSeconds = Math.round(milliseconds / 1_000)
  const hours = Math.floor(totalSeconds / 3_600)
  const minutes = Math.floor((totalSeconds % 3_600) / 60)
  const seconds = totalSeconds % 60
  if (hours > 0) {
    return `${hours}h${String(minutes).padStart(2, '0')}m${String(seconds).padStart(2, '0')}s`
  }
  return `${minutes}m${String(seconds).padStart(2, '0')}s`
}

export function formatLogTokenCount(value: string, locale: string): string {
  if (!/^(?:0|[1-9]\d*)$/u.test(value)) return '—'
  try {
    return new Intl.NumberFormat(locale, { maximumFractionDigits: 0 }).format(BigInt(value))
  } catch {
    return value
  }
}

export function formatLogReasoning(log: RequestLogItemDto, locale: string): string {
  return formatRequestLogReasoning(log.reasoning, locale)
}

export function formatRequestLogReasoning(
  value: RequestLogReasoningDto | null,
  locale: string,
): string {
  if (value === null) return ''
  const details: string[] = []
  if (
    value.mode !== null &&
    (value.mode !== 'enabled' || (value.effort === null && value.budget_tokens === null))
  ) {
    details.push(value.mode)
  }
  if (value.effort !== null) details.push(value.effort)
  if (value.budget_tokens !== null && value.budget_tokens !== '0') {
    if (value.budget_tokens === '-1') {
      if (value.effort === null) details.push('auto')
    } else {
      details.push(formatReasoningBudgetCompact(value.budget_tokens, locale))
    }
  }
  return details.join('/')
}

export function formatLogReasoningBudget(value: string, locale: string): string {
  return formatSignedInteger(value, locale)
}

export function reasoningBudgetSemantic(value: string): 'disabled' | 'dynamic' | null {
  if (value === '0') return 'disabled'
  if (value === '-1') return 'dynamic'
  return null
}

function formatReasoningBudgetCompact(value: string, locale: string): string {
  if (!/^(?:0|[1-9]\d*)$/u.test(value)) return '—'
  try {
    const amount = BigInt(value)
    if (amount < 1_000n) return formatSignedInteger(value, locale)
    if (amount < 1_000_000n) return `${amount / 1_000n}k`
    if (amount < 1_000_000_000n) return `${amount / 1_000_000n}m`
    return `${amount / 1_000_000_000n}b`
  } catch {
    return value
  }
}

function formatSignedInteger(value: string, locale: string): string {
  if (!/^(?:0|-?[1-9]\d*)$/u.test(value)) return '—'
  try {
    return new Intl.NumberFormat(locale, { maximumFractionDigits: 0 }).format(BigInt(value))
  } catch {
    return value
  }
}

export function formatLogOutputRate(log: RequestLogItemDto, locale: string): string {
  if (!log.stream || log.first_response_ms === null || log.duration_ms <= log.first_response_ms) {
    return '—'
  }
  const output = Number(log.output_tokens)
  if (!Number.isSafeInteger(output) || output <= 0) return '—'
  const rate = output / ((log.duration_ms - log.first_response_ms) / 1_000)
  if (!Number.isFinite(rate)) return '—'
  return `${new Intl.NumberFormat(locale, { maximumFractionDigits: 1 }).format(rate)} t/s`
}

export function hasRequestLogCache(log: RequestLogItemDto): boolean {
  return [
    log.cache_read_tokens,
    log.cache_write_5m_tokens,
    log.cache_write_1h_tokens,
    log.cache_write_unknown_tokens,
  ].some((value) => value !== '0')
}

/**
 * 路由链路上的实体（访问密钥 / 分组 / 凭据）在日志里的显示名。
 *
 * 日志是历史记录，实体随时可能已被删除，三者因此共用同一套回退：
 * 有名称就用名称，确认删除就标已删除，两者都不成立时只能给编号
 * （名称来源尚未加载，或该实体压根没有名称数据源）。
 *
 * 编号 0 是控制面观察（模型测活、凭据测活）占用的“没有实体”位，直接显示缺省符
 * ——否则会渲染出“#0”或“已删除 · #0”这种不存在的实体。
 */
export function formatRouteEntity(options: {
  id: number | null
  name: string | null | undefined
  deleted: boolean
  prefix: string
  deletedText: (id: number) => string
}): string {
  if (options.id === null || options.id === 0) return '—'
  const name = options.name?.trim()
  if (name) return name
  if (options.deleted) return options.deletedText(options.id)
  return `${options.prefix}${options.id}`
}

/** 关键原因的来源层级：明确终止的尝试、最后一个带原因的尝试，或请求级摘要。 */
export type RequestLogKeyReasonSource = 'terminal_attempt' | 'summary_attempt' | 'request'

export interface RequestLogKeyReason {
  /** 提供该原因的尝试序号；来自请求级摘要时为 null。 */
  sequence: number | null
  error_code: string
  error_summary: string
  source: RequestLogKeyReasonSource
}

function normalizeReasonText(value: string): string {
  return value.replace(/\s+/gu, ' ').trim()
}

/** 错误码或错误摘要任一非空，即表示该层存在可判读的原因。 */
export function hasRequestLogReason(value: { error_code: string; error_summary: string }): boolean {
  return value.error_code.trim() !== '' || value.error_summary.trim() !== ''
}

/**
 * 原因身份：优先使用规范化后的错误摘要，摘要为空时退回错误码。
 * 两个原因身份相同即视为同一个原因，请求级只汇总一次。
 * 摘要形态与仅错误码形态共享同一身份，因此同一原因不会跨形态重复展示。
 */
export function requestLogReasonIdentity(value: {
  error_code: string
  error_summary: string
}): string {
  const summary = normalizeReasonText(value.error_summary)
  return summary !== '' ? summary : normalizeReasonText(value.error_code)
}

/**
 * 尝试首屏展示的原因文本：优先摘要，摘要为空时降级为错误码。
 * 与请求级关键原因共用同一身份规则，保证「同一原因只展示一次」。
 */
export function requestLogAttemptReasonText(value: {
  error_code: string
  error_summary: string
}): string {
  const summary = value.error_summary.trim()
  return summary !== '' ? value.error_summary : value.error_code.trim()
}

/**
 * 最后一个明确终止请求的尝试。网关把“该尝试之后不再更换候选”的决定记录为
 * `action === 'terminate'`，因此它是关键原因的首选来源。全部尝试都还能重试时返回 null。
 */
export function terminalRequestLogAttempt(
  attempts: readonly RequestLogAttemptDto[],
): RequestLogAttemptDto | null {
  for (let index = attempts.length - 1; index >= 0; index -= 1) {
    if (attempts[index].action === 'terminate') return attempts[index]
  }
  return null
}

function latestRequestLogReasonAttempt(
  attempts: readonly RequestLogAttemptDto[],
): RequestLogAttemptDto | null {
  for (let index = attempts.length - 1; index >= 0; index -= 1) {
    if (normalizeReasonText(attempts[index].error_summary) !== '') return attempts[index]
  }
  return null
}

function reasonFromAttempt(
  attempt: RequestLogAttemptDto,
  source: RequestLogKeyReasonSource,
): RequestLogKeyReason {
  return {
    sequence: attempt.sequence,
    error_code: attempt.error_code,
    error_summary: attempt.error_summary,
    source,
  }
}

/**
 * 请求级关键原因，取值顺序固定为：
 * 1. 最后一个明确终止的尝试（`action === 'terminate'`，允许仅有错误码）；
 * 2. 最后一个规范化后摘要非空的尝试（只有错误码的中间尝试不属于这一级）；
 * 3. 请求级 `error_code` / `error_summary`。
 *
 * 成功请求不回退到已被成功覆盖的失败尝试，避免把重试过程中的旧错误当成结论；
 * 空白原因不构成关键原因。列表投影没有 attempts 字段，因此只会走到第 3 级。
 */
export function requestLogKeyReason(
  log: RequestLogItemDto | RequestLogDetailDto,
): RequestLogKeyReason | null {
  const candidates: RequestLogKeyReason[] = []
  if (log.status !== 'success') {
    const attempts = 'attempts' in log ? log.attempts : []
    const terminal = terminalRequestLogAttempt(attempts)
    if (terminal) candidates.push(reasonFromAttempt(terminal, 'terminal_attempt'))
    const latest = latestRequestLogReasonAttempt(attempts)
    if (latest) candidates.push(reasonFromAttempt(latest, 'summary_attempt'))
  }
  candidates.push({
    sequence: null,
    error_code: log.error_code,
    error_summary: log.error_summary,
    source: 'request',
  })

  for (const candidate of candidates) {
    if (hasRequestLogReason(candidate)) return candidate
  }
  return null
}

/**
 * 关键原因的错误码：可读原因文本已经表达了同一错误码时不再重复展示，
 * 保证相同原因在请求级只出现一次。
 */
export function requestLogKeyReasonCode(reason: RequestLogKeyReason | null): string {
  if (!reason) return ''
  const code = reason.error_code.trim()
  if (code === '') return ''
  return code === requestLogReasonIdentity(reason) ? '' : code
}

/**
 * 请求级去重后，仍需在尝试首屏展示原因的尝试序号。
 * 原因身份与请求级关键原因共享：与关键原因同身份的原因不再重复；多个尝试身份相同时
 * 只保留最早一个；错误码形态与摘要形态视为同一身份；没有任何原因的尝试不返回。
 */
export function requestLogAttemptReasonSequences(log: RequestLogDetailDto): Set<number> {
  const keyReason = requestLogKeyReason(log)
  const shown = new Set<string>()
  if (keyReason) shown.add(requestLogReasonIdentity(keyReason))
  const sequences = new Set<number>()
  for (const attempt of log.attempts) {
    const identity = requestLogReasonIdentity(attempt)
    if (identity === '') continue
    if (shown.has(identity)) continue
    shown.add(identity)
    sequences.add(attempt.sequence)
  }
  return sequences
}

/** 最小 i18n 翻译函数形状，避免 log-format 依赖具体 i18n 运行时。 */
export type RequestLogMessageTranslator = (
  key: string,
  named?: Record<string, string | number>,
) => string

/** 详情抽屉首屏展示的结论字段；标签与文本已本地化。 */
export interface RequestLogFirstScreen {
  status: string
  status_code: number
  attempt_count: number
  /** 关键原因标签；无关键原因时为空串。 */
  key_reason_label: string
  /** 关键原因可读文本；无关键原因时为空串。 */
  key_reason_text: string
  /** 关键原因错误码；与可读文本重复时为空串。 */
  key_reason_code: string
  /** 请求级最终错误码；关键原因已表达同一错误码时为空串。 */
  request_error_code: string
}

/**
 * 详情抽屉首屏投影：最终状态、尝试次数、关键原因（标签/文本/错误码），
 * 以及不与关键原因重复的请求级最终错误码。
 * 成功、取消、无 attempt 与无摘要状态都会得到一个稳定的形状（无原因时字段为空串）。
 */
export function requestLogFirstScreen(
  log: RequestLogDetailDto,
  t: RequestLogMessageTranslator,
): RequestLogFirstScreen {
  const reason = requestLogKeyReason(log)
  const keyReasonText = reason ? requestLogAttemptReasonText(reason) : ''
  const keyReasonLabel = reason
    ? reason.sequence === null
      ? t('monitor.logs.drawer.keyReason')
      : t('monitor.logs.drawer.keyReasonFromAttempt', { sequence: reason.sequence })
    : ''
  const requestCode = log.error_code.trim()
  const requestErrorCode =
    requestCode === '' ||
    (reason !== null && requestCode === reason.error_code.trim()) ||
    requestCode === keyReasonText.trim()
      ? ''
      : requestCode
  return {
    status: t(`monitor.logs.status.${log.status}`),
    status_code: log.status_code,
    attempt_count: log.attempt_count,
    key_reason_label: keyReasonLabel,
    key_reason_text: keyReasonText,
    key_reason_code: requestLogKeyReasonCode(reason),
    request_error_code: requestErrorCode,
  }
}

/** 列表 tooltip 是否值得展示：失败/取消/未完成一定有结论，成功只在发生重试时补充说明。 */
export function requestLogResponseTooltipVisible(log: RequestLogItemDto): boolean {
  return log.status !== 'success' || log.attempt_count > 1
}

/**
 * 列表 tooltip 首屏：最终状态、尝试次数、关键原因（附不同的错误码）。
 * 列表投影没有 attempts 字段，所以关键原因只能取到请求级摘要。
 */
export function requestLogResponseTooltip(
  log: RequestLogItemDto,
  t: RequestLogMessageTranslator,
): string {
  const status = t(`monitor.logs.status.${log.status}`)
  const lines = [
    t('monitor.logs.responseTooltip.status', {
      status: log.status_code > 0 ? `${status} · ${log.status_code}` : status,
    }),
    t('monitor.logs.responseTooltip.attempts', { count: log.attempt_count }),
  ]
  const reason = requestLogKeyReason(log)
  if (reason) {
    lines.push(
      t('monitor.logs.responseTooltip.reason', {
        reason: requestLogAttemptReasonText(reason),
      }),
    )
    const code = requestLogKeyReasonCode(reason)
    if (code !== '') lines.push(t('monitor.logs.responseTooltip.errorCode', { code }))
  }
  return lines.join('\n')
}
