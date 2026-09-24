import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'

// web/ is the parent of scripts/, used as vite root so the '@' alias resolves.
const WEB_ROOT = fileURLToPath(new URL('..', import.meta.url))

// 请求日志 DTO 的最小构造器：本脚本只验证纯函数，不加载真实项目数据。
function attempt(overrides = {}) {
  return {
    sequence: 1,
    group_id: 1,
    group_name: 'Group',
    channel_id: 'openai',
    credential_id: 1,
    credential_name: 'key-••••',
    operation: 'chat_completion',
    route_mode: 'native',
    upstream_model: 'gpt-x',
    upstream_request_id: null,
    dispatch_state: 'not_sent',
    response_started: true,
    upstream_protocol: 'openai',
    reasoning: null,
    status_code: 502,
    duration_ms: 12,
    failure_category: 'upstream_host_error',
    failure_origin: 'upstream',
    failure_scope: 'group',
    retry_directive: 'next_candidate',
    effect: 'none',
    rule_id: 'rule.1',
    action: 'terminate',
    will_retry: false,
    error_code: 'upstream_host_error',
    error_summary: 'upstream exploded',
    committed: false,
    pricing_receipt: null,
    ...overrides,
  }
}

function item(overrides = {}) {
  return {
    request_id: '00000000-0000-4000-8000-000000000000',
    status: 'error',
    status_code: 502,
    attempt_count: 1,
    error_code: 'upstream_host_error',
    error_summary: 'upstream exploded',
    ...overrides,
  }
}

function detail(overrides = {}, attempts = []) {
  return { ...item(overrides), attempts }
}

function retryAttempt(sequence, error_code, error_summary) {
  return attempt({ sequence, action: 'retry', will_retry: true, error_code, error_summary })
}

function terminalAttempt(sequence, error_code, error_summary, overrides = {}) {
  return attempt({ sequence, action: 'terminate', error_code, error_summary, ...overrides })
}

function reason(sequence, error_code, error_summary, source) {
  return { sequence, error_code, error_summary, source }
}

function reasonCase(name, log, expect) {
  return { name, run: (m) => m.requestLogKeyReason(log), expect }
}

function sequenceCase(name, log, expect, sorted = true) {
  return {
    name,
    run: (m) => {
      const sequences = [...m.requestLogAttemptReasonSequences(log)]
      return sorted ? sequences.sort((left, right) => left - right) : sequences
    },
    expect,
  }
}

function screenCase(name, log, expect) {
  return { name, run: (m, ctx) => m.requestLogFirstScreen(log, ctx.translate), expect }
}

function tooltipCase(name, log, expect) {
  return { name, run: (m, ctx) => m.requestLogResponseTooltip(log, ctx.translate), expect }
}

function reasonCodeCase(name, value, expect) {
  return { name, run: (m) => m.requestLogKeyReasonCode(value), expect }
}

function successfulRequest(attempts, overrides = {}) {
  return detail(
    { status: 'success', status_code: 200, error_code: '', error_summary: '', ...overrides },
    attempts,
  )
}

function successfulFinalAttempt() {
  return terminalAttempt(2, '', '', { failure_category: 'ok' })
}

function attemptSequences(m, log) {
  return [...m.requestLogAttemptReasonSequences(log)]
}

function errorItem(attempt_count, status_code, error_code, error_summary) {
  return item({ status: 'error', status_code, attempt_count, error_code, error_summary })
}

function requestWithoutAttempts(status, status_code, attempt_count, error_code, error_summary) {
  return detail({ status, status_code, attempt_count, error_code, error_summary })
}

function noCandidateDetail(attempt_count, attempts, error_summary = '') {
  return detail({ error_code: 'no_available_candidate', error_summary, attempt_count }, attempts)
}

function rateLimitThen(lastAttempt) {
  return detail({}, [retryAttempt(1, 'rate_limit_exceeded', 'was rate limited'), lastAttempt])
}

function codeOnlyRetries(firstSummary) {
  return detail({ error_code: 'no_available_candidate', error_summary: '' }, [
    retryAttempt(1, 'rate_limit_exceeded', firstSummary),
    retryAttempt(2, 'upstream_timeout', ''),
  ])
}

function firstScreenExpectation(status, status_code, attempt_count, reason = {}) {
  return {
    status,
    status_code,
    attempt_count,
    key_reason_label: '',
    key_reason_text: '',
    key_reason_code: '',
    request_error_code: '',
    ...reason,
  }
}

const CASES = [
  reasonCase(
    'terminal attempt wins over an earlier retried summary',
    rateLimitThen(terminalAttempt(2, 'upstream_host_error', 'upstream exploded')),
    reason(2, 'upstream_host_error', 'upstream exploded', 'terminal_attempt'),
  ),
  {
    name: 'terminal attempt identity is the last terminate action',
    run: (m) => {
      const log = detail({}, [
        attempt({ sequence: 1, action: 'terminate' }),
        attempt({ sequence: 2, action: 'retry', will_retry: true }),
        attempt({ sequence: 3, action: 'terminate' }),
      ])
      return m.terminalRequestLogAttempt(log.attempts)?.sequence ?? null
    },
    expect: 3,
  },
  reasonCase(
    'no terminating attempt falls back to the last attempt with a reason',
    rateLimitThen(
      attempt({
        sequence: 2,
        action: 'retry',
        will_retry: false,
        error_code: '',
        error_summary: '',
      }),
    ),
    reason(1, 'rate_limit_exceeded', 'was rate limited', 'summary_attempt'),
  ),
  reasonCase(
    'terminal attempt without a reason falls back to the last summary attempt',
    rateLimitThen(terminalAttempt(2, '', '')),
    reason(1, 'rate_limit_exceeded', 'was rate limited', 'summary_attempt'),
  ),
  reasonCase(
    'request level reason is the fallback when there are no attempts',
    detail({ error_code: 'no_available_candidate', error_summary: 'was rate limited' }),
    reason(null, 'no_available_candidate', 'was rate limited', 'request'),
  ),
  reasonCase(
    'request level reason is the fallback for the list projection without attempts',
    item({ error_code: 'invalid_key', error_summary: 'supplier rejected the key' }),
    reason(null, 'invalid_key', 'supplier rejected the key', 'request'),
  ),
  {
    name: 'success never reports a key reason from a retried failure',
    run: (m) => {
      const log = successfulRequest([
        attempt({
          sequence: 1,
          action: 'retry',
          will_retry: true,
          error_summary: 'was rate limited',
        }),
        successfulFinalAttempt(),
      ])
      return m.requestLogKeyReason(log)
    },
    expect: null,
  },
  {
    name: 'success still surfaces an explicit request level summary',
    run: (m) => {
      const log = detail({
        status: 'success',
        status_code: 200,
        error_code: 'partial',
        error_summary: 'usage incomplete',
      })
      return m.requestLogKeyReason(log)
    },
    expect: reason(null, 'partial', 'usage incomplete', 'request'),
  },
  {
    name: 'empty reasons are never a key reason',
    run: (m) => {
      const log = detail({ error_code: '  ', error_summary: '\n' }, [
        attempt({ error_code: '', error_summary: '   ' }),
      ])
      return m.requestLogKeyReason(log)
    },
    expect: null,
  },
  {
    name: 'reason identity prefers the normalized summary',
    run: (m) =>
      m.requestLogReasonIdentity({
        error_code: 'rate_limit_exceeded',
        error_summary: '  was\nrate limited ',
      }),
    expect: 'was rate limited',
  },
  {
    name: 'reason identity falls back to the normalized error code',
    run: (m) =>
      m.requestLogReasonIdentity({ error_code: ' upstream_host_error ', error_summary: '  ' }),
    expect: 'upstream_host_error',
  },
  {
    name: 'hasRequestLogReason rejects whitespace only reasons',
    run: (m) => [
      m.hasRequestLogReason({ error_code: 'x', error_summary: '' }),
      m.hasRequestLogReason({ error_code: ' ', error_summary: '\t' }),
    ],
    expect: [true, false],
  },
  reasonCodeCase(
    'key reason code is hidden when the summary already shows the same text',
    reason(1, 'rate_limit_exceeded', 'rate_limit_exceeded', 'terminal_attempt'),
    '',
  ),
  reasonCodeCase(
    'key reason code is shown when it adds information',
    reason(1, 'rate_limit_exceeded', 'was rate limited', 'terminal_attempt'),
    'rate_limit_exceeded',
  ),
  reasonCodeCase(
    'key reason code is hidden when there is no code',
    reason(null, ' ', 'upstream exploded', 'request'),
    '',
  ),
  sequenceCase(
    'identical reasons are aggregated once at request level',
    detail({}, [
      retryAttempt(1, 'rate_limit_exceeded', 'was rate limited'),
      retryAttempt(2, 'rate_limit_exceeded', 'was rate limited'),
      terminalAttempt(3, 'upstream_host_error', 'upstream exploded'),
    ]),
    [1],
  ),
  sequenceCase(
    'attempt reason equal to the key reason is not repeated',
    detail({}, [terminalAttempt(1, 'a', 'first reason'), terminalAttempt(2, 'b', 'key reason')]),
    [1],
  ),
  sequenceCase(
    'request level reason aggregates identical attempt reasons once',
    detail({ error_code: 'no_available_candidate', error_summary: 'was rate limited' }, [
      retryAttempt(1, 'rate_limit_exceeded', 'was rate limited'),
      retryAttempt(2, 'rate_limit_exceeded', 'was rate limited'),
    ]),
    [],
  ),
  sequenceCase(
    'repeated non key reasons are shown once per distinct reason',
    detail({}, [
      terminalAttempt(1, 'rate_limit_exceeded', 'was rate limited'),
      terminalAttempt(2, 'upstream_timeout', 'upstream timed out'),
      terminalAttempt(3, 'rate_limit_exceeded', 'was rate limited'),
      terminalAttempt(4, 'upstream_host_error', 'upstream exploded'),
    ]),
    [1, 2],
  ),
  sequenceCase(
    'attempts without any reason are not listed',
    detail({}, [terminalAttempt(1, '', ''), terminalAttempt(2, '', '')]),
    [],
    false,
  ),
  tooltipCase(
    'list tooltip shows final status, attempts and the request level key reason',
    errorItem(3, 502, 'no_available_candidate', 'The upstream account was rate limited.'),
    '最终状态：错误 · 502\n尝试次数：3\n关键原因：The upstream account was rate limited.\n错误码：no_available_candidate',
  ),
  tooltipCase(
    'list tooltip collapses a key reason whose code is the same text',
    errorItem(2, 429, 'rate_limit_exceeded', 'rate_limit_exceeded'),
    '最终状态：错误 · 429\n尝试次数：2\n关键原因：rate_limit_exceeded',
  ),
  tooltipCase(
    'list tooltip omits a zero status code',
    item({
      status: 'canceled',
      status_code: 0,
      attempt_count: 1,
      error_code: 'client_canceled',
      error_summary: '',
    }),
    '最终状态：已取消\n尝试次数：1\n关键原因：client_canceled',
  ),
  {
    name: 'list tooltip is hidden for a single attempt success',
    run: (m) =>
      m.requestLogResponseTooltipVisible(
        item({
          status: 'success',
          status_code: 200,
          attempt_count: 1,
          error_code: '',
          error_summary: '',
        }),
      ),
    expect: false,
  },
  {
    name: 'list tooltip still explains a successful request that retried',
    run: (m, ctx) => [
      m.requestLogResponseTooltipVisible(
        item({ status: 'success', status_code: 200, attempt_count: 2 }),
      ),
      m.requestLogResponseTooltip(
        item({
          status: 'success',
          status_code: 200,
          attempt_count: 2,
          error_code: '',
          error_summary: '',
        }),
        ctx.translate,
      ),
    ],
    expect: [true, '最终状态：成功 · 200\n尝试次数：2'],
  },
  screenCase(
    'drawer first screen: retried failure highlights status, attempts and the terminating attempt',
    detail(
      {
        status_code: 502,
        attempt_count: 3,
        error_code: 'no_available_candidate',
        error_summary: 'upstream exploded',
      },
      [
        retryAttempt(1, 'rate_limit_exceeded', 'was rate limited'),
        retryAttempt(2, 'rate_limit_exceeded', 'was rate limited'),
        terminalAttempt(3, 'upstream_host_error', 'upstream exploded'),
      ],
    ),
    firstScreenExpectation('错误', 502, 3, {
      key_reason_label: '关键原因 · 尝试 #3',
      key_reason_text: 'upstream exploded',
      key_reason_code: 'upstream_host_error',
      request_error_code: 'no_available_candidate',
    }),
  ),
  screenCase(
    'drawer first screen: single attempt failure repeats neither code nor reason',
    detail(
      {
        status_code: 429,
        attempt_count: 1,
        error_code: 'rate_limit_exceeded',
        error_summary: 'was rate limited',
      },
      [terminalAttempt(1, 'rate_limit_exceeded', 'was rate limited', { status_code: 429 })],
    ),
    firstScreenExpectation('错误', 429, 1, {
      key_reason_label: '关键原因 · 尝试 #1',
      key_reason_text: 'was rate limited',
      key_reason_code: 'rate_limit_exceeded',
    }),
  ),
  screenCase(
    'drawer first screen: success after a retry hides the covered failure',
    successfulRequest(
      [
        attempt({
          sequence: 1,
          action: 'retry',
          will_retry: true,
          error_summary: 'was rate limited',
        }),
        successfulFinalAttempt(),
      ],
      { attempt_count: 2 },
    ),
    firstScreenExpectation('成功', 200, 2),
  ),
  screenCase(
    'drawer first screen: canceled request falls back to the code alone',
    requestWithoutAttempts('canceled', 0, 1, 'client_canceled', ''),
    firstScreenExpectation('已取消', 0, 1, {
      key_reason_label: '关键原因',
      key_reason_text: 'client_canceled',
    }),
  ),
  screenCase(
    'drawer first screen: no attempts falls back to the request level reason',
    detail({
      status_code: 503,
      attempt_count: 0,
      error_code: 'no_available_candidate',
      error_summary: 'all candidates failed',
    }),
    firstScreenExpectation('错误', 503, 0, {
      key_reason_label: '关键原因',
      key_reason_text: 'all candidates failed',
      key_reason_code: 'no_available_candidate',
    }),
  ),
  screenCase(
    'drawer first screen: incomplete without any reason stays empty',
    requestWithoutAttempts('incomplete', 200, 0, '', ''),
    firstScreenExpectation('未完成', 200, 0),
  ),
  reasonCase(
    'middle code-only attempt never becomes the request level key reason',
    codeOnlyRetries('was rate limited'),
    reason(1, 'rate_limit_exceeded', 'was rate limited', 'summary_attempt'),
  ),
  reasonCase(
    'all code-only attempts fall back to the request level reason',
    codeOnlyRetries(''),
    reason(null, 'no_available_candidate', '', 'request'),
  ),
  {
    name: 'terminal code-only attempt keeps the key reason while sibling code-only attempts keep theirs',
    run: (m, ctx) => {
      const log = noCandidateDetail(3, [
        retryAttempt(1, 'rate_limit_exceeded', ''),
        retryAttempt(2, 'upstream_timeout', ''),
        terminalAttempt(3, 'upstream_host_error', ''),
      ])
      return {
        reason: m.requestLogKeyReason(log),
        sequences: attemptSequences(m, log).sort((left, right) => left - right),
        firstScreen: m.requestLogFirstScreen(log, ctx.translate),
      }
    },
    expect: {
      reason: reason(3, 'upstream_host_error', '', 'terminal_attempt'),
      sequences: [1, 2],
      firstScreen: firstScreenExpectation('错误', 502, 3, {
        key_reason_label: '关键原因 · 尝试 #3',
        key_reason_text: 'upstream_host_error',
        request_error_code: 'no_available_candidate',
      }),
    },
  },
  {
    name: 'code-only and summary forms of the same identity are shown once',
    run: (m) => {
      const log = detail({ error_code: 'no_available_candidate', error_summary: '' }, [
        retryAttempt(1, 'rate_limit_exceeded', ''),
        retryAttempt(2, 'rate_limit_exceeded', 'rate_limit_exceeded'),
        terminalAttempt(3, 'upstream_host_error', 'upstream exploded'),
      ])
      return {
        sequences: [...m.requestLogAttemptReasonSequences(log)].sort((left, right) => left - right),
        codeOnlyText: m.requestLogAttemptReasonText(log.attempts[0]),
        summaryText: m.requestLogAttemptReasonText(log.attempts[1]),
      }
    },
    expect: {
      sequences: [1],
      codeOnlyText: 'rate_limit_exceeded',
      summaryText: 'rate_limit_exceeded',
    },
  },
  {
    name: 'whitespace-only attempt summary is not a fallback summary',
    run: (m) => {
      const log = detail({ error_code: 'no_available_candidate', error_summary: '' }, [
        retryAttempt(1, 'rate_limit_exceeded', '  \n '),
      ])
      return {
        reason: m.requestLogKeyReason(log),
        sequences: [...m.requestLogAttemptReasonSequences(log)],
        text: m.requestLogAttemptReasonText(log.attempts[0]),
      }
    },
    expect: {
      reason: {
        sequence: null,
        error_code: 'no_available_candidate',
        error_summary: '',
        source: 'request',
      },
      sequences: [1],
      text: 'rate_limit_exceeded',
    },
  },
  {
    name: 'code-only attempt first screen stays consistent with the request level reason',
    run: (m, ctx) => {
      const log = noCandidateDetail(2, [
        retryAttempt(1, 'rate_limit_exceeded', ''),
        terminalAttempt(2, 'upstream_host_error', 'upstream exploded'),
      ])
      return {
        firstScreen: m.requestLogFirstScreen(log, ctx.translate),
        sequences: [...m.requestLogAttemptReasonSequences(log)],
        codeOnlyText: m.requestLogAttemptReasonText(log.attempts[0]),
      }
    },
    expect: {
      firstScreen: firstScreenExpectation('错误', 502, 2, {
        key_reason_label: '关键原因 · 尝试 #2',
        key_reason_text: 'upstream exploded',
        key_reason_code: 'upstream_host_error',
        request_error_code: 'no_available_candidate',
      }),
      sequences: [1],
      codeOnlyText: 'rate_limit_exceeded',
    },
  },
  {
    name: 'success never falls back to a code-only attempt reason',
    run: (m) => {
      const log = successfulRequest([
        attempt({
          sequence: 1,
          action: 'retry',
          will_retry: true,
          failure_category: 'upstream_host_error',
          error_code: 'upstream_host_error',
          error_summary: '',
        }),
        successfulFinalAttempt(),
      ])
      return m.requestLogKeyReason(log)
    },
    expect: null,
  },
]

function createTranslator(messages) {
  return (key, named = {}) => {
    let node = messages
    for (const part of key.split('.')) {
      node = node?.[part]
    }
    if (typeof node !== 'string') throw new Error(`missing message: ${key}`)
    return node.replace(/\{(\w+)\}/gu, (_, name) => String(named[name] ?? `{${name}}`))
  }
}

const DASHBOARD_KEYS = [
  'monitor.logs.drawer.keyReason',
  'monitor.logs.drawer.keyReasonFromAttempt',
  'monitor.logs.drawer.requestErrorCode',
  'monitor.logs.drawer.gatewayAction',
  'monitor.logs.drawer.attemptDetails',
  'monitor.logs.responseTooltip.status',
  'monitor.logs.responseTooltip.attempts',
  'monitor.logs.responseTooltip.reason',
  'monitor.logs.responseTooltip.errorCode',
]

// 三套文案必须同时提供新键，否则侧栏会回退到键名。
function checkLabelParity(messagesByLocale) {
  const missing = []
  for (const [locale, messages] of Object.entries(messagesByLocale)) {
    const translate = createTranslator(messages)
    for (const key of DASHBOARD_KEYS) {
      try {
        if (translate(key, { sequence: 1 }).trim() === '') missing.push(`${locale}:${key}`)
      } catch {
        missing.push(`${locale}:${key}`)
      }
    }
  }
  return missing
}

async function main() {
  const server = await createServer({
    root: WEB_ROOT,
    server: { middlewareMode: true },
    appType: 'custom',
    logLevel: 'error',
  })

  let failed = 0
  try {
    let format
    try {
      format = await server.ssrLoadModule('/src/frontends/classic/features/monitor/log-format.ts')
    } catch (err) {
      console.log(`FAIL  module load: ${err instanceof Error ? err.message : String(err)}`)
      return 1
    }

    const localeNames = ['zh-CN', 'en-US', 'ja-JP']
    const messagesByLocale = {}
    for (const locale of localeNames) {
      const module = await server.ssrLoadModule(
        `/src/frontends/classic/i18n/locales/${locale}/monitor.ts`,
      )
      messagesByLocale[locale] = module.default
    }
    const context = { translate: createTranslator(messagesByLocale['zh-CN']) }

    for (const testCase of CASES) {
      let actual
      try {
        actual = testCase.run(format, context)
      } catch (err) {
        failed++
        console.log(
          `FAIL  ${testCase.name}: threw ${err instanceof Error ? err.message : String(err)}`,
        )
        continue
      }
      const ok = JSON.stringify(actual) === JSON.stringify(testCase.expect)
      if (ok) {
        console.log(`PASS  ${testCase.name}`)
      } else {
        failed++
        console.log(
          `FAIL  ${testCase.name}: expected=${JSON.stringify(testCase.expect)} actual=${JSON.stringify(actual)}`,
        )
      }
    }

    const missingLabels = checkLabelParity(messagesByLocale)
    if (missingLabels.length === 0) {
      console.log('PASS  first-screen labels exist in zh-CN, en-US and ja-JP')
    } else {
      failed++
      console.log(`FAIL  missing first-screen labels: ${missingLabels.join(', ')}`)
    }
  } finally {
    await server.close()
  }

  return failed
}

const failed = await main()
if (failed > 0) {
  console.log(`\n${failed} log-sidebar case(s) failed`)
  process.exitCode = 1
} else {
  console.log(`\nall ${CASES.length} log-format cases and the label parity check passed`)
}
