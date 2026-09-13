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

const CASES = [
  {
    name: 'terminal attempt wins over an earlier retried summary',
    run: (m) => {
      const log = detail({}, [
        attempt({ sequence: 1, action: 'retry', will_retry: true, error_code: 'rate_limit_exceeded', error_summary: 'was rate limited' }),
        attempt({ sequence: 2, action: 'terminate', error_code: 'upstream_host_error', error_summary: 'upstream exploded' }),
      ])
      return m.requestLogKeyReason(log)
    },
    expect: { sequence: 2, error_code: 'upstream_host_error', error_summary: 'upstream exploded', source: 'terminal_attempt' },
  },
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
  {
    name: 'no terminating attempt falls back to the last attempt with a reason',
    run: (m) => {
      const log = detail({}, [
        attempt({ sequence: 1, action: 'retry', will_retry: true, error_code: 'rate_limit_exceeded', error_summary: 'was rate limited' }),
        attempt({ sequence: 2, action: 'retry', will_retry: false, error_code: '', error_summary: '' }),
      ])
      return m.requestLogKeyReason(log)
    },
    expect: { sequence: 1, error_code: 'rate_limit_exceeded', error_summary: 'was rate limited', source: 'summary_attempt' },
  },
  {
    name: 'terminal attempt without a reason falls back to the last summary attempt',
    run: (m) => {
      const log = detail({}, [
        attempt({ sequence: 1, action: 'retry', will_retry: true, error_code: 'rate_limit_exceeded', error_summary: 'was rate limited' }),
        attempt({ sequence: 2, action: 'terminate', error_code: '', error_summary: '' }),
      ])
      return m.requestLogKeyReason(log)
    },
    expect: { sequence: 1, error_code: 'rate_limit_exceeded', error_summary: 'was rate limited', source: 'summary_attempt' },
  },
  {
    name: 'request level reason is the fallback when there are no attempts',
    run: (m) => {
      const log = detail({ error_code: 'no_available_candidate', error_summary: 'was rate limited' }, [])
      return m.requestLogKeyReason(log)
    },
    expect: { sequence: null, error_code: 'no_available_candidate', error_summary: 'was rate limited', source: 'request' },
  },
  {
    name: 'request level reason is the fallback for the list projection without attempts',
    run: (m) => {
      const log = item({ error_code: 'invalid_key', error_summary: 'supplier rejected the key' })
      return m.requestLogKeyReason(log)
    },
    expect: { sequence: null, error_code: 'invalid_key', error_summary: 'supplier rejected the key', source: 'request' },
  },
  {
    name: 'success never reports a key reason from a retried failure',
    run: (m) => {
      const log = detail({ status: 'success', status_code: 200, error_code: '', error_summary: '' }, [
        attempt({ sequence: 1, action: 'retry', will_retry: true, error_summary: 'was rate limited' }),
        attempt({ sequence: 2, action: 'terminate', failure_category: 'ok', error_code: '', error_summary: '' }),
      ])
      return m.requestLogKeyReason(log)
    },
    expect: null,
  },
  {
    name: 'success still surfaces an explicit request level summary',
    run: (m) => {
      const log = detail({ status: 'success', status_code: 200, error_code: 'partial', error_summary: 'usage incomplete' })
      return m.requestLogKeyReason(log)
    },
    expect: { sequence: null, error_code: 'partial', error_summary: 'usage incomplete', source: 'request' },
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
      m.requestLogReasonIdentity({ error_code: 'rate_limit_exceeded', error_summary: '  was\nrate limited ' }),
    expect: 'was rate limited',
  },
  {
    name: 'reason identity falls back to the normalized error code',
    run: (m) => m.requestLogReasonIdentity({ error_code: ' upstream_host_error ', error_summary: '  ' }),
    expect: 'upstream_host_error',
  },
  {
    name: 'hasRequestLogReason rejects whitespace only reasons',
    run: (m) => [m.hasRequestLogReason({ error_code: 'x', error_summary: '' }), m.hasRequestLogReason({ error_code: ' ', error_summary: '\t' })],
    expect: [true, false],
  },
  {
    name: 'key reason code is hidden when the summary already shows the same text',
    run: (m) =>
      m.requestLogKeyReasonCode({ sequence: 1, error_code: 'rate_limit_exceeded', error_summary: 'rate_limit_exceeded', source: 'terminal_attempt' }),
    expect: '',
  },
  {
    name: 'key reason code is shown when it adds information',
    run: (m) =>
      m.requestLogKeyReasonCode({ sequence: 1, error_code: 'rate_limit_exceeded', error_summary: 'was rate limited', source: 'terminal_attempt' }),
    expect: 'rate_limit_exceeded',
  },
  {
    name: 'key reason code is hidden when there is no code',
    run: (m) =>
      m.requestLogKeyReasonCode({ sequence: null, error_code: ' ', error_summary: 'upstream exploded', source: 'request' }),
    expect: '',
  },
  {
    name: 'identical reasons are aggregated once at request level',
    run: (m) => {
      const log = detail({}, [
        attempt({ sequence: 1, action: 'retry', will_retry: true, error_code: 'rate_limit_exceeded', error_summary: 'was rate limited' }),
        attempt({ sequence: 2, action: 'retry', will_retry: true, error_code: 'rate_limit_exceeded', error_summary: 'was rate limited' }),
        attempt({ sequence: 3, action: 'terminate', error_code: 'upstream_host_error', error_summary: 'upstream exploded' }),
      ])
      return [...m.requestLogAttemptReasonSequences(log)].sort((left, right) => left - right)
    },
    expect: [1],
  },
  {
    name: 'attempt reason equal to the key reason is not repeated',
    run: (m) => {
      const log = detail({}, [
        attempt({ sequence: 1, error_code: 'a', error_summary: 'first reason' }),
        attempt({ sequence: 2, error_code: 'b', error_summary: 'key reason' }),
      ])
      return [...m.requestLogAttemptReasonSequences(log)].sort((left, right) => left - right)
    },
    expect: [1],
  },
  {
    name: 'request level reason aggregates identical attempt reasons once',
    run: (m) => {
      const log = detail({ error_code: 'no_available_candidate', error_summary: 'was rate limited' }, [
        attempt({ sequence: 1, action: 'retry', will_retry: true, error_code: 'rate_limit_exceeded', error_summary: 'was rate limited' }),
        attempt({ sequence: 2, action: 'retry', will_retry: true, error_code: 'rate_limit_exceeded', error_summary: 'was rate limited' }),
      ])
      return [...m.requestLogAttemptReasonSequences(log)].sort((left, right) => left - right)
    },
    expect: [],
  },
  {
    name: 'repeated non key reasons are shown once per distinct reason',
    run: (m) => {
      const log = detail({}, [
        attempt({ sequence: 1, error_code: 'rate_limit_exceeded', error_summary: 'was rate limited' }),
        attempt({ sequence: 2, error_code: 'upstream_timeout', error_summary: 'upstream timed out' }),
        attempt({ sequence: 3, error_code: 'rate_limit_exceeded', error_summary: 'was rate limited' }),
        attempt({ sequence: 4, action: 'terminate', error_code: 'upstream_host_error', error_summary: 'upstream exploded' }),
      ])
      return [...m.requestLogAttemptReasonSequences(log)].sort((left, right) => left - right)
    },
    expect: [1, 2],
  },
  {
    name: 'attempts without any reason are not listed',
    run: (m) => {
      const log = detail({}, [
        attempt({ sequence: 1, error_code: '', error_summary: '' }),
        attempt({ sequence: 2, error_code: '', error_summary: '' }),
      ])
      return [...m.requestLogAttemptReasonSequences(log)]
    },
    expect: [],
  },
  {
    name: 'list tooltip shows final status, attempts and the request level key reason',
    run: (m, ctx) =>
      m.requestLogResponseTooltip(
        item({
          status: 'error',
          status_code: 502,
          attempt_count: 3,
          error_code: 'no_available_candidate',
          error_summary: 'The upstream account was rate limited.',
        }),
        ctx.translate,
      ),
    expect: '最终状态：错误 · 502\n尝试次数：3\n关键原因：The upstream account was rate limited.\n错误码：no_available_candidate',
  },
  {
    name: 'list tooltip collapses a key reason whose code is the same text',
    run: (m, ctx) =>
      m.requestLogResponseTooltip(
        item({ status_code: 429, attempt_count: 2, error_code: 'rate_limit_exceeded', error_summary: 'rate_limit_exceeded' }),
        ctx.translate,
      ),
    expect: '最终状态：错误 · 429\n尝试次数：2\n关键原因：rate_limit_exceeded',
  },
  {
    name: 'list tooltip omits a zero status code',
    run: (m, ctx) =>
      m.requestLogResponseTooltip(
        item({ status: 'canceled', status_code: 0, attempt_count: 1, error_code: 'client_canceled', error_summary: '' }),
        ctx.translate,
      ),
    expect: '最终状态：已取消\n尝试次数：1\n关键原因：client_canceled',
  },
  {
    name: 'list tooltip is hidden for a single attempt success',
    run: (m) =>
      m.requestLogResponseTooltipVisible(
        item({ status: 'success', status_code: 200, attempt_count: 1, error_code: '', error_summary: '' }),
      ),
    expect: false,
  },
  {
    name: 'list tooltip still explains a successful request that retried',
    run: (m, ctx) => [
      m.requestLogResponseTooltipVisible(item({ status: 'success', status_code: 200, attempt_count: 2 })),
      m.requestLogResponseTooltip(item({ status: 'success', status_code: 200, attempt_count: 2, error_code: '', error_summary: '' }), ctx.translate),
    ],
    expect: [
      true,
      '最终状态：成功 · 200\n尝试次数：2',
    ],
  },
  {
    name: 'drawer first screen: retried failure highlights status, attempts and the terminating attempt',
    run: (m, ctx) =>
      m.requestLogFirstScreen(
        detail(
          { status_code: 502, attempt_count: 3, error_code: 'no_available_candidate', error_summary: 'upstream exploded' },
          [
            attempt({ sequence: 1, action: 'retry', will_retry: true, error_code: 'rate_limit_exceeded', error_summary: 'was rate limited' }),
            attempt({ sequence: 2, action: 'retry', will_retry: true, error_code: 'rate_limit_exceeded', error_summary: 'was rate limited' }),
            attempt({ sequence: 3, action: 'terminate', error_code: 'upstream_host_error', error_summary: 'upstream exploded' }),
          ],
        ),
        ctx.translate,
      ),
    expect: {
      status: '错误',
      status_code: 502,
      attempt_count: 3,
      key_reason_label: '关键原因 · 尝试 #3',
      key_reason_text: 'upstream exploded',
      key_reason_code: 'upstream_host_error',
      request_error_code: 'no_available_candidate',
    },
  },
  {
    name: 'drawer first screen: single attempt failure repeats neither code nor reason',
    run: (m, ctx) =>
      m.requestLogFirstScreen(
        detail(
          { status_code: 429, attempt_count: 1, error_code: 'rate_limit_exceeded', error_summary: 'was rate limited' },
          [
            attempt({ sequence: 1, action: 'terminate', status_code: 429, error_code: 'rate_limit_exceeded', error_summary: 'was rate limited' }),
          ],
        ),
        ctx.translate,
      ),
    expect: {
      status: '错误',
      status_code: 429,
      attempt_count: 1,
      key_reason_label: '关键原因 · 尝试 #1',
      key_reason_text: 'was rate limited',
      key_reason_code: 'rate_limit_exceeded',
      request_error_code: '',
    },
  },
  {
    name: 'drawer first screen: success after a retry hides the covered failure',
    run: (m, ctx) =>
      m.requestLogFirstScreen(
        detail(
          { status: 'success', status_code: 200, attempt_count: 2, error_code: '', error_summary: '' },
          [
            attempt({ sequence: 1, action: 'retry', will_retry: true, error_summary: 'was rate limited' }),
            attempt({ sequence: 2, action: 'terminate', failure_category: 'ok', error_code: '', error_summary: '' }),
          ],
        ),
        ctx.translate,
      ),
    expect: {
      status: '成功',
      status_code: 200,
      attempt_count: 2,
      key_reason_label: '',
      key_reason_text: '',
      key_reason_code: '',
      request_error_code: '',
    },
  },
  {
    name: 'drawer first screen: canceled request falls back to the code alone',
    run: (m, ctx) =>
      m.requestLogFirstScreen(
        detail({ status: 'canceled', status_code: 0, attempt_count: 1, error_code: 'client_canceled', error_summary: '' }, []),
        ctx.translate,
      ),
    expect: {
      status: '已取消',
      status_code: 0,
      attempt_count: 1,
      key_reason_label: '关键原因',
      key_reason_text: 'client_canceled',
      key_reason_code: '',
      request_error_code: '',
    },
  },
  {
    name: 'drawer first screen: no attempts falls back to the request level reason',
    run: (m, ctx) =>
      m.requestLogFirstScreen(
        detail({ status_code: 503, attempt_count: 0, error_code: 'no_available_candidate', error_summary: 'all candidates failed' }, []),
        ctx.translate,
      ),
    expect: {
      status: '错误',
      status_code: 503,
      attempt_count: 0,
      key_reason_label: '关键原因',
      key_reason_text: 'all candidates failed',
      key_reason_code: 'no_available_candidate',
      request_error_code: '',
    },
  },
  {
    name: 'drawer first screen: incomplete without any reason stays empty',
    run: (m, ctx) =>
      m.requestLogFirstScreen(
        detail({ status: 'incomplete', status_code: 200, attempt_count: 0, error_code: '', error_summary: '' }, []),
        ctx.translate,
      ),
    expect: {
      status: '未完成',
      status_code: 200,
      attempt_count: 0,
      key_reason_label: '',
      key_reason_text: '',
      key_reason_code: '',
      request_error_code: '',
    },
  },
  {
    name: 'middle code-only attempt never becomes the request level key reason',
    run: (m) => {
      const log = detail({ error_code: 'no_available_candidate', error_summary: '' }, [
        attempt({
          sequence: 1,
          action: 'retry',
          will_retry: true,
          error_code: 'rate_limit_exceeded',
          error_summary: 'was rate limited',
        }),
        attempt({
          sequence: 2,
          action: 'retry',
          will_retry: true,
          error_code: 'upstream_timeout',
          error_summary: '',
        }),
      ])
      return m.requestLogKeyReason(log)
    },
    expect: {
      sequence: 1,
      error_code: 'rate_limit_exceeded',
      error_summary: 'was rate limited',
      source: 'summary_attempt',
    },
  },
  {
    name: 'all code-only attempts fall back to the request level reason',
    run: (m) => {
      const log = detail({ error_code: 'no_available_candidate', error_summary: '' }, [
        attempt({
          sequence: 1,
          action: 'retry',
          will_retry: true,
          error_code: 'rate_limit_exceeded',
          error_summary: '',
        }),
        attempt({
          sequence: 2,
          action: 'retry',
          will_retry: true,
          error_code: 'upstream_timeout',
          error_summary: '',
        }),
      ])
      return m.requestLogKeyReason(log)
    },
    expect: { sequence: null, error_code: 'no_available_candidate', error_summary: '', source: 'request' },
  },
  {
    name: 'terminal code-only attempt keeps the key reason while sibling code-only attempts keep theirs',
    run: (m, ctx) => {
      const log = detail(
        { error_code: 'no_available_candidate', error_summary: '', attempt_count: 3 },
        [
          attempt({
            sequence: 1,
            action: 'retry',
            will_retry: true,
            error_code: 'rate_limit_exceeded',
            error_summary: '',
          }),
          attempt({
            sequence: 2,
            action: 'retry',
            will_retry: true,
            error_code: 'upstream_timeout',
            error_summary: '',
          }),
          attempt({
            sequence: 3,
            action: 'terminate',
            error_code: 'upstream_host_error',
            error_summary: '',
          }),
        ],
      )
      return {
        reason: m.requestLogKeyReason(log),
        sequences: [...m.requestLogAttemptReasonSequences(log)].sort((left, right) => left - right),
        firstScreen: m.requestLogFirstScreen(log, ctx.translate),
      }
    },
    expect: {
      reason: {
        sequence: 3,
        error_code: 'upstream_host_error',
        error_summary: '',
        source: 'terminal_attempt',
      },
      sequences: [1, 2],
      firstScreen: {
        status: '错误',
        status_code: 502,
        attempt_count: 3,
        key_reason_label: '关键原因 · 尝试 #3',
        key_reason_text: 'upstream_host_error',
        key_reason_code: '',
        request_error_code: 'no_available_candidate',
      },
    },
  },
  {
    name: 'code-only and summary forms of the same identity are shown once',
    run: (m) => {
      const log = detail({ error_code: 'no_available_candidate', error_summary: '' }, [
        attempt({
          sequence: 1,
          action: 'retry',
          will_retry: true,
          error_code: 'rate_limit_exceeded',
          error_summary: '',
        }),
        attempt({
          sequence: 2,
          action: 'retry',
          will_retry: true,
          error_code: 'rate_limit_exceeded',
          error_summary: 'rate_limit_exceeded',
        }),
        attempt({
          sequence: 3,
          action: 'terminate',
          error_code: 'upstream_host_error',
          error_summary: 'upstream exploded',
        }),
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
        attempt({
          sequence: 1,
          action: 'retry',
          will_retry: true,
          error_code: 'rate_limit_exceeded',
          error_summary: '  \n ',
        }),
      ])
      return {
        reason: m.requestLogKeyReason(log),
        sequences: [...m.requestLogAttemptReasonSequences(log)],
        text: m.requestLogAttemptReasonText(log.attempts[0]),
      }
    },
    expect: {
      reason: { sequence: null, error_code: 'no_available_candidate', error_summary: '', source: 'request' },
      sequences: [1],
      text: 'rate_limit_exceeded',
    },
  },
  {
    name: 'code-only attempt first screen stays consistent with the request level reason',
    run: (m, ctx) => {
      const log = detail(
        { error_code: 'no_available_candidate', error_summary: '', attempt_count: 2 },
        [
          attempt({
            sequence: 1,
            action: 'retry',
            will_retry: true,
            error_code: 'rate_limit_exceeded',
            error_summary: '',
          }),
          attempt({
            sequence: 2,
            action: 'terminate',
            error_code: 'upstream_host_error',
            error_summary: 'upstream exploded',
          }),
        ],
      )
      return {
        firstScreen: m.requestLogFirstScreen(log, ctx.translate),
        sequences: [...m.requestLogAttemptReasonSequences(log)],
        codeOnlyText: m.requestLogAttemptReasonText(log.attempts[0]),
      }
    },
    expect: {
      firstScreen: {
        status: '错误',
        status_code: 502,
        attempt_count: 2,
        key_reason_label: '关键原因 · 尝试 #2',
        key_reason_text: 'upstream exploded',
        key_reason_code: 'upstream_host_error',
        request_error_code: 'no_available_candidate',
      },
      sequences: [1],
      codeOnlyText: 'rate_limit_exceeded',
    },
  },
  {
    name: 'success never falls back to a code-only attempt reason',
    run: (m) => {
      const log = detail({ status: 'success', status_code: 200, error_code: '', error_summary: '' }, [
        attempt({
          sequence: 1,
          action: 'retry',
          will_retry: true,
          failure_category: 'upstream_host_error',
          error_code: 'upstream_host_error',
          error_summary: '',
        }),
        attempt({ sequence: 2, action: 'terminate', failure_category: 'ok', error_code: '', error_summary: '' }),
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
      format = await server.ssrLoadModule('/src/features/monitor/log-format.ts')
    } catch (err) {
      console.log(`FAIL  module load: ${err instanceof Error ? err.message : String(err)}`)
      return 1
    }

    const localeNames = ['zh-CN', 'en-US', 'ja-JP']
    const messagesByLocale = {}
    for (const locale of localeNames) {
      const module = await server.ssrLoadModule(`/src/i18n/locales/${locale}/monitor.ts`)
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
