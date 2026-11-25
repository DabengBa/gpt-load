import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'

const webRoot = fileURLToPath(new URL('..', import.meta.url))
const read = (path) => readFile(fileURLToPath(new URL(path, import.meta.url)), 'utf8')
const server = await createServer({
  root: webRoot,
  server: { middlewareMode: true },
  appType: 'custom',
  logLevel: 'error',
})

try {
  const resources = await server.ssrLoadModule('/src/shared/control/resources/fingerprint.ts')
  const errors = await server.ssrLoadModule('/src/shared/http/errors.ts')
  const run = await server.ssrLoadModule('/src/shared/domain/models/fingerprint-run.ts')
  const fixtures = {}
  for (const decision of ['scored', 'partial', 'unscorable', 'not_model']) {
    const payload = JSON.parse(await read(`../e2e/fixtures/${decision}.json`))
    const actual = resources.projectModelFingerprintResult(payload)
    payload.context_check = null
    // Contract fixtures predate retry metadata; legacy payloads project to
    // explicit null/false defaults so they stay deep-equal here.
    payload.run_id ??= null
    for (const challenge of payload.challenges) challenge.retryable ??= false
    assert.deepEqual(actual, payload)
    assert.equal(actual.analysis.decision, decision)
    if (decision !== 'scored') assert.equal(actual.analysis.probability, null)
    if (decision === 'partial')
      assert.ok(actual.analysis.results.every((item) => item.probability === null))
    if (decision === 'unscorable' || decision === 'not_model') {
      assert.equal(actual.analysis.prediction, '')
      assert.equal(actual.analysis.results.length, 0)
    }
    fixtures[decision] = payload
  }
  const payload = () => structuredClone(fixtures.scored)
  for (const mutate of [
    (v) => {
      v.token = 'secret'
    },
    (v) => {
      v.analysis.extra = true
    },
    (v) => {
      v.analysis.results[0].extra = true
    },
    (v) => {
      v.analysis.calibration.extra = true
    },
    (v) => {
      v.challenges[0].prompt = 'secret'
    },
    (v) => {
      v.analysis.probability = 1.1
    },
    (v) => {
      v.analysis.results[0].probability = -1
    },
    (v) => {
      v.analysis.decision = 'unknown'
    },
    (v) => {
      v.failure = 'analysis_rejected'
    },
    (v) => {
      v.challenges[0].log_id = 'invalid'
    },
  ]) {
    const invalid = payload()
    mutate(invalid)
    assert.throws(() => resources.projectModelFingerprintResult(invalid))
  }
  const failure = payload()
  failure.analysis = null
  failure.failure = 'analysis_unavailable'
  failure.challenges[0].log_id = '9f1c2b3a-4d5e-4f60-8a9b-0c1d2e3f4a5b'
  assert.deepEqual(resources.projectModelFingerprintResult(failure), failure)
  const context = payload()
  context.context_check = {
    status: 'suspected_injection',
    expected_input_tokens: 6,
    reported_input_tokens: 34,
    delta: 28,
    log_id: '9f1c2b3a-4d5e-4f60-8a9b-0c1d2e3f4a5b',
  }
  assert.deepEqual(resources.projectModelFingerprintResult(context), context)
  for (const [field, value] of [
    ['status', 'confirmed_injection'],
    ['expected_input_tokens', -1],
    ['reported_input_tokens', 1.5],
    ['delta', Infinity],
    ['log_id', 'invalid'],
    ['extra', 'secret'],
  ]) {
    const invalid = structuredClone(context)
    invalid.context_check[field] = value
    assert.throws(() => resources.projectModelFingerprintResult(invalid))
  }
  // Retry metadata is optional so legacy payloads (and existing contract
  // fixtures) stay compatible; absent means no run to retry and not retryable.
  const legacy = payload()
  delete legacy.run_id
  for (const challenge of legacy.challenges) delete challenge.retryable
  const projectedLegacy = resources.projectModelFingerprintResult(legacy)
  assert.equal(projectedLegacy.run_id, null)
  assert.ok(projectedLegacy.challenges.every((challenge) => challenge.retryable === false))
  const retryable = payload()
  retryable.run_id = '9f1c2b3a-4d5e-4f60-8a9b-0c1d2e3f4a5b'
  retryable.challenges[0].retryable = true
  assert.deepEqual(resources.projectModelFingerprintResult(retryable), retryable)
  for (const [field, value] of [
    ['run_id', 'not-a-uuid'],
    ['run_id', 1],
  ]) {
    const invalid = structuredClone(retryable)
    invalid[field] = value
    assert.throws(() => resources.projectModelFingerprintResult(invalid))
  }
  const invalidRetryable = structuredClone(retryable)
  invalidRetryable.challenges[0].retryable = 'yes'
  assert.throws(() => resources.projectModelFingerprintResult(invalidRetryable))
  for (const forbidden of ['samples', 'sample', 'outputs']) {
    const invalid = structuredClone(retryable)
    invalid[forbidden] = []
    assert.throws(() => resources.projectModelFingerprintResult(invalid))
  }
  const calls = []
  const target = { group_id: 1, model: 'example' }
  const result = await resources.runModelFingerprint(
    {
      request: async (path, options) => {
        calls.push({ path, options })
        return payload()
      },
    },
    target,
  )
  assert.deepEqual(calls, [
    { path: '/api/modeltrace-probe', options: { method: 'POST', json: target } },
  ])
  const retryCalls = []
  const retryRequest = {
    group_id: 1,
    model: 'example',
    run_id: '9f1c2b3a-4d5e-4f60-8a9b-0c1d2e3f4a5b',
    challenge_id: 'query-01',
  }
  const retried = await resources.retryModelFingerprintChallenge(
    {
      request: async (path, options) => {
        retryCalls.push({ path, options })
        return retryable
      },
    },
    retryRequest,
  )
  assert.deepEqual(retried, retryable)
  assert.deepEqual(retryCalls, [
    {
      path: '/api/modeltrace-probe/retry',
      options: { method: 'POST', json: retryRequest },
    },
  ])
  const running = run.startFingerprintRun(run.initialFingerprintRunState, target)
  assert.equal(run.startFingerprintRun(running, target), running)
  assert.equal(run.closeFingerprintRun(running), running)
  assert.equal(run.completeFingerprintRun(running, result).status, 'succeeded')
  let release
  let count = 0
  const runner = run.createFingerprintRunner({
    run: () => {
      count++
      return new Promise((resolve) => {
        release = resolve
      })
    },
    describeFailure: () => 'failed',
    onChange: () => {},
  })
  runner.start(target)
  runner.start(target)
  runner.close()
  assert.equal(count, 1)
  assert.equal(runner.getState().status, 'running')
  release(result)
  await new Promise((resolve) => setImmediate(resolve))
  assert.equal(runner.getState().status, 'succeeded')
  runner.close()
  assert.equal(runner.getState().status, 'idle')
  let retryCount = 0
  const retryRunner = run.createFingerprintRunner({
    run: async () => result,
    retry: () => {
      retryCount++
      return new Promise((resolve) => {
        release = resolve
      })
    },
    describeFailure: () => 'failed',
    onChange: () => {},
  })
  retryRunner.start(target)
  await new Promise((resolve) => setImmediate(resolve))
  const beforeRetry = retryRunner.getState()
  retryRunner.retryChallenge({ ...retryRequest })
  assert.equal(retryRunner.getState(), beforeRetry)
  assert.equal(retryCount, 0)
  retryRunner.close()
  const retryTarget = target
  const retryingRunner = run.createFingerprintRunner({
    run: async () => retryable,
    retry: () => {
      retryCount++
      return new Promise((resolve) => {
        release = resolve
      })
    },
    describeFailure: () => 'failed',
    onChange: () => {},
  })
  retryingRunner.start(retryTarget)
  await new Promise((resolve) => setImmediate(resolve))
  retryingRunner.retryChallenge(retryRequest)
  retryingRunner.retryChallenge(retryRequest)
  assert.equal(retryCount, 1)
  assert.equal(retryingRunner.getState().status, 'retrying')
  assert.equal(retryingRunner.getState().result, retryable)
  release(retryable)
  await new Promise((resolve) => setImmediate(resolve))
  assert.equal(retryingRunner.getState().status, 'succeeded')
  assert.equal(retryingRunner.getState().result, retryable)
  retryingRunner.close()
  assert.equal(retryingRunner.getState().status, 'idle')
  const failingRetryRunner = run.createFingerprintRunner({
    run: async () => retryable,
    retry: async () => {
      throw new errors.ApiError(502, 'MODELTRACE_UNAVAILABLE', 'down')
    },
    describeFailure: resources.fingerprintRunFailure,
    onChange: () => {},
  })
  failingRetryRunner.start(retryTarget)
  await new Promise((resolve) => setImmediate(resolve))
  failingRetryRunner.retryChallenge(retryRequest)
  await new Promise((resolve) => setImmediate(resolve))
  assert.equal(failingRetryRunner.getState().status, 'retry_failed')
  assert.equal(failingRetryRunner.getState().result, retryable)
  failingRetryRunner.retryChallenge(retryRequest)
  assert.equal(failingRetryRunner.getState().status, 'retrying')
  await new Promise((resolve) => setImmediate(resolve))
  assert.equal(failingRetryRunner.getState().failure, 'unavailable')
  for (const [status, code, expected] of [
    [503, 'MODELTRACE_DISABLED', 'disabled'],
    [503, 'MODELTRACE_MISCONFIGURED', 'misconfigured'],
    [502, 'MODELTRACE_UNAVAILABLE', 'unavailable'],
    [422, 'MODELTRACE_ANALYSIS_REJECTED', 'analysis_rejected'],
    [409, 'NO_ACTIVE_CREDENTIAL', 'no_active_credential'],
    [400, 'BAD_REQUEST', 'unsupported_target'],
    [500, 'INTERNAL_ERROR', 'failed'],
  ]) {
    assert.equal(
      resources.fingerprintRunFailure(new errors.ApiError(status, code, 'error')),
      expected,
    )
  }
  assert.equal(resources.fingerprintRunFailure(new Error('network')), 'failed')
  const failed = run.failFingerprintRun(running, 'unavailable')
  assert.equal(failed.status, 'failed')
  assert.deepEqual(failed.target, target)
  assert.equal(failed.failure, 'unavailable')
  assert.equal(
    run.failFingerprintRun(run.initialFingerprintRunState, 'failed'),
    run.initialFingerprintRunState,
  )
  assert.equal(run.closeFingerprintRun(failed), run.initialFingerprintRunState)
  assert.equal(run.startFingerprintRun(failed, target).status, 'running')
  const failingRunner = run.createFingerprintRunner({
    run: async () => {
      throw new errors.ApiError(502, 'MODELTRACE_UNAVAILABLE', 'down')
    },
    describeFailure: resources.fingerprintRunFailure,
    onChange: () => {},
  })
  for (let attempt = 0; attempt < 2; attempt++) {
    failingRunner.start(target)
    assert.equal(failingRunner.getState().status, 'running')
    await new Promise((resolve) => setImmediate(resolve))
    assert.equal(failingRunner.getState().failure, 'unavailable')
  }
  const tab = await read('../src/frontends/astryx/features/groups/models/GroupModelsTab.tsx')
  for (const pattern of [
    /import \{ useModelFingerprint \}/,
    /import \{ ModelFingerprintDialog \}/,
    /route\.operation === 'probe' && fingerprintProbeProtocols\.has\(/,
    /savedModelIDByKey\.get\(item\.key\) !== id/,
    /fingerprint\.run\.status === 'running'/,
    /monitor\.modelFingerprint\.button/,
    /monitor\.modelFingerprint\.draftHint/,
    /<ModelFingerprintDialog/,
    /onViewLog=\{viewProbeLog\}/,
  ])
    assert.match(tab, pattern)
  const dialog = await read('../src/frontends/astryx/features/models/ModelFingerprintDialog.tsx')
  for (const key of [
    'monitor.modelFingerprint.fields.prediction',
    'monitor.modelFingerprint.fields.probability',
    'monitor.modelFingerprint.candidatesTitle',
    'monitor.modelFingerprint.fields.detectedAt',
    'monitor.modelFingerprint.challenge.log',
    'monitor.modelFingerprint.challenge.summary',
    'monitor.modelFingerprint.evidence',
    'monitor.modelFingerprint.notice',
    'monitor.modelFingerprint.retry',
    'monitor.modelFingerprint.failure.${failure}',
  ])
    assert.ok(dialog.includes(key), `dialog does not render ${key}`)
  const requestLogs = await read('../src/shared/control/resources/request-logs.ts')
  assert.match(requestLogs, /'modeltrace',\n\] as const/)
  const logFilters = await read('../src/shared/domain/monitor/log-filters.ts')
  assert.match(logFilters, /requestLogOperations = requestLogOperationValues/)
  for (const locale of ['zh-CN', 'en-US']) {
    assert.ok(
      (await read(`../src/shared/i18n/locales/${locale}/monitor.ts`)).includes('modeltrace: '),
    )
  }
  for (const path of [
    '../src/shared/control/resources/fingerprint.ts',
    '../src/shared/domain/models/fingerprint-run.ts',
    '../src/frontends/astryx/features/models/use-model-fingerprint.ts',
    '../src/frontends/astryx/features/models/ModelFingerprintDialog.tsx',
  ]) {
    const source = await read(path)
    for (const forbidden of ['api_key', 'apiKey', 'Authorization', 'MODELTRACE_TOKEN'])
      assert.ok(!source.includes(forbidden), `${path} carries ${forbidden}`)
  }
  console.log('fingerprint contract: PASS')
} finally {
  await server.close()
}
process.exit(process.exitCode ?? 0)
