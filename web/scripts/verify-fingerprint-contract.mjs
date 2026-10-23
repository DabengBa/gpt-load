import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'

// Model fingerprint detection contract: the browser only speaks the control
// API (`POST /api/modeltrace-probe`), never upstream credentials or ModelTrace
// itself; the response is a strict projection of the Go control contract; one
// click is one execution.
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

  const target = { group_id: 7, model: 'gpt-5.4' }
  const logID = '9f1c2b3a-4d5e-4f60-8a9b-0c1d2e3f4a5b'
  // Mirrors control.ModelTraceProbeResponse in
  // internal/control/modeltrace_probe.go.
  const payload = () => ({
    group_id: 7,
    group_name: 'main',
    model: 'gpt-5.4',
    service: { base_url: 'http://modeltrace:8080', enabled: true },
    challenges: [
      {
        id: 'query-01',
        expected_count: 218,
        outcome: 'succeeded',
        reason: null,
        log_id: logID,
        latency_ms: 1_240,
        status_code: 200,
      },
      {
        id: 'query-02',
        expected_count: 233,
        outcome: 'failed',
        reason: 'timeout',
        log_id: null,
        latency_ms: null,
        status_code: null,
      },
    ],
    analysis: {
      prediction: 'gpt-5.4',
      prediction_name: 'GPT-5.4',
      probability: 0.912,
      family_prediction: 'gpt',
      family_probability: 0.93,
      used_outputs: 1,
      method: 'global_hellinger',
      candidates: [
        { model: 'gpt-5.4', family: 'gpt', probability: 0.912, conditional_probability: 0.98 },
        { model: 'gpt-5.6-sol', family: 'gpt', probability: 0.081, conditional_probability: 0.02 },
      ],
      diagnostics: [
        { prompt_id: 'query-01', accepted: true, parsed_numbers: 218, minimum_numbers: 200 },
      ],
      bank: { models: 18, built_at: '2026-03-01T00:00:00Z', bank_sha256: 'abc123' },
      upstream: { repository: 'xqy2006/ModelTrace', commit: 'deadbeef' },
    },
    failure: null,
    tested_at_ms: 1_792_714_550_837,
  })

  // 1. Success projection keeps the attribution evidence intact.
  const projected = resources.projectModelFingerprintResult(payload())
  assert.deepEqual(projected, payload())

  // 2. The projection is exact: unknown keys anywhere fail the whole response.
  for (const mutate of [
    (value) => {
      value.capability = true
    },
    (value) => {
      value.analysis.reference_note = 'closed set'
    },
    (value) => {
      value.analysis.candidates[0].family_name = 'GPT'
    },
    (value) => {
      value.analysis.bank.built_by = 'someone'
    },
    (value) => {
      value.challenges[0].prompt = 'leaked prompt'
    },
    (value) => {
      value.token = 'sk-live'
    },
    (value) => {
      value.analysis.api_key = 'sk-live'
    },
  ]) {
    const invalid = payload()
    mutate(invalid)
    assert.throws(() => resources.projectModelFingerprintResult(invalid))
  }

  // 3. Probabilities, identities, enums and the analysis/failure either-or.
  for (const mutate of [
    (value) => {
      value.analysis.probability = 1.01
    },
    (value) => {
      value.analysis.probability = -0.01
    },
    (value) => {
      value.analysis.probability = Number.NaN
    },
    (value) => {
      value.analysis.candidates[0].probability = 1.01
    },
    (value) => {
      value.analysis.candidates[0].conditional_probability = -0.01
    },
    (value) => {
      value.analysis.candidates = []
    },
    (value) => {
      value.analysis.candidates.push({ ...value.analysis.candidates[0] })
    },
    (value) => {
      value.analysis.prediction = ''
    },
    (value) => {
      value.analysis = null
    },
    (value) => {
      value.failure = 'analysis_rejected'
    },
    (value) => {
      value.failure = 'unknown_failure'
    },
    (value) => {
      value.challenges = []
    },
    (value) => {
      value.challenges[0].outcome = 'passed'
    },
    (value) => {
      value.challenges[0].reason = 'made_up'
    },
    (value) => {
      value.challenges[0].log_id = 'not-a-request-id'
    },
    (value) => {
      value.service.enabled = false
    },
    (value) => {
      value.service.base_url = 'ftp://modeltrace'
    },
    (value) => {
      value.model = ' '
    },
  ]) {
    const invalid = payload()
    mutate(invalid)
    assert.throws(() => resources.projectModelFingerprintResult(invalid))
  }

  // A failed analysis keeps every executed challenge but no analysis.
  const rejected = payload()
  rejected.analysis = null
  rejected.failure = 'analysis_rejected'
  const projectedFailure = resources.projectModelFingerprintResult(rejected)
  assert.equal(projectedFailure.analysis, null)
  assert.equal(projectedFailure.failure, 'analysis_rejected')
  assert.equal(projectedFailure.challenges.length, 2)

  // 4. One click sends exactly one saved target to the control endpoint.
  const calls = []
  const client = {
    request: async (path, options) => {
      calls.push({ path, options })
      return payload()
    },
  }
  const result = await resources.runModelFingerprint(client, target)
  assert.equal(result.analysis.prediction, 'gpt-5.4')
  assert.deepEqual(calls, [
    {
      path: '/api/modeltrace-probe',
      options: { method: 'POST', json: { group_id: 7, model: 'gpt-5.4' } },
    },
  ])

  // 5. Control error codes collapse into stable UI failure states.
  assert.equal(
    resources.fingerprintRunFailure(new errors.ApiError(503, 'MODELTRACE_DISABLED', 'off')),
    'disabled',
  )
  assert.equal(
    resources.fingerprintRunFailure(new errors.ApiError(503, 'MODELTRACE_MISCONFIGURED', 'bad')),
    'misconfigured',
  )
  assert.equal(
    resources.fingerprintRunFailure(new errors.ApiError(502, 'MODELTRACE_UNAVAILABLE', 'down')),
    'unavailable',
  )
  assert.equal(
    resources.fingerprintRunFailure(
      new errors.ApiError(422, 'MODELTRACE_ANALYSIS_REJECTED', 'rejected'),
    ),
    'analysis_rejected',
  )
  assert.equal(
    resources.fingerprintRunFailure(new errors.ApiError(409, 'NO_ACTIVE_CREDENTIAL', 'none')),
    'no_active_credential',
  )
  assert.equal(
    resources.fingerprintRunFailure(new errors.ApiError(400, 'BAD_REQUEST', 'bad')),
    'unsupported_target',
  )
  assert.equal(
    resources.fingerprintRunFailure(new errors.ApiError(500, 'INTERNAL_ERROR', 'boom')),
    'failed',
  )
  assert.equal(resources.fingerprintRunFailure(new Error('network')), 'failed')

  // 6. Run state: duplicate starts are refused, only running settles, close is
  //    blocked in flight and resets afterwards.
  const running = run.startFingerprintRun(run.initialFingerprintRunState, target)
  assert.equal(running.status, 'running')
  assert.deepEqual(running.target, target)
  assert.equal(running.result, null)
  assert.equal(running.failure, null)
  assert.equal(run.startFingerprintRun(running, { group_id: 7, model: 'other' }), running)
  assert.equal(run.closeFingerprintRun(running), running)
  assert.equal(run.isFingerprintDialogOpen(running), true)

  const succeeded = run.completeFingerprintRun(running, projected)
  assert.equal(succeeded.status, 'succeeded')
  assert.equal(succeeded.result?.analysis.prediction, 'gpt-5.4')
  assert.equal(succeeded.failure, null)
  assert.equal(run.completeFingerprintRun(succeeded, projected), succeeded)

  const failed = run.failFingerprintRun(running, 'unavailable')
  assert.equal(failed.status, 'failed')
  assert.deepEqual(failed.target, target)
  assert.equal(failed.result, null)
  assert.equal(failed.failure, 'unavailable')
  assert.equal(
    run.failFingerprintRun(run.initialFingerprintRunState, 'failed'),
    run.initialFingerprintRunState,
  )

  assert.equal(run.closeFingerprintRun(succeeded), run.initialFingerprintRunState)
  assert.equal(run.closeFingerprintRun(failed), run.initialFingerprintRunState)
  assert.equal(run.isFingerprintDialogOpen(run.initialFingerprintRunState), false)
  assert.equal(run.startFingerprintRun(failed, target).status, 'running')

  // 7. Runner: a duplicate click during flight never issues a second paid run,
  //    close is refused in flight, and a settled failure keeps its reason.
  const tick = () => new Promise((resolve) => setImmediate(resolve))
  let release
  const runnerCalls = []
  const runner = run.createFingerprintRunner({
    run: (next) => {
      runnerCalls.push(next.model)
      return new Promise((resolve) => {
        release = resolve
      })
    },
    describeFailure: resources.fingerprintRunFailure,
    onChange: () => {},
  })
  runner.start(target)
  runner.start(target)
  assert.deepEqual(runnerCalls, ['gpt-5.4'])
  assert.equal(runner.getState().status, 'running')
  runner.close()
  assert.equal(runner.getState().status, 'running')
  release(projected)
  await tick()
  assert.equal(runner.getState().status, 'succeeded')
  assert.equal(runner.getState().result?.analysis.prediction, 'gpt-5.4')
  runner.close()
  assert.equal(runner.getState().status, 'idle')

  const failingRunner = run.createFingerprintRunner({
    run: async () => {
      throw new errors.ApiError(503, 'MODELTRACE_DISABLED', 'off')
    },
    describeFailure: resources.fingerprintRunFailure,
    onChange: () => {},
  })
  failingRunner.start(target)
  assert.equal(failingRunner.getState().status, 'running')
  await tick()
  assert.equal(failingRunner.getState().status, 'failed')
  assert.equal(failingRunner.getState().failure, 'disabled')
  // A settled failure can be retried: the in-flight guard was released.
  failingRunner.start(target)
  assert.equal(failingRunner.getState().status, 'running')
  await tick()
  assert.equal(failingRunner.getState().status, 'failed')

  // 8. UI wiring: the model row offers the action next to the probe only for a
  //    saved row of a channel that declares text generation, and the dialog
  //    surfaces prediction, probability, candidates, challenges/log links, the
  //    run time and the closed-set notice.
  const tab = await read('../src/frontends/astryx/features/groups/models/GroupModelsTab.tsx')
  assert.match(tab, /import \{ useModelFingerprint \}/)
  assert.match(tab, /import \{ ModelFingerprintDialog \}/)
  assert.match(tab, /route\.operation === 'probe' && fingerprintProbeProtocols\.has\(/)
  assert.match(tab, /savedModelIDByKey\.get\(item\.key\) !== id/)
  assert.match(tab, /fingerprint\.run\.status === 'running'/)
  assert.match(tab, /monitor\.modelFingerprint\.button/)
  assert.match(tab, /monitor\.modelFingerprint\.draftHint/)
  assert.match(tab, /<ModelFingerprintDialog/)
  assert.match(tab, /onViewLog=\{viewProbeLog\}/)

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
  ]) {
    assert.ok(dialog.includes(key), `dialog does not render ${key}`)
  }

  // 9. Challenge executions are formal request logs, so the log projection and
  //    its filter labels must know the operation.
  const requestLogs = await read('../src/shared/control/resources/request-logs.ts')
  assert.match(requestLogs, /export const requestLogOperationValues = \[/)
  assert.match(requestLogs, /'modeltrace',\n\] as const/)
  const logFilters = await read('../src/shared/domain/monitor/log-filters.ts')
  assert.match(logFilters, /requestLogOperations = requestLogOperationValues/)
  for (const locale of ['zh-CN', 'en-US', 'ja-JP']) {
    const catalog = await read(`../src/shared/i18n/locales/${locale}/monitor.ts`)
    assert.ok(catalog.includes('modeltrace: '), `${locale} has no modeltrace log label`)
  }

  // 10. No frontend module in this feature carries an upstream credential or
  //    the ModelTrace token.
  for (const path of [
    '../src/shared/control/resources/fingerprint.ts',
    '../src/shared/domain/models/fingerprint-run.ts',
    '../src/frontends/astryx/features/models/use-model-fingerprint.ts',
    '../src/frontends/astryx/features/models/ModelFingerprintDialog.tsx',
  ]) {
    const source = await read(path)
    for (const forbidden of ['api_key', 'apiKey', 'Authorization', 'MODELTRACE_TOKEN']) {
      assert.ok(!source.includes(forbidden), `${path} carries ${forbidden}`)
    }
  }

  console.log('fingerprint contract: PASS')
} finally {
  await server.close()
}

// The stylex unplugin keeps worker handles alive after server.close();
// exit explicitly so the contract verdict does not hang the gate.
process.exit(process.exitCode ?? 0)
