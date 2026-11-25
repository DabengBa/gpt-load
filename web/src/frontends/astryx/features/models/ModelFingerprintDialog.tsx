import { Banner, Button, Dialog } from '@astryxdesign/core'
import { FingerprintPattern, Info, LoaderCircle, X } from 'lucide-react'
import { useId } from 'react'
import { useIntl } from 'react-intl'

import type {
  FingerprintRunFailure,
  ModelFingerprintResultDto,
} from '@shared/control/resources/fingerprint'
import { formatLocalInstant } from '@shared/lib/format'

import { useT } from '../../app/i18n'
import styles from './ModelFingerprintDialog.module.css'

export function ModelFingerprintDialog({
  open,
  pending,
  failure,
  retryFailure,
  result,
  onOpenChange,
  onRetry,
  onRetryChallenge,
  onViewLog,
}: {
  open: boolean
  pending: boolean
  failure: FingerprintRunFailure | null
  retryFailure: FingerprintRunFailure | null
  result: ModelFingerprintResultDto | null
  onOpenChange: (open: boolean) => void
  onRetry: () => void
  onRetryChallenge: (challengeId: string) => void
  onViewLog: (logId: string) => void
}) {
  const t = useT()
  const intl = useIntl()
  const id = useId()
  const title = t('monitor.modelFingerprint.title')
  const close = t('monitor.modelFingerprint.close')
  const log = t('monitor.modelFingerprint.challenge.log')
  const unknown = t('monitor.modelFingerprint.unknownValue')
  const format = (value: number | null) => (value === null ? unknown : intl.formatNumber(value))
  const percent = (value: number) =>
    intl.formatNumber(value, {
      style: 'percent',
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    })
  const score = (value: number) => intl.formatNumber(value, { maximumFractionDigits: 3 })
  const setOpen = (next: boolean) => {
    if (next || !pending) onOpenChange(next)
  }

  const analysis = result?.analysis ?? null
  const ranked = analysis?.decision === 'partial' || analysis?.decision === 'scored'
  const calibrated =
    analysis?.decision === 'scored' &&
    analysis.probability_status === 'reference_calibrated' &&
    analysis.probability_scope === 'reference-closed-set'
  const candidates = ranked ? analysis.results : []
  const prediction = ranked ? analysis.prediction_name || analysis.prediction : null
  const predictionFamily = ranked
    ? analysis.results.find((candidate) => candidate.model === analysis.prediction)?.family
    : null
  const scores = candidates.map((candidate) => candidate.score)
  // Partial rankings can contain negative scores; shift the visual scale without changing the values.
  const lowestScore = Math.min(0, ...scores)
  const scoreRange = Math.max(lowestScore, ...scores) - lowestScore
  const challenges = result?.challenges ?? []
  const succeeded = challenges.filter((challenge) => challenge.outcome === 'succeeded').length
  const challengeSummary = t('monitor.modelFingerprint.challenge.summary', {
    succeeded,
    total: challenges.length,
  })
  const context = result?.context_check ?? null
  const contextLog = context?.log_id ?? null
  const contextTone =
    context?.status === 'no_anomaly'
      ? 'ok'
      : context?.status === 'suspected_injection'
        ? 'warn'
        : 'neutral'
  const expected = context?.expected_input_tokens ?? null
  const reported = context?.reported_input_tokens ?? null
  const tokenScale = Math.max(1, expected ?? 0, reported ?? 0)
  const tokenWidth = (value: number) => `${(value / tokenScale) * 100}%`
  const delta = context?.delta ?? null
  const deltaLabel =
    delta === null ? unknown : intl.formatNumber(delta, { signDisplay: 'exceptZero' })

  return (
    <Dialog
      isOpen={open}
      onOpenChange={setOpen}
      aria-labelledby={id + '-title'}
      className={styles.shell}
      width="min(840px, calc(100dvw - 16px))"
      maxHeight="calc(100dvh - 16px)"
      padding={0}
    >
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <h2
            id={id + '-title'}
            className={styles.title}
            tabIndex={-1}
            data-autofocus="dialog-title"
          >
            {title}
          </h2>
          {result && (
            <span
              className={styles.route}
              title={`${result.group_name || `#${result.group_id}`} / ${result.model}`}
            >
              <span className={styles.routeGroup}>
                {result.group_name || `#${result.group_id}`}
              </span>
              <span aria-hidden="true">/</span>
              <span className={styles.routeModel}>{result.model}</span>
            </span>
          )}
        </div>
        <div className={styles.headerRight}>
          {result && (
            <time
              className={styles.timestamp}
              dateTime={new Date(result.tested_at_ms).toISOString()}
              aria-label={t('monitor.modelFingerprint.fields.detectedAt')}
            >
              {formatLocalInstant(result.tested_at_ms, intl.locale)}
            </time>
          )}
          <button
            type="button"
            className={styles.iconButton}
            aria-label={close}
            disabled={pending}
            onClick={() => setOpen(false)}
          >
            <X size={16} aria-hidden="true" />
          </button>
        </div>
      </header>

      <div className={styles.body}>
        {pending && (
          <div className={styles.loading} role="status">
            <LoaderCircle size={16} aria-hidden="true" />
            {t('monitor.modelFingerprint.loading')}
          </div>
        )}
        {failure !== null && (
          <>
            <Banner status="error" title={t(`monitor.modelFingerprint.failure.${failure}`)} />
            <div>
              <Button
                variant="secondary"
                size="sm"
                label={t('monitor.modelFingerprint.retry')}
                onClick={onRetry}
              />
            </div>
          </>
        )}
        {retryFailure !== null && (
          <Banner status="error" title={t(`monitor.modelFingerprint.failure.${retryFailure}`)} />
        )}
        {result?.failure != null && (
          <Banner
            status="warning"
            title={t(`monitor.modelFingerprint.failure.${result.failure}`)}
          />
        )}

        {result && (analysis !== null || challenges.length > 0 || context !== null) && (
          <div className={styles.verdict}>
            <div className={styles.verdictMain}>
              <div className={styles.mark}>
                <FingerprintPattern size={20} aria-hidden="true" />
              </div>
              <div className={styles.identity}>
                <div className={styles.modelLine}>
                  <span
                    className={styles.model}
                    title={prediction ? t('monitor.modelFingerprint.fields.prediction') : undefined}
                  >
                    {prediction ||
                      (analysis
                        ? t(`monitor.modelFingerprint.decision.${analysis.decision}`)
                        : title)}
                  </span>
                  {predictionFamily && <span className={styles.vendor}>{predictionFamily}</span>}
                  {ranked && analysis.ranking_score !== null && (
                    <span
                      className={styles.metrics}
                      title={t('monitor.modelFingerprint.rankingScore', {
                        score: score(analysis.ranking_score),
                      })}
                    >
                      {score(analysis.ranking_score)}
                    </span>
                  )}
                </div>
                {ranked && (
                  <div className={styles.flags}>
                    <span
                      className={styles.pill}
                      data-tone={analysis.decision === 'scored' ? 'ok' : 'warn'}
                    >
                      {t(`monitor.modelFingerprint.decision.${analysis.decision}`)}
                    </span>
                    {calibrated && analysis.probability !== null ? (
                      <span className={styles.pill} data-tone="ok">
                        {t('monitor.modelFingerprint.fields.probability')}{' '}
                        <span>{percent(analysis.probability)}</span>
                      </span>
                    ) : analysis.decision !== 'partial' ? (
                      <span className={styles.pill} data-tone="neutral">
                        {t('monitor.modelFingerprint.probabilityUnavailable')}
                      </span>
                    ) : null}
                  </div>
                )}
              </div>
            </div>
            {(challenges.length > 0 || (expected !== null && reported !== null)) && (
              <div className={styles.signals}>
                {challenges.length > 0 && (
                  <span className={styles.segments} title={challengeSummary}>
                    {challenges.map((challenge) => (
                      <span
                        key={challenge.id}
                        aria-hidden="true"
                        className={styles.segment}
                        data-tone={challenge.outcome === 'succeeded' ? 'ok' : 'error'}
                      />
                    ))}
                    <span className={styles.summary}>
                      {succeeded}/{challenges.length}
                    </span>
                  </span>
                )}
                {expected !== null && reported !== null && (
                  <span className={styles.pill} data-tone={contextTone} aria-hidden="true">
                    {format(expected)} → {format(reported)} tok
                  </span>
                )}
              </div>
            )}
          </div>
        )}

        {(candidates.length > 0 || context !== null) && (
          <div className={styles.bento} data-paired={candidates.length > 0 && context !== null}>
            {candidates.length > 0 && analysis && (
              <section className={styles.card} aria-labelledby={id + '-candidates'}>
                <div className={styles.cardHead}>
                  <div className={styles.cardLabel}>
                    <h3 className={styles.cardTitle} id={id + '-candidates'}>
                      {t('monitor.modelFingerprint.candidatesTitle')}
                    </h3>
                    <details className={styles.notice}>
                      <summary aria-label={t('monitor.modelFingerprint.rankingInfo')}>
                        <Info size={14} aria-hidden="true" />
                      </summary>
                      <p>{t('monitor.modelFingerprint.notice')}</p>
                    </details>
                  </div>
                  <span className={styles.meta}>
                    {t(
                      calibrated
                        ? 'monitor.modelFingerprint.probabilityAndScore'
                        : 'monitor.modelFingerprint.rankingScoreTitle',
                    )}
                  </span>
                </div>
                <ol
                  className={styles.candidateList}
                  tabIndex={candidates.length > 5 ? 0 : undefined}
                >
                  {candidates.map((candidate, index) => {
                    const name = candidate.display_name || candidate.model
                    const candidateScore = score(candidate.score)
                    const fill = scoreRange > 0 ? (candidate.score - lowestScore) / scoreRange : 0
                    return (
                      <li
                        key={candidate.model}
                        className={styles.candidateRow}
                        data-top={index === 0}
                      >
                        <svg className={styles.candidateFill} aria-hidden="true">
                          <rect width={`${fill * 100}%`} height="100%" rx="7" />
                        </svg>
                        <span className={styles.rank}>#{index + 1}</span>
                        <span className={styles.candidateIdentity}>
                          <span className={styles.candidateName} title={name}>
                            {name}
                          </span>
                          {candidate.family && (
                            <span className={styles.candidateVendor}>{candidate.family}</span>
                          )}
                        </span>
                        <span className={styles.candidateMetric}>
                          {calibrated && candidate.probability !== null && (
                            <span className={styles.candidateProbability}>
                              {percent(candidate.probability)}
                            </span>
                          )}
                          <span
                            className={styles.candidateScore}
                            title={t('monitor.modelFingerprint.rankingScore', {
                              score: candidateScore,
                            })}
                          >
                            {candidateScore}
                          </span>
                        </span>
                      </li>
                    )
                  })}
                </ol>
              </section>
            )}

            {context !== null && (
              <section className={styles.card} aria-labelledby={id + '-context'}>
                <div className={styles.cardHead}>
                  <div className={styles.cardLabel}>
                    <h3 className={styles.cardTitle} id={id + '-context'}>
                      {t('monitor.modelFingerprint.contextCheck.title')}
                    </h3>
                    <span className={styles.pill} data-tone={contextTone}>
                      {t(`monitor.modelFingerprint.contextCheck.status.${context.status}`)}
                    </span>
                  </div>
                  {contextLog !== null && (
                    <button
                      type="button"
                      className={styles.link}
                      onClick={() => onViewLog(contextLog)}
                    >
                      {log}
                    </button>
                  )}
                </div>
                <div className={styles.fidelity}>
                  <div className={styles.tokenBars}>
                    <div className={styles.tokenRow}>
                      <div className={styles.tokenLabel}>
                        <span>
                          {t('monitor.modelFingerprint.contextCheck.fields.expectedInputTokens')}
                        </span>
                        <strong>{format(expected)}</strong>
                      </div>
                      <div className={styles.track} aria-hidden="true">
                        {expected !== null && (
                          <svg>
                            <rect
                              className={styles.base}
                              width={tokenWidth(expected)}
                              height="100%"
                            />
                          </svg>
                        )}
                      </div>
                    </div>
                    <div className={styles.tokenRow}>
                      <div className={styles.tokenLabel}>
                        <span>
                          {t('monitor.modelFingerprint.contextCheck.fields.reportedInputTokens')}
                        </span>
                        <strong>{format(reported)}</strong>
                      </div>
                      <div className={styles.track} aria-hidden="true" data-tone={contextTone}>
                        {reported !== null && (
                          <svg>
                            <rect
                              className={styles.extra}
                              width={tokenWidth(reported)}
                              height="100%"
                            />
                            <rect
                              className={styles.match}
                              width={tokenWidth(Math.min(reported, expected ?? 0))}
                              height="100%"
                            />
                          </svg>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className={styles.delta}>
                    <span>{t('monitor.modelFingerprint.contextCheck.fields.delta')}</span>
                    <strong className={styles.deltaValue}>{deltaLabel}</strong>
                  </div>
                </div>
              </section>
            )}
          </div>
        )}

        {challenges.length > 0 && (
          <section className={styles.card} aria-labelledby={id + '-challenges'}>
            <div className={styles.cardHead}>
              <h3 className={styles.cardTitle} id={id + '-challenges'}>
                {t('monitor.modelFingerprint.challenge.title')}
              </h3>
              <span className={styles.meta}>
                {analysis
                  ? t('monitor.modelFingerprint.evidence', {
                      used: analysis.used_outputs,
                      total: challenges.length,
                    })
                  : challengeSummary}
              </span>
            </div>
            <ul className={styles.probeList}>
              {challenges.map((challenge) => {
                const logId = challenge.log_id
                const ok = challenge.outcome === 'succeeded'
                const tone = ok ? 'ok' : 'error'
                const reason =
                  challenge.reason === null
                    ? null
                    : t(`monitor.modelProbe.reason.${challenge.reason}`)
                const [, tag = challenge.id, uuid] = /^(probe-\d+)-(.+)$/.exec(challenge.id) ?? []
                return (
                  <li key={challenge.id} className={styles.probeRow} data-error={!ok}>
                    <div className={styles.probeIdentity} title={challenge.id}>
                      <span className={styles.dot} data-tone={tone} aria-hidden="true" />
                      <span className={styles.probeTag}>{tag}</span>
                      {uuid !== undefined && <span className={styles.probeID}>{uuid}</span>}
                    </div>
                    <span className={styles.pill} data-tone={tone}>
                      <span className={styles.srOnly}>
                        {t(`monitor.modelFingerprint.challenge.${challenge.outcome}`)}{' '}
                      </span>
                      {challenge.status_code ?? unknown}
                      {reason !== null && ` ${reason}`}
                    </span>
                    <span className={styles.latency}>
                      {challenge.latency_ms === null
                        ? unknown
                        : t('monitor.modelProbe.latency', { value: format(challenge.latency_ms) })}
                    </span>
                    <div className={styles.actions}>
                      {result?.run_id != null && challenge.retryable && (
                        <button
                          type="button"
                          className={styles.action}
                          data-tone="error"
                          disabled={pending}
                          onClick={() => onRetryChallenge(challenge.id)}
                        >
                          {t('monitor.modelFingerprint.challenge.retry')}
                        </button>
                      )}
                      {logId === null ? (
                        <span className={styles.srOnly}>
                          {t('monitor.modelFingerprint.challenge.noLog')}
                        </span>
                      ) : (
                        <button
                          type="button"
                          className={styles.action}
                          onClick={() => onViewLog(logId)}
                        >
                          {log}
                        </button>
                      )}
                    </div>
                  </li>
                )
              })}
            </ul>
          </section>
        )}
        {result !== null && candidates.length === 0 && (
          <p className={styles.meta}>{t('monitor.modelFingerprint.notice')}</p>
        )}
      </div>
      <footer className={styles.footer}>
        <span className={styles.keyHint} aria-hidden="true">
          <kbd>Esc</kbd>
          {close}
        </span>
        <Button
          variant="secondary"
          size="sm"
          isDisabled={pending}
          label={close}
          onClick={() => setOpen(false)}
        />
      </footer>
    </Dialog>
  )
}
