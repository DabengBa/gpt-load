import * as stylex from '@stylexjs/stylex'
import {
  Badge,
  Banner,
  Button,
  Dialog,
  DialogHeader,
  Layout,
  LayoutContent,
  LayoutFooter,
} from '@astryxdesign/core'
import { LoaderCircle } from 'lucide-react'
import { useIntl } from 'react-intl'

import type {
  FingerprintRunFailure,
  ModelFingerprintResultDto,
} from '@shared/control/resources/fingerprint'
import { formatLocalInstant } from '@shared/lib/format'

import { useT } from '../../app/i18n'

const NARROW = '@media (max-width: 480px)'

const spin = stylex.keyframes({
  to: { transform: 'rotate(360deg)' },
})

const styles = stylex.create({
  body: {
    display: 'grid',
    gap: 'var(--space-4)',
    minWidth: 0,
    lineHeight: 'var(--line-normal)',
  },
  // Classic QueryFeedback state="loading": bordered sunken box + spinner,
  // role="status" (same as the probe dialog).
  loading: {
    display: 'flex',
    minHeight: 48,
    alignItems: 'center',
    gap: 'var(--space-2)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text-muted)',
    padding: 'var(--space-3)',
  },
  loadingSpin: {
    animationName: {
      default: spin,
      '@media (prefers-reduced-motion: reduce)': 'none',
    },
    animationDuration: '1s',
    animationTimingFunction: 'linear',
    animationIterationCount: 'infinite',
  },
  details: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'max-content minmax(0, 1fr)',
      [NARROW]: '1fr',
    },
    gap: { default: '8px var(--space-3)', [NARROW]: 'var(--space-1)' },
    margin: 0,
    fontSize: 'var(--text-sm)',
    lineHeight: 'var(--line-normal)',
  },
  detailsTerm: {
    color: 'var(--color-text-muted)',
  },
  detailsTermSpaced: {
    marginTop: { default: 0, [NARROW]: 'var(--space-2)' },
  },
  detailsDesc: {
    display: 'block',
    minWidth: 0,
    margin: 0,
    alignItems: 'center',
    gap: 'var(--space-2)',
    color: 'var(--color-text)',
    fontVariantNumeric: 'tabular-nums',
    overflowWrap: 'anywhere',
  },
  prediction: {
    fontWeight: 600,
  },
  meta: {
    margin: 0,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  // Classic AppButton variant="link" size="inline" (same as the probe dialog).
  linkButton: {
    display: 'inline',
    minHeight: { default: 0, [NARROW]: 'var(--touch-target)' },
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: { default: 'var(--color-action)', ':hover': 'var(--color-action-hover)' },
    padding: 0,
    verticalAlign: 'baseline',
    cursor: 'pointer',
    textDecorationLine: { default: 'none', ':hover': 'underline' },
    borderRadius: 'var(--radius-control)',
    outline: { default: 'none', ':focus-visible': '2px solid var(--color-focus)' },
    outlineOffset: 2,
  },
  section: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-2)',
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 'var(--space-3)',
  },
  sectionTitle: {
    margin: 0,
    color: 'var(--color-text)',
    fontSize: 'var(--text-body)',
    fontWeight: 600,
  },
  list: {
    display: 'grid',
    gap: 'var(--space-1)',
    margin: 0,
    padding: 0,
    listStyle: 'none',
  },
  row: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) max-content',
    minWidth: 0,
    alignItems: 'baseline',
    justifyContent: 'space-between',
    gap: 'var(--space-2)',
    fontSize: 'var(--text-sm)',
    paddingBlock: 'var(--space-1)',
  },
  rowIdentity: {
    display: 'flex',
    flexWrap: 'wrap',
    minWidth: 0,
    alignItems: 'baseline',
    gap: 'var(--space-2)',
    overflowWrap: 'anywhere',
  },
  rowDetail: {
    flex: 'none',
    color: 'var(--color-text-muted)',
    fontVariantNumeric: 'tabular-nums',
    textAlign: 'end',
    whiteSpace: 'nowrap',
  },
  challenge: {
    paddingBlock: 'var(--space-2)',
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
  },
  identity: {
    minWidth: 0,
    overflowWrap: 'anywhere',
  },
  challengeMeta: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 'var(--space-2)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    overflowWrap: 'anywhere',
    fontVariantNumeric: 'tabular-nums',
  },
})

/**
 * Fingerprint detection result. The dialog states plainly that the attribution
 * is a closed-set probability over the service's reference bank, not proof of
 * the upstream model: an unlisted model is still mapped to its nearest
 * candidate. Every executed challenge keeps its own request-log entry.
 */
export function ModelFingerprintDialog({
  open,
  pending,
  failure,
  result,
  onOpenChange,
  onRetry,
  onViewLog,
}: {
  open: boolean
  pending: boolean
  failure: FingerprintRunFailure | null
  result: ModelFingerprintResultDto | null
  onOpenChange: (open: boolean) => void
  onRetry: () => void
  onViewLog: (logId: string) => void
}) {
  const t = useT()
  const intl = useIntl()

  function percent(value: number): string {
    return intl.formatNumber(value, {
      style: 'percent',
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    })
  }

  // Closing is blocked while pending: Escape, scrim, header X and the footer
  // Close all land here.
  function setOpen(next: boolean): void {
    if (!next && pending) return
    onOpenChange(next)
  }

  const analysis = result?.analysis ?? null
  const challenges = result?.challenges ?? []
  const succeededChallenges = challenges.filter(
    (challenge) => challenge.outcome === 'succeeded',
  ).length

  return (
    <Dialog isOpen={open} onOpenChange={setOpen} width={640} maxHeight="calc(100dvh - 36px)">
      <Layout
        header={
          <DialogHeader
            title={t('monitor.modelFingerprint.title')}
            subtitle={t('monitor.modelFingerprint.description')}
            onOpenChange={setOpen}
            hasDivider
          />
        }
        content={
          <LayoutContent isScrollable>
            <div {...stylex.props(styles.body)}>
              {pending && (
                <div {...stylex.props(styles.loading)} role="status">
                  <LoaderCircle
                    size={18}
                    aria-hidden="true"
                    {...stylex.props(styles.loadingSpin)}
                  />
                  <span>{t('monitor.modelFingerprint.loading')}</span>
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
              {result !== null && result.failure !== null && (
                <Banner
                  status="warning"
                  title={t(`monitor.modelFingerprint.failure.${result.failure}`)}
                />
              )}

              {result !== null && (
                <dl {...stylex.props(styles.details)}>
                  <dt {...stylex.props(styles.detailsTerm)}>
                    {t('monitor.modelFingerprint.fields.group')}
                  </dt>
                  <dd {...stylex.props(styles.detailsDesc)}>
                    {result.group_name === '' ? `#${result.group_id}` : result.group_name}
                  </dd>
                  <dt {...stylex.props(styles.detailsTerm, styles.detailsTermSpaced)}>
                    {t('monitor.modelFingerprint.fields.model')}
                  </dt>
                  <dd {...stylex.props(styles.detailsDesc)}>{result.model}</dd>
                  {analysis !== null && (
                    <>
                      <dt {...stylex.props(styles.detailsTerm, styles.detailsTermSpaced)}>
                        {t('monitor.modelFingerprint.fields.prediction')}
                      </dt>
                      <dd {...stylex.props(styles.detailsDesc)}>
                        <span {...stylex.props(styles.prediction)}>{analysis.prediction}</span>
                      </dd>
                      <dt {...stylex.props(styles.detailsTerm, styles.detailsTermSpaced)}>
                        {t('monitor.modelFingerprint.fields.probability')}
                      </dt>
                      <dd {...stylex.props(styles.detailsDesc)}>{percent(analysis.probability)}</dd>
                    </>
                  )}
                  <dt {...stylex.props(styles.detailsTerm, styles.detailsTermSpaced)}>
                    {t('monitor.modelFingerprint.fields.detectedAt')}
                  </dt>
                  <dd {...stylex.props(styles.detailsDesc)}>
                    {formatLocalInstant(result.tested_at_ms, intl.locale)}
                  </dd>
                </dl>
              )}

              {analysis !== null && (
                <section {...stylex.props(styles.section)} aria-labelledby="fingerprint-candidates">
                  <h3 id="fingerprint-candidates" {...stylex.props(styles.sectionTitle)}>
                    {t('monitor.modelFingerprint.candidatesTitle')}
                  </h3>
                  <ul {...stylex.props(styles.list)}>
                    {analysis.candidates.map((candidate) => (
                      <li key={candidate.model} {...stylex.props(styles.row)}>
                        <span {...stylex.props(styles.rowIdentity)}>
                          <span {...stylex.props(styles.identity)}>{candidate.model}</span>
                          {candidate.family !== '' && (
                            <span {...stylex.props(styles.meta)}>{candidate.family}</span>
                          )}
                        </span>
                        <span {...stylex.props(styles.rowDetail)}>
                          {percent(candidate.probability)}
                        </span>
                      </li>
                    ))}
                  </ul>
                  <p {...stylex.props(styles.meta)}>
                    {t('monitor.modelFingerprint.evidence', {
                      used: analysis.used_outputs,
                      total: challenges.length,
                    })}
                  </p>
                </section>
              )}

              {challenges.length > 0 && (
                <section {...stylex.props(styles.section)} aria-labelledby="fingerprint-challenges">
                  <h3 id="fingerprint-challenges" {...stylex.props(styles.sectionTitle)}>
                    {t('monitor.modelFingerprint.challenge.title')}
                  </h3>
                  <ul {...stylex.props(styles.list)}>
                    {challenges.map((challenge) => {
                      const logId = challenge.log_id
                      return (
                        <li key={challenge.id} {...stylex.props(styles.challenge)}>
                          <div {...stylex.props(styles.row)}>
                            <span {...stylex.props(styles.rowIdentity)}>
                              <span {...stylex.props(styles.identity)}>{challenge.id}</span>
                              <Badge
                                variant={challenge.outcome === 'succeeded' ? 'success' : 'error'}
                                label={t(`monitor.modelFingerprint.challenge.${challenge.outcome}`)}
                              />
                            </span>
                            <span {...stylex.props(styles.rowDetail)}>
                              {challenge.status_code ?? t('monitor.modelFingerprint.unknownValue')}
                            </span>
                          </div>
                          <div {...stylex.props(styles.challengeMeta)}>
                            {challenge.reason !== null && (
                              <span>{t(`monitor.modelProbe.reason.${challenge.reason}`)}</span>
                            )}
                            {challenge.latency_ms !== null && (
                              <span>
                                {t('monitor.modelProbe.latency', {
                                  value: intl.formatNumber(challenge.latency_ms),
                                })}
                              </span>
                            )}
                            {logId === null ? (
                              <span>{t('monitor.modelFingerprint.challenge.noLog')}</span>
                            ) : (
                              <button
                                type="button"
                                {...stylex.props(styles.linkButton)}
                                onClick={() => onViewLog(logId)}
                              >
                                {t('monitor.modelFingerprint.challenge.log')}
                              </button>
                            )}
                          </div>
                        </li>
                      )
                    })}
                  </ul>
                  <p {...stylex.props(styles.meta)}>
                    {t('monitor.modelFingerprint.challenge.summary', {
                      succeeded: succeededChallenges,
                      total: challenges.length,
                    })}
                  </p>
                </section>
              )}

              <Banner status="info" title={t('monitor.modelFingerprint.notice')} />
            </div>
          </LayoutContent>
        }
        footer={
          <LayoutFooter hasDivider>
            <Button
              variant="secondary"
              size="sm"
              isDisabled={pending}
              label={t('monitor.modelFingerprint.close')}
              onClick={() => setOpen(false)}
            />
          </LayoutFooter>
        }
      />
    </Dialog>
  )
}
