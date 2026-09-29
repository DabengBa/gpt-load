import * as stylex from '@stylexjs/stylex'
import {
  Button,
  Dialog,
  DialogHeader,
  Layout,
  LayoutContent,
  LayoutFooter,
} from '@astryxdesign/core'
import { useIntl } from 'react-intl'

import type { CredentialTestResultDto } from '@shared/control/types'
import type { MessageId } from '@shared/i18n/message-ids'
import { formatLocalInstant } from '@shared/lib/format'

import { useT } from '../../../app/i18n'
import { InlineNotice } from '../../../components/InlineNotice'
import { CopyChip } from '../../access-keys/AccessKeyCopyChip'

const small = '@media (max-width: 480px)'

const styles = stylex.create({
  root: {
    display: 'grid',
    gap: 'var(--space-3)',
  },
  credential: {
    display: { default: 'flex', [small]: 'grid' },
    minWidth: 0,
    alignItems: 'baseline',
    gap: { default: 'var(--space-2)', [small]: 'var(--space-1)' },
    margin: 0,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  credentialValue: {
    minWidth: 0,
    color: 'var(--color-text)',
    fontWeight: 600,
    overflowWrap: 'anywhere',
  },
  details: {
    display: 'grid',
    gridTemplateColumns: { default: 'max-content minmax(0, 1fr)', [small]: '1fr' },
    columnGap: 'var(--space-3)',
    rowGap: { default: '8px', [small]: 'var(--space-1)' },
    margin: 0,
    fontSize: 'var(--text-sm)',
    lineHeight: 'var(--line-normal)',
  },
  term: {
    color: 'var(--color-text-muted)',
  },
  definition: {
    minWidth: 0,
    margin: 0,
    color: 'var(--color-text)',
    fontVariantNumeric: 'tabular-nums',
    overflowWrap: 'anywhere',
  },
  logActions: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
})

export function CredentialTestDialog({
  open,
  mask,
  pending,
  requestFailed,
  result,
  onOpenChange,
  onViewLog,
}: {
  open: boolean
  mask: string
  pending: boolean
  requestFailed: boolean
  result?: CredentialTestResultDto
  onOpenChange(open: boolean): void
  onViewLog(logId: string): void
}) {
  const t = useT()
  const intl = useIntl()

  const resultTone =
    result?.outcome === 'passed' ? 'success' : result?.outcome === 'failed' ? 'danger' : 'warning'
  const reasonLabel = result
    ? t(`group.credentials.test.reason.${result.reason ?? 'passed'}` as MessageId)
    : ''

  function setOpen(value: boolean): void {
    if (!value && pending) return
    onOpenChange(value)
  }

  return (
    <Dialog isOpen={open} onOpenChange={setOpen} width={460}>
      <Layout
        header={
          <DialogHeader
            title={t('group.credentials.test.title')}
            subtitle={t('group.credentials.test.description')}
            onOpenChange={pending ? undefined : setOpen}
            hasDivider
          />
        }
        content={
          <LayoutContent isScrollable>
            <div {...stylex.props(styles.root)}>
              <p {...stylex.props(styles.credential)}>
                <span>{t('group.credentials.test.fields.credential')}</span>
                <strong {...stylex.props(styles.credentialValue)}>{mask}</strong>
              </p>
              {pending ? (
                <div role="status" aria-live="polite">
                  {t('group.credentials.test.loading', { mask })}
                </div>
              ) : requestFailed ? (
                <InlineNotice tone="danger" appearance="ledger">
                  {t('group.credentials.test.requestFailed')}
                </InlineNotice>
              ) : result ? (
                <>
                  <InlineNotice tone={resultTone} appearance="ledger">
                    {t(`group.credentials.test.outcome.${result.outcome}` as MessageId)}
                  </InlineNotice>
                  <dl {...stylex.props(styles.details)}>
                    <dt {...stylex.props(styles.term)}>
                      {t('group.credentials.test.fields.model')}
                    </dt>
                    <dd {...stylex.props(styles.definition)}>{result.model}</dd>
                    <dt {...stylex.props(styles.term)}>
                      {t('group.credentials.test.fields.protocol')}
                    </dt>
                    <dd {...stylex.props(styles.definition)}>{result.protocol}</dd>
                    <dt {...stylex.props(styles.term)}>
                      {t('group.credentials.test.fields.latency')}
                    </dt>
                    <dd {...stylex.props(styles.definition)}>
                      {t('group.credentials.test.latency', {
                        value: intl.formatNumber(result.latency_ms),
                      })}
                    </dd>
                    <dt {...stylex.props(styles.term)}>
                      {t('group.credentials.test.fields.reason')}
                    </dt>
                    <dd {...stylex.props(styles.definition)}>{reasonLabel}</dd>
                    {result.outcome === 'passed' && (
                      <>
                        <dt {...stylex.props(styles.term)}>
                          {t('group.credentials.test.fields.recovered')}
                        </dt>
                        <dd {...stylex.props(styles.definition)}>
                          {result.recovered
                            ? t('group.credentials.test.recovered')
                            : t('group.credentials.test.alreadyAvailable')}
                        </dd>
                      </>
                    )}
                    <dt {...stylex.props(styles.term)}>
                      {t('group.credentials.test.fields.testedAt')}
                    </dt>
                    <dd {...stylex.props(styles.definition)}>
                      {formatLocalInstant(result.tested_at_ms, intl.locale)}
                    </dd>
                    <dt {...stylex.props(styles.term)}>
                      {t('monitor.modelProbe.fields.logId')}
                    </dt>
                    <dd {...stylex.props(styles.definition)}>
                      {result.log_id ? (
                        <span {...stylex.props(styles.logActions)}>
                          <CopyChip
                            value={result.log_id}
                            label={t('monitor.modelProbe.fields.logId')}
                            successLabel={t('common.copied')}
                            failureLabel={t('common.copyFailed')}
                          />
                          <Button
                            variant="ghost"
                            size="sm"
                            label={t('monitor.modelProbe.viewLog')}
                            onClick={() => onViewLog(result.log_id!)}
                          />
                        </span>
                      ) : (
                        t('monitor.modelProbe.notExecuted')
                      )}
                    </dd>
                  </dl>
                  {result.outcome === 'passed' && (
                    <InlineNotice tone="success" appearance="ledger">
                      {result.recovered
                        ? t('group.credentials.test.recovered')
                        : t('group.credentials.test.alreadyAvailable')}
                    </InlineNotice>
                  )}
                </>
              ) : null}
            </div>
          </LayoutContent>
        }
        footer={
          <LayoutFooter hasDivider>
            <Button
              variant="secondary"
              size="sm"
              isDisabled={pending}
              label={t('group.credentials.test.close')}
              onClick={() => setOpen(false)}
            />
          </LayoutFooter>
        }
      />
    </Dialog>
  )
}
