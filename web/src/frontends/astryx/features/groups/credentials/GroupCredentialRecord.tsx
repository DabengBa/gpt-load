import * as stylex from '@stylexjs/stylex'
import { Button } from '@astryxdesign/core'
import { RotateCcw } from 'lucide-react'
import { useIntl } from 'react-intl'

import type { CredentialItemDto } from '@shared/control/types'
import type { MessageId } from '@shared/i18n/message-ids'
import { formatLocalInstant } from '@shared/lib/format'
import { presentCredentialFailureCategory } from '@shared/domain/groups/credentials/credential-failure-presenter'

import { useT } from '../../../app/i18n'
import { GroupApiKeyEditor } from './GroupApiKeyEditor'

const narrow = '@media (max-width: 700px)'
const small = '@media (max-width: 520px)'

const styles = stylex.create({
  record: {
    display: 'grid',
    gridTemplateColumns: '1fr',
    padding: 0,
  },
  summary: {
    display: 'grid',
    gridTemplateColumns: { default: 'minmax(0, 1fr) auto', [small]: 'minmax(0, 1fr)' },
    alignItems: 'center',
    minHeight: '56px',
    gap: '12px',
    paddingBlock: { [narrow]: '8px' },
  },

  maskCell: {
    minWidth: 0,
    fontFamily: 'var(--font-mono)',
    fontVariantNumeric: 'tabular-nums',
  },

  actions: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: { default: 'flex-end', [small]: 'flex-start' },
    gap: '4px',
  },
  details: {
    marginBlock: '0 8px',
    marginInline: '14px',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    paddingBlock: '8px',
    paddingInline: '14px',
  },
  detailsList: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '4px 16px',
    margin: 0,
    fontVariantNumeric: 'tabular-nums',
  },
  detailsPair: {
    display: 'flex',
    alignItems: 'baseline',
    gap: '4px',
    maxWidth: '100%',
    minWidth: 0,
  },
  detailsTerm: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    flexShrink: 0,
  },
  detailsValue: {
    margin: 0,
    minWidth: 0,
    overflowWrap: 'anywhere',
  },
})

export function GroupCredentialRecord({
  item,
  groupId,

  busy,
  onRestore,
}: {
  item: CredentialItemDto
  groupId: number

  busy: boolean
  onRestore(item: CredentialItemDto): void
}) {
  const t = useT()
  const intl = useIntl()
  const isProblem = item.effective_status === 'cooldown' || item.effective_status === 'blacklisted'
  const failureCategory = presentCredentialFailureCategory(
    (key) => t(key as MessageId),
    item.last_failure_category,
  )
  const failureSummary = item.last_failure_summary?.trim()
  const showFailureSummary =
    failureSummary &&
    failureSummary.toLocaleLowerCase() !== failureCategory.trim().toLocaleLowerCase()
  const recentLabel =
    item.recent_failure_count === 0
      ? t('group.credentials.recentSuccessOnly', {
          success: intl.formatNumber(item.recent_success_count),
        })
      : t('group.credentials.recent', {
          success: intl.formatNumber(item.recent_success_count),
          failure: intl.formatNumber(item.recent_failure_count),
        })

  return (
    <article {...stylex.props(styles.record)}>
      <div {...stylex.props(styles.summary)}>
        <div {...stylex.props(styles.maskCell)}>
          <GroupApiKeyEditor groupId={groupId} credential={item} disabled={busy} />
        </div>
        {isProblem && (
          <div {...stylex.props(styles.actions)}>
            <Button
              variant="secondary"
              size="sm"
              label={t('group.credentials.restore')}
              isDisabled={busy}
              icon={<RotateCcw size={15} aria-hidden="true" />}
              onClick={() => onRestore(item)}
            />
          </div>
        )}
      </div>
      <div {...stylex.props(styles.details)}>
        <dl {...stylex.props(styles.detailsList)}>
          <div {...stylex.props(styles.detailsPair)}>
            <dt {...stylex.props(styles.detailsTerm)}>{t('group.credentials.columns.recent')}</dt>
            <dd {...stylex.props(styles.detailsValue)}>{recentLabel}</dd>
          </div>
          <div {...stylex.props(styles.detailsPair)}>
            <dt {...stylex.props(styles.detailsTerm)}>{t('group.credentials.detailsFailure')}</dt>
            <dd {...stylex.props(styles.detailsValue)}>
              {failureCategory}
              {showFailureSummary && <span title={failureSummary}> · {failureSummary}</span>}
            </dd>
          </div>
          <div {...stylex.props(styles.detailsPair)}>
            <dt {...stylex.props(styles.detailsTerm)}>{t('group.credentials.detailsRecovery')}</dt>
            <dd {...stylex.props(styles.detailsValue)}>
              {item.effective_status === 'disabled'
                ? t('group.credentials.recovery.groupDisabled')
                : item.recovery.at_ms
                  ? t('group.credentials.recovery.at', {
                      time: formatLocalInstant(item.recovery.at_ms, intl.locale),
                    })
                  : t(`group.credentials.recovery.${item.recovery.mode}` as MessageId)}
            </dd>
          </div>
          <div {...stylex.props(styles.detailsPair)}>
            <dt {...stylex.props(styles.detailsTerm)}>
              {t('group.credentials.detailsConsecutive')}
            </dt>
            <dd {...stylex.props(styles.detailsValue)}>
              {intl.formatNumber(item.consecutive_failure_count)}
            </dd>
          </div>
        </dl>
      </div>
    </article>
  )
}
