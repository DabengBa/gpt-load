import * as stylex from '@stylexjs/stylex'
import { Badge, IconButton, Popover } from '@astryxdesign/core'
import { Activity, ChevronDown, Ellipsis, RotateCcw, Trash2 } from 'lucide-react'
import { useIntl } from 'react-intl'
import { useState } from 'react'

import type { CredentialItemDto } from '@shared/control/types'
import type { MessageId } from '@shared/i18n/message-ids'
import { formatLocalInstant } from '@shared/lib/format'
import { presentCredentialFailureCategory } from '@shared/domain/groups/credentials/credential-failure-presenter'

import { useT } from '../../../app/i18n'
import { credentialStatusBadgeVariant } from '../credential-status'
import { GroupApiKeyEditor } from './GroupApiKeyEditor'

const narrow = '@media (max-width: 700px)'

const styles = stylex.create({
  record: {
    display: 'grid',
    gridTemplateColumns: '1fr',
    padding: 0,
  },
  summary: {
    display: 'grid',
    gridTemplateColumns: {
      default: '34px minmax(150px, 1.4fr) minmax(110px, 0.8fr) minmax(130px, 1fr) 90px',
      [narrow]: '28px minmax(0, 1fr) 86px',
    },
    alignItems: 'center',
    minHeight: '56px',
    gap: '12px',
    paddingBlock: { [narrow]: '8px' },
  },
  select: {
    display: 'flex',
    justifyContent: 'center',
  },
  selectInput: {
    width: '16px',
    height: '16px',
    accentColor: 'var(--color-action)',
  },
  maskCell: {
    minWidth: 0,
    fontFamily: 'var(--font-mono)',
    fontVariantNumeric: 'tabular-nums',
  },
  statusCell: {
    display: { [narrow]: 'none' },
  },
  recentCell: {
    minWidth: 0,
    fontFamily: 'var(--font-mono)',
    fontVariantNumeric: 'tabular-nums',
    display: { [narrow]: 'none' },
  },
  actions: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: '4px',
  },
  menu: {
    display: 'grid',
    minWidth: '160px',
    gap: '2px',
  },
  menuItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: 'var(--color-text)',
    padding: '8px',
    textAlign: 'left',
    font: 'inherit',
    cursor: { default: 'pointer', ':disabled': 'not-allowed' },
    opacity: { ':disabled': 0.55 },
    borderRadius: 'var(--radius-control)',
  },
  menuItemDanger: {
    color: 'var(--color-danger)',
  },
  details: {
    marginBlock: '0 8px',
    marginInline: '14px',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    paddingBlock: '12px',
    paddingInline: '14px',
  },
  detailsList: {
    display: 'grid',
    gridTemplateColumns: { default: 'repeat(3, minmax(0, 1fr))', [narrow]: '1fr' },
    gap: '16px',
    margin: 0,
  },
  detailsTerm: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
  },
  detailsValue: {
    marginBlockStart: '3px',
  },
  mobileLabel: {
    display: { default: 'none', [narrow]: 'block' },
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
  },
  srOnly: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
})

export function GroupCredentialRecord({
  item,
  groupId,
  rowIndex,
  selected,
  busy,
  expanded,
  onSelectedChange,
  onExpandedChange,
  onTest,
  onRestore,
  onRemove,
}: {
  item: CredentialItemDto
  groupId: number
  rowIndex: number
  selected: boolean
  busy: boolean
  expanded: boolean
  onSelectedChange(selected: boolean): void
  onExpandedChange(expanded: boolean): void
  onTest(item: CredentialItemDto): void
  onRestore(item: CredentialItemDto): void
  onRemove(item: CredentialItemDto): void
}) {
  const t = useT()
  const intl = useIntl()
  const [menuOpen, setMenuOpen] = useState(false)

  const detailId = `group-credential-details-${item.credential_id}`
  const isProblem = item.effective_status === 'cooldown' || item.effective_status === 'blacklisted'
  const recentLabel =
    item.recent_failure_count === 0
      ? t('group.credentials.recentSuccessOnly', {
          success: intl.formatNumber(item.recent_success_count),
        })
      : t('group.credentials.recent', {
          success: intl.formatNumber(item.recent_success_count),
          failure: intl.formatNumber(item.recent_failure_count),
        })

  function menuAction(action: () => void): () => void {
    return () => {
      setMenuOpen(false)
      action()
    }
  }

  return (
    <article {...stylex.props(styles.record)} role="row" aria-rowindex={rowIndex}>
      <div {...stylex.props(styles.summary)} role="presentation">
        <div {...stylex.props(styles.select)} role="cell">
          <label>
            <span {...stylex.props(styles.srOnly)}>
              {t('group.credentials.selectCredential', { mask: item.mask })}
            </span>
            <input
              {...stylex.props(styles.selectInput)}
              type="checkbox"
              checked={selected}
              disabled={busy}
              onChange={(event) => onSelectedChange(event.target.checked)}
            />
          </label>
        </div>
        <div {...stylex.props(styles.maskCell)} role="cell">
          <span {...stylex.props(styles.mobileLabel)}>
            {t('group.credentials.columns.credential')}
          </span>
          <GroupApiKeyEditor groupId={groupId} credential={item} disabled={busy} />
        </div>
        <div {...stylex.props(styles.statusCell)} role="cell">
          <span {...stylex.props(styles.mobileLabel)}>{t('group.credentials.columns.status')}</span>
          <Badge
            variant={credentialStatusBadgeVariant(item.effective_status)}
            label={t(`group.credentials.effective.${item.effective_status}` as MessageId)}
          />
        </div>
        <div {...stylex.props(styles.recentCell)} role="cell">
          <span {...stylex.props(styles.mobileLabel)}>{t('group.credentials.columns.recent')}</span>
          {recentLabel}
        </div>
        <div {...stylex.props(styles.actions)} role="cell">
          <Popover
            isOpen={menuOpen}
            onOpenChange={setMenuOpen}
            placement="below"
            alignment="end"
            content={
              <div {...stylex.props(styles.menu)}>
                <button
                  type="button"
                  {...stylex.props(styles.menuItem)}
                  disabled={busy}
                  onClick={menuAction(() => onTest(item))}
                >
                  <Activity size={15} aria-hidden="true" />
                  {t('group.credentials.test.action')}
                </button>
                {isProblem && (
                  <button
                    type="button"
                    {...stylex.props(styles.menuItem)}
                    disabled={busy}
                    onClick={menuAction(() => onRestore(item))}
                  >
                    <RotateCcw size={15} aria-hidden="true" />
                    {t('group.credentials.restore')}
                  </button>
                )}
                <button
                  type="button"
                  {...stylex.props(styles.menuItem, styles.menuItemDanger)}
                  disabled={busy}
                  onClick={menuAction(() => onRemove(item))}
                >
                  <Trash2 size={15} aria-hidden="true" />
                  {t('group.credentials.delete')}
                </button>
              </div>
            }
          >
            <IconButton
              variant="ghost"
              size="sm"
              label={t('group.credentials.moreActions')}
              isDisabled={busy}
              icon={<Ellipsis size={16} aria-hidden="true" />}
            />
          </Popover>
          <IconButton
            variant="ghost"
            size="sm"
            label={expanded ? t('group.credentials.collapse') : t('group.credentials.expand')}
            aria-expanded={expanded}
            aria-controls={detailId}
            icon={<ChevronDown size={16} aria-hidden="true" />}
            onClick={() => onExpandedChange(!expanded)}
          />
        </div>
      </div>
      {expanded && (
        <div id={detailId} {...stylex.props(styles.details)} role="cell">
          <dl {...stylex.props(styles.detailsList)}>
            <div>
              <dt {...stylex.props(styles.detailsTerm)}>{t('group.credentials.detailsFailure')}</dt>
              <dd {...stylex.props(styles.detailsValue)}>
                {presentCredentialFailureCategory(
                  (key) => t(key as MessageId),
                  item.last_failure_category,
                )}
              </dd>
            </div>
            <div>
              <dt {...stylex.props(styles.detailsTerm)}>
                {t('group.credentials.detailsRecovery')}
              </dt>
              <dd {...stylex.props(styles.detailsValue)}>
                {item.recovery.at_ms
                  ? t('group.credentials.recovery.at', {
                      time: formatLocalInstant(item.recovery.at_ms, intl.locale),
                    })
                  : t(`group.credentials.recovery.${item.recovery.mode}` as MessageId)}
              </dd>
            </div>
            <div>
              <dt {...stylex.props(styles.detailsTerm)}>
                {t('group.credentials.detailsConsecutive')}
              </dt>
              <dd {...stylex.props(styles.detailsValue)}>
                {intl.formatNumber(item.consecutive_failure_count)}
              </dd>
            </div>
          </dl>
        </div>
      )}
    </article>
  )
}
