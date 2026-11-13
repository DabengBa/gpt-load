import { Badge, Button, Switch, TextInput } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useIntl } from 'react-intl'
import type { GroupSiteAccessDto } from '@shared/control/types'
import {
  extractSiteRootUrl,
  formatGroupRatio,
  requiresSiteRootConfirmation,
  type GroupSettingsDraft,
} from '@shared/domain/groups/settings/group-settings-patch'
import type { GroupSettingsDto } from '@shared/control/types'
import { retryGroupSiteInfo } from '@shared/control/resources/groups'
import { controlQueryKeys } from '@shared/control/query-keys'
import { useAppServices } from '../../../app/services'
import { useT } from '../../../app/i18n'
import { RouteLink } from '../../../app/route-link'

export function GroupSettingsSiteAccessSection({
  groupId,
  saved,
  draft,
  access,
  disabled,
  forceConfirmation,
  onChange,
}: {
  groupId: number
  saved: GroupSettingsDto
  draft: GroupSettingsDraft
  access: GroupSiteAccessDto
  disabled: boolean
  forceConfirmation: boolean
  onChange(value: Partial<GroupSettingsDraft>): void
}) {
  const t = useT()
  const intl = useIntl()
  const { apiClient } = useAppServices()
  const queryClient = useQueryClient()
  const retry = useMutation({
    mutationFn: () => retryGroupSiteInfo(apiClient, groupId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: controlQueryKeys.groups.all }),
  })
  const root = extractSiteRootUrl(draft.provider_url ?? '')
  const confirmation =
    (forceConfirmation || requiresSiteRootConfirmation(saved, draft)) &&
    !draft.clearSiteAccessToken &&
    !draft.siteAccessToken.trim()
  const snapshot = access.snapshot
  return (
    <>
      <div {...stylex.props(styles.siteField)}>
        <TextInput
          label={t('siteAccess.userId')}
          placeholder="New-Api-User"
          value={draft.siteUserId}
          isDisabled={disabled}
          onChange={(value) => onChange({ siteUserId: value })}
        />
        <span {...stylex.props(styles.rootInfo)}>
          {root
            ? `${t('siteAccess.root')} ${root} · ${t('siteAccess.refresh')}`
            : t('siteAccess.missingRoot')}
        </span>
      </div>
      <div {...stylex.props(styles.siteField)}>
        <TextInput
          label={t('siteAccess.token')}
          type="password"
          value={draft.siteAccessToken}
          isDisabled={disabled || draft.clearSiteAccessToken}
          placeholder={access.token_configured ? t('siteAccess.placeholder') : ''}
          autoComplete="new-password"
          onChange={(value) => onChange({ siteAccessToken: value })}
        />
        {access.token_configured && (
          <Badge
            variant={draft.clearSiteAccessToken ? 'warning' : 'success'}
            label={t(draft.clearSiteAccessToken ? 'siteAccess.clearing' : 'siteAccess.encrypted')}
          />
        )}
        {access.token_configured && (
          <Button
            size="sm"
            variant="ghost"
            label={t(draft.clearSiteAccessToken ? 'siteAccess.undo' : 'siteAccess.clear')}
            isDisabled={disabled}
            onClick={() =>
              onChange({ clearSiteAccessToken: !draft.clearSiteAccessToken, siteAccessToken: '' })
            }
          />
        )}
      </div>
      {(confirmation || snapshot) && (
        <section aria-label={t('siteAccess.title')} {...stylex.props(styles.details)}>
          {confirmation && (
            <div {...stylex.props(styles.confirm)}>
              <span>
                {saved.site_access.site_root_url} → {root || '—'}
              </span>
              <Switch
                label={t('siteAccess.confirm')}
                value={draft.confirmSiteRootChange}
                isDisabled={disabled}
                onChange={(value) => onChange({ confirmSiteRootChange: value })}
              />
            </div>
          )}
          {snapshot && (
            <div {...stylex.props(styles.ledger)}>
              <span>
                {t('siteAccess.balance')}:{' '}
                {snapshot.remaining_amount === null
                  ? '—'
                  : `${snapshot.currency_symbol}${snapshot.remaining_amount.toFixed(2)}`}
              </span>
              <span>
                {t('siteAccess.ratio')}: {snapshot.effective_group || '—'} ·{' '}
                {snapshot.group_ratio === null ? '—' : `${formatGroupRatio(snapshot.group_ratio)}x`}
              </span>
              <RouteLink to={`/announcements?site=${encodeURIComponent(snapshot.site_root_url)}`}>
                {t('siteAccess.announcements')}: {snapshot.announcement_count}
              </RouteLink>
              <span>
                {t('siteAccess.lastSync')}: {t(`siteAccess.status.${snapshot.status}`)}
                {' · '}
                {snapshot.fetched_at_ms === null
                  ? '—'
                  : intl.formatDate(snapshot.fetched_at_ms, {
                      dateStyle: 'short',
                      timeStyle: 'short',
                    })}
              </span>
              {snapshot.status === 'error' && (
                <>
                  <span {...stylex.props(styles.failureHelp)}>
                    {snapshot.last_error || t('siteAccess.failureHelp')}
                  </span>
                  <Button
                    size="sm"
                    variant="secondary"
                    label={t('siteAccess.retry')}
                    isLoading={retry.isPending}
                    onClick={() => retry.mutate()}
                  />
                </>
              )}
            </div>
          )}
        </section>
      )}
    </>
  )
}
const styles = stylex.create({
  details: { gridColumn: '1 / -1', minWidth: 0 },
  rootInfo: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    overflowWrap: 'anywhere',
    minWidth: 0,
  },
  siteField: { minWidth: 0 },
  confirm: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-2)',
    padding: 'var(--space-2) 0',
    flexWrap: 'wrap',
    overflowWrap: 'anywhere',
  },
  ledger: {
    display: 'flex',
    gap: 'var(--space-4)',
    alignItems: 'center',
    flexWrap: 'wrap',
    overflowWrap: 'anywhere',
    padding: 'var(--space-2)',
    marginTop: 'var(--space-3)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    backgroundColor: 'var(--color-surface-sunken)',
    fontVariantNumeric: 'tabular-nums',
  },
  failureHelp: { color: 'var(--color-warning)', fontSize: 'var(--text-sm)' },
})
