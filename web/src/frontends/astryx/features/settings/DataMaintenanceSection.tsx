import { Switch } from '@astryxdesign/core'
import { useEffect, useRef, useState } from 'react'
import { useIntl } from 'react-intl'

import { SettingRow } from '../../components/setting-chrome'
import { useT } from '../../app/i18n'
import { formatInteger } from '@shared/lib/format'
import { isValidRetention } from '@shared/domain/settings/settings-patch'
import {
  SettingNumberInput,
  SettingsSectionFrame,
  useSettingTools,
  type SettingsSectionProps,
} from './section-tools'

const retentionKey = 'request_log_retention_days' as const
const syncKey = 'models_dev_auto_sync_enabled' as const

export function DataMaintenanceSection(props: SettingsSectionProps) {
  const { base, draft, disabled } = props
  const t = useT()
  const tools = useSettingTools(props)
  const { locale } = useIntl()

  // Local string buffer like classic's `retentionInput`: keeps the user's raw
  // keystrokes (e.g. "1e5") instead of echoing back the parsed number.
  const [retentionInput, setRetentionInput] = useState(() =>
    String(draft.values.request_log_retention_days),
  )
  const lastPublishedRetention = useRef<number | undefined>(undefined)
  const draftRetention = draft.values.request_log_retention_days
  useEffect(() => {
    if (!Object.is(draftRetention, lastPublishedRetention.current)) {
      setRetentionInput(String(draftRetention))
    }
  }, [draftRetention])

  const retentionOwned = draft.overrides.has(retentionKey)
  const retentionPendingRestore =
    !retentionOwned && base.settings.overrides.includes(retentionKey)
  const retentionError =
    retentionOwned && !isValidRetention(draft.values.request_log_retention_days)
      ? t('settings.logs.retentionError')
      : undefined
  const retentionValue = retentionPendingRestore
    ? t('settings.runtime.resetPending')
    : t('settings.logs.effectiveValue', {
        value: formatInteger(base.settings.values.request_log_retention_days, locale),
      })

  const setRetentionValue = (value: string): void => {
    setRetentionInput(value)
    const parsed = tools.parseCount(value)
    lastPublishedRetention.current = parsed
    tools.update(retentionKey, (next) => {
      next.values.request_log_retention_days = parsed
    })
  }

  // Retention renders through the local string buffer, not SettingNumberInput's
  // number→string derivation — same visual, different echo semantics.
  const retentionInputError = retentionError
  const retentionInputId = 'settings-value-request_log_retention_days'

  const syncLocked = draft.readOnly.has(syncKey)
  const syncSourceLabel = syncLocked
    ? t('settings.runtime.environmentSource')
    : tools.sourceLabel(syncKey)
  const syncValue = tools.isPendingRestore(syncKey)
    ? t('settings.runtime.resetPending')
    : base.settings.values.models_dev_auto_sync_enabled
      ? t('settings.runtime.enabled')
      : t('settings.runtime.disabled')

  return (
    <SettingsSectionFrame
      id="settings-data-maintenance"
      title={t('settings.logs.title')}
      description={t('settings.logs.description')}
    >
      <SettingRow
        label={t('settings.logs.retention')}
        value={retentionValue}
        sourceLabel={tools.sourceLabel(retentionKey)}
        actionLabel={tools.actionLabel(retentionKey)}
        overridden={retentionOwned}
        pendingRestore={retentionPendingRestore}
        disabled={disabled}
        onToggle={() => tools.toggleOverride(retentionKey)}
        control={
          <SettingNumberInput
            id={retentionInputId}
            value={retentionInput}
            label={t('settings.runtime.valueFor', { field: t('settings.logs.retention') })}
            error={retentionInputError}
            unit={t('settings.logs.days')}
            max="365"
            disabled={disabled}
            onChange={setRetentionValue}
          />
        }
      />

      <SettingRow
        label={t('settings.runtime.models_dev_auto_sync_enabled')}
        value={syncValue}
        help={
          syncLocked
            ? t('settings.runtime.environmentManaged')
            : t('settings.runtime.modelsDevAutoSyncHelp')
        }
        sourceLabel={syncSourceLabel}
        actionLabel={tools.actionLabel(syncKey)}
        overridden={tools.hasOverride(syncKey)}
        pendingRestore={!syncLocked && tools.isPendingRestore(syncKey)}
        locked={syncLocked}
        divided={false}
        disabled={disabled || syncLocked}
        onToggle={() => tools.toggleOverride(syncKey)}
        control={
          <Switch
            value={draft.values.models_dev_auto_sync_enabled}
            isDisabled={disabled || syncLocked}
            label={t('settings.runtime.models_dev_auto_sync_enabled')}
            isLabelHidden
            size="sm"
            onChange={(value) =>
              tools.update(syncKey, (next) => {
                next.values.models_dev_auto_sync_enabled = value
              })
            }
          />
        }
      />
    </SettingsSectionFrame>
  )
}
