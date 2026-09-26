import { useIntl } from 'react-intl'

import { SettingRow } from '../../components/setting-chrome'
import { useT } from '../../app/i18n'
import { formatInteger } from '@shared/lib/format'
import {
  isValidNonNegativeInteger,
  isValidTimeout,
} from '@shared/domain/settings/settings-patch'
import type {
  PolicyCountSettingKey,
} from '@shared/control/resources/settings'
import {
  numberText,
  SettingNumberInput,
  SettingsSectionFrame,
  useSettingTools,
  type SettingsSectionProps,
} from './section-tools'

const policyRows = [
  { key: 'retry_count', helpKey: 'retryCountHelp' },
  { key: 'blacklist_threshold', helpKey: 'blacklistThresholdHelp' },
] as const

export function ReliabilitySection(props: SettingsSectionProps) {
  const { base, draft, disabled } = props
  const t = useT()
  const tools = useSettingTools(props)
  const { locale } = useIntl()

  const policyValue = (key: PolicyCountSettingKey): string => {
    if (tools.isPendingRestore(key)) return t('settings.runtime.resetPending')
    return t('settings.runtime.effectiveCount', {
      value: formatInteger(base.settings.values[key], locale),
    })
  }
  const policyCountError = (key: PolicyCountSettingKey): string | undefined =>
    tools.hasOverride(key) && !isValidNonNegativeInteger(draft.values[key])
      ? t('settings.runtime.nonNegativeIntegerError')
      : undefined
  const blacklistReleaseError = (): string | undefined =>
    tools.hasOverride('blacklist_release_seconds') &&
    !isValidTimeout(draft.values.blacklist_release_seconds)
      ? t('settings.runtime.timeoutError')
      : undefined

  return (
    <SettingsSectionFrame
      id="settings-reliability"
      title={t('settings.reliability.title')}
      description={t('settings.reliability.description')}
    >
      {policyRows.map((policy) => (
        <SettingRow
          key={policy.key}
          label={t(`settings.runtime.${policy.key}`)}
          value={policyValue(policy.key)}
          help={t(`settings.runtime.${policy.helpKey}`)}
          sourceLabel={tools.sourceLabel(policy.key)}
          actionLabel={tools.actionLabel(policy.key)}
          overridden={tools.hasOverride(policy.key)}
          pendingRestore={tools.isPendingRestore(policy.key)}
          disabled={disabled}
          onToggle={() => tools.toggleOverride(policy.key)}
          control={
            <SettingNumberInput
              id={`settings-value-${policy.key}`}
              value={numberText(draft.values[policy.key])}
              label={t('settings.runtime.valueFor', {
                field: t(`settings.runtime.${policy.key}`),
              })}
              error={policyCountError(policy.key)}
              unit={t('settings.runtime.countUnit')}
              min="0"
              disabled={disabled}
              onChange={(value) =>
                tools.update(policy.key, (next) => {
                  next.values[policy.key] = tools.parseCount(value)
                })
              }
            />
          }
        />
      ))}

      <SettingRow
        label={t('settings.runtime.blacklist_release_seconds')}
        value={
          tools.isPendingRestore('blacklist_release_seconds')
            ? t('settings.runtime.resetPending')
            : t('settings.runtime.effectiveValue', {
                value: formatInteger(
                  base.settings.values.blacklist_release_seconds,
                  locale,
                ),
              })
        }
        help={t('settings.runtime.blacklistReleaseHelp')}
        sourceLabel={tools.sourceLabel('blacklist_release_seconds')}
        actionLabel={tools.actionLabel('blacklist_release_seconds')}
        overridden={tools.hasOverride('blacklist_release_seconds')}
        pendingRestore={tools.isPendingRestore('blacklist_release_seconds')}
        divided={false}
        disabled={disabled}
        onToggle={() => tools.toggleOverride('blacklist_release_seconds')}
        control={
          <SettingNumberInput
            id="settings-value-blacklist_release_seconds"
            value={numberText(draft.values.blacklist_release_seconds)}
            label={t('settings.runtime.valueFor', {
              field: t('settings.runtime.blacklist_release_seconds'),
            })}
            error={blacklistReleaseError()}
            unit={t('settings.runtime.seconds')}
            disabled={disabled}
            onChange={(value) =>
              tools.update('blacklist_release_seconds', (next) => {
                next.values.blacklist_release_seconds = tools.parseCount(value)
              })
            }
          />
        }
      />
    </SettingsSectionFrame>
  )
}
