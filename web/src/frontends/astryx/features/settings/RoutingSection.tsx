import { SegmentedControl, SegmentedControlItem, Switch } from '@astryxdesign/core'
import { useIntl } from 'react-intl'

import { SettingRow } from '../../components/setting-chrome'
import { useT } from '../../app/i18n'
import { routeStrategies } from '@shared/control/types'
import { formatInteger } from '@shared/lib/format'
import { isValidAffinityCapacity, isValidTimeout } from '@shared/domain/settings/settings-patch'
import {
  numberText,
  SettingNumberInput,
  SettingsSectionFrame,
  useSettingTools,
  type SettingsSectionProps,
} from './section-tools'

const numericKeys = ['affinity_ttl', 'affinity_capacity'] as const

export function RoutingSection(props: SettingsSectionProps) {
  const { base, draft, disabled } = props
  const t = useT()
  const tools = useSettingTools(props)
  const { locale } = useIntl()

  const enabledValue = base.settings.values.affinity_enabled
    ? t('settings.runtime.enabled')
    : t('settings.runtime.disabled')

  const setRouteStrategy = (value: string) => {
    const strategy = routeStrategies.find((candidate) => candidate === value)
    if (strategy === undefined) return
    tools.update('route_strategy', (next) => {
      next.values.route_strategy = strategy
    })
  }

  const numberFieldError = (key: (typeof numericKeys)[number]): string | undefined => {
    if (!tools.hasOverride(key)) return undefined
    const valid =
      key === 'affinity_ttl'
        ? isValidTimeout(draft.values[key])
        : isValidAffinityCapacity(draft.values[key])
    return valid ? undefined : t(`settings.affinity.${key}Error`)
  }

  const numberFieldValue = (key: (typeof numericKeys)[number]): string => {
    if (tools.isPendingRestore(key)) return t('settings.runtime.resetPending')
    return t(`settings.affinity.${key}Effective`, {
      value: formatInteger(base.settings.values[key], locale),
    })
  }

  return (
    <SettingsSectionFrame
      id="settings-routing"
      title={t('settings.affinity.title')}
      description={t('settings.affinity.description')}
    >
      <SettingRow
        label={t('settings.runtime.route_strategy')}
        value={
          tools.isPendingRestore('route_strategy')
            ? t('settings.runtime.resetPending')
            : t(`settings.runtime.routeStrategies.${base.settings.values.route_strategy}`)
        }
        help={t('settings.runtime.routeStrategyHelp')}
        sourceLabel={tools.sourceLabel('route_strategy')}
        actionLabel={tools.actionLabel('route_strategy')}
        overridden={tools.hasOverride('route_strategy')}
        pendingRestore={tools.isPendingRestore('route_strategy')}
        disabled={disabled}
        onToggle={() => tools.toggleOverride('route_strategy')}
        control={
          <SegmentedControl
            value={draft.values.route_strategy}
            label={t('settings.runtime.route_strategy')}
            size="sm"
            isDisabled={disabled}
            onChange={setRouteStrategy}
          >
            {routeStrategies.map((value) => (
              <SegmentedControlItem
                key={value}
                value={value}
                label={t(`settings.runtime.routeStrategies.${value}`)}
              />
            ))}
          </SegmentedControl>
        }
      />

      <SettingRow
        label={t('settings.affinity.affinity_enabled')}
        value={
          tools.isPendingRestore('affinity_enabled')
            ? t('settings.runtime.resetPending')
            : enabledValue
        }
        help={t('settings.affinity.enabledHelp')}
        sourceLabel={tools.sourceLabel('affinity_enabled')}
        actionLabel={tools.actionLabel('affinity_enabled')}
        overridden={tools.hasOverride('affinity_enabled')}
        pendingRestore={tools.isPendingRestore('affinity_enabled')}
        disabled={disabled}
        onToggle={() => tools.toggleOverride('affinity_enabled')}
        control={
          <Switch
            value={draft.values.affinity_enabled}
            isDisabled={disabled}
            label={t('settings.affinity.affinity_enabled')}
            isLabelHidden
            size="sm"
            onChange={(value) =>
              tools.update('affinity_enabled', (next) => {
                next.values.affinity_enabled = value
              })
            }
          />
        }
      />

      {numericKeys.map((key) => (
        <SettingRow
          key={key}
          label={t(`settings.affinity.${key}`)}
          value={numberFieldValue(key)}
          sourceLabel={tools.sourceLabel(key)}
          actionLabel={tools.actionLabel(key)}
          overridden={tools.hasOverride(key)}
          pendingRestore={tools.isPendingRestore(key)}
          divided={key !== 'affinity_capacity'}
          disabled={disabled}
          onToggle={() => tools.toggleOverride(key)}
          control={
            <SettingNumberInput
              id={`settings-value-${key}`}
              value={numberText(draft.values[key])}
              label={t('settings.runtime.valueFor', { field: t(`settings.affinity.${key}`) })}
              error={numberFieldError(key)}
              unit={
                key === 'affinity_ttl'
                  ? t('settings.runtime.seconds')
                  : t('settings.affinity.entries')
              }
              max={key === 'affinity_ttl' ? '9223372036' : '1000000'}
              disabled={disabled}
              onChange={(value) =>
                tools.update(key, (next) => {
                  next.values[key] = tools.parseCount(value)
                })
              }
            />
          }
        />
      ))}
    </SettingsSectionFrame>
  )
}
