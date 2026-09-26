import { Switch } from '@astryxdesign/core'
import { useIntl } from 'react-intl'

import { ProxyOverrideControl } from '../../components/ProxyOverrideControl'
import { SettingRow } from '../../components/setting-chrome'
import { useT } from '../../app/i18n'
import type { ProxyConfiguredMode, ProxyViewDto } from '@shared/control/types'
import {
  proxyOverrideToggleMode,
} from '@shared/control/resources/proxy'
import type { TimeoutSettingKey } from '@shared/control/resources/settings'
import { formatInteger } from '@shared/lib/format'
import { isValidTimeout } from '@shared/domain/settings/settings-patch'
import {
  numberText,
  SettingNumberInput,
  SettingsSectionFrame,
  useSettingTools,
  type SettingsSectionProps,
} from './section-tools'

const timeoutKeys = [
  'first_byte_timeout',
  'request_timeout',
  'stream_idle_timeout',
] as const satisfies readonly TimeoutSettingKey[]

export interface ConnectionSectionProps extends SettingsSectionProps {
  proxy: ProxyViewDto
  proxyMode: ProxyConfiguredMode
  proxyEndpoint: string
  onProxyModeChange: (value: ProxyConfiguredMode) => void
  onProxyEndpointChange: (value: string) => void
}

export function ConnectionSection({
  proxy,
  proxyMode,
  proxyEndpoint,
  onProxyModeChange,
  onProxyEndpointChange,
  ...props
}: ConnectionSectionProps) {
  const { base, draft, disabled } = props
  const t = useT()
  const tools = useSettingTools(props)
  const { locale } = useIntl()

  const websocketValue = base.settings.values.responses_websocket_enabled
    ? t('settings.runtime.enabled')
    : t('settings.runtime.disabled')

  // The proxy row mirrors the override semantics: inherit = not overridden,
  // direct/custom = explicit override.
  const proxyOverridden = proxyMode !== 'inherit'
  const proxyPendingRestore = proxy.configured_mode !== 'inherit' && proxyMode === 'inherit'
  const proxyEffectiveLabel =
    proxy.display_url ?? t(`common.proxy.mode.${proxy.effective_mode}`)
  const proxyValue = proxyOverridden
    ? t('settings.runtime.overrideValue')
    : proxyPendingRestore
      ? t('settings.runtime.resetPending')
      : proxyEffectiveLabel
  const proxySourceLabel = proxyOverridden
    ? t('settings.runtime.overrideSource')
    : proxyPendingRestore
      ? t('settings.runtime.pendingRestoreSource')
      : t('settings.runtime.defaultSource')
  const proxyActionLabel = proxyOverridden
    ? t('settings.runtime.restoreDefault')
    : t('settings.runtime.override')

  const timeoutValue = (key: TimeoutSettingKey): string => {
    if (tools.isPendingRestore(key)) return t('settings.runtime.resetPending')
    return t('settings.runtime.effectiveValue', {
      value: formatInteger(base.settings.values[key], locale),
    })
  }
  const timeoutError = (key: TimeoutSettingKey): string | undefined =>
    tools.hasOverride(key) && !isValidTimeout(draft.values[key])
      ? t('settings.runtime.timeoutError')
      : undefined

  return (
    <SettingsSectionFrame
      id="settings-connection"
      title={t('settings.runtime.title')}
      description={t('settings.runtime.description')}
    >
      <SettingRow
        label={t('settings.runtime.responses_websocket_enabled')}
        value={
          tools.isPendingRestore('responses_websocket_enabled')
            ? t('settings.runtime.resetPending')
            : websocketValue
        }
        help={t('settings.runtime.websocketHelp')}
        sourceLabel={tools.sourceLabel('responses_websocket_enabled')}
        actionLabel={tools.actionLabel('responses_websocket_enabled')}
        overridden={tools.hasOverride('responses_websocket_enabled')}
        pendingRestore={tools.isPendingRestore('responses_websocket_enabled')}
        disabled={disabled}
        onToggle={() => tools.toggleOverride('responses_websocket_enabled')}
        control={
          <Switch
            id="settings-value-responses_websocket_enabled"
            value={draft.values.responses_websocket_enabled}
            isDisabled={disabled}
            label={t('settings.runtime.responses_websocket_enabled')}
            isLabelHidden
            size="sm"
            onChange={(value) =>
              tools.update('responses_websocket_enabled', (next) => {
                next.values.responses_websocket_enabled = value
              })
            }
          />
        }
      />

      <SettingRow
        label={t('common.proxy.title')}
        value={proxyValue}
        sourceLabel={proxySourceLabel}
        actionLabel={proxyActionLabel}
        overridden={proxyOverridden}
        pendingRestore={proxyPendingRestore}
        disabled={disabled}
        onToggle={() => {
          onProxyModeChange(proxyOverrideToggleMode(proxy, proxyOverridden))
          onProxyEndpointChange('')
        }}
        control={
          <ProxyOverrideControl
            base={proxy}
            mode={proxyMode}
            endpoint={proxyEndpoint}
            disabled={disabled}
            onModeChange={onProxyModeChange}
            onEndpointChange={onProxyEndpointChange}
          />
        }
      />

      {timeoutKeys.map((key) => (
        <SettingRow
          key={key}
          label={t(`settings.runtime.${key}`)}
          value={timeoutValue(key)}
          sourceLabel={tools.sourceLabel(key)}
          actionLabel={tools.actionLabel(key)}
          overridden={tools.hasOverride(key)}
          pendingRestore={tools.isPendingRestore(key)}
          divided={key !== 'stream_idle_timeout'}
          disabled={disabled}
          onToggle={() => tools.toggleOverride(key)}
          control={
            <SettingNumberInput
              id={`settings-value-${key}`}
              value={numberText(draft.values[key])}
              label={t('settings.runtime.valueFor', { field: t(`settings.runtime.${key}`) })}
              error={timeoutError(key)}
              unit={t('settings.runtime.seconds')}
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
