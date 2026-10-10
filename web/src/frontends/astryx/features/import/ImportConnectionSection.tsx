import * as stylex from '@stylexjs/stylex'
import { Field, Selector, Switch, TextInput } from '@astryxdesign/core'

import type { ChannelDto } from '@shared/control/resources/channels'
import { proxyMutation } from '@shared/control/resources/proxy'
import type { ProxyConfiguredMode } from '@shared/control/types'

import {
  hasUpstreamBaseURLVersionMismatch,
  isValidUpstreamBaseURL,
} from '@shared/lib/upstream-base-url'
import type { ImportProxyDraft } from '@shared/domain/import/model-draft'

import { useT } from '../../app/i18n'
import { plainTextInputAttrs } from '../../components/input-attrs'

const styles = stylex.create({
  root: {
    minWidth: 0,
    marginTop: '12px',
  },
  fields: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'minmax(180px, 260px) minmax(0, 1fr)',
      '@media (max-width: 860px)': 'minmax(0, 1fr)',
    },
    alignItems: 'start',
    gap: '10px 14px',
  },
  params: {
    display: 'grid',
    gridColumn: '1 / -1',
    gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 280px), 1fr))',
    gap: '10px 14px',
  },
  fullWidth: {
    gridColumn: { default: '1 / -1', '@media (max-width: 860px)': 'auto' },
    minWidth: 0,
  },
  minWidth: {
    minWidth: 0,
  },

  proxyControls: {
    display: 'flex',
    minWidth: 0,
    alignItems: { default: 'center', '@media (max-width: 860px)': 'stretch' },
    flexDirection: { default: 'row', '@media (max-width: 860px)': 'column' },
    gap: 'var(--space-2)',
  },
  proxyInput: {
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: 'auto',
    minWidth: 0,
  },
  urlInput: {
    fontFamily: 'var(--font-mono)',
  },
  switchRow: {
    display: 'flex',
    minHeight: { default: 'var(--control-xs)', '@media (max-width: 860px)': 'var(--touch-target)' },
    alignItems: 'center',
    gap: 'var(--space-3)',
  },
  urlInputWrap: {
    minWidth: 0,
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: 'auto',
  },
  disabledReason: {
    margin: 0,
    marginTop: '4px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
})

export interface ImportConnectionSectionProps {
  channel: ChannelDto | null
  name: string
  providerUrl: string

  params: Record<string, string>
  proxy: ImportProxyDraft
  paramErrors: Readonly<Record<string, string>>
  baseUrlOverrideEnabled: boolean
  disabled?: boolean
  proxyDisabled?: boolean
  onNameChange(value: string): void
  onProviderUrlChange(value: string): void

  onParamChange(key: string, value: string): void
  onProxyChange(value: ImportProxyDraft): void
  onBaseUrlOverrideChange(enabled: boolean): void
  onParamBlur(key: string): void
}

export function ImportConnectionSection({
  channel,
  name,
  providerUrl,

  params,
  proxy,
  paramErrors,
  baseUrlOverrideEnabled,
  disabled = false,
  proxyDisabled = false,
  onNameChange,
  onProviderUrlChange,

  onParamChange,
  onProxyChange,
  onBaseUrlOverrideChange,
  onParamBlur,
}: ImportConnectionSectionProps) {
  const t = useT()
  const proxySupported = channel?.capabilities.outbound_proxy === true
  const proxyModeOptions = [
    { value: 'inherit', label: t('common.proxy.inherit.group') },
    { value: 'direct', label: t('common.proxy.mode.direct') },
    { value: 'custom', label: t('common.proxy.mode.custom') },
  ]
  const proxyError =
    proxySupported && proxy.mode === 'custom' && proxyMutation(proxy.mode, proxy.url) === undefined
      ? t('common.proxy.invalid')
      : undefined
  const proxyDescription =
    !proxySupported || proxy.mode === 'custom' ? undefined : t(`common.proxy.help.${proxy.mode}`)
  const proxyDisabledReason = !proxySupported
    ? t('common.proxy.unsupportedHelp')
    : proxyDisabled
      ? t('import.connection.proxyLocked')
      : undefined

  function updateProxyMode(value: string): void {
    if (value !== 'inherit' && value !== 'direct' && value !== 'custom') return
    onProxyChange({ mode: value as ProxyConfiguredMode, url: '' })
  }

  function updateProxyURL(value: string): void {
    if (proxy.mode !== 'custom') return
    onProxyChange({ mode: 'custom', url: value })
  }

  function fieldError(key: string): string {
    return paramErrors[key] ?? ''
  }

  function isOptionalBaseURL(key: string, required: boolean): boolean {
    return key === 'base_url' && !required
  }

  function baseURLDescription(): string {
    switch (channel?.channel_id) {
      case 'gpt_load':
        return t('import.connection.gptLoadUrlDescription')
      case 'newapi':
        return t('import.connection.newApiUrlDescription')
      case 'cliproxyapi':
        return t('import.connection.cpaUrlDescription')
      case 'sub2api':
        return t('import.connection.sub2ApiUrlDescription')
      case 'openai_compatible':
        return t('import.connection.compatibleUrlDescription')
    }
    if (!channel?.default_base_url) return t('import.connection.urlDescription')
    return t('import.connection.urlDescriptionWithDefault', {
      url: channel.default_base_url,
    })
  }

  function baseURLVersionWarning(key: string): string | undefined {
    if (key !== 'base_url' || !channel?.default_base_url) return undefined
    const value = params[key]?.trim() ?? ''
    if (!value || !isValidUpstreamBaseURL(value)) return undefined
    return hasUpstreamBaseURLVersionMismatch(channel.default_base_url, value)
      ? t('import.connection.urlVersionWarning')
      : undefined
  }

  return (
    <div {...stylex.props(styles.root)}>
      <div {...stylex.props(styles.fields)}>
        <div {...stylex.props(styles.minWidth)}>
          <TextInput
            value={name}
            isDisabled={disabled}
            autoComplete="off"
            placeholder={t('import.connection.namePlaceholder')}
            label={t('import.connection.name')}
            isOptional
            onChange={onNameChange}
          />
        </div>

        {channel !== null && channel.param_fields.length > 0 && (
          <div {...stylex.props(styles.params)}>
            {channel.param_fields.map((param) =>
              isOptionalBaseURL(param.key, param.required) ? (
                <div key={param.key} {...stylex.props(styles.minWidth)}>
                  <Field
                    label={t('import.connection.customUrl')}
                    description={baseURLDescription()}
                    inputID="import-channel-base-url-override"
                    isRequired={baseUrlOverrideEnabled}
                    status={
                      fieldError(param.key)
                        ? { type: 'error', message: fieldError(param.key) }
                        : baseURLVersionWarning(param.key)
                          ? { type: 'warning', message: baseURLVersionWarning(param.key) }
                          : undefined
                    }
                  >
                    <div {...stylex.props(styles.switchRow)}>
                      <Switch
                        value={baseUrlOverrideEnabled}
                        isDisabled={disabled}
                        label={t('import.connection.customUrl')}
                        isLabelHidden
                        onChange={(checked) => onBaseUrlOverrideChange(checked)}
                      />
                      {baseUrlOverrideEnabled && (
                        <div {...stylex.props(styles.urlInputWrap)}>
                          <TextInput
                            xstyle={styles.urlInput}
                            data-gptload-mono
                            label={t('import.connection.customUrl')}
                            isLabelHidden
                            value={params[param.key] ?? ''}
                            {...{ inputMode: 'url' as const, autoCapitalize: 'none' }}
                            isRequired
                            isDisabled={disabled}
                            status={fieldError(param.key) ? { type: 'error' } : undefined}
                            autoComplete="off"
                            {...plainTextInputAttrs}
                            placeholder="https://"
                            onChange={(value) => onParamChange(param.key, value)}
                            onBlur={() => onParamBlur(param.key)}
                          />
                        </div>
                      )}
                    </div>
                  </Field>
                </div>
              ) : (
                <div key={param.key} {...stylex.props(styles.minWidth)}>
                  <TextInput
                    label={param.key === 'base_url' ? t('import.connection.url') : param.label}
                    description={param.key === 'base_url' ? baseURLDescription() : undefined}
                    data-gptload-mono={param.input_kind === 'url' || undefined}
                    value={params[param.key] ?? ''}
                    {...{
                      inputMode: param.input_kind === 'url' ? ('url' as const) : ('text' as const),
                      autoCapitalize: 'none',
                    }}
                    isDisabled={disabled}
                    autoComplete="off"
                    {...plainTextInputAttrs}
                    placeholder={param.input_kind === 'url' ? 'https://' : undefined}
                    onChange={(value) => onParamChange(param.key, value)}
                    onBlur={() => onParamBlur(param.key)}
                    isRequired={param.required}
                    status={
                      fieldError(param.key)
                        ? { type: 'error', message: fieldError(param.key) }
                        : param.key === 'base_url' && baseURLVersionWarning(param.key)
                          ? { type: 'warning', message: baseURLVersionWarning(param.key) }
                          : undefined
                    }
                  />
                </div>
              ),
            )}
          </div>
        )}

        <div {...stylex.props(styles.fullWidth)}>
          <TextInput
            label={t('group.settings.base.providerUrl')}
            isOptional
            description={t('group.settings.base.providerUrlHelp')}
            data-gptload-mono
            value={providerUrl}
            {...{ inputMode: 'url' as const, autoCapitalize: 'none' }}
            isDisabled={disabled}
            autoComplete="off"
            {...plainTextInputAttrs}
            placeholder="https://"
            onChange={onProviderUrlChange}
          />
        </div>

        {channel && (
          <div {...stylex.props(styles.fullWidth)}>
            <Field
              label={t('common.proxy.title')}
              description={proxyDescription}
              inputID="import-group-proxy-mode"
              status={proxyError ? { type: 'error', message: proxyError } : undefined}
            >
              <div>
                <div {...stylex.props(styles.proxyControls)}>
                  <Selector
                    id="import-group-proxy-mode"
                    label={t('common.proxy.modeLabel')}
                    isLabelHidden
                    options={proxyModeOptions}
                    size="sm"
                    value={proxySupported ? proxy.mode : 'inherit'}
                    isDisabled={disabled || proxyDisabled || !proxySupported}
                    onChange={updateProxyMode}
                  />
                  {proxySupported && proxy.mode === 'custom' && (
                    <TextInput
                      xstyle={styles.proxyInput}
                      data-gptload-mono
                      value={proxy.url}
                      label={t('common.proxy.urlLabel')}
                      isLabelHidden
                      placeholder={t('common.proxy.placeholder')}
                      size="sm"
                      autoComplete="off"
                      {...plainTextInputAttrs}
                      isDisabled={disabled || proxyDisabled}
                      status={proxyError ? { type: 'error', message: proxyError } : undefined}
                      statusVariant="tooltip"
                      onChange={updateProxyURL}
                    />
                  )}
                </div>
                {proxyDisabledReason !== undefined && (
                  <p {...stylex.props(styles.disabledReason)}>{proxyDisabledReason}</p>
                )}
              </div>
            </Field>
          </div>
        )}
      </div>
    </div>
  )
}
