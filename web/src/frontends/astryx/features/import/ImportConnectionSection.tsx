import * as stylex from '@stylexjs/stylex'
import { Field, Selector, TextInput } from '@astryxdesign/core'
import { ChevronRight } from 'lucide-react'

import type { ChannelDto } from '@shared/control/resources/channels'
import { proxyMutation } from '@shared/control/resources/proxy'
import type { ProxyConfiguredMode } from '@shared/control/types'

import {
  hasUpstreamBaseURLVersionMismatch,
  isValidUpstreamBaseURL,
} from '@shared/lib/upstream-base-url'
import { extractSiteRootUrl, isValidProviderUrlForWrite } from '@shared/domain/groups/settings/group-settings-patch'
import type { ImportProxyDraft } from '@shared/domain/import/model-draft'

import { useT } from '../../app/i18n'
import { plainTextInputAttrs } from '../../components/input-attrs'

const styles = stylex.create({
  root: {
    minWidth: 0,
    marginTop: '8px',
  },
  fields: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'repeat(3, minmax(0, 1fr))',
      '@media (max-width: 1100px)': 'minmax(220px, 0.8fr) minmax(0, 1.2fr)',
      '@media (max-width: 860px)': 'minmax(0, 1fr)',
    },
    alignItems: { default: 'end', '@media (max-width: 1100px)': 'start' },
    gap: '8px 14px',
  },
  siteHeading: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    flexWrap: 'wrap',
    marginBottom: 'var(--space-2)',
  },
  siteRoot: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    minWidth: 0,
    overflowWrap: 'anywhere',
  },
  siteGrid: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'repeat(2, minmax(0, 1fr))',
      '@media (max-width: 680px)': 'minmax(0, 1fr)',
    },
    gap: 'var(--space-2)',
  },
  params: {
    display: { default: 'contents', '@media (max-width: 1100px)': 'grid' },
    gridColumn: '1 / -1',
    gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 280px), 1fr))',
    gap: '8px 14px',
  },
  fullWidth: {
    gridColumn: { default: '1 / -1', '@media (max-width: 860px)': 'auto' },
    minWidth: 0,
  },
  minWidth: {
    minWidth: 0,
  },
  moreDetails: {
    gridColumn: { default: '1 / -1', '@media (max-width: 860px)': 'auto' },
    minWidth: 0,
  },
  moreSummary: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 'var(--space-1)',
    color: 'var(--color-text-muted)',
    cursor: 'pointer',
    fontSize: 'var(--text-sm)',
    fontWeight: 560,
    userSelect: 'none',
    listStyle: 'none',
  },
  moreIcon: {
    transform: 'rotate(0deg)',
    transition: 'transform 140ms ease',
  },
  moreBody: {
    marginTop: '8px',
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
  siteUserId: string
  siteAccessToken: string

  params: Record<string, string>
  proxy: ImportProxyDraft
  paramErrors: Readonly<Record<string, string>>

  disabled?: boolean
  onNameChange(value: string): void
  onProviderUrlChange(value: string): void
  onSiteUserIdChange(value: string): void
  onSiteAccessTokenChange(value: string): void

  onParamChange(key: string, value: string): void
  onProxyChange(value: ImportProxyDraft): void

  onParamBlur(key: string): void
}

export function ImportConnectionSection({
  channel,
  name,
  providerUrl,
  siteUserId,
  siteAccessToken,

  params,
  proxy,
  paramErrors,

  disabled = false,
  onNameChange,
  onProviderUrlChange,
  onSiteUserIdChange,
  onSiteAccessTokenChange,

  onParamChange,
  onProxyChange,

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
  const proxyDisabledReason = !proxySupported ? t('common.proxy.unsupportedHelp') : undefined

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

        <div {...stylex.props(styles.minWidth)}>
          <TextInput
            label={t('group.settings.base.providerUrl')}
            isOptional
            data-gptload-mono
            value={providerUrl}
            {...{ inputMode: 'url' as const, autoCapitalize: 'none' }}
            isDisabled={disabled}
            autoComplete="off"
            {...plainTextInputAttrs}
            placeholder="https://"
            status={
              providerUrl.trim() && !isValidProviderUrlForWrite(providerUrl)
                ? { type: 'error', message: t('group.settings.base.providerUrlError') }
                : undefined
            }
            onChange={onProviderUrlChange}
          />
        </div>

        {channel !== null && channel.param_fields.length > 0 && (
          <div {...stylex.props(styles.params)}>
            {channel.param_fields.map((param) => (
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
                  placeholder={
                    param.input_kind === 'url' ? channel.default_base_url || 'https://' : undefined
                  }
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
            ))}
          </div>
        )}

        <section {...stylex.props(styles.fullWidth)} aria-labelledby="import-site-access">
          <div {...stylex.props(styles.siteHeading)}>
            <h3 id="import-site-access">{t('siteAccess.title')}</h3>
            <span {...stylex.props(styles.siteRoot)}>
              {extractSiteRootUrl(providerUrl)
                ? `${t('siteAccess.root')} ${extractSiteRootUrl(providerUrl)}`
                : t('siteAccess.missingRoot')}
            </span>
          </div>
          <div {...stylex.props(styles.siteGrid)}>
            <TextInput
              label={t('siteAccess.userId')}
              value={siteUserId}
              isDisabled={disabled}
              onChange={onSiteUserIdChange}
            />
            <TextInput
              label={t('siteAccess.token')}
              type="password"
              autoComplete="new-password"
              value={siteAccessToken}
              isDisabled={disabled}
              onChange={onSiteAccessTokenChange}
            />
          </div>
        </section>

        {channel && (
          <details
            {...stylex.props(styles.moreDetails)}
            open={proxy.mode !== 'inherit' || Boolean(proxyError) || undefined}
          >
            <summary {...stylex.props(styles.moreSummary)}>
              <ChevronRight size={15} aria-hidden="true" {...stylex.props(styles.moreIcon)} />
              {t('import.connection.moreOptions')}
            </summary>
            <div {...stylex.props(styles.moreBody)}>
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
                      isDisabled={disabled || !proxySupported}
                      onChange={updateProxyMode}
                    />
                    {proxySupported && proxy.mode === 'custom' && (
                      <TextInput
                        controlXstyle={styles.proxyInput}
                        data-gptload-mono
                        value={proxy.url}
                        label={t('common.proxy.urlLabel')}
                        isLabelHidden
                        placeholder={t('common.proxy.placeholder')}
                        size="sm"
                        autoComplete="off"
                        {...plainTextInputAttrs}
                        isDisabled={disabled}
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
          </details>
        )}
      </div>
    </div>
  )
}
