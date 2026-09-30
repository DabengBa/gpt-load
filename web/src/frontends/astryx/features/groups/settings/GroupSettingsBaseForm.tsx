import { Button, Switch, TextInput } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import { useState } from 'react'
import { createPortal } from 'react-dom'

import type { ChannelParamsDto } from '@shared/control/types'
import type { ChannelDto, ChannelFieldDto } from '@shared/control/resources/channels'
import { isValidPriceMultiplier } from '@shared/lib/price-multiplier'
import { useT } from '../../../app/i18n'
import { ChannelPresetPicker } from '../../import/ChannelPresetPicker'

export function GroupSettingsBaseForm({
  section,
  channelId,
  channels,
  selectedChannel,
  channelsLoading,
  channelsError,
  paramFields,
  params,
  name,
  providerUrl,
  priceMultiplier,
  enabled,
  enabledPending = false,
  pending,
  paramsDisabled = false,
  nameError,
  paramErrors,
  showTitle = true,
  showDescription = true,
  unified = false,
  headerActionsTarget,
  onParamChange,
  onChannelSelect,
  onChannelsRetry,
  onNameChange,
  onProviderUrlChange,
  onPriceMultiplierChange,
  onSetEnabled,
}: {
  section: 'general' | 'routing'
  channelId: string
  channels: readonly ChannelDto[]
  selectedChannel: ChannelDto | null
  channelsLoading: boolean
  channelsError: boolean
  paramFields: ChannelFieldDto[]
  params: ChannelParamsDto
  name: string
  providerUrl: string | null
  priceMultiplier: string
  enabled: boolean
  enabledPending?: boolean
  pending: boolean
  paramsDisabled?: boolean
  nameError: string
  paramErrors: Record<string, string>
  showTitle?: boolean
  showDescription?: boolean
  unified?: boolean
  headerActionsTarget?: HTMLElement | null
  onParamChange(key: string, value: string | null): void
  onChannelSelect(channel: ChannelDto): void
  onChannelsRetry(): void
  onNameChange(value: string): void
  onProviderUrlChange(value: string): void
  onPriceMultiplierChange(value: string): void
  onSetEnabled(value: boolean): void
}) {
  const t = useT()
  const [baseUrlOverrideEnabled, setBaseUrlOverrideEnabled] = useState(true)
  const [lastChannelId, setLastChannelId] = useState(channelId)
  if (lastChannelId !== channelId) {
    setLastChannelId(channelId)
    setBaseUrlOverrideEnabled(true)
  }

  const baseUrlParam = params.base_url
  const [lastBaseUrl, setLastBaseUrl] = useState(baseUrlParam)
  if (baseUrlParam !== lastBaseUrl) {
    setLastBaseUrl(baseUrlParam)
    if (baseUrlParam?.trim()) setBaseUrlOverrideEnabled(true)
  }

  function isOptionalBaseURL(field: ChannelFieldDto): boolean {
    return field.key === 'base_url' && !field.required
  }

  function setBaseURLOverride(enabled: boolean): void {
    setBaseUrlOverrideEnabled(enabled)
    if (!enabled) onParamChange('base_url', null)
  }

  function updateParam(field: ChannelFieldDto, value: string): void {
    if (isOptionalBaseURL(field) && !value.trim()) {
      setBaseUrlOverrideEnabled(false)
      onParamChange(field.key, null)
      return
    }
    onParamChange(field.key, value)
  }

  function parameterHelp(field: ChannelFieldDto): string {
    if (field.key === 'base_url') {
      if (channelId === 'gpt_load') return t('group.settings.base.gptLoadUrlDescription')
      if (channelId === 'newapi') return t('group.settings.base.newApiUrlDescription')
      if (channelId === 'cliproxyapi') return t('group.settings.base.cpaUrlDescription')
      if (channelId === 'sub2api') return t('group.settings.base.sub2ApiUrlDescription')
    }
    return t('group.settings.base.urlWarning')
  }

  if (section !== 'general') return null

  const switchRow = (
    <div {...stylex.props(styles.switchRow)}>
      <span {...stylex.props(styles.switchCopy)}>
        <strong {...stylex.props(styles.switchTitle)}>{t('group.settings.base.enabled')}</strong>
        <small {...stylex.props(styles.switchHint)}>{t('group.settings.base.enabledHelp')}</small>
      </span>
      <div {...stylex.props(styles.enabledActions)}>
        <Button
          variant="secondary"
          size="sm"
          isDisabled={pending || enabledPending || enabled}
          onClick={() => onSetEnabled(true)}
          label={t('group.settings.base.enableAll')}
        />
        <Button
          variant="secondary"
          size="sm"
          isDisabled={pending || enabledPending || !enabled}
          onClick={() => onSetEnabled(false)}
          label={t('group.settings.base.disableAll')}
        />
      </div>
    </div>
  )

  return (
    <section id="settings-general" {...stylex.props(styles.section)}>
      {(showTitle || showDescription) && (
        <header>
          {showTitle && (
            <h3 {...stylex.props(styles.sectionHeading)}>{t('group.settings.sections.general')}</h3>
          )}
          {showDescription && (
            <p {...stylex.props(styles.sectionDescription)}>
              {t('group.settings.base.description')}
            </p>
          )}
        </header>
      )}
      <div {...stylex.props(styles.channelField)}>
        <span {...stylex.props(styles.fieldLabel)}>{t('group.settings.base.channel')}</span>
        <small {...stylex.props(styles.fieldHint)}>{t('group.settings.base.channelHelp')}</small>
        <div {...stylex.props(styles.channelPicker)}>
          <ChannelPresetPicker
            value={channelId}
            channels={channels}
            selectedChannel={selectedChannel}
            loading={channelsLoading}
            error={channelsError}
            disabled={pending}
            hideHeader
            compact
            onSelect={onChannelSelect}
            onRetry={onChannelsRetry}
          />
        </div>
      </div>
      <div {...stylex.props(styles.grid)}>
        <label {...stylex.props(styles.field)}>
          <span {...stylex.props(styles.fieldLabel)}>{t('group.settings.base.name')}</span>
          <TextInput
            label={t('group.settings.base.name')}
            isLabelHidden
            value={name}
            isDisabled={pending}
            status={nameError !== '' ? { type: 'error', message: nameError } : undefined}
            onChange={onNameChange}
          />
        </label>
        <label {...stylex.props(styles.field)}>
          <span {...stylex.props(styles.fieldLabel)}>{t('common.priceMultiplier.label')}</span>
          <TextInput
            label={t('common.priceMultiplier.label')}
            isLabelHidden
            value={priceMultiplier}
            isDisabled={pending}
            status={
              !isValidPriceMultiplier(priceMultiplier)
                ? { type: 'error', message: t('common.priceMultiplier.invalid') }
                : undefined
            }
            description={
              isValidPriceMultiplier(priceMultiplier)
                ? t('common.priceMultiplier.groupHelp')
                : undefined
            }
            onChange={onPriceMultiplierChange}
            xstyle={styles.mono}
          />
        </label>
        {paramFields.map((field) => {
          const optionalBaseURL = isOptionalBaseURL(field)
          const error = paramErrors[field.key]
          return (
            <div key={field.key} style={{ display: 'contents' }}>
              {optionalBaseURL && (
                <div {...stylex.props(styles.field, styles.baseUrlToggle)}>
                  <span {...stylex.props(styles.fieldLabel)}>
                    {t('group.settings.base.customUrl')}
                  </span>
                  <div {...stylex.props(styles.baseUrlSwitch)}>
                    <small {...stylex.props(styles.fieldHint)}>
                      {t('group.settings.base.customUrlHelp')}
                    </small>
                    <Switch
                      value={baseUrlOverrideEnabled}
                      isDisabled={pending || paramsDisabled}
                      label={t('group.settings.base.customUrl')}
                      isLabelHidden
                      onChange={setBaseURLOverride}
                    />
                  </div>
                </div>
              )}
              {(!optionalBaseURL || baseUrlOverrideEnabled) && (
                <label
                  {...stylex.props(
                    styles.field,
                    optionalBaseURL ? styles.baseUrlInput : styles.wide,
                  )}
                >
                  <span {...stylex.props(styles.fieldLabel)}>
                    {field.key === 'base_url' ? t('group.settings.base.upstreamUrl') : field.label}
                  </span>
                  <TextInput
                    label={
                      field.key === 'base_url' ? t('group.settings.base.upstreamUrl') : field.label
                    }
                    isLabelHidden
                    type="text"
                    value={params[field.key] ?? ''}
                    isDisabled={pending || paramsDisabled}
                    isRequired={
                      field.required || (field.key === 'base_url' && baseUrlOverrideEnabled)
                    }
                    status={error ? { type: 'error', message: error } : undefined}
                    description={
                      error === undefined && field.input_kind === 'url'
                        ? parameterHelp(field)
                        : undefined
                    }
                    onChange={(value) => updateParam(field, value)}
                    xstyle={styles.mono}
                  />
                </label>
              )}
            </div>
          )
        })}
        <label {...stylex.props(styles.field, styles.wide)}>
          <span {...stylex.props(styles.fieldLabel)}>{t('group.settings.base.providerUrl')}</span>
          <TextInput
            label={t('group.settings.base.providerUrl')}
            isLabelHidden
            type="text"
            value={providerUrl ?? ''}
            isDisabled={pending}
            description={t('group.settings.base.providerUrlHelp')}
            onChange={onProviderUrlChange}
            xstyle={styles.mono}
          />
        </label>
        {unified && headerActionsTarget ? createPortal(switchRow, headerActionsTarget) : switchRow}
      </div>
    </section>
  )
}

const styles = stylex.create({
  section: {
    display: 'grid',
    gap: '15px',
    scrollMarginTop: '76px',
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: '17px',
  },
  sectionHeading: {
    margin: 0,
    fontSize: 'var(--text-body)',
    fontWeight: 650,
  },
  sectionDescription: {
    maxWidth: '580px',
    marginTop: '3px',
    marginBottom: 0,
    marginLeft: 0,
    marginRight: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'repeat(2, minmax(0, 1fr))',
      '@media (max-width: 800px)': '1fr',
    },
    gap: '15px 18px',
  },
  wide: {
    gridColumn: {
      default: '1 / -1',
      '@media (max-width: 800px)': 'auto',
    },
  },
  baseUrlInput: {
    gridColumn: 'auto',
  },
  field: {
    display: 'grid',
    alignContent: 'start',
    gap: '6px',
  },
  fieldLabel: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
    fontWeight: 560,
  },
  fieldHint: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    lineHeight: 'var(--line-normal)',
  },
  channelField: {
    display: 'grid',
    minWidth: 0,
    gap: '4px',
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    paddingBottom: 'var(--space-3)',
  },
  channelPicker: {
    marginTop: '3px',
  },
  mono: {
    fontFamily: 'var(--font-mono)',
  },
  baseUrlToggle: {},
  baseUrlSwitch: {
    display: 'flex',
    minHeight: 'var(--control-xs)',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
  },
  switchRow: {
    display: 'flex',
    minHeight: '48px',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '18px',
    paddingTop: '8px',
    paddingBottom: '8px',
    paddingLeft: '2px',
    paddingRight: '2px',
  },
  switchCopy: {
    display: 'grid',
  },
  switchTitle: {
    fontSize: '12.5px',
  },
  switchHint: {
    color: 'var(--color-text-faint)',
    fontSize: '11px',
  },
  enabledActions: {
    display: 'flex',
    flexShrink: 0,
    alignItems: 'center',
    gap: '8px',
  },
})
