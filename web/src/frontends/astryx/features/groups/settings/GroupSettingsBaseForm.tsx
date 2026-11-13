import { Switch, TextInput } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import { ExternalLink } from 'lucide-react'
import type { ReactNode } from 'react'

import { createPortal } from 'react-dom'

import type { ChannelParamsDto } from '@shared/control/types'
import type { ChannelDto, ChannelFieldDto } from '@shared/control/resources/channels'

import { useT } from '../../../app/i18n'

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
  savedProviderUrl,
  providerUrlInvalid,

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
  siteAccessFields,
  onParamChange,
  onChannelSelect,
  onChannelsRetry,
  onNameChange,
  onProviderUrlChange,

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
  savedProviderUrl: string | null
  providerUrlInvalid: boolean

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
  siteAccessFields: ReactNode
  onParamChange(key: string, value: string | null): void
  onChannelSelect(channel: ChannelDto): void
  onChannelsRetry(): void
  onNameChange(value: string): void
  onProviderUrlChange(value: string): void

  onSetEnabled(value: boolean): void
}) {
  const t = useT()

  function isOptionalBaseURL(field: ChannelFieldDto): boolean {
    return field.key === 'base_url' && !field.required
  }

  function updateParam(field: ChannelFieldDto, value: string): void {
    if (isOptionalBaseURL(field) && !value.trim()) {
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

  function renderParam(field: ChannelFieldDto) {
    const error = paramErrors[field.key]
    const label = field.key === 'base_url' ? t('group.settings.base.upstreamUrl') : field.label
    return (
      <div key={field.key} {...stylex.props(styles.field)}>
        <span {...stylex.props(styles.fieldLabel)}>{label}</span>
        <TextInput
          label={label}
          isLabelHidden
          type="text"
          value={params[field.key] ?? ''}
          isDisabled={pending || paramsDisabled}
          isRequired={field.required}
          status={error ? { type: 'error', message: error } : undefined}
          description={
            error === undefined && field.input_kind === 'url' ? parameterHelp(field) : undefined
          }
          onChange={(value) => updateParam(field, value)}
          controlXstyle={styles.mono}
        />
      </div>
    )
  }

  if (section !== 'general') return null

  const switchRow = (
    <div {...stylex.props(styles.switchRow)}>
      <span {...stylex.props(styles.switchCopy)}>
        <strong {...stylex.props(styles.switchTitle)}>{t('group.settings.base.enabled')}</strong>
        <small {...stylex.props(styles.switchHint)}>{t('group.settings.base.enabledHelp')}</small>
      </span>
      <Switch
        label={t('group.settings.base.enabled')}
        isLabelHidden
        value={enabled}
        isDisabled={pending || enabledPending}
        onChange={onSetEnabled}
      />
    </div>
  )

  return (
    <section
      id="settings-general"
      {...stylex.props(styles.section, unified && styles.panelSection)}
    >
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
      <div {...stylex.props(styles.grid)}>
        <div {...stylex.props(styles.field)}>
          <label {...stylex.props(styles.fieldLabel)} htmlFor="settings-channel">
            {t('group.settings.base.channel')}
          </label>
          <select
            id="settings-channel"
            {...stylex.props(styles.channelSelect)}
            value={channelId}
            disabled={pending || (channels.length === 0 && (channelsLoading || channelsError))}
            aria-describedby="settings-channel-help"
            onChange={(event) => {
              const channel = channels.find((item) => item.channel_id === event.target.value)
              if (channel) onChannelSelect(channel)
            }}
          >
            {!selectedChannel && <option value={channelId}>{channelId}</option>}
            {channels.map((channel) => (
              <option key={channel.channel_id} value={channel.channel_id}>
                {channel.name}
              </option>
            ))}
          </select>
          <small id="settings-channel-help" {...stylex.props(styles.fieldHint)}>
            {t('group.settings.base.channelHelp')}
          </small>
          {channelsLoading && <small role="status">{t('import.presets.loading')}</small>}
          {channelsError && channels.length === 0 && (
            <div role="alert">
              {t('import.presets.loadFailed')}{' '}
              <button type="button" onClick={onChannelsRetry}>
                {t('common.retry')}
              </button>
            </div>
          )}
        </div>
        <div {...stylex.props(styles.field)}>
          <span {...stylex.props(styles.fieldLabel)}>{t('group.settings.base.name')}</span>
          <TextInput
            label={t('group.settings.base.name')}
            isLabelHidden
            value={name}
            isDisabled={pending}
            status={nameError !== '' ? { type: 'error', message: nameError } : undefined}
            onChange={onNameChange}
          />
        </div>

        {paramFields.filter((field) => field.key === 'base_url').map(renderParam)}
        <div {...stylex.props(styles.field)}>
          <span {...stylex.props(styles.fieldLabel)}>{t('group.settings.base.providerUrl')}</span>
          <div {...stylex.props(styles.providerRow)}>
            <div {...stylex.props(styles.providerInput)}>
              <TextInput
                label={t('group.settings.base.providerUrl')}
                isLabelHidden
                type="text"
                value={providerUrl ?? ''}
                isDisabled={pending}
                status={
                  providerUrlInvalid
                    ? { type: 'error', message: t('group.settings.base.providerUrlError') }
                    : undefined
                }
                description={t('group.settings.base.providerUrlHelp')}
                onChange={onProviderUrlChange}
                controlXstyle={styles.mono}
              />
            </div>
            {savedProviderUrl && (
              <a
                {...stylex.props(styles.providerLink)}
                href={savedProviderUrl}
                target="_blank"
                rel="noopener noreferrer"
                aria-label={t('group.openProviderUrl', { url: savedProviderUrl })}
                title={t('group.openProviderUrl', { url: savedProviderUrl })}
              >
                <ExternalLink size={16} aria-hidden="true" />
              </a>
            )}
          </div>
        </div>
        {siteAccessFields}
        {paramFields.filter((field) => field.key !== 'base_url').map(renderParam)}
        {unified && headerActionsTarget ? createPortal(switchRow, headerActionsTarget) : switchRow}
      </div>
    </section>
  )
}

const styles = stylex.create({
  section: {
    display: 'grid',
    gap: 'var(--space-2)',
    scrollMarginTop: '76px',
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 'var(--space-2)',
  },
  sectionHeading: {
    margin: 0,
    fontSize: 'var(--text-body)',
    fontWeight: 650,
  },
  panelSection: {
    borderTopWidth: 0,
    paddingTop: 0,
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
      default: 'repeat(3, minmax(0, 1fr))',
      '@media (max-width: 1100px)': 'repeat(2, minmax(0, 1fr))',
      '@media (max-width: 800px)': '1fr',
    },
    gap: 'var(--space-2) var(--space-3)',
  },

  field: {
    display: 'grid',
    alignContent: 'start',
    gap: 'var(--space-1)',
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
  channelSelect: {
    width: '100%',
    minWidth: 0,
    minHeight: { default: 'var(--control-sm)', '@media (max-width: 680px)': 'var(--touch-target)' },
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text)',
    paddingInline: 'var(--space-2)',
    font: 'inherit',
  },
  mono: {
    fontFamily: 'var(--font-mono)',
  },
  providerRow: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'flex-start',
    gap: 'var(--space-1)',
  },
  providerInput: {
    flex: '1 1 0',
    minWidth: 0,
  },
  providerLink: {
    display: 'inline-flex',
    flexShrink: 0,
    width: 'var(--touch-target)',
    height: 'var(--touch-target)',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 'var(--radius-tag)',
    color: { default: 'var(--color-text-muted)', ':hover': 'var(--color-action)' },
    backgroundColor: { default: 'transparent', ':hover': 'var(--color-surface-sunken)' },
  },
  switchRow: {
    display: 'flex',
    minHeight: 'var(--control-sm)',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-2)',
    paddingTop: 0,
    paddingBottom: 0,
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
})
